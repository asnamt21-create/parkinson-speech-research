"""
wav2vec_extractor.py  (v2)

v2 Extraction Pipeline — official Wav2Vec2FeatureExtractor, no attention_mask,
unpadded tail chunks, explicit tail-drop QC rule.

Key v2 changes from v1:
  * Uses Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base") for
    per-segment zero-mean / unit-variance normalisation (do_normalize=True).
  * Does NOT pass attention_mask — wav2vec2-base uses group feature normalisation
    and officially does not require attention_mask.
  * Does NOT zero-pad the final partial chunk.  The tail is processed at its
    TRUE length using the feature extractor.
  * Drops the final tail if it is shorter than min_tail_seconds (study QC rule;
    default 1.0 s).  The drop is recorded in the cache metadata.
  * Every v2 cache file includes an extraction_version field set to "v2".
    SpeechEmbeddingDataset rejects any cache file whose version != "v2".

Pipeline:
  Raw WAV
   -> in-memory 16 kHz mono (audio_loader.py)
   -> 5-second non-overlapping chunks
   -> official Wav2Vec2FeatureExtractor normalisation (no attention_mask)
   -> frozen Wav2Vec2Model (facebook/wav2vec2-base)
   -> simple mean-pool of all output frames  -> (1, 768) per chunk
   -> stack across chunks                    -> (T, 768) per recording

Cache location: results/stage5_cache_v2/  (completely separate from v1)
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
try:
    import pandas as pd
except ImportError:
    pd = None
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


from src.features.audio_loader import load_and_preprocess_audio

# Extraction version string — must match the version check in dataset.py
EXTRACTION_VERSION = "v2"


# ---------------------------------------------------------------------------
# Utility: deterministic cache key
# ---------------------------------------------------------------------------

def get_cache_filename(file_path: Union[str, Path]) -> str:
    """
    Generates a deterministic, collision-safe cache filename.

    Combines the filename stem with a 12-character SHA-256 hex digest of the
    normalised absolute path.  Shared between Wav2VecEmbeddingExtractor and
    SpeechEmbeddingDataset to guarantee 100 % key consistency.
    """
    path_obj = Path(file_path)
    stem = path_obj.stem
    norm_path = str(path_obj.resolve()).replace("\\", "/").lower()
    path_hash = hashlib.sha256(norm_path.encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{path_hash}.pt"


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class Wav2VecEmbeddingExtractor:
    """
    Extracts temporal sequences of chunk embeddings using a frozen pretrained
    Wav2Vec 2.0 model via the official Wav2Vec2FeatureExtractor.

    v2 protocol
    -----------
    * Official per-segment z-score normalisation via Wav2Vec2FeatureExtractor
      (do_normalize=True, confirmed for facebook/wav2vec2-base).
    * No attention_mask is passed to Wav2Vec2Model.forward() — consistent with
      the official Hugging Face guidance for wav2vec2-base.
    * Partial final tail >= min_tail_seconds: processed at its TRUE length
      (not zero-padded).
    * Partial final tail < min_tail_seconds: DROPPED (QC rule).  The drop is
      recorded per-recording in the cache metadata.

    Recording ID
    ------------
    filename stem is used as recording_id.  Both IPVS and MDVR-KCL stems are
    verified unique within and across datasets (0 duplicates confirmed).

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier.  Default: "facebook/wav2vec2-base".
    device : Optional[str]
        Target device.  Auto-detected if None.
    chunk_duration_sec : float
        Duration of each non-overlapping chunk in seconds.  Default: 5.0.
    target_sr : int
        Expected sampling rate in Hz.  Default: 16000.
    min_tail_seconds : float
        Minimum tail duration to include.  Tails shorter than this are dropped
        and logged.  Default: 1.0 s (study QC rule, not a Wav2Vec2 requirement).
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        device: Optional[str] = None,
        chunk_duration_sec: float = 5.0,
        target_sr: int = 16000,
        min_tail_seconds: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.chunk_duration_sec = chunk_duration_sec
        self.target_sr = target_sr
        self.chunk_samples = int(self.target_sr * self.chunk_duration_sec)
        self.min_tail_seconds = min_tail_seconds
        self.min_tail_samples = int(self.target_sr * self.min_tail_seconds)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading Wav2Vec 2.0 feature extractor + model: '{self.model_name}' on {self.device} ...")

        # Official feature extractor — handles per-segment normalisation
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)

        # Frozen pretrained model
        self.model = Wav2Vec2Model.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        print(
            f"  Hidden size:           {self.model.config.hidden_size}\n"
            f"  Parameters:            frozen\n"
            f"  do_normalize:          {self.feature_extractor.do_normalize}  (official z-score per chunk)\n"
            f"  return_attention_mask: {self.feature_extractor.return_attention_mask}  (not used)\n"
            f"  Chunk duration:        {self.chunk_duration_sec} s ({self.chunk_samples} samples)\n"
            f"  Tail-drop threshold:   < {self.min_tail_seconds} s (< {self.min_tail_samples} samples)"
        )

    # ------------------------------------------------------------------
    # chunk_audio
    # ------------------------------------------------------------------

    def chunk_audio(
        self, audio: np.ndarray
    ) -> Tuple[List[np.ndarray], bool, float]:
        """
        Splits a 1-D audio waveform into non-overlapping 5-second chunks.

        v2 rules
        --------
        * Full chunks (exactly chunk_samples): included as-is.
        * Partial final tail >= min_tail_samples: included at its TRUE length —
          NOT zero-padded.
        * Partial final tail < min_tail_samples: DROPPED.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 array at target_sr Hz.

        Returns
        -------
        chunks : List[np.ndarray]
            Full chunks have shape (chunk_samples,).
            The kept partial tail (if any) has shape (valid_len,) where
            valid_len < chunk_samples.
        tail_dropped : bool
            True if a partial tail existed and was below the QC threshold.
        dropped_tail_seconds : float
            Duration of the dropped tail in seconds (0.0 if nothing was dropped).
        """
        total_samples = len(audio)
        tail_dropped = False
        dropped_tail_seconds = 0.0

        chunks: List[np.ndarray] = []

        for start_idx in range(0, total_samples, self.chunk_samples):
            end_idx = min(start_idx + self.chunk_samples, total_samples)
            chunk = audio[start_idx:end_idx]
            valid_len = len(chunk)

            if valid_len == self.chunk_samples:
                # Full 5-second chunk — include as-is
                chunks.append(chunk.copy())
            else:
                # Partial tail
                if valid_len >= self.min_tail_samples:
                    # Keep at its true length: do NOT zero-pad
                    chunks.append(chunk.copy())
                else:
                    # Drop: too short to be reliable
                    tail_dropped = True
                    dropped_tail_seconds = valid_len / self.target_sr

        return chunks, tail_dropped, dropped_tail_seconds

    # ------------------------------------------------------------------
    # extract_from_audio
    # ------------------------------------------------------------------

    def extract_from_audio(
        self,
        audio: np.ndarray,
        batch_size: int = 8,
    ) -> Tuple[torch.Tensor, bool, float]:
        """
        Extracts chunk embeddings using the official Wav2Vec2FeatureExtractor.

        v2 protocol
        -----------
        * No zero-padding of partial tail.
        * No attention_mask passed to the model.
        * Official per-chunk z-score normalisation via feature extractor.
        * Simple mean-pool of all output frames per chunk.

        Full chunks (all same length) are batched together.
        The partial tail chunk (if kept) is processed individually.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 array at target_sr Hz.
        batch_size : int
            Batch size used for full-length chunks.  Default: 8.

        Returns
        -------
        sequence_embeddings : torch.Tensor  shape (T, 768)
        tail_dropped : bool
        dropped_tail_seconds : float
        """
        chunks, tail_dropped, dropped_tail_seconds = self.chunk_audio(audio)

        if len(chunks) == 0:
            raise ValueError(
                f"Audio produced zero valid chunks after tail-drop QC "
                f"(audio duration: {len(audio) / self.target_sr:.3f} s, "
                f"tail-drop threshold: {self.min_tail_seconds} s). "
                f"Recording may be too short for the study QC rule."
            )

        # Separate full chunks from the partial tail (at most one)
        full_chunks = [c for c in chunks if len(c) == self.chunk_samples]
        partial_chunks = [c for c in chunks if len(c) < self.chunk_samples]

        chunk_embeddings: List[torch.Tensor] = []

        with torch.no_grad():
            # --- Process full chunks in batches (all same length → no padding) ---
            for i in range(0, len(full_chunks), batch_size):
                batch = full_chunks[i: i + batch_size]
                # Official feature extractor: per-chunk z-score normalisation
                # return_attention_mask=False → consistent with wav2vec2-base guidance
                inputs = self.feature_extractor(
                    batch,
                    sampling_rate=self.target_sr,
                    return_tensors="pt",
                    return_attention_mask=False,
                )
                input_values = inputs.input_values.to(self.device)
                outputs = self.model(input_values=input_values)
                # Mean-pool over all output frames: (B, F, 768) -> (B, 768)
                pooled = outputs.last_hidden_state.mean(dim=1)
                chunk_embeddings.append(pooled.cpu())

            # --- Process partial tail chunk at its true length (no padding) ---
            for chunk in partial_chunks:
                inputs = self.feature_extractor(
                    [chunk],
                    sampling_rate=self.target_sr,
                    return_tensors="pt",
                    return_attention_mask=False,
                )
                input_values = inputs.input_values.to(self.device)
                outputs = self.model(input_values=input_values)
                pooled = outputs.last_hidden_state.mean(dim=1)
                chunk_embeddings.append(pooled.cpu())

        sequence_tensor = torch.cat(chunk_embeddings, dim=0)  # (T, 768)
        return sequence_tensor, tail_dropped, dropped_tail_seconds

    # ------------------------------------------------------------------
    # extract_from_file
    # ------------------------------------------------------------------

    def extract_from_file(
        self, file_path: Union[str, Path]
    ) -> Tuple[torch.Tensor, float, int, bool, float]:
        """
        Loads raw audio, preprocesses to 16 kHz mono, and extracts chunk
        embeddings via the v2 pipeline.

        Note: load_and_preprocess_audio applies DC removal and peak normalisation
        (0.95) before the Wav2Vec2FeatureExtractor applies its own per-chunk
        z-score normalisation.  The z-score normalisation supersedes the earlier
        scale, so the final embedding is correctly normalised.

        Returns
        -------
        sequence_embeddings : torch.Tensor  (T, 768)
        duration_seconds : float
        num_chunks : int
        tail_dropped : bool
        dropped_tail_seconds : float
        """
        audio, sr = load_and_preprocess_audio(
            file_path,
            target_sr=self.target_sr,
            normalize_peak=0.95,
        )
        duration_seconds = len(audio) / sr
        embeddings, tail_dropped, dropped_tail_seconds = self.extract_from_audio(audio)
        return embeddings, duration_seconds, embeddings.shape[0], tail_dropped, dropped_tail_seconds


# ---------------------------------------------------------------------------
# Cache population helper
# ---------------------------------------------------------------------------

def cache_embeddings_for_manifest(
    manifest_df: "pd.DataFrame",
    cache_dir: Union[str, Path],
    extractor: Wav2VecEmbeddingExtractor,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """
    Extracts and caches v2 embeddings for all recordings in a manifest.

    v2 cache objects include extraction provenance and tail-drop metadata.
    The cache directory must be the v2 directory (results/stage5_cache_v2/).

    Guarantees
    ----------
    * Cache files store ONLY acoustic representations and recording/speaker IDs.
    * NO class labels ('group') or fold assignments ('fold') are cached —
      preventing any label-leakage into the cache.
    * Every cache file contains extraction_version = "v2".

    Returns
    -------
    cached_paths : Dict[recording_id -> Path]
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    cached_paths: Dict[str, Path] = {}
    total = len(manifest_df)
    tail_drop_count = 0

    print(f"\nCaching v2 embeddings for {total} recordings into: {cache_path}")

    for idx, (_, row) in enumerate(manifest_df.iterrows(), 1):
        file_path = row["file_path"]
        speaker_id = str(row["speaker_id"])
        # recording_id: filename stem, verified unique within and across IPVS/MDVR datasets.
        recording_id = Path(file_path).stem
        cache_filename = get_cache_filename(file_path)
        save_file = cache_path / cache_filename

        if save_file.exists() and not overwrite:
            # --- v2 cache provenance validation (never blindly reuse) ---
            # Load metadata only to verify the five provenance fields.
            # If any field mismatches the current extractor settings, the file
            # is a stale or incompatible cache and must be regenerated.
            try:
                existing = torch.load(save_file, map_location="cpu", weights_only=True)
                expected_norm = "Wav2Vec2FeatureExtractor_official_z-score"
                mismatches = []
                if existing.get("extraction_version") != EXTRACTION_VERSION:
                    mismatches.append(
                        f"extraction_version={existing.get('extraction_version')!r} "
                        f"(expected {EXTRACTION_VERSION!r})"
                    )
                if existing.get("model_name") != extractor.model_name:
                    mismatches.append(
                        f"model_name={existing.get('model_name')!r} "
                        f"(expected {extractor.model_name!r})"
                    )
                if existing.get("normalization_mode") != expected_norm:
                    mismatches.append(
                        f"normalization_mode={existing.get('normalization_mode')!r} "
                        f"(expected {expected_norm!r})"
                    )
                if existing.get("chunk_duration_sec") != extractor.chunk_duration_sec:
                    mismatches.append(
                        f"chunk_duration_sec={existing.get('chunk_duration_sec')!r} "
                        f"(expected {extractor.chunk_duration_sec!r})"
                    )
                if existing.get("min_tail_seconds") != extractor.min_tail_seconds:
                    mismatches.append(
                        f"min_tail_seconds={existing.get('min_tail_seconds')!r} "
                        f"(expected {extractor.min_tail_seconds!r})"
                    )

                if not mismatches:
                    # All five provenance fields match — safe to reuse
                    cached_paths[recording_id] = save_file
                    continue
                else:
                    # Stale or incompatible cache: re-extract and overwrite
                    print(
                        f"  STALE CACHE [{idx}/{total}] {recording_id}: "
                        f"mismatched field(s): {'; '.join(mismatches)}. "
                        f"Re-extracting and overwriting."
                    )
                    # Fall through to extraction block below

            except Exception as load_exc:
                print(
                    f"  CORRUPT CACHE [{idx}/{total}] {recording_id}: "
                    f"could not load existing file ({load_exc}). "
                    f"Re-extracting and overwriting."
                )

        try:
            embeddings, duration, num_chunks, tail_dropped, dropped_tail_seconds = (
                extractor.extract_from_file(file_path)
            )
        except ValueError as exc:
            print(f"  WARNING [{idx}/{total}]: Skipping {recording_id} — {exc}")
            continue

        if tail_dropped:
            tail_drop_count += 1

        # v2 cache object: provenance + tail metadata + embeddings (NO labels/folds)
        cache_obj = {
            # --- Extraction provenance ---
            "extraction_version": EXTRACTION_VERSION,
            "model_name": extractor.model_name,
            "normalization_mode": "Wav2Vec2FeatureExtractor_official_z-score",
            "chunk_duration_sec": extractor.chunk_duration_sec,
            "min_tail_seconds": extractor.min_tail_seconds,
            # --- Tail QC metadata ---
            "tail_dropped": tail_dropped,
            "dropped_tail_seconds": float(dropped_tail_seconds),
            # --- Recording identity (strictly no labels or fold assignments) ---
            "recording_id": recording_id,
            "speaker_id": speaker_id,
            # --- Embeddings ---
            "embeddings": embeddings.to(torch.float32),
            "num_chunks": int(num_chunks),
            "duration_seconds": float(duration),
            "sample_rate": extractor.target_sr,
        }

        torch.save(cache_obj, save_file)
        cached_paths[recording_id] = save_file

        if idx % 10 == 0 or idx == total:
            tail_flag = " [TAIL DROPPED]" if tail_dropped else ""
            print(
                f"  [{idx}/{total}] {recording_id} "
                f"(spk: {speaker_id}, chunks: {num_chunks}{tail_flag})"
            )

    print(
        f"v2 caching complete — {len(cached_paths)}/{total} recordings cached, "
        f"tail dropped for {tail_drop_count} recording(s)."
    )
    return cached_paths
