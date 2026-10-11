"""
audio_loader.py

Safe, in-memory audio loading, mono conversion, 16 kHz resampling,
and amplitude normalization.

GUARANTEE: The raw audio files on disk remain strictly read-only and immutable.
All signal processing operations occur exclusively in volatile RAM.
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
import soundfile as sf
import librosa


def load_and_preprocess_audio(
    file_path: Union[str, Path],
    target_sr: int = 16000,
    normalize_peak: float = 0.95,
) -> Tuple[np.ndarray, int]:
    """
    Safely loads an audio file into memory, converts it to mono, resamples it to target_sr,
    removes DC offset, and normalizes peak amplitude.

    Parameters:
    -----------
    file_path : str or Path
        Path to the original WAV file on disk (read-only).
    target_sr : int, default=16000
        Target sampling rate in Hz (16 kHz standard for speech).
    normalize_peak : float, default=0.95
        Target peak absolute amplitude for normalization. Set <= 0 to skip.

    Returns:
    --------
    audio : np.ndarray
        1D float32 array representing the preprocessed audio waveform.
    sample_rate : int
        The sample rate of the returned audio (always target_sr).
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Audio file not found: {path_obj}")

    # 1. Safe in-memory loading (supports 16-bit, 24-bit, 32-bit PCM)
    audio, orig_sr = sf.read(str(path_obj), dtype="float32")

    # 2. In-memory mono conversion (average channels if multi-channel)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    # Ensure 1D float32 contiguous array
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    # 3. In-memory resampling to target_sr using approved polyphase sinc interpolation
    if orig_sr != target_sr:
        audio = librosa.resample(
            audio,
            orig_sr=orig_sr,
            target_sr=target_sr,
            res_type="polyphase",
        )

    # 4. DC offset removal (center signal around zero amplitude)
    mean_val = np.mean(audio)
    if not np.isnan(mean_val):
        audio = audio - mean_val

    # 5. Peak amplitude normalization (prevent microphone gain differences from biasing models)
    if normalize_peak > 0:
        max_val = np.max(np.abs(audio))
        if max_val > 1e-7:
            audio = (audio / max_val) * np.float32(normalize_peak)

    return audio, target_sr
