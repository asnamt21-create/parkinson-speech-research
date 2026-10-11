"""
streamlit_app.py — Parkinson's Speech Detection: Research Demonstration
Serves the Stage 5 v2 Wav2Vec 2.0 + BiLSTM model via Streamlit Community Cloud.
"""

import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.deep_learning.bilstm_model import Wav2VecBiLSTMClassifier
from src.deep_learning.wav2vec_extractor import Wav2VecEmbeddingExtractor
from src.features.audio_loader import load_and_preprocess_audio

CHECKPOINT_PATH = ROOT_DIR / "results" / "models" / "stage5_v2_final_bilstm.pt"
METADATA_PATH   = ROOT_DIR / "results" / "models" / "stage5_v2_final_bilstm_metadata.json"

CLASS_NAMES = {0: "Healthy Control (EHC)", 1: "Parkinson's Disease (PD)"}
CLASS_SHORT = {0: "Healthy", 1: "Parkinson's Disease"}

st.set_page_config(
    page_title="Parkinson's Speech Detection — MSc Demo",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── Cache models for fast inference and low memory footprint ──────────
@st.cache_resource(show_spinner="Loading Wav2Vec 2.0 and BiLSTM models (first startup only)...")
def load_models():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as fh:
            metadata = json.load(fh)

    # 1. Load BiLSTM checkpoint
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    arch = ckpt["model_architecture"]
    classifier = Wav2VecBiLSTMClassifier(**arch)
    classifier.load_state_dict(ckpt["model_state_dict"])
    classifier.eval()

    # 2. Load frozen Wav2Vec 2.0 backbone
    extractor = Wav2VecEmbeddingExtractor(
        model_name="facebook/wav2vec2-base",
        chunk_duration_sec=5.0,
        min_tail_seconds=1.0,
    )

    # 3. Quick warm-up
    with torch.inference_mode():
        dummy_wav = np.zeros(80000, dtype=np.float32)
        inputs = extractor.feature_extractor(
            [dummy_wav],
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=False,
        )
        out = extractor.model(inputs.input_values.to(extractor.device))
        pooled = out.last_hidden_state.mean(dim=1).cpu()
        dummy_seq = pooled.unsqueeze(0)
        _ = classifier(dummy_seq, lengths=torch.tensor([1], dtype=torch.long))

    return extractor, classifier, metadata


def run_model_inference(extractor, classifier, audio_bytes: bytes, filename: str = "audio.wav"):
    """Runs exact Stage 5 v2 inference pipeline on raw audio bytes."""
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix or ".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        t_infer_0 = time.perf_counter()
        
        # Audio loading & preprocessing (16 kHz mono, DC removal, peak-norm 0.95)
        audio, sr = load_and_preprocess_audio(tmp_path, target_sr=extractor.target_sr, normalize_peak=0.95)
        duration_sec = len(audio) / sr

        # Exact v2 chunking (5.0s chunks, tail >= 1.0s kept, tail < 1.0s dropped)
        chunks, tail_dropped, dropped_tail_sec = extractor.chunk_audio(audio)
        if len(chunks) == 0:
            raise ValueError(
                f"Audio produced zero valid chunks after tail-drop QC "
                f"(duration: {duration_sec:.2f}s). Recording must be at least 1.0 second long."
            )

        full_chunks = [c for c in chunks if len(c) == extractor.chunk_samples]
        partial_chunks = [c for c in chunks if len(c) < extractor.chunk_samples]
        chunk_embeddings = []

        with torch.inference_mode():
            if full_chunks:
                inputs = extractor.feature_extractor(
                    full_chunks,
                    sampling_rate=extractor.target_sr,
                    return_tensors="pt",
                    return_attention_mask=False,
                )
                outputs = extractor.model(input_values=inputs.input_values.to(extractor.device))
                chunk_embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu())

            for tail in partial_chunks:
                inputs_tail = extractor.feature_extractor(
                    [tail],
                    sampling_rate=extractor.target_sr,
                    return_tensors="pt",
                    return_attention_mask=False,
                )
                outputs_tail = extractor.model(input_values=inputs_tail.input_values.to(extractor.device))
                chunk_embeddings.append(outputs_tail.last_hidden_state.mean(dim=1).cpu())

            sequence_embeddings = torch.cat(chunk_embeddings, dim=0)
            x = sequence_embeddings.unsqueeze(0)
            lengths = torch.tensor([sequence_embeddings.shape[0]], dtype=torch.long)

            logits = classifier(x, lengths=lengths)
            probs = F.softmax(logits, dim=-1).squeeze(0)

        t_infer_total = time.perf_counter() - t_infer_0
        pred_class = int(torch.argmax(probs).item())
        prob_healthy = float(probs[0].item())
        prob_pd = float(probs[1].item())

        return {
            "duration_sec": duration_sec,
            "num_chunks": len(chunks),
            "tail_dropped": tail_dropped,
            "dropped_tail_sec": dropped_tail_sec,
            "pred_class": pred_class,
            "pred_label_str": CLASS_SHORT[pred_class],
            "prob_healthy": prob_healthy,
            "prob_pd": prob_pd,
            "confidence_pct": max(prob_healthy, prob_pd) * 100,
            "infer_time_sec": t_infer_total,
        }
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ── UI ────────────────────────────────────────────────────────────────
st.title("🎙️ Speech-Based Parkinson's Detection")
st.caption("MSc Applied Data Science Research Project · Pretrained Wav2Vec 2.0 + BiLSTM Classifier")

# Research Disclaimer
st.warning(
    "⚠️ **Research Demonstration Only:** This system is an academic research prototype and "
    "is **NOT** a medical diagnostic tool. Predictions must not be used for clinical decisions."
)

# Load Models
try:
    extractor, classifier, metadata = load_models()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load models: {e}")
    model_loaded = False

if model_loaded:
    with st.sidebar:
        st.header("🔬 Model Details")
        st.markdown("**Backbone:** `facebook/wav2vec2-base` (Frozen)")
        st.markdown("**Classifier:** 1-Layer BiLSTM (128 hidden, Bidirectional)")
        st.markdown("**Parameters:** 920,066")
        st.markdown(f"**Trained Epoch:** {metadata.get('predetermined_epoch', 30)}")
        st.markdown(f"**Cohort:** {metadata.get('training_cohort', 'Italian PVS B1 (N=49)')}")
        st.markdown("---")
        st.markdown("🌐 [MSc Research Website](https://asnamt21-create.github.io/parkinson-speech-research/)")

    st.subheader("Interactive Speech Analysis")
    st.write(
        "Upload a sustained vowel or continuous reading recording, or record directly using your microphone. "
        "The model will extract 5-second acoustic chunks and compute classification probabilities."
    )

    tab_upload, tab_record = st.tabs(["📁 Upload Audio File", "🎙️ Record Microphone"])
    audio_data = None
    audio_name = ""

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "flac", "ogg", "mp3", "m4a", "webm"],
            help="Supported: WAV, FLAC, OGG, MP3, M4A, WEBM (Max 50MB)",
        )
        if uploaded_file is not None:
            audio_data = uploaded_file.read()
            audio_name = uploaded_file.name
            st.audio(audio_data)

    with tab_record:
        if hasattr(st, "audio_input"):
            recorded_file = st.audio_input("Record continuous speech (speak for 5–10 seconds)")
            if recorded_file is not None:
                audio_data = recorded_file.read()
                audio_name = "mic_recording.wav"
                st.audio(audio_data)
        else:
            st.info("Direct browser recording requires Streamlit >= 1.40 or file upload above.")

    if audio_data is not None:
        if st.button("⚡ Analyze Speech Recording", type="primary", use_container_width=True):
            with st.spinner("Analyzing speech through Wav2Vec 2.0 & BiLSTM..."):
                try:
                    res = run_model_inference(extractor, classifier, audio_data, audio_name)
                    
                    st.markdown("---")
                    st.subheader("Prediction Results")

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if res["pred_class"] == 1:
                            st.error(f"🔴 **{res['pred_label_str']}**")
                        else:
                            st.success(f"🟢 **{res['pred_label_str']}**")
                        st.metric("Model Confidence", f"{res['confidence_pct']:.1f}%")

                    with col2:
                        st.write("**Class Probability Distribution:**")
                        st.write(f"Healthy Control: {res['prob_healthy']*100:.1f}%")
                        st.progress(res['prob_healthy'])
                        st.write(f"Parkinson's Disease: {res['prob_pd']*100:.1f}%")
                        st.progress(res['prob_pd'])

                    st.markdown("### 📋 Acoustic & Timing Metadata")
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    mcol1.metric("Duration", f"{res['duration_sec']:.2f} s")
                    mcol2.metric("Valid 5s Chunks", res['num_chunks'])
                    mcol3.metric("Tail Dropped", f"{res['dropped_tail_sec']:.2f} s" if res['tail_dropped'] else "No")
                    mcol4.metric("Inference Time", f"{res['infer_time_sec']:.2f} s")

                except Exception as err:
                    st.error(f"Inference error: {err}")
