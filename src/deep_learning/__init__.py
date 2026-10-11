"""
Stage 5: Deep Learning Module — Deployment Minimal Subset.
"""

from src.deep_learning.bilstm_model import Wav2VecBiLSTMClassifier
from src.deep_learning.wav2vec_extractor import Wav2VecEmbeddingExtractor

__all__ = [
    "Wav2VecEmbeddingExtractor",
    "Wav2VecBiLSTMClassifier",
]

