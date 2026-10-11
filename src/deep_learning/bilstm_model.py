"""
bilstm_model.py

PyTorch implementation of the BiLSTM classifier for speech-based
Parkinson's Disease detection using frozen Wav2Vec 2.0 chunk embeddings.

Architecture:
- Input: Sequence of chunk-level embeddings (batch_size, seq_len, 768)
- BiLSTM: 1 layer, hidden_size = 128, bidirectional = True (output dim = 256)
- Dropout: 0.30
- Classifier: Linear projection from 256 to 2 output classes (PD vs. Healthy)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Wav2VecBiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier operating on temporal sequences of chunk embeddings.

    Parameters:
    -----------
    input_dim : int, default=768
        Dimension of input chunk embeddings (768 for Wav2Vec2-base).
    hidden_dim : int, default=128
        Hidden dimension per LSTM direction (total bidirectional dim = 256).
    num_layers : int, default=1
        Number of recurrent LSTM layers.
    dropout : float, default=0.30
        Dropout probability applied before the linear classification head.
    num_classes : int, default=2
        Number of output target classes (2 for binary: Healthy vs. PD).
    pooling : str, default='mean'
        Pooling strategy across chunk sequence outputs: 'mean' or 'last'.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.30,
        num_classes: int = 2,
        pooling: str = "mean",
    ):
        super().__init__()

        if pooling not in ("mean", "last"):
            raise ValueError(
                f"Invalid pooling strategy: '{pooling}'. Supported pooling strategies are 'mean' or 'last'."
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.pooling = pooling

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Dropout layer (applied after recurrent aggregation)
        self.dropout = nn.Dropout(p=dropout)

        # Linear classification head mapping (hidden_dim * 2) -> num_classes
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through BiLSTM classifier.

        Parameters:
        -----------
        x : torch.Tensor
            Batch of padded chunk embeddings with shape (batch_size, seq_len, input_dim).
        lengths : torch.Tensor, optional
            1D integer tensor of actual sequence lengths (number of valid chunks per sample).
            If None, all sequences are assumed to have length seq_len.

        Returns:
        --------
        logits : torch.Tensor
            Raw class logits with shape (batch_size, num_classes).
        """
        batch_size, seq_len, _ = x.shape

        if lengths is None:
            lengths = torch.full(
                (batch_size,), seq_len, dtype=torch.long, device=x.device
            )

        # Pack padded sequence for efficient RNN computation
        # (Lengths must be on CPU for pack_padded_sequence in some PyTorch versions)
        lengths_cpu = lengths.to(device="cpu", dtype=torch.int64)

        # Handle case where all lengths are identical or sequences are packed
        packed_x = pack_padded_sequence(
            x,
            lengths=lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, (hn, cn) = self.lstm(packed_x)

        if self.pooling == "last":
            # Concat forward final hidden state and backward initial hidden state
            # hn shape: (num_layers * 2, batch_size, hidden_dim)
            # Forward last: hn[-2], Backward last: hn[-1]
            pooled = torch.cat([hn[-2], hn[-1]], dim=-1)
        else:
            # 'mean' pooling across valid sequence chunks
            out, _ = pad_packed_sequence(packed_out, batch_first=True)
            # Create boolean mask for valid chunks: (batch_size, seq_len, 1)
            time_idx = torch.arange(out.size(1), device=x.device).unsqueeze(0)
            mask = (time_idx < lengths.unsqueeze(1)).unsqueeze(-1).float()
            # Masked average pooling
            sum_pooled = (out * mask).sum(dim=1)
            pooled = sum_pooled / lengths.unsqueeze(1).clamp(min=1).float()

        # Apply dropout
        dropped = self.dropout(pooled)

        # Linear classification
        logits = self.fc(dropped)

        return logits
