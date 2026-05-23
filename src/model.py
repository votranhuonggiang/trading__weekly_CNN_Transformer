from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden_dim = max(32, d_model // 2)
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_logits = self.score(x).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        return torch.sum(x * attn_weights.unsqueeze(-1), dim=1)


class CNNTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int = 3,
        conv_channels: int = 64,
        kernel_size: int = 5,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, conv_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, d_model, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
        )
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Learn which timesteps matter instead of flattening the whole lookback uniformly.
        self.pool = AttentionPooling(d_model=d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = self.pool(x)
        return self.head(x)
