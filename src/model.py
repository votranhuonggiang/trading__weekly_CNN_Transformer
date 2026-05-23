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


class MultiScaleCNNStem(nn.Module):
    def __init__(
        self,
        num_features: int,
        conv_channels: int,
        d_model: int,
        dropout: float,
        kernel_sizes: tuple[int, ...] = (3, 5, 9),
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(num_features, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for kernel_size in kernel_sizes
            ]
        )
        merged_channels = conv_channels * len(kernel_sizes)
        self.projection = nn.Sequential(
            nn.Conv1d(merged_channels, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outputs, dim=1)
        return self.projection(x)


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

        # Candidate 02 isolates the CNN change: use a multi-scale stem, keep
        # the transformer depth and mean pooling identical to the baseline head.
        self.conv = MultiScaleCNNStem(
            num_features=num_features,
            conv_channels=conv_channels,
            d_model=d_model,
            dropout=dropout,
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
        x = x.mean(dim=1)
        return self.head(x)
