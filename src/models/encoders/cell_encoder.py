"""Multi-scale CNN encoder for cell line multi-omics features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Attention-weighted dual pooling (max + sum) over the sequence dimension."""

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = F.softmax(self.score(x), dim=-1)
        attended = x * weights

        max_pooled = attended.max(dim=-1).values
        sum_pooled = attended.sum(dim=-1)

        return torch.cat([max_pooled, sum_pooled], dim=-1), weights.squeeze(1)


class ConvBranch(nn.Module):
    """Two-layer 1D convolution branch with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Two-layer residual convolution with a 1x1 projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2

        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.block = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor, multi_scale: torch.Tensor) -> torch.Tensor:
        return F.relu(self.shortcut(x) + self.block(multi_scale))


class CellLineEncoder(nn.Module):
    """Multi-scale CNN encoder for cell line multi-omics features."""

    OMICS_CHANNELS = 4

    def __init__(
        self,
        n_genes: int,
        in_channels: int = OMICS_CHANNELS,
        embed_dim: int = 256,
        branch_channels: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        if in_channels != self.OMICS_CHANNELS:
            raise ValueError(
                f"CellLineEncoder expects {self.OMICS_CHANNELS} omics channels, "
                f"got in_channels={in_channels}"
            )

        self.n_genes = n_genes
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        total_channels = branch_channels * 3

        self.branch_small = ConvBranch(in_channels, branch_channels, kernel_size=3)
        self.branch_medium = ConvBranch(in_channels, branch_channels, kernel_size=7)
        self.branch_large = ConvBranch(in_channels, branch_channels, kernel_size=15)

        self.residual = ResidualBlock(in_channels, total_channels)

        self.attention_pool = AttentionPooling(total_channels)
        self.projection = nn.Sequential(
            nn.Linear(total_channels * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode multi-omics features into fixed-size embeddings."""
        if x.dim() != 3 or x.shape[1] != self.in_channels or x.shape[2] != self.n_genes:
            raise ValueError(
                f"Expected (batch, {self.in_channels}, {self.n_genes}), "
                f"got {tuple(x.shape)}"
            )

        multi_scale = torch.cat(
            [self.branch_small(x), self.branch_medium(x), self.branch_large(x)],
            dim=1,
        )

        x = self.residual(x, multi_scale)

        pooled, _ = self.attention_pool(x)
        return self.projection(pooled)
