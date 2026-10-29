# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: MIT

"""The column stage: the cell and label embeddings, and the layers' column blocks."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from neuralk_oss.model.attention import BlockStack, InducedSelfAttentionBlock
from neuralk_oss.model.config import FEATURE_GROUP_SIZE, BlockConfig, SeldonConfig


class ColEmbedding(nn.Module):
    """Cell and label embeddings and the column blocks.

    A cell is embedded with the features of its group, ``c + 2^g - 1`` for
    ``g < FEATURE_GROUP_SIZE``, cycling over the table's features. A missing
    cell takes its feature's context mean, and the group's missingness
    pattern adds ``nan_lin``. Categorical columns have their own frequencies
    and projection. A context row's cells also carry its label's embedding.

    The column blocks run between the row stage's blocks; ``Seldon`` drives
    the layers.
    """

    def __init__(self, config: SeldonConfig, block: BlockConfig) -> None:
        super().__init__()
        self.embed_dim = embed_dim = config.embed_dim
        self.max_classes = config.max_classes
        self.fp32_cells = config.fp32_cells

        G, num_freq = FEATURE_GROUP_SIZE, config.num_frequencies
        self.num_freq = nn.Parameter(torch.randn(G, num_freq))
        self.cat_freq = nn.Parameter(torch.randn(G, num_freq))
        self.num_lin = nn.Linear(2 * num_freq, embed_dim)
        self.cat_lin = nn.Linear(2 * num_freq, embed_dim)
        self.nan_lin = nn.Linear(G, embed_dim, bias=False)
        self.y_encoder = (
            nn.Embedding(self.max_classes, embed_dim)
            if self.max_classes > 0
            else nn.Linear(1, embed_dim, bias=False)
        )
        self.tf_col = BlockStack(
            [
                InducedSelfAttentionBlock(block, config.num_inducing_points)
                for _ in range(config.col.num_blocks)
            ]
        )

    @staticmethod
    def context_mean(X: Tensor) -> Tensor:
        """Each feature's mean over the context rows ``X`` ``(B, T, H)``, ignoring NaNs: ``(B, 1, H)``, 0 for a feature with none."""
        mean = X.nanmean(dim=-2, keepdim=True)
        return mean.masked_fill(mean.isnan(), 0.0)

    def check_classes(self, y: Tensor) -> None:
        """Raise if the context labels ``y`` hold more classes than the head predicts."""
        if self.max_classes <= 0 or y.numel() == 0:
            return
        n = int(y.max().item()) + 1
        if n > self.max_classes:
            raise ValueError(
                f"Seldon embeds at most max_classes={self.max_classes} classes, the labels "
                f"hold {n}: run more as error-correcting output codes"
            )

    def encode_targets(self, y: Tensor) -> Tensor:
        """The label embeddings ``(B, T, E)`` of ``y`` ``(B, T)``."""
        if self.max_classes > 0:
            return self.y_encoder(y.long())
        return self.y_encoder(y.float().unsqueeze(-1))

    def embed(
        self,
        X: Tensor,
        mean: Tensor,
        categorical_mask: Tensor,
        y: Tensor | None = None,
    ) -> Tensor:
        """The cells of ``X`` ``(B, T, H)`` embedded: ``(B, T, H, E)``.

        Parameters
        ----------
        X : Tensor
            Features ``(B, T, H)``; a NaN cell takes ``mean``.
        mean : Tensor
            Each feature's context mean, ``(B, 1, H)`` (``context_mean``).
        categorical_mask : Tensor
            Which columns are categorical, ``(B, H)``.
        y : Tensor, optional
            The labels ``(B, T)`` of context rows, added to their cells.
        """
        x, missing, categorical = self._grouped_cells(X, mean, categorical_mask)
        cells = self._project_cells(x, missing, categorical)
        if y is not None:
            cells = cells + self.encode_targets(y).unsqueeze(2).to(cells.dtype)
        return cells

    def _grouped_cells(
        self, X: Tensor, mean: Tensor, categorical_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Each cell's group of imputed values and of missingness, ``(B, T, H, G)``, and its categorical flags ``(B, H, G)``."""
        B, T, H = X.shape
        G = FEATURE_GROUP_SIZE
        missing = X.isnan()
        X = torch.where(missing, mean.to(X.dtype), X)

        shift = 2 ** torch.arange(G, device=X.device) - 1
        index = ((torch.arange(H, device=X.device)[:, None] + shift[None]) % H)[None]
        index = index.expand(B, H, G)
        cell_index = index[:, None].expand(B, T, H, G)
        x = X.unsqueeze(-1).expand(B, T, H, G).gather(2, cell_index)
        missing = missing.unsqueeze(-1).expand(B, T, H, G).gather(2, cell_index)
        categorical = categorical_mask.unsqueeze(-1).expand(B, H, G).gather(1, index)
        return x, missing, categorical

    def _project_cells(self, x: Tensor, missing: Tensor, categorical: Tensor) -> Tensor:
        """``_grouped_cells``' output to the cells' embeddings ``(B, T, H, E)``, in fp32 under ``fp32_cells``."""
        if not self.fp32_cells:
            return self._fourier_projection(x, missing, categorical)
        with torch.autocast(x.device.type, enabled=False):
            return self._fourier_projection(x.float(), missing, categorical)

    def _fourier_projection(self, x: Tensor, missing: Tensor, categorical: Tensor) -> Tensor:
        freq = torch.where(
            categorical[..., None], self.cat_freq.float(), self.num_freq.float()
        )  # (B, H, G, F)
        weight = torch.where(
            categorical[..., None, None], self.cat_lin.weight, self.num_lin.weight
        )  # (B, H, G, E, 2F)
        bias = torch.where(categorical[..., None], self.cat_lin.bias, self.num_lin.bias).sum(
            dim=-2
        )  # (B, H, E)

        angle = x.float().unsqueeze(-1) * freq[:, None]  # (B, T, H, G, F)
        fourier = torch.cat([angle.sin(), angle.cos()], dim=-1).to(weight.dtype)
        # One projection per column: (B, H, T, G*2F) @ (B, H, G*2F, E).
        fourier = fourier.transpose(1, 2).flatten(-2)
        weight = weight.transpose(-3, -2).flatten(-2).mT
        out = torch.matmul(fourier, weight).transpose(1, 2)  # (B, T, H, E)
        out = out + bias[:, None].to(out.dtype)
        return out + self.nan_lin(missing.to(out.dtype))
