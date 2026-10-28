"""The row stage: the readout tokens, the row blocks, and the projection of the readout to the ICL width."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from neuralk_oss.model import kernels
from neuralk_oss.model.attention import BlockStack, MultiheadAttentionBlock, make_norm
from neuralk_oss.model.config import BlockConfig, SeldonConfig

# FlashAttention fails above 65,535 sequences in one call (pytorch#142228).
ROW_BLOCK_MAX_SEQUENCES = 32_768


class RowInteraction(nn.Module):
    """The readout tokens (``cls_tokens``), the row blocks (RoPE, gated query scaling), the final norm and the projection to the ICL width.

    Every row block but the last runs between the column blocks; the last one,
    its queries the readout tokens, makes the row's representation.
    """

    def __init__(self, config: SeldonConfig, block: BlockConfig) -> None:
        super().__init__()
        self.embed_dim = block.d_model
        self.num_cls = config.row.num_cls
        self.tf_row = BlockStack(
            [
                MultiheadAttentionBlock(block, scaling="gated", rope=True)
                for _ in range(config.row.num_blocks)
            ]
        )
        self.cls_tokens = nn.Parameter(torch.empty(self.num_cls, self.embed_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        self.out_ln = make_norm(self.embed_dim, block.use_rmsnorm, block.bias_free_ln)
        width = self.embed_dim * self.num_cls
        self.project = (
            nn.Identity() if width == config.icl_width else nn.Linear(width, config.icl_width)
        )

    def state(self, cells: Tensor) -> Tensor:
        """The readout tokens in front of each row's cells: ``(B, T, H, E)`` -> ``(B, T, K+H, E)``."""
        B, T, _, E = cells.shape
        readout = self.cls_tokens.expand(B, T, self.num_cls, E)
        return torch.cat([readout.to(cells.dtype), cells], dim=2)

    def block(self, i: int, state: Tensor) -> Tensor:
        """Row block ``i`` over the state ``(B, T, K+H, E)``: each row's tokens attend to one another."""
        return self.over_row_slices(self.tf_row.blocks[i], state)

    def forward(self, state: Tensor) -> Tensor:
        """Row representations ``(B, T, icl_width)`` from the state after the last column block."""
        last = self.tf_row.blocks[-1]
        query = state[..., : self.num_cls, :]
        x = self.over_row_slices(last, query, state, state)
        return self.project(kernels.norm(self.out_ln, x).flatten(-2))

    @staticmethod
    def over_row_slices(fn: Callable[..., Tensor], x: Tensor, *args: Tensor) -> Tensor:
        """``fn(x, *args)`` over slices of at most ``ROW_BLOCK_MAX_SEQUENCES`` rows, each argument sliced along ``T`` with ``x``.

        A row block attends within each row, so the slices' outputs are the
        unsliced call's.
        """
        B, T = x.shape[:2]
        step = max(1, ROW_BLOCK_MAX_SEQUENCES // max(B, 1))
        if T <= step:
            return fn(x, *args)
        return torch.cat(
            [
                fn(x[:, t0 : t0 + step], *(a[:, t0 : t0 + step] for a in args))
                for t0 in range(0, T, step)
            ],
            dim=1,
        )
