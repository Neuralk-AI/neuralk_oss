"""The ICL stage: every row attends to the labelled context rows, then the head."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from neuralk_oss.model.attention import BlockStack, MultiheadAttentionBlock, make_norm
from neuralk_oss.model.config import BlockConfig, SeldonConfig


class ICLearning(nn.Module):
    """Labels on the context rows, attention blocks with the log query scaling, norm and head.

    With ``num_icl_key_value_heads_for_query`` set, the query rows attend to
    the context rows with the reduced key/value heads.
    """

    def __init__(self, config: SeldonConfig, block: BlockConfig) -> None:
        super().__init__()
        self.max_classes = config.max_classes
        self.num_quantiles = config.num_quantiles
        self.kv_heads = config.num_icl_key_value_heads_for_query
        self.fp32_head = config.fp32_head
        d_model = block.d_model
        self.y_encoder = (
            nn.Embedding(self.max_classes, d_model)
            if self.max_classes > 0
            else nn.Linear(1, d_model, bias=False)
        )
        self.tf_icl = BlockStack(
            [MultiheadAttentionBlock(block, scaling="log") for _ in range(config.icl.num_blocks)]
        )
        self.ln = make_norm(d_model, block.use_rmsnorm, block.bias_free_ln)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, self.max_classes or self.num_quantiles),
        )

    def _encode_targets(self, y: Tensor) -> Tensor:
        if self.max_classes > 0:
            return self.y_encoder(y.long())
        return self.y_encoder(y.float().unsqueeze(-1))

    def encode_context(self, R: Tensor, y: Tensor) -> list[Tensor]:
        """The context rows' states entering each block, ``(B, train_size, D)`` each, from their representations ``R`` and labels ``y``."""
        context = R + self._encode_targets(y).to(R.dtype)
        states = []
        for block in self.tf_icl.blocks:
            states.append(context)
            context = block(context)
        return states

    def forward(self, R: Tensor, context: list[Tensor]) -> Tensor:
        """Outputs ``(B, T, out_dim)`` of the query rows ``R``, each block's queries attending to its context state."""
        for block, keys in zip(self.tf_icl.blocks, context, strict=True):
            if self.kv_heads is None:
                R = block(R, keys, keys)
            else:
                R = self._reduced_heads(block, R, keys)
        return self._head(R)

    def _reduced_heads(self, block: MultiheadAttentionBlock, q: Tensor, kv: Tensor) -> Tensor:
        """``block`` with its queries attending through the reduced key/value heads."""
        block.attn.query_kv_heads = self.kv_heads
        try:
            return block(q, kv, kv)
        finally:
            block.attn.query_kv_heads = None

    def _head(self, x: Tensor) -> Tensor:
        if not self.fp32_head:
            return self.decoder(self.ln(x))
        with torch.autocast(x.device.type, enabled=False):
            return self.decoder(self.ln(x.float()))
