"""Seldon's attention: the multi-head attention, the pre-norm block, and the column stage's induced block.

``MultiheadAttention`` adds to PyTorch's a RoPE (row blocks), an RMSNorm on
each head's q and k, and a query scaling by the attended-key count.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.backends.cuda import SDPAParams, can_use_flash_attention, flash_sdp_enabled

from neuralk_oss.model.config import BlockConfig

ROPE_THETA = 100_000
GATE_HIDDEN = 64
# Epsilon of the per-head RMSNorm on q and k.
HEAD_NORM_EPS = 1e-6
# FlashAttention launches one block per sequence on the grid's y axis, at
# most 65,535 per call (pytorch#142228).
FLASH_MAX_SEQUENCES = 65_535

Scaling = Literal["none", "log", "gated"]


def make_norm(d_model: int, use_rmsnorm: bool, bias_free_ln: bool) -> nn.Module:
    """RMSNorm, else a LayerNorm without bias under ``bias_free_ln``."""
    if use_rmsnorm:
        return nn.RMSNorm(d_model)
    return nn.LayerNorm(d_model, bias=not bias_free_ln)


class MultiheadAttention(nn.MultiheadAttention):
    """Attention with RoPE (row blocks), an RMSNorm on each head's q and k, and a query scaling by the attended-key count.

    Every input has shape ``(..., len, embed_dim)``.

    Parameters
    ----------
    embed_dim, num_heads : int
        Width, and number of heads.
    bias : bool, default=True
        If False, the in- and out-projections have no bias.
    scaling : {"none", "log", "gated"}, default="none"
        ``"log"`` scales the queries as ``q * log(n) * head_scale`` (column
        induce, ICL), ``"gated"`` times ``1 + tanh(gate(q))`` (row), ``n``
        being the number of attended keys.
    rope : bool, default=False
        Rotate q and k by their position (row blocks).

    Attributes
    ----------
    query_kv_heads : int | None
        Set by ``ICLearning`` around one call: the queries attend with the
        first ``query_kv_heads`` key/value heads, each shared by
        ``num_heads // query_kv_heads`` consecutive query heads.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        *,
        scaling: Scaling = "none",
        rope: bool = False,
    ) -> None:
        super().__init__(embed_dim, num_heads, bias=bias, batch_first=True)
        self.scaling = scaling
        self.query_kv_heads: int | None = None
        if scaling != "none":
            self.head_scale = nn.Parameter(torch.ones(num_heads))
        if scaling == "gated":
            self.gate = nn.Sequential(
                nn.Linear(self.head_dim, GATE_HIDDEN),
                nn.GELU(),
                nn.Linear(GATE_HIDDEN, self.head_dim),
            )
        # A buffer, loaded with the weights: the model was trained with these
        # frequencies, an ulp off rope_inv_freq's.
        self.register_buffer("inv_freq", self.rope_inv_freq(self.head_dim) if rope else None)

    @staticmethod
    def rope_inv_freq(head_dim: int) -> Tensor:
        """The rotation frequency of each pair of channels, ``(head_dim // 2,)``."""
        arange = torch.arange(0, head_dim, 2, dtype=torch.float32)
        return 1.0 / (ROPE_THETA ** (arange / head_dim))

    @staticmethod
    def shared_rows(x: Tensor) -> Tensor | None:
        """The ``(len, E)`` rows ``x`` broadcasts over its batch dims (the inducing points), else None.

        They are normed and projected once rather than once per column.
        """
        lead = x.dim() - 2
        if lead > 0 and not any(x.stride()[:lead]):
            return x[(0,) * lead]
        return None

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Multi-head attention of ``query`` over ``key`` and ``value``: ``(..., tgt_len, embed_dim)``.

        The query scaling's ``n`` is the number of keys.
        """
        if key.shape != value.shape:
            raise ValueError(f"key shape {key.shape} does not match value shape {value.shape}")
        q, k, v = self._project_qkv(query, key, value)
        q, k = self._qk_transform(q, k)
        q = self._scale_q(q, key.shape[-2])
        return self._project_out(self._core(q, k, v))

    def _split_heads(self, x: Tensor, contiguous: bool = True) -> Tensor:
        """(..., len, embed_dim) -> (..., num_heads, len, head_dim)."""
        heads = x.view(*x.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        return heads.contiguous() if contiguous else heads

    def _project_qkv(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Joint in-projection, split into heads; under ``query_kv_heads`` the keys and values keep their first heads only."""
        shared = self.shared_rows(query)
        kv = self.query_kv_heads
        q, k, v = F._in_projection_packed(
            query if shared is None else shared,
            key,
            value,
            self.in_proj_weight,
            self.in_proj_bias,
        )
        q, k, v = (self._split_heads(t, contiguous=kv is None) for t in (q, k, v))
        if kv is not None:
            k, v = k[..., :kv, :, :], v[..., :kv, :, :]
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if shared is not None:
            q = q.expand(*query.shape[:-2], *q.shape)
        return q, k, v

    def _qk_transform(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """RoPE (row blocks), then the per-head RMSNorm, on the per-head queries and keys."""
        return self._head_norm(q), self._head_norm(k)

    def _head_norm(self, x: Tensor) -> Tensor:
        if self.inv_freq is not None:
            x = self._rotate(x)
        return F.rms_norm(x, (self.head_dim,), None, HEAD_NORM_EPS)

    def _rotate(self, x: Tensor) -> Tensor:
        """``x`` of shape ``(..., len, head_dim)`` rotated by its position along ``len``, half the channels against the other half."""
        seq = torch.arange(x.size(-2), device=x.device, dtype=torch.float32)
        freq = seq.view(-1, 1) * self.inv_freq.view(1, -1)
        sin, cos = freq.sin().to(x.dtype), freq.cos().to(x.dtype)
        x1, x2 = x.split(sin.size(-1), dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

    def _scale_q(self, q: Tensor, n: int) -> Tensor:
        """``q`` scaled for ``n`` attended keys."""
        if self.scaling == "none":
            return q
        log_n = q.new_full((), n, dtype=torch.float32).clamp(min=1.0).log().to(q.dtype)
        scaled = q * log_n * self.head_scale.to(q.dtype).view(-1, 1, 1)
        if self.scaling == "gated":
            scaled = scaled * (1 + self.gate(q).tanh())
        return scaled

    def _core(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """SDPA over the flattened batch dims, in slices FlashAttention takes: ``(..., nh, tgt_len, hs)``."""
        shape = q.shape
        q, k, v = (t.reshape(-1, *t.shape[-3:]) for t in (q, k, v))
        if q.shape[0] <= FLASH_MAX_SEQUENCES:
            out = self._sdpa(q, k, v)
        else:
            step = FLASH_MAX_SEQUENCES
            out = torch.cat(
                [
                    self._sdpa(q[i : i + step], k[i : i + step], v[i : i + step])
                    for i in range(0, q.shape[0], step)
                ]
            )
        return out.view(shape)

    def _sdpa(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """The attention kernel on the flattened ``(B, nh, L, hs)`` tensors, grouped-query when k has fewer heads than q."""
        if torch.is_autocast_enabled(q.device.type):
            # The head norms return q and k in fp32 under autocast; the kernel
            # choice must see the autocast dtype.
            dtype = torch.get_autocast_dtype(q.device.type)
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        if k.shape[-3] != q.shape[-3]:
            if self._native_gqa(q, k, v):
                return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
            repeats = q.shape[-3] // k.shape[-3]
            k = k.repeat_interleave(repeats, dim=-3)
            v = v.repeat_interleave(repeats, dim=-3)
        return F.scaled_dot_product_attention(q, k, v)

    @staticmethod
    def _native_gqa(q: Tensor, k: Tensor, v: Tensor) -> bool:
        """Whether SDPA runs this grouped-query call without repeating k and v: on CPU, or on CUDA by FlashAttention."""
        if not q.is_cuda:
            return True
        params = SDPAParams(q, k, v, None, 0.0, False, True)
        return flash_sdp_enabled() and can_use_flash_attention(params)

    def _project_out(self, attn_output: Tensor) -> Tensor:
        """Merge the heads and apply the out-projection: ``(..., tgt_len, E)``."""
        attn_output = attn_output.transpose(-3, -2).contiguous()
        attn_output = attn_output.view(*attn_output.shape[:-2], self.embed_dim)
        return F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)


class MultiheadAttentionBlock(nn.Module):
    """Pre-norm attention block with separate norms on the query (``norm1``) and on the keys/values (``norm_kv``), then a GELU feedforward.

    Parameters
    ----------
    config : BlockConfig
        Settings of the block.
    scaling : {"none", "log", "gated"}
        The attention's query scaling.
    rope : bool, default=False
        Whether the attention rotates q and k by their position.
    """

    def __init__(self, config: BlockConfig, *, scaling: Scaling, rope: bool = False) -> None:
        super().__init__()
        d_model = config.d_model
        self.attn = MultiheadAttention(
            d_model, config.nhead, bias=config.bias, scaling=scaling, rope=rope
        )
        self.linear1 = nn.Linear(d_model, config.dim_feedforward, bias=config.bias)
        self.linear2 = nn.Linear(config.dim_feedforward, d_model, bias=config.bias)
        self.norm1 = make_norm(d_model, config.use_rmsnorm, config.bias_free_ln)
        self.norm2 = make_norm(d_model, config.use_rmsnorm, config.bias_free_ln)
        self.norm_kv = make_norm(d_model, config.use_rmsnorm, config.bias_free_ln)

    def forward(
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        train_size: int | None = None,
    ) -> Tensor:
        """The attention of ``q`` over ``k`` and ``v`` with its residual, then the feedforward with its: ``(..., tgt_len, d_model)``.

        Parameters
        ----------
        q : Tensor
            ``(..., tgt_len, d_model)``.
        k, v : Tensor | None, default=None
            ``(..., src_len, d_model)``; None uses ``q`` (self-attention).
        train_size : int | None, default=None
            When given, K/V are the first ``train_size`` positions of ``q``
            and ``k`` / ``v`` are ignored.
        """
        if train_size is not None:
            k = v = q[..., :train_size, :]
        else:
            k = q if k is None else k
            v = q if v is None else v
        kv = self.norm_kv(k)
        attn = self.attn(self._prenorm(q), kv, kv if v is k else self.norm_kv(v))
        x = q + attn
        return x + self.linear2(F.gelu(self.linear1(self.norm2(x))))

    def _prenorm(self, x: Tensor) -> Tensor:
        """The query's norm, computed once for rows shared over the batch dims (the inducing points)."""
        shared = MultiheadAttention.shared_rows(x)
        if shared is None:
            return self.norm1(x)
        return self.norm1(shared).expand(x.shape)


class InducedSelfAttentionBlock(nn.Module):
    """A column block: the inducing points attend to the column's context rows (``induce``), then every row attends to them (``expand``).

    Parameters
    ----------
    config : BlockConfig
        Settings of both attention blocks.
    num_inds : int
        Number of inducing points.

    References
    ----------
    .. [1] Lee et al. "Set Transformer: A Framework for Attention-based
           Permutation-Invariant Neural Networks", ICML 2019
    """

    def __init__(self, config: BlockConfig, num_inds: int) -> None:
        super().__init__()
        self.multihead_attn1 = MultiheadAttentionBlock(config, scaling="log")
        self.multihead_attn2 = MultiheadAttentionBlock(config, scaling="none")
        self.num_inds = num_inds
        self.ind_vectors = nn.Parameter(torch.empty(num_inds, config.d_model))
        nn.init.trunc_normal_(self.ind_vectors, std=0.02)

    def induce(self, context: Tensor) -> Tensor:
        """Stage 1: the inducing points attend to the context rows ``(..., train_size, d_model)``: ``(..., num_inds, d_model)``."""
        *batch_shape, _, d_model = context.shape
        ind_vectors = self.ind_vectors.expand(*batch_shape, self.num_inds, d_model)
        return self.multihead_attn1(ind_vectors, context, context)

    def expand(self, src: Tensor, induced: Tensor) -> Tensor:
        """Stage 2: every row of ``src`` attends to the induced set."""
        return self.multihead_attn2(src, induced, induced)

    def forward(self, src: Tensor, train_size: int) -> Tensor:
        """Both stages over ``src`` ``(..., seq_len, d_model)``, the first ``train_size`` rows the context."""
        return self.expand(src, self.induce(src[..., :train_size, :]))


class BlockStack(nn.Module):
    """A stage's attention ``blocks``."""

    def __init__(self, blocks: list[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
