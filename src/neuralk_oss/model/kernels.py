# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0

"""The fused RMSNorm of CUDA inference under autocast, from sdm's kernels.

One kernel normalizes in fp32 and writes the autocast dtype, optionally after
RoPE, where PyTorch would write fp32 and cast it again before the next GEMM.
Every other call (CPU, no autocast, gradients) runs PyTorch's norm.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from sdm._kernels import rmsnorm_cast as _rmsnorm_cast
except ImportError:  # pragma: no cover - sdm is a dependency
    _rmsnorm_cast = None

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def fused(x: Tensor) -> bool:
    """Whether the fused norm runs on ``x``: CUDA inference under autocast."""
    return (
        _rmsnorm_cast is not None
        and x.is_cuda
        and torch.is_autocast_enabled("cuda")
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and x.dtype in _DTYPES
    )


def norm(module: nn.Module, x: Tensor) -> Tensor:
    """``module(x)``, an RMSNorm in the autocast dtype when ``fused(x)``; any other norm as is."""
    if not (isinstance(module, nn.RMSNorm) and fused(x)):
        return module(x)
    weight = module.weight
    if weight is not None and weight.dtype not in _DTYPES:
        return module(x)
    eps = module.eps if module.eps is not None else torch.finfo(torch.float32).eps
    return _rmsnorm_cast(x, weight, eps)


def head_norm(x: Tensor, eps: float, inv_freq: Tensor | None = None) -> Tensor:
    """Each head of ``x`` ``(..., L, num_heads, head_dim)`` RMS-normed without weight, after RoPE along ``L`` when ``inv_freq`` is given.

    RoPE rotates half the channels against the other half, rounding in
    ``x``'s dtype.
    """
    rope = None
    if inv_freq is not None:
        seq = torch.arange(x.size(-3), device=x.device, dtype=torch.float32)
        freq = seq.view(-1, 1) * inv_freq.view(1, -1)
        rope = freq.cos().to(x.dtype), freq.sin().to(x.dtype)
    if fused(x):
        return _rmsnorm_cast(x, None, eps, rope)
    if rope is not None:
        cos, sin = (table.unsqueeze(-2) for table in rope)
        x1, x2 = x.split(cos.size(-1), dim=-1)
        x = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return F.rms_norm(x, (x.size(-1),), None, eps)
