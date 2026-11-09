# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0

"""What ``SeldonClassifier`` and ``SeldonRegressor`` share: their configs, tables, member batches, inference context and fitting."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import sdm
import torch
import yaml
from torch.nn.attention import SDPBackend, sdpa_kernel

from neuralk_oss.inference.estimator import SeldonICL
from neuralk_oss.inference.recipe import SeldonRecipe
from neuralk_oss.model.seldon import Task

_TARGET = "target"
_CONFIGS = files("neuralk_oss") / "configs"
# Every attention kernel PyTorch has but cuDNN's (see ``cudnn_attention``).
_NON_CUDNN_ATTENTION = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]
# Context cells (rows x features) one batch of members holds at most by default:
# below it a member alone leaves the GPU idle, so members run together; above
# it they run one at a time, as fast, in a fraction of the memory.
MEMBER_BATCH_CELLS = 400_000


def member_batch_size(n_rows: int, n_features: int, n_estimators: int) -> int:
    """Members run together on a context of ``n_rows`` x ``n_features``, within ``MEMBER_BATCH_CELLS``."""
    return max(1, min(n_estimators, MEMBER_BATCH_CELLS // max(n_rows * n_features, 1)))


@contextmanager
def _inference(device: torch.device, amp: bool, cudnn_attention: bool) -> Iterator[None]:
    """Inference mode, under float16 autocast on CUDA with ``amp``, without cuDNN's attention unless ``cudnn_attention``."""
    amp = amp and device.type == "cuda"
    attention = nullcontext() if cudnn_attention else sdpa_kernel(_NON_CUDNN_ATTENTION)
    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=torch.float16, enabled=amp),
        attention,
    ):
        yield


def _fit_members(
    estimator: SeldonICL,
    x_context: sdm.TableTensor,
    y_context: sdm.TableTensor,
    *,
    n_estimators: int,
    max_context_size: int,
    random_state: int,
    estimator_batch_size: int | Literal["auto"] | None,
    device: torch.device,
    amp: bool,
    cudnn_attention: bool,
) -> bool:
    """Fit ``estimator``'s ``n_estimators`` members on the context; whether each reads its own random subset of it, the query rows then expanded to one copy per member.

    Above ``max_context_size`` rows, the members take consecutive slices of
    concatenated random permutations (NVIDIA's rule), so that every row is
    read about equally often.
    """
    num_estimators: int | None = n_estimators
    n_rows = x_context.shape[-2]
    subsampled = n_rows > max_context_size
    if subsampled:
        generator = torch.Generator().manual_seed(random_state)
        repeats = -(-n_estimators * max_context_size // n_rows)
        permutations = [torch.randperm(n_rows, generator=generator) for _ in range(repeats)]
        shape = (n_estimators, max_context_size)
        index = torch.cat(permutations)[: shape[0] * shape[1]]
        x_context = x_context[index].unflatten(0, shape)
        y_context = y_context[index].unflatten(0, shape)
        num_estimators = None
    batch_size = estimator_batch_size
    if batch_size == "auto":
        batch_size = member_batch_size(*x_context.shape[-2:], n_estimators)
    with _inference(device, amp, cudnn_attention):
        estimator.fit(
            x=x_context.to(device),
            y=y_context.to(device),
            num_estimators=num_estimators,
            estimator_batch_size=batch_size,
            generator=torch.Generator(device).manual_seed(random_state),
        )
    return subsampled


def _move_estimator(estimator: SeldonICL, device: torch.device) -> None:
    """Move ``estimator``, fitted or not, to ``device``: its network, its fitted preprocessing, and a cached context on another CUDA device (see ``SeldonClassifier.to``)."""
    estimator.to(device)
    cache = estimator._cache
    if cache is None:
        return
    recipe = cache["recipe_execution"].recipe
    for processor in (recipe.features, recipe.target, recipe.output):
        processor.to(device)
    # shortcut: sdm's private cache layout (_cache, _apply_tensor), pinned at 0.2.1; recheck on an sdm upgrade.
    estimator._cache = cache._apply_tensor(
        lambda t: t.to(device) if t.is_cuda and t.device != device else t
    )


def _config_params(config: str | os.PathLike, kwargs: dict[str, Any], task: Task) -> dict[str, Any]:
    """The keys of a YAML config of ``task``, a packaged one by name or a file, with ``kwargs`` over them; they must name a ``checkpoint``.

    A packaged regression config's name ends with ``-regressor``; ``template``
    serves both tasks; a ``-forecaster`` config is ``SeldonForecaster``'s.
    """
    path = Path(config)
    if path.suffix not in (".yaml", ".yml") and path.parent == Path():
        names = sorted(
            name
            for p in _CONFIGS.iterdir()
            if (name := p.name.removesuffix(".yaml")) == "template"
            or (
                not name.endswith("-forecaster")
                and name.endswith("-regressor") == (task == "regression")
            )
        )
        if str(config) not in names:
            raise ValueError(
                f"unknown {task} config {config!r}; packaged: {names}, or a .yaml path"
            )
        path = _CONFIGS / f"{config}.yaml"
    params = {**_read_config(path), **kwargs}
    if task == "classification" and "clip_margin" in params:
        raise ValueError(
            f"config {config!r} sets clip_margin, a regressor's: load it with SeldonRegressor.from_config"
        )
    if "checkpoint" not in params:
        raise ValueError(f"config {config!r} names no checkpoint")
    return params


def _read_config(path: Path | Traversable) -> dict[str, Any]:
    """A YAML config's keys, its ``recipe`` built as a ``SeldonRecipe``."""
    params = yaml.safe_load(path.read_text()) or {}
    if isinstance(params.get("recipe"), dict):
        params["recipe"] = SeldonRecipe(**params["recipe"])
    return params


def _frame(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """``X`` as a DataFrame with string column names, which sdm requires, and signed integers for unsigned ones, which CUDA cannot index with."""
    if isinstance(X, pd.DataFrame):
        X = X.rename(columns=str)
    else:
        X = np.asarray(X)
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    unsigned = {
        name: "Int64" if isinstance(dtype, pd.api.extensions.ExtensionDtype) else "int64"
        for name, dtype in X.dtypes.items()
        if pd.api.types.is_unsigned_integer_dtype(dtype)
    }
    # shortcut: a uint64 above 2**63 wraps (numpy) or raises (nullable); none expected in tables.
    return X.astype(unsigned) if unsigned else X
