# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0

"""Seldon as a scikit-learn style classifier over pandas tables."""

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
from neuralk_oss.model import Seldon
from neuralk_oss.model.seldon import HUB_CHECKPOINTS

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


class SeldonClassifier:
    """A Seldon classifier: ``fit`` keeps the training table as context, ``predict_proba`` reads the test rows against it.

    sdm infers each column's type (categorical columns are embedded as such,
    NaNs kept for the model's imputation and missingness term), preprocesses
    each ensemble member with ``recipe``, and averages the members' logits
    before the softmax.

    Parameters
    ----------
    model : Seldon
        A classifier, e.g. ``Seldon.from_pretrained(path)``.
    n_estimators : int, default=8
        Ensemble members.
    device : str or torch.device, optional
        Device of the model and the tables; CUDA when available by default.
    max_context_size : int, default=200_000
        Training rows one member reads in context; a larger table gives each
        member its own random subset.
    amp : bool, default=True
        float16 autocast on CUDA; False runs in float32.
    cudnn_attention : bool, default=False
        Let PyTorch's attention use cuDNN's kernel. It is faster once a table
        shape has run, but builds a plan per new shape, seconds each, which a
        table fitted once pays in full; by default the FlashAttention and
        memory-efficient kernels run instead.
    random_state : int, default=0
        Seed of the context subsets and of sdm's per-member preprocessing.
    recipe : SeldonRecipe, optional
        The preprocessing; ``SeldonRecipe()`` by default.
    query_chunk_size : int, default=8192
        Test rows run through the model together, per member.
    estimator_batch_size : int, None or "auto", default="auto"
        Members run through the model together; None runs them all at once,
        "auto" as many as ``member_batch_size`` gives for the context. sdm
        still separates members whose preprocessed tables differ in shape or
        classes, and each member's outputs are the same up to rounding.
    max_cache_gb : float or None, default=8.0
        GiB of context encodings ``fit`` keeps for ``predict_proba``, over all
        members; members past it encode their context again at each call,
        slower but within memory. None keeps them all.
    """

    def __init__(
        self,
        model: Seldon,
        *,
        n_estimators: int = 8,
        device: str | torch.device | None = None,
        max_context_size: int = 200_000,
        amp: bool = True,
        random_state: int = 0,
        cudnn_attention: bool = False,
        recipe: SeldonRecipe | None = None,
        query_chunk_size: int = 8192,
        estimator_batch_size: int | Literal["auto"] | None = "auto",
        max_cache_gb: float | None = 8.0,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.n_estimators = n_estimators
        self.max_context_size = max_context_size
        self.amp = amp
        self.cudnn_attention = cudnn_attention
        self.random_state = random_state
        self.estimator_batch_size = estimator_batch_size
        self.estimator = SeldonICL(model, recipe, query_chunk_size, max_cache_gb).to(self.device)

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | os.PathLike,
        *,
        subfolder: str | None = None,
        revision: str | None = None,
        token: str | None = None,
        **kwargs: Any,
    ) -> SeldonClassifier:
        """The classifier of a saved model: a released model's name, ``"seldon"`` or ``"seldon-mini"``, or what ``Seldon.from_pretrained`` takes (a local directory, a checkpoint name such as ``"seldon-mini-classifier"``, a Hub repository); ``kwargs`` are the constructor's.

        A released model's name also takes the settings of its packaged config
        (``configs/<name>.yaml``: n_estimators, recipe, ...), which ``kwargs``
        override; anything else takes the constructor's defaults.
        """
        # A model name is its classification checkpoint; a local directory of that name wins.
        checkpoint = f"{name_or_path}-classifier"
        if subfolder is None and checkpoint in HUB_CHECKPOINTS and not Path(name_or_path).is_dir():
            settings = _read_config(_CONFIGS / f"{name_or_path}.yaml")
            settings.pop("checkpoint", None)
            kwargs = {**settings, **kwargs}
            name_or_path = checkpoint
        model = Seldon.from_pretrained(
            name_or_path, subfolder=subfolder, revision=revision, token=token
        )
        return cls(model, **kwargs)

    @classmethod
    def from_config(cls, config: str | os.PathLike, **kwargs: Any) -> SeldonClassifier:
        """The classifier of a YAML config: a packaged one by name, such as ``"seldon-mini"``, or a file.

        The config maps ``checkpoint`` to ``from_pretrained``'s first argument,
        ``recipe`` to ``SeldonRecipe``'s keyword arguments, and any other key to
        a keyword argument of ``from_pretrained``; ``kwargs`` override its keys.
        """
        path = Path(config)
        if path.suffix not in (".yaml", ".yml") and path.parent == Path():
            names = sorted(p.name.removesuffix(".yaml") for p in _CONFIGS.iterdir())
            if str(config) not in names:
                raise ValueError(f"unknown config {config!r}; packaged: {names}, or a .yaml path")
            path = _CONFIGS / f"{config}.yaml"
        params = {**_read_config(path), **kwargs}
        if "checkpoint" not in params:
            raise ValueError(f"config {config!r} names no checkpoint")
        return cls.from_pretrained(params.pop("checkpoint"), **params)

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> SeldonClassifier:
        """Keep ``X`` and ``y`` as the context of every member."""
        X = _frame(X)
        self.classes_, codes = np.unique(np.asarray(y), return_inverse=True)
        self._stypes = sdm.infer_stypes(X, _low_cardinality="infer")
        x_context = sdm.TableTensor.from_pandas(df=X, stypes=self._stypes)
        y_context = sdm.TableTensor.from_pandas(
            df=pd.DataFrame({_TARGET: codes}), stypes={_TARGET: "categorical"}
        )
        num_estimators: int | None = self.n_estimators
        self._subsampled = len(X) > self.max_context_size
        if self._subsampled:
            generator = torch.Generator().manual_seed(self.random_state)
            repeats = -(-self.n_estimators * self.max_context_size // len(X))
            permutations = [torch.randperm(len(X), generator=generator) for _ in range(repeats)]
            shape = (self.n_estimators, self.max_context_size)
            index = torch.cat(permutations)[: shape[0] * shape[1]]
            x_context = x_context[index].unflatten(0, shape)
            y_context = y_context[index].unflatten(0, shape)
            num_estimators = None
        batch_size = self.estimator_batch_size
        if batch_size == "auto":
            n_rows, n_features = x_context.shape[-2:]
            batch_size = member_batch_size(n_rows, n_features, self.n_estimators)
        with self._inference():
            self.estimator.fit(
                x=x_context.to(self.device),
                y=y_context.to(self.device),
                num_estimators=num_estimators,
                estimator_batch_size=batch_size,
                generator=torch.Generator(self.device).manual_seed(self.random_state),
            )
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Class probabilities of the rows of ``X``, ``(n_rows, n_classes)`` in the order of ``classes_``."""
        X = _frame(X)
        query = sdm.TableTensor.from_pandas(df=X, stypes=self._stypes, device=self.device)
        if self._subsampled:
            query = query.expand(self.n_estimators, *query.size())
        with self._inference():
            out = self.estimator.predict(query)
        codes = [int(code) for code in out.columns[sdm.Stype.numerical]]
        proba = np.zeros((len(X), len(self.classes_)), dtype=np.float32)
        proba[:, codes] = out.numerical.float().cpu().numpy()
        return proba

    def to(self, device: str | torch.device) -> SeldonClassifier:
        """Move the classifier, fitted or not, to ``device``: the network, the fitted preprocessing and the next tables' device.

        The members' cached contexts stay where ``fit`` left them: fitted on
        CUDA with several members, sdm keeps each batch's in pinned CPU memory
        and ``predict_proba`` streams them to the device one at a time, so they
        are never all on the GPU at once. Only a cache on another CUDA device
        (a single member's) moves.
        """
        self.device = torch.device(device)
        self.estimator.to(self.device)
        cache = self.estimator._cache
        if cache is None:
            return self
        recipe = cache["recipe_execution"].recipe
        for processor in (recipe.features, recipe.target, recipe.output):
            processor.to(self.device)
        # shortcut: sdm's private cache layout (_cache, _apply_tensor), pinned at 0.2.1; recheck on an sdm upgrade.
        self.estimator._cache = cache._apply_tensor(
            lambda t: t.to(self.device) if t.is_cuda and t.device != self.device else t
        )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """The most probable class of each row of ``X``."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    @contextmanager
    def _inference(self) -> Iterator[None]:
        amp = self.amp and self.device.type == "cuda"
        attention = nullcontext() if self.cudnn_attention else sdpa_kernel(_NON_CUDNN_ATTENTION)
        with (
            torch.inference_mode(),
            torch.autocast(self.device.type, dtype=torch.float16, enabled=amp),
            attention,
        ):
            yield


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
