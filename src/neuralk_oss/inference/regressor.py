# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0

"""Seldon as a scikit-learn style regressor over pandas tables."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import sdm
import torch
from torch import Tensor

from neuralk_oss.inference._inference import (
    _CONFIGS,
    _TARGET,
    _config_params,
    _fit_members,
    _frame,
    _inference,
    _move_estimator,
    _read_config,
)
from neuralk_oss.inference.estimator import SeldonICL
from neuralk_oss.inference.recipe import SeldonRecipe
from neuralk_oss.model import Seldon
from neuralk_oss.model.seldon import HUB_CHECKPOINTS

_PREDICT_ROWS = 2**15


class SeldonRegressor:
    """A Seldon regressor: ``fit`` keeps the training table as context, ``predict`` reads the test rows against it.

    sdm infers each column's type (categorical columns are embedded as such,
    NaNs kept for the model's imputation and missingness term), preprocesses
    each ensemble member with ``recipe`` (the target standardized), and
    averages the members' quantiles, back in the target's units and sorted,
    by sdm's trimmed mean (``trim_fraction=0.2``). With ``clip_margin``, the
    quantiles are then clamped to the training target's range, widened by
    ``clip_margin`` times that range on each side. The prediction is the mean
    of the quantiles.

    Parameters
    ----------
    model : Seldon
        A regressor, e.g. ``Seldon.from_pretrained(path, task="regression")``.
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
        Let PyTorch's attention use cuDNN's kernel (see ``SeldonClassifier``).
    random_state : int, default=0
        Seed of the context subsets and of sdm's per-member preprocessing.
    recipe : SeldonRecipe, optional
        The preprocessing; ``SeldonRecipe()`` by default. Its classification
        parameters (``class_shift``, ``average_logits``,
        ``softmax_temperature``) do not apply.
    query_chunk_size : int, default=8192
        Test rows run through the model together, per member.
    estimator_batch_size : int, None or "auto", default="auto"
        Members run through the model together (see ``SeldonClassifier``).
    max_cache_gb : float or None, default=8.0
        GiB of context encodings ``fit`` keeps over all members (see
        ``SeldonClassifier``).
    clip_margin : float or None, default=None
        How far beyond the training target's range the quantiles may reach,
        in units of that range; None leaves them unclamped. The released
        models' configs set 0.1.
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
        clip_margin: float | None = None,
    ) -> None:
        if model.num_quantiles <= 0:
            raise ValueError("SeldonRegressor runs a regressor; the model is a classifier")
        if clip_margin is not None and clip_margin < 0:
            raise ValueError(f"clip_margin must be None or at least 0, got {clip_margin}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.n_estimators = n_estimators
        self.max_context_size = max_context_size
        self.amp = amp
        self.cudnn_attention = cudnn_attention
        self.random_state = random_state
        self.estimator_batch_size = estimator_batch_size
        self.clip_margin = clip_margin
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
    ) -> SeldonRegressor:
        """The regressor of a saved model: a released model's name, ``"seldon"`` or ``"seldon-mini"``, or what ``Seldon.from_pretrained`` takes (a local directory, a checkpoint name such as ``"seldon-mini-regressor"``, a Hub repository); ``kwargs`` are the constructor's.

        A released model's name, or its regression checkpoint's, also takes
        the settings of its packaged regression config
        (``configs/<model>-regressor.yaml``: n_estimators, clip_margin,
        recipe, ...), which ``kwargs`` override; anything else takes the
        constructor's defaults.
        """
        name = str(name_or_path)
        checkpoint = name if name.endswith("-regressor") else f"{name}-regressor"
        if subfolder is None and checkpoint in HUB_CHECKPOINTS and not Path(name_or_path).is_dir():
            settings = _read_config(_CONFIGS / f"{checkpoint}.yaml")
            settings.pop("checkpoint", None)
            kwargs = {**settings, **kwargs}
            name_or_path = checkpoint
        model = Seldon.from_pretrained(
            name_or_path, task="regression", subfolder=subfolder, revision=revision, token=token
        )
        return cls(model, **kwargs)

    @classmethod
    def from_config(cls, config: str | os.PathLike, **kwargs: Any) -> SeldonRegressor:
        """The regressor of a YAML config: a packaged one by name, such as ``"seldon-mini-regressor"``, or a file.

        The config maps ``checkpoint`` to ``from_pretrained``'s first argument,
        ``recipe`` to ``SeldonRecipe``'s keyword arguments, and any other key to
        a keyword argument of ``from_pretrained``; ``kwargs`` override its keys.
        """
        params = _config_params(config, kwargs, "regression")
        return cls.from_pretrained(params.pop("checkpoint"), **params)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        *,
        stypes: dict[str, str | sdm.Stype] | None = None,
    ) -> SeldonRegressor:
        """Keep ``X`` and ``y`` as the context of every member; ``y`` is one finite number per row, ``(n_rows,)`` or ``(n_rows, 1)``.

        ``stypes`` sets the type of the columns it names (``"numerical"``,
        ``"categorical"``), over sdm's inference, which reads a column of few
        distinct numbers as categorical.
        """
        X = _frame(X)
        y = np.asarray(y, dtype=float)
        if y.shape not in ((len(X),), (len(X), 1)):
            raise ValueError(f"y must hold one target per row of X, {len(X)} rows; got {y.shape}")
        y = y.reshape(-1)
        if not np.isfinite(y).all():
            raise ValueError("y holds NaN or infinite values")
        self._bounds = None
        if self.clip_margin is not None:
            low, high, m = float(np.min(y)), float(np.max(y)), self.clip_margin
            self._bounds = ((1 + m) * low - m * high, (1 + m) * high - m * low)
        self._stypes = sdm.infer_stypes(X, _low_cardinality="infer")
        if stypes:
            if unknown := sorted(set(stypes) - set(self._stypes)):
                raise ValueError(f"stypes names columns X does not have: {unknown}")
            self._stypes = {**self._stypes, **{k: sdm.Stype(v) for k, v in stypes.items()}}
        x_context = sdm.TableTensor.from_pandas(df=X, stypes=self._stypes)
        y_context = sdm.TableTensor.from_pandas(
            df=pd.DataFrame({_TARGET: y}), stypes={_TARGET: "numerical"}
        )
        self._subsampled = _fit_members(
            self.estimator,
            x_context,
            y_context,
            n_estimators=self.n_estimators,
            max_context_size=self.max_context_size,
            random_state=self.random_state,
            estimator_batch_size=self.estimator_batch_size,
            device=self.device,
            amp=self.amp,
            cudnn_attention=self.cudnn_attention,
        )
        return self

    def predict_quantiles(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """The quantiles of each row of ``X``, ascending, clamped by ``clip_margin``: ``(n_rows, num_quantiles)``."""
        empty = torch.empty(0, self.estimator.model.num_quantiles)
        return torch.cat([empty, *(q.cpu() for q in self._quantiles(X))]).numpy()

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """The mean of each row's quantiles, ``(n_rows,)``."""
        empty = torch.empty(0)
        return torch.cat([empty, *(q.mean(dim=-1).cpu() for q in self._quantiles(X))]).numpy()

    def to(self, device: str | torch.device) -> SeldonRegressor:
        """Move the regressor, fitted or not, to ``device`` (see ``SeldonClassifier.to``)."""
        self.device = torch.device(device)
        _move_estimator(self.estimator, self.device)
        return self

    def _quantiles(self, X: pd.DataFrame | np.ndarray) -> Iterator[Tensor]:
        """The clamped float32 quantiles of the rows of ``X``, ``_PREDICT_ROWS`` rows at a time, on the device."""
        X = _frame(X)
        query = sdm.TableTensor.from_pandas(df=X, stypes=self._stypes, device=self.device)
        for start in range(0, len(X), _PREDICT_ROWS):
            chunk = query[start : start + _PREDICT_ROWS]
            if self._subsampled:
                chunk = chunk.expand(self.n_estimators, *chunk.size())
            with _inference(self.device, self.amp, self.cudnn_attention):
                quantiles = self.estimator.predict(chunk).numerical.float()
            if self._bounds is not None:
                quantiles = quantiles.clamp(*self._bounds)
            yield quantiles
