# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0

"""Seldon as a scikit-learn style classifier over pandas tables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import sdm
import torch

from neuralk_oss.inference._inference import (
    _CONFIGS,
    _TARGET,
    MEMBER_BATCH_CELLS,
    _config_params,
    _fit_members,
    _frame,
    _inference,
    _move_estimator,
    _read_config,
    member_batch_size,
)
from neuralk_oss.inference.estimator import SeldonICL
from neuralk_oss.inference.recipe import SeldonRecipe
from neuralk_oss.model import Seldon
from neuralk_oss.model.seldon import HUB_CHECKPOINTS

# Importable from here as before the regressor shared them.
__all__ = ["MEMBER_BATCH_CELLS", "SeldonClassifier", "member_batch_size"]


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
        if model.max_classes <= 0:
            raise ValueError("SeldonClassifier runs a classifier; the model is a regressor")
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
            name_or_path, task="classification", subfolder=subfolder, revision=revision, token=token
        )
        return cls(model, **kwargs)

    @classmethod
    def from_config(cls, config: str | os.PathLike, **kwargs: Any) -> SeldonClassifier:
        """The classifier of a YAML config: a packaged one by name, such as ``"seldon-mini"``, or a file.

        The config maps ``checkpoint`` to ``from_pretrained``'s first argument,
        ``recipe`` to ``SeldonRecipe``'s keyword arguments, and any other key to
        a keyword argument of ``from_pretrained``; ``kwargs`` override its keys.
        """
        params = _config_params(config, kwargs, "classification")
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

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Class probabilities of the rows of ``X``, ``(n_rows, n_classes)`` in the order of ``classes_``."""
        X = _frame(X)
        query = sdm.TableTensor.from_pandas(df=X, stypes=self._stypes, device=self.device)
        if self._subsampled:
            query = query.expand(self.n_estimators, *query.size())
        with _inference(self.device, self.amp, self.cudnn_attention):
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
        _move_estimator(self.estimator, self.device)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """The most probable class of each row of ``X``."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
