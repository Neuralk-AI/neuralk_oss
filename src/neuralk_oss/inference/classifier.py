"""Seldon as a scikit-learn style classifier over pandas tables."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import numpy as np
import pandas as pd
import sdm
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from neuralk_oss.inference.estimator import SeldonICL
from neuralk_oss.inference.recipe import SeldonRecipe
from neuralk_oss.model import Seldon

_TARGET = "target"
# Every attention kernel PyTorch has but cuDNN's (see ``cudnn_attention``).
_NON_CUDNN_ATTENTION = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


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
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.n_estimators = n_estimators
        self.max_context_size = max_context_size
        self.amp = amp
        self.cudnn_attention = cudnn_attention
        self.random_state = random_state
        self.estimator = SeldonICL(model, recipe, query_chunk_size).to(self.device)

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
        """The classifier of a saved model: a local directory, a checkpoint name such as ``"seldon-small-classifier"``, or a Hub repository (see ``Seldon.from_pretrained``); ``kwargs`` are the constructor's."""
        model = Seldon.from_pretrained(
            name_or_path, subfolder=subfolder, revision=revision, token=token
        )
        return cls(model, **kwargs)

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
        with self._inference():
            self.estimator.fit(
                x=x_context.to(self.device),
                y=y_context.to(self.device),
                num_estimators=num_estimators,
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


def _frame(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """``X`` as a DataFrame with string column names, which sdm requires."""
    if isinstance(X, pd.DataFrame):
        return X.rename(columns=str)
    X = np.asarray(X)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
