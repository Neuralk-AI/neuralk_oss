# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0

"""Seldon as an sdm in-context model: sdm's preprocessing, ensembling, context cache and error-correcting output codes around ``Seldon``."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from sdm import Recipe, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import ECOC, ICLModel
from torch import Tensor

from neuralk_oss.inference.recipe import SeldonRecipe
from neuralk_oss.model import Seldon

_CACHE_KEY = "seldon"
# The context rows themselves, kept instead of their encoding past the budget.
_RAW_KEY = "seldon_raw"


class SeldonICL(ICLModel):
    """``Seldon`` behind sdm's ``fit`` / ``predict``.

    ``fit`` runs each member's context rows through the model once and caches
    what the query rows read of them (``Seldon.encode_context``); ``predict``
    runs the query rows against it, ``query_chunk_size`` rows at a time.
    Encodings past ``max_cache_gb`` in total are not kept: those members keep
    their context rows and encode them at each ``predict``. Members run in
    batches of the same shape, so the last batch's encoding tells a batch's
    size before it runs, and members past the budget are not encoded at ``fit``.
    Above ``max_classes`` classes, sdm's error-correcting output codes run the
    model once per code.

    Parameters
    ----------
    model : Seldon
        A classifier.
    recipe : SeldonRecipe, optional
        The preprocessing; ``SeldonRecipe()`` by default.
    query_chunk_size : int, default=8192
        Query rows of one table run through the model together.
    max_cache_gb : float or None, default=8.0
        GiB of context encodings ``fit`` keeps over all members, None for no
        bound. The encoding grows with features as well as rows (Seldon's is
        about 0.75 GiB per member at 500 features), and sdm keeps it in pinned
        CPU memory, which a large table can exhaust.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset({Stype.numerical})
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset({Stype.categorical})
    supports_multi_target: ClassVar[bool] = False
    supports_related_tables: ClassVar[bool] = False

    def __init__(
        self,
        model: Seldon,
        recipe: SeldonRecipe | None = None,
        query_chunk_size: int = 8192,
        max_cache_gb: float | None = 8.0,
    ) -> None:
        if model.max_classes <= 0:
            raise ValueError("SeldonICL runs a classifier; the model is a regressor")
        super().__init__(task="classification")
        self.model = model
        self.recipe = SeldonRecipe() if recipe is None else recipe
        self.query_chunk_size = query_chunk_size
        self.max_cache_gb = max_cache_gb
        self._cached_bytes = 0
        self._member_bytes = 0.0
        self.ecoc = ECOC(max_classes=model.max_classes)
        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        return SeldonRecipe().build()

    def fit(self, *args: Any, recipe: Recipe | None = None, **kwargs: Any) -> None:
        """sdm's ``fit``, with this estimator's ``recipe`` unless one is given."""
        self._cached_bytes = self._member_bytes = 0
        super().fit(*args, recipe=self.recipe.build() if recipe is None else recipe, **kwargs)

    def _forward(
        self,
        x_context: TableTensor | None,
        y_context: TableTensor | None,
        x_query: TableTensor | None,
        related_context_tables: RelatedTables[TableTensor] | None,
        related_query_tables: RelatedTables[TableTensor] | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        *,
        categorical_mask: Tensor,
        **kwargs: Any,
    ) -> TableTensor:
        if x_query is None:
            assert x_context is not None
            x = x_context.numerical
        elif x_context is None:
            x = x_query.numerical
        else:
            x = torch.cat([x_context.numerical, x_query.numerical], dim=-2)

        if y_context is not None:
            y = y_context.categorical.code.squeeze(-1)
            classes = y_context.categorical.categories[0]
        else:
            assert cache is not None
            classes = cache["classes"]
            y = x.new_empty((*x.shape[:-2], 0), dtype=torch.int64)

        out = self.ecoc(
            model=self._network,
            x=x,
            y=y,
            num_classes=len(classes),
            num_members=x.size(0) if x.dim() > 2 else 1,
            cache=cache,
            generator=generator,
            categorical_mask=categorical_mask.expand(*x.shape[:-2], -1),
        )
        return TableTensor(
            columns={Stype.numerical: [str(i) for i in classes.tolist()]},
            numerical=out,
        )

    def _network(
        self,
        x: Tensor,
        y: Tensor,
        *,
        categorical_mask: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """The model on ``x`` ``(..., R, C)``, its first ``y.shape[-1]`` rows the context: logits of the others, ``(..., R - y.shape[-1], max_classes)``.

        Recording, ``x`` holds the context rows only and their encoding goes
        to ``cache`` (the rows themselves past ``max_cache_gb``); replaying,
        ``x`` holds query rows only, ``y`` none.
        """
        lead, (R, C) = x.shape[:-2], x.shape[-2:]
        X = x.reshape(-1, R, C)
        mask = categorical_mask.expand(*lead, C).reshape(-1, C)
        train_size = y.shape[-1]
        if cache is not None and cache.is_replaying:
            if _RAW_KEY in cache:
                raw = cache[_RAW_KEY]
                context = self.model.encode_context(raw["x"], raw["y"], mask)
            else:
                context = cache[_CACHE_KEY]
            out = self._predict(X, context, mask)
        else:
            x_context, y_context = X[:, :train_size], y.reshape(-1, train_size)
            raw = {"x": x_context, "y": y_context}
            # Without query rows, a batch the last one's size puts past the budget is not encoded.
            if (
                cache is not None
                and R == train_size
                and self._over_budget(self._member_bytes * len(X))
            ):
                cache[_RAW_KEY] = raw
                return X.new_zeros((*lead, 0, self.model.max_classes))
            context = self.model.encode_context(x_context, y_context, mask)
            if cache is not None:
                size = _nbytes(context)
                self._member_bytes = size / len(X)
                if self._over_budget(size):
                    cache[_RAW_KEY] = raw
                else:
                    self._cached_bytes += size
                    cache[_CACHE_KEY] = context
            out = self._predict(X[:, train_size:], context, mask)
        return out.view(*lead, *out.shape[-2:])

    def _over_budget(self, size: float) -> bool:
        """Whether caching ``size`` more bytes passes ``max_cache_gb``."""
        budget = self.max_cache_gb
        return budget is not None and self._cached_bytes + size > budget * 2**30

    def _predict(self, X: Tensor, context: dict[str, Any], mask: Tensor) -> Tensor:
        if X.shape[1] == 0:
            return X.new_zeros((X.shape[0], 0, self.model.max_classes))
        return self.model.predict_query(X, context, mask, chunk_size=self.query_chunk_size)


def _nbytes(context: dict[str, Any]) -> int:
    """Bytes of the tensors of an encoded context."""
    tensors = [t for v in context.values() for t in (v if isinstance(v, list) else [v])]
    return sum(t.nbytes for t in tensors)
