# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0

"""Seldon's preprocessing: what sdm does to a table before the model reads it, and to the model's outputs.

``SeldonRecipe`` builds an ``sdm.Recipe`` from one method per step, each
returning fresh processors. Its parameters are keyword arguments::

    SeldonClassifier.from_pretrained(path, recipe=SeldonRecipe(transforms=("identity", "robust")))

To change a step beyond them, subclass it and override that method (or set a
class attribute), then hand an instance to the estimator::

    class NoPowerTransform(SeldonRecipe):
        def numerical_transforms(self):
            return [sp.Identity(), sp.RankGaussian(max_knots=self.rank_gaussian_knots)]

    SeldonClassifier.from_pretrained(path, recipe=NoPowerTransform())

The default is sdm's recipe for this architecture, step for step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import sdm.processing as sp
import torch
from sdm import EnsembleTable


class SeldonRecipe:
    """The steps of Seldon's preprocessing, each an overridable method.

    Parameters
    ----------
    **params
        Any of the attributes below, overriding its default.

    Attributes
    ----------
    max_features : int
        Columns the model reads; more are cut after the column shuffle.
    standardize_eps : float
        Epsilon of the per-column standardization.
    clip_value : float
        Bound of the standardized values, before the transform.
    rank_gaussian_knots : int
        Knots of the rank-Gaussian transform.
    clip_sigma : float
        Bound, in standard deviations, after the transform.
    category_count_min_cardinality : int
        A categorical column with at least this many categories gains a column
        of each category's count.
    transforms : tuple of str
        The numerical transforms the ensemble members take in turn, one each:
        ``"identity"``, ``"power"`` (Yeo-Johnson), ``"robust"`` (median and
        quartile scaling, soft-clipped to +-3) and ``"rank_gaussian"``; their
        number is the number of distinct views.
    shuffle_categories : bool
        Whether half the members read each categorical column with its codes
        in a random order, one permutation per column, instead of sorted by
        value: members in blocks of ``len(transforms)``, sorted then shuffled,
        so each view of the numbers meets both orders. With
        ``n_estimators`` up to ``len(transforms)``, no member shuffles.
    flip_sign : bool
        Whether each member negates a random half of its numerical columns.
    feature_shuffle : {"latin", "random"} or None
        How each member permutes its columns: coupled Latin permutations,
        independent ones, or none.
    class_shift : bool
        Whether each member shifts the class codes cyclically, by its own offset.
    average_logits : bool
        Whether the members' logits are averaged before the softmax; otherwise
        their probabilities are averaged after it.
    softmax_temperature : float
        Divisor of the logits before the softmax.
    """

    max_features = 500
    standardize_eps = 1e-6
    clip_value = 100.0
    rank_gaussian_knots = 8192
    clip_sigma = 4.0
    category_count_min_cardinality = 50
    transforms: tuple[str, ...] = ("identity", "power", "robust", "rank_gaussian")
    shuffle_categories = True
    flip_sign = True
    feature_shuffle: Literal["latin", "random"] | None = "latin"
    class_shift = True
    average_logits = True
    softmax_temperature = 1.0

    def __init__(self, **params: Any) -> None:
        for name, value in params.items():
            cls = type(self)
            if name.startswith("_") or not hasattr(cls, name) or callable(getattr(cls, name)):
                raise TypeError(f"SeldonRecipe has no parameter {name!r}")
            setattr(self, name, value)
        unknown = set(self.transforms) - set(_TRANSFORMS)
        if not self.transforms or unknown:
            raise ValueError(
                f"transforms must be a non-empty tuple of {sorted(_TRANSFORMS)}, got {self.transforms!r}"
            )

    def build(self) -> sp.Recipe:
        """The ``sdm.Recipe`` of these steps."""
        return sp.Recipe(features=self.features(), target=self.target(), output=self.output())

    def features(self) -> list[sp.Processor]:
        """The steps of the feature columns: per type, then shuffled, cut to ``max_features`` and cast."""
        shuffle = [sp.ShuffleColumns(method=self.feature_shuffle)] if self.feature_shuffle else []
        return [
            sp.StypeDispatch(
                numerical=self.numerical_features(),
                categorical=self.categorical_features(),
            ),
            *shuffle,
            sp.SelectColumns(self.max_features, method="first"),
            sp.Cast(torch.float32),
        ]

    def numerical_features(self) -> list[sp.Processor]:
        """The steps of a numerical column."""
        return [self.numerical_processor(), *([sp.FlipSign()] if self.flip_sign else [])]

    def categorical_features(self) -> list[sp.Processor]:
        """The steps of a categorical column: its codes (shuffled in some members), with their counts for a large vocabulary, read as numbers."""
        shuffle = sp.ShuffleCategories(method="random")
        block = len(self.transforms)
        return [
            sp.AlignCategories(sort_by="value"),
            *(
                [_BlockChoice(sp.Identity(), shuffle, block=block)]
                if self.shuffle_categories
                else []
            ),
            sp.AddCategoryCounts(min_cardinality=self.category_count_min_cardinality),
            sp.ToNumerical(),
            self.numerical_processor(),
        ]

    def numerical_processor(self) -> sp.Sequential:
        """The scaling of a column of numbers: standardized and clipped, transformed, then clipped in standard deviations."""
        return sp.Sequential(
            # In single precision, a few huge outliers collapse the
            # standardized values of all other rows onto one value.
            sp.Cast(torch.float64),
            sp.DropConstantColumns(),
            sp.Standardize(eps=self.standardize_eps),
            sp.Clip(-self.clip_value, self.clip_value),
            sp.Choice(*self.numerical_transforms(), method="round_robin"),
            sp.ClipSigma(threshold=self.clip_sigma),
        )

    def numerical_transforms(self) -> list[sp.Processor | list[sp.Processor]]:
        """The transforms the ensemble members take in turn, one each: those named in ``transforms``."""
        return [_TRANSFORMS[name](self) for name in self.transforms]

    def target(self) -> list[sp.Processor]:
        """The steps of the target: class codes shifted per member, or a standardized number."""
        return [
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(sort_by="value"),
                    *([sp.ShuffleCategories(method="shift")] if self.class_shift else []),
                ],
                numerical=[sp.Standardize(), sp.FlipSign()],
            ),
        ]

    def output(self) -> sp.TaskDispatch:
        """The steps of the members' outputs: class logits averaged then softmaxed (or the reverse); sorted quantiles with a trimmed mean."""
        classification = [sp.AverageEstimators(), sp.Softmax(temperature=self.softmax_temperature)]
        return sp.TaskDispatch(
            classification=classification if self.average_logits else classification[::-1],
            regression=[sp.SortQuantiles(), sp.AverageEstimators(trim_fraction=0.2)],
        )


class _BlockChoice(sp.Choice):
    """A choice by blocks of members: the first ``block`` take option 0, the next option 1, and so on, cycling."""

    def __init__(self, *options: object, block: int) -> None:
        super().__init__(*options, method="round_robin")
        self.block = block

    # shortcut: overrides sdm's private hook, pinned at 0.2.1; recheck on an sdm upgrade.
    def _draw_option_ids(self, ensemble_table: EnsembleTable, **kwargs: Any) -> tuple[int, ...]:
        return tuple(
            (member // self.block) % len(self.options) for member in range(len(ensemble_table))
        )


_TRANSFORMS: dict[str, Callable[[SeldonRecipe], sp.Processor | list[sp.Processor]]] = {
    "identity": lambda recipe: sp.Identity(),
    "power": lambda recipe: sp.PowerTransform(),
    "robust": lambda recipe: [sp.RobustScale(), sp.ClipSoft(3.0)],
    "rank_gaussian": lambda recipe: sp.RankGaussian(max_knots=recipe.rank_gaussian_knots),
}
