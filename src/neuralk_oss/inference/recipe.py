"""Seldon's preprocessing: what sdm does to a table before the model reads it, and to the model's outputs.

``SeldonRecipe`` builds an ``sdm.Recipe`` from one method per step, each
returning fresh processors. To change a step, subclass it and override that
method (or set a class attribute), then hand an instance to the estimator::

    class NoPowerTransform(SeldonRecipe):
        def numerical_transforms(self):
            return [sp.Identity(), sp.RankGaussian(max_knots=self.rank_gaussian_knots)]

    SeldonClassifier.from_pretrained(path, recipe=NoPowerTransform())

The default is sdm's recipe for this architecture, step for step.
"""

from __future__ import annotations

import sdm.processing as sp
import torch


class SeldonRecipe:
    """The steps of Seldon's preprocessing, each an overridable method.

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
    """

    max_features = 500
    standardize_eps = 1e-6
    clip_value = 100.0
    rank_gaussian_knots = 8192
    clip_sigma = 4.0
    category_count_min_cardinality = 50

    def build(self) -> sp.Recipe:
        """The ``sdm.Recipe`` of these steps."""
        return sp.Recipe(features=self.features(), target=self.target(), output=self.output())

    def features(self) -> list[sp.Processor]:
        """The steps of the feature columns: per type, then shuffled, cut to ``max_features`` and cast."""
        return [
            sp.StypeDispatch(
                numerical=self.numerical_features(),
                categorical=self.categorical_features(),
            ),
            sp.ShuffleColumns(method="latin"),
            sp.SelectColumns(self.max_features, method="first"),
            sp.Cast(torch.float32),
        ]

    def numerical_features(self) -> list[sp.Processor]:
        """The steps of a numerical column."""
        return [self.numerical_processor(), sp.FlipSign()]

    def categorical_features(self) -> list[sp.Processor]:
        """The steps of a categorical column: its codes, with their counts for a large vocabulary, read as numbers."""
        return [
            sp.AlignCategories(sort_by="value"),
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
        """The transforms the ensemble members take in turn, one each."""
        return [
            sp.Identity(),
            sp.PowerTransform(),
            [sp.RobustScale(), sp.ClipSoft(3.0)],
            sp.RankGaussian(max_knots=self.rank_gaussian_knots),
        ]

    def target(self) -> list[sp.Processor]:
        """The steps of the target: class codes shifted per member, or a standardized number."""
        return [
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(sort_by="value"),
                    sp.ShuffleCategories(method="shift"),
                ],
                numerical=[sp.Standardize(), sp.FlipSign()],
            ),
        ]

    def output(self) -> sp.TaskDispatch:
        """The steps of the members' outputs: class logits averaged then softmaxed; sorted quantiles with a trimmed mean."""
        return sp.TaskDispatch(
            classification=[sp.AverageEstimators(), sp.Softmax()],
            regression=[sp.SortQuantiles(), sp.AverageEstimators(trim_fraction=0.2)],
        )
