# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0

"""Inference: Seldon behind sdm's preprocessing and ensembling, as a classifier."""

from neuralk_oss.inference.classifier import SeldonClassifier
from neuralk_oss.inference.estimator import SeldonICL
from neuralk_oss.inference.recipe import SeldonRecipe

__all__ = ["SeldonClassifier", "SeldonICL", "SeldonRecipe"]
