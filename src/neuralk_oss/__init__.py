# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: MIT

"""Neuralk-AI open-source models."""

from importlib.metadata import version

from neuralk_oss.inference import SeldonClassifier, SeldonICL, SeldonRecipe
from neuralk_oss.model import Seldon, SeldonConfig

__all__ = ["Seldon", "SeldonClassifier", "SeldonConfig", "SeldonICL", "SeldonRecipe"]
__version__ = version("neuralk-oss")
