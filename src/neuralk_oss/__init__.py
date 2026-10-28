"""Neuralk-AI open-source models."""

from neuralk_oss.inference import SeldonClassifier, SeldonICL, SeldonRecipe
from neuralk_oss.model import Seldon, SeldonConfig

__all__ = ["Seldon", "SeldonClassifier", "SeldonConfig", "SeldonICL", "SeldonRecipe"]
__version__ = "0.1.0"
