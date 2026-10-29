"""Golden tests: outputs of the committed tiny checkpoints on fixed tables, against saved references.

A change that moves any of them (the model, sdm's preprocessing, a torch
upgrade) fails here. If the change is intended, regenerate the references
with ``UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py`` and review them.
"""

import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from neuralk_oss import Seldon, SeldonClassifier

GOLDEN = Path(__file__).parent / "golden"
UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"


@functools.cache
def checkpoint(name: str) -> Seldon:
    return Seldon.from_pretrained(GOLDEN / name)


def check(name: str, actual: np.ndarray) -> None:
    path = GOLDEN / "expected" / f"{name}.npy"
    if UPDATE:
        path.parent.mkdir(exist_ok=True)
        np.save(path, actual)
        return
    expected = np.load(path)
    assert actual.shape == expected.shape
    # Loose enough for BLAS and torch versions, tight enough for any real change.
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def _numerical(rng, n, columns=4, nan=0.0):
    X = pd.DataFrame(rng.normal(size=(n, columns)), columns=[f"f{i}" for i in range(columns)])
    return X.mask(rng.random(X.shape) < nan)


def _labels(rng, n, classes):
    return pd.Series([f"c{k}" for k in rng.integers(0, classes, n)])


def binary_numerical(rng):
    X = _numerical(rng, 80)
    y = (X["f0"] + X["f1"] > 0).astype(int)
    return X, y, {}


def multiclass_mixed(rng):
    X = _numerical(rng, 80, nan=0.1)
    X["cat"] = pd.Categorical(rng.choice(["u", "v", "w"], size=80))
    return X, _labels(rng, 80, 4), {}


def many_classes(rng):
    # More classes than the head's 10: error-correcting output codes.
    return _numerical(rng, 120), _labels(rng, 120, 13), {}


def high_cardinality_categorical(rng):
    # 60 categories, above the recipe's 50: the column gains its category counts.
    X = _numerical(rng, 120, columns=2)
    X["cat"] = pd.Categorical(rng.choice([f"k{i}" for i in range(60)], size=120))
    return X, _labels(rng, 120, 3), {}


def unseen_category(rng):
    X = _numerical(rng, 80, columns=2)
    X["cat"] = pd.Categorical(rng.choice(["u", "v"], size=80), categories=["u", "v", "new"])
    X.loc[70:, "cat"] = "new"  # only in the test rows
    return X, _labels(rng, 80, 3), {}


def degenerate_columns(rng):
    X = _numerical(rng, 80, columns=3)
    X["constant"] = 1.0
    X["all_nan"] = np.nan
    return X, _labels(rng, 80, 3), {}


def subsampled_context(rng):
    X, y, _ = multiclass_mixed(rng)
    return X, y, {"max_context_size": 25}


def numpy_input(rng):
    return _numerical(rng, 80).to_numpy(), _labels(rng, 80, 3).to_numpy(), {}


def single_query_row(rng):
    X, y, _ = multiclass_mixed(rng)
    return X[:61], y[:61], {}


def variant_checkpoint(rng):
    X, y, _ = multiclass_mixed(rng)
    return X, y, {"checkpoint": "seldon-tiny-classifier-variant"}


CLASSIFIER_CASES = [
    binary_numerical,
    multiclass_mixed,
    many_classes,
    high_cardinality_categorical,
    unseen_category,
    degenerate_columns,
    subsampled_context,
    numpy_input,
    single_query_row,
    variant_checkpoint,
]


@pytest.mark.parametrize("case", CLASSIFIER_CASES, ids=lambda case: case.__name__)
def test_classifier_golden(case):
    X, y, kwargs = case(np.random.default_rng(0))
    model = checkpoint(kwargs.pop("checkpoint", "seldon-tiny-classifier"))
    clf = SeldonClassifier(model, n_estimators=3, device="cpu", **kwargs)
    n_train = 60 if len(X) <= 80 else 100
    proba = clf.fit(X[:n_train], y[:n_train]).predict_proba(X[n_train:])
    check(f"classifier-{case.__name__}", proba)


@pytest.mark.parametrize(
    "name",
    ["seldon-tiny-classifier", "seldon-tiny-classifier-variant", "seldon-tiny-regressor"],
)
@torch.no_grad()
def test_model_golden(name):
    model = checkpoint(name)
    g = torch.Generator().manual_seed(0)
    X = torch.randn(2, 40, 5, generator=g)
    X[torch.rand(X.shape, generator=g) < 0.1] = float("nan")
    mask = torch.tensor([[False, True, False, False, True]] * 2)
    if model.num_quantiles:
        y = torch.randn(2, 30, generator=g)
    else:
        y = torch.randint(0, 3, (2, 30), generator=g)
    check(f"model-{name}", model(X, y, 30, categorical_mask=mask).numpy())
