# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0

"""Golden tests: outputs of the committed tiny checkpoints on fixed tables, against saved references.

A change that moves any of them (the model, sdm's preprocessing, a torch
upgrade) fails here. If the change is intended, regenerate the references
with ``UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py`` and review them.

With CUDA, the same cases also run on it under float16 autocast, the
default inference path through the fused kernels, with a looser tolerance.
The model's CUDA outputs are checked against the CPU references; the
classifier's have their own, in ``expected/cuda``: its seeded generator is on
the model's device, and CUDA's random streams differ from CPU's, so each
member is preprocessed differently. Regenerating them needs a CUDA machine.
"""

import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from neuralk_oss import Seldon, SeldonClassifier, SeldonRecipe

GOLDEN = Path(__file__).parent / "golden"
UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"

# CPU in float32; CUDA in float16 autocast, as the classifier runs by default.
DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    ),
]


def tolerance(test: str, device: str) -> dict[str, float]:
    """``assert_allclose``'s tolerances against the references."""
    if device == "cpu":
        # Loose enough for BLAS and torch versions, tight enough for any real change.
        return {"rtol": 1e-4, "atol": 1e-5}
    # About 4x the float16 drift from float32 on an H100: 2.3e-4 on probabilities,
    # 2.6e-3 on logits, which leaves room for other GPUs.
    return {"rtol": 0.0, "atol": {"classifier": 1e-3, "model": 1e-2}[test]}


@functools.cache
def checkpoint(name: str, device: str) -> Seldon:
    # One per device: the classifier moves its model in place.
    return Seldon.from_pretrained(GOLDEN / name, device=device)


def check(path: Path, actual: np.ndarray, tolerance: dict[str, float], *, reference: bool) -> None:
    """Compare ``actual`` to the reference at ``path``; with ``UPDATE_GOLDEN=1``, overwrite it if ``reference``."""
    if UPDATE:
        if reference:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, actual)
        return
    expected = np.load(path)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, **tolerance)


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


def recipe_two_transforms(rng):
    X, y, _ = multiclass_mixed(rng)
    recipe = SeldonRecipe(
        transforms=("identity", "rank_gaussian"),
        shuffle_categories=False,
        feature_shuffle="random",
        softmax_temperature=0.8,
    )
    return X, y, {"recipe": recipe}


def recipe_single_view(rng):
    # One transform, no category or column shuffle, sign flip or class shift:
    # the members are identical, and their probabilities are averaged.
    X, y, _ = multiclass_mixed(rng)
    recipe = SeldonRecipe(
        transforms=("robust",),
        shuffle_categories=False,
        flip_sign=False,
        feature_shuffle=None,
        class_shift=False,
        average_logits=False,
    )
    return X, y, {"recipe": recipe}


def shuffled_categories(rng):
    X, y, _ = multiclass_mixed(rng)
    # Blocks of 2 members, so the third of the golden tests' 3 shuffles.
    recipe = SeldonRecipe(transforms=("identity", "power"))
    return X, y, {"recipe": recipe}


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
    recipe_two_transforms,
    recipe_single_view,
    shuffled_categories,
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", CLASSIFIER_CASES, ids=lambda case: case.__name__)
def test_classifier_golden(case, device):
    X, y, kwargs = case(np.random.default_rng(0))
    model = checkpoint(kwargs.pop("checkpoint", "seldon-tiny-classifier"), device)
    clf = SeldonClassifier(model, n_estimators=3, device=device, **kwargs)
    n_train = 60 if len(X) <= 80 else 100
    proba = clf.fit(X[:n_train], y[:n_train]).predict_proba(X[n_train:])
    expected = GOLDEN / "expected" / ("cuda" if device == "cuda" else "")
    check(
        expected / f"classifier-{case.__name__}.npy",
        proba,
        tolerance("classifier", device),
        reference=True,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name",
    ["seldon-tiny-classifier", "seldon-tiny-classifier-variant", "seldon-tiny-regressor"],
)
@torch.inference_mode()
def test_model_golden(name, device):
    model = checkpoint(name, device)
    g = torch.Generator().manual_seed(0)
    X = torch.randn(2, 40, 5, generator=g)
    X[torch.rand(X.shape, generator=g) < 0.1] = float("nan")
    mask = torch.tensor([[False, True, False, False, True]] * 2)
    if model.num_quantiles:
        y = torch.randn(2, 30, generator=g)
    else:
        y = torch.randint(0, 3, (2, 30), generator=g)
    with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        out = model(X.to(device), y.to(device), 30, categorical_mask=mask)
    check(
        GOLDEN / "expected" / f"model-{name}.npy",
        out.float().cpu().numpy(),
        tolerance("model", device),
        reference=device == "cpu",
    )
