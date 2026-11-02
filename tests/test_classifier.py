# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from neuralk_oss import SeldonClassifier, SeldonRecipe
from neuralk_oss.inference.classifier import MEMBER_BATCH_CELLS, member_batch_size


def _frame(n, classes, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    X.iloc[rng.random((n, 4)) < 0.1] = np.nan
    X["cat"] = pd.Categorical(rng.choice(["u", "v", "w"], size=n))
    y = pd.Series(np.array([f"c{k}" for k in rng.integers(0, classes, n)]))
    return X, y


@pytest.mark.parametrize("classes", [2, 4, 13])
def test_fit_predict_proba(tiny_model, classes):
    X, y = _frame(80, classes)
    clf = SeldonClassifier(tiny_model, n_estimators=3, device="cpu")
    proba = clf.fit(X[:60], y[:60]).predict_proba(X[60:])
    assert proba.shape == (20, len(np.unique(y[:60])))
    np.testing.assert_allclose(proba.sum(1), 1.0, rtol=1e-5)
    assert set(clf.predict(X[60:])) <= set(clf.classes_)


def test_query_chunks_match(tiny_model):
    X, y = _frame(80, 3)
    whole = SeldonClassifier(tiny_model, n_estimators=2, device="cpu")
    chunked = SeldonClassifier(tiny_model, n_estimators=2, device="cpu", query_chunk_size=7)
    np.testing.assert_allclose(
        chunked.fit(X[:60], y[:60]).predict_proba(X[60:]),
        whole.fit(X[:60], y[:60]).predict_proba(X[60:]),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(("classes", "max_context_size"), [(3, 200_000), (13, 200_000), (3, 40)])
def test_member_batches_match(tiny_model, classes, max_context_size):
    X, y = _frame(80, classes)
    kwargs = dict(n_estimators=4, device="cpu", max_context_size=max_context_size)
    one = SeldonClassifier(tiny_model, estimator_batch_size=1, **kwargs)
    for batch in (None, "auto", 3):
        batched = SeldonClassifier(tiny_model, estimator_batch_size=batch, **kwargs)
        np.testing.assert_allclose(
            batched.fit(X[:60], y[:60]).predict_proba(X[60:]),
            one.fit(X[:60], y[:60]).predict_proba(X[60:]),
            rtol=1e-5,
            atol=1e-6,
        )


def test_member_batch_size():
    assert member_batch_size(500, 10, 4) == 4
    assert member_batch_size(1_000, 200, 4) == 2
    assert member_batch_size(4_000, 200, 4) == 1
    assert member_batch_size(1, 1, 8) == 8
    assert member_batch_size(MEMBER_BATCH_CELLS, 2, 8) == 1


def test_subsampled_context(tiny_model):
    X, y = _frame(80, 3)
    clf = SeldonClassifier(tiny_model, n_estimators=2, device="cpu", max_context_size=25)
    assert clf.fit(X[:60], y[:60]).predict_proba(X[60:]).shape == (20, 3)


def test_recipe_override(tiny_model):
    class FewerFeatures(SeldonRecipe):
        max_features = 2

    X, y = _frame(80, 3)
    clf = SeldonClassifier(tiny_model, n_estimators=2, device="cpu", recipe=FewerFeatures())
    default = SeldonClassifier(tiny_model, n_estimators=2, device="cpu")
    p = clf.fit(X[:60], y[:60]).predict_proba(X[60:])
    assert not np.allclose(p, default.fit(X[:60], y[:60]).predict_proba(X[60:]))


def test_numpy_input(tiny_model):
    X, y = _frame(50, 2)
    clf = SeldonClassifier(tiny_model, n_estimators=2, device="cpu")
    assert clf.fit(X[list("abcd")].to_numpy()[:40], y[:40].to_numpy()).predict_proba(
        X[list("abcd")].to_numpy()[40:]
    ).shape == (10, 2)


def test_from_pretrained_model_names(tiny_model, tmp_path, monkeypatch):
    from neuralk_oss.model import Seldon

    loaded = []
    monkeypatch.setattr(
        Seldon, "from_pretrained", lambda name, **_: loaded.append(str(name)) or tiny_model
    )
    for name in ["seldon", "seldon-mini", "seldon-mini-classifier", "org/repo"]:
        SeldonClassifier.from_pretrained(name)
    # A local directory named like a model is read as such.
    (tmp_path / "seldon").mkdir()
    monkeypatch.chdir(tmp_path)
    SeldonClassifier.from_pretrained("seldon")
    assert loaded == [
        "seldon-classifier",
        "seldon-mini-classifier",
        "seldon-mini-classifier",
        "org/repo",
        "seldon",
    ]
