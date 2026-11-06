# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0

import copy
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sdm
import sdm.processing as sp
import torch

from neuralk_oss import SeldonClassifier, SeldonRecipe, SeldonRegressor
from neuralk_oss.inference import regressor
from neuralk_oss.model import Seldon
from tests.test_model import _save_checkpoints

GOLDEN = Path(__file__).parent / "golden"
TINY = GOLDEN / "seldon-tiny-regressor"
CONFIGS = Path(__file__).parents[1] / "src" / "neuralk_oss" / "configs"


def _frame(n, seed=0):
    """A table with NaNs and a categorical column; its target spans little of its spread, so the tiny regressor's quantiles pass its range."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    X.iloc[rng.random((n, 4)) < 0.1] = np.nan
    X["cat"] = pd.Categorical(rng.choice(["u", "v", "w"], size=n))
    y = pd.Series(100.0 + 5.0 * (X["a"] > 0) + rng.uniform(size=n))
    return X, y


def _fit(model, *, n_estimators=3, **kwargs):
    """A regressor fitted on the first 60 rows of an 80-row table, its 20 test rows, its training target."""
    X, y = _frame(80)
    reg = SeldonRegressor(model, n_estimators=n_estimators, device="cpu", **kwargs)
    return reg.fit(X[:60], y[:60]), X[60:], y[:60]


def test_fit_predict(tiny_regressor):
    reg, X_test, _ = _fit(tiny_regressor)
    quantiles = reg.predict_quantiles(X_test)
    assert quantiles.shape == (20, 9) and quantiles.dtype == np.float32
    assert (np.diff(quantiles, axis=1) >= 0).all()
    np.testing.assert_allclose(reg.predict(X_test), quantiles.mean(axis=1), rtol=1e-6)


def test_estimator_outputs_quantile_columns(tiny_regressor):
    reg, X_test, _ = _fit(tiny_regressor)
    query = sdm.TableTensor.from_pandas(df=X_test, stypes=reg._stypes)
    with torch.inference_mode():
        out = reg.estimator.predict(query)
    # sdm's names, q001 to q999 for the released heads.
    assert list(out.columns[sdm.Stype.numerical]) == [f"q{i:03d}" for i in range(1, 10)]
    assert out.numerical.shape == (20, 9)


@pytest.mark.parametrize("clip_margin", [0.0, 0.1])
def test_clip_margin(tiny_regressor, clip_margin):
    free, X_test, y = _fit(tiny_regressor)
    clamped, _, _ = _fit(tiny_regressor, clip_margin=clip_margin)
    span = y.max() - y.min()
    bounds = (y.min() - clip_margin * span, y.max() + clip_margin * span)
    unclamped = free.predict_quantiles(X_test)
    # The clamp acts on this table.
    assert ((unclamped < bounds[0]) | (unclamped > bounds[1])).any()
    quantiles = clamped.predict_quantiles(X_test)
    np.testing.assert_allclose(quantiles, np.clip(unclamped, *bounds), rtol=1e-6)
    np.testing.assert_allclose(clamped.predict(X_test), quantiles.mean(axis=1), rtol=1e-6)


def test_predict_row_blocks_match(tiny_regressor, monkeypatch):
    reg, X_test, _ = _fit(tiny_regressor, clip_margin=0.1)
    quantiles, points = reg.predict_quantiles(X_test), reg.predict(X_test)
    calls = []
    predict = reg.estimator.predict
    monkeypatch.setattr(reg.estimator, "predict", lambda query: calls.append(1) or predict(query))
    monkeypatch.setattr(regressor, "_PREDICT_ROWS", 7)
    np.testing.assert_allclose(reg.predict_quantiles(X_test), quantiles, rtol=1e-6)
    np.testing.assert_allclose(reg.predict(X_test), points, rtol=1e-6)
    assert len(calls) == 2 * 3  # 20 rows in blocks of 7, twice


@pytest.mark.parametrize("max_context_size", [200_000, 40])
def test_member_batches_match(tiny_regressor, max_context_size):
    kwargs = dict(n_estimators=4, max_context_size=max_context_size)
    one, X_test, _ = _fit(tiny_regressor, estimator_batch_size=1, **kwargs)
    expected = one.predict_quantiles(X_test)
    for batch in (None, "auto", 3):
        batched, _, _ = _fit(tiny_regressor, estimator_batch_size=batch, **kwargs)
        np.testing.assert_allclose(batched.predict_quantiles(X_test), expected, rtol=1e-5)


def test_subsampled_context(tiny_regressor, monkeypatch):
    encode = tiny_regressor.encode_context
    rows = []
    monkeypatch.setattr(
        tiny_regressor,
        "encode_context",
        lambda X, *a: rows.append(tuple(X.shape[:2])) or encode(X, *a),
    )
    reg, X_test, _ = _fit(tiny_regressor, max_context_size=25, estimator_batch_size=1)
    # Each of the 3 members reads its own 25 of the 60 training rows.
    assert rows == [(1, 25)] * 3
    assert reg.predict(X_test).shape == (20,)


@pytest.mark.parametrize(
    ("n_estimators", "shuffled"), [(8, [0, 0, 0, 0, 1, 1, 1, 1]), (4, [0, 0, 1, 1]), (3, [0, 0, 1])]
)
def test_half_shuffle(tiny_regressor, n_estimators, shuffled):
    reg, _, _ = _fit(
        tiny_regressor, n_estimators=n_estimators, recipe=SeldonRecipe(half_shuffle=True)
    )
    built = reg.estimator._cache["recipe_execution"].recipe
    choices = [m for m in built.features.modules() if isinstance(m, sp.Choice)]
    assert {m._option_ids for m in choices if type(m).__name__ == "_HalfChoice"} == {
        tuple(shuffled)
    }


def test_numpy_input(tiny_regressor):
    X, y = _frame(50)
    X = X[list("abcd")].to_numpy()
    reg = SeldonRegressor(tiny_regressor, n_estimators=2, device="cpu")
    assert reg.fit(X[:40], y[:40].to_numpy()).predict(X[40:]).shape == (10,)


def test_column_target(tiny_regressor):
    flat, X_test, y = _fit(tiny_regressor)
    X, _ = _frame(80)
    column = SeldonRegressor(tiny_regressor, n_estimators=3, device="cpu")
    column.fit(X[:60], y.to_frame())
    np.testing.assert_array_equal(column.predict(X_test), flat.predict(X_test))
    with pytest.raises(ValueError, match="one target per row"):
        column.fit(X[:60], np.stack([y, y], axis=1))
    with pytest.raises(ValueError, match="one target per row"):
        column.fit(X[:60], y[:59])


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_non_finite_target(tiny_regressor, value):
    X, y = _frame(80)
    y = y[:60].copy()
    y.iloc[3] = value
    with pytest.raises(ValueError, match="NaN or infinite"):
        SeldonRegressor(tiny_regressor, device="cpu").fit(X[:60], y)


def test_clip_margin_is_at_least_zero(tiny_regressor):
    with pytest.raises(ValueError, match="clip_margin"):
        SeldonRegressor(tiny_regressor, clip_margin=-0.1)


def test_empty_test_table(tiny_regressor):
    reg, X_test, _ = _fit(tiny_regressor, clip_margin=0.1)
    assert reg.predict(X_test[:0]).shape == (0,)
    assert reg.predict_quantiles(X_test[:0]).shape == (0, 9)


def test_cache_budget_reencodes_the_same(tiny_regressor):
    kwargs = dict(estimator_batch_size=1)
    cached, X_test, _ = _fit(tiny_regressor, **kwargs)
    bounded, _, _ = _fit(tiny_regressor, max_cache_gb=0, **kwargs)
    np.testing.assert_allclose(
        bounded.predict_quantiles(X_test), cached.predict_quantiles(X_test), rtol=1e-5
    )
    # Past the budget only the context rows are kept, far smaller than their encoding.
    assert bounded.estimator._cache.size() < cached.estimator._cache.size()


def test_heads_are_checked(tiny_model, tiny_regressor):
    with pytest.raises(ValueError, match="the model is a classifier"):
        SeldonRegressor(tiny_model)
    with pytest.raises(ValueError, match="the model is a regressor"):
        SeldonClassifier(tiny_regressor)
    with pytest.raises(ValueError, match="classification model, not regression"):
        SeldonRegressor.from_pretrained(GOLDEN / "seldon-tiny-classifier")


def test_directory_of_both_heads(tiny_model, tiny_regressor, tmp_path):
    _save_checkpoints(tmp_path, {"tiny-classifier": tiny_model, "tiny-regressor": tiny_regressor})
    reg = SeldonRegressor.from_pretrained(tmp_path, device="cpu")
    clf = SeldonClassifier.from_pretrained(tmp_path, device="cpu")
    assert (reg.estimator.model.num_quantiles, clf.estimator.model.max_classes) == (9, 10)


def test_from_pretrained_model_names(tiny_regressor, tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr(
        Seldon,
        "from_pretrained",
        lambda name, *, task, **_: loaded.append((str(name), task)) or tiny_regressor,
    )
    for name in ["seldon", "seldon-mini", "seldon-mini-regressor", "org/repo"]:
        SeldonRegressor.from_pretrained(name, device="cpu")
    # The classifier's checkpoint goes as it is, for Seldon.from_pretrained to refuse its task.
    reg = SeldonRegressor.from_pretrained("seldon-mini-classifier", device="cpu")
    assert (reg.n_estimators, reg.clip_margin) == (8, None)
    # A local directory named like a model is read as such.
    (tmp_path / "seldon").mkdir()
    monkeypatch.chdir(tmp_path)
    SeldonRegressor.from_pretrained("seldon", device="cpu")
    assert loaded == [
        ("seldon-regressor", "regression"),
        ("seldon-mini-regressor", "regression"),
        ("seldon-mini-regressor", "regression"),
        ("org/repo", "regression"),
        ("seldon-mini-classifier", "regression"),
        ("seldon", "regression"),
    ]


def test_model_names_take_their_packaged_config(tiny_regressor, monkeypatch):
    monkeypatch.setattr(Seldon, "from_pretrained", lambda name, **_: tiny_regressor)
    # A model name or its regression checkpoint's: the same checkpoint, the same settings.
    for name in ["seldon", "seldon-mini", "seldon-regressor", "seldon-mini-regressor"]:
        reg = SeldonRegressor.from_pretrained(name, device="cpu")
        assert (reg.n_estimators, reg.max_context_size, reg.random_state, reg.amp) == (
            4,
            200_000,
            0,
            True,
        )
        assert reg.clip_margin == 0.1
        assert repr(reg.estimator.recipe.build()) == repr(SeldonRecipe(half_shuffle=True).build())
    # Arguments override the config.
    reg = SeldonRegressor.from_pretrained("seldon", clip_margin=None, n_estimators=2, device="cpu")
    assert (reg.clip_margin, reg.n_estimators) == (None, 2)
    # Any other checkpoint takes the constructor's defaults.
    assert SeldonRegressor.from_pretrained(TINY, device="cpu").clip_margin is None


@pytest.mark.parametrize("name", ["seldon-regressor", "seldon-mini-regressor"])
def test_packaged_config(name):
    # The packaged checkpoint is downloaded: the tiny one stands in for it.
    reg = SeldonRegressor.from_config(name, checkpoint=TINY, device="cpu")
    assert (reg.n_estimators, reg.clip_margin) == (4, 0.1)
    assert repr(reg.estimator.recipe.build()) == repr(SeldonRecipe(half_shuffle=True).build())
    X, y = _frame(80)
    assert reg.fit(X[:60], y[:60]).predict(X[60:]).shape == (20,)


def test_from_config_takes_its_own_task(tmp_path):
    with pytest.raises(ValueError, match="unknown regression config 'seldon-mini'"):
        SeldonRegressor.from_config("seldon-mini")
    with pytest.raises(ValueError, match="unknown classification config 'seldon-mini-regressor'"):
        SeldonClassifier.from_config("seldon-mini-regressor")
    path = tmp_path / "run.yaml"
    path.write_text(f"checkpoint: {TINY}\nclip_margin: 0.1\n")
    with pytest.raises(ValueError, match="SeldonRegressor.from_config"):
        SeldonClassifier.from_config(path)


def test_template_config_lists_the_regressor_argument(tiny_regressor, tmp_path, monkeypatch):
    text = (CONFIGS / "template.yaml").read_text()
    signature = inspect.signature
    assert set(signature(SeldonRegressor.__init__).parameters) - set(
        signature(SeldonClassifier.__init__).parameters
    ) == {"clip_margin"}
    assert set(signature(SeldonRegressor.from_pretrained).parameters) == set(
        signature(SeldonClassifier.from_pretrained).parameters
    )
    # Commented out, as the classifier rejects it; uncommented, the defaults.
    assert "\n# clip_margin: null\n" in text
    path = tmp_path / "template.yaml"
    path.write_text(text.replace("# clip_margin: null", "clip_margin: null"))
    reg = SeldonRegressor.from_config(path, checkpoint=TINY, device="cpu")
    default = SeldonRegressor.from_pretrained(TINY, device="cpu")
    assert {k: v for k, v in vars(reg).items() if k != "estimator"} == {
        k: v for k, v in vars(default).items() if k != "estimator"
    }
    assert repr(reg.estimator.recipe.build()) == repr(default.estimator.recipe.build())
    # With the template's model name, clip_margin left out is its packaged config's.
    monkeypatch.setattr(Seldon, "from_pretrained", lambda name, **_: tiny_regressor)
    assert SeldonRegressor.from_config("template", device="cpu").clip_margin == 0.1
    assert SeldonRegressor.from_config(path, device="cpu").clip_margin is None


def test_to_before_fit(tiny_regressor):
    X, y = _frame(80)
    reg = SeldonRegressor(tiny_regressor, n_estimators=2, device="cpu").to("cpu")
    assert reg.fit(X[:60], y[:60]).predict(X[60:]).shape == (20,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="moves a CUDA-fitted regressor")
def test_to_moves_a_fitted_regressor(tiny_regressor):
    X, y = _frame(80)
    reg = SeldonRegressor(copy.deepcopy(tiny_regressor), n_estimators=4, device="cuda", amp=False)
    on_gpu = reg.fit(X[:60], y[:60]).predict_quantiles(X[60:])
    on_cpu = reg.to("cpu").predict_quantiles(X[60:])
    np.testing.assert_allclose(on_cpu, on_gpu, rtol=1e-4)
    assert not any(t.is_cuda for t in reg.estimator._cache._tensors())
