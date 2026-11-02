# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

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


@pytest.mark.parametrize(
    "params",
    [
        {"max_features": 2},
        {"transforms": ("identity", "rank_gaussian")},
        {"shuffle_categories": False},
        {"flip_sign": False},
        {"feature_shuffle": "random"},
        {"feature_shuffle": None},
        {"class_shift": False},
        {"average_logits": False},
        {"softmax_temperature": 2.0},
    ],
    ids=lambda params: "-".join(f"{k}={v}" for k, v in params.items()),
)
def test_recipe_parameters(tiny_model, params):
    X, y = _frame(80, 3)
    # 8 members: shuffle_categories leaves the first len(transforms), 4, alone.
    clf = SeldonClassifier(tiny_model, n_estimators=8, device="cpu", recipe=SeldonRecipe(**params))
    default = SeldonClassifier(tiny_model, n_estimators=8, device="cpu")
    p = clf.fit(X[:60], y[:60]).predict_proba(X[60:])
    np.testing.assert_allclose(p.sum(axis=1), 1, rtol=1e-5)
    assert not np.allclose(p, default.fit(X[:60], y[:60]).predict_proba(X[60:]))


def test_shuffle_categories_blocks(tiny_model):
    X, y = _frame(80, 3)
    clf = SeldonClassifier(
        tiny_model,
        n_estimators=8,
        device="cpu",
        recipe=SeldonRecipe(shuffle_categories=True, transforms=("identity", "power")),
    )
    clf.fit(X[:60], y[:60])
    recipe = clf.estimators[0]._cache["recipe_execution"].recipe
    (choice,) = [m for m in recipe.features.modules() if type(m).__name__ == "_BlockChoice"]
    # Blocks of len(transforms) members: sorted codes, then shuffled ones.
    assert choice._option_ids == (0, 0, 1, 1, 0, 0, 1, 1)


def test_recipe_rejects_unknown_parameters():
    with pytest.raises(TypeError, match="max_feature"):
        SeldonRecipe(max_feature=2)
    with pytest.raises(TypeError, match="build"):
        SeldonRecipe(build=None)
    with pytest.raises(ValueError, match="transforms"):
        SeldonRecipe(transforms=("identity", "quantile"))
    with pytest.raises(ValueError, match="transforms"):
        SeldonRecipe(transforms=())


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


TINY = Path(__file__).parent / "golden" / "seldon-tiny-classifier"


def test_packaged_config_is_the_default_recipe():
    # The packaged checkpoint is downloaded: the tiny one stands in for it.
    clf = SeldonClassifier.from_config("seldon-mini", checkpoint=TINY, device="cpu")
    assert (clf.n_estimators, clf.max_context_size, clf.random_state) == (8, 200_000, 0)
    assert repr(clf.estimators[0].recipe.build()) == repr(SeldonRecipe().build())
    X, y = _frame(80, 3)
    assert clf.fit(X[:60], y[:60]).predict_proba(X[60:]).shape == (20, 3)


def test_config_file(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        f"checkpoint: {TINY}\nn_estimators: 3\nrecipe:\n  transforms: [identity, robust]\n"
    )
    clf = SeldonClassifier.from_config(path, device="cpu")
    assert clf.n_estimators == 3
    assert clf.estimators[0].recipe.transforms == ["identity", "robust"]
    path.write_text(f"checkpoint: {TINY}\nrecipe:\n  max_feature: 3\n")
    with pytest.raises(TypeError, match="max_feature"):
        SeldonClassifier.from_config(path)
    with pytest.raises(ValueError, match="seldon-mini"):
        SeldonClassifier.from_config("seldon-maxi")


def test_template_config_lists_every_argument_at_its_default():
    path = Path(__file__).parents[1] / "configs" / "template.yaml"
    config = yaml.safe_load(path.read_text())
    arguments = {
        *inspect.signature(SeldonClassifier.__init__).parameters,
        *inspect.signature(SeldonClassifier.from_pretrained).parameters,
    } - {"self", "model", "name_or_path", "kwargs", "recipe"}
    assert set(config) - {"recipe", "checkpoint"} == arguments
    recipe = SeldonRecipe(**config["recipe"])
    defaults = {n for n in vars(SeldonRecipe) if not n.startswith("_")}
    assert set(config["recipe"]) == {n for n in defaults if not callable(getattr(SeldonRecipe, n))}
    assert repr(recipe.build()) == repr(SeldonRecipe().build())
    clf = SeldonClassifier.from_config(path, checkpoint=TINY, device="cpu")
    default = SeldonClassifier.from_pretrained(TINY, device="cpu")
    assert {k: v for k, v in vars(clf).items() if k != "estimators"} == {
        k: v for k, v in vars(default).items() if k != "estimators"
    }


def test_recipes_per_member_group(tiny_model):
    X, y = _frame(80, 3)
    kwargs = {"n_estimators": 4, "device": "cpu"}
    one = SeldonClassifier(tiny_model, **kwargs).fit(X[:60], y[:60]).predict_proba(X[60:])
    listed = SeldonClassifier(tiny_model, recipe=[SeldonRecipe()], **kwargs)
    np.testing.assert_array_equal(listed.fit(X[:60], y[:60]).predict_proba(X[60:]), one)
    halves = [
        SeldonRecipe(members=[0, 1], shuffle_categories=False),
        SeldonRecipe(members=[2, 3], shuffle_categories="all"),
    ]
    clf = SeldonClassifier(tiny_model, recipe=halves, **kwargs)
    p = clf.fit(X[:60], y[:60]).predict_proba(X[60:])
    np.testing.assert_allclose(p.sum(axis=1), 1, rtol=1e-5)
    assert not np.allclose(p, one)
    assert clf._group_sizes == [2, 2]


def test_shuffle_categories_all(tiny_model):
    X, y = _frame(80, 3)
    recipe = SeldonRecipe(shuffle_categories="all")
    clf = SeldonClassifier(tiny_model, n_estimators=2, device="cpu", recipe=recipe)
    clf.fit(X[:60], y[:60])
    built = clf.estimators[0]._cache["recipe_execution"].recipe
    names = [type(m).__name__ for m in built.features.modules()]
    assert "ShuffleCategories" in names and "_BlockChoice" not in names


@pytest.mark.parametrize(
    ("members", "match"),
    [
        (["all", [0, 1]], "only one"),
        ([[0, 1], [1, 2]], r"missing \[3\], repeated \[1\]"),
        ([[0, 1], [2, 5]], r"missing \[3\].*out of range \[5\]"),
    ],
)
def test_recipe_members_must_cover_the_ensemble(tiny_model, members, match):
    recipes = [SeldonRecipe(members=m) for m in members]
    with pytest.raises(ValueError, match=match):
        SeldonClassifier(tiny_model, n_estimators=4, device="cpu", recipe=recipes)


def test_config_with_several_recipes(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        f"""checkpoint: {TINY}
n_estimators: 4
recipe:
  - members: [0, 1]
    shuffle_categories: false
  - members: [2, 3]
    shuffle_categories: all
"""
    )
    clf = SeldonClassifier.from_config(path, device="cpu")
    assert [e.recipe.shuffle_categories for e in clf.estimators] == [False, "all"]


def test_recipe_groups_on_a_subsampled_context(tiny_model):
    X, y = _frame(80, 3)
    recipes = [SeldonRecipe(members=[0]), SeldonRecipe(members=[1, 2])]
    clf = SeldonClassifier(
        tiny_model, n_estimators=3, device="cpu", max_context_size=25, recipe=recipes
    )
    assert clf.fit(X[:60], y[:60]).predict_proba(X[60:]).shape == (20, 3)
