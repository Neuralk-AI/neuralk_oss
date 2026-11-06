# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from safetensors.torch import save_file

from neuralk_oss import Seldon
from neuralk_oss.model import seldon
from tests.conftest import tiny_config


def _table(B=2, T=40, H=5, train_size=30, classes=3):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(B, T, H, generator=g)
    X[torch.rand(B, T, H, generator=g) < 0.1] = float("nan")
    y = torch.randint(0, classes, (B, train_size), generator=g)
    mask = torch.rand(B, H, generator=g) < 0.4
    return X, y, train_size, mask


@torch.no_grad()
def test_forward_shape_and_finite(tiny_model):
    X, y, ts, mask = _table()
    out = tiny_model(X, y, ts, categorical_mask=mask)
    assert out.shape == (2, 10, 10)
    assert out.isfinite().all()


@torch.no_grad()
def test_query_rows_are_independent(tiny_model):
    X, y, ts, mask = _table()
    context = tiny_model.encode_context(X[:, :ts], y, mask)
    whole = tiny_model.predict_query(X[:, ts:], context, mask)
    rows = torch.cat(
        [tiny_model.predict_query(X[:, t : t + 3], context, mask) for t in range(ts, 40, 3)], 1
    )
    torch.testing.assert_close(rows, whole, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_query_does_not_change_context(tiny_model):
    X, y, ts, mask = _table()
    out = tiny_model(X, y, ts, categorical_mask=mask)
    X2 = X.clone()
    X2[:, ts + 1 :] = torch.randn_like(X2[:, ts + 1 :])
    out2 = tiny_model(X2, y, ts, categorical_mask=mask)
    torch.testing.assert_close(out2[:, 0], out[:, 0], rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_reduced_kv_heads_run():
    model = Seldon(tiny_config(num_icl_key_value_heads_for_query=2)).eval()
    X, y, ts, mask = _table()
    assert model(X, y, ts, categorical_mask=mask).isfinite().all()


def test_too_many_classes(tiny_model):
    X, y, ts, mask = _table(classes=3)
    y[0, 0] = 10
    with pytest.raises(ValueError, match="max_classes"):
        tiny_model.encode_context(X[:, :ts], y, mask)


@torch.no_grad()
def test_save_load_roundtrip(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path)
    loaded = Seldon.from_pretrained(tmp_path)
    assert not loaded.training
    X, y, ts, mask = _table()
    torch.testing.assert_close(
        loaded(X, y, ts, categorical_mask=mask), tiny_model(X, y, ts, categorical_mask=mask)
    )


def _tiny_pair():
    """A tiny classifier and a regressor of the same architecture, random weights."""
    models = {}
    for seed, head in enumerate([None, {"task": "regression", "num_quantiles": 9}]):
        torch.manual_seed(seed)
        model = Seldon(tiny_config() if head is None else tiny_config(head=head)).eval()
        with torch.no_grad():
            for param in model.parameters():
                param.normal_(0, 0.3)
        models["seldon-tiny-regressor" if head else "seldon-tiny-classifier"] = model
    return models


def _save_checkpoints(path, models):
    """``models`` as one directory of checkpoints: their shared architecture and heads in config.json, ``<name>.safetensors`` each."""
    path.mkdir(parents=True, exist_ok=True)
    config = {"model_version": Seldon.VERSION, **next(iter(models.values())).config.model_dump()}
    config.pop("head")
    config["checkpoints"] = {name: model.config.head.model_dump() for name, model in models.items()}
    (path / "config.json").write_text(json.dumps(config))
    for name, model in models.items():
        save_file(model.state_dict(), path / f"{name}.safetensors")


@torch.no_grad()
def test_checkpoints_directory(tmp_path):
    models = _tiny_pair()
    _save_checkpoints(tmp_path, models)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "config.json",
        "seldon-tiny-classifier.safetensors",
        "seldon-tiny-regressor.safetensors",
    ]
    X, y, ts, mask = _table()
    for task, name in [
        ("classification", "seldon-tiny-classifier"),
        ("regression", "seldon-tiny-regressor"),
    ]:
        loaded = Seldon.from_pretrained(tmp_path, task=task)
        assert loaded.config == models[name].config
        y_ = y.float() if task == "regression" else y
        torch.testing.assert_close(
            loaded(X, y_, ts, categorical_mask=mask), models[name](X, y_, ts, categorical_mask=mask)
        )
    with pytest.raises(ValueError, match="pass the task"):
        Seldon.from_pretrained(tmp_path)
    # A single checkpoint's task is checked.
    models["seldon-tiny-regressor"].save_pretrained(tmp_path / "single")
    Seldon.from_pretrained(tmp_path / "single")
    with pytest.raises(ValueError, match="regression model, not classification"):
        Seldon.from_pretrained(tmp_path / "single", task="classification")


def test_from_pretrained_hub_names(tmp_path, monkeypatch):
    _save_checkpoints(tmp_path / "hub_root", _tiny_pair())
    _tiny_pair()["seldon-tiny-classifier"].save_pretrained(
        tmp_path / "hub" / "seldon-tiny-classifier"
    )
    calls = []

    def fake_download(repo_id, *, revision, token, allow_patterns):
        calls.append((repo_id, revision, allow_patterns))
        return str(tmp_path / ("hub_root" if repo_id == "org/tiny" else "hub"))

    monkeypatch.setattr(seldon, "snapshot_download", fake_download)
    for name in ["seldon-tiny-classifier", "seldon-tiny-regressor"]:
        monkeypatch.setitem(seldon.HUB_CHECKPOINTS, name, "org/tiny")
    monkeypatch.chdir(tmp_path)
    # A checkpoint name reads the root config.json, then its own weights only.
    assert Seldon.from_pretrained("seldon-tiny-regressor").num_quantiles == 9
    # A repository id: its checkpoint by task, or a single one in a subfolder, at a revision.
    assert Seldon.from_pretrained("org/tiny", task="classification").max_classes == 10
    Seldon.from_pretrained("org/repo", subfolder="seldon-tiny-classifier", revision="v1")
    assert calls == [
        ("org/tiny", None, ["config.json"]),
        ("org/tiny", None, ["seldon-tiny-regressor.safetensors"]),
        ("org/tiny", None, ["config.json"]),
        ("org/tiny", None, ["seldon-tiny-classifier.safetensors"]),
        ("org/repo", "v1", ["seldon-tiny-classifier/config.json"]),
        ("org/repo", "v1", ["seldon-tiny-classifier/model.safetensors"]),
    ]
    # A local directory is read in place.
    Seldon.from_pretrained(tmp_path / "hub", subfolder="seldon-tiny-classifier")
    assert len(calls) == 6
    with pytest.raises(ValueError, match="regression model, not classification"):
        Seldon.from_pretrained("seldon-tiny-regressor", task="classification")
    with pytest.raises(ValueError, match="seldon-classifier, seldon-regressor"):
        Seldon.from_pretrained("seldon-missing-classifier")
    with pytest.raises(FileNotFoundError):
        Seldon.from_pretrained("org/repo", subfolder="seldon-missing-classifier")


def test_hub_checkpoints():
    assert seldon.HUB_CHECKPOINTS == {
        "seldon-classifier": "Neuralk-AI/seldon",
        "seldon-regressor": "Neuralk-AI/seldon",
        "seldon-mini-classifier": "Neuralk-AI/seldon-mini",
        "seldon-mini-regressor": "Neuralk-AI/seldon-mini",
    }


@torch.no_grad()
def test_context_chunks_match_one_pass(tiny_model, monkeypatch):
    X, y, ts, mask = _table()
    whole = tiny_model.encode_context(X[:, :ts], y, mask)
    # Slices of 7 rows (the last one short) and of 1 column.
    monkeypatch.setattr(seldon, "split_size", lambda n, item_bytes, device: max(1, n // 4))
    chunked = tiny_model.encode_context(X[:, :ts], y, mask)
    torch.testing.assert_close(chunked, whole, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_split_fits_free_memory(monkeypatch):
    # sdm's limit allows 100 items of a byte, but only 30 bytes are free: 4 balanced chunks.
    cuda = torch.device("cuda")
    monkeypatch.setattr(seldon, "split_size", lambda n, item_bytes, device: 100)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (20, 0))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 15)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 5)
    assert seldon._split(100, 1, cuda) == 25
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1000, 0))
    assert seldon._split(100, 1, cuda) == 100
