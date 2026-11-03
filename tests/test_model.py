# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

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


def test_from_pretrained_hub_names(tiny_model, tmp_path, monkeypatch):
    from neuralk_oss.model import seldon

    tiny_model.save_pretrained(tmp_path / "hub" / "seldon-tiny-classifier")
    tiny_model.save_pretrained(tmp_path / "hub_root")
    calls = []

    def fake_download(repo_id, *, revision, token, allow_patterns):
        calls.append((repo_id, revision, allow_patterns))
        return str(tmp_path / ("hub_root" if repo_id == "org/tiny" else "hub"))

    monkeypatch.setattr(seldon, "snapshot_download", fake_download)
    monkeypatch.setitem(seldon.HUB_CHECKPOINTS, "seldon-tiny-classifier", "org/tiny")
    monkeypatch.chdir(tmp_path)
    # A checkpoint name downloads the root of its own repository.
    Seldon.from_pretrained("seldon-tiny-classifier")
    # A repository id, with the checkpoint's subfolder and a revision.
    Seldon.from_pretrained("org/repo", subfolder="seldon-tiny-classifier", revision="v1")
    assert calls == [
        ("org/tiny", None, ["config.json", "model.safetensors"]),
        (
            "org/repo",
            "v1",
            ["seldon-tiny-classifier/config.json", "seldon-tiny-classifier/model.safetensors"],
        ),
    ]
    # A local directory is read in place.
    Seldon.from_pretrained(tmp_path / "hub", subfolder="seldon-tiny-classifier")
    assert len(calls) == 2
    with pytest.raises(ValueError, match="seldon-classifier, seldon-mini-classifier"):
        Seldon.from_pretrained("seldon-missing-classifier")
    with pytest.raises(FileNotFoundError):
        Seldon.from_pretrained("org/repo", subfolder="seldon-missing-classifier")


def test_hub_checkpoints():
    from neuralk_oss.model.seldon import HUB_CHECKPOINTS

    assert HUB_CHECKPOINTS == {
        "seldon-classifier": "Neuralk-AI/seldon",
        "seldon-mini-classifier": "Neuralk-AI/seldon-mini",
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
