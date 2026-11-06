# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from neuralk_oss import Seldon, SeldonConfig


def tiny_config(**overrides) -> SeldonConfig:
    config = {
        "embed_dim": 16,
        "use_rmsnorm": True,
        "head": {"task": "classification", "max_classes": 10},
        "col": {"num_blocks": 2, "nhead": 2, "bias_free_ln": True},
        "row": {"num_blocks": 2, "nhead": 2, "num_cls": 2, "bias_free_ln": True},
        "icl": {"num_blocks": 2, "nhead": 4, "bias_free_ln": True},
        "num_inducing_points": 8,
        "num_frequencies": 4,
        "icl_channels": 24,
        "num_icl_key_value_heads_for_query": None,
        "bias_free_linear": True,
        "fp32_cells": True,
        "fp32_head": True,
    }
    return SeldonConfig.model_validate({**config, **overrides})


def _random_model(config: SeldonConfig) -> Seldon:
    torch.manual_seed(0)
    model = Seldon(config).eval()
    # Random weights everywhere, so no zero-initialized layer hides a path.
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 0.3)
    return model


@pytest.fixture
def tiny_model() -> Seldon:
    return _random_model(tiny_config())


@pytest.fixture
def tiny_regressor() -> Seldon:
    return _random_model(tiny_config(head={"task": "regression", "num_quantiles": 9}))
