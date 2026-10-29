"""Write the tiny checkpoints the golden tests load: ``uv run python -m tests.golden.make_checkpoints``.

Random weights, seeded, so every layer is exercised; each checkpoint covers
other branches of the config. Regenerating them moves every golden output.
"""

from pathlib import Path

import torch

from neuralk_oss import Seldon
from tests.conftest import tiny_config

GOLDEN = Path(__file__).parent

CHECKPOINTS = {
    # The small checkpoint's architecture, scaled down.
    "seldon-tiny-classifier": tiny_config(),
    # LayerNorm, biases, no ICL projection, grouped query heads, no fp32 islands.
    "seldon-tiny-classifier-variant": tiny_config(
        use_rmsnorm=False,
        col={"num_blocks": 2, "nhead": 2, "bias_free_ln": False},
        row={"num_blocks": 2, "nhead": 2, "num_cls": 2, "bias_free_ln": False},
        icl={"num_blocks": 2, "nhead": 4, "bias_free_ln": False},
        icl_channels=None,
        num_icl_key_value_heads_for_query=2,
        bias_free_linear=False,
        fp32_cells=False,
        fp32_head=False,
    ),
    "seldon-tiny-regressor": tiny_config(head={"task": "regression", "num_quantiles": 9}),
}


def main() -> None:
    for seed, (name, config) in enumerate(CHECKPOINTS.items()):
        torch.manual_seed(seed)
        model = Seldon(config)
        with torch.no_grad():
            for param in model.parameters():
                param.normal_(0, 0.3)
        model.save_pretrained(GOLDEN / name)
        print(f"wrote {GOLDEN / name}")


if __name__ == "__main__":
    main()
