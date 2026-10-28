# neuralk_oss

Neuralk-AI open-source models.

## Layout

```
src/neuralk_oss/
├── model/       # model definitions and checkpoint loading
└── inference/   # inference utilities
tests/
```

## Installation

```bash
pip install -e .
```

For development (with [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
uv run pytest
uv run ruff check .
```

## Usage

```python
import torch
from neuralk_oss import load_model, predict

model = load_model("weights.pt", MyModel(), device="cpu")
outputs = predict(model, torch.randn(8, 16))
```
