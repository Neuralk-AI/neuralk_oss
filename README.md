# neuralk_oss

Neuralk-AI open-source models.

## Seldon

Seldon is a tabular in-context-learning foundation model: `fit` keeps the
training table as context, `predict_proba` classifies new rows against it in
one forward pass, with no gradient step.

```python
from neuralk_oss import SeldonClassifier

clf = SeldonClassifier.from_pretrained("seldon-small-classifier")
clf.fit(X_train, y_train)  # pandas DataFrame (or array) and labels
proba = clf.predict_proba(X_test)  # (n_rows, n_classes), columns in clf.classes_ order
```

### Checkpoints

| name | task | parameters |
| --- | --- | --- |
| `seldon-small-classifier` | classification, up to 10 classes per model call | 27.4M |

A checkpoint is a directory holding `config.json` and `model.safetensors`.
`from_pretrained` takes:

- a checkpoint name, such as `"seldon-small-classifier"`: downloaded from the
  Hugging Face repository [`Neuralk-AI/seldon`](https://huggingface.co/Neuralk-AI/seldon),
  where each checkpoint is a directory, on the first call, then read from the
  Hugging Face cache (`~/.cache/huggingface`, or `HF_HOME`);
- a local directory, read in place: `from_pretrained("path/to/seldon-small-classifier")`;
- any Hugging Face repository, with the checkpoint's directory and a revision:
  `from_pretrained("org/repo", subfolder="seldon-small-classifier", revision="v1")`.

The repository is private for now: log in once with an account that can read
it (`hf auth login`), or pass `token=...` to `from_pretrained`.

Column types are inferred (categorical columns are embedded as such, NaNs are
kept for the model's own imputation), each of the `n_estimators` ensemble
members (8 by default) sees its own preprocessing, and their logits are
averaged. Inference runs in float16 autocast on CUDA by default
(`amp=False` for float32). Tables with more classes than the model's head
(10) go through error-correcting output codes.

### Preprocessing

The preprocessing and ensembling come from structured-data-models (`sdm`,
Apache-2.0), pinned in `pyproject.toml`. `SeldonRecipe` builds its
recipe one overridable method per step:

```python
import sdm.processing as sp
from neuralk_oss import SeldonClassifier, SeldonRecipe


class MyRecipe(SeldonRecipe):
    max_features = 200

    def numerical_transforms(self):
        return [sp.Identity(), sp.RankGaussian(max_knots=self.rank_gaussian_knots)]


clf = SeldonClassifier.from_pretrained("seldon-small-classifier", recipe=MyRecipe())
```

### The model alone

```python
from neuralk_oss import Seldon

model = Seldon.from_pretrained("seldon-small-classifier")
context = model.encode_context(X_context, y_context, categorical_mask)  # (B, T, H), (B, T), (B, H)
logits = model.predict_query(X_query, context, categorical_mask)  # (B, T_query, 10)
```

## Layout

```
src/neuralk_oss/
├── model/       # Seldon's architecture, config and checkpoint format
└── inference/   # sdm estimator, preprocessing recipe, scikit-learn style classifier
tests/
```

## Development

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

`tests/test_golden.py` compares the outputs of tiny random-weight checkpoints
(`tests/golden/`) to saved references. After an intended change of outputs,
regenerate them with `UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py` and
commit the new `tests/golden/expected/`.
