# neuralk_oss

Neuralk-AI open-source models.

## Seldon

Seldon is a tabular in-context-learning foundation model: `fit` keeps the
training table as context, `predict_proba` classifies new rows against it in
one forward pass, with no gradient step.

```python
from neuralk_oss import SeldonClassifier

clf = SeldonClassifier.from_pretrained("seldon-small-classifier")  # directory: config.json + model.safetensors
clf.fit(X_train, y_train)            # pandas DataFrame (or array) and labels
proba = clf.predict_proba(X_test)    # (n_rows, n_classes), columns in clf.classes_ order
```

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
logits = model.predict_query(X_query, context, categorical_mask)       # (B, T_query, 10)
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
