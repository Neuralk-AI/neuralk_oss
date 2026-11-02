![Neuralk, the predictive AI company](assets/banner.png)

# Neuralk OSS

Neuralk-AI open-source models.

## Seldon

Seldon is a tabular in-context-learning foundation model: `fit` keeps the
training table as context, `predict_proba` classifies new rows against it in
one forward pass, with no gradient step.

```python
from neuralk_oss import SeldonClassifier

clf = SeldonClassifier.from_pretrained("seldon-mini")  # or "seldon"
clf.fit(X_train, y_train)  # pandas DataFrame (or array) and labels
proba = clf.predict_proba(X_test)  # (n_rows, n_classes), columns in clf.classes_ order
```

### Checkpoints

| model | checkpoint | task | parameters |
| --- | --- | --- | --- |
| `seldon` | `seldon-classifier` | classification, up to 10 classes per model call | 213.5M |
| `seldon-mini` | `seldon-mini-classifier` | classification, up to 10 classes per model call | 27.4M |

`SeldonClassifier.from_pretrained` takes the model name and loads its
classification checkpoint; `Seldon.from_pretrained` takes the checkpoint name.

A checkpoint is a directory holding `config.json` and `model.safetensors`.
`from_pretrained` takes:

- a checkpoint name of the table above (or, for `SeldonClassifier`, a model
  name): downloaded from its Hugging Face repository,
  [`Neuralk-AI/seldon`](https://huggingface.co/Neuralk-AI/seldon) or
  [`Neuralk-AI/seldon-mini`](https://huggingface.co/Neuralk-AI/seldon-mini),
  where the checkpoint is a directory of that name, on the first call, then
  read from the Hugging Face cache (`~/.cache/huggingface`, or `HF_HOME`);
- a local directory, read in place: `from_pretrained("path/to/seldon-mini-classifier")`;
- any Hugging Face repository, with the checkpoint's directory and a revision:
  `from_pretrained("org/repo", subfolder="seldon-mini-classifier", revision="v1")`.

The repositories are private for now: log in once with an account that can read
them (`hf auth login`), or pass `token=...` to `from_pretrained`.

Column types are inferred (categorical columns are embedded as such, NaNs are
kept for the model's own imputation), each of the `n_estimators` ensemble
members (8 by default) sees its own preprocessing, and their logits are
averaged. Inference runs in float16 autocast on CUDA by default
(`amp=False` for float32). Tables with more classes than the model's head
(10) go through error-correcting output codes.

### Preprocessing

The preprocessing and ensembling come from structured-data-models (`sdm`,
Apache-2.0), pinned in `pyproject.toml`. `SeldonRecipe` builds its recipe, and
takes its parameters as keyword arguments:

```python
from neuralk_oss import SeldonClassifier, SeldonRecipe

recipe = SeldonRecipe(transforms=("identity", "rank_gaussian"), softmax_temperature=0.9)
clf = SeldonClassifier.from_pretrained("seldon-small-classifier", recipe=recipe)
```

| parameter | default | |
| --- | --- | --- |
| `transforms` | `("identity", "power", "robust", "rank_gaussian")` | numerical transforms the members take in turn, one each |
| `flip_sign` | `True` | each member negates a random half of its numerical columns |
| `feature_shuffle` | `"latin"` | column permutation per member: `"latin"`, `"random"` or `None` |
| `class_shift` | `True` | each member shifts the class codes by its own offset |
| `average_logits` | `True` | average the members' logits, else their probabilities |
| `softmax_temperature` | `1.0` | divisor of the logits before the softmax |
| `max_features` | `500` | columns the model reads, cut after the shuffle |

The rest (clipping bounds, rank-Gaussian knots, category counts) is in
`SeldonRecipe`'s docstring. Beyond parameters, each step is an overridable method:

```python
import sdm.processing as sp
from neuralk_oss import SeldonClassifier, SeldonRecipe


class MyRecipe(SeldonRecipe):
    max_features = 200

    def numerical_transforms(self):
        return [sp.Identity(), sp.RankGaussian(max_knots=self.rank_gaussian_knots)]


clf = SeldonClassifier.from_pretrained("seldon-mini", recipe=MyRecipe())
```

### The model alone

```python
from neuralk_oss import Seldon

model = Seldon.from_pretrained("seldon-mini-classifier")
context = model.encode_context(X_context, y_context, categorical_mask)  # (B, T, H), (B, T), (B, H)
logits = model.predict_query(X_query, context, categorical_mask)  # (B, T_query, 10)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md): layout, development and versions.

## License

[Apache 2.0](LICENSE).
