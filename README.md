![Neuralk, the predictive AI company](assets/banner.png)

# Neuralk OSS

Neuralk-AI open-source models.

## Seldon

Seldon is a tabular in-context-learning foundation model: `fit` keeps the
training table as context, `predict_proba` classifies new rows against it (or
a regressor's `predict` estimates their target, see [Regression](#regression))
in one forward pass, with no gradient step.

```python
from neuralk_oss import SeldonClassifier

clf = SeldonClassifier.from_pretrained("seldon-mini")  # or "seldon"
clf.fit(X_train, y_train)  # pandas DataFrame (or array) and labels
proba = clf.predict_proba(X_test)  # (n_rows, n_classes), columns in clf.classes_ order
```

A model's name loads it with the settings it is meant to run with, from its
packaged config (see [Configs](#configs)): `seldon` with 8 ensemble members,
`seldon-mini` with 4, half of them reading categories in a shuffled order.
Keyword arguments override them, e.g. `from_pretrained("seldon", device="cpu")`.

### Checkpoints

| model | checkpoint | task | parameters |
| --- | --- | --- | --- |
| `seldon` | `seldon-classifier` | classification, up to 10 classes per model call | 213.5M |
| `seldon` | `seldon-regressor` | regression, 999 quantiles | 215.5M |
| `seldon-mini` | `seldon-mini-classifier` | classification, up to 10 classes per model call | 27.4M |
| `seldon-mini` | `seldon-mini-regressor` | regression, 999 quantiles | 28.4M |

`SeldonClassifier.from_pretrained` takes the model name and loads its
classification checkpoint, `SeldonRegressor.from_pretrained` its regression
checkpoint; `Seldon.from_pretrained` takes the checkpoint name.

A checkpoint directory holds one model, `config.json` and `model.safetensors`
(as `Seldon.save_pretrained` writes it), or several sharing an architecture:
`config.json` then names them under `checkpoints`, each with its head, and the
weights of checkpoint `name` are `name.safetensors`. `from_pretrained` takes:

- a checkpoint name of the table above (or, for `SeldonClassifier` and
  `SeldonRegressor`, a model name): its model's Hugging Face repository,
  [`Neuralk-AI/seldon`](https://huggingface.co/Neuralk-AI/seldon) or
  [`Neuralk-AI/seldon-mini`](https://huggingface.co/Neuralk-AI/seldon-mini),
  holds both checkpoints at its root; the first call downloads `config.json`
  and that checkpoint's weights, later ones read the Hugging Face cache
  (`~/.cache/huggingface`, or `HF_HOME`);
- a local directory, read in place: `from_pretrained("path/to/seldon-mini-classifier")`;
- any Hugging Face repository, its root or a checkpoint's directory, and a
  revision: `from_pretrained("Neuralk-AI/seldon-mini", task="regression")`,
  `from_pretrained("org/repo", subfolder="seldon-mini-classifier", revision="v1")`.
  `task` picks the checkpoint in a directory of several (`SeldonClassifier`
  passes `"classification"`, `SeldonRegressor` `"regression"`).

The weights come under their own license, which each repository's page asks
to accept: accept it once with a Hugging Face account (access is granted at
once), then log in with it (`hf auth login`) or pass `token=...` to
`from_pretrained`.

Column types are inferred (categorical columns are embedded as such, NaNs are
kept for the model's own imputation), each of the `n_estimators` ensemble
members (8 by default) sees its own preprocessing, and their logits are
averaged. Inference runs in float16 autocast on CUDA by default
(`amp=False` for float32). Tables with more classes than the model's head
(10) go through error-correcting output codes.

### Regression

```python
from neuralk_oss import SeldonRegressor

reg = SeldonRegressor.from_pretrained("seldon-mini")  # or "seldon"
reg.fit(X_train, y_train)  # pandas DataFrame (or array) and a numerical target
y_pred = reg.predict(X_test)  # (n_rows,): the mean of each row's quantiles
quantiles = reg.predict_quantiles(X_test)  # (n_rows, 999), ascending
```

`SeldonRegressor` takes the classifier's arguments, plus `clip_margin`. The
target is standardized for the model; each member's quantiles, back in the
target's units, are sorted, then averaged over the members by a trimmed mean.
With `clip_margin`, they are clamped to the training target's range, widened
by `clip_margin` times that range on each side; the constructor's default,
`None`, leaves them unclamped. A model's name loads its regression checkpoint
with its packaged config, `seldon-regressor` or `seldon-mini-regressor`: half
the members read categories in a shuffled order, and `clip_margin=0.1`.

### Forecasting

```python
from neuralk_oss import SeldonForecaster

fc = SeldonForecaster.from_pretrained("seldon-mini")  # or "seldon"
# context_df: item_id (optional), timestamp, target, covariates
out = fc.predict_df(context_df, prediction_length=24)  # or future_df=... with known covariates
# indexed by (item_id, timestamp): target (the median), 0.1, ..., 0.9
```

`SeldonForecaster` runs the [TabPFN-TS](https://github.com/PriorLabs/tabpfn-time-series)
pipeline (its features ported in `neuralk_oss.forecasting.features`) with
Seldon's regressor in TabPFN's place. Each series becomes a table, one row
per timestamp and its value the label, whose features are known before the
forecast: lags {1, season} and a trailing mean of the known covariates, the
row's running index, the year and the calendar fields' sin and cos, and sin
and cos of the running index over the 12 strongest periods of the series'
spectrum. The regressor is fit on each series' history and predicts the
quantiles of its horizon; the target's past values are never features.

Per series, context rows without a target are dropped (a series with at most
one value has the others set to 0), and the latest `max_context_length`
(32768) are kept. Nothing else is subsampled: a series whose context passes
the regressor's `max_context_size`, or whose table passes the recipe's
`max_features` (`cut_features=False`, required), raises instead of being cut. A covariate with values on the
horizon is a known one; a context column without is past-only, read as
missing on the horizon. Quantile levels between the model's 999 are
interpolated; `point_forecast="mean"` makes `target` the mean of the
quantiles. The packaged configs `seldon-forecaster` and
`seldon-mini-forecaster` take the regressors' settings without their clamp to
the context's range (`clip_margin`), which would keep a trend inside it.

`forecast` takes the covariates `x_context`, the targets `y_context` (one or
more, each forecast on its own) and the horizon `x_query`, on timestamps, and
returns per target `t` the columns `t` and `t__q<100 x level>`:

```python
out = fc.forecast(x_context, y_context, x_query, quantiles=[0.1, 0.5, 0.9])
```

### Preprocessing

The preprocessing and ensembling come from structured-data-models (`sdm`,
Apache-2.0), pinned in `pyproject.toml`. `SeldonRecipe` builds its recipe, and
takes its parameters as keyword arguments:

```python
from neuralk_oss import SeldonClassifier, SeldonRecipe

recipe = SeldonRecipe(transforms=("identity", "rank_gaussian"), softmax_temperature=0.9)
clf = SeldonClassifier.from_pretrained("seldon-mini", recipe=recipe)
```

| parameter | default | |
| --- | --- | --- |
| `transforms` | `("identity", "power", "robust", "rank_gaussian")` | numerical transforms the members take in turn, one each |
| `half_shuffle` | `False` | the second half of the members read categorical codes in a random order per column, the first half sorted |
| `flip_sign` | `True` | each member negates a random half of its numerical columns |
| `feature_shuffle` | `"latin"` | column permutation per member: `"latin"`, `"random"` or `None` |
| `class_shift` | `True` | each member shifts the class codes by its own offset |
| `average_logits` | `True` | average the members' logits, else their probabilities |
| `softmax_temperature` | `1.0` | divisor of the logits before the softmax |
| `max_features` | `500` | columns the model reads, cut after the shuffle |
| `cut_features` | `True` | cut the columns past `max_features`; `False`: `fit` raises `TooManyFeatures` instead (the forecaster's configs) |

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

### Configs

For full control over a run, beyond `from_pretrained`'s defaults, a YAML
config sets the checkpoint, the classifier and its recipe in one file:
`checkpoint`, any keyword argument of `from_pretrained`, and `recipe`,
`SeldonRecipe`'s keyword arguments. Each released model's settings are a packaged config of its name,
which `from_pretrained` loads: `seldon` and `seldon-mini` for the classifiers,
`seldon-regressor` and `seldon-mini-regressor` for the regressors
(`SeldonRegressor.from_config("seldon-regressor")`), `seldon-forecaster` and
`seldon-mini-forecaster` for the forecasters.

```yaml
# seldon.yaml
checkpoint: seldon
n_estimators: 8
max_context_size: 200000
random_state: 0
amp: true
recipe:
  half_shuffle: true
```

```python
clf = SeldonClassifier.from_config("seldon")  # the same as from_pretrained("seldon")
clf = SeldonClassifier.from_config(
    "my_run.yaml", device="cpu"
)  # keyword arguments override the file
```

[`template.yaml`](src/neuralk_oss/configs/template.yaml), packaged next to
them, lists every argument a config can set, at its default, with what
it does: copy it and keep the keys you change. Keys a config leaves out take
the model's own settings for `seldon` and `seldon-mini`, the constructor's
defaults otherwise.

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
