# Contributing

Follow the [code of conduct](CODE_OF_CONDUCT.md); report vulnerabilities as
[SECURITY.md](SECURITY.md) says, not in public issues.

## Layout

```
src/neuralk_oss/
├── model/        # Seldon's architecture, config and checkpoint format
├── inference/    # sdm estimator, preprocessing recipe, scikit-learn style classifier and regressor
└── forecasting/  # TabPFN-TS's time features and pipeline around the regressor
tests/
```

## Development

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

To run ruff's check (with fixes) and formatter on every commit, install the
[pre-commit](https://pre-commit.com) hooks once:

```bash
uvx pre-commit install
```

`tests/test_golden.py` compares the outputs of tiny random-weight checkpoints
(`tests/golden/`) to saved references. After an intended change of outputs,
regenerate them with `UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py` and
commit the new `tests/golden/expected/`. With CUDA, the same cases also run on
the GPU in float16 autocast; the classifier's and the regressor's CUDA
references (`tests/golden/expected/cuda/`) are only regenerated on a CUDA machine.

On a machine with a CUDA GPU, [tox](https://tox.wiki) runs the whole suite in
two environments, `gpu-ours` (the versions of `uv.lock`) and `gpu-tabarena`
(the versions [TabArena](https://github.com/autogluon/tabarena) installs our
dependencies at, `tests/envs/tabarena.txt`); both fail if torch sees no GPU:

```bash
uvx --with tox-uv tox                  # both
uvx --with tox-uv tox -e gpu-tabarena  # one; pytest arguments after `--`
```

## Versions

Versions follow [semantic versioning](https://semver.org), `MAJOR.MINOR.PATCH`,
set in `pyproject.toml` only (`__version__` reads it from the installed package).
While in `0.x`:

- **patch** (`0.1.0` → `0.1.1`): fixes and speedups with the same outputs;
  the golden references are unchanged;
- **minor** (`0.1.0` → `0.2.0`): a change of the public API, of outputs
  (golden references regenerated) or of a dependency pin, a new checkpoint.

To release: bump the version in a PR, merge it, then tag the merge commit on
`main` and push the tag:

```bash
uv version --bump patch  # or minor; edits pyproject.toml and uv.lock
# PR, merge, then on main:
git tag v0.1.1 && git push origin v0.1.1
```

The `Release` workflow (`.github/workflows/release.yml`) checks the tag matches
`pyproject.toml` and publishes to PyPI, once `paulnovello-nk` or `nicolasAtza`
approves the run (required reviewers of the `pypi` environment). Running it by
hand from the Actions tab publishes to TestPyPI instead, with no approval. A
version is uploaded once per index.
