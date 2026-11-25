## What and why

<!-- What this PR changes and why. Link the issue it fixes, if any. -->

## Checklist

- [ ] `uv run pytest` passes
- [ ] `uv run ruff check . && uv run ruff format --check .` is clean (the pre-commit hooks run both)
- [ ] If outputs changed: golden references regenerated (`UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py`) and `tests/golden/expected/` committed
- [ ] The version in `pyproject.toml` is unchanged, unless this is a release PR
