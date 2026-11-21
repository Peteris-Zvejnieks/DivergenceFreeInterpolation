# Project rules

- All Python code must be formatted with `uv run ruff format` and pass `uv run ruff check`
  (config in `pyproject.toml`; CI enforces both). The editor formats with Ruff on save, so
  unformatted files turn into large diffs as soon as they are touched.
  After editing Python files (including Python code blocks in Markdown, which ruff also formats),
  run `uv run ruff check --fix` and `uv run ruff format`, and fix anything that can't be fixed
  automatically. Import blocks are sorted: stdlib, third-party, `divergence_free_interpolant`,
  then local test helpers such as `conftest`. Put comments on their own line rather than after
  code that may get wrapped.
- Tests: `uv run pytest`. Warnings are treated as errors.
