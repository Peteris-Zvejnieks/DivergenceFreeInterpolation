# Project rules

- All Python code must be formatted with `uv run ruff format` and pass `uv run ruff check`
  (config in `pyproject.toml`; CI enforces both). The editor formats with Ruff on save, so
  unformatted files turn into large diffs as soon as they are touched.
  After editing Python files (including Python code blocks in Markdown, which ruff also formats),
  run `uv run ruff check --fix` and `uv run ruff format`, and fix anything that can't be fixed
  automatically. Import blocks are sorted: stdlib, third-party, `divergence_free_interpolant`,
  then local test helpers such as `conftest`. Put comments on their own line rather than after
  code that may get wrapped.
- Rust code (the `rust/` crate, built as `divergence_free_interpolant._core`) must be formatted
  with `cargo fmt` and pass `cargo clippy --all-targets -- -D warnings`, both without and with
  `--features python`; run them from `rust/` (CI enforces all of this). `cargo test` tests the
  core without Python. After changing Rust code, `uv sync` rebuilds the extension.
- `src/divergence_free_interpolant/_wendland_table.py` is generated: never edit it by hand, change
  `tools/gen_kernels.py` (or the derivation in `_wendland_sympy.py`) and rerun it. sympy is an
  optional dependency: only `_wendland_sympy.py` and sympy-defined boundaries may import it, and
  only lazily.
- Tests: `uv run pytest`. Warnings are treated as errors. The numpy reference implementation
  must keep passing too: `DFI_BACKEND=numpy uv run pytest`.
