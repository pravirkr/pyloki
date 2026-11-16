# Working on PyLoki

Read `docs/architecture.md` before changing code. It is the frozen layout and
the rules every change must follow (public API, dispatch).

Before finishing a change:

- `uv run ruff check` and `uv run ruff format --check`
- `uv run ty check`
- `uv run pytest` (default; no `--runslow` unless the change touches slow tests)
- Expensive integration tests: `@pytest.mark.slow`

Do not stage or commit unless asked.
