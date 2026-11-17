# pyloki

Leverage Optimal significance to detect Keplerian orbIt pulsars.

[![GitHub CI](https://github.com/pravirkr/pyloki/actions/workflows/ci.yml/badge.svg)](https://github.com/pravirkr/pyloki/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pravirkr/pyloki/graph/badge.svg)](https://codecov.io/gh/pravirkr/pyloki)

PyLoki is the Python prototyping and analysis companion to [**LOKI**](https://github.com/pravirkr/loki), the production-grade C++/CUDA implementation for large-scale HPC searches.

Example notebooks live in [`examples/`](https://github.com/pravirkr/pyloki/tree/main/examples).

## Install

**Requirements:** Python 3.12+.

### uv (recommended for development)

```bash
git clone https://github.com/pravirkr/pyloki.git
cd pyloki
uv sync
```

### pip

```bash
pip install "pyloki @ git+https://github.com/pravirkr/pyloki.git"
```

## Development

```bash
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest
```

Expensive seed-sweep tests are skipped by default; run them with `uv run pytest --runslow`.

See [docs/architecture.md](https://github.com/pravirkr/pyloki/blob/main/docs/architecture.md) for development rules and architecture guidelines.
