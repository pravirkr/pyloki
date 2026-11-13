# pyloki

Leverage Optimal significance to detect Keplerian orbIt pulsars.

[![GitHub CI](https://github.com/pravirkr/pyloki/actions/workflows/ci.yml/badge.svg)](https://github.com/pravirkr/pyloki/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pravirkr/pyloki/graph/badge.svg)](https://codecov.io/gh/pravirkr/pyloki)

Example notebooks live in [`examples/`](examples/).

## Install

**Requirements:** Python 3.12+, [git](https://git-scm.com/) (needed to install the
`sigpyproc` dependency from GitHub).

PyPI is not published yet; install from this repository.

### uv (recommended for development)

```bash
git clone https://github.com/pravirkr/pyloki.git
cd pyloki
uv sync --extra tests --extra dev
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

See [docs/architecture.md](docs/architecture.md) for development rules and architecture guidelines.
