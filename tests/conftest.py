"""Shared pytest configuration.

`slow` marks tests that are correct and wanted but too expensive for every run --
chiefly the seed sweeps, which repeat a full search pipeline once per noise
realisation. They are **skipped with a reason** rather than silently deselected, so
a run that did not include them says so in its output.

    pytest                   # skips them, and prints why
    pytest --runslow         # runs them
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked slow (seed sweeps: minutes, not seconds)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="needs --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
