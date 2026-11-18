"""Memory and output regression for `determine_scheme` / `evaluate_scheme`.

Both functions used to append every stage's `Folds` to a list they only ever read at
`[istage - 1]`, so peak memory grew linearly with the number of stages on top of the
single stage's ``ntrials x nbins``. The memory tests pin the growth. The output tests
check that keeping one stage still yields every stage's state, with the invariants a
stage-to-stage chain must satisfy, deterministically under a seed.
"""

from __future__ import annotations

import tracemalloc
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyloki.detection import thresholding

if TYPE_CHECKING:
    from pyloki.detection.schemes import StateInfo

if TYPE_CHECKING:
    from collections.abc import Callable

SEED = 42
KW = {"ref_ducy": 0.1, "nbins": 64, "snr_final": 8.0, "ducy_max": 0.5, "wtsp": 1.2}


def _peak(fn: Callable[[], object]) -> int:
    fn()  # JIT and allocator warm-up outside the traced window
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def _determine(nstages: int) -> Callable[[], object]:
    bp = np.ones(nstages)
    return lambda: thresholding.determine_scheme(
        1 / bp,
        bp,
        ntrials=4096,
        seed=SEED,
        **KW,
    )


def _evaluate(nstages: int) -> Callable[[], object]:
    bp = np.ones(nstages)
    thresholds = np.full(nstages, -np.inf)
    return lambda: thresholding.evaluate_scheme(
        thresholds,
        bp,
        ntrials=4096,
        seed=SEED,
        **KW,
    )


@pytest.mark.parametrize("make", [_determine, _evaluate], ids=["determine", "evaluate"])
def test_peak_memory_does_not_grow_with_stages(
    make: Callable[[int], Callable[[], object]],
) -> None:
    # Retaining every stage made 64 stages cost 5.9x (determine) and 8.7x (evaluate)
    # what 8 stages cost; keeping one stage, 1.55x and 1.01x. determine_scheme's 1.55x
    # is simulate_folds refilling to ~2x ntrials rows once an H1 trial is lost.
    assert _peak(make(64)) < 3 * _peak(make(8))


PATTERNS = {
    "mixed": np.array([1.0, 2.0, 1.0, 3.0, 1.0, 1.5, 1.0, 2.0]),
    "unbranched": np.ones(40),
}
OUT_KW = {
    "ref_ducy": 0.1,
    "nbins": 32,
    "ntrials": 512,
    "snr_final": 8.0,
    "ducy_max": 0.5,
    "wtsp": 1.2,
}


def _check_chain(entries: list[StateInfo], bp: np.ndarray) -> None:
    """Invariants of a stage-to-stage chain, whatever the Monte Carlo draws."""
    # Every stage is reported: none is dropped by keeping only the previous one.
    assert len(entries) == len(bp)
    np.testing.assert_array_equal([e.nbranches for e in entries], bp)
    h1 = np.array([e.success_h1 for e in entries])
    h1_cumul = np.array([e.success_h1_cumul for e in entries])
    assert np.all((h1 >= 0.0) & (h1 <= 1.0))
    # Cumulative survival is the product of the per-stage survivals ...
    np.testing.assert_allclose(h1_cumul, np.cumprod(h1), rtol=1e-5)
    # ... and cumulative complexity only grows.
    assert np.all(np.diff([e.complexity_cumul for e in entries]) > 0.0)


@pytest.mark.parametrize("name", list(PATTERNS))
def test_determine_scheme_output(name: str) -> None:
    bp = PATTERNS[name]
    got = thresholding.determine_scheme(1 / bp, bp, seed=SEED, **OUT_KW).entries
    _check_chain(got, bp)
    # Each threshold is set so the requested fraction of H0 trials survives.
    np.testing.assert_allclose([e.success_h0 for e in got], 1 / bp, atol=0.01)
    assert got == thresholding.determine_scheme(1 / bp, bp, seed=SEED, **OUT_KW).entries
    # The comparison can fail: another seed gives other states.
    assert (
        got
        != thresholding.determine_scheme(1 / bp, bp, seed=SEED + 1, **OUT_KW).entries
    )


@pytest.mark.parametrize("name", list(PATTERNS))
def test_evaluate_scheme_output(name: str) -> None:
    bp = PATTERNS[name]
    thresholds = np.linspace(0.5, 3.0, len(bp))
    got = thresholding.evaluate_scheme(thresholds, bp, seed=SEED, **OUT_KW).entries
    _check_chain(got, bp)
    # The ladder that was asked for is the ladder that was applied.
    np.testing.assert_allclose([e.threshold for e in got], thresholds, rtol=1e-6)
    assert (
        got == thresholding.evaluate_scheme(thresholds, bp, seed=SEED, **OUT_KW).entries
    )
    assert (
        got
        != thresholding.evaluate_scheme(thresholds, bp, seed=SEED + 1, **OUT_KW).entries
    )
