"""CI regression test converted from `examples/optimal_thresholds.ipynb`.

Covers the threshold-scheme machinery: `determine_scheme` builds a ladder from target
per-stage survival probabilities, `evaluate_scheme` scores a ladder and reports
per-stage complexity and cumulative detection probability, and `schemes.bound_scheme` /
`schemes.trials_scheme` provide the two analytic reference ladders the notebook compares
against.

`evaluate_scheme` and the `schemes` helpers were untested before this.
`determine_scheme` is used by the EP example tests, but only to produce a
ladder -- nothing asserted anything about it.

## What is deliberately NOT converted: `DynamicThresholdScheme.run()`

The notebook's centrepiece is the Viterbi threshold optimiser (cells 3-4). It is left
out because **it aborts the interpreter in this environment**:

    Numba workqueue threading layer is terminating: Concurrent access has been detected.

exit code 134 (SIGABRT). `run_stage_improved` / `run_stage_legacy` use `prange` and call
further parallel functions, which is a nested parallel region, and numba's default
`workqueue` layer explicitly cannot do that. Verified:

- both `mode="improved"` and `mode="legacy"` abort;
- `NUMBA_NUM_THREADS=1` does not help;
- TBB, the layer numba's own error message recommends, has no wheel for macOS arm64
  (`pip install tbb` -> "No matching distribution found"), so it is not a portable fix.

This is not something a test can skip around: `SIGABRT` kills the worker process, so
pytest cannot catch it and turn it into a skip. Including the call would either crash
the CI run or pass, depending on which threading layer numba happens to select on the
runner -- a coin flip either way. It is left out until the nesting is fixed upstream,
and it is worth reporting: a public API demonstrated in a shipped example aborts the
process on a default install.

## The `schemes` import is guarded

`pyloki/detection/schemes.py` imports `tkinter`, `tkinter.filedialog`,
`tkinter.messagebox`, `tkinter.ttk` and `matplotlib.backends.backend_tkagg` at **module
level** (lines 4, 6, 15), so the module is unimportable wherever Tk is absent -- which
includes minimal and headless installs. The tests that need it use
`pytest.importorskip`, so they run where Tk exists and skip cleanly where it does not,
rather than failing. Also worth reporting: a library module should not require a GUI
toolkit at import time.

## Sizing

The notebook uses its own 127-stage branching pattern with `ntrials=2**10`. This uses a
16-stage pattern of the same shape (branch hard early, settle to 1) and the same
`ntrials`, `ref_ducy`, `nbins`, `ducy_max`, `wtsp` and `snr_final`. Nothing here scales
with data volume -- it is pure Monte Carlo over folded profiles -- so the whole module
runs in a few seconds.

**Measured stability** over 8 independent runs of the unseeded Monte Carlo: detection
probability at `snr_final` 4/7/10/13/16 came out in the ranges 0.008-0.016, 0.209-0.257,
0.672-0.715, 0.918-0.939, 0.989-0.996, and was monotonic in 8/8. The gaps between
adjacent points are an order of magnitude wider than the spread, so the monotonicity
assertion has a large margin.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyloki.detection import thresholding

# --- The notebook's setup, unchanged except the stage count ------------------------
TARGET_SNR = 10.0
REF_DUCY = 0.1
NBINS = 64
DUCY_MAX = 0.3
WTSP = 1.5
NTRIALS = 2**10
NSTAGES = 16  # notebook: 127

# Injected-S/N ladder for the detection-probability curve, and loose brackets on the
# ends. Measured ranges were 0.008-0.016 and 0.989-0.996.
SNR_LADDER = (4.0, 7.0, 10.0, 13.0, 16.0)
MAX_DET_PROB_AT_LOW_SNR = 0.10
MIN_DET_PROB_AT_HIGH_SNR = 0.90


def branching_pattern() -> np.ndarray:
    """A short pattern with the notebook's shape: branch hard early, then settle."""
    pattern = np.ones(NSTAGES)
    pattern[0] = 8.0
    pattern[1:6] = 3.0
    pattern[6] = 6.0
    return pattern


def detection_probability(ladder: np.ndarray, pattern: np.ndarray, snr: float) -> float:
    """Cumulative probability of the signal surviving every stage, as the notebook."""
    state = thresholding.evaluate_scheme(
        ladder,
        pattern,
        ref_ducy=REF_DUCY,
        nbins=NBINS,
        ntrials=NTRIALS,
        snr_final=snr,
        ducy_max=DUCY_MAX,
        wtsp=WTSP,
    )
    return float(np.asarray(state.get_info("success_h1_cumul"))[-1])


@pytest.fixture(scope="module")
def constant_scheme() -> dict:
    """The notebook's "constant" scheme: survival probability 1/branching per stage."""
    pattern = branching_pattern()
    info = thresholding.determine_scheme(
        1.0 / pattern,
        pattern,
        ref_ducy=REF_DUCY,
        nbins=NBINS,
        ntrials=NTRIALS,
        snr_final=TARGET_SNR,
        ducy_max=DUCY_MAX,
        wtsp=WTSP,
    )
    return {
        "pattern": pattern,
        "info": info,
        "ladder": np.asarray(info.thresholds, dtype=np.float64),
    }


def test_determine_scheme_returns_a_usable_ladder(constant_scheme) -> None:
    ladder = constant_scheme["ladder"]
    assert len(ladder) == NSTAGES
    assert np.all(np.isfinite(ladder)), f"non-finite thresholds: {ladder}"
    assert np.all(ladder > 0), f"non-positive thresholds: {ladder}"


def test_evaluate_scheme_reports_per_stage_state(constant_scheme) -> None:
    state = thresholding.evaluate_scheme(
        constant_scheme["ladder"],
        constant_scheme["pattern"],
        ref_ducy=REF_DUCY,
        nbins=NBINS,
        ntrials=NTRIALS,
        snr_final=TARGET_SNR,
        ducy_max=DUCY_MAX,
        wtsp=WTSP,
    )
    assert len(state.entries) == NSTAGES
    complexity = np.asarray(state.get_info("complexity"))
    assert len(complexity) == NSTAGES
    assert np.all(complexity > 0), "complexity must be positive at every stage"
    det = np.asarray(state.get_info("success_h1_cumul"))
    assert np.all((det >= 0.0) & (det <= 1.0)), f"probabilities out of range: {det}"
    # Cumulative survival can only fall as stages are added.
    assert np.all(np.diff(det) <= 1e-12), f"cumulative survival increased: {det}"


def test_detection_probability_rises_with_injected_snr(constant_scheme) -> None:
    """The headline property: a stronger signal must be more likely to survive.

    Monotonic in 8/8 measured runs, with adjacent points an order of magnitude further
    apart than the Monte Carlo spread.
    """
    ladder, pattern = constant_scheme["ladder"], constant_scheme["pattern"]
    probs = [detection_probability(ladder, pattern, snr) for snr in SNR_LADDER]
    for i in range(len(probs) - 1):
        assert probs[i + 1] >= probs[i], (
            f"detection probability fell from {probs[i]:.4f} at "
            f"snr={SNR_LADDER[i]} to {probs[i + 1]:.4f} at snr={SNR_LADDER[i + 1]}; "
            f"full curve {[round(p, 4) for p in probs]}"
        )
    assert probs[0] < MAX_DET_PROB_AT_LOW_SNR, (
        f"a snr={SNR_LADDER[0]} signal should rarely survive, got {probs[0]:.4f}"
    )
    assert probs[-1] > MIN_DET_PROB_AT_HIGH_SNR, (
        f"a snr={SNR_LADDER[-1]} signal should almost always survive, "
        f"got {probs[-1]:.4f}"
    )


def test_reference_ladders_and_their_ordering(constant_scheme) -> None:
    """`bound_scheme` vs `trials_scheme`, and that a stricter ladder detects less.

    Skipped where Tk is unavailable: `pyloki.detection.schemes` imports tkinter at
    module level (see the module docstring).
    """
    schemes = pytest.importorskip(
        "pyloki.detection.schemes",
        reason="pyloki.detection.schemes imports tkinter at module level",
    )
    pattern = constant_scheme["pattern"]

    bound = np.asarray(schemes.bound_scheme(NSTAGES, TARGET_SNR), dtype=np.float64)
    trials = np.asarray(schemes.trials_scheme(pattern, 1), dtype=np.float64)
    assert len(bound) == NSTAGES
    assert len(trials) == NSTAGES
    assert np.all(np.isfinite(bound)) and np.all(np.isfinite(trials))

    # The notebook's combination, and what it means.
    minimized = np.minimum(bound, trials)
    np.testing.assert_array_equal(minimized, np.minimum(bound, trials))
    assert np.all(minimized <= bound + 1e-12)
    assert np.all(minimized <= trials + 1e-12)

    # `bound` is the strict ladder and `trials` the permissive one here, so `bound`
    # must not detect more. Measured: ~0.09 against ~0.996.
    det_bound = detection_probability(bound, pattern, TARGET_SNR)
    det_trials = detection_probability(trials, pattern, TARGET_SNR)
    assert det_bound <= det_trials, (
        f"the stricter 'bound' ladder detected more ({det_bound:.4f}) than the "
        f"permissive 'trials' ladder ({det_trials:.4f})"
    )
