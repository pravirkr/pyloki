"""Memory and output regression for `determine_scheme` / `evaluate_scheme`.

Both functions used to append every stage's `Folds` to a list they only ever read at
`[istage - 1]`, so peak memory grew linearly with the number of stages on top of the
single stage's ``ntrials x nbins``. The memory tests pin the growth. The output tests
compare against the old keep-every-stage loop, reproduced below, on the same machine,
so exact equality holds whatever the platform's floating point does.
"""

from __future__ import annotations

import tracemalloc
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyloki.detection import thresholding
from pyloki.detection.schemes import StateInfo
from pyloki.simulation.pulse import generate_folded_profile

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
        1 / bp, bp, ntrials=4096, seed=SEED, **KW,
    )


def _evaluate(nstages: int) -> Callable[[], object]:
    bp = np.ones(nstages)
    thresholds = np.full(nstages, -np.inf)
    return lambda: thresholding.evaluate_scheme(
        thresholds, bp, ntrials=4096, seed=SEED, **KW,
    )


@pytest.mark.parametrize("make", [_determine, _evaluate], ids=["determine", "evaluate"])
def test_peak_memory_does_not_grow_with_stages(
    make: Callable[[int], Callable[[], object]],
) -> None:
    # Retaining every stage made 64 stages cost 5.9x (determine) and 8.7x (evaluate)
    # what 8 stages cost; keeping one stage, 1.55x and 1.01x. determine_scheme's 1.55x
    # is simulate_folds refilling to ~2x ntrials rows once an H1 trial is lost.
    assert _peak(make(64)) < 3 * _peak(make(8))


def _keep_every_stage(
    gen_next: Callable[..., tuple[np.ndarray, thresholding.Folds]],
    per_stage: np.ndarray,
    bp: np.ndarray,
    *,
    ntrials: int,
    seed: int,
    ref_ducy: float,
    nbins: int,
    snr_final: float,
    ducy_max: float,
    wtsp: float,
) -> list[StateInfo]:
    """Run the loop as it was before the fix, retaining every stage's folds."""
    nstages = len(bp)
    profile = generate_folded_profile(nbins=nbins, ducy=ref_ducy)
    bias_snr = snr_final / np.sqrt(nstages + 1)
    rng = np.random.default_rng(seed)
    folds = np.zeros((ntrials, len(profile)), dtype=np.float32)
    h0, _ = thresholding.simulate_folds(folds, 0, profile, rng, 0, 1.0, ntrials)
    h1, _ = thresholding.simulate_folds(folds, 0, profile, rng, bias_snr, 1.0, ntrials)
    initial_state = np.ones(1, dtype=thresholding.state_dtype)[0]
    initial_state["threshold"] = -1
    initial_state["threshold_prev"] = -1
    initial_fold_state = thresholding.Folds(h0, h1, 1.0)
    states, fold_states = [], []
    for istage in range(nstages):
        prev_state = initial_state if istage == 0 else states[istage - 1]
        prev_folds = initial_fold_state if istage == 0 else fold_states[istage - 1]
        if istage > 0 and prev_folds.is_empty:
            break
        cur_state, cur_folds = gen_next(
            prev_state, prev_folds, per_stage[istage], bp[istage], bias_snr,
            profile, rng, ntrials, ducy_max, wtsp,
        )
        states.append(cur_state[0])
        fold_states.append(cur_folds)
    return [StateInfo.from_record(state) for state in states]


PATTERNS = {
    "mixed": np.array([1.0, 2.0, 1.0, 3.0, 1.0, 1.5, 1.0, 2.0]),
    "unbranched": np.ones(40),
}
OUT_KW = {"ref_ducy": 0.1, "nbins": 32, "ntrials": 512, "snr_final": 8.0,
          "ducy_max": 0.5, "wtsp": 1.2}


@pytest.mark.parametrize("name", list(PATTERNS))
def test_determine_scheme_output_unchanged(name: str) -> None:
    bp = PATTERNS[name]
    got = thresholding.determine_scheme(1 / bp, bp, seed=SEED, **OUT_KW).entries
    want = _keep_every_stage(thresholding.gen_next_using_surv_prob, 1 / bp, bp,
                             seed=SEED, **OUT_KW)
    assert got == want
    # The comparison can fail: another seed gives other states.
    assert got != _keep_every_stage(thresholding.gen_next_using_surv_prob, 1 / bp, bp,
                                    seed=SEED + 1, **OUT_KW)


@pytest.mark.parametrize("name", list(PATTERNS))
def test_evaluate_scheme_output_unchanged(name: str) -> None:
    bp = PATTERNS[name]
    thresholds = np.linspace(0.5, 3.0, len(bp))
    got = thresholding.evaluate_scheme(thresholds, bp, seed=SEED, **OUT_KW).entries
    want = _keep_every_stage(thresholding.gen_next_using_thresh, thresholds, bp,
                             seed=SEED, **OUT_KW)
    assert got == want
    assert got != _keep_every_stage(thresholding.gen_next_using_thresh, thresholds, bp,
                                    seed=SEED + 1, **OUT_KW)
