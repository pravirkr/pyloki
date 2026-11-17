"""CI regression test converted from `examples/pyloki_ep_circular.ipynb`.

The circular-orbit EP search at `prune_poly_order=5`, recovering all five kinematic
derivatives (crackle, snap, jerk, accel, freq) of a pulsar in a circular binary. Read
`test_example_ep_accel.py`'s docstring first for the shared pipeline; only the
differences are covered here.

Scaling: `nsamps 2**24 -> 2**21`, 128 stages -> 64, `n_runs 16 -> 1`,
`max_sugg 2**21 -> 2**18`, 4 workers -> 1, `branch_max 128 -> 32`, thresholds derived
rather than hardcoded, `snr 11 -> 40`. The branching kind is `"circ_taylor_moving"`,
as in the notebook.

`p_orb` is scaled with `tobs` to preserve the notebook's `tobs / p_orb = 0.895`, so the
orbit stays as well sampled as the example intends: `p_orb = 1200 -> 149.96 s`. Simply
keeping `p_orb = 1200` while shrinking `tobs` would leave only ~11% of an orbit covered
and change what the test exercises.

**Two sizing findings, both measured:**

*Fewer stages makes this example worse, not better.* At 8 stages the branching pattern
starts `[160, 180, ...]` -- cumulative ~1.9e8 leaves -- and the run never finishes
(killed after 10 min, at `branch_max` 128 and 32 alike). The total refinement is fixed
by `tobs`, so squeezing it into fewer stages makes each one explosive. At 64 stages the
pattern starts `[4, 9, 1, 6.35, ...]` and the prune takes ~15 s. This is the opposite of
the accel example, where 8 stages was the cheap choice.

*`snr` had to go to 40, well above the other two tests' 20.* At `snr=20` this test was
badly flaky: over 4 realisations, one returned **no candidates at all**, one returned a
confidently *wrong* best candidate (score 11.5, jerk error 151 = 62x its reported
uncertainty), and only two recovered cleanly. The notebook's `n_runs=16` is not merely
statistics -- a single pruning run can lose the signal -- and dropping to `n_runs=1`
removes that protection. Raising `snr` restores it: **10/10 clean at `snr=40`**, scores
38.2-39.6.

**Measured error over the reported uncertainty**, worst over 20 seeded realisations:
crackle 0.24, snap 0.77, jerk 1.34, accel 1.79, freq 2.10. The first four are checked
by the shared `k = 2` rule (`tests/_pipelines.py`); accel's 1.79 is the tightest margin.
Frequency is the exception, as in the jerk example: `dfreq` is optimistic here, so it
keeps an absolute tolerance. This is the most expensive of the three EP examples,
because `poly_order=5` over 64 stages is simply a big search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyloki.config import ParamLimits
from pyloki.simulation.pulse import PulseSignalConfig
from tests._pipelines import (
    EpSearch,
    check_candidates,
    check_detection_strength,
    check_recovered,
    check_stage_count,
    run_ep,
)

if TYPE_CHECKING:
    from pathlib import Path

SEED = 42  # Pinned so CI is reproducible.

# Measured over 20 seeded realisations. These three seeds are the worst observed
# case (error / d<param>) for different parameters, so between them they exercise
# all five:
#   1  -> worst crackle (0.24), jerk (1.34) and freq (2.10)
#   14 -> worst snap (0.77)
#   7  -> worst accel (1.79)
SEED_SWEEP = (1, 14, 7)

# --- The notebook's physical setup ------------------------------------------------
PULSAR_PERIOD = 0.007
DT = 64e-6
NBINS = 64
ETA = 1
DUCY = 0.1
FREQ_RANGE = (142.5, 143.5)
M_P_MIN = 1.2
M_C_MAX = 10.0
M_C = 8.0
PSI = np.pi / 4.1
TOBS_OVER_PORB = 0.895  # notebook: tobs=1073.7 with p_orb=1200

# --- Scaled for CI (see the module docstring) --------------------------------------
NSAMPS = 2**21
NSEGMENTS = 64
FFA_LEVELS = 16
SNR = 40.0
MAX_SUGG = 2**18
BRANCH_MAX = 32

# --- Recovery: the shared `check_recovered` rule, |err| <= 2 x d<param> -------------
# param -> max uncertainty as a fraction of |true| (the non-vacuity guard)
MAX_UNC_FRACTION = {
    "crackle": 0.05,  # dcrackle is 0.7% of 7.12
    "snap": 0.05,  # dsnap is 0.2% of 163.6
    "jerk": 0.05,  # djerk is 0.06% of 4058
    "accel": 0.05,  # daccel is 0.04% of 93221
    "freq": 1e-4,  # dfreq is 1.3e-6 of freq
}
# Exception: dfreq is optimistic (err/dfreq up to 2.10), so freq gets an absolute
# tolerance; worst measured error 3.9e-4.
ABS_TOL = {"freq": 1e-3}

TOBS = NSAMPS * DT
SEARCH = EpSearch(
    nsegments=NSEGMENTS,
    prune_poly_order=5,
    kind="circ_taylor_moving",
    max_sugg=MAX_SUGG,
    ffa_levels=FFA_LEVELS,
    nbins=NBINS,
    eta=ETA,
    branch_max=BRANCH_MAX,
    config_kwargs={"m_p_min": M_P_MIN, "m_c_max": M_C_MAX, "p_orb_min": TOBS},
)


def _run_search(seed: int, outdir: Path) -> dict:
    """Run the notebook's pipeline at one noise realisation."""
    signal = PulseSignalConfig(
        period=PULSAR_PERIOD,
        dt=DT,
        nsamps=NSAMPS,
        snr=SNR,
        ducy=DUCY,
        mod_kwargs={"p_orb": TOBS / TOBS_OVER_PORB, "psi": PSI, "m_c": M_C},
        mod_type="circular",
        seed=seed,
    )
    limits = ParamLimits.from_circular(
        FREQ_RANGE, TOBS, M_C_MAX, m_p_min=M_P_MIN, poly_order=5
    )
    result = run_ep(signal, limits.limits, SEARCH, seed, outdir)
    # The truth is the Taylor gauge of the circular orbit, as the notebook computes
    # it. `to_derivatives_gauge` names it "acc"; the periodogram column is "accel".
    truth = signal.mod_func.to_derivatives_gauge(signal.freq)
    result["truth"] = {
        "crackle": float(truth["crackle"]),
        "snap": float(truth["snap"]),
        "jerk": float(truth["jerk"]),
        "accel": float(truth["acc"]),
        "freq": float(truth["freq"]),
    }
    return result


@pytest.fixture(scope="module")
def ep_circular_search(tmp_path_factory) -> dict:
    """Run the pinned realisation once; the fast tests share it."""
    return _run_search(SEED, tmp_path_factory.mktemp("ep_circular"))


def _check_param(result: dict, param: str, seed: object = SEED) -> None:
    """Check one derivative. One source for the predicate, shared by the sweep."""
    check_recovered(
        result,
        param,
        result["truth"][param],
        MAX_UNC_FRACTION[param],
        seed,
        abs_tol=ABS_TOL.get(param),
    )


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEED_SWEEP)
def test_recovery_does_not_depend_on_the_noise_realisation(
    seed: int,
    tmp_path: Path,
) -> None:
    """All five derivatives must come back at every realisation, not just `SEED`.

    Fixed seeds, never random: a random seed makes a failure unreproducible, which
    is the defect this suite was fixed to not have. The seeds are the measured
    worst case for each parameter; see `SEED_SWEEP`.
    """
    result = _run_search(seed, tmp_path)
    for param in MAX_UNC_FRACTION:
        _check_param(result, param, seed)
    check_detection_strength(result, seed)


def test_pipeline_produces_candidates(ep_circular_search) -> None:
    check_candidates(
        ep_circular_search, [c for name in MAX_UNC_FRACTION for c in (name, f"d{name}")]
    )


def test_stage_count_matches_configuration(ep_circular_search) -> None:
    check_stage_count(ep_circular_search, NSEGMENTS)


@pytest.mark.parametrize("param", list(MAX_UNC_FRACTION))
def test_recovers_injected_parameter(ep_circular_search, param: str) -> None:
    """All five kinematic derivatives must come back, each with a non-vacuity guard."""
    _check_param(ep_circular_search, param)


def test_best_candidate_carries_the_injected_snr(ep_circular_search) -> None:
    check_detection_strength(ep_circular_search, SEED)
