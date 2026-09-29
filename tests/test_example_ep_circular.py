"""CI regression test converted from `examples/pyloki_ep_circular.ipynb`.

The circular-orbit EP search at `prune_poly_order=5`, recovering all five kinematic
derivatives (crackle, snap, jerk, accel, freq) of a pulsar in a circular binary. Read
`test_example_ep_accel.py`'s docstring first for the shared pipeline; only the
differences are covered here.

**The notebook does not run against the current API.** Its cell 2 calls
`ParamLimits.from_circular(..., m_p=m_p_min, ...)`, but the parameter is named
`m_p_min` (`config.py:206`), so that cell raises `TypeError`. This test uses the
correct keyword. Converting the example is what surfaced it -- worth fixing in the
notebook separately.

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

**Measured worst case over those 10 realisations** (reported uncertainty in brackets):
crackle 0.0149 [0.0484], snap 0.260 [0.345], jerk 3.29 [2.45], accel 70.3 [39.3],
freq 3.39e-4 [1.84e-4]. Every one is within 0.2% of the injected value. Tolerances
below are ~3x those worst cases.

Cost: ~28 s locally on a cold numba cache, ~16 s warm -- the most expensive of the three
example tests by an order of magnitude, because `poly_order=5` over 64 stages is simply
a big search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyloki.config import ParamLimits, PulsarSearchConfig
from pyloki.detection import thresholding
from pyloki.ffa import DynamicProgramming
from pyloki.periodogram import ScatteredPeriodogram
from pyloki.prune import prune_dyp_tree
from pyloki.simulation.pulse import PulseSignalConfig

if TYPE_CHECKING:
    from pathlib import Path

# Pinned so CI is reproducible. Before the library took a `seed`, the noise and the
# threshold ladder were drawn from unseeded generators inside pyloki and this file
# had no way to reach them; see tests/test_rng_seeding.py.
SEED = 42

# Measured over 20 seeded realisations. Every parameter passed at every seed, with
# error/tolerance peaking at 0.386 (freq). These three seeds are the worst observed
# case for different parameters, so between them they exercise all five:
#   1  -> worst crackle (0.234), jerk (0.329) and freq (0.386)
#   14 -> worst snap (0.330)
#   7  -> worst accel (0.319)
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

# --- Tolerances: ~3x the measured worst case over 10 realisations -------------------
# param: (absolute tolerance, max uncertainty as a fraction of |true| for the guard)
TOLERANCES = {
    "crackle": (0.05, 0.05),  # worst err 0.0149, unc 0.0484 (0.7% of 7.12)
    "snap": (0.8, 0.05),  # worst err 0.260, unc 0.345 (0.2% of 163.6)
    "jerk": (10.0, 0.05),  # worst err 3.29, unc 2.45 (0.06% of 4058)
    "accel": (220.0, 0.05),  # worst err 70.3, unc 39.3 (0.04% of 93221)
    "freq": (1e-3, 1e-4),  # worst err 3.39e-4, unc 1.84e-4
}


def _run_search(seed: int, outdir: Path) -> dict:
    """Run the notebook's pipeline at one noise realisation."""
    tobs = NSAMPS * DT
    p_orb = tobs / TOBS_OVER_PORB
    cfg = PulseSignalConfig(
        period=PULSAR_PERIOD,
        dt=DT,
        nsamps=NSAMPS,
        snr=SNR,
        ducy=DUCY,
        mod_kwargs={"p_orb": p_orb, "psi": PSI, "m_c": M_C},
        mod_type="circular",
        seed=seed,
    )
    tim_data = cfg.generate(shape="gaussian")
    # The truth is the Taylor gauge of the circular orbit, as the notebook computes it.
    truth = cfg.mod_func.to_derivatives_gauge(cfg.freq)

    # NOTE: m_p_min, not m_p -- the notebook's spelling is stale (see docstring).
    limits = ParamLimits.from_circular(
        FREQ_RANGE, tobs, M_C_MAX, m_p_min=M_P_MIN, poly_order=5
    )
    bseg_ffa = NSAMPS // NSEGMENTS
    search_cfg = PulsarSearchConfig(
        nsamps=cfg.nsamps,
        tsamp=cfg.dt,
        nbins=NBINS,
        eta=ETA,
        param_limits=limits.limits,
        bseg_brute=bseg_ffa // FFA_LEVELS,
        bseg_ffa=bseg_ffa,
        prune_poly_order=5,
        m_p_min=M_P_MIN,
        m_c_max=M_C_MAX,
        p_orb_min=tobs,
        ducy_max=0.5,
        wtsp=1.2,
        use_fourier=True,
        branch_max=BRANCH_MAX,
    )
    dyp = DynamicProgramming(tim_data, search_cfg)
    dyp.initialize()
    dyp.execute()

    branching_pattern = np.asarray(
        search_cfg.generate_branching_pattern(
            kind="circ_taylor_moving", ref_seg=dyp.nsegments // 2
        ),
        dtype=np.float64,
    )
    scheme = thresholding.determine_scheme(
        1.0 / branching_pattern,
        branching_pattern,
        ref_ducy=DUCY,
        nbins=NBINS,
        ntrials=1024,
        snr_final=SNR,
        ducy_max=0.5,
        wtsp=1.2,
        seed=seed,
    )
    thresholds = np.asarray(scheme.thresholds, dtype=np.float64)

    result_file = prune_dyp_tree(
        dyp,
        thresholds,
        n_runs=1,
        max_sugg=MAX_SUGG,
        outdir=str(outdir),
        file_prefix="test_circular",
        poly_basis="taylor",
        n_workers=1,
        use_moving_grid=True,
    )
    pgram = ScatteredPeriodogram.load(result_file)
    return {
        "data": pgram.data,
        # to_derivatives_gauge names it "acc"; the periodogram column is "accel".
        "truth": {
            "crackle": float(truth["crackle"]),
            "snap": float(truth["snap"]),
            "jerk": float(truth["jerk"]),
            "accel": float(truth["acc"]),
            "freq": float(truth["freq"]),
        },
        "nsegments": dyp.nsegments,
        "thresholds": thresholds,
        "branching_pattern": branching_pattern,
    }


@pytest.fixture(scope="module")
def ep_circular_search(tmp_path_factory) -> dict:
    """Run the pinned realisation once; the fast tests share it."""
    return _run_search(SEED, tmp_path_factory.mktemp("ep_circular"))


def _check_param(result: dict, param: str, seed: object = SEED) -> None:
    """Check one derivative. One source for the predicate, shared by the sweep."""
    data = result["data"]
    true = result["truth"][param]
    tol, max_unc_fraction = TOLERANCES[param]
    best = data.loc[data["score"].idxmax()]

    uncertainty = float(best[f"d{param}"])
    assert uncertainty < max_unc_fraction * abs(true), (
        f"{param} is not actually constrained (seed={seed}): d{param}="
        f"{uncertainty:.5g} against |{param}|={abs(true):.5g} -- the recovery "
        f"check below would be vacuous"
    )
    error = abs(float(best[param]) - true)
    assert error < tol, (
        f"{param} not recovered (seed={seed}): got {float(best[param]):.6g}, "
        f"want {true:.6g} (error {error:.5g} > {tol:.5g}, d{param}="
        f"{uncertainty:.5g})"
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
    for param in TOLERANCES:
        _check_param(result, param, seed)


def test_pipeline_produces_candidates(ep_circular_search) -> None:
    data = ep_circular_search["data"]
    assert len(data) > 0, "no candidates survived; the search found nothing at all"
    for name in TOLERANCES:
        assert name in data.columns, f"missing column {name!r}"
        assert f"d{name}" in data.columns, f"missing column d{name!r}"


def test_stage_count_matches_configuration(ep_circular_search) -> None:
    assert ep_circular_search["nsegments"] == NSEGMENTS
    assert len(ep_circular_search["thresholds"]) == len(
        ep_circular_search["branching_pattern"]
    )


@pytest.mark.parametrize("param", list(TOLERANCES))
def test_recovers_injected_parameter(ep_circular_search, param: str) -> None:
    """All five kinematic derivatives must come back, each with a non-vacuity guard."""
    _check_param(ep_circular_search, param)


def test_best_candidate_clears_the_final_threshold(ep_circular_search) -> None:
    data = ep_circular_search["data"]
    best_score = float(data["score"].max())
    final_threshold = float(ep_circular_search["thresholds"][-1])
    assert best_score > final_threshold, (
        f"best score {best_score:.3f} does not clear the final threshold "
        f"{final_threshold:.3f}"
    )
