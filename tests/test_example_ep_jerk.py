"""CI regression test converted from `examples/pyloki_ep_jerk.ipynb`.

Same pipeline as `test_example_ep_accel.py` but at `prune_poly_order=3`, searching
jerk as well as acceleration. Read that module's docstring first; only the differences
are covered here.

Scaling: `nsamps 2**25 -> 2**22`, 128 stages -> 64, `n_runs 16 -> 1`,
`max_sugg 2**22 -> 2**18`, 4 workers -> 1, thresholds derived rather than hardcoded,
`snr 10.0 -> 20`. `ref_seg` is the middle segment as in the notebook (`64` of 128
there, `nsegments // 2` here), and `tiling_strategy="aggressive"` is kept from the
notebook (it is also the library default).

**This test is deliberately larger than the accel one** -- `2**22` and 64 stages against
`2**21` and 8 -- because that is what it takes to constrain jerk. The sizing was
measured, not guessed:

- `2**21`, 8 stages: jerk **unconstrained**. The injected `jerk = 2.0` comes back as
  `0.0` with `djerk = 5.0`, exactly the full `(-2.5, 2.5)` search range, i.e. the axis
  is never branched and the leaf keeps its seed value and seed uncertainty. Identical
  across 6 realisations.
- `2**21`, 64 stages: **still unconstrained**, `djerk = 5.0`. More stages do not help;
  the branching decision is per-axis, and jerk's phase contribution over a 134 s span
  never trips the `eta` threshold. Note the search still *detects* the signal (score
  ~19) and recovers accel and frequency while leaving jerk unresolved.
- `2**22`, 64 stages: **constrained**. `djerk` falls 5.0 -> 0.556 (28% of the injected
  value) and the error is 0.222-0.333. Jerk resolution scales as `1 / tobs**3`, so the
  span is what matters here, not the stage count.

So the jerk check is a genuine recovery assertion rather than a coverage one, and it is
guarded: `djerk` must be under half the injected jerk before the recovery tolerance is
applied, so the assertion cannot pass merely because the uncertainty is huge.

**Measured stability** (5 independent noise realisations at the final configuration):
5/5 recovered, `jerk` error 0.222 in 4 of 5 and 0.333 in the fifth (`djerk = 0.556`),
`accel` error 0.0063-0.0082 (`daccel = 8.29`), `freq` error 1.6e-4 to 3.0e-4
(`dfreq = 9.2e-5`).

That 5-realisation sample missed the failure mode. Over 28 seeded realisations the
`accel` error is **bimodal**: 26 in 0.0059-0.0082 and 2 at 8.28, i.e. exactly one
`daccel` step. The best candidate either lands in the right acceleration cell or in an
adjacent one; there is no continuum between. Each of the three recovery checks is
therefore sized against one grid step rather than against the observed spread --
`JERK_TOL = 0.75` already exceeds `djerk = 0.556` and `FREQ_TOL = 2e-3` already exceeds
`dfreq = 9.2e-5`, so both survive a one-cell miss, but the old absolute
`ACCEL_TOL = 1.0` did not, and that was the whole of the flake.

Note the frequency error *exceeds* the reported `dfreq` by 1.8-3.2x here, unlike the
accel example where it stayed below it. The frequency check therefore uses an absolute
tolerance rather than a multiple of `dfreq`: the reported frequency uncertainty is
optimistic at this configuration.

Cost: ~11 s locally on a cold numba cache, ~1.5 s warm.
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

    import pandas as pd

# --- The notebook's physical setup, unchanged -------------------------------------
PULSAR_PERIOD = 0.007
DT = 64e-6
ACCEL = 500.0
JERK = 2.0
NBINS = 64
ETA = 1
DUCY = 0.1
FREQ_RANGE = (142.0, 144.0)
JERK_SEARCH_RANGE = (-2.5, 2.5)

# --- Scaled for CI ------------------------------------------------------------------
NSAMPS = 2**22
NSEGMENTS = 64
FFA_LEVELS = 16
SNR = 20.0
MAX_SUGG = 2**18

# --- Tolerances, well above the measured spread -------------------------------------
JERK_TOL = 0.75  # measured error <= 0.333, i.e. ~2.3x margin -- and see below
# Acceleration is checked against the search's OWN reported uncertainty, not an
# absolute number. The error here is bimodal, not continuous: the best candidate
# either lands in the correct accel cell (error ~0.008) or one cell away (error
# ~8.285, which IS `daccel`). Measured over 28 seeded realisations: 26 at
# 0.0059-0.0082, 2 at 8.2787/8.2915, nothing in between. An absolute `ACCEL_TOL =
# 1.0` therefore looked like a >100x margin while actually demanding that the peak
# land in exactly the right cell -- 8x tighter than the uncertainty the search
# itself quotes -- and that is what made this test flaky.
ACCEL_TOL_DACCEL = 1.5  # i.e. recovery within ~1 grid step of the reported value
FREQ_TOL = 2e-3  # measured error <= 3.0e-4; deliberately NOT a multiple of dfreq
# Guards so the recovery checks cannot pass vacuously.
MAX_DJERK_FRACTION = 0.50  # djerk 0.556 is 28% of the injected jerk -> passes
MAX_DACCEL_FRACTION = 0.10  # daccel 8.29 is 1.7% of accel -> passes

# Pinned so CI is reproducible. Before the library took a `seed`, the noise and the
# threshold ladder were drawn from unseeded generators inside pyloki and this file
# had no way to reach them; see tests/test_rng_seeding.py.
SEED = 42

# The sweep. Fixed and enumerated, never random -- a random seed would make a
# failure unreproducible, which is the defect this file exists to not have.
# 7 and 11 are the measured one-cell-miss realisations; the rest are ordinary.
SEED_SWEEP = (7, 11, 1, 15, 26)


def _run_search(seed: int, outdir: Path) -> dict:
    """Run the notebook's pipeline at one noise realisation."""
    cfg = PulseSignalConfig(
        period=PULSAR_PERIOD,
        dt=DT,
        nsamps=NSAMPS,
        snr=SNR,
        ducy=DUCY,
        mod_kwargs={"acc": ACCEL, "jerk": JERK},
        seed=seed,
    )
    tim_data = cfg.generate(shape="gaussian")

    limits = ParamLimits.from_upper(
        FREQ_RANGE, [JERK, ACCEL], JERK_SEARCH_RANGE, cfg.tobs
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
        prune_poly_order=3,
        ducy_max=0.5,
        wtsp=1.2,
        use_fourier=True,
        tiling_strategy="aggressive",
        branch_max=32,
    )
    dyp = DynamicProgramming(tim_data, search_cfg)
    dyp.initialize()
    dyp.execute()

    branching_pattern = np.asarray(
        search_cfg.generate_branching_pattern(
            kind="poly_taylor_moving", ref_seg=dyp.nsegments // 2
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
        file_prefix="test_jerk",
        poly_basis="taylor",
        n_workers=1,
        use_moving_grid=True,
    )
    pgram = ScatteredPeriodogram.load(result_file)
    return {
        "data": pgram.data,
        "true_freq": cfg.freq,
        "nsegments": dyp.nsegments,
        "thresholds": thresholds,
        "branching_pattern": branching_pattern,
    }


@pytest.fixture(scope="module")
def ep_jerk_search(tmp_path_factory) -> dict:
    """Run the pinned realisation once; the fast tests share it."""
    return _run_search(SEED, tmp_path_factory.mktemp("ep_jerk"))


def _best(result: dict) -> pd.Series:
    data = result["data"]
    return data.loc[data["score"].idxmax()]


# One source for each recovery predicate. The pinned tests and the seed sweep both
# call these, so a sweep cannot quietly assert something weaker than the fast path.


def _check_accel(result: dict, seed: object = SEED) -> None:
    best = _best(result)
    assert best["daccel"] < MAX_DACCEL_FRACTION * ACCEL, (
        f"acceleration is not actually constrained (seed={seed}): "
        f"daccel={best['daccel']:.3f} against accel={ACCEL:.1f} -- the recovery "
        f"check below would be vacuous"
    )
    tol = ACCEL_TOL_DACCEL * float(best["daccel"])
    error = abs(float(best["accel"]) - ACCEL)
    assert error < tol, (
        f"acceleration not recovered (seed={seed}): got {best['accel']:.5f}, "
        f"want {ACCEL:.1f} (error {error:.5f} > {tol:.5f} = "
        f"{ACCEL_TOL_DACCEL} x daccel {best['daccel']:.3f})"
    )


def _check_jerk(result: dict, seed: object = SEED) -> None:
    best = _best(result)
    assert best["djerk"] < MAX_DJERK_FRACTION * JERK, (
        f"jerk is not actually constrained (seed={seed}): "
        f"djerk={best['djerk']:.4f} against jerk={JERK} -- the recovery check "
        f"below would be vacuous"
    )
    error = abs(float(best["jerk"]) - JERK)
    assert error < JERK_TOL, (
        f"jerk not recovered (seed={seed}): got {best['jerk']:.4f}, want {JERK} "
        f"(error {error:.4f} > {JERK_TOL})"
    )


def _check_freq(result: dict, seed: object = SEED) -> None:
    best = _best(result)
    error = abs(float(best["freq"]) - result["true_freq"])
    assert error < FREQ_TOL, (
        f"frequency not recovered (seed={seed}): got {best['freq']:.10f}, "
        f"want {result['true_freq']:.10f} (error {error:.3e} > {FREQ_TOL:.3e})"
    )


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEED_SWEEP)
def test_recovery_does_not_depend_on_the_noise_realisation(
    seed: int,
    tmp_path: Path,
) -> None:
    """The recovery predicates must hold at every realisation, not just `SEED`.

    Fixed seeds, not random ones. A randomly drawn seed would reintroduce exactly
    the flakiness this file was fixed to remove: an unreproducible failure that
    cannot be re-examined. Each case here is deterministic and names its seed.

    **7 and 11 are in the list on purpose.** They are the two realisations found,
    in a 28-seed sweep, where the best candidate lands one acceleration cell off
    the truth -- the case that the old absolute `ACCEL_TOL = 1.0` failed. A sweep
    over easy seeds would be weaker evidence than the single pinned run; it is the
    known-hard ones that make it worth the runtime.

    What this does and does not establish: passing at every seed here bounds the
    per-realisation failure rate only loosely (0 of 5 is consistent with anything
    under ~45%). It establishes that recovery is not grossly realisation-dependent
    and that the known-hard cases pass. The measured rate is in
    `FINDINGS_rng_seeding.md`, from 60 runs, and that is the number to quote.
    """
    result = _run_search(seed, tmp_path)
    _check_accel(result, seed)
    _check_jerk(result, seed)
    _check_freq(result, seed)


def test_pipeline_produces_candidates(ep_jerk_search) -> None:
    data = ep_jerk_search["data"]
    assert len(data) > 0, "no candidates survived; the search found nothing at all"
    for column in ("jerk", "djerk", "accel", "daccel", "freq", "dfreq", "score"):
        assert column in data.columns, f"missing column {column!r}"


def test_stage_count_matches_configuration(ep_jerk_search) -> None:
    assert ep_jerk_search["nsegments"] == NSEGMENTS
    assert len(ep_jerk_search["thresholds"]) == len(
        ep_jerk_search["branching_pattern"]
    )


def test_recovers_injected_frequency(ep_jerk_search) -> None:
    _check_freq(ep_jerk_search)


def test_recovers_injected_acceleration(ep_jerk_search) -> None:
    _check_accel(ep_jerk_search)


def test_recovers_injected_jerk(ep_jerk_search) -> None:
    """Jerk recovery -- the parameter this example exists to search.

    Constraining it is what forced this test to `2**22` / 64 stages rather than the
    accel test's `2**21` / 8; see the module docstring.
    """
    _check_jerk(ep_jerk_search)


def test_best_candidate_clears_the_final_threshold(ep_jerk_search) -> None:
    data = ep_jerk_search["data"]
    best_score = float(data["score"].max())
    final_threshold = float(ep_jerk_search["thresholds"][-1])
    assert best_score > final_threshold, (
        f"best score {best_score:.3f} does not clear the final threshold "
        f"{final_threshold:.3f}"
    )
