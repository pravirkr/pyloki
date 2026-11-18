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

**Measured stability.** The `accel` error is **bimodal**: over 28 seeded realisations,
26 landed in the right acceleration cell (error ~0.008) and 2 exactly one `daccel` step
away (8.28); nothing in between. That is why recovery is checked against the reported
uncertainty (shared `k = 2` rule, `tests/_pipelines.py`) and not an absolute number --
an absolute `ACCEL_TOL = 1.0` demanded the exact cell and was flaky. Measured
error/d<param> over the pinned seed and `SEED_SWEEP`: `jerk` 0.40-0.60, `accel`
0.0007-0.0010 or 1.00 (the one-cell misses, seeds 7 and 11).

The exception is frequency: its error *exceeds* the reported `dfreq` by 1.8-4.2x here
(unlike the accel example), so the reported frequency uncertainty is optimistic at this
configuration and the frequency check uses an absolute tolerance instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

# --- Recovery: the shared `check_recovered` rule, |err| <= 2 x d<param> -------------
# Non-vacuity guards: each uncertainty must be small relative to the injected value.
MAX_DJERK_FRACTION = 0.50  # djerk 0.556 is 28% of the injected jerk
MAX_DACCEL_FRACTION = 0.10  # daccel 8.29 is 1.7% of accel
MAX_DFREQ_FRACTION = 1e-4  # dfreq 9.2e-5 is 6.4e-7 of freq
# Exception: dfreq is optimistic here (err/dfreq measured 1.8-4.2), so freq gets an
# absolute tolerance; measured error <= 3.9e-4.
FREQ_TOL = 2e-3

SEED = 42  # Pinned so CI is reproducible.

# The sweep. Fixed and enumerated, never random -- a random seed would make a
# failure unreproducible, which is the defect this file exists to not have.
# 7 and 11 are the measured one-cell-miss realisations; the rest are ordinary.
SEED_SWEEP = (7, 11, 1, 15, 26)

SEARCH = EpSearch(
    nsegments=NSEGMENTS,
    prune_poly_order=3,
    kind="poly_taylor_moving",
    max_sugg=MAX_SUGG,
    ffa_levels=FFA_LEVELS,
    nbins=NBINS,
    eta=ETA,
    config_kwargs={"tiling_strategy": "aggressive"},
)


def _run_search(seed: int, outdir: Path) -> dict:
    """Run the notebook's pipeline at one noise realisation."""
    signal = PulseSignalConfig(
        period=PULSAR_PERIOD,
        dt=DT,
        nsamps=NSAMPS,
        snr=SNR,
        ducy=DUCY,
        mod_kwargs={"acc": ACCEL, "jerk": JERK},
        seed=seed,
    )
    limits = ParamLimits.from_upper(
        FREQ_RANGE, [JERK, ACCEL], JERK_SEARCH_RANGE, signal.tobs
    )
    return {
        **run_ep(signal, limits.limits, SEARCH, seed, outdir),
        "true_freq": signal.freq,
    }


@pytest.fixture(scope="module")
def ep_jerk_search(tmp_path_factory) -> dict:
    """Run the pinned realisation once; the fast tests share it."""
    return _run_search(SEED, tmp_path_factory.mktemp("ep_jerk"))


# One source for the recovery predicates, shared by the pinned tests and the sweep.


def _check_accel(result: dict, seed: object = SEED) -> None:
    check_recovered(result, "accel", ACCEL, MAX_DACCEL_FRACTION, seed)


def _check_jerk(result: dict, seed: object = SEED) -> None:
    check_recovered(result, "jerk", JERK, MAX_DJERK_FRACTION, seed)


def _check_freq(result: dict, seed: object = SEED) -> None:
    check_recovered(
        result, "freq", result["true_freq"], MAX_DFREQ_FRACTION, seed, abs_tol=FREQ_TOL
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
    the truth -- the case that an absolute `ACCEL_TOL = 1.0` once failed. A sweep
    over easy seeds would be weaker evidence than the single pinned run; it is the
    known-hard ones that make it worth the runtime.

    What this does and does not establish: passing at every seed here bounds the
    per-realisation failure rate only loosely (0 of 5 is consistent with anything
    under ~45%). It establishes that recovery is not grossly realisation-dependent
    and that the known-hard cases pass.
    """
    result = _run_search(seed, tmp_path)
    _check_accel(result, seed)
    _check_jerk(result, seed)
    _check_freq(result, seed)
    check_detection_strength(result, seed)


def test_pipeline_produces_candidates(ep_jerk_search) -> None:
    check_candidates(
        ep_jerk_search, ["jerk", "djerk", "accel", "daccel", "freq", "dfreq"]
    )


def test_stage_count_matches_configuration(ep_jerk_search) -> None:
    check_stage_count(ep_jerk_search, NSEGMENTS)


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


def test_best_candidate_carries_the_injected_snr(ep_jerk_search) -> None:
    check_detection_strength(ep_jerk_search, SEED)
