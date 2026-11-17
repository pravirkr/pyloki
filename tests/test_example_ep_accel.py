"""CI regression test converted from `examples/pyloki_ep_accel.ipynb`.

The notebook demonstrates the Extreme Pruning search recovering an injected
constant-acceleration pulsar. This runs the same pipeline end to end -- simulate,
`DynamicProgramming` FFA, branching pattern, threshold scheme, `prune_dyp_tree`,
`ScatteredPeriodogram` -- and asserts the injected `accel` and `freq` come back.

It is a **scaled-down** conversion, because the notebook's configuration is far too
large for CI. Deviations, all deliberate:

===================  ====================  ======================================
notebook             here                  why
===================  ====================  ======================================
`nsamps = 2**25`     `2**21`               36 min of data -> 2.2 min. The
                     (tobs 2147s -> 134s)  *smallest* size at which `accel` is
                                           still constrained; see below.
128 pruning stages   8                     `bseg_ffa = nsamps // 8`.
`n_runs = 32`        1                     One realisation suffices.
`max_sugg = 2**21`   `2**16`               Peak survivor count is ~3e4.
`n_workers = 4`      1                     Determinism; CI runners are small.
hardcoded 128        derived by            The fixed array is the wrong length
thresholds           `determine_scheme`    once the stage count changes; deriving
                                           from the pattern tracks it.
`snr = 8.5`          20                    Near-threshold recovery is a coin
                                           flip; a CI test must not be.
plots                dropped               Nothing to assert.
===================  ====================  ======================================

**Why not smaller.** At `2**18` the pipeline still runs and frequency is recovered, but
the acceleration is *unconstrained*: the reported `daccel` comes back equal to the
`accel` value itself, and the recovered `accel` lands 1500 m/s^2 from the injection
(i.e. anywhere) between realisations. Asserting recovery there would pass vacuously.
The accel grid step scales as `1 / tseg_ffa**2`, so `tobs` has to be large enough for
the accel axis to resolve at all. `2**21` is the first power of two that does.

**Measured stability** (pinned seed and `SEED_SWEEP`): error over the reported
uncertainty is 0.099 for `accel` every time and 0.33-0.67 for `freq`, both well inside
the shared `k = 2` rule (`tests/_pipelines.py`). Each run is a few seconds once numba
has compiled.
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


SEED = 42  # Pinned so CI is reproducible.

# Measured over 40 seeded realisations: this example produced exactly TWO distinct
# outcomes, differing only in frequency (error/dfreq 0.33 or 0.67), with the
# acceleration error identical every time. A one-cell acceleration miss would still
# pass, because the tolerance is 2 x daccel.
#
# So unlike ep_jerk, this example has no hard realisation to pin -- the sweep below
# is regression insurance, not coverage of a measured risk. 4 and 16 are from the
# worse of the two outcomes, 1 from the better.
SEED_SWEEP = (4, 16, 1)

# --- The notebook's physical setup, unchanged -------------------------------------
PULSAR_PERIOD = 0.007
DT = 64e-6
ACCEL = 1500.0
NBINS = 64
ETA = 1
DUCY = 0.1
FREQ_RANGE = (140.0, 150.0)
ACCEL_SEARCH_RANGE = (-2500.0, 2500.0)

# --- Scaled for CI (see the module docstring) --------------------------------------
NSAMPS = 2**21
NSEGMENTS = 8
FFA_LEVELS = 16  # bseg_ffa // bseg_brute
SNR = 20.0
MAX_SUGG = 2**16

# --- Non-vacuity guards for the shared `check_recovered` rule ----------------------
# The search must actually constrain each parameter, not report an uncertainty that
# swallows any answer.
MAX_DACCEL_FRACTION = 0.10  # daccel must be < 10% of the injected accel (it is 3%)
MAX_DFREQ_FRACTION = 1e-4  # dfreq must be < 1e-4 of the frequency (it is 1.4e-6)

SEARCH = EpSearch(
    nsegments=NSEGMENTS,
    prune_poly_order=2,
    kind="poly_taylor_moving",
    max_sugg=MAX_SUGG,
    ffa_levels=FFA_LEVELS,
    nbins=NBINS,
    eta=ETA,
)


def _run_search(seed: int, outdir: Path) -> dict:
    """Run the notebook's pipeline at one noise realisation."""
    signal = PulseSignalConfig(
        period=PULSAR_PERIOD,
        dt=DT,
        nsamps=NSAMPS,
        snr=SNR,
        ducy=DUCY,
        mod_kwargs={"acc": ACCEL},
        seed=seed,
    )
    limits = ParamLimits.from_upper(
        FREQ_RANGE, [ACCEL], ACCEL_SEARCH_RANGE, signal.tobs
    )
    return {
        **run_ep(signal, limits.limits, SEARCH, seed, outdir),
        "true_freq": signal.freq,
    }


@pytest.fixture(scope="module")
def ep_accel_search(tmp_path_factory) -> dict:
    """Run the pinned realisation once; the fast tests share it."""
    return _run_search(SEED, tmp_path_factory.mktemp("ep_accel"))


# One source for the recovery predicates, shared by the pinned tests and the sweep.


def _check_freq(result: dict, seed: object = SEED) -> None:
    check_recovered(result, "freq", result["true_freq"], MAX_DFREQ_FRACTION, seed)


def _check_accel(result: dict, seed: object = SEED) -> None:
    check_recovered(result, "accel", ACCEL, MAX_DACCEL_FRACTION, seed)


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEED_SWEEP)
def test_recovery_does_not_depend_on_the_noise_realisation(
    seed: int,
    tmp_path: Path,
) -> None:
    """The recovery predicates must hold at every realisation, not just `SEED`.

    Fixed seeds, never random: a random seed makes a failure unreproducible, which
    is the defect this suite was fixed to not have.

    This example is genuinely realisation-insensitive -- see `SEED_SWEEP` for the
    measurement -- so unlike the `ep_jerk` sweep, this one is not exercising a
    known-hard case. It is here to catch a future change that *makes* the result
    realisation-dependent, which the pinned single seed could not.
    """
    result = _run_search(seed, tmp_path)
    _check_freq(result, seed)
    _check_accel(result, seed)
    check_detection_strength(result, seed)


def test_pipeline_produces_candidates(ep_accel_search) -> None:
    check_candidates(ep_accel_search, ["accel", "daccel", "freq", "dfreq"])


def test_stage_count_matches_configuration(ep_accel_search) -> None:
    check_stage_count(ep_accel_search, NSEGMENTS)


def test_recovers_injected_frequency(ep_accel_search) -> None:
    _check_freq(ep_accel_search)


def test_recovers_injected_acceleration(ep_accel_search) -> None:
    _check_accel(ep_accel_search)


def test_best_candidate_carries_the_injected_snr(ep_accel_search) -> None:
    check_detection_strength(ep_accel_search, SEED)
