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

**Measured stability** (8 independent noise realisations at this configuration): 8/8
recovered, `accel` error 4.56 m/s^2 every time with `daccel = 46.3`, and `freq` error at
most `0.67 * dfreq`. The tolerances below sit an order of magnitude above those figures.

Cost: ~14 s locally on a cold numba cache, ~2.5 s warm. CI runners have measured about
6x slower on this suite, so budget ~90 s there.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyloki.config import ParamLimits, PulsarSearchConfig
from pyloki.detection import thresholding
from pyloki.ffa import DynamicProgramming
from pyloki.periodogram import ScatteredPeriodogram
from pyloki.prune import prune_dyp_tree
from pyloki.simulation.pulse import PulseSignalConfig

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
N_RUNS = 1

# --- Tolerances, ~10x the measured spread ------------------------------------------
ACCEL_TOL = 50.0  # measured error 4.56, reported daccel 46.3
FREQ_TOL = 1e-3  # measured error <= 1.4e-4, reported dfreq 2.07e-4
# Guards so the assertions above cannot pass vacuously: the search must actually
# constrain the parameter, not just report a huge uncertainty that swallows any answer.
MAX_DACCEL_FRACTION = 0.10  # daccel must be < 10% of the injected accel
MAX_DFREQ_FRACTION = 1e-4  # dfreq must be < 1e-4 of the frequency


@pytest.fixture(scope="module")
def ep_accel_search(tmp_path_factory) -> dict:
    """Run the notebook's pipeline once and return the recovered candidates."""
    cfg = PulseSignalConfig(
        period=PULSAR_PERIOD,
        dt=DT,
        nsamps=NSAMPS,
        snr=SNR,
        ducy=DUCY,
        mod_kwargs={"acc": ACCEL},
    )
    tim_data = cfg.generate(shape="gaussian")

    limits = ParamLimits.from_upper(
        FREQ_RANGE, [ACCEL], ACCEL_SEARCH_RANGE, cfg.tobs
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
        prune_poly_order=2,
        ducy_max=0.5,
        wtsp=1.2,
        use_fourier=True,
        branch_max=32,
    )
    dyp = DynamicProgramming(tim_data, search_cfg)
    dyp.initialize()
    dyp.execute()

    branching_pattern = np.asarray(
        search_cfg.generate_branching_pattern(kind="poly_taylor_moving", ref_seg=0),
        dtype=np.float64,
    )
    # The notebook hardcodes 128 thresholds, which only fit its stage count. Deriving
    # them from the branching pattern with a constant survivor population per stage is
    # how examples/optimal_thresholds.ipynb builds its "constant" scheme, and it tracks
    # any change in the pattern.
    scheme = thresholding.determine_scheme(
        1.0 / branching_pattern,
        branching_pattern,
        ref_ducy=DUCY,
        nbins=NBINS,
        ntrials=1024,
        snr_final=SNR,
        ducy_max=0.5,
        wtsp=1.2,
    )
    thresholds = np.asarray(scheme.thresholds, dtype=np.float64)

    outdir = tmp_path_factory.mktemp("ep_accel")
    result_file = prune_dyp_tree(
        dyp,
        thresholds,
        n_runs=N_RUNS,
        max_sugg=MAX_SUGG,
        outdir=str(outdir),
        file_prefix="test_accel",
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


def test_pipeline_produces_candidates(ep_accel_search) -> None:
    """The search runs to completion and survives thresholding."""
    data = ep_accel_search["data"]
    assert len(data) > 0, "no candidates survived; the search found nothing at all"
    for column in ("accel", "daccel", "freq", "dfreq", "score"):
        assert column in data.columns, f"missing column {column!r}"


def test_stage_count_matches_configuration(ep_accel_search) -> None:
    """`nsegments` is driven by `bseg_ffa`, and the threshold ladder must match it."""
    assert ep_accel_search["nsegments"] == NSEGMENTS
    assert len(ep_accel_search["thresholds"]) == len(
        ep_accel_search["branching_pattern"]
    )


def test_recovers_injected_frequency(ep_accel_search) -> None:
    data, true_freq = ep_accel_search["data"], ep_accel_search["true_freq"]
    best = data.loc[data["score"].idxmax()]

    # Guard first: a huge reported uncertainty would make the recovery check vacuous.
    assert best["dfreq"] / true_freq < MAX_DFREQ_FRACTION, (
        f"frequency is not actually constrained: dfreq={best['dfreq']:.3e} "
        f"on freq={true_freq:.6f}"
    )
    error = abs(float(best["freq"]) - true_freq)
    assert error < FREQ_TOL, (
        f"frequency not recovered: got {best['freq']:.10f}, "
        f"want {true_freq:.10f} (error {error:.3e} > {FREQ_TOL:.3e})"
    )


def test_recovers_injected_acceleration(ep_accel_search) -> None:
    data = ep_accel_search["data"]
    best = data.loc[data["score"].idxmax()]

    # Guard: at smaller nsamps this reports daccel == accel, i.e. no constraint at all.
    assert best["daccel"] < MAX_DACCEL_FRACTION * ACCEL, (
        f"acceleration is not actually constrained: daccel={best['daccel']:.3f} "
        f"against accel={ACCEL:.1f} -- the recovery check below would be vacuous"
    )
    error = abs(float(best["accel"]) - ACCEL)
    assert error < ACCEL_TOL, (
        f"acceleration not recovered: got {best['accel']:.4f}, want {ACCEL:.1f} "
        f"(error {error:.3f} > {ACCEL_TOL:.1f})"
    )


def test_best_candidate_clears_the_final_threshold(ep_accel_search) -> None:
    """A recovered signal should score above the last stage's threshold."""
    data = ep_accel_search["data"]
    best_score = float(data["score"].max())
    final_threshold = float(ep_accel_search["thresholds"][-1])
    assert best_score > final_threshold, (
        f"best score {best_score:.3f} does not clear the final threshold "
        f"{final_threshold:.3f}"
    )
