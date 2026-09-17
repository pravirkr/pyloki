"""Regression test: the `max_sugg` overflow ratchet must be visible in the log.

When the candidate buffer overflows, pruning raises the effective cut to
`max(threshold, top-K, median)` and never lowers it again for that level
(`utils/world_tree.py:prune_on_overload_func`). Before `threshold_eff` was recorded,
`PruneStats` logged only the nominal scheme value, so a run whose thresholds had been
silently tightened was indistinguishable from one that ran at the scheme's thresholds.

`score_min` does not substitute for it: that is taken over every scored leaf before
thresholding, not over the survivors.

This runs the scaled-down Extreme-Pruning pipeline of `test_example_ep_accel.py` on one
time series at two buffers and asserts the log distinguishes them.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyloki.config import ParamLimits, PulsarSearchConfig
from pyloki.detection import thresholding
from pyloki.ffa import DynamicProgramming
from pyloki.prune import prune_dyp_tree
from pyloki.simulation.pulse import PulseSignalConfig

NSAMPS, NSEGMENTS, FFA_LEVELS, NBINS, SNR, DUCY = 2**21, 8, 16, 64, 20.0, 0.1

LINE = re.compile(r"score thresh:\s*([-\d.]+), eff:\s*([-\d.naN]+)")


def build():
    cfg = PulseSignalConfig(period=0.007, dt=64e-6, nsamps=NSAMPS, snr=SNR,
                            ducy=DUCY, mod_kwargs={"acc": 1500.0})
    tim = cfg.generate(shape="gaussian")
    limits = ParamLimits.from_upper((140.0, 150.0), [1500.0], (-2500.0, 2500.0),
                                    cfg.tobs)
    bseg_ffa = NSAMPS // NSEGMENTS
    scfg = PulsarSearchConfig(
        nsamps=cfg.nsamps, tsamp=cfg.dt, nbins=NBINS, eta=1,
        param_limits=limits.limits, bseg_brute=bseg_ffa // FFA_LEVELS,
        bseg_ffa=bseg_ffa, prune_poly_order=2, ducy_max=0.5, wtsp=1.2,
        use_fourier=True, branch_max=32)
    dyp = DynamicProgramming(tim, scfg)
    dyp.initialize()
    dyp.execute()
    bp = np.asarray(scfg.generate_branching_pattern(kind="poly_taylor_moving",
                                                    ref_seg=0), dtype=np.float64)
    scheme = thresholding.determine_scheme(1.0 / bp, bp, ref_ducy=DUCY, nbins=NBINS,
                                           ntrials=1024, snr_final=SNR,
                                           ducy_max=0.5, wtsp=1.2)
    return dyp, np.asarray(scheme.thresholds, dtype=np.float64)


def run(dyp, thresholds, max_sugg: int) -> list[tuple[float, float]]:
    with tempfile.TemporaryDirectory() as td:
        prune_dyp_tree(dyp, thresholds, n_runs=1, max_sugg=max_sugg, outdir=td,
                       file_prefix="v", poly_basis="taylor", n_workers=1,
                       use_moving_grid=True)
        logs = list(Path(td).rglob("*_log.txt"))
        if not logs:
            raise SystemExit(f"no log written in {td}")
        text = max(logs, key=lambda p: p.stat().st_size).read_text()
    return [(float(a), float(b)) for a, b in LINE.findall(text)
            if b.lower() != "nan"]


@pytest.fixture(scope="module")
def pipeline():
    return build()


def test_threshold_eff_matches_nominal_when_buffer_does_not_bind(pipeline):
    dyp, thresholds = pipeline
    pairs = run(dyp, thresholds, 2**20)
    assert pairs, "no levels logged an effective threshold"
    assert all(e <= t + 1e-9 for t, e in pairs), (
        f"buffer should not bind at 2^20, but got {[(t, e) for t, e in pairs if e > t]}"
    )


def test_threshold_eff_exceeds_nominal_when_buffer_binds(pipeline):
    dyp, thresholds = pipeline
    pairs = run(dyp, thresholds, 2**10)
    ratcheted = [(t, e) for t, e in pairs if e > t + 1e-9]
    assert ratcheted, (
        "at max_sugg = 2^10 the buffer must bind and the log must show it; "
        "this is exactly the tightening that used to be silent"
    )
