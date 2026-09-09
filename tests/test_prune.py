"""Smoke coverage for the pruning pipeline.

The suite previously exercised no part of ``prune_dyp_tree``, which is how a crash
that killed *every* pruning run went unnoticed: ``pruning_iteration_batched``
returned a dict mixing int counts with float scores, and numba cannot box a
heterogeneous ``LiteralStrKey`` dict back to Python -- it returns NULL with an
exception set, so the caller's tuple unpack dereferenced NULL and the process died
with SIGSEGV. A segfault cannot be caught in-process, so this test guards the path
by running it: if the boxing regresses, the test crashes the worker instead of
silently passing.
"""

import tempfile

import numpy as np
import pytest

from pyloki.config import ParamLimits, PulsarSearchConfig
from pyloki.detection import thresholding
from pyloki.ffa import DynamicProgramming
from pyloki.prune import prune_dyp_tree
from pyloki.simulation.pulse import PulseSignalConfig


@pytest.fixture(scope="module")
def small_search():
    nsamps, dt, period, nbins = 2**14, 64e-6, 0.007, 32
    freq = 1.0 / period
    cfg = PulseSignalConfig(
        period=period, dt=dt, nsamps=nsamps, snr=15.0, ducy=0.1,
        mod_kwargs={"acc": 500.0, "jerk": 6.0},
    )
    tim_data = cfg.generate(shape="gaussian")
    tobs = nsamps * dt
    limits = ParamLimits.from_upper(
        (freq - 1, freq + 1), [6.0, 500.0], (-8.0, 8.0), tobs
    )
    search_cfg = PulsarSearchConfig(
        nsamps=nsamps, tsamp=dt, nbins=nbins, eta=1, param_limits=limits.limits,
        bseg_brute=nsamps // 8, bseg_ffa=nsamps // 2, prune_poly_order=3,
        ducy_max=0.5, wtsp=1.2, use_fourier=True,
        tiling_strategy="aggressive", branch_max=16,
    )
    dyp = DynamicProgramming(tim_data, search_cfg)
    dyp.initialize()
    dyp.execute()
    return dyp, search_cfg, nbins


def test_prune_dyp_tree_completes(small_search) -> None:
    dyp, search_cfg, nbins = small_search
    branching_pattern = search_cfg.generate_branching_pattern(
        kind="poly_taylor_moving", ref_seg=dyp.nsegments // 2
    )
    thresholds = np.linspace(1.5, 6.0, len(branching_pattern))
    thresholding.evaluate_scheme(
        thresholds, branching_pattern, ref_ducy=0.1, nbins=nbins,
        ntrials=256, snr_final=9.0, ducy_max=0.5, wtsp=1.2,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = prune_dyp_tree(
            dyp, thresholds, n_runs=1, max_sugg=2**12, outdir=tmpdir,
            file_prefix="smoke", poly_basis="taylor", n_workers=1,
            use_moving_grid=True,
        )
        assert result_file is not None
