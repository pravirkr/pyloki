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

import h5py
import numpy as np
import pytest

from pyloki.config import ParamLimits, PulsarSearchConfig
from pyloki.ffa import DynamicProgramming
from pyloki.prune import Pruning, prune_dyp_tree
from pyloki.simulation.pulse import PulseSignalConfig

# Pinned so CI is reproducible. Before the library took a `seed`, the noise and the
# threshold ladder were drawn from unseeded generators inside pyloki and this file
# had no way to reach them; see tests/test_rng_seeding.py.
SEED = 42


@pytest.fixture(scope="module")
def small_search():
    nsamps, dt, period, nbins = 2**14, 64e-6, 0.007, 32
    freq = 1.0 / period
    cfg = PulseSignalConfig(
        period=period,
        dt=dt,
        nsamps=nsamps,
        snr=15.0,
        ducy=0.1,
        mod_kwargs={"acc": 500.0, "jerk": 6.0},
        seed=SEED,
    )
    tim_data = cfg.generate(shape="gaussian")
    tobs = nsamps * dt
    limits = ParamLimits.from_upper(
        (freq - 1, freq + 1), [6.0, 500.0], (-8.0, 8.0), tobs
    )
    search_cfg = PulsarSearchConfig(
        nsamps=nsamps,
        tsamp=dt,
        nbins=nbins,
        eta=1,
        param_limits=limits.limits,
        bseg_brute=nsamps // 8,
        bseg_ffa=nsamps // 2,
        prune_poly_order=3,
        ducy_max=0.5,
        wtsp=1.2,
        use_fourier=True,
        tiling_strategy="aggressive",
        branch_max=16,
    )
    dyp = DynamicProgramming(tim_data, search_cfg)
    dyp.initialize()
    dyp.execute()
    return dyp, search_cfg, nbins


def test_prune_dyp_tree_writes_its_run(small_search, tmp_path) -> None:
    dyp, search_cfg, _ = small_search
    ref_seg = dyp.nsegments // 2
    branching_pattern = search_cfg.generate_branching_pattern(
        kind="poly_taylor_moving", ref_seg=ref_seg
    )
    thresholds = np.linspace(1.5, 6.0, len(branching_pattern))
    result_file = prune_dyp_tree(
        dyp,
        thresholds,
        n_runs=1,
        max_sugg=2**12,
        outdir=tmp_path,
        file_prefix="smoke",
        poly_basis="taylor",
        n_workers=1,
        use_moving_grid=True,
    )
    with h5py.File(result_file, "r") as h5:
        assert h5.attrs["nsegments"] == dyp.nsegments
        assert h5.attrs["max_sugg"] == 2**12
        assert list(h5["runs"]) == [f"{ref_seg:03d}_00"]
        run = h5["runs"][f"{ref_seg:03d}_00"]
        level_stats = run["level_stats"][()]
        scores = run["scores"][()]
        param_sets = run["param_sets"][()]
    # One stats row per segment, and every segment is visited exactly once.
    assert len(level_stats) == dyp.nsegments
    assert sorted(level_stats["seg_idx"]) == list(range(dyp.nsegments))
    # The injected signal survives, and every survivor clears the final threshold.
    assert len(scores) == len(param_sets) > 0
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= thresholds[-1])


def test_pruning_execute_without_task_id(small_search, tmp_path) -> None:
    # Pruning.execute declares task_id: int | None = None, but formatted it with
    # :02d when naming the run, so calling it directly without one raised TypeError.
    dyp, search_cfg, _ = small_search
    ref_seg = dyp.nsegments // 2
    branching_pattern = search_cfg.generate_branching_pattern(
        kind="poly_taylor_moving", ref_seg=ref_seg
    )
    thresholds = np.linspace(1.5, 6.0, len(branching_pattern))
    Pruning(dyp, thresholds, max_sugg=2**12).execute(ref_seg, outdir=tmp_path)
    assert (tmp_path / f"tmp_{ref_seg:03d}_00_results.h5").exists()
