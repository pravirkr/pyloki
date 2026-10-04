"""A parallel ``prune_dyp_tree`` must merge only the temporary files it wrote.

With ``n_workers > 1`` each worker writes a temporary result file into ``outdir``, and the
call then merges them into its result file. ``merge_prune_result_files`` globbed every
``tmp_*`` file in the directory, so a second search sharing ``outdir`` (running
concurrently, or interrupted earlier) had its passes merged into this search's result and
deleted. The test plants such a file deterministically instead of racing two searches.
"""

import h5py
import numpy as np
import pytest

from pyloki.config import ParamLimits, PulsarSearchConfig
from pyloki.ffa import DynamicProgramming
from pyloki.prune import Pruning, prune_dyp_tree
from pyloki.simulation.pulse import PulseSignalConfig
from pyloki.utils.np_utils import determine_ref_segs_pareto

SEED = 42
N_RUNS = 2


@pytest.fixture(scope="module")
def eight_segment_search():
    nsamps, dt, period, nbins = 2**14, 64e-6, 0.007, 32
    freq = 1.0 / period
    cfg = PulseSignalConfig(period=period, dt=dt, nsamps=nsamps, snr=15.0, ducy=0.1,
                            mod_kwargs={"acc": 500.0, "jerk": 6.0}, seed=SEED)
    tim_data = cfg.generate(shape="gaussian")
    limits = ParamLimits.from_upper((freq - 1, freq + 1), [6.0, 500.0], (-8.0, 8.0), nsamps * dt)
    search_cfg = PulsarSearchConfig(
        nsamps=nsamps, tsamp=dt, nbins=nbins, eta=1, param_limits=limits.limits,
        bseg_brute=nsamps // 64, bseg_ffa=nsamps // 8, prune_poly_order=3,
        ducy_max=0.5, wtsp=1.2, use_fourier=True, tiling_strategy="aggressive", branch_max=16,
    )
    dyp = DynamicProgramming(tim_data, search_cfg)
    dyp.initialize()
    dyp.execute()
    return dyp


def test_parallel_prune_merges_only_its_own_runs(eight_segment_search, tmp_path) -> None:
    dyp = eight_segment_search
    thresholds = np.linspace(1.5, 6.0, dyp.nsegments - 1)
    want = determine_ref_segs_pareto(dyp.nsegments, N_RUNS)
    stray_seg = next(s for s in range(dyp.nsegments) if s not in want)

    # A temporary result another search left in the same directory.
    stray_h5 = tmp_path / f"tmp_{stray_seg:03d}_07_results.h5"
    stray_log = tmp_path / f"tmp_{stray_seg:03d}_07_log.txt"
    Pruning(dyp, thresholds, max_sugg=2**12).execute(
        stray_seg, outdir=tmp_path, log_file=stray_log, result_file=stray_h5, task_id=7)
    assert stray_h5.exists()

    result = prune_dyp_tree(dyp, thresholds, n_runs=N_RUNS, max_sugg=2**12, outdir=tmp_path,
                            file_prefix="mine", poly_basis="taylor", n_workers=2)

    with h5py.File(result, "r") as h:
        segs = sorted(int(name.split("_")[0]) for name in h["runs"])
    assert segs == sorted(want)          # exactly its own passes, and not the stray one
    assert stray_h5.exists()             # and the other search's file is left alone
    assert not list(tmp_path.glob("tmp_mine_*"))   # its own temporaries are cleaned up
