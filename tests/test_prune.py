"""Integration tests for the pruning pipeline, `pyloki.prune`.

- `prune_dyp_tree` runs a pruning pass to completion and writes it to its result
  file. The pass returns a dict of int counts and float scores out of a numba kernel;
  a heterogeneous ``LiteralStrKey`` dict cannot be boxed back to Python and crashes
  the process with SIGSEGV, which cannot be caught in-process. So the path is guarded
  by running it: if the boxing regresses, this module crashes its worker.
- `Pruning.execute` works without the optional ``task_id``.
- With ``n_workers > 1``, `prune_dyp_tree` merges exactly the temporary files its own
  workers wrote, and leaves another search's temporaries in a shared ``outdir`` alone.
  The merge itself is tested on synthetic files in ``test_cands.py``.

All three share one FFA run over eight segments, so the prune kernels compile once.
"""

import h5py
import numpy as np
import pytest

from pyloki.config import ParamLimits, PulsarSearchConfig
from pyloki.ffa import DynamicProgramming
from pyloki.io.cands import PruneResultWriter
from pyloki.prune import Pruning, prune_dyp_tree
from pyloki.simulation.pulse import PulseSignalConfig
from pyloki.utils.np_utils import determine_ref_segs_pareto

# Seeds the simulated noise, so the survivors asserted on below are reproducible.
SEED = 42
MAX_SUGG = 2**12


@pytest.fixture(scope="module")
def search() -> DynamicProgramming:
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
    limits = ParamLimits.from_upper(
        (freq - 1, freq + 1), [6.0, 500.0], (-8.0, 8.0), nsamps * dt
    )
    search_cfg = PulsarSearchConfig(
        nsamps=nsamps,
        tsamp=dt,
        nbins=nbins,
        eta=1,
        param_limits=limits.limits,
        bseg_brute=nsamps // 64,
        bseg_ffa=nsamps // 8,
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
    # Two parallel runs must leave a segment free for another search's stray file.
    assert dyp.nsegments == 8
    return dyp


def _thresholds(dyp: DynamicProgramming) -> np.ndarray:
    ref_seg = dyp.nsegments // 2
    branching_pattern = dyp.cfg.generate_branching_pattern(
        kind="poly_taylor_moving", ref_seg=ref_seg
    )
    return np.linspace(1.5, 6.0, len(branching_pattern))


def test_prune_dyp_tree_writes_its_run(search, tmp_path) -> None:
    dyp = search
    ref_seg = dyp.nsegments // 2  # n_runs=1 anchors on the centre segment
    thresholds = _thresholds(dyp)
    result_file = prune_dyp_tree(
        dyp,
        thresholds,
        n_runs=1,
        max_sugg=MAX_SUGG,
        outdir=tmp_path,
        file_prefix="smoke",
        poly_basis="taylor",
        n_workers=1,
        use_moving_grid=True,
    )
    with h5py.File(result_file, "r") as h5:
        assert h5.attrs["nsegments"] == dyp.nsegments
        assert h5.attrs["max_sugg"] == MAX_SUGG
        assert list(h5["runs"]) == [f"{ref_seg:03d}_00"]
        run = h5["runs"][f"{ref_seg:03d}_00"]
        level_stats = run["level_stats"][()]
        scores = run["scores"][()]
        param_sets = run["param_sets"][()]
    # One stats row per segment, and every segment is visited exactly once.
    assert len(level_stats) == dyp.nsegments
    assert sorted(level_stats["seg_idx"]) == list(range(dyp.nsegments))
    # Level k is cut at thresholds[k - 1] (level 0 is the seed), and leaves survive
    # every cut.
    by_level = np.sort(level_stats, order="level")
    np.testing.assert_allclose(by_level["threshold"][1:], thresholds, rtol=1e-6)
    assert np.all(by_level["n_leaves_surv"] > 0)
    # The injected signal survives. The final ascend re-scores every survivor, so a
    # written score may fall below the last cut; the best one clears it.
    assert len(scores) == len(param_sets) > 0
    assert np.all(np.isfinite(scores))
    assert scores.max() >= thresholds[-1]


def test_pruning_execute_without_task_id(search, tmp_path) -> None:
    """`task_id` is optional; without one the run is named with task 0."""
    dyp = search
    ref_seg = dyp.nsegments // 2
    Pruning(dyp, _thresholds(dyp), max_sugg=MAX_SUGG).execute(ref_seg, outdir=tmp_path)
    assert (tmp_path / f"tmp_{ref_seg:03d}_00_results.h5").exists()


def test_parallel_prune_merges_only_its_own_runs(search, tmp_path) -> None:
    dyp = search
    want = determine_ref_segs_pareto(dyp.nsegments, 2)
    stray_seg = next(s for s in range(dyp.nsegments) if s not in want)

    # A temporary result another search left in the same directory.
    stray_h5 = tmp_path / f"tmp_{stray_seg:03d}_07_results.h5"
    stray_log = tmp_path / f"tmp_{stray_seg:03d}_07_log.txt"
    with PruneResultWriter(stray_h5) as writer:
        writer.runs_group.create_group(f"{stray_seg:03d}_07")
    stray_log.write_text("another search\n")

    result = prune_dyp_tree(
        dyp,
        _thresholds(dyp),
        n_runs=2,
        max_sugg=MAX_SUGG,
        outdir=tmp_path,
        file_prefix="mine",
        poly_basis="taylor",
        n_workers=2,
    )

    with h5py.File(result, "r") as h:
        segs = sorted(int(name.split("_")[0]) for name in h["runs"])
    assert segs == sorted(want)  # exactly its own passes, and not the stray one
    assert stray_h5.exists()  # and the other search's files are left alone
    assert stray_log.exists()
    assert not list(tmp_path.glob("tmp_mine_*"))  # its own temporaries are cleaned up
