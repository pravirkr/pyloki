"""Regression test: pyloki must read the pruning results LOKI writes.

LOKI's `level_stats` record has 13 fields (`create_compound_prune_stats` in
`lib/search/cands.cpp`, at LOKI main bcb6cfe and on the ep-sweep-seedfix line), and
its timer record lacks pyloki's `batch_add` and adds `rfi`. Before this fix
`PruningStatsPlotter.add_run` asserted a field count of 9 or 10, and
`PruneStatsCollection.from_arrays` indexed every pyloki timer by name, so both raised
on any current LOKI file. They now read by name.
"""

from __future__ import annotations

import h5py
import matplotlib as mpl
import numpy as np
import pytest

from pyloki.io.cands import PruneStats, PruneStatsCollection
from pyloki.periodogram import PruningStatsPlotter

mpl.use("Agg")

LOKI_LEVEL_FIELDS = [
    ("level", np.uint64),
    ("seg_idx", np.uint64),
    ("threshold", np.float32),
    ("score_min", np.float32),
    ("score_max", np.float32),
    ("n_branches", np.uint64),
    ("n_leaves", np.uint64),
    ("n_leaves_resolved", np.uint64),
    ("n_leaves_phy", np.uint64),
    ("n_leaves_surv", np.uint64),
    ("n_leaves_masked", np.uint64),
    ("n_leaves_vetoed", np.uint64),
    ("n_harvested", np.uint64),
]
LOKI_TIMER_FIELDS = [
    "branch",
    "validate",
    "resolve",
    "shift_add",
    "score",
    "transform",
    "threshold",
    "rfi",
]


def _loki_level_stats(n: int = 4) -> np.ndarray:
    ls = np.zeros(n, dtype=np.dtype(LOKI_LEVEL_FIELDS))
    ls["level"] = np.arange(1, n + 1)
    ls["threshold"] = np.linspace(1.0, 4.0, n)
    ls["n_leaves_surv"] = [100, 80, 60, 40][:n]
    return ls


def _loki_timer_stats() -> np.ndarray:
    dtype = np.dtype([(name, np.float32) for name in LOKI_TIMER_FIELDS])
    ts = np.zeros(1, dtype=dtype)
    ts[0]["score"] = 1.5
    ts[0]["rfi"] = 0.5
    return ts


def _pyloki_level_stats() -> np.ndarray:
    c = PruneStatsCollection()
    for level, surv in enumerate([160, 120, 80, 40], start=1):
        c.update_stats(
            PruneStats(
                level=level,
                seg_idx=level,
                threshold=1.0,
                threshold_eff=1.0,
                score_min=0.0,
                score_max=5.0,
                n_branches=2,
                n_leaves=2 * surv,
                n_leaves_phy=2 * surv,
                n_leaves_surv=surv,
            ),
        )
    return c.to_array()[0]


def test_plotter_accepts_loki_level_stats() -> None:
    plotter = PruningStatsPlotter()
    plotter.add_run(_loki_level_stats(), "loki")
    assert plotter.n_runs == 1
    assert "n_harvested" in plotter.get_level_stats("loki").columns


def test_plotter_still_rejects_records_without_required_fields() -> None:
    ls = _loki_level_stats()
    names = [n for n in ls.dtype.names if n != "n_leaves_surv"]
    with pytest.raises(ValueError, match="n_leaves_surv"):
        PruningStatsPlotter().add_run(ls[names].copy(), "broken")


def test_from_arrays_reads_loki_files_by_name() -> None:
    back = PruneStatsCollection.from_arrays(_loki_level_stats(), _loki_timer_stats())
    assert [s.n_leaves_surv for s in back.stats_list] == [100, 80, 60, 40]
    assert all(np.isnan(s.threshold_eff) for s in back.stats_list)
    assert back.timers["score"] == pytest.approx(1.5)
    assert back.timers["rfi"] == pytest.approx(0.5)
    assert back.timers["batch_add"] == 0.0  # LOKI has no such stage: it took no time


def test_timer_summaries_of_a_loki_collection_add_up() -> None:
    back = PruneStatsCollection.from_arrays(_loki_level_stats(), _loki_timer_stats())
    for summary in (back.get_timer_summary(), back.get_concise_timer_summary()):
        assert "nan" not in summary
    assert "Total: 2.0s" in back.get_concise_timer_summary()


def test_load_reads_pyloki_and_loki_runs_from_one_file(tmp_path) -> None:
    path = tmp_path / "mixed.h5"
    with h5py.File(path, "w") as f:
        f.attrs["pruning_version"] = "test"
        runs = f.create_group("runs")
        runs.create_group("pyloki")["level_stats"] = _pyloki_level_stats()
        runs.create_group("loki")["level_stats"] = _loki_level_stats()
        runs.create_group("empty")  # LOKI omits level_stats for an empty run
    plotter = PruningStatsPlotter.load(str(path))
    assert plotter.n_runs == 2
    surv = plotter.get_level_stats("pyloki")["n_leaves_surv"].tolist()
    assert surv == [160, 120, 80, 40]
    surv = plotter.get_level_stats("loki")["n_leaves_surv"].tolist()
    assert surv == [100, 80, 60, 40]
    fig = plotter.plot_level_stats()
    assert len(fig.axes[0].get_lines()) >= 2
