"""Tests for `pyloki.io.cands`: reading pruning result files and merging them.

- pyloki reads the results LOKI writes. LOKI's `level_stats` record has 13 fields
  (`create_compound_prune_stats` in `lib/search/cands.cpp`, at LOKI main bcb6cfe and
  on the ep-sweep-seedfix line), and its timer record lacks pyloki's `batch_add` and
  adds `rfi`. `PruningStatsPlotter.add_run` and `PruneStatsCollection.from_arrays`
  read both records by field name, not by field count or position.
- `merge_prune_result_files` merges a parallel search's per-worker temporaries into
  its result file: only the listed ones when given a list, otherwise every ``tmp_*``
  file in ref_seg order. Missing files are skipped, the first run of a name wins, and
  every merged temporary is deleted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import matplotlib as mpl
import numpy as np
import pytest

from pyloki.io.cands import (
    PruneStats,
    PruneStatsCollection,
    merge_prune_result_files,
)
from pyloki.periodogram import PruningStatsPlotter

if TYPE_CHECKING:
    from pathlib import Path

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


# ------------------------------------------------- merge_prune_result_files


def _temp_pair(
    directory: Path, ref_seg: int, task: int, *, run: str | None = None, value: int = 0
) -> tuple[Path, Path]:
    """Write a worker's (log, result) temporaries, holding one run."""
    log = directory / f"tmp_{ref_seg:03d}_{task:02d}_log.txt"
    h5 = directory / f"tmp_{ref_seg:03d}_{task:02d}_results.h5"
    log.write_text(f"log {ref_seg}\n")
    run = run or f"{ref_seg:03d}_{task:02d}"
    with h5py.File(h5, "w") as f:
        f.create_group("runs").create_group(run)["value"] = value
    return log, h5


def _merged(result: Path) -> dict[str, int]:
    with h5py.File(result, "r") as f:
        return {name: int(f["runs"][name]["value"][()]) for name in f["runs"]}


@pytest.fixture
def merge_targets(tmp_path) -> tuple[Path, Path]:
    log = tmp_path / "main_log.txt"
    log.write_text("Pruning log\n")
    return log, tmp_path / "main_results.h5"


def test_merge_takes_only_the_listed_files(tmp_path, merge_targets) -> None:
    log, result = merge_targets
    listed = [_temp_pair(tmp_path, 5, 1), _temp_pair(tmp_path, 2, 0)]
    stray_log, stray_h5 = _temp_pair(tmp_path, 3, 7)  # another search's
    merge_prune_result_files(tmp_path, log, result, temp_files=listed)
    assert set(_merged(result)) == {"005_01", "002_00"}
    # Listed order, not ref_seg order, when the caller passes the list.
    assert log.read_text() == "Pruning log\nlog 5\nlog 2\n"
    assert stray_log.exists()
    assert stray_h5.exists()
    for path in (p for pair in listed for p in pair):
        assert not path.exists()


def test_merge_skips_missing_files(tmp_path, merge_targets) -> None:
    """A worker that failed may have written only one temporary, or none."""
    log, result = merge_targets
    no_log = _temp_pair(tmp_path, 1, 0)
    no_log[0].unlink()
    no_h5 = _temp_pair(tmp_path, 2, 0)
    no_h5[1].unlink()
    neither = (tmp_path / "tmp_004_00_log.txt", tmp_path / "tmp_004_00_results.h5")
    merge_prune_result_files(tmp_path, log, result, temp_files=[no_log, no_h5, neither])
    assert set(_merged(result)) == {"001_00"}
    assert log.read_text() == "Pruning log\nlog 2\n"
    assert not list(tmp_path.glob("tmp_*"))


def test_merge_keeps_the_first_run_of_a_name(tmp_path, merge_targets) -> None:
    log, result = merge_targets
    first = _temp_pair(tmp_path, 1, 0, run="001_00", value=1)
    second = _temp_pair(tmp_path, 1, 1, run="001_00", value=2)
    merge_prune_result_files(tmp_path, log, result, temp_files=[first, second])
    assert _merged(result) == {"001_00": 1}
    assert not second[1].exists()  # the duplicate's file is still consumed


def test_merge_without_a_list_globs_in_ref_seg_order(tmp_path, merge_targets) -> None:
    log, result = merge_targets
    # Written out of order, so directory order cannot pass for ref_seg order.
    for ref_seg in (7, 0, 12, 3):
        _temp_pair(tmp_path, ref_seg, 0, run="dup", value=ref_seg)
    merge_prune_result_files(tmp_path, log, result)
    assert log.read_text() == "Pruning log\nlog 0\nlog 3\nlog 7\nlog 12\n"
    assert _merged(result) == {"dup": 0}  # the lowest ref_seg was merged first
    assert not list(tmp_path.glob("tmp_*"))
