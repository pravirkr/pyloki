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
    SECONDS_PER_DAY,
    CandidateClusterOptions,
    CandidateCollection,
    HarmonicMatcher,
    KnownPeriodEntry,
    PhaseModelKind,
    PruneResultWriter,
    PruneStats,
    PruneStatsCollection,
    SurveyCandidateGroups,
    bin2d,
    build_phase_model,
    cluster_candidates_single_dm,
    dominance_ratio,
    evaluate_tempo2_predictor,
    export_segment_ephemeris,
    export_tempo2_predictor,
    extract_ref_seg,
    extract_segment_waterfall,
    match_known_harmonics,
    merge_prune_result_files,
    phase_span,
    phase_track_distance,
    plot_candidate_diagnostic_card,
    reduce_dominated_candidates,
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


def test_zero_leaf_and_zero_timer_summaries() -> None:
    empty_stats = PruneStats(
        level=1,
        seg_idx=0,
        threshold=2.0,
        threshold_eff=2.0,
        n_branches=0,
        n_leaves=0,
        n_leaves_phy=0,
        n_leaves_surv=0,
    )
    assert empty_stats.lb_leaves == 0.0
    assert empty_stats.lb_leaves_phys == 0.0
    assert "inf" not in empty_stats.get_summary().lower()

    coll = PruneStatsCollection()
    assert coll.get_stats_summary() == "Score: 0.00, Leaves: 0.00"
    coll.update_stats(empty_stats)
    timer_summary = coll.get_timer_summary()
    assert "Timing breakdown: 0.00s" in timer_summary


def test_extract_ref_seg_and_merge_prefixed_temporaries(
    tmp_path: Path, merge_targets: tuple[Path, Path]
) -> None:
    assert extract_ref_seg(tmp_path / "tmp_003_01_log.txt") == 3
    assert extract_ref_seg(tmp_path / "tmp_mine_nstages_8_012_01_results.h5") == 12
    assert extract_ref_seg(tmp_path / "tmp_foo_1024_12_log.txt") == 1024

    log, result = merge_targets
    for ref_seg in (12, 3):
        h5 = tmp_path / f"tmp_mine_pruning_nstages_16_{ref_seg:03d}_01_results.h5"
        lg = tmp_path / f"tmp_mine_pruning_nstages_16_{ref_seg:03d}_01_log.txt"
        lg.write_text(f"log {ref_seg}\n")
        with h5py.File(h5, "w") as f:
            f.create_group("runs").create_group("dup")["value"] = ref_seg
    merge_prune_result_files(tmp_path, log, result)
    assert log.read_text() == "Pruning log\nlog 3\nlog 12\n"
    assert _merged(result) == {"dup": 3}


# ---------------------------------------------------------------------------
# Stage 5: Catalogs, CandidateCollection, Sifting, Harmonics & Predictors
# ---------------------------------------------------------------------------


def _write_schema_v1_h5(
    path: Path,
    *,
    dm: float,
    freqs: np.ndarray,
    accels: np.ndarray,
    snrs: np.ndarray,
) -> None:
    c_val = 299792458.0
    n = freqs.size
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "1.0"
        f.attrs["complete"] = 1
        f.attrs["param_names"] = ["accel", "freq"]
        obs = f.create_group("observation")
        obs.attrs["dm"] = dm
        obs.attrs["reference_mjd"] = 60000.5
        obs.attrs["barycentric"] = True
        grp = f.create_group("candidates")
        grp.create_dataset("id", data=np.arange(n, dtype=np.uint64))
        grp.create_dataset("cluster_id", data=np.arange(n, dtype=np.int64))
        grp.create_dataset("param_sets", data=np.column_stack([accels, freqs]))
        grp.create_dataset("snr", data=snrs.astype(np.float64))
        grp.create_dataset("width", data=np.full(n, 4, dtype=np.int64))
        grp.create_dataset("nbins", data=np.full(n, 64, dtype=np.int64))
        grp.create_dataset("f0", data=freqs)
        grp.create_dataset("f1", data=-freqs * accels / c_val)
        grp.create_dataset("f2", data=np.zeros(n, dtype=np.float64))
        grp.create_dataset("duty_cycle", data=np.full(n, 4 / 64, dtype=np.float64))


def test_candidate_collection_reads_schema_v1_legacy_and_prune(tmp_path: Path) -> None:
    # 1. Schema v1
    v1_path = tmp_path / "dm100.h5"
    _write_schema_v1_h5(
        v1_path,
        dm=100.0,
        freqs=np.array([100.0, 200.0]),
        accels=np.array([5.0, -2.0]),
        snrs=np.array([12.5, 9.0]),
    )
    c_v1 = CandidateCollection.from_hdf5(v1_path)
    assert len(c_v1) == 2
    assert c_v1.param_names == ("accel", "freq")
    assert np.allclose(c_v1.dm, 100.0)
    assert c_v1.metadata["reference_mjd"] == pytest.approx(60000.5)

    # 2. Legacy flat FFA
    leg_path = tmp_path / "legacy_ffa.h5"
    with h5py.File(leg_path, "w") as f:
        f.attrs["ffa_version"] = "0.9"
        f.attrs["complete"] = 1
        f.attrs["dm"] = 50.0
        f.attrs["param_names"] = ["accel", "freq"]
        f.create_dataset("param_sets", data=np.array([[1.0, 150.0]]))
        f.create_dataset("snr", data=np.array([11.0]))
        f.create_dataset("width", data=np.array([2]))
        f.create_dataset("nbins", data=np.array([64]))
    c_leg = CandidateCollection.from_hdf5(leg_path)
    assert len(c_leg) == 1
    assert c_leg.f0[0] == pytest.approx(150.0)
    assert c_leg.dm[0] == pytest.approx(50.0)

    # 3. PruneResultWriter output
    pr_path = tmp_path / "prune.h5"
    with PruneResultWriter(pr_path) as writer:
        writer.write_metadata(
            param_names=["jerk", "accel", "freq"],
            nsegments=8,
            max_sugg=16,
            threshold_scheme=np.ones(8, dtype=np.float32),
        )
        leaves_report = np.zeros((2, 5, 2), dtype=np.float64)
        leaves_report[0, :3, 0] = [0.01, -3.0, 86.48]
        leaves_report[1, :3, 0] = [0.02, 4.0, 120.5]
        writer.write_run_results(
            run_name="000_00",
            scheme=np.ones(8, dtype=np.float32),
            leaves_report=leaves_report,
            scores=np.array([15.2, 8.7], dtype=np.float32),
            scores_ep=np.array([15.0, 8.5], dtype=np.float32),
            pstats=PruneStatsCollection(),
        )
    c_pr = CandidateCollection.from_hdf5(pr_path, dm=238.75)
    assert len(c_pr) == 2
    assert c_pr.param_names == ("jerk", "accel", "freq")
    assert c_pr.f0[0] == pytest.approx(86.48)
    assert np.allclose(c_pr.dm, 238.75)


def test_phase_span_dominance_and_single_dm_clustering(tmp_path: Path) -> None:
    tobs = 1000.0
    # 1. Exact phase_span on linear & curved tracks
    # df = 1e-4 Hz over 1000 s => phase span = df * tobs = 0.1 cycles
    span_lin = phase_span([100.0, 0.0], [100.0001, 0.0], tobs)
    assert span_lin == pytest.approx(0.1, rel=1e-9)
    dist = phase_track_distance([100.0, 0.0], 0.05, [100.0001, 0.0], 0.1, tobs)
    assert dist == pytest.approx(2.0, rel=1e-9)
    # Quadratic track has interior extremum at x = 0: f1 * (T/2)^2 / 2
    span_quad = phase_span([100.0, 0.0], [100.0, 1e-6], tobs)
    assert span_quad == pytest.approx(0.5 * 1e-6 * (0.5 * tobs) ** 2, rel=1e-12)

    # 2. Dominance ratio & reduction
    assert dominance_ratio(np.zeros(129), 0.05) == pytest.approx(1.0)
    p_dom = tmp_path / "dom.h5"
    _write_schema_v1_h5(
        p_dom,
        dm=100.0,
        freqs=np.array([100.0, 100.00004, 250.0]),
        accels=np.array([0.0, 0.0, 0.0]),
        snrs=np.array([20.0, 12.0, 15.0]),
    )
    coll_dom = CandidateCollection.from_hdf5(p_dom)
    reduced, kept_idx = reduce_dominated_candidates(
        coll_dom,
        tobs,
        dominance_kappa=1.5,
        radius=1.0,
        phase_radius=0.05,
    )
    assert len(reduced) == 2
    assert kept_idx.tolist() == [0, 2]

    # 3. Transitive union-find vs anchored grouping (bridge candidate!)
    # width=4, nbins=64 => duty = 0.0625, limit = 0.0625 cycles => df_lim = 6.25e-5 Hz
    # Three points spaced by 5e-5 Hz: A-B linked, B-C linked, A-C NOT directly linked
    p_tr = tmp_path / "transitive.h5"
    _write_schema_v1_h5(
        p_tr,
        dm=238.0,
        freqs=np.array([100.0, 100.00005, 100.00010, 200.0]),
        accels=np.array([10.0, 10.0, 10.0, 10.0]),
        snrs=np.array([20.0, 14.0, 12.0, 15.0]),
    )
    coll_tr = CandidateCollection.from_hdf5(p_tr)
    res_tr = cluster_candidates_single_dm(
        coll_tr,
        tobs,
        options=CandidateClusterOptions(
            grouping="transitive",
            radius=1.0,
            phase_radius=0.05,
            flag_harmonics=True,
            remove_harmonics=True,
        ),
    )
    # Transitive merges {0, 1, 2} into cluster 0; cluster 1 (200 Hz) is 2/1 harmonic
    assert len(res_tr.representatives) == 2
    assert res_tr.member_counts.tolist() == [3, 1]
    assert res_tr.phase_extents[0] == pytest.approx(0.1, rel=1e-9)
    assert res_tr.parent_cluster_ids.tolist() == [-1, 0]
    assert res_tr.harmonic_numerators.tolist() == [0, 2]
    assert res_tr.harmonic_denominators.tolist() == [0, 1]
    assert res_tr.selected.tolist() == [True, False]

    # Anchored mode does NOT bridge A-C directly
    res_anch = cluster_candidates_single_dm(
        coll_tr,
        tobs,
        options=CandidateClusterOptions(
            grouping="anchored",
            radius=1.0,
            phase_radius=0.05,
        ),
    )
    assert len(res_anch.representatives) == 3
    assert res_anch.member_counts.tolist() == [2, 1, 1]


def test_harmonic_matching_tiers() -> None:
    known: list[KnownPeriodEntry] = [
        ("J1748-2446A", 11.5632e-3, "strong"),
        ("J1748-2446G", 21.6719e-3, "weak"),
    ]
    matcher = HarmonicMatcher(known)
    ter5a_p = 11.5632e-3
    # Strong pulsar matches high coprime harmonic 13/11
    m_strong = matcher.match_period(ter5a_p * (13.0 / 11.0))
    assert m_strong.matched
    assert m_strong.pulsar_name == "J1748-2446A"
    assert (m_strong.k, m_strong.m) == (13, 11)

    # Weak pulsar matches 3/2, but NOT 13/11 (tier cap = 3)
    weak_p = 21.6719e-3
    m_weak_ok = matcher.match_period(weak_p * (3.0 / 2.0))
    assert m_weak_ok.matched
    assert m_weak_ok.pulsar_name == "J1748-2446G"
    m_weak_high = match_known_harmonics(
        weak_p * (13.0 / 11.0),
        known,
        rel_tol=1e-4,
    )
    assert not m_weak_high.matched

    # Novel period does not match
    m_novel = matcher.match_period(0.137912345, rel_tol=1e-4)
    assert not m_novel.matched


def test_multi_dm_survey_sifting_and_persistence(tmp_path: Path) -> None:
    ter5a_p = 11.5632e-3
    ter5a_f = 1.0 / ter5a_p
    known = {"J1748-2446A": ter5a_p}
    # Build 4 DM trials: DM=0 (RFI), DM=238, DM=242 (peak for Ter5A + novel), DM=246
    trials = [
        (0.0, [50.0], [0.0], [25.0]),
        (238.0, [50.001, ter5a_f, 173.456], [0.0, 1.0, 2.0], [18.0, 12.0, 9.0]),
        (242.0, [49.999, ter5a_f, 173.457], [0.0, 1.0, 2.0], [15.0, 22.0, 16.0]),
        (246.0, [50.0, ter5a_f, 173.455], [0.0, 1.0, 2.0], [14.0, 11.0, 10.0]),
    ]
    colls = []
    for dm, freqs, accels, snrs in trials:
        fpath = tmp_path / f"dm_{int(dm)}.h5"
        _write_schema_v1_h5(
            fpath,
            dm=dm,
            freqs=np.array(freqs),
            accels=np.array(accels),
            snrs=np.array(snrs),
        )
        colls.append(CandidateCollection.from_hdf5(fpath))

    merged = CandidateCollection.merge(colls)
    groups = SurveyCandidateGroups.from_collection(merged, rel_freq_tol=1e-3)
    assert len(groups) == 3
    # Verify single-append (total members across all groups equals len(merged))
    assert sum(g.member_indices.size for g in groups.groups) == len(merged)

    survivors = groups.filter_dm0_and_span(rfi_ratio=0.5)
    assert len(survivors) == 2
    rfi_groups = [g for g in groups.groups if g.disposition == "rfi"]
    assert len(rfi_groups) == 1
    assert rfi_groups[0].has_dm0_hit

    n_harm = groups.tag_known_harmonics(known)
    assert n_harm == 1
    novel = [g for g in groups.groups if g.disposition == "new"]
    assert len(novel) == 1
    assert novel[0].dm_peak == pytest.approx(242.0)

    groups.set_annotation(novel[0].group_id, notes="Promising new MSP candidate")

    json_path = tmp_path / "sifted.json"
    groups.save_json(json_path)
    loaded_json = SurveyCandidateGroups.load_json(json_path)
    assert len(loaded_json) == 3
    assert loaded_json.groups[novel[0].group_id].notes == "Promising new MSP candidate"

    h5_path = tmp_path / "sifted.h5"
    groups.save_hdf5(h5_path)
    loaded_h5 = SurveyCandidateGroups.load_hdf5(h5_path)
    assert len(loaded_h5) == 3
    assert loaded_h5.groups[novel[0].group_id].notes == "Promising new MSP candidate"


def test_segment_ephemeris_and_tempo2_predictor_accuracy(tmp_path: Path) -> None:
    ref_mjd = 60000.0
    t_end = 1800.0

    # 1. High-order Taylor with snap + crackle (previously blocked by PulsarX F0/F1/F2)
    taylor_params = [1e-7, -2e-5, 3e-3, -15.0, 200.0]
    seg_json = tmp_path / "ephem.json"
    seg_csv = tmp_path / "ephem.csv"
    rows_json = export_segment_ephemeris(
        seg_json,
        model="taylor",
        params=taylor_params,
        ref_mjd=ref_mjd,
        t_start=0.0,
        t_end=t_end,
        n_segments=10,
        fmt="json",
    )
    rows_csv = export_segment_ephemeris(
        seg_csv,
        model="taylor",
        params=taylor_params,
        ref_mjd=ref_mjd,
        t_start=0.0,
        t_end=t_end,
        n_segments=10,
        fmt="csv",
    )
    assert len(rows_json) == 10
    assert len(rows_csv) == 10
    assert seg_json.exists()
    assert seg_csv.exists()
    assert rows_json[0]["mjd"] > ref_mjd

    # 2. TEMPO2 Chebyshev predictor round-trip on Taylor, circular, and Keplerian
    cases: list[tuple[PhaseModelKind, list[float], float]] = [
        ("taylor", taylor_params, t_end),
        ("circular", [2.0 * np.pi / 3600.0, 250.0, 0.8, 0.6], 3600.0),
        ("keplerian", [150.0, 3000.0, 1.5, 0.25, 0.7, 0.4], 6000.0),
    ]
    for kind, params, span_s in cases:
        pred_path = tmp_path / f"t2pred_{kind}.dat"
        mjd_hi = ref_mjd + span_s / SECONDS_PER_DAY
        info = export_tempo2_predictor(
            pred_path,
            model=kind,
            params=params,
            ref_mjd=ref_mjd,
            mjd_start=ref_mjd,
            mjd_end=mjd_hi,
            max_phase_err=1e-4,
        )
        assert info["max_phase_err"] < 1e-4
        test_t = np.linspace(0.0, span_s, 200, dtype=np.float64)
        test_mjd = ref_mjd + test_t / SECONDS_PER_DAY
        phi_true = build_phase_model(kind, params, t_ref=0.0)(test_t)
        phi_pred = evaluate_tempo2_predictor(pred_path, test_mjd)
        assert float(np.max(np.abs(phi_pred - phi_true))) < 1e-4


def test_bin2d_waterfall_and_diagnostic_card() -> None:
    from matplotlib import pyplot as plt  # noqa: PLC0415

    # S/N-preserving rebinning: constant 1.0 signal scales by sqrt(bt * bp)
    ones = np.ones((8, 16), dtype=np.float64)
    rebinned = bin2d(ones, bt=2, bp=4)
    assert rebinned.shape == (4, 4)
    assert np.allclose(rebinned, np.sqrt(8.0))

    # Build synthetic aligned 3D (n_seg, 2, n_bins) folds with a pulse at bin 16
    rng = np.random.default_rng(123)
    n_seg, n_bins = 12, 32
    folds = np.empty((n_seg, 2, n_bins), dtype=np.float32)
    folds[:, 0, :] = rng.normal(0.0, 1.0, size=(n_seg, n_bins))
    folds[:, 1, :] = 1.0
    shifts = np.arange(n_seg, dtype=np.int64) % 5
    for i in range(n_seg):
        folds[i, 0, (16 - int(shifts[i])) % n_bins] += 6.0

    wf = extract_segment_waterfall(folds, phase_shifts=shifts)
    assert wf.amp_map.shape == (n_seg, n_bins)
    assert wf.integrated_profile.shape == (n_bins,)
    assert wf.best_snr > 12.0
    assert int(np.argmax(wf.integrated_profile)) == 16

    t_seg = np.linspace(0.0, 1200.0, n_seg)
    f_curve = 100.0 + 0.01 * np.sin(2.0 * np.pi * t_seg / 1200.0)
    fig = plot_candidate_diagnostic_card(
        wf,
        t_seg,
        f_curve,
        runner_up_freqs=[f_curve + 1e-3],
        rebin=(2, 2),
        title="Test Candidate",
    )
    assert len(fig.axes) == 4
    plt.close(fig)


def test_tempo2_predictor_cheby2d_scaling_and_header(tmp_path: Path) -> None:
    """Verify TEMPO2 Cheby2D 0.25*c_0 + 0.5*sum_{k>=1} c_k T_k(u) & header."""
    from numpy.polynomial import chebyshev as np_cheb  # noqa: PLC0415

    ref_mjd = 60000.0
    span_s = 1800.0
    params = [-15.0, 200.0]
    pred_path = tmp_path / "t2pred_cheby2d.dat"
    info = export_tempo2_predictor(
        pred_path,
        model="taylor",
        params=params,
        ref_mjd=ref_mjd,
        mjd_start=ref_mjd,
        mjd_end=ref_mjd + span_s / SECONDS_PER_DAY,
        max_phase_err=1e-4,
    )
    lines = pred_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"ChebyModelSet {info['n_blocks']} segments"

    # Independently parse raw COEFFS and apply real TEMPO2 Cheby2D_Evaluate formula
    blocks: list[tuple[float, float, np.ndarray]] = []
    lo = hi = 0.0
    raw_coeffs: list[float] = []
    for raw_line in lines:
        parts = raw_line.strip().split()
        if not parts:
            continue
        if parts[0] == "TIME_RANGE":
            lo, hi = float(parts[1]), float(parts[2])
        elif parts[0] == "COEFFS":
            raw_coeffs.append(float(parts[1]))
        elif raw_line.strip() == "ChebyModel END":
            blocks.append((lo, hi, np.asarray(raw_coeffs, dtype=np.float64)))
            raw_coeffs = []

    test_t = np.linspace(0.0, span_s, 100, dtype=np.float64)
    test_mjd = ref_mjd + test_t / SECONDS_PER_DAY
    phi_true = build_phase_model("taylor", params, t_ref=0.0)(test_t)

    phi_indep = np.empty_like(test_mjd)
    for idx, m in enumerate(test_mjd):
        b_lo, b_hi, c = next(
            (blk for blk in blocks if blk[0] <= m <= blk[1]),
            blocks[-1],
        )
        u = ((m - b_lo) - (b_hi - m)) / (b_hi - b_lo)
        # Cheby2D_Evaluate with NCOEFF_FREQ=1: 0.25*c[0] + 0.5*sum_{k>=1} c[k]*T_k(u)
        t_vals = np.array(
            [float(np_cheb.chebval(u, np.eye(c.size)[k])) for k in range(c.size)],
            dtype=np.float64,
        )
        phi_indep[idx] = 0.25 * c[0] + 0.5 * float(np.dot(c[1:], t_vals[1:]))

    assert float(np.max(np.abs(phi_indep - phi_true))) < 1e-4


def test_export_segment_ephemeris_no_cancellation(tmp_path: Path) -> None:
    """Analytical f1 & accel avoid float64 cancellation at dt_seg=1s."""
    from pyloki.utils.psr_utils import C_VAL  # noqa: PLC0415

    ref_mjd = 60000.0
    accel_true = -15.0
    f0_true = 200.0
    f1_true = -f0_true * accel_true / C_VAL

    for n_seg in (180, 1800):
        rows = export_segment_ephemeris(
            tmp_path / f"ephem_{n_seg}.json",
            model="taylor",
            params=[accel_true, f0_true],
            ref_mjd=ref_mjd,
            t_start=0.0,
            t_end=1800.0,
            n_segments=n_seg,
            fmt="json",
        )
        accels = np.array([r["accel_mss"] for r in rows])
        freqs = np.array([r["freq_hz"] for r in rows])
        f1s = np.array([r["f1_hz_s"] for r in rows])
        assert np.allclose(f1s, f1_true, rtol=1e-10, atol=0.0)
        assert np.allclose(accels, -f1_true / freqs * C_VAL, rtol=1e-10, atol=0.0)
        assert np.allclose(accels, accel_true, rtol=1e-3, atol=0.0)

    # Also verify circular, chebyshev, and keplerian models produce accurate f1_hz_s
    omega = 2.0 * np.pi / 3600.0
    rows_circ = export_segment_ephemeris(
        tmp_path / "ephem_circ.json",
        model="circular",
        params=[omega, 250.0, 0.8, 0.6],
        ref_mjd=ref_mjd,
        t_start=0.0,
        t_end=1800.0,
        n_segments=180,
        fmt="json",
    )
    t_mid0 = rows_circ[0]["t_offset_s"]
    z0 = 0.6 * np.sin(omega * t_mid0) + 0.8 * np.cos(omega * t_mid0)
    assert rows_circ[0]["f1_hz_s"] == pytest.approx(250.0 * omega**2 * z0, rel=1e-10)

    rows_cheb = export_segment_ephemeris(
        tmp_path / "ephem_cheb.json",
        model="chebyshev",
        params=[200.0, 1e-3, 0.0],
        ref_mjd=ref_mjd,
        t_start=0.0,
        t_end=1800.0,
        n_segments=180,
        fmt="json",
    )
    # f(u) = 200 + 1e-3 * u with half_span = 900 => df/dt = 1e-3 / 900
    assert rows_cheb[50]["f1_hz_s"] == pytest.approx(1e-3 / 900.0, rel=1e-10)

    rows_kep = export_segment_ephemeris(
        tmp_path / "ephem_kep.json",
        model="keplerian",
        params=[150.0, 3000.0, 1.5, 0.25, 0.7, 0.4],
        ref_mjd=ref_mjd,
        t_start=0.0,
        t_end=1800.0,
        n_segments=180,
        fmt="json",
    )
    assert all(np.isfinite(r["f1_hz_s"]) and r["f1_hz_s"] != 0.0 for r in rows_kep)


def test_diagnostic_card_panel01_overlays_winning_boxcar() -> None:
    """Panel (0, 1) of diagnostic card overlays the winning boxcar template."""
    from matplotlib import pyplot as plt  # noqa: PLC0415

    n_seg, n_bins = 8, 32
    folds = np.zeros((n_seg, 2, n_bins), dtype=np.float32)
    folds[:, 1, :] = 1.0
    folds[:, 0, 10:14] = 4.0
    wf = extract_segment_waterfall(folds)
    t_seg = np.linspace(0.0, 80.0, n_seg)
    f_curve = np.full(n_seg, 100.0)
    fig = plot_candidate_diagnostic_card(wf, t_seg, f_curve)
    ax_p = fig.axes[1]
    labels = [line.get_label() for line in ax_p.lines]
    assert "Profile" in labels
    assert any(lbl.startswith("Boxcar (w=") for lbl in labels)
    plt.close(fig)
