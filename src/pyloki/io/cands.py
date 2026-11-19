from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import attrs
import h5py
import numpy as np
from numpy.polynomial import chebyshev as np_cheb
from numpy.polynomial import polynomial as np_poly

from pyloki import kepler
from pyloki.detection import scoring
from pyloki.utils.misc import C_VAL

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import ClassVar, Self

    from matplotlib import figure as mpl_figure
    from numpy import typing as npt


@attrs.define(auto_attribs=True, slots=True, kw_only=True)
class PruneStats:
    """Container for pruning statistics at a single pruning level.

    Attributes
    ----------
    level : int
        The current pruning level (1 for the first addition).
    seg_idx : int
        The segment index being added.
    threshold : float
        The nominal threshold from the threshold scheme.
    threshold_eff : float
        The threshold actually applied. Equal to `threshold` unless the candidate
        buffer overflowed, in which case pruning ratchets the cut up to
        `max(threshold, top-K, median)` and never lowers it again for that level.
    score_min : float
        The minimum leaf score.
    score_max : float
        The maximum leaf score.
    n_branches : int
        The total number of tree branches.
    n_leaves : int
        The total number of leaves.
    n_leaves_phy : int
        The total number of physical leaves.
    n_leaves_surv : int
        The total number of surviving leaves after pruning.
    lb_leaves : float
        The log base 2 of the number of physical leaves.
    branch_frac : float
        Branching fraction (average number of leaves per branch).
    phys_frac : float
        The fraction of physical leaves to total leaves.
    surv_frac : float
        The fraction of surviving leaves to physical leaves.
    """

    level: int
    seg_idx: int
    threshold: float
    threshold_eff: float = float("nan")
    score_min: float = 0.0
    score_max: float = 0.0
    n_branches: int = 1
    n_leaves: int = 1
    n_leaves_phy: int = 1
    n_leaves_surv: int = 1

    @property
    def lb_leaves(self) -> float:
        if self.n_leaves <= 0:
            return 0.0
        return np.round(np.log2(self.n_leaves), 2)

    @property
    def lb_leaves_phys(self) -> float:
        if self.n_leaves_phy <= 0:
            return 0.0
        return np.round(np.log2(self.n_leaves_phy), 2)

    @property
    def branch_frac(self) -> float:
        if self.n_branches == 0:
            return 0.0
        return np.round(self.n_leaves_phy / self.n_branches, 2)

    @property
    def phys_frac(self) -> float:
        if self.n_leaves == 0:
            return 0.0
        return np.round(self.n_leaves_phy / self.n_leaves, 2)

    @property
    def surv_frac(self) -> float:
        if self.n_leaves_phy == 0:
            return 0.0
        return np.round(self.n_leaves_surv / self.n_leaves_phy, 2)

    def update(self, stats_dict: dict[str, float]) -> None:
        for key, value in stats_dict.items():
            setattr(self, key, value)

    def get_summary(self) -> str:
        summary = []
        summary.append(
            f"Prune level: {self.level:3d}, seg_idx: {self.seg_idx:3d}, "
            f"leaves: {self.lb_leaves:5.2f}, leaves_phys: {self.lb_leaves_phys:5.2f}, "
            f"branch_frac: {self.branch_frac:5.2f},",
        )
        summary.append(
            f"score thresh: {self.threshold:5.2f}, eff: {self.threshold_eff:5.2f}, "
            f"max: {self.score_max:5.2f}, "
            f"min: {self.score_min:5.2f}, P(surv): {self.surv_frac:4.2f}",
        )
        return "".join(summary) + "\n"


@attrs.define(auto_attribs=True, slots=True, kw_only=True)
class PruneStatsCollection:
    """Collection of PruneStats objects across multiple pruning levels."""

    stats_list: list[PruneStats] = attrs.Factory(list)
    timers: dict[str, float] = attrs.Factory(dict)

    TIMER_NAMES: ClassVar[list[str]] = [
        "branch",
        "validate",
        "resolve",
        "shift_add",
        "score",
        "threshold",
        "transform",
        "batch_add",
    ]

    def __attrs_post_init__(self) -> None:
        self.timers = dict.fromkeys(self.TIMER_NAMES, 0.0)

    @property
    def nstages(self) -> int:
        return len(self.stats_list)

    def update_stats(
        self,
        stats: PruneStats,
        timer_arr: np.ndarray | None = None,
    ) -> None:
        """Update the collection with new PruneStats and timer values."""
        self.stats_list.append(stats)
        if timer_arr is not None:
            if len(timer_arr) != len(self.TIMER_NAMES):
                msg = f"Invalid timer array length: {len(timer_arr)}"
                raise ValueError(msg)
            for i, name in enumerate(self.TIMER_NAMES):
                self.timers[name] += timer_arr[i]

    def get_stats(self, level: int) -> PruneStats | None:
        """Get PruneStats for a specific level."""
        for stats in self.stats_list:
            if stats.level == level:
                return stats
        return None

    def get_all_summaries(self) -> str:
        """Get formatted summaries of all pruning levels."""
        return "".join(
            stats.get_summary()
            for stats in sorted(self.stats_list, key=lambda x: x.level)
        )

    def get_stats_summary(self) -> str:
        """Get formatted stats summaries for final level."""
        if not self.stats_list:
            return "Score: 0.00, Leaves: 0.00"
        return (
            f"Score: {self.stats_list[-1].score_max:.2f}, "
            f"Leaves: {self.stats_list[-1].lb_leaves:.2f}"
        )

    def get_timer_summary(self) -> str:
        """Get formatted timer summary for final level."""
        total_time = sum(self.timers.values())
        timer_percent = {
            name: (time / total_time) * 100 if total_time > 0 else 0.0
            for name, time in self.timers.items()
        }
        lines = [f"Timing breakdown: {total_time:.2f}s"]
        lines.extend(
            f"  {name:10s}: {value:6.1f}%"
            for name, value in sorted(
                timer_percent.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        return "\n".join(lines)

    def get_concise_timer_summary(self) -> str:
        """Get a concise single-line timer summary for terminal output."""
        total_time = sum(self.timers.values())
        sorted_times = sorted(self.timers.items(), key=lambda x: x[1], reverse=True)
        top_times = sorted_times[:4]  # Take top 4 operations
        formatted_times = [
            f"{name}: {(time / total_time) * 100:.0f}%"
            for name, time in top_times
            if time > 0
        ]
        time_breakdown = " | ".join(formatted_times)

        return f"Total: {total_time:.1f}s ({time_breakdown})"

    def to_dict(self) -> dict:
        """Convert stats collection to a dictionary for saving to file."""
        return {
            "nstages": self.nstages,
            "levels": [attrs.asdict(stats) for stats in self.stats_list],
            "timers": self.timers,
        }

    def to_array(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert stats collection to numpy arrays for HDF5 storage."""
        level_dtype = np.dtype(
            [
                ("level", np.int32),
                ("seg_idx", np.int32),
                ("threshold", np.float32),
                ("threshold_eff", np.float32),
                ("score_min", np.float32),
                ("score_max", np.float32),
                ("n_branches", np.int32),
                ("n_leaves", np.int32),
                ("n_leaves_phy", np.int32),
                ("n_leaves_surv", np.int32),
            ],
        )
        level_stats = np.zeros(len(self.stats_list), dtype=level_dtype)
        for i, stats in enumerate(self.stats_list):
            level_stats[i] = (
                stats.level,
                stats.seg_idx,
                stats.threshold,
                stats.threshold_eff,
                stats.score_min,
                stats.score_max,
                stats.n_branches,
                stats.n_leaves,
                stats.n_leaves_phy,
                stats.n_leaves_surv,
            )
        timer_dtype = np.dtype([(name, np.float32) for name in self.TIMER_NAMES])
        timer_stats = np.zeros(1, dtype=timer_dtype)
        for name in self.TIMER_NAMES:
            timer_stats[0][name] = self.timers[name]

        return level_stats, timer_stats

    @classmethod
    def from_dict(cls, data: dict) -> PruneStatsCollection:
        """Create PruneStatsCollection from a dictionary."""
        collection = cls()
        for level_data in data["levels"]:
            stats = PruneStats(**level_data)
            collection.update_stats(stats)
        collection.timers = data["timers"]
        return collection

    @classmethod
    def from_arrays(
        cls,
        level_stats: np.ndarray,
        timer_stats: np.ndarray,
    ) -> PruneStatsCollection:
        """Create PruneStatsCollection from numpy arrays.

        Fields are read by name, so extra level fields (LOKI's files carry more) are
        ignored. Files written before `threshold_eff` was saved have no such field;
        it is restored as NaN. Timers are read by name over 0.0 defaults: a timer the
        file lacks stays 0.0, and one pyloki does not track is kept (but `to_array`
        writes only `TIMER_NAMES`).
        """
        collection = cls()
        has_eff = "threshold_eff" in (level_stats.dtype.names or ())
        for row in level_stats:
            stats = PruneStats(
                level=int(row["level"]),
                seg_idx=int(row["seg_idx"]),
                threshold=float(row["threshold"]),
                threshold_eff=float(row["threshold_eff"]) if has_eff else float("nan"),
                score_min=float(row["score_min"]),
                score_max=float(row["score_max"]),
                n_branches=int(row["n_branches"]),
                n_leaves=int(row["n_leaves"]),
                n_leaves_phy=int(row["n_leaves_phy"]),
                n_leaves_surv=int(row["n_leaves_surv"]),
            )
            collection.stats_list.append(stats)

        # Reconstruct timers by name over the 0.0 defaults: a stage the file has no
        # timer for (LOKI writes no `batch_add`) took no time, and a timer pyloki does
        # not track (LOKI's `rfi`) is kept, so the summaries still add up.
        for name in timer_stats.dtype.names or ():
            collection.timers[name] = float(timer_stats[0][name])

        return collection


class PruneResultWriter:
    def __init__(
        self,
        filename: str | Path,
        mode: str = "w",
    ) -> None:
        self.filename = Path(filename)
        self.file = None
        self.mode = mode
        self.runs_group: h5py.Group | None = None

    def __enter__(self) -> Self:
        self.file = h5py.File(self.filename, self.mode)
        if self.file is None:
            msg = f"Could not open file: {self.filename}"
            raise ValueError(msg)
        if "runs" not in self.file:
            self.file.create_group("runs")
        self.runs_group = self.file["runs"]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        if self.file:
            self.file.close()

    def write_metadata(
        self,
        param_names: list[str],
        nsegments: int,
        max_sugg: int,
        threshold_scheme: np.ndarray,
    ) -> None:
        if self.file is None:
            msg = "File not open."
            raise ValueError(msg)
        # Store package version
        self.file.attrs["pruning_version"] = metadata.version("pyloki")
        self.file.attrs["param_names"] = param_names
        self.file.attrs["nsegments"] = nsegments
        self.file.attrs["max_sugg"] = max_sugg
        self.file.create_dataset(
            "threshold_scheme",
            data=threshold_scheme,
            compression="gzip",
            compression_opts=9,
        )

    def write_run_results(
        self,
        run_name: str,
        scheme: np.ndarray,
        leaves_report: np.ndarray,
        scores: np.ndarray,
        scores_ep: np.ndarray,
        pstats: PruneStatsCollection,
    ) -> None:
        if self.runs_group is None:
            msg = "No runs group found in the file."
            raise ValueError(msg)
        if run_name in self.runs_group:
            msg = f"Run name {run_name} already exists in the file."
            raise ValueError(msg)
        run_group = self.runs_group.create_group(run_name)
        level_stats, timer_stats = pstats.to_array()
        for name, data in [
            ("scheme", scheme),
            ("param_sets", leaves_report),
            ("scores_ep", scores_ep),
            ("scores", scores),
            ("level_stats", level_stats),
            ("timer_stats", timer_stats),
        ]:
            run_group.create_dataset(
                name,
                data=data,
                compression="gzip",
                compression_opts=9,
            )


def extract_ref_seg(file_path: Path) -> int:
    # Extracts ref_seg from e.g. tmp_003_01_log.txt or tmp_base_003_01_results.h5
    match = re.search(
        r"^tmp_(?:.+_)?(\d+)_\d+_(?:log\.txt|results\.h5)$",
        file_path.name,
    )
    if match:
        return int(match.group(1))
    return 0


def merge_prune_result_files(
    results_dir: str | Path,
    log_file: Path,
    result_file: Path,
    temp_files: list[tuple[Path, Path]] | None = None,
) -> None:
    """Merge temporary HDF5 and log files into result files.

    This function merges temporary HDF5 files created during the multiprocessing
    of pruning results into a final result file. It also merges log files into a
    single log file. The temporary files are deleted after merging. Merging order
    is determined by ref_seg.

    Parameters
    ----------
    results_dir : str
        Directory containing the temporary files.
    log_file : Path
        Path to the log file to be merged.
    result_file : Path
        Path to the final result file.
    temp_files : list[tuple[Path, Path]] | None, optional
        The (log, result) temporaries to merge, in order. Pass the files this call
        wrote: without it, every ``tmp_*`` file in `results_dir` is merged, including
        another search's that shares the directory. A missing file is skipped (a
        worker that failed may have written none).

    """
    if temp_files is not None:
        temp_log_files = [log for log, _ in temp_files if log.exists()]
        temp_h5_files = [h5 for _, h5 in temp_files if h5.exists()]
    else:
        temp_log_files = list(Path(results_dir).glob("tmp_*_log.txt"))
        temp_h5_files = list(Path(results_dir).glob("tmp_*_results.h5"))

        # Sort files by ref_seg
        temp_log_files.sort(key=extract_ref_seg)
        temp_h5_files.sort(key=extract_ref_seg)

    with log_file.open("a") as main_log:
        for temp_log in temp_log_files:
            with temp_log.open("r") as log:
                main_log.write(log.read())
            temp_log.unlink()

    # Open the master result file in append mode
    with h5py.File(result_file, "a") as main_h5:
        if "runs" not in main_h5:
            main_h5.create_group("runs")
        for temp_h5_path in temp_h5_files:
            with h5py.File(temp_h5_path, "r") as temp_h5:
                if "runs" in temp_h5:
                    for run_name in temp_h5["runs"]:
                        if run_name in main_h5["runs"]:
                            continue
                        temp_h5.copy(f"runs/{run_name}", main_h5["runs"])
            temp_h5_path.unlink()


# ---------------------------------------------------------------------------
# Unified Candidate I/O, Single-DM Clustering, Multi-DM Sifting & Export
# ---------------------------------------------------------------------------

CANDIDATE_TABLES = ("candidates", "clusters", "detections")
Disposition = Literal["new", "known_harmonic", "rfi", "noise"]
StrengthTier = Literal["strong", "medium", "weak"]
KnownPeriodEntry = tuple[str, float] | tuple[str, float, StrengthTier]
KnownPeriodsInput = Mapping[str, float] | Sequence[float] | Sequence[KnownPeriodEntry]
SECONDS_PER_DAY = 86400.0


def _decode_str(value: Any) -> str:  # noqa: ANN401
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _canonical_f0_f1_f2(
    param_sets: np.ndarray,
    param_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = param_sets.shape[0]
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty
    if not param_names or param_names[-1] != "freq":
        f0 = param_sets[:, -1].astype(np.float64)
        zeros = np.zeros(n, dtype=np.float64)
        return f0, zeros, zeros
    f0 = param_sets[:, -1].astype(np.float64)
    f1 = (
        -f0 * param_sets[:, -2].astype(np.float64) / C_VAL
        if param_sets.shape[1] >= 2
        else np.zeros(n, dtype=np.float64)
    )
    f2 = (
        -f0 * param_sets[:, -3].astype(np.float64) / C_VAL
        if param_sets.shape[1] >= 3
        else np.zeros(n, dtype=np.float64)
    )
    return f0, f1, f2


@attrs.define(auto_attribs=True, kw_only=True)
class CandidateCollection:
    """Unified in-memory candidate table across HDF5 v1, legacy FFA, and prune files."""

    ids: npt.NDArray[np.uint64]
    param_sets: npt.NDArray[np.float64]
    param_names: tuple[str, ...]
    snr: npt.NDArray[np.float64]
    width: npt.NDArray[np.int64]
    nbins: npt.NDArray[np.int64]
    f0: npt.NDArray[np.float64]
    f1: npt.NDArray[np.float64]
    f2: npt.NDArray[np.float64]
    dm: npt.NDArray[np.float64]
    metadata: dict[str, Any] = attrs.Factory(dict)
    cluster_ids: npt.NDArray[np.int64] | None = None

    def __len__(self) -> int:
        return int(self.snr.size)

    @classmethod
    def from_hdf5(
        cls,
        path: str | Path,
        *,
        table: str = "candidates",
        dm: float | None = None,
        run_name: str | None = None,
    ) -> CandidateCollection:
        """Load candidates from a schema v1, legacy flat-FFA, or prune HDF5 file."""
        if table not in CANDIDATE_TABLES:
            msg = (
                f"Unknown candidate table {table!r}; expected one of {CANDIDATE_TABLES}"
            )
            raise ValueError(msg)
        filepath = Path(path)
        with h5py.File(filepath, "r") as handle:
            if "schema_version" in handle.attrs:
                return cls._read_schema_v1(handle, table=table, dm_override=dm)
            if "ffa_version" in handle.attrs:
                return cls._read_legacy_ffa(handle, dm_override=dm)
            if "runs" in handle:
                return cls._read_prune_results(
                    handle,
                    dm_override=dm,
                    run_name=run_name,
                )
        msg = f"Unrecognized candidate HDF5 layout: {filepath}"
        raise ValueError(msg)

    @classmethod
    def _read_schema_v1(
        cls,
        handle: h5py.File,
        *,
        table: str,
        dm_override: float | None,
    ) -> CandidateCollection:
        if int(handle.attrs.get("complete", 0)) != 1:
            msg = "Candidate file is incomplete"
            raise ValueError(msg)
        version = _decode_str(handle.attrs["schema_version"])
        if version.split(".")[0] != "1":
            msg = f"Unsupported candidate schema version: {version}"
            raise ValueError(msg)
        if table not in handle:
            msg = f"Table /{table} not found in candidate file"
            raise ValueError(msg)
        if "observation" not in handle:
            msg = "Candidate file has no /observation metadata group"
            raise ValueError(msg)

        obs_meta = {
            k: (_decode_str(v) if isinstance(v, bytes) else v)
            for k, v in handle["observation"].attrs.items()
        }
        param_names = tuple(_decode_str(x) for x in handle.attrs.get("param_names", []))
        group = handle[table]
        snr = np.asarray(group["snr"][:], dtype=np.float64)
        n = snr.size
        ids = np.asarray(group["id"][:], dtype=np.uint64)
        param_sets = np.asarray(group["param_sets"][:], dtype=np.float64)
        width = np.asarray(group["width"][:], dtype=np.int64)
        nbins = np.asarray(group["nbins"][:], dtype=np.int64)
        f0 = np.asarray(group["f0"][:], dtype=np.float64)
        f1 = np.asarray(group["f1"][:], dtype=np.float64)
        f2 = np.asarray(group["f2"][:], dtype=np.float64)
        cluster_ids = (
            np.asarray(group["cluster_id"][:], dtype=np.int64)
            if "cluster_id" in group
            else None
        )
        dm_val = float(
            dm_override if dm_override is not None else obs_meta.get("dm", 0.0),
        )
        dm_arr = np.full(n, dm_val, dtype=np.float64)
        return cls(
            ids=ids,
            param_sets=param_sets,
            param_names=param_names,
            snr=snr,
            width=width,
            nbins=nbins,
            f0=f0,
            f1=f1,
            f2=f2,
            dm=dm_arr,
            metadata=obs_meta,
            cluster_ids=cluster_ids,
        )

    @classmethod
    def _read_legacy_ffa(
        cls,
        handle: h5py.File,
        *,
        dm_override: float | None,
    ) -> CandidateCollection:
        if int(handle.attrs.get("complete", 0)) != 1:
            msg = "Candidate file is incomplete"
            raise ValueError(msg)
        meta = {
            k: (_decode_str(v) if isinstance(v, bytes) else v)
            for k, v in handle.attrs.items()
        }
        param_names = tuple(_decode_str(x) for x in handle.attrs.get("param_names", []))
        snr = np.asarray(handle["snr"][:], dtype=np.float64)
        n = snr.size
        param_sets = np.asarray(handle["param_sets"][:], dtype=np.float64)
        width = np.asarray(handle["width"][:], dtype=np.int64)
        nbins = np.asarray(handle["nbins"][:], dtype=np.int64)
        f0, f1, f2 = _canonical_f0_f1_f2(param_sets, param_names)
        dm_val = float(dm_override if dm_override is not None else meta.get("dm", 0.0))
        return cls(
            ids=np.arange(n, dtype=np.uint64),
            param_sets=param_sets,
            param_names=param_names,
            snr=snr,
            width=width,
            nbins=nbins,
            f0=f0,
            f1=f1,
            f2=f2,
            dm=np.full(n, dm_val, dtype=np.float64),
            metadata=meta,
        )

    @classmethod
    def _read_prune_results(
        cls,
        handle: h5py.File,
        *,
        dm_override: float | None,
        run_name: str | None,
    ) -> CandidateCollection:
        meta = {
            k: (_decode_str(v) if isinstance(v, bytes) else v)
            for k, v in handle.attrs.items()
        }
        raw_names = tuple(_decode_str(x) for x in handle.attrs.get("param_names", []))
        runs_grp = handle["runs"]
        run_keys = [run_name] if run_name is not None else list(runs_grp.keys())
        params_list: list[np.ndarray] = []
        snr_list: list[np.ndarray] = []
        for rk in run_keys:
            if rk not in runs_grp:
                msg = f"Run {rk!r} not found in prune results file"
                raise KeyError(msg)
            rg = runs_grp[rk]
            if "param_sets" not in rg or "scores" not in rg:
                continue
            ps = np.asarray(rg["param_sets"][:], dtype=np.float64)
            sc = np.asarray(rg["scores"][:], dtype=np.float64)
            if ps.ndim == 3:
                # leaves_report shape (n_surv, n_reported + 2, 2): column 0 is values
                ps = ps[:, :-2, 0] if ps.shape[1] > 2 else ps[:, :, 0]
            if ps.shape[0] == 0:
                continue
            params_list.append(ps)
            snr_list.append(sc)

        if not params_list:
            n_cols = len(raw_names)
            empty_f = np.empty(0, dtype=np.float64)
            return cls(
                ids=np.empty(0, dtype=np.uint64),
                param_sets=np.empty((0, n_cols), dtype=np.float64),
                param_names=raw_names,
                snr=empty_f,
                width=np.empty(0, dtype=np.int64),
                nbins=np.empty(0, dtype=np.int64),
                f0=empty_f,
                f1=empty_f,
                f2=empty_f,
                dm=empty_f,
                metadata=meta,
            )

        param_sets = np.concatenate(params_list, axis=0)
        snr = np.concatenate(snr_list, axis=0)
        n = snr.size
        n_cols = param_sets.shape[1]
        if len(raw_names) >= n_cols:
            param_names = raw_names[-n_cols:]
        else:
            canonical = ("crackle", "snap", "jerk", "accel", "freq")
            param_names = canonical[-n_cols:]
        f0, f1, f2 = _canonical_f0_f1_f2(param_sets, param_names)
        dm_val = float(dm_override if dm_override is not None else meta.get("dm", 0.0))
        return cls(
            ids=np.arange(n, dtype=np.uint64),
            param_sets=param_sets,
            param_names=param_names,
            snr=snr,
            width=np.ones(n, dtype=np.int64),
            nbins=np.full(n, int(meta.get("nbins", 64)), dtype=np.int64),
            f0=f0,
            f1=f1,
            f2=f2,
            dm=np.full(n, dm_val, dtype=np.float64),
            metadata=meta,
        )

    @classmethod
    def merge(cls, collections: Sequence[CandidateCollection]) -> CandidateCollection:
        """Concatenate multiple ``CandidateCollection`` tables (e.g. across DMs)."""
        non_empty = [c for c in collections if len(c) > 0]
        if not non_empty:
            empty_f = np.empty(0, dtype=np.float64)
            return cls(
                ids=np.empty(0, dtype=np.uint64),
                param_sets=np.empty((0, 0), dtype=np.float64),
                param_names=(),
                snr=empty_f,
                width=np.empty(0, dtype=np.int64),
                nbins=np.empty(0, dtype=np.int64),
                f0=empty_f,
                f1=empty_f,
                f2=empty_f,
                dm=empty_f,
            )
        ref_names = non_empty[0].param_names
        if any(c.param_names != ref_names for c in non_empty[1:]):
            msg = "All merged CandidateCollections must share the same param_names"
            raise ValueError(msg)
        n_total = sum(len(c) for c in non_empty)
        return cls(
            ids=np.arange(n_total, dtype=np.uint64),
            param_sets=np.concatenate([c.param_sets for c in non_empty], axis=0),
            param_names=ref_names,
            snr=np.concatenate([c.snr for c in non_empty], axis=0),
            width=np.concatenate([c.width for c in non_empty], axis=0),
            nbins=np.concatenate([c.nbins for c in non_empty], axis=0),
            f0=np.concatenate([c.f0 for c in non_empty], axis=0),
            f1=np.concatenate([c.f1 for c in non_empty], axis=0),
            f2=np.concatenate([c.f2 for c in non_empty], axis=0),
            dm=np.concatenate([c.dm for c in non_empty], axis=0),
            metadata=dict(non_empty[0].metadata),
        )


_PHASE_DEGREE = 6
_DOMINANCE_SAMPLES = 129
_DOMINANCE_MAX_DUTY = 0.3
_DOMINANCE_WIDTH_STEP = 1.5


@attrs.frozen(auto_attribs=True, kw_only=True)
class CandidateClusterOptions:
    """Configuration for single-DM phase-track candidate clustering."""

    grouping: Literal["transitive", "anchored"] = "transitive"
    radius: float = 1.0
    phase_radius: float = 0.05
    flag_harmonics: bool = False
    remove_harmonics: bool = False
    harmonic_denominator_max: int = 16
    harmonic_phase_distance: float = 1.0
    harmonic_snr_distance: float = 3.0
    harmonic_doppler_tolerance: float = 0.0
    harmonic_doppler_min_snr: float = 20.0
    max_candidates: int = 0

    def validate(self) -> None:
        if self.grouping not in ("transitive", "anchored"):
            msg = f"grouping must be 'transitive' or 'anchored', got {self.grouping!r}"
            raise ValueError(msg)
        if not math.isfinite(self.radius) or self.radius <= 0.0:
            msg = "radius must be finite and positive"
            raise ValueError(msg)
        if not math.isfinite(self.phase_radius) or self.phase_radius < 0.0:
            msg = "phase_radius must be finite and nonnegative"
            raise ValueError(msg)
        if self.remove_harmonics and not self.flag_harmonics:
            msg = "remove_harmonics requires flag_harmonics=True"
            raise ValueError(msg)
        if not (1 <= self.harmonic_denominator_max <= 10000):
            msg = "harmonic_denominator_max must be in [1, 10000]"
            raise ValueError(msg)
        if (
            not math.isfinite(self.harmonic_phase_distance)
            or self.harmonic_phase_distance <= 0.0
        ):
            msg = "harmonic_phase_distance must be finite and positive"
            raise ValueError(msg)
        if (
            not math.isfinite(self.harmonic_snr_distance)
            or self.harmonic_snr_distance <= 0.0
        ):
            msg = "harmonic_snr_distance must be finite and positive"
            raise ValueError(msg)
        if not (0.0 <= self.harmonic_doppler_tolerance < 0.1):
            msg = "harmonic_doppler_tolerance must be in [0, 0.1)"
            raise ValueError(msg)
        if self.max_candidates < 0:
            msg = "max_candidates must be nonnegative"
            raise ValueError(msg)


def _collection_derivatives(collection: CandidateCollection) -> npt.NDArray[np.float64]:
    """Extract midpoint frequency derivatives d^k f / dt^k (shape ``(N, 6)``)."""
    n = len(collection)
    derivs = np.zeros((n, _PHASE_DEGREE), dtype=np.float64)
    if n == 0:
        return derivs
    p = collection.param_sets
    n_cols = p.shape[1] if p.ndim == 2 else 0
    if n_cols >= 1 and collection.param_names and collection.param_names[-1] == "freq":
        f0 = p[:, -1].astype(np.float64)
        derivs[:, 0] = f0
        max_k = min(n_cols - 1, _PHASE_DEGREE - 1)
        for k in range(1, max_k + 1):
            derivs[:, k] = -f0 * p[:, -1 - k].astype(np.float64) / C_VAL
    else:
        derivs[:, 0] = collection.f0
        derivs[:, 1] = collection.f1
        derivs[:, 2] = collection.f2
    return derivs


def _phase_diff_poly(
    derivs_a: np.ndarray,
    derivs_b: np.ndarray,
    tobs: float,
    scale_a: float = 1.0,
) -> np.ndarray:
    """Phase-difference polynomial coefficients c[0..6] on x in [-1, 1]."""
    coeffs = np.zeros(_PHASE_DEGREE + 1, dtype=np.float64)
    factor = 1.0
    half_t = 0.5 * tobs
    n_d = min(derivs_a.size, derivs_b.size, _PHASE_DEGREE)
    for k in range(n_d):
        factor *= half_t / float(k + 1)
        coeffs[k + 1] = (scale_a * float(derivs_a[k]) - float(derivs_b[k])) * factor
    # Trim trailing zeros
    last = coeffs.size
    while last > 1 and coeffs[last - 1] == 0.0:
        last -= 1
    return coeffs[:last]


def phase_span(
    derivs_a: npt.ArrayLike,
    derivs_b: npt.ArrayLike,
    tobs: float,
    *,
    scale_a: float = 1.0,
    reject_above: float | None = None,
) -> float:
    """Constant-phase-invariant span (cycles) over ``[-tobs/2, tobs/2]``."""
    if not math.isfinite(tobs) or tobs <= 0.0:
        msg = "tobs must be finite and positive"
        raise ValueError(msg)
    da = np.asarray(derivs_a, dtype=np.float64)
    db = np.asarray(derivs_b, dtype=np.float64)
    coeffs = _phase_diff_poly(da, db, tobs, scale_a=scale_a)
    if coeffs.size <= 1:
        return 0.0

    at_hi = float(np_poly.polyval(1.0, coeffs))
    at_lo = float(np_poly.polyval(-1.0, coeffs))
    low = min(at_hi, at_lo)
    high = max(at_hi, at_lo)
    if reject_above is not None:
        bound = max(high - low, abs(at_hi), abs(at_lo)) * (1.0 - 1e-6)
        if bound > reject_above:
            return bound

    if coeffs.size > 2:
        dcoeffs = np_poly.polyder(coeffs)
        roots_raw = np_poly.polyroots(dcoeffs)
        for r in roots_raw:
            if abs(r.imag) <= 1e-9:
                x = float(r.real)
                if -1.0 < x < 1.0:
                    y = float(np_poly.polyval(x, coeffs))
                    low = min(low, y)
                    high = max(high, y)
    return high - low


def phase_track_distance(
    derivs_a: npt.ArrayLike,
    duty_a: float,
    derivs_b: npt.ArrayLike,
    duty_b: float,
    tobs: float,
) -> float:
    """Phase-track span normalized by the narrower pulse duty cycle."""
    narrow = min(duty_a, duty_b)
    if narrow <= 0.0:
        msg = "duty cycles must be positive"
        raise ValueError(msg)
    return phase_span(derivs_a, derivs_b, tobs) / narrow


def _link_reach(span: float, tobs: float) -> float:
    """Conservative frequency prefilter via Bernstein's inequality."""
    return 1.01 * float(_PHASE_DEGREE) * span / tobs


def dominance_ratio(dphi: npt.ArrayLike, duty: float) -> float:
    """Boxcar S/N recovery fraction for duty ``duty`` along residual ``dphi``."""
    arr = np.sort(np.asarray(dphi, dtype=np.float64))
    n = arr.size
    if n == 0 or duty <= 0.0:
        return 0.0
    best = 0.0
    width = duty
    while True:
        most = 0
        j = 0
        for i in range(n):
            while j < n and arr[j] <= arr[i] + width:
                j += 1
            most = max(most, j - i)
        best = max(best, (most / float(n)) * math.sqrt(duty / width))
        if width >= _DOMINANCE_MAX_DUTY:
            break
        width = min(width * _DOMINANCE_WIDTH_STEP, _DOMINANCE_MAX_DUTY)
    return best


def reduce_dominated_candidates(
    collection: CandidateCollection,
    tobs: float,
    *,
    dominance_kappa: float = 1.5,
    radius: float = 1.0,
    phase_radius: float = 0.05,
    freq_reach_bins: float | None = None,
) -> tuple[CandidateCollection, npt.NDArray[np.int64]]:
    """Absorb weaker trials explained by a stronger trial's phase track or boxcar lobe.

    Returns ``(kept_collection, kept_indices)`` preserving descending S/N order.
    """
    n = len(collection)
    if n == 0 or dominance_kappa <= 0.0:
        idx = np.arange(n, dtype=np.int64)
        return collection, idx

    derivs = _collection_derivatives(collection)
    duties = collection.width.astype(np.float64) / np.maximum(
        collection.nbins.astype(np.float64),
        1.0,
    )
    order = np.argsort(-collection.snr, kind="mergesort")
    x_grid = np.linspace(-1.0, 1.0, _DOMINANCE_SAMPLES, dtype=np.float64)
    reach_hz = (
        (2.0 * freq_reach_bins / tobs)
        if freq_reach_bins is not None and freq_reach_bins > 0.0
        else None
    )

    kept: list[int] = []
    for pos in range(n):
        i = int(order[pos])
        fi = float(derivs[i, 0])
        wi = float(duties[i])
        snr_i = float(collection.snr[i])
        explained = False
        for k in kept:
            if reach_hz is not None and abs(fi - float(derivs[k, 0])) > reach_hz:
                continue
            snr_k = float(collection.snr[k])
            if snr_i > dominance_kappa * snr_k:
                continue
            wk = float(duties[k])
            limit = max(phase_radius, radius * min(wi, wk))
            if phase_span(derivs[k], derivs[i], tobs, reject_above=limit) <= limit:
                explained = True
                break
            coeffs = _phase_diff_poly(derivs[k], derivs[i], tobs)
            dphi = np_poly.polyval(x_grid, coeffs)
            if snr_i <= dominance_kappa * snr_k * dominance_ratio(dphi, wk):
                explained = True
                break
        if not explained:
            kept.append(i)

    kept_arr = np.asarray(kept, dtype=np.int64)
    reduced = CandidateCollection(
        ids=collection.ids[kept_arr],
        param_sets=collection.param_sets[kept_arr],
        param_names=collection.param_names,
        snr=collection.snr[kept_arr],
        width=collection.width[kept_arr],
        nbins=collection.nbins[kept_arr],
        f0=collection.f0[kept_arr],
        f1=collection.f1[kept_arr],
        f2=collection.f2[kept_arr],
        dm=collection.dm[kept_arr],
        metadata=dict(collection.metadata),
        cluster_ids=(
            collection.cluster_ids[kept_arr]
            if collection.cluster_ids is not None
            else None
        ),
    )
    return reduced, kept_arr


@attrs.frozen(auto_attribs=True, kw_only=True)
class SingleDMClusterResult:
    """Output of single-DM phase-track candidate clustering."""

    representatives: CandidateCollection
    cluster_ids: npt.NDArray[np.int64]
    member_counts: npt.NDArray[np.int64]
    phase_extents: npt.NDArray[np.float64]
    parent_cluster_ids: npt.NDArray[np.int64]
    harmonic_numerators: npt.NDArray[np.int64]
    harmonic_denominators: npt.NDArray[np.int64]
    selected: npt.NDArray[np.bool_]


def _flag_cluster_harmonics(
    rep_derivs: np.ndarray,
    rep_duties: np.ndarray,
    rep_snrs: np.ndarray,
    tobs: float,
    options: CandidateClusterOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flag self-harmonic families among single-DM cluster representatives."""
    n_clusters = rep_snrs.size
    parents = np.full(n_clusters, -1, dtype=np.int64)
    nums = np.zeros(n_clusters, dtype=np.int64)
    dens = np.zeros(n_clusters, dtype=np.int64)
    if n_clusters <= 1:
        return parents, nums, dens

    max_order = options.harmonic_denominator_max
    ratios = [
        (p * q, p / q, p, q)
        for p in range(1, max_order + 1)
        for q in range(1, max_order + 1)
        if math.gcd(p, q) == 1
    ]
    ratios.sort(key=lambda item: (item[0], item[1]))
    sorted_vals = (
        np.sort(np.array([r[1] for r in ratios], dtype=np.float64))
        if options.harmonic_doppler_tolerance > 0.0
        else np.empty(0, dtype=np.float64)
    )
    doppler_tol = options.harmonic_doppler_tolerance
    track_window = (
        2.0
        * options.harmonic_phase_distance
        * float(_PHASE_DEGREE * _PHASE_DEGREE)
        / tobs
    )

    roots: list[int] = []
    for h in range(n_clusters):
        fh = float(rep_derivs[h, 0])
        snr_h = float(rep_snrs[h])
        duty_h = float(rep_duties[h])
        best_order = 0
        best_parent = -1
        best_pq: tuple[int, int] | None = None

        for order_val, r_val, p_val, q_val in ratios:
            if best_pq is not None and order_val > best_order:
                break
            lo_freq = (fh - track_window) / r_val
            hi_freq = (fh + track_window) / r_val
            if doppler_tol > 0.0:
                target_c = fh / r_val
                lo_freq = min(lo_freq, (target_c / (1.0 + doppler_tol)) * (1.0 - 1e-12))
                hi_freq = max(hi_freq, (target_c / (1.0 - doppler_tol)) * (1.0 + 1e-12))
            for f in roots:
                if best_pq is not None and f >= best_parent:
                    continue
                fc_freq = float(rep_derivs[f, 0])
                if fc_freq < lo_freq or fc_freq > hi_freq:
                    continue
                snr_f = float(rep_snrs[f])
                if snr_h > snr_f + options.harmonic_snr_distance:
                    continue
                target_f = r_val * fc_freq
                df = abs(fh - target_f)
                related = False
                if (
                    doppler_tol > 0.0
                    and snr_f >= options.harmonic_doppler_min_snr
                    and df <= doppler_tol * target_f * (1.0 + 1e-12)
                ):
                    ratio_obs = fh / fc_freq
                    v_lo = (ratio_obs / (1.0 + doppler_tol)) * (1.0 - 1e-12)
                    v_hi = (ratio_obs / (1.0 - doppler_tol)) * (1.0 + 1e-12)
                    i_lo = int(np.searchsorted(sorted_vals, v_lo, side="left"))
                    i_hi = int(np.searchsorted(sorted_vals, v_hi, side="right"))
                    in_window = 0
                    for vi in range(i_lo, i_hi):
                        vf = float(sorted_vals[vi]) * fc_freq
                        if abs(fh - vf) <= doppler_tol * vf * (1.0 + 1e-12):
                            in_window += 1
                            if in_window > 1:
                                break
                    related = in_window == 1
                if not related:
                    duty_fast = duty_h if fh >= fc_freq else float(rep_duties[f])
                    limit = options.harmonic_phase_distance * duty_fast
                    related = (
                        df * (0.5 * tobs)
                        <= limit * float(_PHASE_DEGREE * _PHASE_DEGREE)
                        and phase_span(
                            rep_derivs[f],
                            rep_derivs[h],
                            tobs,
                            scale_a=r_val,
                            reject_above=limit,
                        )
                        <= limit
                    )
                if related:
                    best_order = order_val
                    best_parent = f
                    best_pq = (p_val, q_val)

        if best_pq is not None:
            parents[h] = best_parent
            nums[h], dens[h] = best_pq
        else:
            roots.append(h)

    return parents, nums, dens


def _cluster_anchored(
    derivs: np.ndarray,
    duties: np.ndarray,
    order: np.ndarray,
    tobs: float,
    opts: CandidateClusterOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = order.size
    rep_indices: list[int] = []
    counts: list[int] = []
    extents: list[float] = []
    cluster_ids = np.full(n, -1, dtype=np.int64)
    for pos in range(n):
        i = int(order[pos])
        fi = float(derivs[i, 0])
        wi = float(duties[i])
        reach = _link_reach(max(opts.radius * wi, opts.phase_radius), tobs)
        match = -1
        for cid, rep_idx in enumerate(rep_indices):
            if abs(float(derivs[rep_idx, 0]) - fi) > reach:
                continue
            limit = max(
                opts.phase_radius, opts.radius * min(float(duties[rep_idx]), wi)
            )
            if (
                phase_span(derivs[rep_idx], derivs[i], tobs, reject_above=limit)
                <= limit
            ):
                match = cid
                break
        if match >= 0:
            cluster_ids[i] = match
            counts[match] += 1
            extents[match] = max(
                extents[match],
                phase_span(derivs[rep_indices[match]], derivs[i], tobs),
            )
        else:
            cid = len(rep_indices)
            cluster_ids[i] = cid
            rep_indices.append(i)
            counts.append(1)
            extents.append(0.0)
    return (
        np.asarray(rep_indices, dtype=np.int64),
        cluster_ids,
        np.asarray(counts, dtype=np.int64),
        np.asarray(extents, dtype=np.float64),
    )


def _cluster_transitive(
    derivs: np.ndarray,
    duties: np.ndarray,
    order: np.ndarray,
    tobs: float,
    opts: CandidateClusterOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = order.size
    node_rep: list[int] = []
    node_duty: list[float] = []
    node_parent: list[int] = []
    node_count: list[int] = []
    track_to_node: dict[tuple[float, ...], int] = {}
    prov_node = np.empty(n, dtype=np.int64)

    def _find_root(u: int) -> int:
        while node_parent[u] != u:
            node_parent[u] = node_parent[node_parent[u]]
            u = node_parent[u]
        return u

    for pos in range(n):
        i = int(order[pos])
        key = tuple(float(v) for v in derivs[i])
        wi = float(duties[i])
        if key in track_to_node:
            u = track_to_node[key]
            prov_node[i] = u
            node_count[_find_root(u)] += 1
            if wi <= node_duty[u]:
                continue
            node_duty[u] = wi
        else:
            u = len(node_rep)
            node_rep.append(i)
            node_duty.append(wi)
            node_parent.append(u)
            node_count.append(1)
            track_to_node[key] = u
            prov_node[i] = u

        fi = float(derivs[i, 0])
        reach = _link_reach(max(opts.phase_radius, opts.radius * node_duty[u]), tobs)
        for other in range(len(node_rep)):
            if other == u:
                continue
            if abs(float(derivs[node_rep[other], 0]) - fi) > reach:
                continue
            ra = _find_root(u)
            rb = _find_root(other)
            if ra == rb:
                continue
            limit = max(
                opts.phase_radius,
                opts.radius * min(node_duty[u], node_duty[other]),
            )
            if (
                phase_span(
                    derivs[i],
                    derivs[node_rep[other]],
                    tobs,
                    reject_above=limit,
                )
                <= limit
            ):
                if ra > rb:
                    ra, rb = rb, ra
                node_parent[rb] = ra
                node_count[ra] += node_count[rb]

    n_nodes = len(node_rep)
    remap = np.empty(n_nodes, dtype=np.int64)
    rep_indices: list[int] = []
    counts: list[int] = []
    for u in range(n_nodes):
        if _find_root(u) == u:
            remap[u] = len(rep_indices)
            rep_indices.append(node_rep[u])
            counts.append(node_count[u])
    extents_arr = np.zeros(len(rep_indices), dtype=np.float64)
    for u in range(n_nodes):
        cid = int(remap[_find_root(u)])
        remap[u] = cid
        extents_arr[cid] = max(
            extents_arr[cid],
            phase_span(derivs[rep_indices[cid]], derivs[node_rep[u]], tobs),
        )
    return (
        np.asarray(rep_indices, dtype=np.int64),
        remap[prov_node],
        np.asarray(counts, dtype=np.int64),
        extents_arr,
    )


def cluster_candidates_single_dm(
    collection: CandidateCollection,
    tobs: float,
    *,
    options: CandidateClusterOptions | None = None,
) -> SingleDMClusterResult:
    """Cluster single-DM candidates by exact constant-phase-invariant track span.

    Supports both ``"transitive"`` (union-find over phase-track links) and
    ``"anchored"`` (representative-anchored) grouping over ``[-tobs/2, tobs/2]``,
    and optional self-harmonic family flagging.
    """
    opts = options or CandidateClusterOptions()
    opts.validate()
    if not math.isfinite(tobs) or tobs <= 0.0:
        msg = "tobs must be finite and positive"
        raise ValueError(msg)

    n = len(collection)
    if n == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return SingleDMClusterResult(
            representatives=collection,
            cluster_ids=empty_i,
            member_counts=empty_i,
            phase_extents=np.empty(0, dtype=np.float64),
            parent_cluster_ids=empty_i,
            harmonic_numerators=empty_i,
            harmonic_denominators=empty_i,
            selected=np.empty(0, dtype=np.bool_),
        )

    derivs = _collection_derivatives(collection)
    duties = collection.width.astype(np.float64) / np.maximum(
        collection.nbins.astype(np.float64),
        1.0,
    )
    order = np.argsort(-collection.snr, kind="mergesort")

    if opts.grouping == "anchored":
        rep_arr, cluster_ids, counts_arr, extents_arr = _cluster_anchored(
            derivs,
            duties,
            order,
            tobs,
            opts,
        )
    else:
        rep_arr, cluster_ids, counts_arr, extents_arr = _cluster_transitive(
            derivs,
            duties,
            order,
            tobs,
            opts,
        )

    n_clusters = rep_arr.size
    if opts.flag_harmonics:
        parent_ids, h_nums, h_dens = _flag_cluster_harmonics(
            derivs[rep_arr],
            duties[rep_arr],
            collection.snr[rep_arr],
            tobs,
            opts,
        )
    else:
        parent_ids = np.full(n_clusters, -1, dtype=np.int64)
        h_nums = np.zeros(n_clusters, dtype=np.int64)
        h_dens = np.zeros(n_clusters, dtype=np.int64)

    selected = np.ones(n_clusters, dtype=np.bool_)
    n_sel = 0
    for cid in range(n_clusters):
        keep = not (opts.remove_harmonics and parent_ids[cid] >= 0)
        if keep and opts.max_candidates > 0 and n_sel >= opts.max_candidates:
            keep = False
        selected[cid] = keep
        if keep:
            n_sel += 1

    reps = CandidateCollection(
        ids=collection.ids[rep_arr],
        param_sets=collection.param_sets[rep_arr],
        param_names=collection.param_names,
        snr=collection.snr[rep_arr],
        width=collection.width[rep_arr],
        nbins=collection.nbins[rep_arr],
        f0=collection.f0[rep_arr],
        f1=collection.f1[rep_arr],
        f2=collection.f2[rep_arr],
        dm=collection.dm[rep_arr],
        metadata=dict(collection.metadata),
        cluster_ids=np.arange(n_clusters, dtype=np.int64),
    )
    return SingleDMClusterResult(
        representatives=reps,
        cluster_ids=cluster_ids,
        member_counts=counts_arr,
        phase_extents=extents_arr,
        parent_cluster_ids=parent_ids,
        harmonic_numerators=h_nums,
        harmonic_denominators=h_dens,
        selected=selected,
    )


# ---------------------------------------------------------------------------
# Known-Pulsar Rational Harmonic Matching
# ---------------------------------------------------------------------------


@attrs.frozen(auto_attribs=True, kw_only=True)
class HarmonicMatch:
    """Result of matching a candidate period against a known-pulsar catalog."""

    matched: bool
    pulsar_name: str | None = None
    k: int = 0
    m: int = 0
    rel_err: float = float("inf")
    tier: StrengthTier | None = None


def _coprime_ratios(k_max: int) -> list[tuple[float, int, int]]:
    return [
        (k / m, k, m)
        for k in range(1, k_max + 1)
        for m in range(1, k_max + 1)
        if math.gcd(k, m) == 1
    ]


def _normalize_known_periods(
    known_periods: KnownPeriodsInput,
    *,
    default_tier: StrengthTier = "weak",
) -> list[tuple[str, float, StrengthTier]]:
    if isinstance(known_periods, Mapping):
        return [
            (str(name), float(p), default_tier) for name, p in known_periods.items()
        ]
    entries: list[tuple[str, float, StrengthTier]] = []
    for idx, item in enumerate(known_periods):
        if isinstance(item, tuple):
            if len(item) == 2:
                entries.append((str(item[0]), float(item[1]), default_tier))
            elif len(item) == 3:
                entries.append((str(item[0]), float(item[1]), item[2]))
            else:
                msg = f"Expected (name, period_s[, tier]) tuple, got length {len(item)}"
                raise ValueError(msg)
        else:
            entries.append((f"psr_{idx}", float(item), default_tier))
    return entries


class HarmonicMatcher:
    """Fast O(log M) binary-search rational harmonic matcher against known periods."""

    def __init__(
        self,
        known_periods: KnownPeriodsInput,
        *,
        default_tier: StrengthTier = "weak",
        k_max_strong: int = 15,
        k_max_medium: int = 3,
        k_max_weak: int = 3,
    ) -> None:
        normalized = _normalize_known_periods(known_periods, default_tier=default_tier)
        tier_kmax: dict[StrengthTier, int] = {
            "strong": k_max_strong,
            "medium": k_max_medium,
            "weak": k_max_weak,
        }
        ratio_cache = {kmax: _coprime_ratios(kmax) for kmax in set(tier_kmax.values())}

        entries: list[tuple[float, str, int, int, StrengthTier]] = []
        for name, period_s, tier in normalized:
            kmax = tier_kmax[tier]
            for ratio, k, m in ratio_cache[kmax]:
                entries.append((period_s * ratio, name, k, m, tier))

        entries.sort(key=lambda item: item[0])
        self._periods = np.array([e[0] for e in entries], dtype=np.float64)
        self._names = tuple(e[1] for e in entries)
        self._ks = np.array([e[2] for e in entries], dtype=np.int64)
        self._ms = np.array([e[3] for e in entries], dtype=np.int64)
        self._tiers: tuple[StrengthTier, ...] = tuple(e[4] for e in entries)

    def match_period(
        self,
        period_s: float,
        *,
        rel_tol: float = 5e-4,
    ) -> HarmonicMatch:
        """Match a candidate spin period ``period_s`` against precomputed harmonics."""
        if self._periods.size == 0 or period_s <= 0.0:
            return HarmonicMatch(matched=False)
        lo_val = period_s * (1.0 - rel_tol)
        hi_val = period_s * (1.0 + rel_tol)
        lo_idx = int(np.searchsorted(self._periods, lo_val, side="left"))
        hi_idx = int(np.searchsorted(self._periods, hi_val, side="right"))
        if lo_idx >= hi_idx:
            return HarmonicMatch(matched=False)

        window = self._periods[lo_idx:hi_idx]
        errs = np.abs(window - period_s) / window
        best_local = int(np.argmin(errs))
        best_err = float(errs[best_local])
        if best_err > rel_tol:
            return HarmonicMatch(matched=False)
        idx = lo_idx + best_local
        return HarmonicMatch(
            matched=True,
            pulsar_name=self._names[idx],
            k=int(self._ks[idx]),
            m=int(self._ms[idx]),
            rel_err=best_err,
            tier=self._tiers[idx],
        )

    def match_frequency(
        self,
        freq_hz: float,
        *,
        rel_tol: float = 5e-4,
    ) -> HarmonicMatch:
        if freq_hz <= 0.0:
            return HarmonicMatch(matched=False)
        return self.match_period(1.0 / freq_hz, rel_tol=rel_tol)


def match_known_harmonics(
    period_or_freq: float,
    known_periods: KnownPeriodsInput,
    *,
    is_frequency: bool = False,
    default_tier: StrengthTier = "weak",
    rel_tol: float = 5e-4,
    k_max_strong: int = 15,
    k_max_medium: int = 3,
    k_max_weak: int = 3,
) -> HarmonicMatch:
    """Check whether a candidate period or frequency is a rational harmonic ``k/m``."""
    matcher = HarmonicMatcher(
        known_periods,
        default_tier=default_tier,
        k_max_strong=k_max_strong,
        k_max_medium=k_max_medium,
        k_max_weak=k_max_weak,
    )
    if is_frequency:
        return matcher.match_frequency(period_or_freq, rel_tol=rel_tol)
    return matcher.match_period(period_or_freq, rel_tol=rel_tol)


# ---------------------------------------------------------------------------
# Multi-DM Survey Candidate Grouping & Persistence
# ---------------------------------------------------------------------------


@attrs.define(auto_attribs=True, kw_only=True)
class SurveyCandidateGroup:
    """Multi-DM frequency track with SNR(DM) summary and persistent annotations."""

    group_id: int
    f0_peak: float
    p0_peak: float
    peak_snr: float
    dm_peak: float
    dm_min: float
    dm_max: float
    dm_span: float
    n_dms: int
    has_dm0_hit: bool
    dm0_snr: float
    dms: npt.NDArray[np.float64]
    snrs: npt.NDArray[np.float64]
    member_indices: npt.NDArray[np.int64]
    disposition: Disposition = "new"
    notes: str = ""
    harmonic_match: HarmonicMatch | None = None

    def to_dict(self) -> dict[str, Any]:
        hm = None
        if self.harmonic_match is not None:
            hm = {
                "matched": self.harmonic_match.matched,
                "pulsar_name": self.harmonic_match.pulsar_name,
                "k": self.harmonic_match.k,
                "m": self.harmonic_match.m,
                "rel_err": self.harmonic_match.rel_err,
                "tier": self.harmonic_match.tier,
            }
        return {
            "group_id": self.group_id,
            "f0_peak": self.f0_peak,
            "p0_peak": self.p0_peak,
            "peak_snr": self.peak_snr,
            "dm_peak": self.dm_peak,
            "dm_min": self.dm_min,
            "dm_max": self.dm_max,
            "dm_span": self.dm_span,
            "n_dms": self.n_dms,
            "has_dm0_hit": self.has_dm0_hit,
            "dm0_snr": self.dm0_snr,
            "dms": self.dms.tolist(),
            "snrs": self.snrs.tolist(),
            "member_indices": self.member_indices.tolist(),
            "disposition": self.disposition,
            "notes": self.notes,
            "harmonic_match": hm,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurveyCandidateGroup:
        hm_raw = data.get("harmonic_match")
        hm = HarmonicMatch(**hm_raw) if isinstance(hm_raw, dict) else None
        return cls(
            group_id=int(data["group_id"]),
            f0_peak=float(data["f0_peak"]),
            p0_peak=float(data["p0_peak"]),
            peak_snr=float(data["peak_snr"]),
            dm_peak=float(data["dm_peak"]),
            dm_min=float(data["dm_min"]),
            dm_max=float(data["dm_max"]),
            dm_span=float(data["dm_span"]),
            n_dms=int(data["n_dms"]),
            has_dm0_hit=bool(data["has_dm0_hit"]),
            dm0_snr=float(data["dm0_snr"]),
            dms=np.asarray(data["dms"], dtype=np.float64),
            snrs=np.asarray(data["snrs"], dtype=np.float64),
            member_indices=np.asarray(data["member_indices"], dtype=np.int64),
            disposition=data.get("disposition", "new"),
            notes=str(data.get("notes", "")),
            harmonic_match=hm,
        )


@attrs.define(auto_attribs=True, kw_only=True)
class SurveyCandidateGroups:
    """Multi-DM survey sifting container with DM=0 RFI filtering and persistence."""

    groups: list[SurveyCandidateGroup] = attrs.Factory(list)

    def __len__(self) -> int:
        return len(self.groups)

    @classmethod
    def from_collection(
        cls,
        collection: CandidateCollection,
        *,
        rel_freq_tol: float = 1e-3,
        dm0_tol: float = 1e-6,
    ) -> SurveyCandidateGroups:
        """Partition candidates across DMs into frequency tracks."""
        n = len(collection)
        if n == 0:
            return cls(groups=[])

        order = np.argsort(collection.f0)
        raw_groups: list[list[int]] = [[int(order[0])]]
        anchor_f0 = float(collection.f0[order[0]])
        for pos in range(1, n):
            idx = int(order[pos])
            f_cur = float(collection.f0[idx])
            if abs(f_cur - anchor_f0) <= rel_freq_tol * max(abs(anchor_f0), 1e-12):
                raw_groups[-1].append(idx)
            else:
                raw_groups.append([idx])
                anchor_f0 = f_cur

        groups: list[SurveyCandidateGroup] = []
        for members in raw_groups:
            m_arr = np.asarray(members, dtype=np.int64)
            m_snrs = collection.snr[m_arr]
            m_dms = collection.dm[m_arr]
            m_f0s = collection.f0[m_arr]
            best_local = int(np.argmax(m_snrs))
            f0_peak = float(m_f0s[best_local])
            dm_peak = float(m_dms[best_local])
            peak_snr = float(m_snrs[best_local])

            # Build per-unique-DM peak SNR(DM) curve
            dm_order = np.argsort(m_dms)
            sorted_dms = m_dms[dm_order]
            sorted_snrs = m_snrs[dm_order]
            uniq_dms, inv = np.unique(sorted_dms, return_inverse=True)
            curve_snrs = np.full(uniq_dms.size, -np.inf, dtype=np.float64)
            np.maximum.at(curve_snrs, inv, sorted_snrs)

            dm0_mask = np.abs(uniq_dms) <= dm0_tol
            has_dm0 = bool(np.any(dm0_mask))
            dm0_snr = float(np.max(curve_snrs[dm0_mask])) if has_dm0 else 0.0

            groups.append(
                SurveyCandidateGroup(
                    group_id=0,
                    f0_peak=f0_peak,
                    p0_peak=1.0 / f0_peak if f0_peak > 0.0 else float("inf"),
                    peak_snr=peak_snr,
                    dm_peak=dm_peak,
                    dm_min=float(uniq_dms[0]),
                    dm_max=float(uniq_dms[-1]),
                    dm_span=float(uniq_dms[-1] - uniq_dms[0]),
                    n_dms=int(uniq_dms.size),
                    has_dm0_hit=has_dm0,
                    dm0_snr=dm0_snr,
                    dms=uniq_dms,
                    snrs=curve_snrs,
                    member_indices=m_arr,
                ),
            )

        # Rank groups descending by peak S/N and assign stable group IDs
        groups.sort(key=lambda g: g.peak_snr, reverse=True)
        for gid, grp in enumerate(groups):
            grp.group_id = gid
        return cls(groups=groups)

    def filter_dm0_and_span(
        self,
        *,
        rfi_ratio: float = 0.5,
        max_dm_width: float | None = None,
        survey_dm_min: float = 0.0,
    ) -> list[SurveyCandidateGroup]:
        """Flag broadband DM=0 or unbounded-DM-span groups as ``"rfi"``.

        Returns the list of non-RFI groups.
        """
        survivors: list[SurveyCandidateGroup] = []
        for grp in self.groups:
            is_dm0_rfi = grp.has_dm0_hit and (grp.dm0_snr >= rfi_ratio * grp.peak_snr)
            is_wide_rfi = (
                max_dm_width is not None
                and grp.dm_span > max_dm_width
                and grp.dm_min <= survey_dm_min + 1e-6
            )
            if is_dm0_rfi or is_wide_rfi:
                grp.disposition = "rfi"
            else:
                survivors.append(grp)
        return survivors

    def tag_known_harmonics(
        self,
        known_periods: KnownPeriodsInput,
        *,
        default_tier: StrengthTier = "weak",
        rel_tol: float = 5e-4,
        k_max_strong: int = 15,
        k_max_medium: int = 3,
        k_max_weak: int = 3,
    ) -> int:
        """Annotate groups matching known pulsars or their rational harmonics."""
        matcher = HarmonicMatcher(
            known_periods,
            default_tier=default_tier,
            k_max_strong=k_max_strong,
            k_max_medium=k_max_medium,
            k_max_weak=k_max_weak,
        )
        n_matched = 0
        for grp in self.groups:
            hm = matcher.match_frequency(grp.f0_peak, rel_tol=rel_tol)
            grp.harmonic_match = hm
            if hm.matched:
                if grp.disposition == "new":
                    grp.disposition = "known_harmonic"
                n_matched += 1
        return n_matched

    def set_annotation(
        self,
        group_id: int,
        *,
        disposition: Disposition | None = None,
        notes: str | None = None,
    ) -> None:
        for grp in self.groups:
            if grp.group_id == group_id:
                if disposition is not None:
                    grp.disposition = disposition
                if notes is not None:
                    grp.notes = notes
                return
        msg = f"Group ID {group_id} not found"
        raise KeyError(msg)

    def save_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"groups": [g.to_dict() for g in self.groups]}
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> SurveyCandidateGroups:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            groups=[SurveyCandidateGroup.from_dict(g) for g in payload["groups"]],
        )

    def save_hdf5(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"groups": [g.to_dict() for g in self.groups]})
        with h5py.File(out, "w") as handle:
            handle.attrs["format"] = "pyloki_survey_groups_v1"
            handle.create_dataset(
                "groups_json",
                data=np.bytes_(payload.encode("utf-8")),
            )

    @classmethod
    def load_hdf5(cls, path: str | Path) -> SurveyCandidateGroups:
        with h5py.File(Path(path), "r") as handle:
            raw = bytes(handle["groups_json"][()]).decode("utf-8")
        payload = json.loads(raw)
        return cls(
            groups=[SurveyCandidateGroup.from_dict(g) for g in payload["groups"]],
        )


# ---------------------------------------------------------------------------
# Arbitrary-Order Phase Models, Segment Ephemeris & TEMPO2 Predictor Export
# ---------------------------------------------------------------------------


PhaseModelKind = Literal["taylor", "chebyshev", "circular", "keplerian", "callable"]


def build_phase_model(
    model: PhaseModelKind,
    params: Sequence[float] | None = None,
    *,
    t_ref: float = 0.0,
    t_span: tuple[float, float] | None = None,
    phase_func: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Construct a vectorized ``phase_cycles(t_sec)`` evaluator for any search basis.

    Parameters
    ----------
    model : {"taylor", "chebyshev", "circular", "keplerian", "callable"}
        Trajectory parameterization:
        - ``"taylor"``: ``[..., crackle, snap, jerk, accel, freq]`` kinematic Taylor.
        - ``"chebyshev"``: Chebyshev polynomial coefficients of ``f(t)`` on ``t_span``.
        - ``"circular"``: ``[omega, freq, x_sin_phi, x_cos_phi]``.
        - ``"keplerian"``: ``[f0, p_orb, x_orb, ecc, aop, phi]``.
        - ``"callable"``: custom ``phase_func(t_sec) -> phase_cycles``.
    params : sequence of float | None
        Model parameters (required unless ``model == "callable"``).
    t_ref : float, optional
        Reference time in seconds where ``phase(t_ref) == 0``, by default 0.0.
    t_span : tuple[float, float] | None, optional
        Domain ``(t_start, t_end)`` in seconds (required for ``"chebyshev"``).
    phase_func : callable | None, optional
        Custom vectorized phase function when ``model == "callable"``.
    """
    if model == "callable":
        if phase_func is None:
            msg = "phase_func must be provided when model='callable'"
            raise ValueError(msg)
        return phase_func

    if params is None:
        msg = f"params must be provided for model={model!r}"
        raise ValueError(msg)
    p = np.asarray(params, dtype=np.float64)

    if model == "taylor":
        f0 = float(p[-1])
        kin = p[:-1][::-1]  # [accel, jerk, snap, crackle, ...]

        def _taylor_phase(t: np.ndarray) -> np.ndarray:
            dt = np.asarray(t, dtype=np.float64) - t_ref
            phi = f0 * dt
            for order_idx, a_k in enumerate(kin, start=1):
                phi = phi - (
                    f0 * float(a_k) / (C_VAL * math.factorial(order_idx + 1))
                ) * (dt ** (order_idx + 1))
            return phi

        return _taylor_phase

    if model == "chebyshev":
        if t_span is None or t_span[1] <= t_span[0]:
            msg = (
                "t_span=(t_start, t_end) with t_end > t_start is required "
                "for model='chebyshev'"
            )
            raise ValueError(msg)
        t_lo, t_hi = t_span
        half_span = 0.5 * (t_hi - t_lo)
        mid = 0.5 * (t_lo + t_hi)
        # Integrate frequency Chebyshev expansion on [-1, 1] and scale by half_span
        int_coeffs = np_cheb.chebint(p) * half_span
        u_ref = (t_ref - mid) / half_span
        phi_ref = float(np_cheb.chebval(u_ref, int_coeffs))

        def _cheb_phase(t: np.ndarray) -> np.ndarray:
            u = (np.asarray(t, dtype=np.float64) - mid) / half_span
            return np_cheb.chebval(u, int_coeffs) - phi_ref

        return _cheb_phase

    if model == "circular":
        if p.size != 4:
            msg = "circular params must be [omega, freq, x_sin_phi, x_cos_phi]"
            raise ValueError(msg)
        omega, f0, x_sin_phi, x_cos_phi = (float(v) for v in p)
        z0_c = x_sin_phi

        def _circ_phase(t: np.ndarray) -> np.ndarray:
            dt = np.asarray(t, dtype=np.float64) - t_ref
            z_c = x_cos_phi * np.sin(omega * dt) + x_sin_phi * np.cos(omega * dt)
            return f0 * (dt - (z_c - z0_c))

        return _circ_phase

    if model == "keplerian":
        if p.size != 6:
            msg = "keplerian params must be [f0, p_orb, x_orb, ecc, aop, phi]"
            raise ValueError(msg)
        f0, p_orb, x_orb, ecc, aop, phi0 = (float(v) for v in p)
        z_ref = float(
            kepler.keplerian_z(
                np.array([t_ref], dtype=np.float64),
                p_orb,
                ecc,
                phi0,
                x_orb,
                aop,
                0.5 * math.pi,
            )[0],
        )

        def _kep_phase(t: np.ndarray) -> np.ndarray:
            t_arr = np.asarray(t, dtype=np.float64)
            z_sec = kepler.keplerian_z(
                t_arr,
                p_orb,
                ecc,
                phi0,
                x_orb,
                aop,
                0.5 * math.pi,
            )
            return f0 * ((t_arr - t_ref) - (z_sec - z_ref))

        return _kep_phase

    msg = f"Unsupported phase model: {model!r}"
    raise ValueError(msg)


def export_segment_ephemeris(
    output: str | Path,
    *,
    model: PhaseModelKind,
    params: Sequence[float] | None = None,
    ref_mjd: float,
    t_start: float,
    t_end: float,
    n_segments: int,
    t_ref: float = 0.0,
    phase_func: Callable[[np.ndarray], np.ndarray] | None = None,
    fmt: Literal["json", "csv"] = "json",
) -> list[dict[str, float]]:
    """Sample a candidate trajectory at ``n_segments`` midpoints and export JSON/CSV.

    Emits ``(mjd, t_offset_s, freq_hz, period_s, f1_hz_s, accel_mss, phase_cycles)``
    with explicit MJD timestamps and SI units.
    """
    if n_segments < 1 or t_end <= t_start:
        msg = "n_segments must be >= 1 and t_end > t_start"
        raise ValueError(msg)
    phi_fn = build_phase_model(
        model,
        params,
        t_ref=t_ref,
        t_span=(t_start, t_end),
        phase_func=phase_func,
    )
    dt_seg = (t_end - t_start) / n_segments
    t_mid = t_start + (np.arange(n_segments, dtype=np.float64) + 0.5) * dt_seg
    eps = max(1e-4, 1e-4 * dt_seg)

    phi_0 = phi_fn(t_mid)
    phi_p = phi_fn(t_mid + eps)
    phi_m = phi_fn(t_mid - eps)
    freq_hz = (phi_p - phi_m) / (2.0 * eps)
    f1_hz_s = (phi_p - 2.0 * phi_0 + phi_m) / (eps * eps)

    rows: list[dict[str, float]] = []
    for i in range(n_segments):
        f_i = float(freq_hz[i])
        f1_i = float(f1_hz_s[i])
        rows.append(
            {
                "mjd": float(ref_mjd + (t_mid[i] - t_ref) / SECONDS_PER_DAY),
                "t_offset_s": float(t_mid[i]),
                "freq_hz": f_i,
                "period_s": 1.0 / f_i if f_i > 0.0 else float("nan"),
                "f1_hz_s": f1_i,
                "accel_mss": -f1_i / f_i * C_VAL if f_i > 0.0 else float("nan"),
                "phase_cycles": float(phi_0[i]),
            },
        )

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        payload = {
            "model": model,
            "ref_mjd": ref_mjd,
            "t_ref": t_ref,
            "t_start": t_start,
            "t_end": t_end,
            "segments": rows,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif fmt == "csv":
        fieldnames = [
            "mjd",
            "t_offset_s",
            "freq_hz",
            "period_s",
            "f1_hz_s",
            "accel_mss",
            "phase_cycles",
        ]
        with out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        msg = f"Unsupported export format {fmt!r}; expected 'json' or 'csv'"
        raise ValueError(msg)
    return rows


def _fit_cheby_block(
    phi_fn: Callable[[np.ndarray], np.ndarray],
    mjd_lo: float,
    mjd_hi: float,
    ref_mjd: float,
    t_ref: float,
    ncoeff: int,
) -> tuple[np.ndarray, float]:
    """Fit a 1D Chebyshev polynomial on Chebyshev-Gauss-Lobatto nodes."""
    k = np.arange(ncoeff, dtype=np.float64)
    u_nodes = np.cos(math.pi * k / max(ncoeff - 1, 1))
    t_lo = t_ref + (mjd_lo - ref_mjd) * SECONDS_PER_DAY
    t_hi = t_ref + (mjd_hi - ref_mjd) * SECONDS_PER_DAY
    t_mid = 0.5 * (t_lo + t_hi)
    t_half = 0.5 * (t_hi - t_lo)
    t_nodes = t_mid + t_half * u_nodes
    y_nodes = phi_fn(t_nodes)
    coeffs = np_cheb.chebfit(u_nodes, y_nodes, deg=ncoeff - 1)

    # Verify max residual on a dense interior grid
    u_check = np.linspace(-1.0, 1.0, max(4 * ncoeff, 65), dtype=np.float64)
    t_check = t_mid + t_half * u_check
    err = float(np.max(np.abs(np_cheb.chebval(u_check, coeffs) - phi_fn(t_check))))
    return coeffs, err


def export_tempo2_predictor(
    output: str | Path,
    *,
    model: PhaseModelKind,
    params: Sequence[float] | None = None,
    ref_mjd: float,
    mjd_start: float,
    mjd_end: float,
    t_ref: float = 0.0,
    phase_func: Callable[[np.ndarray], np.ndarray] | None = None,
    psr_name: str = "CAND",
    site_name: str = "coe",
    freq_range_mhz: tuple[float, float] = (1000.0, 2000.0),
    dispersion_constant: float = 0.0,
    ncoeff: int = 15,
    max_phase_err: float = 1e-4,
    max_blocks: int = 256,
) -> dict[str, Any]:
    """Export an arbitrary-order candidate trajectory as a TEMPO2 ``t2pred.dat`` file.

    Automatically subdivides ``[mjd_start, mjd_end]`` into Chebyshev time blocks
    (``ncoeff <= 15`` per block, TEMPO2 Clenshaw convention with ``c_0 = 2 * a_0``)
    until the maximum phase error across every block is below ``max_phase_err``
    cycles (default ``1e-4``).
    """
    if mjd_end <= mjd_start:
        msg = "mjd_end must be greater than mjd_start"
        raise ValueError(msg)
    if not (2 <= ncoeff <= 15):
        msg = f"ncoeff must be in [2, 15], got {ncoeff}"
        raise ValueError(msg)

    t_start = t_ref + (mjd_start - ref_mjd) * SECONDS_PER_DAY
    t_end = t_ref + (mjd_end - ref_mjd) * SECONDS_PER_DAY
    phi_fn = build_phase_model(
        model,
        params,
        t_ref=t_ref,
        t_span=(t_start, t_end),
        phase_func=phase_func,
    )

    blocks: list[tuple[float, float, np.ndarray, float]] = []
    queue: list[tuple[float, float]] = [(mjd_start, mjd_end)]
    while queue:
        lo, hi = queue.pop(0)
        coeffs, err = _fit_cheby_block(phi_fn, lo, hi, ref_mjd, t_ref, ncoeff)
        if err > max_phase_err and (len(blocks) + len(queue) + 2) <= max_blocks:
            mid = 0.5 * (lo + hi)
            queue.insert(0, (mid, hi))
            queue.insert(0, (lo, mid))
        else:
            blocks.append((lo, hi, coeffs, err))

    blocks.sort(key=lambda b: b[0])
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"ChebyModelSet {len(blocks)}"]
    for lo, hi, coeffs, _ in blocks:
        # TEMPO2 Cheby2D uses phi(x) = 0.5 * c_0 + sum_{k>=1} c_k T_k(x)
        t2_coeffs = coeffs.copy()
        t2_coeffs[0] *= 2.0
        lines.extend(
            [
                "ChebyModel BEGIN",
                f"PSRNAME {psr_name}",
                f"SITENAME {site_name}",
                f"TIME_RANGE {lo:.17g} {hi:.17g}",
                f"FREQ_RANGE {freq_range_mhz[0]:.6f} {freq_range_mhz[1]:.6f}",
                f"DISPERSION_CONSTANT {dispersion_constant:.17g}",
                f"NCOEFF_TIME {ncoeff}",
                "NCOEFF_FREQ 1",
                *(f"COEFFS {c:.17g}" for c in t2_coeffs),
                "ChebyModel END",
            ],
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    max_err = max(b[3] for b in blocks)
    return {
        "n_blocks": len(blocks),
        "ncoeff": ncoeff,
        "max_phase_err": max_err,
        "path": str(out),
    }


def evaluate_tempo2_predictor(
    path: str | Path,
    mjd: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Evaluate a TEMPO2 ``t2pred.dat`` Chebyshev predictor at arbitrary MJDs."""
    blocks: list[tuple[float, float, np.ndarray]] = []
    lo = hi = 0.0
    coeffs: list[float] = []
    in_block = False
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "ChebyModel BEGIN":
            in_block = True
            coeffs = []
        elif line == "ChebyModel END":
            arr = np.asarray(coeffs, dtype=np.float64)
            arr[0] *= 0.5  # Convert TEMPO2 Clenshaw c_0 back to standard Chebyshev a_0
            blocks.append((lo, hi, arr))
            in_block = False
        elif in_block:
            parts = line.split()
            if parts[0] == "TIME_RANGE":
                lo, hi = float(parts[1]), float(parts[2])
            elif parts[0] == "COEFFS":
                coeffs.append(float(parts[1]))

    if not blocks:
        msg = f"No ChebyModel blocks found in {path}"
        raise ValueError(msg)

    mjd_arr = np.asarray(mjd, dtype=np.float64)
    flat = mjd_arr.ravel()
    out = np.empty_like(flat)
    for idx, m in enumerate(flat):
        chosen = blocks[-1]
        for blk in blocks:
            if blk[0] <= m <= blk[1] or m < blk[0]:
                chosen = blk
                break
        b_lo, b_hi, b_coeffs = chosen
        u = ((m - b_lo) - (b_hi - m)) / (b_hi - b_lo)
        out[idx] = float(np_cheb.chebval(u, b_coeffs))
    return out.reshape(mjd_arr.shape)


# ---------------------------------------------------------------------------
# 2D Segment Waterfalls, S/N-Preserving Rebinning & 4-Panel Diagnostic Card
# ---------------------------------------------------------------------------


def bin2d(
    folds: npt.ArrayLike,
    bt: int = 1,
    bp: int = 1,
) -> npt.NDArray[np.float64]:
    """S/N-preserving 2D block rebinning by ``(bt, bp)`` along ``(time, phase)``.

    Sums non-overlapping ``bt x bp`` blocks and divides by ``sqrt(bt * bp)``
    so unit-variance Gaussian noise remains unit-variance after rebinning.
    """
    arr = np.asarray(folds, dtype=np.float64)
    if arr.ndim != 2:
        msg = f"folds must be 2D (n_time, n_phase), got shape {arr.shape}"
        raise ValueError(msg)
    if bt < 1 or bp < 1:
        msg = f"Rebin factors must be >= 1, got bt={bt}, bp={bp}"
        raise ValueError(msg)
    n_t, n_p = arr.shape
    eff_bt = min(bt, n_t)
    eff_bp = min(bp, n_p)
    trimmed = arr[: n_t - (n_t % eff_bt), : n_p - (n_p % eff_bp)]
    return trimmed.reshape(
        trimmed.shape[0] // eff_bt,
        eff_bt,
        trimmed.shape[1] // eff_bp,
        eff_bp,
    ).sum(axis=(1, 3)) / math.sqrt(float(eff_bt * eff_bp))


@attrs.frozen(auto_attribs=True, kw_only=True)
class SegmentWaterfall:
    """Aligned per-segment folds, unit-noise waterfall, and integrated profile."""

    segment_folds: npt.NDArray[np.float32]
    amp_map: npt.NDArray[np.float64]
    integrated_fold: npt.NDArray[np.float32]
    integrated_profile: npt.NDArray[np.float64]
    best_snr: float
    best_width: int


def _normalize_ev_row(e_row: np.ndarray, v_row: np.ndarray) -> np.ndarray:
    """Mean-subtracted unit-noise profile ``(E - E_bar * V / V_bar) / sqrt(V)``."""
    v = np.asarray(v_row, dtype=np.float64)
    e = np.asarray(e_row, dtype=np.float64)
    out = np.zeros_like(e, dtype=np.float64)
    valid = v > 0.0
    if not np.any(valid):
        return out
    v_sum = float(np.sum(v[valid]))
    e_sum = float(np.sum(e[valid]))
    mean_ratio = e_sum / v_sum if v_sum > 0.0 else 0.0
    out[valid] = (e[valid] - mean_ratio * v[valid]) / np.sqrt(v[valid])
    return out


def extract_segment_waterfall(
    folds: npt.NDArray[np.float32],
    *,
    a_indices: npt.ArrayLike | None = None,
    f_indices: npt.ArrayLike | None = None,
    phase_shifts: npt.ArrayLike | None = None,
    widths: npt.NDArray[np.int64] | None = None,
) -> SegmentWaterfall:
    """Extract an aligned 2D ``(segment x phase)`` S/N waterfall and integrated profile.

    Accepts either:
    - A 5D P-FFA fold cube ``(n_seg, n_acc, n_freq, 2, n_bins)`` together with
      ``a_indices`` and ``f_indices`` (and optional ``phase_shifts``), OR
    - Already-extracted 3D per-segment folds ``(n_seg, 2, n_bins)`` with
      optional ``phase_shifts``.
    """
    arr = np.asarray(folds, dtype=np.float32)
    if arr.ndim == 5:
        if a_indices is None or f_indices is None:
            msg = "a_indices and f_indices are required for 5D P-FFA fold cubes"
            raise ValueError(msg)
        a_idx = np.asarray(a_indices, dtype=np.int64)
        f_idx = np.asarray(f_indices, dtype=np.int64)
        n_seg = arr.shape[0]
        n_bins = arr.shape[4]
        seg_folds = np.empty((n_seg, 2, n_bins), dtype=np.float32)
        for i in range(n_seg):
            seg_folds[i] = arr[i, int(a_idx[i]), int(f_idx[i])]
    elif arr.ndim == 3 and arr.shape[1] == 2:
        seg_folds = arr.copy()
        n_seg, _, n_bins = seg_folds.shape
    else:
        msg = (
            "folds must have shape (n_seg, n_acc, n_freq, 2, n_bins) or "
            f"(n_seg, 2, n_bins), got {arr.shape}"
        )
        raise ValueError(msg)

    if phase_shifts is not None:
        shifts = np.asarray(phase_shifts, dtype=np.int64)
        if shifts.shape != (n_seg,):
            msg = f"phase_shifts must have shape ({n_seg},), got {shifts.shape}"
            raise ValueError(msg)
        for i in range(n_seg):
            if int(shifts[i]) != 0:
                seg_folds[i] = np.roll(seg_folds[i], int(shifts[i]), axis=-1)

    amp_map = np.empty((n_seg, n_bins), dtype=np.float64)
    for i in range(n_seg):
        amp_map[i] = _normalize_ev_row(seg_folds[i, 0], seg_folds[i, 1])

    integrated = np.sum(seg_folds, axis=0).astype(np.float32)
    integrated_prof = _normalize_ev_row(integrated[0], integrated[1])

    w_arr = (
        scoring.generate_box_width_trials(n_bins)
        if widths is None
        else np.ascontiguousarray(widths, dtype=np.int64)
    )
    snrs = scoring.boxcar_snr_glrt_1d(
        np.ascontiguousarray(integrated[0], dtype=np.float32),
        np.ascontiguousarray(integrated[1], dtype=np.float32),
        w_arr,
    )
    best_idx = int(np.argmax(snrs))
    return SegmentWaterfall(
        segment_folds=seg_folds,
        amp_map=amp_map,
        integrated_fold=integrated,
        integrated_profile=integrated_prof,
        best_snr=float(snrs[best_idx]),
        best_width=int(w_arr[best_idx]),
    )


def plot_candidate_diagnostic_card(
    waterfall: SegmentWaterfall,
    segment_times: npt.ArrayLike,
    freq_curve: npt.ArrayLike,
    *,
    phase_residuals_ms: npt.ArrayLike | None = None,
    runner_up_freqs: Sequence[npt.ArrayLike] | None = None,
    rebin: tuple[int, int] = (1, 1),
    title: str | None = None,
    fig: mpl_figure.Figure | None = None,
) -> mpl_figure.Figure:
    """Render a 4-panel candidate confirmation diagnostic card.

    Panels:
    - ``(0, 0)``: spin frequency curve ``f(t)`` (with optional runner-up tracks).
    - ``(0, 1)``: integrated unit-noise pulse profile with S/N and boxcar width.
    - ``(1, 0)``: detrended orbital arrival-time residual (ms).
    - ``(1, 1)``: S/N-preserving ``bin2d`` ``(time x phase)`` waterfall.
    """
    from matplotlib import pyplot as plt  # noqa: PLC0415

    t_arr = np.asarray(segment_times, dtype=np.float64)
    f_arr = np.asarray(freq_curve, dtype=np.float64)
    if fig is None:
        fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    axes = fig.subplots(2, 2)

    # (0, 0): Frequency track f(t)
    ax_f = axes[0, 0]
    if runner_up_freqs is not None:
        for rf in runner_up_freqs:
            ax_f.plot(t_arr, np.asarray(rf, dtype=np.float64), color="0.7", lw=1.0)
    ax_f.plot(t_arr, f_arr, color="tab:blue", lw=1.8, label="Best")
    ax_f.set_xlabel("Time (s)")
    ax_f.set_ylabel("Frequency (Hz)")
    ax_f.set_title("Spin Frequency Track")

    # (0, 1): Integrated profile
    ax_p = axes[0, 1]
    n_bins = waterfall.integrated_profile.size
    phase = np.linspace(0.0, 1.0, n_bins, endpoint=False)
    ax_p.step(phase, waterfall.integrated_profile, where="mid", color="tab:blue")
    ax_p.axhline(0.0, color="0.6", ls="--", lw=0.8)
    ax_p.set_xlabel("Pulse Phase")
    ax_p.set_ylabel("Normalized S/N")
    ax_p.set_title(
        f"Integrated Profile (S/N={waterfall.best_snr:.2f}, w={waterfall.best_width})",
    )

    # (1, 0): Detrended arrival-time residual (ms)
    ax_r = axes[1, 0]
    if phase_residuals_ms is not None:
        res_ms = np.asarray(phase_residuals_ms, dtype=np.float64)
    else:
        # Derive detrended timing residual from integral of (f(t) - f_mean) / f_mean
        f_mean = max(float(np.mean(f_arr)), 1e-12)
        dt = np.diff(t_arr, prepend=t_arr[0])
        tau_s = -np.cumsum((f_arr - f_mean) / f_mean * dt)
        res_ms = (tau_s - float(np.mean(tau_s))) * 1e3
    ax_r.plot(t_arr, res_ms, color="tab:orange", lw=1.5)
    ax_r.axhline(0.0, color="0.6", ls="--", lw=0.8)
    ax_r.set_xlabel("Time (s)")
    ax_r.set_ylabel("Delay Residual (ms)")
    ax_r.set_title("Detrended Orbital Phase Residual")

    # (1, 1): 2D (time x phase) waterfall
    ax_w = axes[1, 1]
    rebinned = bin2d(waterfall.amp_map, bt=rebin[0], bp=rebin[1])
    ax_w.imshow(
        rebinned,
        aspect="auto",
        origin="lower",
        extent=(0.0, 1.0, float(t_arr[0]), float(t_arr[-1])),
        cmap="viridis",
    )
    ax_w.set_xlabel("Pulse Phase")
    ax_w.set_ylabel("Time (s)")
    ax_w.set_title(f"Time-Phase Waterfall (rebin={rebin[0]}x{rebin[1]})")

    if title:
        fig.suptitle(title)
    return fig
