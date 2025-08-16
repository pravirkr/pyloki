from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

import attrs
import h5py
import numpy as np

if TYPE_CHECKING:
    from typing import ClassVar, Self


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
        The threshold value.
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
    score_min: float = 0.0
    score_max: float = 0.0
    n_branches: int = 1
    n_leaves: int = 1
    n_leaves_phy: int = 1
    n_leaves_surv: int = 1

    @property
    def lb_leaves(self) -> float:
        return np.round(np.log2(self.n_leaves), 2)

    @property
    def lb_leaves_phys(self) -> float:
        return np.round(np.log2(self.n_leaves_phy), 2)

    @property
    def branch_frac(self) -> float:
        return np.round(self.n_leaves_phy / self.n_branches, 2)

    @property
    def phys_frac(self) -> float:
        return np.round(self.n_leaves_phy / self.n_leaves, 2)

    @property
    def surv_frac(self) -> float:
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
            f"score thresh: {self.threshold:5.2f}, max: {self.score_max:5.2f}, "
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
        "transform",
        "threshold",
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
        return (
            f"Score: {self.stats_list[-1].score_max:.2f}, "
            f"Leaves: {self.stats_list[-1].lb_leaves:.2f}"
        )

    def get_timer_summary(self) -> str:
        """Get formatted timer summary for final level."""
        total_time = sum(self.timers.values())
        timer_percent = {
            name: (time / total_time) * 100 for name, time in self.timers.items()
        }
        lines = [f"Timing breakdown: {total_time:.2f}s"]
        lines.extend(
            f"  {name:10s}: {value:6.1f}%" for name, value in timer_percent.items()
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
            timer_stats[name] = self.timers[name]

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
        """Create PruneStatsCollection from numpy arrays."""
        collection = cls()
        for row in level_stats:
            stats = PruneStats(
                level=int(row["level"]),
                seg_idx=int(row["seg_idx"]),
                threshold=float(row["threshold"]),
                score_min=float(row["score_min"]),
                score_max=float(row["score_max"]),
                n_branches=int(row["n_branches"]),
                n_leaves=int(row["n_leaves"]),
                n_leaves_phy=int(row["n_leaves_phy"]),
                n_leaves_surv=int(row["n_leaves_surv"]),
            )
            collection.stats_list.append(stats)

        # Reconstruct timers
        for name in cls.TIMER_NAMES:
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
        # Store package version
        self.file.attrs["pruning_version"] = importlib.metadata.version("pyloki")
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
        param_sets: np.ndarray,
        scores: np.ndarray,
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
            ("param_sets", param_sets),
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
    # Extracts the ref_seg from the filename, e.g., tmp_003_01_log.txt -> 3
    match = re.search(r"tmp_(\d{3})_\d{2}_(log\.txt|results\.h5)", file_path.name)
    if match:
        return int(match.group(1))
    return 0


def merge_prune_result_files(
    results_dir: str | Path,
    log_file: Path,
    result_file: Path,
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

    """
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
