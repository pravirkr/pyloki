from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, types

from pyloki.core import (
    PruneChebyshevDPFuncts,
    PruneTaylorComplexDPFuncts,
    PruneTaylorDPFuncts,
    set_prune_load_func,
)
from pyloki.io.cands import (
    PruneResultWriter,
    PruneStats,
    PruneStatsCollection,
    merge_prune_result_files,
)
from pyloki.utils.misc import (
    MultiprocessProgressTracker,
    PicklableStructRefWrapper,
    get_logger,
    get_worker_logger,
    prune_track,
)
from pyloki.utils.psr_utils import SnailScheme
from pyloki.utils.timing import Timer, nb_time_now

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable
    from multiprocessing.managers import DictProxy
    from queue import Queue

    from pyloki.ffa import DynamicProgramming
    from pyloki.utils.suggestion import SuggestionStruct, SuggestionStructComplex

DP_FUNCS_TYPE = (
    PruneTaylorDPFuncts | PruneTaylorComplexDPFuncts | PruneChebyshevDPFuncts
)

logger = get_logger(__name__)

"""
# Cache is disabled because there are random fails becasue of callable args
@njit(cache=False, fastmath=True)
def pruning_iteration(
    sugg: SuggestionStruct,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
    coord_add: tuple[float, float],
    coord_valid: tuple[float, float],
    prune_funcs: DP_FUNCS_TYPE,
    threshold: float,
    load_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    sugg_max: int = 2**17,
) -> tuple[
    SuggestionStruct,
    types.DictType[str, float],
    np.ndarray,
]:
    sugg_new = sugg.get_new(sugg_max)
    n_leaves = 0
    n_leaves_phy = 0
    score_min = np.inf
    score_max = -np.inf
    timers = np.zeros(7, dtype=np.float32)

    trans_matrix = prune_funcs.get_transform_matrix(coord_cur, coord_prev)
    validation_check = False
    if validation_check:
        validation_params = prune_funcs.get_validation_params(coord_valid)

    n_branches = sugg.valid_size
    # Preallocate backtrack array
    backtrack = np.empty(sugg.backtracks.shape[1], dtype=np.int32)

    for isuggest in range(n_branches):
        tmp = nb_time_now()
        leaves_arr = prune_funcs.branch(sugg.param_sets[isuggest], coord_cur)
        n_leaves += len(leaves_arr)
        timers[0] += nb_time_now() - tmp

        tmp = nb_time_now()
        if validation_check:
            leaves_arr = prune_funcs.validate(
                leaves_arr,
                coord_valid,
                validation_params,
            )
        n_leaves_phy += len(leaves_arr)
        timers[1] += nb_time_now() - tmp

        for ileaf in range(len(leaves_arr)):
            leaf = leaves_arr[ileaf]

            tmp2 = nb_time_now()
            param_idx, phase_shift = prune_funcs.resolve(
                leaf,
                coord_add,
                coord_init,
            )
            timers[2] += nb_time_now() - tmp2

            tmp2 = nb_time_now()
            partial_res = prune_funcs.shift(
                load_func(fold_segment, param_idx),
                phase_shift,
            )
            combined_res = prune_funcs.add(sugg.folds[isuggest], partial_res)
            timers[3] += nb_time_now() - tmp2

            tmp2 = nb_time_now()
            score = prune_funcs.score(combined_res)
            score_min = min(score_min, score)
            score_max = max(score_max, score)
            timers[4] += nb_time_now() - tmp2

            if score >= threshold:
                tmp3 = nb_time_now()
                leaf_trans = prune_funcs.transform(
                    leaf,
                    coord_cur,
                    trans_matrix,
                )
                backtrack[0] = isuggest
                backtrack[1 : len(param_idx) + 1] = param_idx
                backtrack[-1] = phase_shift
                timers[5] += nb_time_now() - tmp3

                tmp3 = nb_time_now()
                is_success = sugg_new.add(leaf_trans, combined_res, score, backtrack)
                if not is_success:
                    # Handle buffer full case
                    threshold = sugg_new.trim_threshold()
                timers[6] += nb_time_now() - tmp3

    sugg_new = sugg_new.trim_empty()
    stats = {
        "n_leaves": float(n_leaves),
        "n_leaves_phy": float(n_leaves_phy),
        "score_min": score_min,
        "score_max": score_max,
    }
    return sugg_new, stats, timers
"""


# Cache is disabled because there are random fails becasue of callable args
@njit(cache=False, fastmath=True)
def pruning_iteration_batched(
    sugg: SuggestionStruct,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
    coord_add: tuple[float, float],
    coord_valid: tuple[float, float],
    prune_funcs: DP_FUNCS_TYPE,
    threshold: float,
    load_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    sugg_max: int = 2**18,
    batch_size: int = 1024,
) -> tuple[
    SuggestionStruct,
    types.DictType[str, float],
    np.ndarray,
]:
    """Perform a single iteration of the pruning algorithm using batch processing.

    Parameters
    ----------
    sugg : SuggestionStruct
        The suggestion structure to be pruned.
    fold_segment : np.ndarray
        The fold segment of the current pruning level to be used for pruning.
    prune_funcs : PruningTaylorDPFunctions
        A container for the functions to be used in the pruning algorithm.
    scheme : SnailScheme
        A container describing the indexing scheme used in the pruning algorithm.
    threshold : float
        The threshold score for the current pruning level.
    prune_level : int
        The current pruning level.
    load_func : Callable[[np.ndarray, np.ndarray], np.ndarray]
        A function to load the desired fold from the input structure.
    sugg_max : int, optional
        Maximum number of suggestions to keep in the output SuggestionStruct,
        by default 2**17

    Returns
    -------
    tuple[SuggestionStruct, PruneStats]
        The pruned suggestion structure and the statistics of the pruning iteration.
    """
    sugg_new = sugg.get_new(sugg_max)
    n_leaves = 0
    n_leaves_phy = 0
    score_min = np.inf
    score_max = -np.inf
    timers = np.zeros(7, dtype=np.float32)
    current_threshold = threshold

    trans_matrix = prune_funcs.get_transform_matrix(coord_cur, coord_prev)
    validation_check = False
    if validation_check:
        validation_params = prune_funcs.get_validation_params(coord_valid)

    n_branches = sugg.valid_size
    batch_size = max(1, min(batch_size, n_branches))

    # Loop over branches in batches
    for i_batch_start in range(0, n_branches, batch_size):
        i_batch_end = min(i_batch_start + batch_size, n_branches)
        cur_batch_indices = np.arange(i_batch_start, i_batch_end)

        # Branching
        t_start = nb_time_now()
        batch_leaves, batch_leaf_origins = prune_funcs.branch(
            sugg.param_sets[i_batch_start:i_batch_end],
            coord_cur,
        )
        n_leaves_batch = len(batch_leaves)
        n_leaves += n_leaves_batch
        timers[0] += nb_time_now() - t_start
        if n_leaves_batch == 0:
            continue

        # Validation
        t_start = nb_time_now()
        if validation_check:
            batch_leaves = prune_funcs.validate(
                batch_leaves,
                coord_valid,
                validation_params,
            )
        n_leaves_batch = len(batch_leaves)
        n_leaves_phy += n_leaves_batch
        timers[1] += nb_time_now() - t_start

        # Resolve
        t_start = nb_time_now()
        batch_param_idx, batch_phase_shift = prune_funcs.resolve(
            batch_leaves,
            coord_add,
            coord_init,
        )
        timers[2] += nb_time_now() - t_start

        # Load, shift, add
        t_start = nb_time_now()
        batch_loaded_data = load_func(fold_segment, batch_param_idx)
        # Map batch_leaf_origins (0 to current_batch_size-1) to global indices
        isuggest_batch = cur_batch_indices[batch_leaf_origins]
        batch_combined_res = prune_funcs.shift_add(
            batch_loaded_data,
            batch_phase_shift,
            sugg.folds,
            isuggest_batch,
        )
        timers[3] += nb_time_now() - t_start

        # Score
        t_start = nb_time_now()
        batch_scores = prune_funcs.score(batch_combined_res)
        if n_leaves_batch > 0:
            score_min = min(score_min, np.min(batch_scores))
            score_max = max(score_max, np.max(batch_scores))
        timers[4] += nb_time_now() - t_start

        # Thresholding & Filtering
        t_start = nb_time_now()
        passing_mask = batch_scores >= threshold
        num_passing = np.sum(passing_mask)
        if num_passing == 0:
            timers[6] += nb_time_now() - t_start
            continue

        # Filter results needed for adding to sugg_new
        filtered_indices = np.where(passing_mask)[0]
        filtered_leaves = batch_leaves[filtered_indices]
        filtered_scores = batch_scores[filtered_indices]
        filtered_combined_res = batch_combined_res[filtered_indices]
        timers[6] += nb_time_now() - t_start  # Part of time for Threshold step

        # Transform & Backtrack
        t_start = nb_time_now()
        filtered_leaves_trans = prune_funcs.transform(
            filtered_leaves,
            coord_cur,
            trans_matrix,
        )
        timers[5] += nb_time_now() - t_start

        # Construct Backtrack & Add to sugg_new
        t_start = nb_time_now()
        n_add = len(filtered_scores)
        bt_nparams = batch_param_idx.shape[1]
        batch_backtrack = np.empty((n_add, sugg.backtracks.shape[1]), dtype=np.int32)
        batch_backtrack[:, 0] = isuggest_batch[filtered_indices]
        batch_backtrack[:, 1 : bt_nparams + 1] = batch_param_idx[filtered_indices]
        batch_backtrack[:, -1] = batch_phase_shift[filtered_indices]
        current_threshold = sugg_new.add_batch(
            filtered_leaves_trans,
            filtered_combined_res,
            filtered_scores,
            batch_backtrack,
            current_threshold,
        )
        timers[6] += nb_time_now() - t_start

    # Finalize
    sugg_new = sugg_new.trim_empty()
    stats = {
        "n_leaves": float(n_leaves),
        "n_leaves_phy": float(n_leaves_phy),
        "score_min": score_min if np.isfinite(score_min) else 0.0,
        "score_max": score_max if np.isfinite(score_max) else 0.0,
    }
    return sugg_new, stats, timers


class Pruning:
    """A class to perform the pruning algorithm on the FFA search results.

    Time is linearly advancing. The algorithm starts with a reference segment
    and iteratively prunes the parameter space based on the scores of the
    suggestions.

    Parameters
    ----------
    dyp : DynamicProgramming
        An instance of the DynamicProgramming class.
    threshold_scheme : np.ndarray
        An array of thresholds for each pruning level. Thresholds should
        maximise the Prob(detecting signal) / (computational complexity) ratio.
    max_sugg : int, optional
        Maximum suggestions to store in memory (to avoid tree explosion),
        by default 2**17.
    kind : {"taylor", "chebyshev"}, optional
        The kind of pruning algorithm to use, by default "taylor".
    """

    def __init__(
        self,
        dyp: DynamicProgramming,
        threshold_scheme: np.ndarray,
        max_sugg: int = 2**17,
        batch_size: int = 1024,
        kind: str = "taylor",
        logger: logging.Logger | None = None,
    ) -> None:
        self._dyp = dyp
        self._threshold_scheme = threshold_scheme
        self._max_sugg = max_sugg
        self._batch_size = batch_size
        self._setup_pruning(kind)
        self._logger = logger or get_logger(__name__)

    @property
    def dyp(self) -> DynamicProgramming:
        return self._dyp

    @property
    def threshold_scheme(self) -> np.ndarray:
        return self._threshold_scheme

    @property
    def max_sugg(self) -> int:
        return self._max_sugg

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def prune_funcs(self) -> DP_FUNCS_TYPE:
        return self._prune_funcs.get_instance()

    @property
    def load_func(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        return self._load_func

    @property
    def prune_level(self) -> int:
        return self._prune_level

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def scheme(self) -> SnailScheme:
        return self._scheme

    @property
    def suggestion(self) -> SuggestionStruct | SuggestionStructComplex:
        return self._suggestion

    @property
    def pstats(self) -> PruneStatsCollection:
        return self._pstats

    @property
    def backtrack_arr(self) -> np.ndarray:
        return self._backtrack_arr

    @property
    def best_intermediate_arr(self) -> np.ndarray:
        return self._best_intermediate_arr

    def initialize(self, ref_seg: int, log_file: Path) -> None:
        """Initialize the pruning algorithm.

        Parameters
        ----------
        ref_seg : int
            The reference segment to start the pruning algorithm.
        log_file : Path
            File path to store the log of the pruning algorithm.

        Notes
        -----
        Reference time for the parameters will be the middle of the reference segment.
        """
        self._scheme = SnailScheme(
            nseg=self.dyp.nsegments,
            ref_idx=ref_seg,
            tseg=self.dyp.tseg,
        )
        self._complete = False
        self._prune_level = 0
        self.logger.info(
            f"Initializing pruning run with ref segment: {self.scheme.ref_idx}",
        )

        fold_segment = self.prune_funcs.load(self.dyp.fold, self.scheme.ref_idx)

        # Initialize the suggestions with the first segment
        coord = self.scheme.get_coord(self.prune_level)
        self._suggestion = self.prune_funcs.suggest(fold_segment, coord)
        # Records to track the numerical stability of the algorithm
        self._backtrack_arr = np.zeros(
            (self.max_sugg, self.dyp.cfg.prune_poly_order + 2),
            dtype=np.int32,
        )
        self._best_intermediate_arr = np.zeros(
            self.dyp.nsegments - 1,
            dtype=np.dtype(
                [
                    ("param_sets", np.float64, (self.dyp.cfg.prune_poly_order + 2, 2)),
                    ("folds", self.dyp.fold.dtype, fold_segment.shape[-2:]),
                    ("scores", np.float32),
                ],
            ),
        )
        self._pstats = PruneStatsCollection()
        pstats_cur = PruneStats(
            level=self.prune_level,
            seg_idx=self.scheme.get_idx(self.prune_level),
            threshold=0,
            score_min=self.suggestion.score_min,
            score_max=self.suggestion.score_max,
            n_branches=self.suggestion.size,
            n_leaves=self.suggestion.size,
            n_leaves_phy=self.suggestion.size,
            n_leaves_surv=self.suggestion.size,
        )
        with log_file.open("a") as f:
            f.write(pstats_cur.get_summary())
        self._pstats.update_stats(pstats_cur)

    def execute(
        self,
        ref_seg: int,
        outdir: str = "./",
        log_file: Path | None = None,
        result_file: Path | None = None,
        shared_progress: DictProxy | None = None,
        task_id: int | None = None,
    ) -> None:
        """Execute the pruning algorithm.

        Parameters
        ----------
        ref_seg : int
            The reference segment to start the pruning algorithm.
        outdir : str, optional
            The output directory to store the results, by default "./".
        log_file : Path, optional
            The file to store the log of the pruning algorithm, by default None.
        result_file : Path, optional
            The file to store the results of the pruning algorithm, by default None.
        shared_progress : DictProxy, optional
            The shared progress dictionary, by default None.
        task_id : int, optional
            The task ID for progress tracking, by default None.
        """
        run_name = f"{ref_seg:03d}_{task_id:02d}"

        if log_file is None:
            log_file = Path(outdir) / f"tmp_{run_name}_log.txt"
            log_file.touch()
        if result_file is None:
            result_file = Path(outdir) / f"tmp_{run_name}_results.h5"
        with log_file.open("a") as f:
            f.write(f"Pruning log for ref segment: {ref_seg}\n")
        with Timer(name="prune_initialize", logger=self.logger.info):
            self.initialize(ref_seg, log_file)
        for _ in prune_track(
            range(self.dyp.nsegments - 1),
            description=f"Pruning segment {ref_seg:03d}",
            shared_progress=shared_progress,
            task_id=task_id,
            total=self.dyp.nsegments - 1,
            get_score=lambda: self.suggestion.score_max,
            get_leaves=lambda: self.suggestion.size_lb,
        ):
            self.execute_iter(log_file)
        # Transform the suggestion oarams to middle of the data
        delta_t = self.scheme.get_delta(self.prune_level)
        with PruneResultWriter(result_file, mode="a") as writer:
            writer.write_run_results(
                run_name,
                self.scheme.data,
                self.suggestion.get_transformed(delta_t),
                self.suggestion.scores,
                self.pstats,
            )
        with log_file.open("a") as f:
            f.write(f"Pruning complete for ref segment: {ref_seg}\n\n")
            f.write(f"Time: {self.pstats.get_timer_summary()}\n")
        self.logger.info(f"Pruning run complete for ref segment: {ref_seg}")
        self.logger.info(f"Pruning stats: {self.pstats.get_stats_summary()}")
        self.logger.info(f"Pruning time: {self.pstats.get_concise_timer_summary()}")

    def execute_iter(self, log_file: Path) -> None:
        if self.is_complete:
            return
        self._prune_level += 1
        seg_idx_cur = self.scheme.get_idx(self.prune_level)
        fold_segment = self.prune_funcs.load(self.dyp.fold, seg_idx_cur)
        threshold = self.threshold_scheme[self.prune_level - 1]
        # Get the useful precomputed values: coord_cur := coord_prev + coord_add
        coord_init = self.scheme.get_coord(0)
        coord_cur = self.scheme.get_coord(self.prune_level)
        coord_prev = self.scheme.get_coord(self.prune_level - 1)
        coord_add = self.scheme.get_seg_coord(self.prune_level)
        coord_valid = self.scheme.get_valid(self.prune_level)
        suggestion, stats_dict, timers = pruning_iteration_batched(
            self.suggestion,
            fold_segment,
            coord_init,
            coord_cur,
            coord_prev,
            coord_add,
            coord_valid,
            self.prune_funcs,
            threshold,
            self.load_func,
            self.max_sugg,
            self.batch_size,
        )
        pstats_cur = PruneStats(
            level=self.prune_level,
            seg_idx=seg_idx_cur,
            threshold=threshold,
            n_branches=self.suggestion.valid_size,
            n_leaves_surv=suggestion.valid_size,
            **stats_dict,
        )
        with log_file.open("a") as f:
            f.write(pstats_cur.get_summary())
        self._pstats.update_stats(pstats_cur, timers)
        if suggestion.size == 0:
            self._complete = True
            self._suggestion = suggestion
            self.logger.info(f"Pruning run complete at level: {self.prune_level}")
            return
        self._best_intermediate_arr[self.prune_level - 1] = suggestion.get_best()
        self._backtrack_arr[: suggestion.size] = suggestion.backtracks.copy()
        self._suggestion = suggestion

    def get_branching_pattern(self, nstages: int, isuggest: int = 0) -> np.ndarray:
        """Get the branching pattern of the pruning algorithm."""
        branching_pattern = []
        fold_segment = self.prune_funcs.load(self.dyp.fold, self.scheme.ref_idx)
        coord = self.scheme.get_coord(0)
        leaf = self.prune_funcs.suggest(fold_segment, coord).param_sets[isuggest]
        for prune_level in range(1, nstages + 1):
            coord_cur = self.scheme.get_coord(prune_level)
            coord_prev = self.scheme.get_coord(prune_level - 1)
            trans_matrix = self.prune_funcs.get_transform_matrix(coord_cur, coord_prev)
            leaves_arr = self.prune_funcs.branch(leaf, coord_cur)
            branching_pattern.append(len(leaves_arr))
            leaf = self.prune_funcs.transform(
                leaves_arr[isuggest],
                coord_cur,
                trans_matrix,
            )
        return np.array(branching_pattern)

    def _setup_pruning(self, kind: str) -> None:
        if self.dyp.fold.ndim > 7:
            msg = "Pruning only supports initial data with up to 4 param dimensions."
            raise ValueError(msg)
        self._load_func = set_prune_load_func(self.dyp.fold.ndim - 3)
        if kind == "taylor" and self.dyp.cfg.use_fft_shifts:
            self._prune_funcs = PicklableStructRefWrapper[PruneTaylorComplexDPFuncts](
                PruneTaylorComplexDPFuncts,
                self.dyp.param_arr,
                self.dyp.dparams_limited,
                self.dyp.tseg,
                self.dyp.cfg,
            )
        elif kind == "taylor" and not self.dyp.cfg.use_fft_shifts:
            self._prune_funcs = PicklableStructRefWrapper[PruneTaylorDPFuncts](
                PruneTaylorDPFuncts,
                self.dyp.param_arr,
                self.dyp.dparams_limited,
                self.dyp.tseg,
                self.dyp.cfg,
            )
        elif kind == "chebyshev":
            self._prune_funcs = PicklableStructRefWrapper[PruneChebyshevDPFuncts](
                PruneChebyshevDPFuncts,
                self.dyp.param_arr,
                self.dyp.dparams_limited,
                self.dyp.tseg,
                self.dyp.cfg,
            )
        else:
            msg = f"Invalid pruning kind: {kind}"
            raise ValueError(msg)


def _prune_dyp_seg(
    dyp: DynamicProgramming,
    ref_seg: int,
    threshold_scheme: np.ndarray,
    shared_progress: DictProxy,
    task_id: int,
    log_queue: Queue,
    outdir: str = "./",
    max_sugg: int = 2**18,
    batch_size: int = 1024,
    kind: str = "taylor",
    log_file: Path | None = None,
    result_file: Path | None = None,
) -> None:
    """Execute the pruning for multiprocessing."""
    logger = get_worker_logger(f"worker_{ref_seg:03d}", log_queue=log_queue)
    prn = Pruning(
        dyp,
        threshold_scheme,
        max_sugg=max_sugg,
        batch_size=batch_size,
        kind=kind,
        logger=logger,
    )
    prn.execute(
        ref_seg,
        outdir=outdir,
        log_file=log_file,
        result_file=result_file,
        shared_progress=shared_progress,
        task_id=task_id,
    )


def prune_dyp_tree(
    dyp: DynamicProgramming,
    threshold_scheme: np.ndarray,
    n_runs: int | None = None,
    ref_segs: list[int] | None = None,
    max_sugg: int = 2**18,
    batch_size: int = 1024,
    outdir: str = "./",
    file_prefix: str = "test",
    kind: str = "taylor",
    n_workers: int = 4,
) -> str:
    if not isinstance(n_workers, int | np.integer):
        msg = f"n_workers must be an integer, got {type(n_workers)}"
        raise TypeError(msg)
    if n_workers < 1:
        msg = f"n_workers must be greater than 0, got {n_workers}"
        raise ValueError(msg)

    filebase = f"{file_prefix}_pruning_nstages_{dyp.nsegments}"
    log_file = Path(outdir) / f"{filebase}_log.txt"
    result_file = Path(outdir) / f"{filebase}_results.h5"
    if log_file.exists() or result_file.exists():
        msg = f"Output files already exist: {log_file}, {result_file}"
        raise FileExistsError(msg)
    # n_runs takes precedence over ref_segs
    if n_runs is not None:
        if not 1 <= n_runs <= dyp.nsegments:
            msg = f"n_runs must be between 1 and {dyp.nsegments}, got {n_runs}"
            raise ValueError(msg)
        ref_segs = list(np.linspace(0, dyp.nsegments - 1, n_runs, dtype=int))
    elif ref_segs is None:
        msg = "Either n_runs or ref_segs must be provided"
        raise ValueError(msg)

    logger.info(f"Starting Pruning for {len(ref_segs)} runs, with {n_workers} workers")
    log_file.write_text("Pruning log\n")
    with PruneResultWriter(result_file) as writer:
        writer.write_metadata(
            dyp.cfg.param_names,
            dyp.nsegments,
            max_sugg,
            threshold_scheme,
        )
    if n_workers == 1:
        prn = Pruning(
            dyp,
            threshold_scheme,
            max_sugg=max_sugg,
            batch_size=batch_size,
            kind=kind,
            logger=logger,
        )
        for ref_seg in ref_segs:
            prn.execute(
                ref_seg,
                outdir=outdir,
                log_file=log_file,
                result_file=result_file,
                task_id=0,
            )
    else:
        with (
            MultiprocessProgressTracker("Pruning tree") as tracker,
            ProcessPoolExecutor(max_workers=n_workers) as executor,
        ):
            futures_to_seg = {}
            # Important: pass a reference to the tracker dict to the worker
            # processes to avoid pickling issues
            shared_progress = tracker.shared_progress
            log_queue = tracker.log_queue

            for ref_seg in ref_segs:
                task_id = tracker.add_task(
                    f"Pruning segment {ref_seg:03d}",
                    total=dyp.nsegments - 1,
                )
                future = executor.submit(
                    _prune_dyp_seg,
                    dyp,
                    ref_seg,
                    threshold_scheme,
                    shared_progress,
                    task_id,
                    log_queue,
                    outdir,
                    max_sugg,
                    batch_size,
                    kind,
                )
                futures_to_seg[future] = ref_seg

            # Collect results with integrated progress tracking
            results, errors = tracker.collect_results(futures_to_seg)
            with log_file.open("a") as f:
                for ref_seg, error_msg in errors:
                    f.write(f"Error processing ref_seg {ref_seg}: {error_msg}\n")
                    logger.error(f"Error processing ref_seg {ref_seg}: {error_msg}")
        merge_prune_result_files(outdir, log_file, result_file)
    logger.info(f"Pruning complete. Results saved to {result_file}")
    return result_file.as_posix()
