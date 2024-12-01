from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import attrs
import numpy as np
from numba import njit, types
from numba.experimental import jitclass

from pyloki.core import (
    PruningChebychevDPFunctions,
    PruningTaylorDPFunctions,
    SuggestionStruct,
    set_prune_load_func,
)
from pyloki.utils.misc import get_logger, prune_track
from pyloki.utils.timing import Timer

if TYPE_CHECKING:
    from pyloki.ffa import DynamicProgramming

DP_FUNCS_TYPE = PruningTaylorDPFunctions | PruningChebychevDPFunctions

logger = get_logger(__name__)


@jitclass(
    spec=[
        ("nsegments", types.i8),
        ("ref_idx", types.i8),
        ("data", types.i8[:]),
    ],
)
class SnailScheme:
    """A class to describe the indexing scheme used in the pruning algorithm.

    The scheme allow for "middle-out" enumeration of the segments.

    Parameters
    ----------
    nsegments : int
        The number of segments to be pruned.
    ref_idx : int
        Reference (starting) segment index for pruning.
    """

    def __init__(self, nsegments: int, ref_idx: int) -> None:
        self.nsegments = nsegments
        self.ref_idx = ref_idx
        self.data = np.argsort(np.abs(np.arange(nsegments) - ref_idx))

    @property
    def ref(self) -> float:
        """:obj:`float`: Reference time (middle of the reference segment)."""
        return self.ref_idx + 0.5

    def get_idx(self, prune_level: int) -> int:
        """
        Get the segment index for the given pruning level.

        Parameters
        ----------
        prune_level : int
            The pruning level.

        Returns
        -------
        int
            The segment index.
        """
        return self.data[prune_level]

    def get_coord(self, prune_level: int) -> tuple[float, float]:
        """
        Get the coord (reference and scale) for the given pruning level.

        Parameters
        ----------
        prune_level : int
            The pruning level.

        Returns
        -------
        tuple[float, float]
            The reference and scale for the given pruning level.
        """
        scheme_till_now = self.data[: prune_level + 1]
        ref = (np.min(scheme_till_now) + np.max(scheme_till_now) + 1) / 2
        scale = ref - np.min(scheme_till_now)
        return ref, scale

    def get_coord_add(self, prune_level: int) -> tuple[float, float]:
        ref = self.get_idx(prune_level) + 0.5
        scale = 0.5
        return ref, scale

    def get_valid(self, prune_level: int) -> tuple[float, float]:
        scheme_till_now = self.data[:prune_level]
        return np.min(scheme_till_now), np.max(scheme_till_now)


@attrs.define(auto_attribs=True, slots=True, kw_only=True)
class PruneStats:
    level: int
    seg_idx: int
    threshold: float
    score_min: float = 0.0
    score_max: float = 0.0
    n_branches: int = 1
    n_leaves: int = 1
    n_leaves_phy: int = 0
    n_leaves_surv: int = 0

    @property
    def lb_leaves(self) -> float:
        return np.round(np.log2(self.n_leaves), 2)

    @property
    def branch_frac(self) -> float:
        return np.round(self.n_leaves / self.n_branches, 2)

    @property
    def branch_frac_phy(self) -> float:
        return np.round(self.n_leaves_phy / self.n_branches, 2)

    @property
    def surv_frac(self) -> float:
        return np.round(self.n_leaves_surv / self.n_leaves, 2)

    def update(self, stats_dict: dict[str, int]) -> None:
        for key, value in stats_dict.items():
            setattr(self, key, value)

    def get_summary(self) -> str:
        summary = []
        summary.append(
            f"Prune level: {self.level}, seg_idx: {self.seg_idx}, "
            f"lb_leaves: {self.lb_leaves:.2f}, branch_frac: {self.branch_frac:.2f},",
        )
        summary.append(
            f"score thresh: {self.threshold:.2f}, max: {self.score_max:.2f}, "
            f"min: {self.score_min:.2f}, P(surv): {self.surv_frac:.2f}",
        )
        return "".join(summary) + "\n"


@njit(fastmath=True)
def pruning_iteration(
    sugg: SuggestionStruct,
    fold_segment: np.ndarray,
    prune_funcs: DP_FUNCS_TYPE,
    scheme: SnailScheme,
    threshold: float,
    prune_level: int,
    load_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    sugg_max: int = 2**17,
) -> tuple[SuggestionStruct, types.DictType[str, int]]:
    """
    Perform a single iteration of the pruning algorithm.

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
    iparam = 0
    n_leaves = 0
    n_leaves_phy = 0
    n_branches = sugg.size

    # Get the useful precomputed values: coord_cur := coord_prev + coord_add
    coord_init = scheme.get_coord(0)
    coord_cur = scheme.get_coord(prune_level)
    coord_prev = scheme.get_coord(prune_level - 1)
    coord_add = scheme.get_coord_add(prune_level)
    coord_valid = scheme.get_valid(prune_level)
    trans_matrix = prune_funcs.get_transform_matrix(coord_cur, coord_prev)
    validation_check = False
    if validation_check:
        validation_params = prune_funcs.get_validation_params(coord_valid)

    for isuggest in range(n_branches):
        leaves_arr = prune_funcs.branch(sugg.param_sets[isuggest], coord_cur)
        n_leaves += len(leaves_arr)
        if validation_check:
            leaves_arr = prune_funcs.validate(
                leaves_arr,
                coord_valid,
                validation_params,
            )
        n_leaves_phy += len(leaves_arr)

        for ileaf in range(len(leaves_arr)):
            leaf = leaves_arr[ileaf]
            param_idx, phase_shift = prune_funcs.resolve(
                leaf,
                coord_add,
                coord_init,
            )
            partial_res = prune_funcs.shift(
                load_func(fold_segment, param_idx),
                phase_shift,
            )
            combined_res = prune_funcs.add(sugg.folds[isuggest], partial_res)
            score = prune_funcs.score(combined_res)

            if score >= threshold:
                sugg_new.param_sets[iparam] = prune_funcs.transform(
                    leaf,
                    coord_cur,
                    trans_matrix,
                )
                sugg_new.folds[iparam] = combined_res
                sugg_new.scores[iparam] = score
                sugg_new.backtracks[iparam] = np.array(
                    [isuggest, *list(param_idx), phase_shift],
                )

                iparam += 1
                if iparam == sugg_max:
                    threshold = np.median(sugg_new.scores)
                    sugg_new = sugg_new.apply_threshold(threshold)
                    iparam = sugg_new.actual_size

    sugg_new = sugg_new.trim_empty(iparam)
    stats = {
        "n_branches": n_branches,
        "n_leaves": n_leaves,
        "n_leaves_phy": n_leaves_phy,
        "n_leaves_surv": sugg_new.size,
    }
    return sugg_new, stats


class Pruning:
    """
    Pruning class to perform the pruning algorithm on the dynamic programming.

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
        by default 2**17
    kind : {"taylor", "chebyshev"}, optional
        The kind of pruning algorithm to use, by default "taylor"
    """

    def __init__(
        self,
        dyp: DynamicProgramming,
        threshold_scheme: np.ndarray,
        max_sugg: int = 2**17,
        kind: str = "taylor",
        logfile: str = "prune.log",
    ) -> None:
        self._dyp = dyp
        self._threshold_scheme = threshold_scheme
        self._max_sugg = max_sugg
        self._logfile = logfile
        self._setup_pruning(kind)

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
    def logfile(self) -> str:
        return self._logfile

    @property
    def prune_funcs(self) -> DP_FUNCS_TYPE:
        return self._prune_funcs

    @property
    def load_func(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        return self._load_func

    @property
    def prune_level(self) -> int:
        return self._prune_level

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def scheme(self) -> SnailScheme:
        return self._scheme

    @property
    def suggestion(self) -> SuggestionStruct:
        return self._suggestion

    @property
    def t_ref(self) -> float:
        return self.scheme.ref * self.prune_funcs.tseg_ffa

    @property
    def backtrack_arr(self) -> np.ndarray:
        return self._backtrack_arr

    @property
    def best_intermediate_arr(self) -> np.ndarray:
        return self._best_intermediate_arr

    @Timer(name="prune_initialize", logger=logger.info)
    def initialize(self, ref_seg: int = 12) -> None:
        """Initialize the pruning algorithm.

        Parameters
        ----------
        ref_seg : int, optional
            The reference segment to start the pruning algorithm, by default 12.

        Notes
        -----
        Reference time for the parameters will be the middle of the reference segment.
        """
        self._scheme = SnailScheme(self.dyp.nsegments, ref_seg)
        self._complete = False
        self._prune_level = 0
        logger.info(f"Initializing pruning with ref segment: {self.scheme.ref_idx}")

        fold_segment = self.prune_funcs.load(self.dyp.fold, self.scheme.ref_idx)

        # Initialize the suggestions with the first segment
        coord = self.scheme.get_coord(self.prune_level)
        self._suggestion = self.prune_funcs.suggest(fold_segment, coord)
        # Records to track the numercal stability of the algorithm
        self._backtrack_arr = np.zeros(
            (self.dyp.fold.shape[0], self.max_sugg, len(self.dyp.fold.shape[1:])),
        )
        self._best_intermediate_arr = np.empty(
            (self.dyp.fold.shape[0], 3),
            dtype=object,
        )

    def execute(
        self,
        snr_lim: float,
        ref_seg_list: np.ndarray | None = None,
        *,
        lazy: bool = True,
    ) -> list[tuple[int, SuggestionStruct]]:
        """Execute the pruning algorithm.

        Parameters
        ----------
        snr_lim : float
            The signal-to-noise ratio limit for the pruning to stop.
        ref_seg_list : np.ndarray | None, optional
            The reference segment list to start the pruning algorithm, by default None
        lazy : bool, optional
            If True, return the result after the first pass, by default True

        Returns
        -------
        list[tuple[int, SuggestionStruct]]
            A list of tuples containing the reference segment and the suggestion.
        """
        res = []
        Path(self.logfile).write_text("Pruning log\n")
        if ref_seg_list is None:
            ref_seg_list = np.arange(0, self.dyp.nsegments, self.dyp.nsegments // 16)
        for ref_seg in ref_seg_list:
            with Path(self.logfile).open("a") as f:
                f.write(f"Pruning log for ref segment: {ref_seg}\n")
            self.initialize(ref_seg=ref_seg)
            for _ in prune_track(
                range(self.dyp.nsegments - 1),
                description="Pruning",
                total=self.dyp.nsegments - 1,
                get_score=lambda: self.suggestion.score_max,
                get_leaves=lambda: self.suggestion.size_lb,
            ):
                self.execute_iter()
            with Path(self.logfile).open("a") as f:
                f.write(f"Pruning complete for ref segment: {ref_seg}\n\n")
            if self.suggestion.size > 0 and np.max(self.suggestion.scores) > snr_lim:
                res.append((ref_seg, self.suggestion))
                if lazy:
                    return res
        return res

    def execute_iter(self) -> None:
        if self.is_complete:
            return
        self._prune_level += 1
        seg_idx_cur = self.scheme.get_idx(self.prune_level)
        fold_segment = self.prune_funcs.load(self.dyp.fold, seg_idx_cur)
        threshold = self.threshold_scheme[self.prune_level]
        pstats = PruneStats(
            level=self.prune_level,
            seg_idx=seg_idx_cur,
            threshold=threshold,
        )
        suggestion, stats = pruning_iteration(
            self.suggestion,
            fold_segment,
            self.prune_funcs,
            self.scheme,
            threshold,
            self.prune_level,
            self.load_func,
            self.max_sugg,
        )
        pstats.update(stats)
        pstats.update(
            {
                "level": self.prune_level,
                "seg_idx": seg_idx_cur,
                "threshold": threshold,
                "score_min": suggestion.score_min,
                "score_max": suggestion.score_max,
            },
        )
        with Path(self.logfile).open("a") as f:
            f.write(pstats.get_summary())
        if suggestion.size == 0:
            self._complete = True
            self._suggestion = suggestion
            logger.info(f"Pruning complete at level: {self.prune_level}")
            return
        self._best_intermediate_arr[self.prune_level] = suggestion.get_best()
        self._backtrack_arr[self.prune_level, : suggestion.size] = suggestion.backtracks
        self._suggestion = suggestion

    def generate_branching_pattern(self, n_iters: int, isuggest: int = 0) -> np.ndarray:
        """Need to fix this function."""
        branching_pattern = []
        leaf_param_sets = self.suggestion.param_sets
        coord_cur = self.scheme.get_coord(self.prune_level)
        for _ in range(1, n_iters + 1):
            leaves_arr = self.prune_funcs.branch(leaf_param_sets[isuggest], coord_cur)
            branching_pattern.append(len(leaves_arr))
            leaf_param_sets = leaves_arr
        return np.array(branching_pattern)

    def _setup_pruning(self, kind: str) -> None:
        if self.dyp.fold.ndim > 7:
            msg = "Pruning only supports initial data with up to 4 param dimensions."
            raise ValueError(msg)
        self._load_func = set_prune_load_func(self.dyp.fold.ndim - 3)
        if kind == "taylor":
            self._prune_funcs: DP_FUNCS_TYPE = PruningTaylorDPFunctions(
                self.dyp.cfg,
                self.dyp.param_arr,
                self.dyp.dparams_limited,
                self.dyp.tseg,
                self.dyp.cfg.prune_poly_order,
            )
        elif kind == "chebyshev":
            self._prune_funcs = PruningChebychevDPFunctions(
                self.dyp.cfg,
                self.dyp.param_arr,
                self.dyp.dparams_limited,
                self.dyp.tseg,
                self.dyp.cfg.prune_poly_order,
                self.dyp.cfg.prune_n_derivs,
            )
        else:
            msg = f"Invalid pruning kind: {kind}"
            raise ValueError(msg)
