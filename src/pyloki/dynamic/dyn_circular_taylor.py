from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np
from numba import njit, typed, types
from numba.experimental import structref
from numba.extending import overload_method

from pyloki.core import circular, common, taylor
from pyloki.detection import scoring
from pyloki.utils.snail import MiddleOutScheme
from pyloki.utils.world_tree import WorldTree, WorldTreeComplex

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyloki.config import PulsarSearchConfig


@structref.register
class PruneCircTaylorDPFunctsTemplate(types.StructRef):
    pass


@structref.register
class PruneCircTaylorComplexDPFunctsTemplate(types.StructRef):
    pass


class PruneCircTaylorDPFuncts(structref.StructRefProxy):
    def __new__(
        cls,
        param_arr: list[np.ndarray],
        dparams: np.ndarray,
        param_grid_count_init: np.ndarray,
        tseg_ffa: float,
        cfg: PulsarSearchConfig,
        use_moving_grid: bool = True,
    ) -> Self:
        """Create a new instance of PruneCircTaylorDPFuncts."""
        return prune_circ_taylor_dp_functs_init(
            typed.List(param_arr),
            dparams,
            param_grid_count_init,
            tseg_ffa,
            cfg.nbins,
            cfg.eta,
            cfg.param_limits,
            cfg.bseg_brute,
            cfg.score_widths,
            cfg.prune_poly_order,
            cfg.branch_max,
            cfg.tiling_strategy,
            cfg.p_orb_min,
            cfg.x_mass_const,
            cfg.propagator_significance,
            cfg.validation_significance,
            use_moving_grid,
            getattr(cfg, "score_mode_id", scoring.SCORE_MODE_BOXCAR),
            getattr(cfg, "score_n_harmonics", 16),
            getattr(cfg, "kadane_biases", scoring.generate_kadane_biases(cfg.nbins)),
        )

    def load(self, fold: np.ndarray, seg_idx: int) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    def seed(
        self,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> WorldTree:
        return seed_func(self, fold_segment, coord_init)

    def branch(
        self,
        leaves_batch: np.ndarray,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, leaves_batch, coord_cur, coord_prev)

    def validate(
        self,
        leaves_batch: np.ndarray,
        leaves_origins: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return validate_func(self, leaves_batch, leaves_origins, coord_cur)

    def get_validation_params(
        self,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return get_validation_params_func(self, coord_add)

    def resolve(
        self,
        leaves_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_cur: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_func(self, leaves_batch, coord_add, coord_cur, coord_init)

    def shift_add(
        self,
        segment_batch: np.ndarray,
        shift_batch: np.ndarray,
        folds: np.ndarray,
        isuggest_batch: np.ndarray,
    ) -> np.ndarray:
        return shift_add_func(
            self,
            segment_batch,
            shift_batch,
            folds,
            isuggest_batch,
        )

    def score(self, combined_res_batch: np.ndarray) -> np.ndarray:
        return score_func(self, combined_res_batch)

    def transform(
        self,
        leaves_batch: np.ndarray,
        coord_next: tuple[float, float],
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        return transform_func(self, leaves_batch, coord_next, coord_cur)

    def get_transform_matrix(
        self,
        coord_next: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_next, coord_prev)

    def pack(self, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)

    def ascend(
        self,
        world_tree: WorldTree,
        dyp_folds: np.ndarray,
        load_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        idx_segments: np.ndarray,
        coord_segments: np.ndarray,
        coord_cur: tuple[float, float],
        batch_size: int,
    ) -> None:
        return ascend_func(
            self,
            world_tree,
            dyp_folds,
            load_func,
            idx_segments,
            coord_segments,
            coord_cur,
            batch_size,
        )

    def report(
        self,
        leaves_batch: np.ndarray,
        coord_report: tuple[float, float],
        coord_end: tuple[float, float],
    ) -> np.ndarray:
        return report_func(self, leaves_batch, coord_report, coord_end)


class PruneCircTaylorComplexDPFuncts(structref.StructRefProxy):
    def __new__(
        cls,
        param_arr: list[np.ndarray],
        dparams: np.ndarray,
        param_grid_count_init: np.ndarray,
        tseg_ffa: float,
        cfg: PulsarSearchConfig,
        use_moving_grid: bool = True,
    ) -> Self:
        """Create a new instance of PruneCircTaylorComplexDPFuncts."""
        return prune_circ_taylor_complex_dp_functs_init(
            typed.List(param_arr),
            dparams,
            param_grid_count_init,
            tseg_ffa,
            cfg.nbins,
            cfg.eta,
            cfg.param_limits,
            cfg.bseg_brute,
            cfg.score_widths,
            cfg.prune_poly_order,
            cfg.branch_max,
            cfg.tiling_strategy,
            cfg.p_orb_min,
            cfg.x_mass_const,
            cfg.propagator_significance,
            cfg.validation_significance,
            use_moving_grid,
            getattr(cfg, "score_mode_id", scoring.SCORE_MODE_BOXCAR),
            getattr(cfg, "score_n_harmonics", 16),
            getattr(cfg, "kadane_biases", scoring.generate_kadane_biases(cfg.nbins)),
        )

    def load(self, fold: np.ndarray, seg_idx: int) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    def seed(
        self,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> WorldTreeComplex:
        return seed_complex_func(self, fold_segment, coord_init)

    def branch(
        self,
        leaves_batch: np.ndarray,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, leaves_batch, coord_cur, coord_prev)

    def validate(
        self,
        leaves_batch: np.ndarray,
        leaves_origins: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return validate_func(self, leaves_batch, leaves_origins, coord_cur)

    def get_validation_params(
        self,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return get_validation_params_func(self, coord_add)

    def resolve(
        self,
        leaves_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_cur: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_func(self, leaves_batch, coord_add, coord_cur, coord_init)

    def shift_add(
        self,
        segment_batch: np.ndarray,
        shift_batch: np.ndarray,
        folds: np.ndarray,
        isuggest_batch: np.ndarray,
    ) -> np.ndarray:
        return shift_add_complex_func(
            self,
            segment_batch,
            shift_batch,
            folds,
            isuggest_batch,
        )

    def score(self, batch_combined_res: np.ndarray) -> np.ndarray:
        return score_complex_func(self, batch_combined_res)

    def transform(
        self,
        leaves_batch: np.ndarray,
        coord_next: tuple[float, float],
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        return transform_func(self, leaves_batch, coord_next, coord_cur)

    def get_transform_matrix(
        self,
        coord_next: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_next, coord_prev)

    def pack(self, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)

    def ascend(
        self,
        world_tree: WorldTreeComplex,
        dyp_folds: np.ndarray,
        load_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        idx_segments: np.ndarray,
        coord_segments: np.ndarray,
        coord_cur: tuple[float, float],
        batch_size: int,
    ) -> None:
        return ascend_complex_func(
            self,
            world_tree,
            dyp_folds,
            load_func,
            idx_segments,
            coord_segments,
            coord_cur,
            batch_size,
        )

    def report(
        self,
        leaves_batch: np.ndarray,
        coord_report: tuple[float, float],
        coord_end: tuple[float, float],
    ) -> np.ndarray:
        return report_func(self, leaves_batch, coord_report, coord_end)


fields_prune_circ_taylor_dp_funcs = [
    ("param_arr", types.ListType(types.Array(types.f8, 1, "C"))),
    ("dparams", types.f8[::1]),
    ("param_grid_count_init", types.i8[::1]),
    ("tseg_ffa", types.f8),
    ("nbins", types.int64),
    ("eta", types.f8),
    ("param_limits", types.f8[:, ::1]),
    ("bseg_brute", types.int64),
    ("score_widths", types.i8[::1]),
    ("poly_order", types.i8),
    ("branch_max", types.i8),
    ("tiling_strategy", types.unicode_type),
    ("p_orb_min", types.f8),
    ("x_mass_const", types.f8),
    ("propagator_significance", types.f8),
    ("validation_significance", types.f8),
    ("use_moving_grid", types.bool_),
    ("score_mode_id", types.i8),
    ("score_n_harmonics", types.i8),
    ("kadane_biases", types.f4[::1]),
]

structref.define_boxing(PruneCircTaylorDPFunctsTemplate, PruneCircTaylorDPFuncts)
PruneCircTaylorDPFunctsType = PruneCircTaylorDPFunctsTemplate(
    fields_prune_circ_taylor_dp_funcs,
)

structref.define_boxing(
    PruneCircTaylorComplexDPFunctsTemplate,
    PruneCircTaylorComplexDPFuncts,
)
PruneCircTaylorComplexDPFunctsType = PruneCircTaylorComplexDPFunctsTemplate(
    fields_prune_circ_taylor_dp_funcs,
)


@njit(cache=True, fastmath=True)
def prune_circ_taylor_dp_functs_init(
    param_arr: list[np.ndarray],
    dparams: np.ndarray,
    param_grid_count_init: np.ndarray,
    tseg_ffa: float,
    nbins: int,
    eta: float,
    param_limits: np.ndarray,
    bseg_brute: int,
    score_widths: np.ndarray,
    poly_order: int,
    branch_max: int,
    tiling_strategy: str,
    p_orb_min: float,
    x_mass_const: float,
    propagator_significance: float,
    validation_significance: float,
    use_moving_grid: bool,
    score_mode_id: int,
    score_n_harmonics: int,
    kadane_biases: np.ndarray,
) -> PruneCircTaylorDPFuncts:
    """Initialize the PruneCircTaylorDPFuncts object."""
    self = structref.new(PruneCircTaylorDPFunctsType)  # ty: ignore[missing-argument]
    self.param_arr = typed.List(param_arr)
    self.dparams = dparams
    self.param_grid_count_init = param_grid_count_init
    self.tseg_ffa = tseg_ffa
    self.nbins = nbins
    self.eta = eta
    self.param_limits = param_limits
    self.bseg_brute = bseg_brute
    self.score_widths = score_widths
    self.poly_order = poly_order
    self.branch_max = branch_max
    self.tiling_strategy = tiling_strategy
    self.p_orb_min = p_orb_min
    self.x_mass_const = x_mass_const
    self.propagator_significance = propagator_significance
    self.validation_significance = validation_significance
    self.use_moving_grid = use_moving_grid
    self.score_mode_id = score_mode_id
    self.score_n_harmonics = score_n_harmonics
    self.kadane_biases = kadane_biases
    if self.poly_order != 5:
        msg = (
            f"Poly order {self.poly_order} is not supported for circular orbit pruning"
        )
        raise ValueError(msg)
    return self


@njit(cache=True, fastmath=True)
def prune_circ_taylor_complex_dp_functs_init(
    param_arr: list[np.ndarray],
    dparams: np.ndarray,
    param_grid_count_init: np.ndarray,
    tseg_ffa: float,
    nbins: int,
    eta: float,
    param_limits: np.ndarray,
    bseg_brute: int,
    score_widths: np.ndarray,
    poly_order: int,
    branch_max: int,
    tiling_strategy: str,
    p_orb_min: float,
    x_mass_const: float,
    propagator_significance: float,
    validation_significance: float,
    use_moving_grid: bool,
    score_mode_id: int,
    score_n_harmonics: int,
    kadane_biases: np.ndarray,
) -> PruneCircTaylorComplexDPFuncts:
    """Initialize the PruneCircTaylorComplexDPFuncts object."""
    self = structref.new(PruneCircTaylorComplexDPFunctsType)  # ty: ignore[missing-argument]
    self.param_arr = typed.List(param_arr)
    self.dparams = dparams
    self.param_grid_count_init = param_grid_count_init
    self.tseg_ffa = tseg_ffa
    self.nbins = nbins
    self.eta = eta
    self.param_limits = param_limits
    self.bseg_brute = bseg_brute
    self.score_widths = score_widths
    self.poly_order = poly_order
    self.branch_max = branch_max
    self.tiling_strategy = tiling_strategy
    self.p_orb_min = p_orb_min
    self.x_mass_const = x_mass_const
    self.propagator_significance = propagator_significance
    self.validation_significance = validation_significance
    self.use_moving_grid = use_moving_grid
    self.score_mode_id = score_mode_id
    self.score_n_harmonics = score_n_harmonics
    self.kadane_biases = kadane_biases
    if self.poly_order != 5:
        msg = (
            f"Poly order {self.poly_order} is not supported for circular orbit pruning"
        )
        raise ValueError(msg)
    return self


@njit(cache=True, fastmath=True)
def load_func(
    self: PruneCircTaylorDPFuncts | PruneCircTaylorComplexDPFuncts,
    fold: np.ndarray,
    seg_idx: int,
) -> np.ndarray:
    return fold[seg_idx]


@njit(cache=True, fastmath=True)
def seed_func(
    self: PruneCircTaylorDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> WorldTree:
    leaves = taylor.poly_taylor_seed(
        self.param_arr,
        self.dparams,
        self.poly_order,
        coord_init,
    )
    n_leaves = len(leaves)
    folds = fold_segment.reshape((n_leaves, *fold_segment.shape[-2:]))
    scores = np.zeros(n_leaves, dtype=np.float32)
    scores = self.score(folds)
    backtracks = np.zeros((n_leaves, self.poly_order + 2), dtype=np.int32)
    return WorldTree(leaves, folds, scores, backtracks)


@njit(cache=True, fastmath=True)
def seed_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> WorldTreeComplex:
    leaves = taylor.poly_taylor_seed(
        self.param_arr,
        self.dparams,
        self.poly_order,
        coord_init,
    )
    n_leaves = len(leaves)
    folds = fold_segment.reshape((n_leaves, *fold_segment.shape[-2:]))
    scores = np.zeros(n_leaves, dtype=np.float32)
    scores = self.score(folds)
    backtracks = np.zeros((n_leaves, self.poly_order + 2), dtype=np.int32)
    return WorldTreeComplex(leaves, folds, scores, backtracks)


@njit(cache=True, fastmath=True)
def branch_func(
    self: PruneCircTaylorDPFuncts | PruneCircTaylorComplexDPFuncts,
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    # Pass coord_cur for moving grid, coord_cur_fixed for fixed grid
    return circular.circ_taylor_branch_batch(
        leaves_batch,
        coord_cur,
        self.nbins,
        self.eta,
        self.poly_order,
        self.branch_max,
        self.propagator_significance,
    )


@njit(cache=True, fastmath=True)
def validate_func(
    self: PruneCircTaylorDPFuncts | PruneCircTaylorComplexDPFuncts,
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    coord_cur: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    return circular.circ_taylor_validate_batch(
        leaves_batch,
        leaves_origins,
        self.p_orb_min,
        self.x_mass_const,
        self.validation_significance,
    )


@njit(cache=True, fastmath=True)
def get_validation_params_func(
    self: PruneCircTaylorDPFuncts | PruneCircTaylorComplexDPFuncts,
    coord_add: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    return common.get_validation_params(coord_add)


@njit(cache=True, fastmath=True)
def resolve_func(
    self: PruneCircTaylorDPFuncts | PruneCircTaylorComplexDPFuncts,
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if self.use_moving_grid:
        return circular.circ_taylor_resolve_batch(
            leaves_batch,
            coord_add,
            coord_cur,
            coord_init,
            self.param_grid_count_init,
            self.param_limits,
            self.nbins,
            self.propagator_significance,
        )
    return circular.circ_taylor_fixed_resolve_batch(
        leaves_batch,
        coord_add,
        coord_cur,
        coord_init,
        self.param_grid_count_init,
        self.param_limits,
        self.nbins,
        self.propagator_significance,
    )


@njit(cache=True, fastmath=True)
def shift_add_func(
    self: PruneCircTaylorDPFuncts,
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> np.ndarray:
    return common.shift_add_batch(segment_batch, shift_batch, folds, isuggest_batch)


@njit(cache=True, fastmath=True)
def shift_add_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> np.ndarray:
    return common.shift_add_complex_batch(
        segment_batch,
        shift_batch,
        folds,
        isuggest_batch,
    )


@njit(cache=True, fastmath=True)
def score_func(
    self: PruneCircTaylorDPFuncts,
    combined_res_batch: np.ndarray,
) -> np.ndarray:
    return scoring.score_batch_ev_njit(
        combined_res_batch,
        self.score_widths,
        self.score_mode_id,
        self.score_n_harmonics,
        self.kadane_biases,
    )


@njit(cache=True, fastmath=True)
def score_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    combined_res_batch: np.ndarray,
) -> np.ndarray:
    return scoring.score_batch_ev_complex_njit(
        combined_res_batch,
        self.score_widths,
        self.score_mode_id,
        self.score_n_harmonics,
        self.kadane_biases,
    )


@njit(cache=True, fastmath=True)
def transform_func(
    self: PruneCircTaylorDPFuncts | PruneCircTaylorComplexDPFuncts,
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
) -> np.ndarray:
    if self.use_moving_grid:
        return circular.circ_taylor_transform_batch(
            leaves_batch,
            coord_next,
            coord_cur,
            self.tiling_strategy,
            self.propagator_significance,
        )
    return leaves_batch


@njit(cache=True, fastmath=True)
def get_transform_matrix_func(
    self: PruneCircTaylorDPFuncts | PruneCircTaylorComplexDPFuncts,
    coord_next: tuple[float, float],
    coord_prev: tuple[float, float],
) -> np.ndarray:
    return common.get_trans_matrix(coord_next, coord_prev)


@njit(cache=True, fastmath=True)
def pack_func(
    self: PruneCircTaylorDPFuncts | PruneCircTaylorComplexDPFuncts,
    data: np.ndarray,
) -> np.ndarray:
    return common.pack(data)


@njit(cache=True, fastmath=True)
def ascend_func(
    self: PruneCircTaylorDPFuncts,
    world_tree: WorldTree,
    dyp_folds: np.ndarray,
    load_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    idx_segments: np.ndarray,
    coord_segments: np.ndarray,
    coord_cur: tuple[float, float],
    batch_size: int,
) -> None:
    n_leaves = world_tree.valid_size
    if n_leaves <= 0:
        return
    # Copy the scores to the scores_ep
    world_tree.scores_ep[:n_leaves] = world_tree.scores[:n_leaves]
    batch_size = max(1, min(batch_size, n_leaves))
    # Loop over branches in batches
    for i_batch_start in range(0, n_leaves, batch_size):
        i_batch_end = min(i_batch_start + batch_size, n_leaves)
        batch_leaves = world_tree.leaves[i_batch_start:i_batch_end]
        param_idx_arr, relative_phase_arr = circular.circ_taylor_ascend_resolve_batch(
            batch_leaves,
            coord_segments,
            coord_cur,
            self.param_grid_count_init,
            self.param_limits,
            self.nbins,
            self.propagator_significance,
        )
        batch_combined_res = common.shift_add_ascend_batch(
            dyp_folds,
            load_func,
            idx_segments,
            param_idx_arr,
            relative_phase_arr,
        )
        batch_scores = scoring.score_batch_ev_njit(
            batch_combined_res,
            self.score_widths,
            self.score_mode_id,
            self.score_n_harmonics,
            self.kadane_biases,
        )
        world_tree.folds[i_batch_start:i_batch_end] = batch_combined_res
        world_tree.scores[i_batch_start:i_batch_end] = batch_scores


@njit(cache=True, fastmath=True)
def ascend_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    world_tree: WorldTreeComplex,
    dyp_folds: np.ndarray,
    load_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    idx_segments: np.ndarray,
    coord_segments: np.ndarray,
    coord_cur: tuple[float, float],
    batch_size: int,
) -> None:
    n_leaves = world_tree.valid_size
    if n_leaves <= 0:
        return
    # Copy the scores to the scores_ep
    world_tree.scores_ep[:n_leaves] = world_tree.scores[:n_leaves]
    batch_size = max(1, min(batch_size, n_leaves))
    # Loop over branches in batches
    for i_batch_start in range(0, n_leaves, batch_size):
        i_batch_end = min(i_batch_start + batch_size, n_leaves)
        batch_leaves = world_tree.leaves[i_batch_start:i_batch_end]
        param_idx_arr, relative_phase_arr = circular.circ_taylor_ascend_resolve_batch(
            batch_leaves,
            coord_segments,
            coord_cur,
            self.param_grid_count_init,
            self.param_limits,
            self.nbins,
            self.propagator_significance,
        )
        batch_combined_res = common.shift_add_ascend_complex_batch(
            dyp_folds,
            load_func,
            idx_segments,
            param_idx_arr,
            relative_phase_arr,
        )
        batch_scores = scoring.score_batch_ev_complex_njit(
            batch_combined_res,
            self.score_widths,
            self.score_mode_id,
            self.score_n_harmonics,
            self.kadane_biases,
        )
        world_tree.folds[i_batch_start:i_batch_end] = batch_combined_res
        world_tree.scores[i_batch_start:i_batch_end] = batch_scores


@njit(cache=True, fastmath=True)
def report_func(
    self: PruneCircTaylorDPFuncts | PruneCircTaylorComplexDPFuncts,
    leaves_batch: np.ndarray,
    coord_report: tuple[float, float],
    coord_end: tuple[float, float],
) -> np.ndarray:
    if not self.use_moving_grid:
        # Shift to the middle of the segment
        leaves_batch = circular.circ_taylor_transform_batch(
            leaves_batch,
            coord_report,
            coord_end,
            self.tiling_strategy,
            self.propagator_significance,
        )
    return taylor.poly_taylor_report_batch(leaves_batch)


@overload_method(PruneCircTaylorDPFunctsTemplate, "load")
def ol_load_func(
    self: PruneCircTaylorDPFuncts,
    fold: np.ndarray,
    seg_idx: int,
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        fold: np.ndarray,
        seg_idx: int,
    ) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    return impl


@overload_method(PruneCircTaylorDPFunctsTemplate, "seed")
def ol_seed_func(
    self: PruneCircTaylorDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> WorldTree:
        return seed_func(self, fold_segment, coord_init)

    return impl


@overload_method(PruneCircTaylorDPFunctsTemplate, "branch")
def ol_branch_func(
    self: PruneCircTaylorDPFuncts,
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        leaves_batch: np.ndarray,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, leaves_batch, coord_cur, coord_prev)

    return impl


@overload_method(PruneCircTaylorDPFunctsTemplate, "validate")
def ol_validate_func(
    self: PruneCircTaylorDPFuncts,
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        leaves_batch: np.ndarray,
        leaves_origins: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return validate_func(self, leaves_batch, leaves_origins, coord_cur)

    return impl


@overload_method(PruneCircTaylorDPFunctsTemplate, "get_validation_params")
def ol_get_validation_params_func(
    self: PruneCircTaylorDPFuncts,
    coord_add: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return get_validation_params_func(self, coord_add)

    return impl


@overload_method(PruneCircTaylorDPFunctsTemplate, "resolve")
def ol_resolve_func(
    self: PruneCircTaylorDPFuncts,
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        leaves_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_cur: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_func(self, leaves_batch, coord_add, coord_cur, coord_init)

    return impl


@overload_method(PruneCircTaylorDPFunctsTemplate, "shift_add")
def ol_shift_add_func(
    self: PruneCircTaylorDPFuncts,
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        segment_batch: np.ndarray,
        shift_batch: np.ndarray,
        folds: np.ndarray,
        isuggest_batch: np.ndarray,
    ) -> np.ndarray:
        return shift_add_func(self, segment_batch, shift_batch, folds, isuggest_batch)

    return impl


@overload_method(PruneCircTaylorDPFunctsTemplate, "score")
def ol_score_func(
    self: PruneCircTaylorDPFuncts,
    combined_res_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        combined_res_batch: np.ndarray,
    ) -> np.ndarray:
        return score_func(self, combined_res_batch)

    return impl


@overload_method(PruneCircTaylorDPFunctsTemplate, "transform")
def ol_transform_func(
    self: PruneCircTaylorDPFuncts,
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        leaves_batch: np.ndarray,
        coord_next: tuple[float, float],
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        return transform_func(self, leaves_batch, coord_next, coord_cur)

    return impl


@overload_method(PruneCircTaylorDPFunctsTemplate, "get_transform_matrix")
def ol_get_transform_matrix_func(
    self: PruneCircTaylorDPFuncts,
    coord_next: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        coord_next: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_next, coord_prev)

    return impl


@overload_method(PruneCircTaylorDPFunctsTemplate, "pack")
def ol_pack_func(self: PruneCircTaylorDPFuncts, data: np.ndarray) -> types.FunctionType:
    def impl(self: PruneCircTaylorDPFuncts, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)

    return impl


@overload_method(PruneCircTaylorDPFunctsTemplate, "ascend")
def ol_ascend_func(
    self: PruneCircTaylorDPFuncts,
    world_tree: WorldTree,
    dyp_folds: np.ndarray,
    load_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    idx_segments: np.ndarray,
    coord_segments: np.ndarray,
    coord_cur: tuple[float, float],
    batch_size: int,
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        world_tree: WorldTree,
        dyp_folds: np.ndarray,
        load_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        idx_segments: np.ndarray,
        coord_segments: np.ndarray,
        coord_cur: tuple[float, float],
        batch_size: int,
    ) -> None:
        return ascend_func(
            self,
            world_tree,
            dyp_folds,
            load_func,
            idx_segments,
            coord_segments,
            coord_cur,
            batch_size,
        )

    return impl


@overload_method(PruneCircTaylorDPFunctsTemplate, "report")
def ol_report_func(
    self: PruneCircTaylorDPFuncts,
    leaves_batch: np.ndarray,
    coord_report: tuple[float, float],
    coord_end: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        leaves_batch: np.ndarray,
        coord_report: tuple[float, float],
        coord_end: tuple[float, float],
    ) -> np.ndarray:
        return report_func(self, leaves_batch, coord_report, coord_end)

    return impl


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "load")
def ol_load_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    fold: np.ndarray,
    seg_idx: int,
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        fold: np.ndarray,
        seg_idx: int,
    ) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    return impl


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "seed")
def ol_seed_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> WorldTreeComplex:
        return seed_complex_func(self, fold_segment, coord_init)

    return impl


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "branch")
def ol_branch_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        leaves_batch: np.ndarray,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, leaves_batch, coord_cur, coord_prev)

    return impl


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "validate")
def ol_validate_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        leaves_batch: np.ndarray,
        leaves_origins: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return validate_func(self, leaves_batch, leaves_origins, coord_cur)

    return impl


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "get_validation_params")
def ol_get_validation_params_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    coord_add: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return get_validation_params_func(self, coord_add)

    return impl


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "resolve")
def ol_resolve_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        leaves_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_cur: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_func(self, leaves_batch, coord_add, coord_cur, coord_init)

    return impl


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "shift_add")
def ol_shift_add_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        segment_batch: np.ndarray,
        shift_batch: np.ndarray,
        folds: np.ndarray,
        isuggest_batch: np.ndarray,
    ) -> np.ndarray:
        return shift_add_complex_func(
            self,
            segment_batch,
            shift_batch,
            folds,
            isuggest_batch,
        )

    return impl


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "score")
def ol_score_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    combined_res_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        combined_res_batch: np.ndarray,
    ) -> np.ndarray:
        return score_complex_func(self, combined_res_batch)

    return impl


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "transform")
def ol_transform_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        leaves_batch: np.ndarray,
        coord_next: tuple[float, float],
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        return transform_func(self, leaves_batch, coord_next, coord_cur)

    return impl


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "get_transform_matrix")
def ol_get_transform_matrix_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    coord_next: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        coord_next: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_next, coord_prev)

    return impl


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "pack")
def ol_pack_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    data: np.ndarray,
) -> types.FunctionType:
    def impl(self: PruneCircTaylorComplexDPFuncts, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)

    return impl


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "ascend")
def ol_ascend_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    world_tree: WorldTreeComplex,
    dyp_folds: np.ndarray,
    load_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    idx_segments: np.ndarray,
    coord_segments: np.ndarray,
    coord_cur: tuple[float, float],
    batch_size: int,
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        world_tree: WorldTreeComplex,
        dyp_folds: np.ndarray,
        load_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        idx_segments: np.ndarray,
        coord_segments: np.ndarray,
        coord_cur: tuple[float, float],
        batch_size: int,
    ) -> None:
        return ascend_complex_func(
            self,
            world_tree,
            dyp_folds,
            load_func,
            idx_segments,
            coord_segments,
            coord_cur,
            batch_size,
        )

    return impl


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "report")
def ol_report_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    leaves_batch: np.ndarray,
    coord_report: tuple[float, float],
    coord_end: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        leaves_batch: np.ndarray,
        coord_report: tuple[float, float],
        coord_end: tuple[float, float],
    ) -> np.ndarray:
        return report_func(self, leaves_batch, coord_report, coord_end)

    return impl


@dataclass(frozen=True, kw_only=True)
class Circular4DContinuationResult:
    """Surviving 4D physical circular leaves after Taylor -> 4D stage handoff."""

    leaves_4d: np.ndarray
    scores: np.ndarray
    folds: np.ndarray
    coord_epoch: tuple[float, float]
    coord_init: tuple[float, float]
    n_levels_continued: int


def run_circular_4d_continuation(
    dyp_fold: np.ndarray,
    taylor_leaves: np.ndarray,
    taylor_folds: np.ndarray,
    *,
    transition_level: int,
    nsegments: int,
    tseg: float,
    param_grid_count_init: np.ndarray,
    param_limits: np.ndarray,
    cfg: PulsarSearchConfig,
    ref_seg: int = 0,
    threshold_scheme: np.ndarray | None = None,
    use_moving_grid: bool = True,
    max_sugg: int = 2**14,
) -> Circular4DContinuationResult:
    """Continue circular-Taylor survivors in exact 4D physical circular coordinates.

    Converts surviving 7-slot circular-Taylor leaves at ``transition_level``
    into 6-slot 4D circular leaves ``[omega, f0, x_sin_phi, x_cos_phi, 0, 3]``
    and integrates remaining segments ``transition_level + 1 .. nsegments - 1``
    branching only ``(omega, f0)``.
    """
    scheme = MiddleOutScheme(
        nsegments=nsegments,
        ref_idx=ref_seg,
        tsegment=tseg,
        stride=1,
    )
    coord_init = scheme.get_coord(0)
    coord_epoch = scheme.get_previous_coord(transition_level + 1, use_moving_grid)

    leaves_4d, origins = circular.convert_circ_taylor_to_4d_leaves(
        np.ascontiguousarray(taylor_leaves, dtype=np.float64),
        float(cfg.p_orb_min),
    )
    if leaves_4d.shape[0] == 0:
        return Circular4DContinuationResult(
            leaves_4d=leaves_4d,
            scores=np.empty(0, dtype=np.float32),
            folds=np.empty((0, 2, cfg.nbins), dtype=np.float32),
            coord_epoch=coord_epoch,
            coord_init=coord_init,
            n_levels_continued=0,
        )

    cur_folds = np.ascontiguousarray(taylor_folds[origins], dtype=np.float32)
    score_mode_id = getattr(cfg, "score_mode_id", 0)
    score_n_harmonics = getattr(cfg, "score_n_harmonics", 16)
    kadane_biases = getattr(
        cfg,
        "kadane_biases",
        scoring.generate_kadane_biases(cfg.nbins),
    )
    cur_scores = scoring.score_batch_ev_njit(
        cur_folds,
        cfg.score_widths,
        score_mode_id,
        score_n_harmonics,
        kadane_biases,
    )
    p_limits_2d = np.ascontiguousarray(param_limits[-2:], dtype=np.float64)
    p_counts_2d = np.ascontiguousarray(param_grid_count_init[-2:], dtype=np.int64)
    n_acc = int(p_counts_2d[0])
    n_freq = int(p_counts_2d[1])
    fold_5d = np.ascontiguousarray(
        dyp_fold.reshape(nsegments, -1, n_acc, n_freq, 2, cfg.nbins)[:, 0],
        dtype=np.float32,
    )

    n_continued = 0
    for lvl in range(transition_level + 1, nsegments):
        seg_idx = scheme.get_segment_idx(lvl)
        coord_cur = scheme.get_current_coord(lvl, use_moving_grid)
        coord_prev = scheme.get_previous_coord(lvl, use_moving_grid)
        coord_add = scheme.get_segment_coord(lvl)

        child_leaves, child_origins = circular.cir_physical_branch_batch(
            leaves_4d,
            coord_cur,
            coord_prev,
            cfg.nbins,
            cfg.eta,
            cfg.branch_max,
        )
        if child_leaves.shape[0] == 0:
            leaves_4d = child_leaves
            cur_folds = np.empty((0, 2, cfg.nbins), dtype=np.float32)
            cur_scores = np.empty(0, dtype=np.float32)
            break

        pidx, pshift = circular.poly_circular_resolve_epoch_batch(
            child_leaves,
            coord_add,
            coord_epoch,
            coord_init,
            p_counts_2d,
            p_limits_2d,
            cfg.nbins,
        )
        fold_seg = fold_5d[seg_idx]
        loaded = np.ascontiguousarray(
            fold_seg[pidx[:, 0], pidx[:, 1]],
            dtype=np.float32,
        )
        combined = common.shift_add_batch(loaded, pshift, cur_folds, child_origins)
        scores = scoring.score_batch_ev_njit(
            combined,
            cfg.score_widths,
            score_mode_id,
            score_n_harmonics,
            kadane_biases,
        )

        thr = (
            float(threshold_scheme[lvl - 1])
            if threshold_scheme is not None and lvl - 1 < len(threshold_scheme)
            else -np.inf
        )
        keep = np.flatnonzero(scores >= thr)
        if keep.size == 0:
            leaves_4d = np.empty((0, 6, 2), dtype=np.float64)
            cur_folds = np.empty((0, 2, cfg.nbins), dtype=np.float32)
            cur_scores = np.empty(0, dtype=np.float32)
            break
        if keep.size > max_sugg:
            top = np.argsort(scores[keep])[::-1][:max_sugg]
            keep = keep[top]

        leaves_4d = np.ascontiguousarray(child_leaves[keep], dtype=np.float64)
        cur_folds = np.ascontiguousarray(combined[keep], dtype=np.float32)
        cur_scores = np.ascontiguousarray(scores[keep], dtype=np.float32)
        n_continued += 1

    if cur_scores.size > 0:
        order = np.argsort(cur_scores)[::-1]
        leaves_4d = leaves_4d[order]
        cur_folds = cur_folds[order]
        cur_scores = cur_scores[order]

    return Circular4DContinuationResult(
        leaves_4d=leaves_4d,
        scores=cur_scores,
        folds=cur_folds,
        coord_epoch=coord_epoch,
        coord_init=coord_init,
        n_levels_continued=n_continued,
    )
