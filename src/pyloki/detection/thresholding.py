# ruff: noqa: ARG001

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import attrs
import h5py
import numpy as np
import numpy.typing as npt
from numba import njit, prange, typed, types
from numba.experimental import structref
from numba.extending import overload
from rich.progress import track

from pyloki.detection import scoring
from pyloki.detection.schemes import StateInfo, StatesInfo, bound_scheme, trials_scheme
from pyloki.simulation.pulse import generate_folded_profile
from pyloki.utils import np_utils
from pyloki.utils.misc import CONSOLE, get_logger
from pyloki.utils.timing import Timer

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyloki.detection.scoring import ScoreMode

logger = get_logger(__name__)


@njit("i8[::1](i8[::1], i8, i8)", cache=True, fastmath=True)
def neighbouring_indices(
    beam_indices: npt.NDArray[np.int64],
    target_idx: int,
    num: int,
) -> npt.NDArray[np.int64]:
    # Find the index of target_idx in beam_indices
    target_beam_idx = np.searchsorted(beam_indices, target_idx)

    # Calculate the window around the target
    left = max(0, int(target_beam_idx - num // 2))
    right = min(len(beam_indices), left + num)

    # Adjust left if we're at the right edge
    left = max(0, right - num)

    return beam_indices[left:right]


@njit(cache=True, fastmath=True)
def simulate_folds(
    folds: npt.NDArray[np.float32],
    var_cur: float,
    profile: npt.NDArray[np.float32],
    rng: np.random.Generator,
    bias_snr: float = 0,
    var_add: float = 1,
    ntrials_min: int = 1024,
) -> tuple[npt.NDArray[np.float32], float]:
    """Simulate folded profiles by adding signal + noise to the template profile.

    Parameters
    ----------
    folds : NDArray[np.float32]
        2D Array of folded data with shape (ntrials, nbins).
    var_cur : float
        Current variance of the noise in the folded profiles.
    profile : NDArray[np.float32]
        Normalized signal profile with the same units as the bias_snr.
    rng : np.random.Generator
        Random number generator.
    bias_snr : float, optional
        Bias signal-to-noise ratio.
    var_add : float, optional
        Variance of the added noise, by default 1.
    ntrials_min : int, optional
        Minimum number of trials to simulate, by default 1024.

    Returns
    -------
    tuple[NDArray[np.float32], float]
        Array of simulated folded data and the updated variance.
    """
    ntrials_prev, nbins = folds.shape
    if ntrials_prev == 0:
        msg = "No trials in the input folds"
        raise ValueError(msg)
    repeat_factor = int(np.ceil(ntrials_min / ntrials_prev))
    ntrials = ntrials_prev * repeat_factor
    folds_sim = np.empty((ntrials, nbins), dtype=np.float32)
    # Pre-compute template profile scaling once
    profile_scaled = profile * bias_snr
    noise_std = np.sqrt(var_add)
    noise = rng.normal(0, noise_std, (ntrials, nbins)).astype(np.float32)
    for i in range(ntrials):
        orig_trial = i % ntrials_prev
        for j in range(nbins):
            folds_sim[i, j] = folds[orig_trial, j] + noise[i, j] + profile_scaled[j]
    var_sim = var_cur + var_add
    return folds_sim, var_sim


@njit("f4[:,::1](f4[:,::1], f8, f8, f8, f8)", cache=True, fastmath=True)
def prune_folds(
    folds: npt.NDArray[np.float32],
    var_cur: float,
    snr_thresh: float,
    ducy_max: float = 0.2,
    wtsp: float = 1.0,
) -> npt.NDArray[np.float32]:
    """Prune the folds that did not pass the threshold.

    Parameters
    ----------
    folds : NDArray[np.float32]
        2D Array of folded data with shape (ntrials, nbins).
    var_cur : float
        Current variance of the noise in the folded profiles.
    snr_thresh : float
        Signal-to-noise ratio threshold.
    ducy_max : float, optional
        Maximum duty cycle for boxcar search, by default 0.2

    Returns
    -------
    NDArray[np.float32]
        Pruned folds that passed the threshold.
    """
    folds_var = var_cur * np.ones_like(folds)
    folds_norm = folds / np.sqrt(folds_var)
    folds_norm = folds_norm.astype(np.float32)
    widths = scoring.generate_box_width_trials(
        folds.shape[1],
        ducy_max=ducy_max,
        wtsp=wtsp,
    )
    scores_arr = np_utils.nb_max(
        scoring.boxcar_snr_2d_serial(folds_norm, widths, 1.0),
        axis=1,
    )
    good_scores_idx = np.nonzero(scores_arr > snr_thresh)[0]
    return np.ascontiguousarray(folds[good_scores_idx])


@njit("Tuple((f8, f4[:,::1]))(f4[:,::1], f8, f8, f8, f8)", cache=True, fastmath=True)
def prune_folds_survival(
    folds: npt.NDArray[np.float32],
    var_cur: float,
    survive_prob: float,
    ducy_max: float = 0.2,
    wtsp: float = 1.0,
) -> tuple[float, npt.NDArray[np.float32]]:
    if folds.shape[0] == 0:
        return np.inf, np.empty((0, folds.shape[1]), dtype=np.float32)
    folds_var = var_cur * np.ones_like(folds)
    folds_norm = folds / np.sqrt(folds_var)
    folds_norm = folds_norm.astype(np.float32)
    widths = scoring.generate_box_width_trials(
        folds.shape[1],
        ducy_max=ducy_max,
        wtsp=wtsp,
    )
    scores_arr = np_utils.nb_max(
        scoring.boxcar_snr_2d_serial(folds_norm, widths, 1.0),
        axis=1,
    )
    n_total = len(scores_arr)
    n_surviving = min(n_total, int(survive_prob * n_total))
    if n_surviving <= 0:
        return (
            float(np.max(scores_arr)),
            np.empty((0, folds.shape[1]), dtype=np.float32),
        )
    good_scores_idx = np.flip(np.argsort(scores_arr))[:n_surviving]
    threshold = float(scores_arr[good_scores_idx[-1]])
    return threshold, np.ascontiguousarray(folds[good_scores_idx])


@structref.register
class FoldsTemplate(types.StructRef):
    pass


class Folds(structref.StructRefProxy):
    """Folds class to keep track of the folded profiles.

    Parameters
    ----------
    folds_h0 : np.ndarray
        Surviving H0 folded profiles.
    folds_h1 : np.ndarray
        Surviving H1 folded profiles.
    scores_h0 : np.ndarray, optional
        Per-trial max boxcar SNR for H0 (improved mode only).
    scores_h1 : np.ndarray, optional
        Per-trial max boxcar SNR for H1 (improved mode only).
    variance : float, optional
        Variance of the noise in the folded profiles, by default 1
    """

    def __new__(
        cls,
        folds_h0: npt.NDArray[np.float32],
        folds_h1: npt.NDArray[np.float32],
        third: npt.NDArray[np.float32] | float = 1.0,
        fourth: npt.NDArray[np.float32] | None = None,
        fifth: float | None = None,
    ) -> Self:
        """Create a new instance of Folds."""
        if fourth is None:
            if isinstance(third, bool) or not isinstance(
                third,
                (int, float, np.integer, np.floating),
            ):
                msg = "Legacy Folds expects variance as the third argument"
                raise TypeError(msg)
            return folds_init_legacy(folds_h0, folds_h1, float(third))
        if (
            fifth is None
            or isinstance(fifth, bool)
            or not isinstance(fifth, (int, float, np.integer, np.floating))
        ):
            msg = "Scored Folds requires variance as the fifth argument"
            raise TypeError(msg)
        if not isinstance(third, np.ndarray):
            msg = "Scored Folds expects score arrays as third and fourth arguments"
            raise TypeError(msg)
        return folds_init(folds_h0, folds_h1, third, fourth, float(fifth))

    @property
    @njit(cache=True, fastmath=True)
    def folds_h0(self) -> npt.NDArray[np.float32]:
        return self.folds_h0

    @property
    @njit(cache=True, fastmath=True)
    def folds_h1(self) -> npt.NDArray[np.float32]:
        return self.folds_h1

    @property
    @njit(cache=True, fastmath=True)
    def scores_h0(self) -> npt.NDArray[np.float32]:
        return self.scores_h0

    @property
    @njit(cache=True, fastmath=True)
    def scores_h1(self) -> npt.NDArray[np.float32]:
        return self.scores_h1

    @property
    @njit(cache=True, fastmath=True)
    def variance(self) -> float:
        return self.variance

    @property
    @njit(cache=True, fastmath=True)
    def is_empty(self) -> int:
        return self.is_empty


fields_folds = [
    ("folds_h0", types.f4[:, ::1]),
    ("folds_h1", types.f4[:, ::1]),
    ("scores_h0", types.f4[::1]),
    ("scores_h1", types.f4[::1]),
    ("variance", types.f8),
    ("is_empty", types.int8),
]

structref.define_boxing(FoldsTemplate, Folds)
FoldsType = FoldsTemplate(fields_folds)


@njit(
    FoldsType(
        types.f4[:, ::1],
        types.f4[:, ::1],
        types.f4[::1],
        types.f4[::1],
        types.f8,
    ),
    cache=True,
    fastmath=True,
)
def folds_init(
    folds_h0: npt.NDArray[np.float32],
    folds_h1: npt.NDArray[np.float32],
    scores_h0: npt.NDArray[np.float32],
    scores_h1: npt.NDArray[np.float32],
    variance: float,
) -> Folds:
    self = structref.new(FoldsType)  # ty: ignore[missing-argument]
    self.folds_h0 = folds_h0
    self.folds_h1 = folds_h1
    self.scores_h0 = scores_h0
    self.scores_h1 = scores_h1
    self.variance = variance
    self.is_empty = 0 if (folds_h0.shape[0] > 0 and folds_h1.shape[0] > 0) else 1
    return self


@njit(
    FoldsType(types.f4[:, ::1], types.f4[:, ::1], types.f8),
    cache=True,
    fastmath=True,
)
def folds_init_legacy(
    folds_h0: npt.NDArray[np.float32],
    folds_h1: npt.NDArray[np.float32],
    variance: float,
) -> Folds:
    scores_h0 = np.empty(0, dtype=np.float32)
    scores_h1 = np.empty(0, dtype=np.float32)
    return folds_init(folds_h0, folds_h1, scores_h0, scores_h1, variance)


@overload(Folds)
def overload_folds_legacy(
    folds_h0: npt.NDArray[np.float32],
    folds_h1: npt.NDArray[np.float32],
    variance: float = 1.0,
) -> types.FunctionType:
    def impl(
        folds_h0: npt.NDArray[np.float32],
        folds_h1: npt.NDArray[np.float32],
        variance: float = 1.0,
    ) -> Folds:
        return folds_init_legacy(folds_h0, folds_h1, variance)

    return impl


@overload(Folds)
def overload_folds_scored(
    folds_h0: npt.NDArray[np.float32],
    folds_h1: npt.NDArray[np.float32],
    scores_h0: npt.NDArray[np.float32],
    scores_h1: npt.NDArray[np.float32],
    variance: float,
) -> types.FunctionType:
    def impl(
        folds_h0: npt.NDArray[np.float32],
        folds_h1: npt.NDArray[np.float32],
        scores_h0: npt.NDArray[np.float32],
        scores_h1: npt.NDArray[np.float32],
        variance: float,
    ) -> Folds:
        return folds_init(folds_h0, folds_h1, scores_h0, scores_h1, variance)

    return impl


@njit(cache=True, fastmath=True)
def create_empty_folds() -> Folds:
    empty_h0 = np.empty((0, 1), dtype=np.float32)  # Shape (0, nbins) indicates empty
    empty_h1 = np.empty((0, 1), dtype=np.float32)
    scores_h0 = np.empty(0, dtype=np.float32)
    scores_h1 = np.empty(0, dtype=np.float32)
    return folds_init(empty_h0, empty_h1, scores_h0, scores_h1, 0.0)


state_dtype = np.dtype(
    [
        ("success_h0", np.float32),
        ("success_h1", np.float32),
        ("complexity", np.float32),
        ("complexity_cumul", np.float32),
        ("success_h1_cumul", np.float32),
        ("nbranches", np.float32),
        ("threshold", np.float32),
        ("cost", np.float32),
        ("threshold_prev", np.float32),
        ("success_h1_cumul_prev", np.float32),
        ("is_empty", np.bool_),
    ],
)


@njit(cache=True, fastmath=True)
def get_next_state(
    state_cur: npt.NDArray,
    threshold: float,
    success_h0: float,
    success_h1: float,
    nbranches: float,
) -> npt.NDArray:
    nleaves_next = state_cur["complexity"] * nbranches
    nleaves_surv = nleaves_next * success_h0
    complexity_cumul = state_cur["complexity_cumul"] + nleaves_next
    success_h1_cumul = state_cur["success_h1_cumul"] * success_h1

    # Create a new state struct
    state_next = np.zeros(1, dtype=state_dtype)  # Single struct
    state_next[0]["success_h0"] = success_h0
    state_next[0]["success_h1"] = success_h1
    state_next[0]["complexity"] = nleaves_surv
    state_next[0]["complexity_cumul"] = complexity_cumul
    state_next[0]["success_h1_cumul"] = success_h1_cumul
    state_next[0]["nbranches"] = nbranches
    state_next[0]["threshold"] = threshold
    state_next[0]["cost"] = (
        complexity_cumul / success_h1_cumul if success_h1_cumul > 0.0 else np.inf
    )
    # For backtracking
    state_next[0]["threshold_prev"] = state_cur["threshold"]
    state_next[0]["success_h1_cumul_prev"] = state_cur["success_h1_cumul"]
    return state_next


@njit(cache=True, fastmath=True)
def gen_next_using_thresh(
    state_cur: npt.NDArray,
    folds_cur: Folds,
    threshold: float,
    nbranches: float,
    bias_snr: float,
    profile: npt.NDArray[np.float32],
    rng: np.random.Generator,
    ntrials: int = 1024,
    ducy_max: float = 0.2,
    wtsp: float = 1.0,
) -> tuple[npt.NDArray, Folds]:
    folds_h0, variance = simulate_folds(
        folds_cur.folds_h0,
        folds_cur.variance,
        profile,
        rng,
        bias_snr=0,
        var_add=1,
        ntrials_min=ntrials,
    )
    folds_h0_pruned = prune_folds(
        folds_h0,
        variance,
        threshold,
        ducy_max=ducy_max,
        wtsp=wtsp,
    )
    success_h0 = len(folds_h0_pruned) / len(folds_h0)
    folds_h1, variance = simulate_folds(
        folds_cur.folds_h1,
        folds_cur.variance,
        profile,
        rng,
        bias_snr=bias_snr,
        var_add=1,
        ntrials_min=ntrials,
    )
    folds_h1_pruned = prune_folds(
        folds_h1,
        variance,
        threshold,
        ducy_max=ducy_max,
        wtsp=wtsp,
    )
    success_h1 = len(folds_h1_pruned) / len(folds_h1)
    state_next = get_next_state(state_cur, threshold, success_h0, success_h1, nbranches)
    folds_next = Folds(folds_h0_pruned, folds_h1_pruned, variance)
    return state_next, folds_next


@njit(cache=True, fastmath=True)
def gen_next_using_surv_prob(
    state_cur: npt.NDArray,
    folds_cur: Folds,
    surv_prob_h0: float,
    nbranches: float,
    bias_snr: float,
    profile: npt.NDArray[np.float32],
    rng: np.random.Generator,
    ntrials: int = 1024,
    ducy_max: float = 0.2,
    wtsp: float = 1.0,
) -> tuple[npt.NDArray, Folds]:
    folds_h0, variance = simulate_folds(
        folds_cur.folds_h0,
        folds_cur.variance,
        profile,
        rng,
        bias_snr=0,
        var_add=1,
        ntrials_min=ntrials,
    )
    thres_h0, folds_h0_pruned = prune_folds_survival(
        folds_h0,
        variance,
        surv_prob_h0,
        ducy_max=ducy_max,
        wtsp=wtsp,
    )
    success_h0 = len(folds_h0_pruned) / len(folds_h0)
    folds_h1, variance = simulate_folds(
        folds_cur.folds_h1,
        folds_cur.variance,
        profile,
        rng,
        bias_snr=bias_snr,
        var_add=1,
        ntrials_min=ntrials,
    )
    folds_h1_pruned = prune_folds(
        folds_h1,
        variance,
        thres_h0,
        ducy_max=ducy_max,
        wtsp=wtsp,
    )
    success_h1 = len(folds_h1_pruned) / len(folds_h1)
    state_next = get_next_state(state_cur, thres_h0, success_h0, success_h1, nbranches)
    folds_next = Folds(folds_h0_pruned, folds_h1_pruned, variance)
    return state_next, folds_next


@njit("f4[::1](f4[:,::1], f8, f8, f8)", cache=True, fastmath=True)
def fold_max_scores(
    folds: npt.NDArray[np.float32],
    var_cur: float,
    ducy_max: float = 0.2,
    wtsp: float = 1.0,
) -> npt.NDArray[np.float32]:
    """Per-trial max boxcar SNR (threshold-independent part of pruning)."""
    folds_var = var_cur * np.ones_like(folds)
    folds_norm = folds / np.sqrt(folds_var)
    folds_norm = folds_norm.astype(np.float32)
    widths = scoring.generate_box_width_trials(
        folds.shape[1],
        ducy_max=ducy_max,
        wtsp=wtsp,
    )
    return np_utils.nb_max(
        scoring.boxcar_snr_2d_serial(folds_norm, widths, 1.0),
        axis=1,
    )


@njit(cache=True, fastmath=True)
def transition_state(
    state_cur: npt.NDArray,
    folds_cur: Folds,
    threshold: float,
    nbranches: float,
) -> tuple[npt.NDArray, Folds]:
    good_scores_idx_h0 = np.nonzero(folds_cur.scores_h0 > threshold)[0]
    folds_h0_pruned = np.ascontiguousarray(folds_cur.folds_h0[good_scores_idx_h0])
    scores_h0_pruned = np.ascontiguousarray(folds_cur.scores_h0[good_scores_idx_h0])
    success_h0 = len(folds_h0_pruned) / len(folds_cur.folds_h0)
    good_scores_idx_h1 = np.nonzero(folds_cur.scores_h1 > threshold)[0]
    folds_h1_pruned = np.ascontiguousarray(folds_cur.folds_h1[good_scores_idx_h1])
    scores_h1_pruned = np.ascontiguousarray(folds_cur.scores_h1[good_scores_idx_h1])
    success_h1 = len(folds_h1_pruned) / len(folds_cur.folds_h1)
    state_next = get_next_state(state_cur, threshold, success_h0, success_h1, nbranches)
    folds_next = Folds(
        folds_h0_pruned,
        folds_h1_pruned,
        scores_h0_pruned,
        scores_h1_pruned,
        folds_cur.variance,
    )
    return state_next, folds_next


@njit(cache=True, parallel=True, fastmath=True)
def pre_simulate_stage_folds(
    prev_states: npt.NDArray,
    folds_in: list[Folds],
    beam_idx_prev: npt.NDArray,
    bias_snr: float,
    profile: npt.NDArray[np.float32],
    rngs: list[np.random.Generator],
    ntrials: int,
    nthresholds: int,
    nprobs: int,
    ducy_max: float,
    wtsp: float,
) -> list[Folds]:
    sim_folds = typed.List.empty_list(FoldsType)
    for _ in range(nthresholds * nprobs):
        sim_folds.append(create_empty_folds())

    n_beam = len(beam_idx_prev)
    for ii in prange(n_beam * nprobs):
        # One generator per iteration, so the result does not depend on which
        # thread runs which iteration. See DynamicThresholdScheme._stage_rngs.
        rng = rngs[np.int64(ii)]  # prange indices are uint64
        kprob = ii % nprobs
        ibeam = ii // nprobs
        jthresh = beam_idx_prev[ibeam]

        prev_state = prev_states[jthresh, kprob]
        if prev_state["is_empty"]:
            continue
        fold_idx = jthresh * nprobs + kprob
        prev_fold_state = folds_in[fold_idx]
        if prev_fold_state.is_empty == 0:
            folds_h0, variance = simulate_folds(
                prev_fold_state.folds_h0,
                prev_fold_state.variance,
                profile,
                rng,
                bias_snr=0.0,
                var_add=1.0,
                ntrials_min=ntrials,
            )
            folds_h1, variance = simulate_folds(
                prev_fold_state.folds_h1,
                prev_fold_state.variance,
                profile,
                rng,
                bias_snr=bias_snr,
                var_add=1.0,
                ntrials_min=ntrials,
            )
            scores_h0 = fold_max_scores(folds_h0, variance, ducy_max, wtsp)
            scores_h1 = fold_max_scores(folds_h1, variance, ducy_max, wtsp)
            sim_folds[fold_idx] = Folds(
                folds_h0,
                folds_h1,
                scores_h0,
                scores_h1,
                variance,
            )
    return sim_folds


@njit(cache=True, parallel=True, fastmath=True)
def run_stage_legacy(
    istage: int,
    beam_idx_cur: npt.NDArray,
    beam_idx_prev: npt.NDArray,
    states: npt.NDArray,
    folds_in: list[Folds],
    folds_out: list[Folds],
    probs: npt.NDArray,
    nbranches: int,
    thresholds: npt.NDArray,
    bias_snr: float,
    profile: npt.NDArray[np.float32],
    rngs: list[np.random.Generator],
    ntrials: int,
    ducy_max: float,
    wtsp: float,
    thres_neigh: int,
) -> None:
    nprobs = len(probs)
    for ibeam_cur in prange(len(beam_idx_cur)):
        # One generator per iteration, so the result does not depend on which
        # thread runs which iteration. See DynamicThresholdScheme._stage_rngs.
        rng = rngs[np.int64(ibeam_cur)]  # prange indices are uint64
        ithres = int(beam_idx_cur[ibeam_cur])
        # Find nearest neighbors in the previous beam
        neighbour_beam_indices = neighbouring_indices(
            beam_idx_prev,
            ithres,
            thres_neigh,
        )
        for jthresh in neighbour_beam_indices:
            for kprob in range(nprobs):
                fold_idx = int(jthresh * nprobs + kprob)
                prev_state = states[istage - 1, jthresh, kprob]
                if prev_state["is_empty"]:
                    continue
                prev_fold_state = folds_in[fold_idx]
                if prev_fold_state.is_empty == 0:
                    cur_state, cur_fold_state = gen_next_using_thresh(
                        prev_state,
                        prev_fold_state,
                        thresholds[ithres],
                        nbranches,
                        bias_snr,
                        profile,
                        rng,
                        ntrials,
                        ducy_max,
                        wtsp,
                    )
                    # Keep `cur_state` as the one-element ARRAY. Pulling the record
                    # out (`cur_state[0]`) and storing it whole writes zeros under
                    # numba, and a zeroed record reads as `is_empty == False`, so the
                    # scheme fills up with states that look real and hold nothing.
                    # Slice assignment is the form that round-trips; see
                    # tests/test_thresholding.py::TestRecordAssignment.
                    iprob_val = np.digitize(cur_state[0]["success_h1_cumul"], probs) - 1
                    iprob = int(iprob_val.item())  # Extract scalar
                    if iprob < 0 or iprob >= nprobs:  # Clamp to valid range
                        continue
                    existing_state = states[istage, ithres, iprob]
                    if (
                        existing_state["is_empty"]
                        or cur_state[0]["complexity_cumul"]
                        < existing_state["complexity_cumul"]
                    ):
                        states[istage, ithres, iprob : iprob + 1] = cur_state
                        folds_out[ithres * nprobs + iprob] = cur_fold_state


@njit(cache=True, parallel=True, fastmath=True)
def run_stage_improved(
    istage: int,
    beam_idx_cur: npt.NDArray,
    beam_idx_prev: npt.NDArray,
    states: npt.NDArray,
    sim_folds: list[Folds],
    folds_out: list[Folds],
    probs: npt.NDArray,
    nbranches: int,
    thresholds: npt.NDArray,
    nprobs: int,
    thres_neigh: int,
) -> None:
    for ibeam_cur in prange(len(beam_idx_cur)):
        ithres = int(beam_idx_cur[ibeam_cur])
        neighbour_beam_indices = neighbouring_indices(
            beam_idx_prev,
            ithres,
            thres_neigh,
        )
        for jthresh in neighbour_beam_indices:
            for kprob in range(nprobs):
                fold_idx = int(jthresh * nprobs + kprob)
                prev_state = states[istage - 1, jthresh, kprob]
                if prev_state["is_empty"]:
                    continue
                prev_sim_fold = sim_folds[fold_idx]
                if prev_sim_fold.is_empty == 0:
                    cur_state, cur_fold_state = transition_state(
                        prev_state,
                        prev_sim_fold,
                        thresholds[ithres],
                        nbranches,
                    )
                    # See run_stage_legacy: keep the array, assign to a slice.
                    iprob_val = np.digitize(cur_state[0]["success_h1_cumul"], probs) - 1
                    iprob = int(iprob_val.item())  # Extract scalar
                    if iprob < 0 or iprob >= nprobs:  # Clamp to valid range
                        continue

                    existing_state = states[istage, ithres, iprob]
                    if (
                        existing_state["is_empty"]
                        or cur_state[0]["complexity_cumul"]
                        < existing_state["complexity_cumul"]
                    ):
                        states[istage, ithres, iprob : iprob + 1] = cur_state
                        folds_out[ithres * nprobs + iprob] = cur_fold_state


@attrs.frozen(auto_attribs=True, kw_only=True)
class SharedOptions:
    """Configuration options for ``shared`` and ``robust`` threshold scheme modes."""

    min_pd: float = 0.1
    w_terminal: float = 1.0
    beam_sigma: float = 3.0
    reach_sigma: float = 1.0
    ntrials_pilot: int = 8192
    ntrials_sig_bank: int = 16384
    ntrials_noise_bank: int = 4096


@attrs.frozen(auto_attribs=True, kw_only=True)
class SchemeSolution:
    """Sample-split solution returned by ``DynamicThresholdScheme.solve()``."""

    thresholds: npt.NDArray[np.float32]
    pd: float
    pd_se: float
    pd_trials: int
    c: float
    cost: float
    search_thresholds: npt.NDArray[np.float32]
    search_pd: float
    search_c: float
    delivery: Literal["no_path", "met", "within_tolerance", "short"]
    min_pd: float
    w_terminal: float
    eval_states: StatesInfo


def simulate_unpruned_signal_scores(
    nstages: int,
    profile: npt.NDArray[np.float32],
    bias_snr: float,
    ntrials: int,
    ducy_max: float,
    wtsp: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float32]:
    """Simulate unpruned cumulative signal boxcar scores ``(nstages, ntrials)``."""
    nbins = len(profile)
    widths = scoring.generate_box_width_trials(nbins, ducy_max=ducy_max, wtsp=wtsp)
    sig_step = (profile * np.float32(bias_snr)).astype(np.float32)
    folds = rng.standard_normal((ntrials, nbins)).astype(np.float32) + sig_step
    scores = np.empty((nstages, ntrials), dtype=np.float32)
    for istage in range(nstages):
        folds += rng.standard_normal((ntrials, nbins)).astype(np.float32) + sig_step
        inv_std = np.float32(1.0 / math.sqrt(istage + 2.0))
        norm_folds = np.ascontiguousarray(folds * inv_std, dtype=np.float32)
        scores[istage] = np.max(scoring.boxcar_snr_2d(norm_folds, widths, 1.0), axis=1)
    return scores


class DynamicThresholdScheme:
    def __init__(
        self,
        branching_pattern: np.ndarray,
        ref_ducy: float,
        nbins: int = 64,
        ntrials: int = 1024,
        nprobs: int = 10,
        prob_min: float = 0.05,
        snr_final: float = 8,
        nthresholds: int = 100,
        ducy_max: float = 0.2,
        wtsp: float = 1.0,
        beam_width: float = 0.7,
        mode: Literal["legacy", "improved", "shared", "robust"] = "legacy",
        seed: int | np.random.Generator | None = None,
        shared_options: SharedOptions | None = None,
    ) -> None:
        if mode not in ("legacy", "improved", "shared", "robust"):
            msg = (
                "mode must be 'legacy', 'improved', 'shared', or 'robust', "
                f"got {mode!r}"
            )
            raise ValueError(msg)
        self.mode = mode
        self.shared_options = (
            SharedOptions() if shared_options is None else shared_options
        )
        self.rng = np.random.default_rng(seed)
        # `self.rng` serves the serial init paths. The parallel kernels cannot use it:
        # they consume it inside a `prange`, so which thread takes which draw depends
        # on scheduling and the result is not reproducible even from a fixed seed.
        # They get one generator per loop iteration instead, derived from this entropy
        # and the stage index, which makes them independent of thread order and of
        # NUMBA_NUM_THREADS. A `seed` of None still yields fresh entropy per instance.
        self._entropy = (
            int(seed.integers(np.iinfo(np.int64).max))
            if isinstance(seed, np.random.Generator)
            else np.random.SeedSequence(seed).entropy
        )
        self.branching_pattern = branching_pattern
        self.ref_ducy = ref_ducy
        self.profile = generate_folded_profile(nbins=nbins, ducy=ref_ducy)
        self.ntrials = ntrials
        self.snr_final = snr_final
        self.ducy_max = ducy_max
        self.wtsp = wtsp
        self.beam_width = beam_width

        # later define as snr^2
        self.thresholds = np.linspace(0.1, self.snr_final, nthresholds)
        self.prob_min = prob_min
        if self.mode in ("shared", "robust"):
            min_pd = self.shared_options.min_pd
            above = np.logspace(np.log10(min_pd), 0.0, nprobs)
            n_below = min(16, max(1, nprobs // 2))
            below = np.logspace(np.log10(min_pd * 0.5), np.log10(min_pd), n_below + 1)[
                :-1
            ]
            self.probs = np.concatenate((below, above))
        else:
            self.probs = np.logspace(np.log10(prob_min), 0, nprobs)

        state = np.ones(1, dtype=state_dtype)
        state["threshold"] = -1.0
        state["threshold_prev"] = -1.0
        self.states = np.full((self.nstages, self.nthresholds, self.nprobs), state)
        self.folds_in = typed.List.empty_list(FoldsType)
        self.folds_out = typed.List.empty_list(FoldsType)
        for _ in range(self.nthresholds * self.nprobs):
            self.folds_in.append(create_empty_folds())
            self.folds_out.append(create_empty_folds())
        self.guess_path = np.minimum(
            bound_scheme(self.nstages, self.snr_final),
            trials_scheme(self.branching_pattern, trials_start=1),
        )
        self._pilot_sigma: npt.NDArray[np.float64] = np.ones(
            self.nstages,
            dtype=np.float64,
        )
        self._beam_lo: npt.NDArray[np.float64] = np.zeros(
            self.nstages,
            dtype=np.float64,
        )
        self._beam_hi: npt.NDArray[np.float64] = np.full(
            self.nstages,
            self.snr_final,
            dtype=np.float64,
        )
        self._shared_scores: npt.NDArray[np.float32] | None = None
        self._shared_masks_in: dict[tuple[int, int], npt.NDArray[np.bool_]] = {}
        self._shared_h0_in: dict[tuple[int, int], npt.NDArray[np.float32]] = {}
        self._shared_parents: dict[tuple[int, int, int], tuple[int, int]] = {}

        if self.mode == "improved":
            self._init_improved()
        elif self.mode in ("shared", "robust"):
            self._init_shared()
        else:
            self._init_legacy()

    @property
    def nstages(self) -> int:
        return len(self.branching_pattern)

    @property
    def bias_snr(self) -> float:
        return self.snr_final / np.sqrt(self.nstages + 1)

    @property
    def nthresholds(self) -> int:
        return len(self.thresholds)

    @property
    def nprobs(self) -> int:
        return len(self.probs)

    @property
    def nbins(self) -> int:
        return len(self.profile)

    def get_current_thresholds_idx(self, istage: int) -> np.ndarray:
        if self.mode in ("shared", "robust"):
            lo = float(self._beam_lo[istage])
            hi = float(self._beam_hi[istage])
            idx = np.where((self.thresholds >= lo) & (self.thresholds <= hi))[0]
            if len(idx) == 0:
                target = float(np.clip(self.guess_path[istage], lo, hi))
                nearest = int(np.argmin(np.abs(self.thresholds - target)))
                return np.array([nearest], dtype=np.int64)
            return idx
        guess = self.guess_path[istage]
        half_extent = self.beam_width
        lower_bound = max(0, guess - half_extent)
        upper_bound = min(self.snr_final, guess + half_extent)
        idx = np.where(
            (self.thresholds >= lower_bound) & (self.thresholds <= upper_bound),
        )[0]
        if len(idx) == 0:
            target = np.clip(guess, 0.0, self.snr_final)
            nearest = int(np.argmin(np.abs(self.thresholds - target)))
            return np.array([nearest], dtype=np.int64)
        return idx

    def _init_legacy(self) -> None:
        with Timer(name="DynamicThresholdScheme init", logger=logger.info):
            var_init = 1.0
            folds = np.zeros((self.ntrials, self.nbins), dtype=np.float32)
            folds_h0, _ = simulate_folds(
                folds,
                0,
                self.profile,
                self.rng,
                bias_snr=0,
                var_add=var_init,
                ntrials_min=self.ntrials,
            )
            folds_h1, _ = simulate_folds(
                folds,
                0,
                self.profile,
                self.rng,
                bias_snr=self.bias_snr,
                var_add=var_init,
                ntrials_min=self.ntrials,
            )
            state = np.ones(1, dtype=state_dtype)[0]
            state["threshold"] = -1.0
            state["threshold_prev"] = -1.0
            fold_state = Folds(folds_h0, folds_h1, var_init)
            thresholds_idx = self.get_current_thresholds_idx(0)
            for ithres in thresholds_idx:
                cur_state, cur_fold_state = gen_next_using_thresh(
                    state,
                    fold_state,
                    self.thresholds[ithres],
                    self.branching_pattern[0],
                    self.bias_snr,
                    self.profile,
                    self.rng,
                    self.ntrials,
                    self.ducy_max,
                    self.wtsp,
                )
                cur_state = cur_state[0]  # Get record from array
                iprob = (
                    int(
                        np.digitize(
                            float(cur_state["success_h1_cumul"]),
                            self.probs,
                        ),
                    )
                    - 1
                )
                if iprob < 0 or iprob >= self.nprobs:
                    continue
                self.states[0, ithres, iprob] = cur_state
                folds_idx = int(ithres * self.nprobs + iprob)
                self.folds_in[folds_idx] = cur_fold_state

    def _init_improved(self) -> None:
        with Timer(name="DynamicThresholdScheme init", logger=logger.info):
            var_init = 1.0
            folds = np.zeros((self.ntrials, self.nbins), dtype=np.float32)
            folds_h0, _ = simulate_folds(
                folds,
                0,
                self.profile,
                self.rng,
                bias_snr=0,
                var_add=var_init,
                ntrials_min=self.ntrials,
            )
            folds_h1, _ = simulate_folds(
                folds,
                0,
                self.profile,
                self.rng,
                bias_snr=self.bias_snr,
                var_add=var_init,
                ntrials_min=self.ntrials,
            )
            state = np.ones(1, dtype=state_dtype)[0]
            state["threshold"] = -1.0
            state["threshold_prev"] = -1.0

            sim_folds_h0, var_next = simulate_folds(
                folds_h0,
                var_init,
                self.profile,
                self.rng,
                bias_snr=0.0,
                var_add=1.0,
                ntrials_min=self.ntrials,
            )
            sim_folds_h1, var_next = simulate_folds(
                folds_h1,
                var_init,
                self.profile,
                self.rng,
                bias_snr=self.bias_snr,
                var_add=1.0,
                ntrials_min=self.ntrials,
            )
            scores_h0 = fold_max_scores(
                sim_folds_h0,
                var_next,
                self.ducy_max,
                self.wtsp,
            )
            scores_h1 = fold_max_scores(
                sim_folds_h1,
                var_next,
                self.ducy_max,
                self.wtsp,
            )
            fold_sim_state = Folds(
                sim_folds_h0,
                sim_folds_h1,
                scores_h0,
                scores_h1,
                var_next,
            )

            thresholds_idx = self.get_current_thresholds_idx(0)
            for ithres in thresholds_idx:
                cur_state, cur_fold_state = transition_state(
                    state,
                    fold_sim_state,
                    self.thresholds[ithres],
                    self.branching_pattern[0],
                )
                cur_state = cur_state[0]

                iprob = (
                    int(
                        np.digitize(
                            float(cur_state["success_h1_cumul"]),
                            self.probs,
                        ),
                    )
                    - 1
                )
                if iprob < 0 or iprob >= self.nprobs:
                    continue
                self.states[0, ithres, iprob] = cur_state
                folds_idx = int(ithres * self.nprobs + iprob)
                self.folds_in[folds_idx] = cur_fold_state

    def _init_shared(self) -> None:
        with Timer(name="DynamicThresholdScheme shared init", logger=logger.info):
            opts = self.shared_options
            pilot_rng = np.random.default_rng(
                np.random.SeedSequence(self._entropy, spawn_key=(9001,)),
            )
            pilot_scores = simulate_unpruned_signal_scores(
                self.nstages,
                self.profile,
                self.bias_snr,
                opts.ntrials_pilot,
                self.ducy_max,
                self.wtsp,
                pilot_rng,
            )
            med_s = np.median(pilot_scores, axis=1)
            sigma_s = np.maximum(np.std(pilot_scores, axis=1), 0.25)
            p_lo = float(self.probs[0])
            q_lo = np.quantile(
                pilot_scores,
                max(1e-4, 0.02 * p_lo / max(self.nstages, 1)),
                axis=1,
            )
            q_hi = np.quantile(pilot_scores, min(0.9999, 1.0 - 0.5 * p_lo), axis=1)
            self._pilot_sigma = sigma_s.astype(np.float64)
            self.guess_path = np.minimum(
                med_s,
                trials_scheme(self.branching_pattern, trials_start=1),
            )
            self._set_guided_beam(self.guess_path, q_lo, q_hi)

            n_sig_shared = min(16 * self.ntrials, 2**16)
            shared_rng = np.random.default_rng(
                np.random.SeedSequence(self._entropy, spawn_key=(9002,)),
            )
            self._shared_scores = simulate_unpruned_signal_scores(
                self.nstages,
                self.profile,
                self.bias_snr,
                n_sig_shared,
                self.ducy_max,
                self.wtsp,
                shared_rng,
            )
            self._init_shared_stage0(pass_idx=0)

    def _set_guided_beam(
        self,
        center: np.ndarray,
        q_lo: np.ndarray | None = None,
        q_hi: np.ndarray | None = None,
    ) -> None:
        half = self.shared_options.beam_sigma * self._pilot_sigma
        lo = np.maximum(0.0, center - half)
        hi = np.minimum(self.snr_final, center + half)
        if q_lo is not None:
            lo = np.maximum(lo, q_lo)
        if q_hi is not None:
            hi = np.minimum(hi, q_hi)
        self._beam_lo = np.minimum(lo, hi)
        self._beam_hi = np.maximum(lo, hi)

    def _init_shared_stage0(self, pass_idx: int) -> None:
        assert self._shared_scores is not None  # noqa: S101
        state_empty = np.ones(1, dtype=state_dtype)
        state_empty["threshold"] = -1.0
        state_empty["threshold_prev"] = -1.0
        self.states = np.full(
            (self.nstages, self.nthresholds, self.nprobs),
            state_empty,
        )
        self._shared_masks_in = {}
        self._shared_h0_in = {}
        self._shared_parents = {}

        rng0 = np.random.default_rng(
            np.random.SeedSequence(self._entropy, spawn_key=(9100 + pass_idx, 0)),
        )
        folds0 = rng0.standard_normal((self.ntrials, self.nbins)).astype(np.float32)
        folds1 = folds0 + rng0.standard_normal((self.ntrials, self.nbins)).astype(
            np.float32,
        )
        scores_h0 = fold_max_scores(folds1, 2.0, self.ducy_max, self.wtsp)
        sig0 = self._shared_scores[0]
        h0_floor = min(8, self.ntrials) / float(self.ntrials)
        bp0 = float(self.branching_pattern[0])
        is_last = self.nstages == 1
        w_term = self.shared_options.w_terminal if is_last else 0.0

        for ithres in self.get_current_thresholds_idx(0):
            thresh = float(self.thresholds[ithres])
            mask_sig = sig0 > thresh
            pd_c = float(np.mean(mask_sig))
            if pd_c <= 0.0:
                continue
            surv_idx = np.flatnonzero(scores_h0 > thresh)
            if surv_idx.size == 0:
                continue
            h0_rate = max(surv_idx.size / float(self.ntrials), h0_floor)
            nleaves_next = bp0
            nleaves_surv = nleaves_next * h0_rate
            comp_cumul = 1.0 + nleaves_next
            rank_cost = (comp_cumul + w_term * nleaves_surv) / pd_c

            iprob = int(np.digitize(pd_c, self.probs)) - 1
            if iprob < 0 or iprob >= self.nprobs:
                continue
            cur = self.states[0, ithres, iprob]
            if cur["threshold"] < 0 or rank_cost < float(cur["cost"]):
                cur["success_h0"] = h0_rate
                cur["success_h1"] = pd_c
                cur["complexity"] = nleaves_surv
                cur["complexity_cumul"] = comp_cumul
                cur["success_h1_cumul"] = pd_c
                cur["nbranches"] = bp0
                cur["threshold"] = thresh
                cur["cost"] = rank_cost
                cur["threshold_prev"] = -1.0
                cur["success_h1_cumul_prev"] = 1.0
                self._shared_masks_in[(int(ithres), iprob)] = mask_sig
                self._shared_h0_in[(int(ithres), iprob)] = np.ascontiguousarray(
                    folds1[surv_idx],
                )

    def _run_shared_pass(self, center: np.ndarray, pass_idx: int) -> None:
        assert self._shared_scores is not None  # noqa: S101
        h0_floor = min(8, self.ntrials) / float(self.ntrials)
        reach_sigma = self.shared_options.reach_sigma
        widths = scoring.generate_box_width_trials(
            self.nbins,
            ducy_max=self.ducy_max,
            wtsp=self.wtsp,
        )

        for istage in track(
            range(1, self.nstages),
            description="Running shared stages",
            console=CONSOLE,
            transient=True,
        ):
            beam_cur = self.get_current_thresholds_idx(istage)
            sig_s = self._shared_scores[istage]
            var_next = float(istage + 2)
            inv_std = np.float32(1.0 / math.sqrt(var_next))
            bp_s = float(self.branching_pattern[istage])
            is_last = istage == self.nstages - 1
            w_term = self.shared_options.w_terminal if is_last else 0.0
            max_doff = reach_sigma * float(self._pilot_sigma[istage])
            center_prev = float(center[istage - 1])
            center_cur = float(center[istage])

            masks_out: dict[tuple[int, int], npt.NDArray[np.bool_]] = {}
            h0_out: dict[tuple[int, int], npt.NDArray[np.float32]] = {}
            parent_keys = sorted(self._shared_masks_in.keys())
            stage_seq = np.random.SeedSequence(
                self._entropy,
                spawn_key=(9100 + pass_idx, istage),
            )
            parent_rngs = [
                np.random.default_rng(child)
                for child in stage_seq.spawn(max(1, len(parent_keys)))
            ]

            for pidx, (iprev, jprob) in enumerate(parent_keys):
                prev_state = self.states[istage - 1, iprev, jprob]
                if prev_state["threshold"] < 0:
                    continue
                mask_parent = self._shared_masks_in[(iprev, jprob)]
                h0_parent = self._shared_h0_in[(iprev, jprob)]
                if h0_parent.shape[0] == 0:
                    continue
                prev_off = float(self.thresholds[iprev]) - center_prev
                rng_p = parent_rngs[pidx]
                rep = math.ceil(self.ntrials / h0_parent.shape[0])
                tiled = np.tile(h0_parent, (rep, 1))[: self.ntrials]
                h0_next = tiled + rng_p.standard_normal(
                    (self.ntrials, self.nbins),
                ).astype(np.float32)
                scores_h0 = np.max(
                    scoring.boxcar_snr_2d(
                        np.ascontiguousarray(h0_next * inv_std, dtype=np.float32),
                        widths,
                        1.0,
                    ),
                    axis=1,
                )

                prev_comp = float(prev_state["complexity"])
                prev_comp_cumul = float(prev_state["complexity_cumul"])
                prev_pd = float(prev_state["success_h1_cumul"])
                nleaves_next = prev_comp * bp_s
                comp_cumul = prev_comp_cumul + nleaves_next

                for ithres in beam_cur:
                    thresh = float(self.thresholds[ithres])
                    cur_off = thresh - center_cur
                    if abs(cur_off - prev_off) > max_doff:
                        continue
                    mask_child = mask_parent & (sig_s > thresh)
                    pd_c = float(np.mean(mask_child))
                    if pd_c <= 0.0:
                        continue
                    surv_idx = np.flatnonzero(scores_h0 > thresh)
                    if surv_idx.size == 0:
                        continue
                    h0_rate = max(surv_idx.size / float(self.ntrials), h0_floor)
                    nleaves_surv = nleaves_next * h0_rate
                    rank_cost = (comp_cumul + w_term * nleaves_surv) / pd_c

                    iprob = int(np.digitize(pd_c, self.probs)) - 1
                    if iprob < 0 or iprob >= self.nprobs:
                        continue
                    cur = self.states[istage, ithres, iprob]
                    if cur["threshold"] < 0 or rank_cost < float(cur["cost"]):
                        cur["success_h0"] = h0_rate
                        cur["success_h1"] = pd_c / max(prev_pd, 1e-12)
                        cur["complexity"] = nleaves_surv
                        cur["complexity_cumul"] = comp_cumul
                        cur["success_h1_cumul"] = pd_c
                        cur["nbranches"] = bp_s
                        cur["threshold"] = thresh
                        cur["cost"] = rank_cost
                        cur["threshold_prev"] = float(prev_state["threshold"])
                        cur["success_h1_cumul_prev"] = prev_pd
                        masks_out[(int(ithres), iprob)] = mask_child
                        h0_out[(int(ithres), iprob)] = np.ascontiguousarray(
                            h0_next[surv_idx],
                        )
                        self._shared_parents[(istage, int(ithres), iprob)] = (
                            iprev,
                            jprob,
                        )

            self._shared_masks_in = masks_out
            self._shared_h0_in = h0_out

    def _extract_best_path(
        self,
        min_pd: float,
    ) -> tuple[npt.NDArray[np.float32], float, float]:
        last = self.states[self.nstages - 1]
        valid = (last["threshold"] >= 0) & (last["success_h1_cumul"] >= min_pd)
        if not np.any(valid):
            valid = last["threshold"] >= 0
        if not np.any(valid):
            return np.empty(0, dtype=np.float32), 0.0, float("inf")
        costs = np.where(valid, last["cost"], np.inf)
        best_flat = int(np.argmin(costs))
        ithres, iprob = divmod(best_flat, self.nprobs)
        best_state = last[ithres, iprob]
        search_pd = float(best_state["success_h1_cumul"])
        w_term = (
            self.shared_options.w_terminal if self.mode in ("shared", "robust") else 0.0
        )
        search_c = float(best_state["complexity_cumul"]) + w_term * float(
            best_state["complexity"],
        )

        path = np.empty(self.nstages, dtype=np.float32)
        cur_t, cur_p = ithres, iprob
        for istage in range(self.nstages - 1, -1, -1):
            st = self.states[istage, cur_t, cur_p]
            path[istage] = np.float32(st["threshold"])
            if istage > 0:
                if (istage, cur_t, cur_p) in self._shared_parents:
                    cur_t, cur_p = self._shared_parents[(istage, cur_t, cur_p)]
                else:
                    t_prev = float(st["threshold_prev"])
                    p_prev = float(st["success_h1_cumul_prev"])
                    cur_t = int(np.argmin(np.abs(self.thresholds - t_prev)))
                    cur_p = int(
                        np.clip(
                            int(np.digitize(p_prev, self.probs)) - 1,
                            0,
                            self.nprobs - 1,
                        ),
                    )
        return path, search_pd, search_c

    def solve(
        self,
        min_pd: float | None = None,
        seed: int | None = None,
    ) -> SchemeSolution:
        """Sample-split tighten, repair, and unbiased evaluation of the best schedule.

        Uses three independent signal banks (Bank A to shift/tighten, Bank B to
        repair if needed, and Bank C — untouched by any selection — for unbiased
        ``pd`` and binomial ``pd_se``) plus a fresh noise evaluation for ``c``.
        """
        opts = self.shared_options
        target_pd = opts.min_pd if min_pd is None else float(min_pd)
        w_term = opts.w_terminal if self.mode in ("shared", "robust") else 0.0
        search_path, search_pd, search_c = self._extract_best_path(target_pd)
        if search_path.size == 0:
            return SchemeSolution(
                thresholds=np.empty(0, dtype=np.float32),
                pd=0.0,
                pd_se=0.0,
                pd_trials=0,
                c=float("inf"),
                cost=float("inf"),
                search_thresholds=search_path,
                search_pd=0.0,
                search_c=float("inf"),
                delivery="no_path",
                min_pd=target_pd,
                w_terminal=w_term,
                eval_states=StatesInfo([]),
            )

        base_seq = (
            np.random.SeedSequence(seed)
            if seed is not None
            else np.random.SeedSequence(self._entropy, spawn_key=(9500,))
        )
        rng_a, rng_b, rng_c, rng_noise = [
            np.random.default_rng(s) for s in base_seq.spawn(4)
        ]
        n_sig = opts.ntrials_sig_bank
        scores_a = simulate_unpruned_signal_scores(
            self.nstages,
            self.profile,
            self.bias_snr,
            n_sig,
            self.ducy_max,
            self.wtsp,
            rng_a,
        )
        scores_b = simulate_unpruned_signal_scores(
            self.nstages,
            self.profile,
            self.bias_snr,
            n_sig,
            self.ducy_max,
            self.wtsp,
            rng_b,
        )
        scores_c = simulate_unpruned_signal_scores(
            self.nstages,
            self.profile,
            self.bias_snr,
            n_sig,
            self.ducy_max,
            self.wtsp,
            rng_c,
        )

        # Target includes +2 binomial standard errors to protect delivery on Bank C
        se_target = math.sqrt(max(target_pd * (1.0 - target_pd), 1e-6) / n_sig)
        k_target = min(n_sig, math.ceil((target_pd + 2.0 * se_target) * n_sig))

        # 1. Bank A: shift down if under target, then tighten each stage
        gamma = search_path.astype(np.float64).copy()
        margins_a = np.min(scores_a - gamma[:, None], axis=0)
        if int(np.count_nonzero(margins_a > 0.0)) < k_target:
            delta_a = float(np.sort(margins_a)[-k_target]) - 1e-6
            gamma = np.maximum(0.0, gamma + delta_a)
        surv_a = np.all(scores_a > gamma[:, None], axis=0)
        if np.any(surv_a):
            tight = np.min(scores_a[:, surv_a], axis=1) - 1e-6
            gamma = np.maximum(gamma, tight)

        # 2. Bank B: repair if Bank B survival falls below target
        margins_b = np.min(scores_b - gamma[:, None], axis=0)
        if int(np.count_nonzero(margins_b > 0.0)) < k_target:
            delta_b = float(np.sort(margins_b)[-k_target]) - 1e-6
            gamma = np.maximum(0.0, gamma + delta_b)

        gamma_f32 = gamma.astype(np.float32)

        # 3. Bank C: unbiased Pd report + fresh noise cost evaluation
        surv_c = np.all(scores_c > gamma_f32[:, None], axis=0)
        pd = float(np.mean(surv_c))
        pd_se = math.sqrt(max(pd * (1.0 - pd), 0.0) / n_sig)
        eval_states = evaluate_scheme(
            gamma_f32,
            self.branching_pattern,
            self.ref_ducy,
            nbins=self.nbins,
            ntrials=opts.ntrials_noise_bank,
            snr_final=self.snr_final,
            ducy_max=self.ducy_max,
            wtsp=self.wtsp,
            seed=rng_noise,
        )
        if eval_states.entries:
            last_st = eval_states.entries[-1]
            c_val = float(last_st.complexity_cumul) + w_term * float(last_st.complexity)
        else:
            c_val = float("inf")
        cost_val = c_val / pd if pd > 0.0 else float("inf")

        delivery: Literal["no_path", "met", "within_tolerance", "short"]
        if pd >= target_pd:
            delivery = "met"
        elif pd >= 0.9 * target_pd:
            delivery = "within_tolerance"
        else:
            delivery = "short"

        return SchemeSolution(
            thresholds=gamma_f32,
            pd=pd,
            pd_se=pd_se,
            pd_trials=n_sig,
            c=c_val,
            cost=cost_val,
            search_thresholds=search_path,
            search_pd=search_pd,
            search_c=search_c,
            delivery=delivery,
            min_pd=target_pd,
            w_terminal=w_term,
            eval_states=eval_states,
        )

    def _stage_rngs(self, istage: int, n: int) -> typed.List:
        """One generator per parallel-loop iteration of `istage`.

        Derived from `(self._entropy, istage)` rather than spawned off a running
        stream, so the result depends only on the seed, the stage and the iteration
        index -- never on call order or on how many threads numba happens to use.
        """
        seq = np.random.SeedSequence(self._entropy, spawn_key=(istage,))
        return typed.List([np.random.default_rng(child) for child in seq.spawn(n)])

    def run(self, thres_neigh: int = 11) -> None:
        with Timer(name="DynamicThresholdScheme run", logger=logger.info):
            logger.info("Running dynamic threshold scheme (%s mode)", self.mode)
            if self.mode in ("shared", "robust"):
                self._run_shared_pass(self.guess_path, pass_idx=0)
                if self.mode == "robust":
                    path1, _, _ = self._extract_best_path(self.shared_options.min_pd)
                    if path1.size == self.nstages:
                        self.guess_path = path1.astype(np.float64)
                        self._set_guided_beam(self.guess_path)
                        self._init_shared_stage0(pass_idx=1)
                        self._run_shared_pass(self.guess_path, pass_idx=1)
                return
            for istage in track(
                range(1, self.nstages),
                description="Running stages",
                console=CONSOLE,
                transient=True,
            ):
                beam_idx_cur = self.get_current_thresholds_idx(istage)
                beam_idx_prev = self.get_current_thresholds_idx(istage - 1)
                if self.mode == "improved":
                    sim_folds = pre_simulate_stage_folds(
                        self.states[istage - 1],
                        self.folds_in,
                        beam_idx_prev,
                        self.bias_snr,
                        self.profile,
                        self._stage_rngs(istage, len(beam_idx_prev) * self.nprobs),
                        self.ntrials,
                        self.nthresholds,
                        self.nprobs,
                        self.ducy_max,
                        self.wtsp,
                    )
                    run_stage_improved(
                        istage,
                        beam_idx_cur,
                        beam_idx_prev,
                        self.states,
                        sim_folds,
                        self.folds_out,
                        self.probs,
                        self.branching_pattern[istage],
                        self.thresholds,
                        self.nprobs,
                        thres_neigh,
                    )
                else:
                    run_stage_legacy(
                        istage,
                        beam_idx_cur,
                        beam_idx_prev,
                        self.states,
                        self.folds_in,
                        self.folds_out,
                        self.probs,
                        self.branching_pattern[istage],
                        self.thresholds,
                        self.bias_snr,
                        self.profile,
                        self._stage_rngs(istage, len(beam_idx_cur)),
                        self.ntrials,
                        self.ducy_max,
                        self.wtsp,
                        thres_neigh,
                    )
                self.folds_in = self.folds_out
                self.folds_out = typed.List.empty_list(FoldsType)
                for _ in range(self.nthresholds * self.nprobs):
                    self.folds_out.append(create_empty_folds())

    def save(self, outdir: str = ".") -> str:
        """Save the StatesInfo object to an hdf5 file."""
        filebase = (
            f"dynscheme_nstages_{self.nstages:03d}_"
            f"nthresh_{self.nthresholds:03d}_nprobs_{self.nprobs:03d}_"
            f"ntrials_{self.ntrials:04d}_snr_{self.snr_final:04.1f}_"
            f"ducy_{self.ref_ducy:04.2f}_beam_{self.beam_width:03.1f}"
        )
        filename = Path(outdir) / f"{filebase}.h5"
        with h5py.File(filename, "w") as f:
            # Save simple attributes
            for attr in [
                "ntrials",
                "snr_final",
                "ref_ducy",
                "ducy_max",
                "wtsp",
                "beam_width",
                "mode",
            ]:
                f.attrs[attr] = getattr(self, attr)
            # Save numpy arrays
            for arr in [
                "branching_pattern",
                "profile",
                "thresholds",
                "probs",
                "guess_path",
            ]:
                f.create_dataset(
                    arr,
                    data=getattr(self, arr),
                    compression="gzip",
                    compression_opts=9,
                )
            f.create_dataset(
                "states",
                data=self.states,
                compression="gzip",
                compression_opts=9,
            )
        logger.info(f"Saved dynamic threshold scheme to {filename}")
        return filename.as_posix()


def schedule_survive_probs(
    branching_pattern: np.ndarray,
    *,
    protect: int = 4,
    return_stages: int = 3,
    survive_nonbranch: float = 1.0,
) -> tuple[np.ndarray, list[int], float]:
    """Survive probabilities that protect the early pruning stages and return to size.

    The plain choice, 1/bp at every branching stage, puts the first cuts where a weak
    signal's partial S/N is lowest: at stages 1-2 a final S/N 14-19 pulsar has partial
    S/N 2.5-3.6 and the cuts sit at 2.6-3.3. This schedule keeps every leaf at the first
    ``protect`` stages, then spreads the protected stages' branching product as an extra
    factor on the cuts at the next ``return_stages`` branching stages, so that the noise
    tree returns to the plain ladder's size at the last of them and stays there (each
    later stage keeps 1/bp of what arrives). Non-branching stages keep
    ``survive_nonbranch`` (1 means no cut, as the plain ladder's minimum over trials).
    The window is stated in branching stages, never in levels: a window fixed in levels
    can hold a single branching stage, and the whole factor on one cut is worse than no
    protection. ``protect=0`` returns the plain vector.

    Parameters
    ----------
    branching_pattern : np.ndarray
        Branching factor per stage, as `determine_scheme` takes it.
    protect : int, optional
        Number of leading stages with survive probability 1, by default 4.
    return_stages : int, optional
        Number of branching stages after the protected ones that carry the extra
        factor, by default 3. Fewer are used when fewer exist.
    survive_nonbranch : float, optional
        Survive probability at the non-branching stages, by default 1.0.

    Returns
    -------
    tuple[np.ndarray, list[int], float]
        The survive probabilities, the (1-based) stages that carry the extra factor,
        and the factor.
    """
    bp = np.asarray(branching_pattern, dtype=np.float64)
    probs = 1.0 / bp
    probs[np.abs(bp - 1.0) < 1e-9] = survive_nonbranch
    protect = int(protect)
    if protect <= 0:
        return probs, [], 1.0
    probs[:protect] = 1.0
    stages = [i + 1 for i in range(protect, len(bp)) if bp[i] > 1.0 + 1e-4][
        : max(int(return_stages), 0)
    ]
    if not stages:
        return probs, [], 1.0
    factor = float(np.prod(bp[:protect])) ** (1.0 / len(stages))
    for st in stages:
        probs[st - 1] /= factor
    return probs, stages, factor


def determine_scheme(
    survive_probs: np.ndarray,
    branching_pattern: np.ndarray,
    ref_ducy: float,
    nbins: int = 64,
    ntrials: int = 2048,
    snr_final: float = 8,
    ducy_max: float = 0.2,
    wtsp: float = 1.0,
    seed: int | np.random.Generator | None = None,
) -> StatesInfo:
    if len(survive_probs) != len(branching_pattern):
        msg = "Number of survive_probs must match the number of stages"
        raise ValueError(msg)

    var_init = 1.0
    nstages = len(branching_pattern)
    profile = generate_folded_profile(nbins=nbins, ducy=ref_ducy)
    bias_snr = snr_final / np.sqrt(nstages + 1)
    rng = np.random.default_rng(seed)
    states: list[np.recarray] = []
    folds = np.zeros((ntrials, len(profile)), dtype=np.float32)
    folds_h0, _ = simulate_folds(folds, 0, profile, rng, 0, var_init, ntrials)
    folds_h1, _ = simulate_folds(folds, 0, profile, rng, bias_snr, var_init, ntrials)
    initial_state = np.ones(1, dtype=state_dtype)[0]
    initial_state["threshold"] = -1
    initial_state["threshold_prev"] = -1
    prev_fold_state = Folds(folds_h0, folds_h1, var_init)
    for istage in range(nstages):
        prev_state = initial_state if istage == 0 else states[istage - 1]
        if istage > 0 and prev_fold_state.is_empty:
            logger.info("Path not viable, No trials survived, stopping")
            break
        cur_state, cur_fold_state = gen_next_using_surv_prob(
            prev_state,
            prev_fold_state,
            survive_probs[istage],
            branching_pattern[istage],
            bias_snr,
            profile,
            rng,
            ntrials,
            ducy_max,
            wtsp,
        )
        cur_state = cur_state[0]  # Get record from array
        states.append(cur_state)
        prev_fold_state = cur_fold_state
    return StatesInfo([StateInfo.from_record(state) for state in states])


def evaluate_scheme(
    thresholds: np.ndarray,
    branching_pattern: np.ndarray,
    ref_ducy: float,
    nbins: int = 64,
    ntrials: int = 2048,
    snr_final: float = 8,
    ducy_max: float = 0.2,
    wtsp: float = 1.0,
    seed: int | np.random.Generator | None = None,
) -> StatesInfo:
    var_init = 1.0
    nstages = len(branching_pattern)
    profile = generate_folded_profile(nbins=nbins, ducy=ref_ducy)
    bias_snr = snr_final / np.sqrt(nstages + 1)
    rng = np.random.default_rng(seed)
    if len(thresholds) != nstages:
        msg = "Number of thresholds must match the number of stages"
        raise ValueError(msg)
    states: list[np.recarray] = []
    folds = np.zeros((ntrials, len(profile)), dtype=np.float32)
    folds_h0, _ = simulate_folds(folds, 0, profile, rng, 0, var_init, ntrials)
    folds_h1, _ = simulate_folds(folds, 0, profile, rng, bias_snr, var_init, ntrials)
    initial_state = np.ones(1, dtype=state_dtype)[0]
    initial_state["threshold"] = -1
    initial_state["threshold_prev"] = -1
    prev_fold_state = Folds(folds_h0, folds_h1, var_init)
    for istage in range(nstages):
        prev_state = initial_state if istage == 0 else states[istage - 1]
        if istage > 0 and prev_fold_state.is_empty:
            logger.info("Path not viable, No trials survived, stopping")
            break
        cur_state, cur_fold_state = gen_next_using_thresh(
            prev_state,
            prev_fold_state,
            thresholds[istage],
            branching_pattern[istage],
            bias_snr,
            profile,
            rng,
            ntrials,
            ducy_max,
            wtsp,
        )
        cur_state = cur_state[0]  # Get record from array
        states.append(cur_state)
        prev_fold_state = cur_fold_state
    return StatesInfo([StateInfo.from_record(state) for state in states])


def threshold_simulation(
    survive_probs: np.ndarray,
    branching_pattern: np.ndarray,
    ref_ducy: float = 0.1,
    *,
    nbins: int = 64,
    ntrials: int = 2048,
    snr_final: float = 8.0,
    ducy_max: float = 0.2,
    wtsp: float = 1.0,
    score_mode: ScoreMode = "boxcar",
    score_fn: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]
    | None = None,
    n_harmonics: int = 16,
    seed: int | np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.float32], StatesInfo]:
    """Empirically calibrate pruning thresholds for any profile scorer.

    Simulates null-hypothesis (``H0``) and signal (``H1``) cumulative ``(E, V)``
    folds along ``branching_pattern``, scoring them at each stage with either
    ``score_fn`` or ``scoring.score_profiles(..., mode=score_mode)`` so
    non-boxcar scorers (``double_boxcar``, ``double_matched``, ``harmonic_z2``,
    ``combined``, ``boxcar_glrt``, ``kadane``) obtain properly calibrated
    empirical survival thresholds.

    Returns
    -------
    tuple[NDArray[np.float32], StatesInfo]
        Empirical per-stage thresholds and stage-by-stage ``StatesInfo``.
    """
    bp = np.asarray(branching_pattern, dtype=np.float64)
    sp = np.asarray(survive_probs, dtype=np.float64)
    if sp.shape != bp.shape:
        msg = "survive_probs and branching_pattern must have the same length"
        raise ValueError(msg)
    nstages = len(bp)
    widths = scoring.generate_box_width_trials(nbins, ducy_max=ducy_max, wtsp=wtsp)
    profile = generate_folded_profile(nbins=nbins, ducy=ref_ducy).astype(np.float32)
    bias_snr = float(snr_final / math.sqrt(nstages + 1))
    rng = np.random.default_rng(seed)

    folds_h0 = rng.standard_normal((ntrials, nbins)).astype(np.float32)
    folds_h1 = rng.standard_normal((ntrials, nbins)).astype(np.float32) + (
        profile * np.float32(bias_snr)
    )
    sig_step = (profile * np.float32(bias_snr)).astype(np.float32)

    def _eval_scores(folds: npt.NDArray[np.float32], var_cur: float) -> np.ndarray:
        n = folds.shape[0]
        folds_ev = np.empty((n, 2, nbins), dtype=np.float32)
        folds_ev[:, 0, :] = folds
        folds_ev[:, 1, :] = np.float32(var_cur)
        if score_fn is not None:
            return np.asarray(score_fn(folds_ev), dtype=np.float32)
        return scoring.score_profiles(
            folds_ev,
            widths,
            mode=score_mode,
            n_harmonics=n_harmonics,
        )

    thresholds = np.empty(nstages, dtype=np.float32)
    states_list: list[StateInfo] = []
    complexity = 1.0
    complexity_cumul = 1.0
    h1_cumul = 1.0

    for istage in range(nstages):
        var_cur = float(istage + 2)
        # Stratified refill to ntrials before adding the next segment
        rep0 = math.ceil(ntrials / folds_h0.shape[0])
        h0_sim = np.tile(folds_h0, (rep0, 1))[:ntrials] + rng.standard_normal(
            (ntrials, nbins),
        ).astype(np.float32)
        scores_h0 = _eval_scores(h0_sim, var_cur)

        p_surv = float(np.clip(sp[istage], 0.0, 1.0))
        n_keep = max(1, min(ntrials, round(p_surv * ntrials)))
        order_h0 = np.argsort(scores_h0)[::-1]
        keep_h0 = order_h0[:n_keep]
        thresh = float(scores_h0[keep_h0[-1]])
        thresholds[istage] = np.float32(thresh)
        folds_h0 = np.ascontiguousarray(h0_sim[keep_h0])
        h0_rate = keep_h0.size / float(ntrials)

        if folds_h1.shape[0] > 0:
            rep1 = math.ceil(ntrials / folds_h1.shape[0])
            h1_sim = (
                np.tile(folds_h1, (rep1, 1))[:ntrials]
                + rng.standard_normal((ntrials, nbins)).astype(np.float32)
                + sig_step
            )
            scores_h1 = _eval_scores(h1_sim, var_cur)
            keep_h1 = np.flatnonzero(scores_h1 >= thresh)
            h1_rate = keep_h1.size / float(ntrials)
            folds_h1 = np.ascontiguousarray(h1_sim[keep_h1])
        else:
            h1_rate = 0.0

        nbranches = float(bp[istage])
        nleaves_next = complexity * nbranches
        complexity = nleaves_next * h0_rate
        complexity_cumul += nleaves_next
        h1_cumul *= h1_rate

        states_list.append(
            StateInfo(
                success_h0=h0_rate,
                success_h1=h1_rate,
                complexity=complexity,
                complexity_cumul=complexity_cumul,
                success_h1_cumul=h1_cumul,
                nbranches=nbranches,
                threshold=thresh,
            ),
        )

    return thresholds, StatesInfo(states_list)
