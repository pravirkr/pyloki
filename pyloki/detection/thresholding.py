# ruff: noqa: ARG001

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

import attrs
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from numba import njit, prange, typed, types
from numba.experimental import structref
from numba.extending import overload
from rich.progress import track
from scipy import stats
from sigpyproc.viz.styles import set_seaborn

from pyloki.detection import scoring
from pyloki.utils import np_utils
from pyloki.utils.misc import CONSOLE, get_logger

logger = get_logger(__name__)


def bound_scheme(nstages: int, snr_bound: float) -> npt.NDArray[np.float32]:
    """Threshold scheme using the bound on the target S/N.

    Parameters
    ----------
    nstages : int
        Number of stages in the threshold scheme.
    snr_bound : float
        Upper bound on the target S/N.

    Returns
    -------
    NDArray[np.float32]
        Thresholds for each stage.
    """
    nsegments = nstages + 1
    thresh_sn2 = np.arange(1, nsegments + 1) * snr_bound**2 / nsegments
    thresh_sn2 = thresh_sn2.astype(np.float32)
    return np.sqrt(thresh_sn2[1:])


def trials_scheme(
    branching_pattern: npt.NDArray[np.float32],
    trials_start: int = 1,
) -> npt.NDArray[np.float32]:
    """Threshold scheme using the FAR of the tree.

    Parameters
    ----------
    branching_pattern : NDArray[np.float32]
        Branching pattern for each stage.
    trials_start : int
        Starting number of trials at stage 0, by default 1.

    Returns
    -------
    NDArray[np.float32]
        Thresholds for each stage.
    """
    trials = np.cumprod(branching_pattern) * trials_start
    return stats.norm.isf(1 / trials)


@njit(cache=True, fastmath=True)
def neighbouring_indices(
    beam_indices: npt.NDArray[np.int32],
    target_idx: int,
    num: int,
) -> npt.NDArray[np.int32]:
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
    template: npt.NDArray[np.float32],
    rng: np.random.Generator,
    bias_snr: float = 0,
    var_add: float = 1,
    ntrials_min: int = 1024,
) -> tuple[npt.NDArray[np.float32], float]:
    """Simulate folded profiles by adding signal + noise to the template.

    Parameters
    ----------
    folds : NDArray[np.float32]
        2D Array of folded data with shape (ntrials, nbins).
    var_cur : float
        Current variance of the noise in the folded profiles.
    template : NDArray[np.float32]
        Normalized template of the signal with the same units as the bias_snr.
    bias_snr : float, optional
        Bias signal-to-noise ratio.
    var_add : float, optional
        Variance of the added noise, by default 1.
    ntrials : int, optional
        Number of trials, by default 1024.

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
    folds_sim = np.zeros((ntrials, nbins), dtype=np.float32)
    for irepeat in range(repeat_factor):
        folds_sim[irepeat * ntrials_prev : (irepeat + 1) * ntrials_prev] = folds
    noise = rng.normal(0, np.sqrt(var_add), (ntrials, nbins)).astype(np.float32)
    folds_sim += noise + (bias_snr * template)
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
    scores_arr = np_utils.nb_max(scoring.boxcar_snr_2d(folds_norm, widths, 1.0), axis=1)
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
    folds_var = var_cur * np.ones_like(folds)
    folds_norm = folds / np.sqrt(folds_var)
    folds_norm = folds_norm.astype(np.float32)
    widths = scoring.generate_box_width_trials(
        folds.shape[1],
        ducy_max=ducy_max,
        wtsp=wtsp,
    )
    scores_arr = np_utils.nb_max(scoring.boxcar_snr_2d(folds_norm, widths, 1.0), axis=1)
    n_surviving = int(survive_prob * len(scores_arr))
    good_scores_idx = np.flip(np.argsort(scores_arr))[: int(n_surviving)]
    threshold = scores_arr[good_scores_idx[-1]]
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
    variance : float, optional
        Variance of the noise in the folded profiles, by default 1
    """

    def __new__(
        cls,
        folds_h0: npt.NDArray[np.float32],
        folds_h1: npt.NDArray[np.float32],
        variance: float = 1.0,
    ) -> Self:
        """Create a new instance of Folds."""
        return folds_init(folds_h0, folds_h1, variance)

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
    def variance(self) -> float:
        return self.variance

    @property
    @njit(cache=True, fastmath=True)
    def is_empty(self) -> int:
        return self.is_empty


fields_folds = [
    ("folds_h0", types.f4[:, :]),
    ("folds_h1", types.f4[:, :]),
    ("variance", types.f4),
    ("is_empty", types.int8),
]

structref.define_boxing(FoldsTemplate, Folds)
FoldsType = FoldsTemplate(fields_folds)


@njit(
    FoldsType(types.f4[:, ::1], types.f4[:, ::1], types.f4),
    cache=True,
    fastmath=True,
)
def folds_init(
    folds_h0: npt.NDArray[np.float32],
    folds_h1: npt.NDArray[np.float32],
    variance: float = 1.0,
) -> Folds:
    self = structref.new(FoldsType)
    self.folds_h0 = folds_h0
    self.folds_h1 = folds_h1
    self.variance = variance
    self.is_empty = 0 if (folds_h0.shape[0] > 0 and folds_h1.shape[0] > 0) else 1
    return self


@overload(Folds)
def overload_folds(
    folds_h0: npt.NDArray[np.float32],
    folds_h1: npt.NDArray[np.float32],
    variance: float = 1.0,
) -> types.FunctionType:
    def impl(
        folds_h0: npt.NDArray[np.float32],
        folds_h1: npt.NDArray[np.float32],
        variance: float = 1.0,
    ) -> Folds:
        return folds_init(folds_h0, folds_h1, variance)

    return impl


@njit(cache=True, fastmath=True)
def create_empty_folds() -> Folds:
    empty_h0 = np.empty((0, 1), dtype=np.float32)  # Shape (0, nbins) indicates empty
    empty_h1 = np.empty((0, 1), dtype=np.float32)
    return folds_init(empty_h0, empty_h1, 0.0)  # variance=0 for empty


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
        ("is_empty", np.int8),
    ],
)


@attrs.define(frozen=True)
class StateInfo:
    """Class to save the state of the threshold scheme.

    Parameters
    ----------
    success_h0 : float, optional
        Success probability for H0 hypothesis, by default 1.
    success_h1 : float, optional
        Success probability for H1 hypothesis, by default 1.
    complexity : float, optional
        Number of options for H0 hypothesis, by default 1.
    complexity_cumul : float, optional
        Cumulative complexity/number of options for H0 hypothesis, by default 1.
    success_h1_cumul : float, optional
        Cumulative success/survival probability for H1 hypothesis, by default 1.
    nbranches : float, optional
        Number of branches for the current stage, by default 1.
    """

    success_h0: float
    success_h1: float
    complexity: float
    complexity_cumul: float
    success_h1_cumul: float
    nbranches: int
    threshold: float

    @property
    def cost(self) -> float:
        return self.complexity_cumul / self.success_h1_cumul

    @classmethod
    def from_state(cls, state: np.record) -> Self:
        if state.dtype != state_dtype:
            msg = (
                f"State dtype is not correct, expected {state_dtype}, got {state.dtype}"
            )
            raise ValueError(msg)
        return cls(
            success_h0=float(state["success_h0"]),
            success_h1=float(state["success_h1"]),
            complexity=float(state["complexity"]),
            complexity_cumul=float(state["complexity_cumul"]),
            success_h1_cumul=float(state["success_h1_cumul"]),
            nbranches=int(state["nbranches"]),
            threshold=float(state["threshold"]),
        )


@attrs.define(frozen=True)
class StatesInfo:
    """Class to handle the information of the states in the threshold scheme."""

    entries: list[StateInfo] = attrs.Factory(list)

    @property
    def thresholds(self) -> npt.NDArray[np.float32]:
        """Get list of thresholds for this scheme."""
        return np.array([entry.threshold for entry in self.entries], dtype=np.float32)

    def get_info(self, key: str) -> npt.NDArray[np.float32]:
        """Get list of values for a given key for all entries."""
        return np.array(
            [getattr(entry, key) for entry in self.entries],
            dtype=np.float32,
        )

    def print_summary(self) -> None:
        """Print a summary of the threshold scheme."""
        branching_pattern = self.get_info("nbranches")
        survive_prob = self.get_info("success_h1_cumul")[-1]
        pruning_complexity = self.get_info("complexity_cumul")[-1]
        total_cost = self.get_info("cost")[-1]
        mean_exp_bp = 2 ** (np.mean(np.log2(branching_pattern)))
        n_options = np.sum(np.log2(branching_pattern))
        n_independent = len(self.thresholds) / self.thresholds[-1]
        total_survive_prob = 1 - (1 - survive_prob) ** n_independent

        info_str = "Threshold scheme summary:\n"
        info_str += f"Branching mean exponential growth: {mean_exp_bp:.2f}\n"
        info_str += f"Branching max exponential growth: {max(branching_pattern)}\n"
        info_str += f"Total enumerated options: {n_options:.2f}\n"
        info_str += f"Pruning complexity: {pruning_complexity:.2f}\n"
        info_str += f"Crude survival probability: {survive_prob:.2f}\n"
        info_str += f"Total cost: {total_cost:.2f}\n"
        info_str += f"Number of independent trials: {n_independent:.2f}\n"
        info_str += f"Total survival probability: {total_survive_prob:.2f}"
        print(info_str)  # noqa: T201

    def save(self, filename: str) -> None:
        """Save the StatesInfo object to a file."""
        with Path(filename).open("w") as fp:
            json.dump(attrs.asdict(self), fp)

    @classmethod
    def load(cls, filename: str) -> StatesInfo:
        """Load a StatesInfo object from a file."""
        with Path(filename).open("r") as fp:
            data = json.load(fp)
            entries = [StateInfo(**entry) for entry in data["entries"]]
            return cls(entries=entries)


@njit(cache=True, fastmath=True)
def get_next_state(
    state_cur: npt.NDArray,
    threshold: float,
    success_h0: float,
    success_h1: float,
    nbranches: int,
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
    state_next[0]["cost"] = complexity_cumul / success_h1_cumul
    # For backtracking
    state_next[0]["threshold_prev"] = state_cur["threshold"]
    state_next[0]["success_h1_cumul_prev"] = state_cur["success_h1_cumul"]
    return state_next


@njit(cache=True, fastmath=True)
def gen_next_using_thresh(
    state_cur: npt.NDArray,
    folds_cur: Folds,
    threshold: float,
    nbranches: int,
    bias_snr: float,
    template: npt.NDArray[np.float32],
    rng: np.random.Generator,
    ntrials: int = 1024,
    ducy_max: float = 0.2,
    wtsp: float = 1.0,
) -> tuple[npt.NDArray, Folds]:
    folds_h0, variance = simulate_folds(
        folds_cur.folds_h0,
        folds_cur.variance,
        template,
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
        template,
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
    nbranches: int,
    bias_snr: float,
    template: npt.NDArray[np.float32],
    rng: np.random.Generator,
    ntrials: int = 1024,
    ducy_max: float = 0.2,
    wtsp: float = 1.0,
) -> tuple[npt.NDArray, Folds]:
    folds_h0, variance = simulate_folds(
        folds_cur.folds_h0,
        folds_cur.variance,
        template,
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
        template,
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


@njit(cache=True, parallel=True, fastmath=True)
def run_stage(
    istage: int,
    beam_idx_cur: npt.NDArray,
    beam_idx_prev: npt.NDArray,
    states: npt.NDArray,
    folds_in: types.ListType[FoldsTemplate],
    folds_out: types.ListType[FoldsTemplate],
    probs: npt.NDArray,
    nbranches: int,
    thresholds: npt.NDArray[np.float32],
    bias_snr: float,
    template: npt.NDArray[np.float32],
    rng: np.random.Generator,
    ntrials: int,
    ducy_max: float,
    wtsp: float,
    thres_neigh: int,
) -> None:
    nprobs = len(probs)
    for ibeam_cur in prange(len(beam_idx_cur)):
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
                if prev_state["is_empty"] == 1:
                    continue
                prev_fold_state = folds_in[fold_idx]
                if prev_fold_state.is_empty == 0:
                    cur_state, cur_fold_state = gen_next_using_thresh(
                        prev_state,
                        prev_fold_state,
                        thresholds[ithres],
                        nbranches,
                        bias_snr,
                        template,
                        rng,
                        ntrials,
                        ducy_max,
                        wtsp,
                    )
                    cur_state = cur_state[0]  # Get record from array
                    iprob_val = np.digitize(cur_state["success_h1_cumul"], probs) - 1
                    iprob = int(iprob_val.item())  # Extract scalar
                    if iprob < 0 or iprob >= nprobs:  # Clamp to valid range
                        continue
                    existing_state = states[istage, ithres, iprob]
                    if (
                        existing_state["is_empty"] == 1
                        or cur_state["complexity_cumul"]
                        < existing_state["complexity_cumul"]
                    ):
                        states[istage, ithres, iprob] = cur_state
                        folds_out[ithres * nprobs + iprob] = cur_fold_state


class DynamicThresholdScheme:
    def __init__(
        self,
        branching_pattern: np.ndarray,
        template: np.ndarray,
        ntrials: int = 1024,
        nprobs: int = 10,
        prob_min: float = 0.05,
        snr_final: float = 8,
        nthresholds: int = 100,
        ducy_max: float = 0.2,
        wtsp: float = 1.0,
        beam_width: float = 0.7,
    ) -> None:
        self.rng = np.random.default_rng()
        self.branching_pattern = branching_pattern
        self.template = template / np.sqrt(np.sum(template**2))
        self.ntrials = ntrials
        self.snr_final = snr_final
        self.ducy_max = ducy_max
        self.wtsp = wtsp
        self.beam_width = beam_width

        # later define as snr^2
        self.thresholds = np.linspace(0.1, self.snr_final, nthresholds)
        self.prob_min = prob_min
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
        self.init()

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
        return len(self.template)

    def get_current_thresholds_idx(self, istage: int) -> np.ndarray:
        guess = self.guess_path[istage]
        half_extent = self.beam_width
        lower_bound = max(0, guess - half_extent)
        upper_bound = min(self.snr_final, guess + half_extent)
        return np.where(
            (self.thresholds >= lower_bound) & (self.thresholds <= upper_bound),
        )[0]

    def init(self) -> None:
        var_init = 1.0
        folds = np.zeros((self.ntrials, self.nbins), dtype=np.float32)
        folds_h0, _ = simulate_folds(
            folds,
            0,
            self.template,
            self.rng,
            bias_snr=0,
            var_add=var_init,
            ntrials_min=self.ntrials,
        )
        folds_h1, _ = simulate_folds(
            folds,
            0,
            self.template,
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
                self.template,
                self.rng,
                self.ntrials,
                self.ducy_max,
                self.wtsp,
            )
            cur_state = cur_state[0]  # Get record from array
            iprob = np.digitize(cur_state["success_h1_cumul"], self.probs) - 1
            if iprob < 0:
                continue
            self.states[0, ithres, iprob] = cur_state
            folds_idx = int(ithres * self.nprobs + iprob)
            self.folds_in[folds_idx] = cur_fold_state

    def run(self, thres_neigh: int = 11) -> None:
        for istage in track(
            range(1, self.nstages),
            description="Running stages",
            console=CONSOLE,
            transient=True,
        ):
            beam_idx_cur = self.get_current_thresholds_idx(istage)
            beam_idx_prev = self.get_current_thresholds_idx(istage - 1)
            run_stage(
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
                self.template,
                self.rng,
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
            f"dynscheme_nstages_{self.nstages}_"
            f"nthresh_{self.nthresholds}_nprobs_{self.nprobs}_"
            f"ntrials_{self.ntrials}_snr_{self.snr_final:.1f}_"
            f"beam_{self.beam_width:.1f}"
        )
        filename = Path(outdir) / f"{filebase}.h5"
        with h5py.File(filename, "w") as f:
            # Save simple attributes
            for attr in ["ntrials", "snr_final", "ducy_max", "wtsp", "beam_width"]:
                f.attrs[attr] = getattr(self, attr)
            # Save numpy arrays
            for arr in [
                "branching_pattern",
                "template",
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
        return filename.as_posix()


class DynamicThresholdSchemeAnalyser:
    def __init__(
        self,
        states: npt.NDArray,
        thresholds: npt.NDArray[np.float32],
        probs: npt.NDArray[np.float32],
        branching_pattern: npt.NDArray[np.float32],
        guess_path: npt.NDArray[np.float32],
        beam_width: float,
    ) -> None:
        self.states = states
        self.thresholds = thresholds
        self.probs = probs
        self.branching_pattern = branching_pattern
        self.guess_path = guess_path
        self.beam_width = beam_width
        self.nstages, self.nthresholds, self.nprobs = states.shape

    def backtrack_best(self, min_probs: list[float] | None) -> list[StatesInfo]:
        """Backtrack the best paths in the threshold scheme.

        Parameters
        ----------
        min_probs : list[float] | None
            Minimum success probability for the best path, by default None.

        Returns
        -------
        list[StatesInfo]
            List of the best paths as StatesInfo objects.
        """
        if min_probs is None:
            min_probs = [self.probs[0]]
        final_states = self.states[-1][self.states[-1]["is_empty"] == 0]
        backtrack_states_info = []
        for min_prob in min_probs:
            filtered_states = final_states[final_states["success_h1_cumul"] >= min_prob]
            if len(filtered_states) == 0:
                logger.warning(f"No states found for min_prob {min_prob}")
                backtrack_states_info.append(StatesInfo([]))
                continue
            best_state = filtered_states[np.argmin(filtered_states["cost"])]
            backtrack_states = [StateInfo.from_state(best_state)]
            prev_threshold = best_state["threshold_prev"]
            prev_success_h1_cumul = best_state["success_h1_cumul_prev"]
            for istage in range(self.nstages - 2, -1, -1):
                ithres = np.argmin(np.abs(self.thresholds - prev_threshold))
                iprob = np.digitize(prev_success_h1_cumul, self.probs) - 1
                if iprob < 0:
                    msg = (
                        f"Backtracking failed at stage {istage} for threshold "
                        f"{prev_threshold} and success probability "
                        f"{prev_success_h1_cumul}"
                    )
                    raise ValueError(msg)
                prev_state = self.states[istage, ithres, iprob]
                if prev_state["is_empty"] == 1:
                    msg = (
                        f"Backtracking failed at stage {istage} for threshold "
                        f"{prev_threshold} and success probability "
                        f"{prev_success_h1_cumul}"
                    )
                    raise ValueError(msg)
                backtrack_states.insert(0, StateInfo.from_state(prev_state))
                prev_threshold = prev_state["threshold_prev"]
                prev_success_h1_cumul = prev_state["success_h1_cumul_prev"]
            backtrack_states_info.append(StatesInfo(backtrack_states))
        return backtrack_states_info

    def backtrack_all(self, min_prob: float) -> list[StatesInfo]:
        """Backtrack all paths in the threshold scheme that meet the minimum success.

        Parameters
        ----------
        min_prob : float
            Minimum success probability for the best path.

        Returns
        -------
        list[StatesInfo]
            List of the best paths as StatesInfo objects.
        """
        final_states = self.states[-1][self.states[-1]["is_empty"] == 0]
        filtered_states = final_states[final_states["success_h1_cumul"] >= min_prob]
        if len(filtered_states) == 0:
            return []
        backtrack_states_info = []
        for final_state in filtered_states:
            backtrack_states = [StateInfo.from_state(final_state)]
            prev_threshold = final_state["threshold_prev"]
            prev_success_h1_cumul = final_state["success_h1_cumul_prev"]
            for istage in range(self.nstages - 2, -1, -1):
                ithres = np.argmin(np.abs(self.thresholds - prev_threshold))
                iprob = np.digitize(prev_success_h1_cumul, self.probs) - 1
                if iprob < 0:
                    msg = (
                        f"Backtracking failed at stage {istage} for threshold "
                        f"{prev_threshold} and success probability "
                        f"{prev_success_h1_cumul}"
                    )
                    raise ValueError(msg)
                prev_state = self.states[istage, ithres, iprob]
                if prev_state["is_empty"] == 1:
                    msg = (
                        f"Backtracking failed at stage {istage} for threshold "
                        f"{prev_threshold} and success probability "
                        f"{prev_success_h1_cumul}"
                    )
                    raise ValueError(msg)
                backtrack_states.insert(0, StateInfo.from_state(prev_state))
                prev_threshold = prev_state["threshold_prev"]
                prev_success_h1_cumul = prev_state["success_h1_cumul_prev"]
            backtrack_states_info.append(StatesInfo(backtrack_states))
        return backtrack_states_info

    def plot_paths(self, best_prob: float, min_prob: float) -> plt.Figure:
        paths = self.backtrack_all(min_prob)
        best_path = self.backtrack_best(min_probs=[best_prob])[0]
        set_seaborn(**{"lines.linewidth": 1.5})
        fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(12, 8))  # type: ignore[misc]
        x = np.arange(1, self.nstages + 1)

        label = f"Best: P(H1) = {best_prob:.2f}"
        for path in paths:
            ax1.plot(x, path.thresholds**2, "b-", alpha=0.2)
        ax1.plot(x, best_path.thresholds**2, "r-", label=label)
        ax1.plot(x, self.guess_path**2, color="navy", ls="--", label="Guess path")
        upper_bound = np.minimum(self.guess_path + self.beam_width, self.thresholds[-1])
        lower_bound = np.maximum(self.guess_path - self.beam_width, 0)
        ax1.fill_between(
            x,
            lower_bound**2,
            upper_bound**2,
            color="cornflowerblue",
            alpha=0.2,
            label="Beam width",
        )
        ax1.set_xlabel("Pruning stage")
        ax1.set_ylabel("S/N squared")
        ax1.set_title("Threshold scheme")
        ax1.legend(fontsize="small")
        ax1.set_ylim(-0.5, self.thresholds[-1] ** 2 + 0.5)
        for ax, info, title, ylabel in [
            (
                ax2,
                "complexity",
                "False Alarm",
                r"Number of $H_{0}$ options",
            ),
            (
                ax3,
                "complexity_cumul",
                r"Cumulative complexity $H_{0}$",
                r"Cumulative $H_{0}$ complexity",
            ),
            (
                ax4,
                "success_h1_cumul",
                r"Cumulative Success $H_{1}$",
                "Detection Probability",
            ),
        ]:
            for path in paths:
                ax.plot(x, path.get_info(info), "b-", alpha=0.2)
            ax.plot(x, best_path.get_info(info), "r-", label="Best path")
            ax.set_xlabel("Pruning stage")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_yscale("log")
        ax2_current_ylim = ax2.get_ylim()
        ax2.plot(
            x,
            np.cumprod(self.branching_pattern),
            color="k",
            ls="--",
            label="Total options",
        )
        ax2.set_ylim(bottom=0.05, top=ax2_current_ylim[1])
        ax2.legend(fontsize="small", loc="upper right")
        # Show grid in probability plots
        for pval in self.probs[1:]:
            ax4.axhline(y=pval, color="gray", alpha=0.15, linestyle="-", zorder=0)
        ax4.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        plt.tight_layout()
        return fig

    def plot_slice(
        self,
        stage: int,
        attribute: str = "success_h1_cumul",
        fmt: str = ".3f",
        cmap: str = "viridis",
        figsize: tuple[float, float] = (12, 8),
        annot_size: float = 8,
    ) -> plt.Figure:
        if not 0 <= stage < self.nstages:
            msg = f"Stage must be between 0 and {self.nstages - 1}, got {stage}"
            raise ValueError(msg)
        if self.states.dtype.names is None or attribute not in self.states.dtype.names:
            msg = f"Attribute must be one of {self.states.dtype.names}, got {attribute}"
            raise ValueError(msg)

        # Extract 2D slice directly from structured array
        cum_score = self.states[stage, :, :][attribute].astype(float)
        mask = self.states[stage, :, :]["is_empty"] == 1
        cum_score[mask] = np.nan

        df = pd.DataFrame(
            cum_score,
            index=pd.Index(self.thresholds, name="Thresholds", dtype=np.float32),
            columns=pd.Index(range(self.nprobs), name="Success Prob bins"),
        )
        set_seaborn()
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            df,
            annot=True,
            annot_kws={"size": annot_size},
            fmt=fmt,
            cmap=cmap,
            linewidth=0.5,
            linecolor="gray",
            cbar_kws={
                "label": attribute.replace("_", " ").title(),
                "shrink": 0.8,
                "pad": 0.02,
            },
            ax=ax,
        )
        ax.set_title(f"Stage {stage}: {attribute.replace('_', ' ').title()}")
        ax.invert_yaxis()
        plt.tight_layout()
        return fig

    def save(self, filename: str) -> None:
        """Save the StatesInfo object to a file."""
        np.savez(
            filename,
            thresholds=self.thresholds,
            probs=self.probs,
            states=self.states,
            allow_pickle=True,
        )

    @classmethod
    def from_file(cls, filename: str) -> DynamicThresholdSchemeAnalyser:
        """Load a DynamicThresholdScheme object from an hdf5 file."""
        with h5py.File(filename, "r") as f:
            branching_pattern = f["branching_pattern"][:]
            thresholds = f["thresholds"][:]
            probs = f["probs"][:]
            guess_path = f["guess_path"][:]
            beam_width = f.attrs["beam_width"]
            states = f["states"][:]
        return cls(
            states,
            thresholds,
            probs,
            branching_pattern,
            guess_path,
            beam_width,
        )


def determine_scheme(
    survive_probs: np.ndarray,
    branching_pattern: np.ndarray,
    profile: np.ndarray,
    ntrials: int = 2048,
    snr_final: float = 8,
    ducy_max: float = 0.2,
    wtsp: float = 1.0,
) -> StatesInfo:
    if len(survive_probs) != len(branching_pattern):
        msg = "Number of survive_probs must match the number of stages"
        raise ValueError(msg)

    var_init = 1.0
    nstages = len(branching_pattern)
    template = profile / np.sqrt(np.sum(profile**2))
    bias_snr = snr_final / np.sqrt(nstages + 1)
    rng = np.random.default_rng()
    states: list[np.record] = []
    fold_states: list[Folds] = []
    folds = np.zeros((ntrials, len(template)), dtype=np.float32)
    folds_h0, _ = simulate_folds(folds, 0, template, rng, 0, var_init, ntrials)
    folds_h1, _ = simulate_folds(folds, 0, template, rng, bias_snr, var_init, ntrials)
    initial_state = np.ones(1, dtype=state_dtype)[0]
    initial_state["threshold"] = 0
    initial_state["is_empty"] = 0
    initial_fold_state = Folds(folds_h0, folds_h1, var_init)
    for istage in range(nstages):
        prev_state = initial_state if istage == 0 else states[istage - 1]
        prev_fold_state = initial_fold_state if istage == 0 else fold_states[istage - 1]
        if istage > 0 and prev_fold_state.is_empty == 1:
            logger.info("Path not viable, No trials survived, stopping")
            break
        cur_state, cur_fold_state = gen_next_using_surv_prob(
            prev_state,
            prev_fold_state,
            survive_probs[istage],
            branching_pattern[istage],
            bias_snr,
            template,
            rng,
            ntrials,
            ducy_max,
            wtsp,
        )
        cur_state = cur_state[0]  # Get record from array
        states.append(cur_state)
        fold_states.append(cur_fold_state)
    return StatesInfo([StateInfo.from_state(state) for state in states])


def evaluate_scheme(
    thresholds: np.ndarray,
    branching_pattern: np.ndarray,
    profile: np.ndarray,
    ntrials: int = 2048,
    snr_final: float = 8,
    ducy_max: float = 0.2,
    wtsp: float = 1.0,
) -> StatesInfo:
    var_init = 1.0
    nstages = len(branching_pattern)
    template = profile / np.sqrt(np.sum(profile**2))
    bias_snr = snr_final / np.sqrt(nstages + 1)
    rng = np.random.default_rng()
    if len(thresholds) != nstages:
        msg = "Number of thresholds must match the number of stages"
        raise ValueError(msg)
    states: list[np.record] = []
    fold_states: list[Folds] = []
    folds = np.zeros((ntrials, len(template)), dtype=np.float32)
    folds_h0, _ = simulate_folds(folds, 0, template, rng, 0, var_init, ntrials)
    folds_h1, _ = simulate_folds(folds, 0, template, rng, bias_snr, var_init, ntrials)
    initial_state = np.ones(1, dtype=state_dtype)[0]
    initial_state["threshold"] = 0
    initial_state["is_empty"] = 0
    initial_fold_state = Folds(folds_h0, folds_h1, var_init)
    for istage in range(nstages):
        prev_state = initial_state if istage == 0 else states[istage - 1]
        prev_fold_state = initial_fold_state if istage == 0 else fold_states[istage - 1]
        if istage > 0 and prev_fold_state.is_empty == 1:
            logger.info("Path not viable, No trials survived, stopping")
            break
        cur_state, cur_fold_state = gen_next_using_thresh(
            prev_state,
            prev_fold_state,
            thresholds[istage],
            branching_pattern[istage],
            bias_snr,
            template,
            rng,
            ntrials,
            ducy_max,
            wtsp,
        )
        cur_state = cur_state[0]  # Get record from array
        states.append(cur_state)
        fold_states.append(cur_fold_state)
    return StatesInfo([StateInfo.from_state(state) for state in states])
