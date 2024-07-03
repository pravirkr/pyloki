from __future__ import annotations

import itertools
import pickle
from pathlib import Path

import attrs
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numba import njit, types
from numba.experimental import jitclass
from rich.progress import track
from scipy import stats

from pruning import kernels, scores


def threshold_scheme_bound(nsegments: int, bound: float) -> np.ndarray:
    """Threshold scheme using the bound on the target S/N.

    Parameters
    ----------
    nsegments : int
        Number of data segments
    bound : float
        Upper bound on the target S/N

    Returns
    -------
    np.ndarray
        Thresholds for each iteration
    """
    thresh_sn2 = np.arange(1, nsegments + 1) * bound**2 / nsegments
    return np.sqrt(thresh_sn2)


def threshold_scheme_trials(
    nsegments: int,
    nparams: int,
    sugg_size: int = 1,
) -> np.ndarray:
    """Threshold scheme using the number of param trials.

    Parameters
    ----------
    nsegments : int
        Number of data segments
    sugg_size : int
        Current number of suggestions
    nparams : int
        Number of search parameters

    Returns
    -------
    np.ndarray
        Thresholds for each iteration
    """
    complexity_scaling = np.sum(np.arange(nparams) + 1)
    trials = sugg_size * (np.arange(nsegments) + 1) ** complexity_scaling
    return stats.norm.isf(1 / trials)


@njit
def neighbouring_indices(arr: np.ndarray, target: float, num: int) -> np.ndarray:
    index = np.argwhere(arr == target)[0][0]

    # Calculate the window around the target
    left = max(0, index - num // 2)
    right = min(len(arr), index + num // 2 + 1)

    # Adjust the window if necessary for the end points
    while right - left < num:
        if left == 0:
            right += 1
        elif right == len(arr) or index - left < right - index:
            left -= 1
        else:
            right += 1

    return np.arange(left, right)


def init_state_using_surv_prob(
    survive_prob: float,
    template: np.ndarray,
    bias_snr: float,
    nbranches: int,
    rng: np.random.Generator,
    ntrials: int = 1024,
    ducy_max: float = 0.2,
    var_init: float = 2,
) -> tuple[State, Folds]:
    folds = np.zeros((ntrials, len(template)), dtype=np.float32)
    folds_h0, variance = simulate_folds(
        folds,
        0,
        template,
        rng,
        bias_snr=0,
        var_add=var_init,
        ntrials=ntrials,
    )
    folds_h1, variance = simulate_folds(
        folds,
        0,
        template,
        rng,
        bias_snr=bias_snr,
        var_add=var_init,
        ntrials=ntrials,
    )
    thresold_h0, success_h0, folds_h0 = measure_threshold(
        folds_h0,
        variance,
        survive_prob,
        ducy_max=ducy_max,
    )
    success_h1, folds_h1 = measure_success(
        folds_h1,
        variance,
        thresold_h0,
        ducy_max=ducy_max,
    )
    complexity = nbranches * success_h0
    backtrack = [(thresold_h0, success_h1)]
    state = State(
        success_h0,
        success_h1,
        complexity,
        complexity,
        success_h1,
        backtrack,
    )
    fold_state = Folds(folds_h0, folds_h1, variance)
    return state, fold_state


def init_states_dynamic(
    template: np.ndarray,
    thresholds: np.ndarray,
    bias_snr: float,
    nbranches: int,
    rng: np.random.Generator,
    ntrials: int = 1024,
    ducy_max: float = 0.2,
    var_init: float = 2,
) -> tuple[list[State], list[Folds]]:
    folds = np.zeros((ntrials, len(template)), dtype=np.float32)
    folds_h0, variance = simulate_folds(
        folds,
        0,
        template,
        rng,
        bias_snr=0,
        var_add=var_init,
        ntrials=ntrials,
    )
    folds_h1, variance = simulate_folds(
        folds,
        0,
        template,
        rng,
        bias_snr=bias_snr,
        var_add=var_init,
        ntrials=ntrials,
    )
    states = []
    fold_states = []
    for ithres in range(len(thresholds)):
        threshold = thresholds[ithres]
        success_h0, folds_h0_pass = measure_success(
            folds_h0,
            variance,
            threshold,
            ducy_max=ducy_max,
        )
        success_h1, folds_h1_pass = measure_success(
            folds_h1,
            variance,
            threshold,
            ducy_max=ducy_max,
        )
        complexity = nbranches * success_h0
        backtrack = [(threshold, success_h1)]
        state_init = State(
            success_h0,
            success_h1,
            complexity,
            complexity,
            success_h1,
            backtrack,
        )
        fold_state_init = Folds(folds_h0_pass, folds_h1_pass, variance)
        states.append(state_init)
        fold_states.append(fold_state_init)
    return states, fold_states


@jitclass(
    [
        ("folds_h0", types.f4[:, :]),
        ("folds_h1", types.f4[:, :]),
        ("variance", types.f8),
    ],
)
class Folds:
    def __init__(
        self,
        folds_h0: np.ndarray,
        folds_h1: np.ndarray,
        variance: float = 1,
    ) -> None:
        self.folds_h0 = folds_h0
        self.folds_h1 = folds_h1
        self.variance = variance

    @property
    def is_empty(self) -> bool:
        return len(self.folds_h0) == 0 or len(self.folds_h1) == 0


@attrs.define(frozen=True)
class State:
    """Class to handle the information of the state in the threshold scheme."""

    success_h0: float
    success_h1: float
    complexity: float
    complexity_cumul: float
    success_h1_cumul: float
    backtrack: list[tuple[float, float]]

    @property
    def cost(self) -> float:
        return self.complexity_cumul / self.success_h1_cumul

    def gen_next_using_surv_prob(
        self,
        prev_fold_state: Folds,
        surv_prob_h0: float,
        bias_snr: float,
        template: np.ndarray,
        nbranches: int,
        rng: np.random.Generator,
        ntrials: int = 1024,
        ducy_max: float = 0.2,
    ) -> tuple[State, Folds]:
        folds_h0, variance = simulate_folds(
            prev_fold_state.folds_h0,
            prev_fold_state.variance,
            template,
            rng,
            bias_snr=0,
            var_add=1,
            ntrials=ntrials,
        )
        thresold_h0, success_h0, folds_h0 = measure_threshold(
            folds_h0,
            variance,
            surv_prob_h0,
            ducy_max=ducy_max,
        )
        return self._gen_next_h1(
            prev_fold_state,
            folds_h0,
            success_h0,
            thresold_h0,
            bias_snr,
            template,
            nbranches,
            rng,
            ntrials,
            ducy_max,
        )

    def gen_next_using_thresh(
        self,
        prev_fold_state: Folds,
        threshold: float,
        bias_snr: float,
        template: np.ndarray,
        nbranches: int,
        rng: np.random.Generator,
        ntrials: int = 1024,
        ducy_max: float = 0.2,
    ) -> tuple[State, Folds]:
        folds_h0, variance = simulate_folds(
            prev_fold_state.folds_h0,
            prev_fold_state.variance,
            template,
            rng,
            bias_snr=0,
            var_add=1,
            ntrials=ntrials,
        )
        success_h0, folds_h0 = measure_success(
            folds_h0,
            variance,
            threshold,
            ducy_max=ducy_max,
        )
        return self._gen_next_h1(
            prev_fold_state,
            folds_h0,
            success_h0,
            threshold,
            bias_snr,
            template,
            nbranches,
            rng,
            ntrials,
            ducy_max,
        )

    def _gen_next_h1(
        self,
        prev_fold_state: Folds,
        folds_h0: np.ndarray,
        success_h0: float,
        threshold: float,
        bias_snr: float,
        template: np.ndarray,
        nbranches: int,
        rng: np.random.Generator,
        ntrials: int = 1024,
        ducy_max: float = 0.2,
    ) -> tuple[State, Folds]:
        folds_h1, variance = simulate_folds(
            prev_fold_state.folds_h1,
            prev_fold_state.variance,
            template,
            rng,
            bias_snr=bias_snr,
            var_add=1,
            ntrials=ntrials,
        )
        success_h1, folds_h1 = measure_success(
            folds_h1,
            variance,
            threshold,
            ducy_max=ducy_max,
        )
        complexity = self.complexity * nbranches * success_h0
        complexity_cumul = self.complexity_cumul + complexity
        success_h1_cumul = self.success_h1_cumul * success_h1
        backtrack = self.backtrack.copy()
        backtrack.append((threshold, success_h1_cumul))
        next_state = State(
            success_h0,
            success_h1,
            complexity,
            complexity_cumul,
            success_h1_cumul,
            backtrack,
        )
        next_fold_state = Folds(folds_h0, folds_h1, variance)
        return next_state, next_fold_state


@attrs.define(frozen=True)
class StatesInfo:
    """Class to handle the information of the states in the threshold scheme."""

    entries: list[State] = attrs.Factory(list)

    def get_info(self, key: str) -> list:
        """Get list of values for a given key for all entries."""
        return [getattr(entry, key) for entry in self.entries]

    def save(self, filename: str) -> None:
        """Save the StatesInfo object to a file."""
        with Path(filename).open("wb") as fp:
            pickle.dump(self, fp)

    @classmethod
    def load(cls, filename: str) -> StatesInfo:
        """Load a StatesInfo object from a file."""
        with Path(filename).open("rb") as fp:
            return pickle.load(fp)


@njit(cache=True, fastmath=True)
def simulate_folds(
    folds: np.ndarray,
    var_cur: float,
    template: np.ndarray,
    rng: np.random.Generator,
    bias_snr: float = 0,
    var_add: float = 1,
    ntrials: int = 1024,
) -> tuple[np.ndarray, float]:
    """Simulate folded profiles by adding signal + noise to the template.

    Parameters
    ----------
    folds : np.ndarray
        2D Array of folded data with shape (ntrials, nbins).
    var_cur : float
        Current variance of the noise.
    template : np.ndarray
        Normalized template of the signal with the same units as the bias_snr.
    bias_snr : float, optional
        Bias signal-to-noise ratio
    var_add : float, optional
        Variance of the added noise, by default 1
    ntrials : int, optional
        Number of trials, by default 1024

    Returns
    -------
    tuple[np.ndarray, float]
        Array of simulated folded data and the updated variance.
    """
    ntrials_prev, nbins = folds.shape
    folds_sim = np.zeros((ntrials, nbins), dtype=folds.dtype)
    for itrial in range(ntrials):
        folds_sim[itrial] = folds[itrial % ntrials_prev]
    noise = rng.normal(0, np.sqrt(var_add), (ntrials, nbins)).astype(folds.dtype)
    folds_sim += noise + bias_snr * template
    var_cur += var_add
    return folds_sim, var_cur


@njit(cache=True, fastmath=True)
def measure_success(
    folds: np.ndarray,
    var_cur: float,
    snr_thresh: float,
    ducy_max: float = 0.2,
) -> tuple[float, np.ndarray]:
    """Measure the success probability of signal detection.

    Parameters
    ----------
    folds : np.ndarray
        2D Array of folded data with shape (ntrials, nbins).
    var_cur : float
        Current variance of the noise.
    snr_thresh : float
        Signal-to-noise ratio threshold.
    ducy_max : float, optional
        Maximum duty cycle for boxcar search, by default 0.2

    Returns
    -------
    tuple[float, np.ndarray]
        Success probability and the folds that passed the threshold.
    """
    folds_norm = folds / np.sqrt(var_cur * np.ones_like(folds))
    widths = scores.generate_width_trials(folds.shape[1], ducy_max=ducy_max, wtsp=1)
    scores_arr = kernels.nb_max(scores.boxcar_snr(folds_norm, widths), axis=1)
    good_scores_idx = np.nonzero(scores_arr > snr_thresh)[0]
    succ_prob = len(good_scores_idx) / len(scores_arr)
    return succ_prob, folds[good_scores_idx]


@njit(cache=True, fastmath=True)
def measure_threshold(
    folds: np.ndarray,
    var_cur: float,
    survive_prob: float,
    ducy_max: float = 0.2,
) -> tuple[float, float, np.ndarray]:
    folds_norm = folds / np.sqrt(var_cur * np.ones_like(folds))
    widths = scores.generate_width_trials(folds.shape[1], ducy_max=ducy_max, wtsp=1)
    scores_arr = kernels.nb_max(scores.boxcar_snr(folds_norm, widths), axis=1)
    n_surviving = int(survive_prob * len(scores_arr))
    good_scores_idx = np.flip(np.argsort(scores_arr))[: int(n_surviving)]
    succ_prob = len(good_scores_idx) / len(scores_arr)
    threshold = scores_arr[good_scores_idx[-1]]
    return threshold, succ_prob, folds[good_scores_idx]


class DynamicThresholdScheme:
    def __init__(
        self,
        branching_pattern: np.ndarray,
        template: np.ndarray,
        ntrials: int = 1024,
        nprobs: int = 10,
        snr_final: float = 8,
        snr_step: float = 0.1,
        ducy_max: float = 0.2,
    ) -> None:
        self.ntrials = ntrials
        self.branching_pattern = branching_pattern
        self.snr_final = snr_final
        self.template = template / np.sqrt(np.sum(template**2))
        self.ducy_max = ducy_max
        self.rng = np.random.default_rng()

        # later define as snr^2
        self._thresholds = np.arange(0, snr_final, snr_step) + snr_step
        self._probs = 1 - np.logspace(-3, 0, nprobs, base=np.e)[::-1]
        self._states = np.empty(
            (self.nsegments, self.nthresholds, self.nprobs),
            dtype=object,
        )
        self._folds_in = np.empty((self.nthresholds, self.nprobs), dtype=object)
        self._folds_out = np.empty((self.nthresholds, self.nprobs), dtype=object)
        self.thresh_bound_scheme = threshold_scheme_bound(self.nsegments, snr_final)
        self.init()

    @property
    def nsegments(self) -> int:
        return len(self.branching_pattern)

    @property
    def bias_snr(self) -> float:
        return self.snr_final / np.sqrt(self.nsegments)

    @property
    def thresholds(self) -> np.ndarray:
        return self._thresholds

    @property
    def nthresholds(self) -> int:
        return len(self.thresholds)

    @property
    def probs(self) -> np.ndarray:
        return self._probs

    @property
    def nprobs(self) -> int:
        return len(self.probs)

    @property
    def states(self) -> np.ndarray[State]:
        return self._states

    def init(self) -> None:
        print("Initializing the threshold scheme")
        states, fold_states = init_states_dynamic(
            self.template,
            self.thresholds,
            self.bias_snr,
            self.branching_pattern[0],
            self.rng,
            self.ntrials,
            self.ducy_max,
            var_init=2,
        )
        for ithres, state_init in enumerate(states):
            iprob = np.digitize(state_init.success_h1_cumul, self.probs) - 1
            self._states[0, ithres, iprob] = state_init
            self._folds_in[ithres, iprob] = fold_states[ithres]

    def run(self, thresh_neighbours_tol: float = 20) -> None:
        for istage in track(range(1, self.nsegments), description="Running"):
            self.run_stage(istage, thresh_neighbours_tol)
            self._folds_in = self._folds_out
            self._folds_out = np.empty((self.nthresholds, self.nprobs), dtype=object)

    def run_stage(self, istage: int, thresh_neighbours_tol: float = 20) -> None:
        for ithres in range(self.nthresholds):
            # Iterate over all prevoius neighbouring thresholds
            neighbour_thresholds_ids = neighbouring_indices(
                self.thresholds,
                self.thresholds[ithres],
                thresh_neighbours_tol,
            )
            cur_states = []
            for jthres, kprob in itertools.product(
                neighbour_thresholds_ids,
                range(self.nprobs),
            ):
                prev_state = self.states[istage - 1, jthres, kprob]
                prev_fold_state = self._folds_in[jthres, kprob]
                if prev_fold_state is None or prev_fold_state.is_empty:
                    continue
                cur_state, cur_fold_state = prev_state.gen_next_using_thresh(
                    prev_fold_state,
                    self.thresholds[ithres],
                    self.bias_snr,
                    self.template,
                    self.branching_pattern[istage],
                    self.rng,
                    self.ntrials,
                    self.ducy_max,
                )
                cur_states.append((cur_state, cur_fold_state))
            # Find the state with the minimum complexity
            for iprob in range(self.nprobs):
                filtered_states = [
                    (state, fold_state)
                    for state, fold_state in cur_states
                    if np.digitize(state.success_h1_cumul, self.probs) - 1 == iprob
                ]
                if len(filtered_states) == 0:
                    continue
                best_state_tup = min(
                    filtered_states,
                    key=lambda tup: tup[0].complexity_cumul,
                )
                self._states[istage, ithres, iprob] = best_state_tup[0]
                self._folds_out[ithres, iprob] = best_state_tup[1]

    def backtrack(self) -> StatesInfo:
        final_states = [state for state in self.states[-1].ravel() if state is not None]
        best_state = min(final_states, key=lambda st: st.cost)
        backtrack_states = []
        for istage, (thresh, success_h1_cumul) in enumerate(best_state.backtrack):
            thresh_idx = np.argmin(np.abs(self.thresholds - thresh))
            prob_idx = np.digitize(success_h1_cumul, self.probs) - 1
            bk_state = self.states[istage, thresh_idx, prob_idx]
            backtrack_states.append(bk_state)
        return StatesInfo(backtrack_states)

    def backtrack_many(self, min_probs: list) -> list[StatesInfo]:
        final_states = [state for state in self.states[-1].ravel() if state is not None]
        backtrack_states_info = []
        for min_prob in min_probs:
            final_states_filtered = [
                state for state in final_states if state.success_h1_cumul >= min_prob
            ]
            best_state = min(final_states_filtered, key=lambda st: st.cost)
            backtrack_states = []
            for istage, (thresh, success_h1_cumul) in enumerate(best_state.backtrack):
                thresh_idx = np.argmin(np.abs(self.thresholds - thresh))
                prob_idx = np.digitize(success_h1_cumul, self.probs) - 1
                bk_state = self.states[istage, thresh_idx, prob_idx]
                backtrack_states.append(bk_state)
            backtrack_states_info.append(StatesInfo(backtrack_states))
        return backtrack_states_info

    def plot_slice(
        self,
        stage: int,
        attribute: str = "success_h1",
        fmt: str = ".3f",
    ) -> plt.Figure:
        cum_score = np.empty((self.nthresholds, self.nprobs), dtype=float)
        for ithresh, jprob in itertools.product(
            range(self.nthresholds),
            range(self.nprobs),
        ):
            cum_score[ithresh, jprob] = getattr(
                self.states[stage, ithresh, jprob],
                attribute,
                0,
            )
        df = pd.DataFrame(
            cum_score,
            index=pd.Index(self.thresholds, name="Thresholds", dtype=np.float32),
            columns=pd.Index(range(self.nprobs), name="Success Prob bins"),
        )
        fig, ax = plt.subplots(figsize=(12, 18))
        sns.heatmap(df, annot=True, fmt=fmt, linewidth=0.5, ax=ax)
        ax.invert_yaxis()
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

    def load_from_file(self, filename: str) -> None:
        data = np.load(filename, allow_pickle=True)
        self._thresholds = data["thresholds"]
        self._probs = data["probs"]
        self._states = data["states"]


def determine_threshold_scheme(
    survive_probs: np.ndarray,
    branching_pattern: np.ndarray,
    template: np.ndarray,
    ntrials: int = 1024,
    snr_final: float = 8,
    ducy_max: float = 0.2,
) -> StatesInfo:
    nsegments = len(branching_pattern)
    template = template / np.sqrt(np.sum(template**2))
    bias_snr = snr_final / np.sqrt(nsegments)
    rng = np.random.default_rng()
    states: list[State] = []
    fold_states: list[Folds] = []
    state_init, fold_state_init = init_state_using_surv_prob(
        survive_probs[0],
        template,
        bias_snr,
        branching_pattern[0],
        rng,
        ntrials,
        ducy_max,
        var_init=2,
    )
    states.append(state_init)
    fold_states.append(fold_state_init)
    for istage in track(
        range(1, nsegments),
        description="Running",
        total=nsegments - 1,
    ):
        state = states[istage - 1]
        fold_state = fold_states[istage - 1]
        state_next, fold_state_next = state.gen_next_using_surv_prob(
            fold_state,
            survive_probs[istage],
            bias_snr,
            template,
            branching_pattern[istage],
            rng,
            ntrials,
            ducy_max,
        )
        states.append(state_next)
        fold_states.append(fold_state_next)
    return StatesInfo(states)


def evaluate_threshold_scheme(
    thresholds: np.ndarray,
    branching_pattern: np.ndarray,
    template: np.ndarray,
    ntrials: int = 1024,
    snr_final: float = 8,
    ducy_max: float = 0.2,
) -> StatesInfo:
    nsegments = len(branching_pattern)
    template = template / np.sqrt(np.sum(template**2))
    bias_snr = snr_final / np.sqrt(nsegments)
    rng = np.random.default_rng()
    states: list[State] = []
    fold_states: list[Folds] = []
    state_init_list, fold_state_init_list = init_states_dynamic(
        template,
        thresholds[:1],
        bias_snr,
        branching_pattern[0],
        rng,
        ntrials,
        ducy_max,
        var_init=2,
    )
    states.append(state_init_list[0])
    fold_states.append(fold_state_init_list[0])
    for istage in track(
        range(1, nsegments),
        description="Running",
        total=nsegments - 1,
    ):
        state = states[istage - 1]
        fold_state = fold_states[istage - 1]
        state_next, fold_state_next = state.gen_next_using_thresh(
            fold_state,
            thresholds[istage],
            bias_snr,
            template,
            branching_pattern[istage],
            rng,
            ntrials,
            ducy_max,
        )
        states.append(state_next)
        fold_states.append(fold_state_next)
    return StatesInfo(states)
