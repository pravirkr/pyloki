from __future__ import annotations
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
from numba import njit, types, typed
from numba.experimental import jitclass
from rich.progress import track
from matplotlib import pyplot as plt

from pruning import scores, kernels


@njit
def neighbouring_indices(arr: np.ndarray, target: float, num: int):
    index = np.argwhere(arr == target)[0][0]

    # Calculate the window around the target
    left = max(0, index - num // 2)
    right = min(len(arr), index + num // 2 + 1)

    # Adjust the window if necessary for the end points
    while right - left < num:
        if left == 0:
            right += 1
        elif right == len(arr):
            left -= 1
        else:
            if index - left < right - index:
                left -= 1
            else:
                right += 1

    return np.arange(left, right)


@njit
def init_state(
    bias_snr: float, template: np.ndarray, threshold: float, ntrials=1024, ducy_max=0.2
) -> State:
    nbins = len(template)
    folds_h0 = np.random.normal(0, np.sqrt(2), (ntrials, nbins)).astype(np.float32)
    folds_h1 = np.random.normal(0, np.sqrt(2), (ntrials, nbins)).astype(np.float32)
    folds_h1 += bias_snr * template
    success_h0, folds_h0 = measure_success_prob(folds_h0, threshold, ducy_max=ducy_max)
    success_h1, folds_h1 = measure_success_prob(folds_h1, threshold, ducy_max=ducy_max)
    backtrack = typed.List([(threshold, success_h1)])
    return State(
        folds_h0.astype(np.float32),
        folds_h1.astype(np.float32),
        success_h0,
        success_h1,
        success_h0,
        success_h1,
        success_h0,
        backtrack,
    )


# Perhaps we need a multi-state class that keep tracks of the path per success
# probability so we can guarantee it does not plunge below 5% per trial.
@jitclass(
    [
        ("folds_h0", types.f4[:, :]),
        ("folds_h1", types.f4[:, :]),
        ("success_h0", types.f8),
        ("success_h1", types.f8),
        ("n_h0", types.f8),
        ("success_h1_cumul", types.f8),
        ("complexity_cumul", types.f8),
        ("backtrack", types.ListType(types.Tuple((types.f8, types.f8)))),
    ]
)
class State(object):
    def __init__(
        self,
        folds_h0: np.ndarray,
        folds_h1: np.ndarray,
        success_h0: float = 1,
        success_h1: float = 1,
        n_h0: float = 1,
        success_h1_cumul: float = 1,
        complexity_cumul: float = 1,
        backtrack: typed.List[tuple[float, float]] = None,
    ) -> None:
        self.folds_h0 = folds_h0
        self.folds_h1 = folds_h1
        self.success_h0 = success_h0
        self.success_h1 = success_h1
        self.n_h0 = n_h0
        self.success_h1_cumul = success_h1_cumul
        self.complexity_cumul = complexity_cumul
        self.backtrack = backtrack

    @property
    def cost(self) -> float:
        return self.complexity_cumul / self.success_h1_cumul

    @property
    def is_empty(self) -> bool:
        return len(self.folds_h0) == 0 or len(self.folds_h1) == 0

    def gen_next(
        self,
        threshold: float,
        bias_snr: float,
        segment_cur: int,
        nbranches: int,
        template: np.ndarray,
        ntrials: int = 1024,
        ducy_max: float = 0.2,
    ) -> State:
        folds_h0 = simulate_folds(self.folds_h0, template, bias_snr=0, ntrials=ntrials)
        success_h0, folds_h0 = measure_success_prob(
            folds_h0, threshold, segment_cur=segment_cur, ducy_max=ducy_max
        )
        folds_h1 = simulate_folds(
            self.folds_h1, template, bias_snr=bias_snr, ntrials=ntrials
        )
        success_h1, folds_h1 = measure_success_prob(
            folds_h1, threshold, segment_cur=segment_cur, ducy_max=ducy_max
        )
        complexity_cumul = self.complexity_cumul + self.n_h0 * nbranches * success_h0
        n_h0 = self.n_h0 * success_h0 * nbranches
        success_h1_cumul = self.success_h1_cumul * success_h1
        backtrack = self.backtrack.copy()
        backtrack.append((threshold, success_h1_cumul))
        return State(
            folds_h0,
            folds_h1,
            success_h0,
            success_h1,
            n_h0,
            success_h1_cumul,
            complexity_cumul,
            backtrack,
        )


# make sure the template is normalized correctly to have units that correspond the bias_snr
@njit
def simulate_folds(
    folds: np.ndarray, template: np.ndarray, bias_snr: float = 0, ntrials: int = 1024
) -> np.ndarray:
    """Simulate the folded data for the bias S/N

    Parameters
    ----------
    folds : np.ndarray
        Array of simulated folded data.
    template : np.ndarray
        Template for the signal.
    bias_snr : float, optional
        Bias of the signal, by default 0
    ntrials : int, optional
        Number of trials, by default 1024

    Returns
    -------
    np.ndarray
        Array of simulated folded data.
    """
    nfolds = len(folds)
    folds_sim = np.empty((ntrials, len(template)), dtype=folds.dtype)
    for itrial in range(ntrials):
        folds_sim[itrial] = folds[itrial % nfolds]
    noise = np.random.normal(0, 1, folds_sim.shape) + bias_snr * template
    folds_sim += noise
    return folds_sim


@njit
def measure_success_prob(
    folds: np.ndarray, threshold: float, segment_cur: int = 0, ducy_max: float = 0.2
) -> tuple[float, np.ndarray]:
    """Measure the success probability of signal detection.

    Parameters
    ----------
    folds : np.ndarray
        Array of simulated folded data.
    threshold : float
        Threshold for the detection statistic.
    ducy_max : float, optional
        Maximum duty cycle for boxcar search, by default 0.2
    segment_cur : int
        Index of the current segment.

    Returns
    -------
    tuple[float, np.ndarray]
        Success probability and the folds that passed the threshold.
    """
    nbins = folds.shape[1]
    folds_norm = folds / np.sqrt((segment_cur + 2) * np.ones_like(folds))
    widths = scores.generate_width_trials(nbins, ducy_max=ducy_max, wtsp=1.1)
    scores_arr = kernels.nb_max(scores.boxcar_snr(folds_norm, widths), axis=1)
    good_scores = np.nonzero(scores_arr > threshold)[0]
    succ_prob = len(good_scores) / len(scores_arr)
    return succ_prob, folds[good_scores]


class DynamicThresholding(object):
    def __init__(
        self,
        branching_pattern,
        template,
        ntrials=1024,
        snr_final=8,
        snr_step=0.1,
    ):
        self.ntrials = ntrials
        self.branching_pattern = branching_pattern
        self.snr_final = snr_final
        self.template = template

        # later define as snr^2
        self._thresholds = np.arange(0, snr_final, snr_step) + snr_step
        self._probs = 1 - np.logspace(-3, 0, 10, base=np.e)[::-1]

        self._states = None

    @property
    def nsegments(self):
        return len(self.branching_pattern)

    @property
    def bias_snr(self):
        return self.snr_final / np.sqrt(self.nsegments)

    @property
    def thresholds(self):
        return self._thresholds

    @property
    def nthresholds(self):
        return len(self.thresholds)

    @property
    def probs(self):
        return self._probs

    @property
    def nprobs(self):
        return len(self.probs)

    @property
    def states(self) -> np.ndarray[State]:
        return self._states

    def init(self):
        self._states = np.empty(
            (self.nsegments, self.nthresholds, self.nprobs), dtype=object
        )
        for ithres, threshold in enumerate(self.thresholds):
            state_init = init_state(self.bias_snr, self.template, threshold, self.ntrials)
            prob_idx = np.digitize(state_init.success_h1_cumul, self.probs) - 1
            self._states[0, ithres, prob_idx] = state_init

    def run(self, threshold_tolerance=20):
        for istage, ithres in track(
            itertools.product(range(1, self.nsegments), range(self.nthresholds)),
            description="Running",
            total=(self.nsegments - 1) * self.nthresholds,
        ):
            # Iterate over all prevoius neighbouring thresholds
            neighbour_thresholds_ids = neighbouring_indices(
                self.thresholds, self.thresholds[ithres], threshold_tolerance
            )
            cur_states = []
            for jthres, kprob in itertools.product(
                neighbour_thresholds_ids, range(self.nprobs)
            ):
                prev_state = self.states[istage - 1, jthres, kprob]
                if prev_state is None or prev_state.is_empty:
                    continue
                cur_state = prev_state.gen_next(
                    self.thresholds[ithres],
                    self.bias_snr,
                    istage,
                    self.branching_pattern[istage - 1],
                    self.template,
                    self.ntrials,
                    0.2,
                )
                cur_states.append(cur_state)
            # Find the state with the minimum complexity
            for iprob in range(self.nprobs):
                filtered_states = [
                    state
                    for state in cur_states
                    if np.digitize(state.success_h1_cumul, self.probs) - 1 == iprob
                ]
                if len(filtered_states) == 0:
                    continue
                best_state = min(filtered_states, key=lambda st: st.complexity_cumul)
                self._states[istage, ithres, iprob] = best_state

    def backtrack(self):
        final_states = [state for state in self.states[-1].ravel() if state is not None]
        best_state = min(final_states, key=lambda st: st.cost)
        backtrack_states = []
        for istage, (thresh, success_h1_cumul) in enumerate(best_state.backtrack):
            thresh_idx = np.argmin(np.abs(self.thresholds - thresh))
            prob_idx = np.digitize(success_h1_cumul, self.probs) - 1
            backtrack_states.append(self.states[istage, thresh_idx, prob_idx])
        return backtrack_states, best_state

    def plot_slice(self, stage, attribute="success_h1", fmt=".3f"):
        cum_score = np.empty((self.nthresholds, self.nprobs), dtype=float)
        for ithresh, jprob in itertools.product(
            range(self.nthresholds), range(self.nprobs)
        ):
            cum_score[ithresh, jprob] = getattr(
                self.states[stage, ithresh, jprob], attribute, 0
            )
        df = pd.DataFrame(cum_score)
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.heatmap(df, annot=True, fmt=fmt, linewidth=0.5, ax=ax)
        ax.invert_yaxis()
        return fig
