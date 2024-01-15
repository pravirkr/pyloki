from __future__ import annotations
import numpy as np
from scipy import stats

from dataclasses import dataclass
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from pruning import kernels, scores


def threshold_scheme_bound(nsegments: int, bound: float, margin: float = 0) -> np.ndarray:
    """Returns the threshold scheme using the bound on the target S/N.

    Parameters
    ----------
    nsegments : int
        Number of data segments
    bound : float
        Upper bound on the target S/N
    margin : float, optional
        S/N margin to be subtracted from the threshold, by default 0

    Returns
    -------
    np.ndarray
        Array of thresholds, one for each iteration
    """
    return bound / np.sqrt(nsegments) * np.sqrt(np.arange(nsegments) + 1) - margin


def threshold_scheme_trials(
    nsegments: int, nparams: int, sugg_size: int = 1
) -> np.ndarray:
    """Returns the threshold scheme using the number of trials.

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
        Array of thresholds, one for each iteration
    """
    complexity_scaling = np.sum(np.arange(nparams) + 1)
    trials = sugg_size * (np.arange(nsegments) + 1) ** complexity_scaling
    return stats.norm.isf(1 / trials)


@dataclass
class StartState:
    folds_h0: np.ndarray
    folds_h1: np.ndarray
    success_h0: float = 0
    success_h1: float = 2
    threshold: float = 0

    def gen_next(
        self,
        survive_prob: float,
        bias_snr: float,
        segment_cur,
        template,
        ntrials=1024,
    ):
        folds_h0 = simulate_folds(self.folds_h0, template, 0, ntrials)
        thresold_h0, success_h0, folds_h0 = measure_threshold(
            folds_h0, survive_prob, segment_cur
        )
        folds_h1 = simulate_folds(self.folds_h1, template, bias_snr, ntrials)
        success_h1, folds_h1 = measure_success_prob(folds_h1, thresold_h0, segment_cur)
        return StartState(folds_h0, folds_h1, success_h0, success_h1, thresold_h0)

    def plot_hist(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.hist(self.folds_h0.flatten(), bins=100, alpha=0.5, label="H0")
        ax.hist(self.folds_h1.flatten(), bins=100, alpha=0.5, label="H1")
        ax.axvline(self.threshold, color="r", linestyle="--")
        ax.legend()
        ax.set_xlabel("Folded S/N")
        ax.set_ylabel("Counts")

    @classmethod
    def init(cls, bias_snr, template, survive_prob, ntrials=1024):
        folds_h0 = np.random.normal(0, np.sqrt(2), [ntrials, len(template)])
        folds_h1 = np.random.normal(
            bias_snr * 2 * template, np.sqrt(2), [4 * ntrials, len(template)]
        )
        thresold_h0, success_h0, folds_h0 = measure_threshold(folds_h0, survive_prob)
        success_h1, folds_h1 = measure_success_prob(folds_h1, thresold_h0)
        return cls(folds_h0, folds_h1, success_h0, success_h1, thresold_h0)


def simulate_folds(
    folds: np.ndarray, template: np.ndarray, bias_snr: float = 0, ntrials: int = 1024
) -> np.ndarray:
    nfolds = len(folds)
    folds_sim = np.empty((ntrials, len(template)), dtype=folds.dtype)
    for itrial in range(ntrials):
        folds_sim[itrial] = folds[itrial % nfolds]
    noise = np.random.normal(0, 1, folds_sim.shape) + bias_snr * template
    folds_sim += noise
    return folds_sim


def measure_success_prob(
    folds: np.ndarray, threshold: float, segment_cur: int = 0, ducy_max: float = 0.2
) -> tuple[float, np.ndarray]:
    nbins = folds.shape[1]
    folds_norm = folds / np.sqrt((segment_cur + 2) * np.ones_like(folds))
    widths = scores.generate_width_trials(nbins, ducy_max=ducy_max, wtsp=1.1)
    scores_arr = kernels.nb_max(scores.boxcar_snr(folds_norm, widths), axis=1)
    good_scores = np.nonzero(scores_arr > threshold)[0]
    succ_prob = len(good_scores) / len(scores_arr)
    return succ_prob, folds[good_scores]


def measure_threshold(
    folds: np.ndarray, survive_prob: float, segment_cur: int = 0
) -> tuple[float, np.ndarray]:
    nbins = folds.shape[1]
    folds_norm = folds / np.sqrt((segment_cur + 2) * np.ones_like(folds))
    widths = scores.generate_width_trials(nbins, ducy_max=0.2, wtsp=1.1)
    scores_arr = kernels.nb_max(scores.boxcar_snr(folds_norm, widths), axis=1)
    n_surviving = int(survive_prob * len(scores_arr))
    good_scores = np.flip(np.argsort(scores_arr))[: int(n_surviving)]
    succ_prob = len(good_scores) / len(scores_arr)
    threshold = scores_arr[good_scores[-1]]
    return threshold, succ_prob, folds[good_scores]


def determine_threshold_scheme_start_new(ntrials, survive_prob_arr, bias_snr, template):
    template = template / np.sqrt(np.sum(template**2))
    thresholds = []
    survive_prob_good = []
    survive_prob_bad = []
    fig, ax = plt.subplots()

    state = None
    # Function to initialize the plot
    def init():
        # Set up your plot initialization here if needed
        return (ax,)

    # Update function for the animation
    def update(iprob):
        nonlocal state
        if iprob == 0 or state is None:
            state = StartState.init(bias_snr, template, survive_prob_arr[0], ntrials)
        else:
            state = state.gen_next(
                survive_prob_arr[iprob], bias_snr, iprob, template, ntrials
            )
        thresholds.append(state.threshold)
        survive_prob_good.append(state.success_h1)
        survive_prob_bad.append(state.success_h0)
        ax.clear()
        state.plot_hist(ax=ax)
        return (ax,)

    ani = FuncAnimation(
        fig, update, frames=len(survive_prob_arr), init_func=init, blit=False
    )

    # Save the animation
    ani.save("threshold_animation.mp4", fps=2)  # Adjust fps as needed

    return thresholds, survive_prob_good, survive_prob_bad
