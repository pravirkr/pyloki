from __future__ import annotations

import itertools
import json
from pathlib import Path

import attrs
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from numba import njit, typed, types
from numba.experimental import jitclass
from rich.progress import track
from scipy import stats

from pyloki.detection import scoring
from pyloki.utils import np_utils
from pyloki.utils.misc import get_logger
from pyloki.utils.plotter import set_seaborn

logger = get_logger(__name__)


def bound_scheme(nstages: int, snr_bound: float) -> np.ndarray:
    """Threshold scheme using the bound on the target S/N.

    Parameters
    ----------
    nstages : int
        Number of stages in the threshold scheme.
    snr_bound : float
        Upper bound on the target S/N.

    Returns
    -------
    np.ndarray
        Thresholds for each stage.
    """
    nsegments = nstages + 1
    thresh_sn2 = np.arange(1, nsegments + 1) * snr_bound**2 / nsegments
    return np.sqrt(thresh_sn2[1:])


def trials_scheme(branching_pattern: np.ndarray, trials_start: int = 1) -> np.ndarray:
    """Threshold scheme using the FAR of the tree.

    Parameters
    ----------
    branching_pattern : np.ndarray
        Branching pattern for each stage.
    trials_start : int
        Starting number of trials at stage 0, by default 1.

    Returns
    -------
    np.ndarray
        Thresholds for each stage.
    """
    trials = np.cumprod(branching_pattern) * trials_start
    return stats.norm.isf(1 / trials)


@njit(cache=True, fastmath=True)
def neighbouring_indices(
    beam_indices: np.ndarray,
    target_idx: int,
    num: int,
) -> np.ndarray:
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
    folds: np.ndarray,
    var_cur: float,
    template: np.ndarray,
    rng: np.random.Generator,
    bias_snr: float = 0,
    var_add: float = 1,
    ntrials_min: int = 1024,
) -> tuple[np.ndarray, float]:
    """Simulate folded profiles by adding signal + noise to the template.

    Parameters
    ----------
    folds : np.ndarray
        2D Array of folded data with shape (ntrials, nbins).
    var_cur : float
        Current variance of the noise in the folded profiles.
    template : np.ndarray
        Normalized template of the signal with the same units as the bias_snr.
    bias_snr : float, optional
        Bias signal-to-noise ratio.
    var_add : float, optional
        Variance of the added noise, by default 1.
    ntrials : int, optional
        Number of trials, by default 1024.

    Returns
    -------
    tuple[np.ndarray, float]
        Array of simulated folded data and the updated variance.
    """
    ntrials_prev, nbins = folds.shape
    repeat_factor = int(np.ceil(ntrials_min / ntrials_prev))
    ntrials = ntrials_prev * repeat_factor
    folds_sim = np.zeros((ntrials, nbins), dtype=folds.dtype)
    for irepeat in range(repeat_factor):
        folds_sim[irepeat * ntrials_prev : (irepeat + 1) * ntrials_prev] = folds
    noise = rng.normal(0, np.sqrt(var_add), (ntrials, nbins)).astype(folds.dtype)
    folds_sim += noise + (bias_snr * template)
    var_sim = var_cur + var_add
    return folds_sim, var_sim


@njit(fastmath=True)
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
        Current variance of the noise in the folded profiles.
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
    widths = scoring.generate_box_width_trials(
        folds.shape[1],
        ducy_max=ducy_max,
        spacing_factor=1,
    )
    scores_arr = np_utils.nb_max(scoring.boxcar_snr(folds_norm, widths), axis=1)
    good_scores_idx = np.nonzero(scores_arr > snr_thresh)[0]
    succ_prob = len(good_scores_idx) / len(scores_arr)
    return succ_prob, folds[good_scores_idx]


@njit(fastmath=True)
def measure_threshold(
    folds: np.ndarray,
    var_cur: float,
    survive_prob: float,
    ducy_max: float = 0.2,
) -> tuple[float, float, np.ndarray]:
    folds_norm = folds / np.sqrt(var_cur * np.ones_like(folds))
    widths = scoring.generate_box_width_trials(
        folds.shape[1],
        ducy_max=ducy_max,
        spacing_factor=1,
    )
    scores_arr = np_utils.nb_max(scoring.boxcar_snr(folds_norm, widths), axis=1)
    n_surviving = int(survive_prob * len(scores_arr))
    good_scores_idx = np.flip(np.argsort(scores_arr))[: int(n_surviving)]
    succ_prob = len(good_scores_idx) / len(scores_arr)
    threshold = scores_arr[good_scores_idx[-1]]
    return threshold, succ_prob, folds[good_scores_idx]


@jitclass(
    spec=[
        ("folds_h0", types.f4[:, :]),
        ("folds_h1", types.f4[:, :]),
        ("variance", types.f4),
    ],
)
class Folds:
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
class SaveState:
    """Class to save the state of the threshold scheme."""

    success_h0: float
    success_h1: float
    complexity: float
    complexity_cumul: float
    success_h1_cumul: float
    nbranches: int
    backtrack: list[tuple[float, float]]

    @property
    def cost(self) -> float:
        return self.complexity_cumul / self.success_h1_cumul


@jitclass(
    spec=[
        ("success_h0", types.f4),
        ("success_h1", types.f4),
        ("complexity", types.f4),
        ("complexity_cumul", types.f4),
        ("success_h1_cumul", types.f4),
        ("nbranches", types.f4),
        ("backtrack", types.ListType(types.Tuple((types.f4, types.f4)))),
    ],
)
class State:
    """State class to keep track of the current state of the threshold scheme.

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
    backtrack : typed.List[tuple[float, float]], optional
        List of the thresholds and success probabilities for the backtrack,
        by default None.
    """

    def __init__(
        self,
        success_h0: float = 1,
        success_h1: float = 1,
        complexity: float = 1,
        complexity_cumul: float = 1,
        success_h1_cumul: float = 1,
        nbranches: float = 1,
        backtrack: typed.List[tuple[float, float]] = typed.List.empty_list(  # noqa: B008
            types.Tuple((types.f4, types.f4)),  # noqa: B008
        ),
    ) -> None:
        self.success_h0 = success_h0
        self.success_h1 = success_h1
        self.complexity = complexity
        self.complexity_cumul = complexity_cumul
        self.success_h1_cumul = success_h1_cumul
        self.nbranches = nbranches
        self.backtrack = backtrack

    @property
    def cost(self) -> float:
        return self.complexity_cumul / self.success_h1_cumul

    def get_next_state(
        self,
        threshold: float,
        success_h0: float,
        success_h1: float,
        nbranches: int,
    ) -> State:
        complexity_cumul = self.complexity_cumul + (self.complexity * nbranches)
        complexity = self.complexity * nbranches * success_h0
        success_h1_cumul = self.success_h1_cumul * success_h1
        backtrack = self.backtrack.copy()
        backtrack.append((threshold, success_h1_cumul))
        return State(
            success_h0,
            success_h1,
            complexity,
            complexity_cumul,
            success_h1_cumul,
            nbranches,
            backtrack,
        )

    def gen_next_using_surv_prob(
        self,
        fold_state: Folds,
        surv_prob_h0: float,
        nbranches: int,
        bias_snr: float,
        template: np.ndarray,
        rng: np.random.Generator,
        ntrials: int = 1024,
        ducy_max: float = 0.2,
    ) -> tuple[State, Folds]:
        folds_h0, variance = simulate_folds(
            fold_state.folds_h0,
            fold_state.variance,
            template,
            rng,
            bias_snr=0,
            var_add=1,
            ntrials_min=ntrials,
        )
        thresold_h0, success_h0, folds_h0 = measure_threshold(
            folds_h0,
            variance,
            surv_prob_h0,
            ducy_max=ducy_max,
        )
        folds_h1, variance = simulate_folds(
            fold_state.folds_h1,
            fold_state.variance,
            template,
            rng,
            bias_snr=bias_snr,
            var_add=1,
            ntrials_min=ntrials,
        )
        success_h1, folds_h1 = measure_success(
            folds_h1,
            variance,
            thresold_h0,
            ducy_max=ducy_max,
        )
        next_state = self.get_next_state(thresold_h0, success_h0, success_h1, nbranches)
        next_fold_state = Folds(folds_h0, folds_h1, variance)
        return next_state, next_fold_state

    def gen_next_using_thresh(
        self,
        fold_state: Folds,
        threshold: float,
        nbranches: int,
        bias_snr: float,
        template: np.ndarray,
        rng: np.random.Generator,
        ntrials: int = 1024,
        ducy_max: float = 0.2,
    ) -> tuple[State, Folds]:
        folds_h0, variance = simulate_folds(
            fold_state.folds_h0,
            fold_state.variance,
            template,
            rng,
            bias_snr=0,
            var_add=1,
            ntrials_min=ntrials,
        )
        success_h0, folds_h0 = measure_success(
            folds_h0,
            variance,
            threshold,
            ducy_max=ducy_max,
        )
        folds_h1, variance = simulate_folds(
            fold_state.folds_h1,
            fold_state.variance,
            template,
            rng,
            bias_snr=bias_snr,
            var_add=1,
            ntrials_min=ntrials,
        )
        success_h1, folds_h1 = measure_success(
            folds_h1,
            variance,
            threshold,
            ducy_max=ducy_max,
        )
        next_state = self.get_next_state(threshold, success_h0, success_h1, nbranches)
        next_fold_state = Folds(folds_h0, folds_h1, variance)
        return next_state, next_fold_state


@attrs.define(frozen=True)
class StatesInfo:
    """Class to handle the information of the states in the threshold scheme."""

    entries: list[State] = attrs.Factory(list)

    @property
    def thresholds(self) -> np.ndarray:
        """Get list of thresholds for this scheme."""
        thresh, _ = zip(*self.entries[-1].backtrack)
        return np.array(thresh)

    def get_info(self, key: str) -> list:
        """Get list of values for a given key for all entries."""
        return [getattr(entry, key) for entry in self.entries]

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
            entries = [State(**entry) for entry in data["entries"]]
            return cls(entries=entries)


class DynamicThresholdScheme:
    def __init__(
        self,
        branching_pattern: np.ndarray,
        template: np.ndarray,
        ntrials: int = 1024,
        nprobs: int = 10,
        snr_final: float = 8,
        nthresholds: int = 100,
        ducy_max: float = 0.2,
        beam_width: float = 0.7,
    ) -> None:
        self.rng = np.random.default_rng()
        self.branching_pattern = branching_pattern
        self.template = template / np.sqrt(np.sum(template**2))
        self.ntrials = ntrials
        self.snr_final = snr_final
        self.ducy_max = ducy_max
        self.beam_width = beam_width

        # later define as snr^2
        self.thresholds = np.linspace(0.1, self.snr_final, nthresholds)
        self.probs = 1 - np.logspace(-3, 0, nprobs, base=np.e)[::-1]
        self.states = np.empty(
            (self.nstages, self.nthresholds, self.nprobs),
            dtype=object,
        )
        self.folds_in = np.empty((self.nthresholds, self.nprobs), dtype=object)
        self.folds_out = np.empty((self.nthresholds, self.nprobs), dtype=object)
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
        folds = np.zeros((self.ntrials, len(self.template)), dtype=np.float32)
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
        state = State()
        fold_state = Folds(folds_h0, folds_h1, var_init)
        thresholds_idx = self.get_current_thresholds_idx(0)
        for ithres in thresholds_idx:
            cur_state, cur_fold_state = state.gen_next_using_thresh(
                fold_state,
                self.thresholds[ithres],
                self.branching_pattern[0],
                self.bias_snr,
                self.template,
                self.rng,
                self.ntrials,
                self.ducy_max,
            )
            iprob = np.digitize(cur_state.success_h1_cumul, self.probs) - 1
            self.states[0, ithres, iprob] = cur_state
            self.folds_in[ithres, iprob] = cur_fold_state

    def run(self, thres_neigh: int = 21) -> None:
        for istage in track(range(1, self.nstages), description="Running stages"):
            self.run_stage(istage, thres_neigh)
            self.folds_in = self.folds_out
            self.folds_out = np.empty((self.nthresholds, self.nprobs), dtype=object)

    def run_stage(self, istage: int, thres_neigh: int = 11) -> None:
        beam_idx_cur = self.get_current_thresholds_idx(istage)
        beam_idx_prev = self.get_current_thresholds_idx(istage - 1)
        for ibeam_cur in range(len(beam_idx_cur)):
            ithres = beam_idx_cur[ibeam_cur]
            # Find nearest neighbors in the previous beam
            neighbour_beam_indices = neighbouring_indices(
                beam_idx_prev,
                ithres,
                thres_neigh,
            )
            candidates = []
            for jthresh, kprob in itertools.product(
                neighbour_beam_indices,
                range(self.nprobs),
            ):
                prev_state = self.states[istage - 1, jthresh, kprob]
                prev_fold_state = self.folds_in[jthresh, kprob]
                if prev_fold_state is not None and not prev_fold_state.is_empty:
                    cur_state, cur_fold_state = prev_state.gen_next_using_thresh(
                        prev_fold_state,
                        self.thresholds[ithres],
                        self.branching_pattern[istage],
                        self.bias_snr,
                        self.template,
                        self.rng,
                        self.ntrials,
                        self.ducy_max,
                    )
                    candidates.append((cur_state, cur_fold_state))
            # Find the state with the minimum complexity
            for candidate in candidates:
                state, fold_state = candidate
                iprob = np.digitize(state.success_h1_cumul, self.probs) - 1
                existing_state = self.states[istage, ithres, iprob]
                if (
                    existing_state is None
                    or state.complexity_cumul < existing_state.complexity_cumul
                ):
                    self.states[istage, ithres, iprob] = state
                    self.folds_out[ithres, iprob] = fold_state

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
            for attr in ["ntrials", "snr_final", "ducy_max", "beam_width"]:
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

            # Prepare state data
            state_data = []
            backtrack_data = []
            backtrack_lengths = []
            grid_indices = []
            for istage in range(self.nstages):
                for ithres in range(self.nthresholds):
                    for iprob in range(self.nprobs):
                        jit_state = self.states[istage, ithres, iprob]
                        if jit_state is not None:
                            state_dict = {
                                "success_h0": jit_state.success_h0,
                                "success_h1": jit_state.success_h1,
                                "complexity": jit_state.complexity,
                                "complexity_cumul": jit_state.complexity_cumul,
                                "success_h1_cumul": jit_state.success_h1_cumul,
                                "nbranches": jit_state.nbranches,
                            }
                            backtrack = list(jit_state.backtrack)
                            backtrack_data.extend(backtrack)
                            backtrack_lengths.append(len(backtrack))
                            state_data.append(state_dict)
                            grid_indices.append((istage, ithres, iprob))
            # Create a structured numpy array
            dtype = [(key, "f8") for key in state_data[0]]
            state_array = np.array([tuple(d.values()) for d in state_data], dtype=dtype)
            f.create_dataset(
                "states",
                data=state_array,
                compression="gzip",
                compression_opts=9,
            )
            f.create_dataset(
                "backtrack",
                data=np.array(backtrack_data, dtype="f8"),
                compression="gzip",
                compression_opts=9,
            )
            f.create_dataset(
                "backtrack_lengths",
                data=np.array(backtrack_lengths, dtype="i4"),
                compression="gzip",
                compression_opts=9,
            )
            f.create_dataset(
                "grid_indices",
                data=np.array(grid_indices, dtype="i4"),
                compression="gzip",
                compression_opts=9,
            )
        return filename.as_posix()

    def save1(self, outdir: str = ".") -> str:
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
            for attr in ["ntrials", "snr_final", "ducy_max", "beam_width"]:
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

            # create groups for the states
            states_group = f.create_group("states")
            for istage in range(self.nstages):
                stage_group = states_group.create_group(f"stage_{istage}")
                for ithres in range(self.nthresholds):
                    threshold_group = stage_group.create_group(f"threshold_{ithres}")
                    for iprob in range(self.nprobs):
                        jit_state = self.states[istage, ithres, iprob]
                        if jit_state is not None:
                            state = SaveState(
                                jit_state.success_h0,
                                jit_state.success_h1,
                                jit_state.complexity,
                                jit_state.complexity_cumul,
                                jit_state.success_h1_cumul,
                                jit_state.nbranches,
                                list(jit_state.backtrack),
                            )
                            state_group = threshold_group.create_group(f"prob_{iprob}")
                            for key in attrs.asdict(state):
                                state_group.attrs[key] = getattr(state, key)
        return filename.as_posix()


class DynamicThresholdSchemeAnalyser:
    def __init__(
        self,
        states: np.ndarray,
        thresholds: np.ndarray,
        probs: np.ndarray,
        branching_pattern: np.ndarray,
        guess_path: np.ndarray,
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
        final_states = [state for state in self.states[-1].ravel() if state is not None]
        backtrack_states_info = []
        for min_prob in min_probs:
            final_states_filtered = [
                state for state in final_states if state.success_h1_cumul >= min_prob
            ]
            best_state = min(final_states_filtered, key=lambda st: st.cost)
            backtrack_states = []
            for istage, (thresh, success_h1_cumul) in enumerate(best_state.backtrack):
                ithresh = np.argmin(np.abs(self.thresholds - thresh))
                iprob = np.digitize(success_h1_cumul, self.probs) - 1
                bk_state = self.states[istage, ithresh, iprob]
                backtrack_states.append(bk_state)
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
        final_states = [
            state
            for state in self.states[-1].ravel()
            if state is not None and state.success_h1_cumul >= min_prob
        ]
        backtrack_states_info = []
        for final_state in final_states:
            backtrack_states = []
            for istage, (thresh, success_h1_cumul) in enumerate(final_state.backtrack):
                ithresh = np.argmin(np.abs(self.thresholds - thresh))
                iprob = np.digitize(success_h1_cumul, self.probs) - 1
                bk_state = self.states[istage, ithresh, iprob]
                backtrack_states.append(bk_state)
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

    @classmethod
    def from_file(cls, filename: str) -> DynamicThresholdSchemeAnalyser:
        """Load a DynamicThresholdScheme object from an hdf5 file."""
        with h5py.File(filename, "r") as f:
            branching_pattern = f["branching_pattern"][:]
            thresholds = f["thresholds"][:]
            probs = f["probs"][:]
            guess_path = f["guess_path"][:]
            beam_width = f.attrs["beam_width"]
            nstages = len(branching_pattern)
            nthresholds = len(thresholds)
            nprobs = len(probs)
            # Load datasets
            states = f["states"][:]
            backtrack_data = f["backtrack"][:]
            backtrack_lengths = f["backtrack_lengths"][:]
            grid_indices = f["grid_indices"][:]
            re_states = np.empty((nstages, nthresholds, nprobs), dtype=object)

            start = 0
            for i, (istage, ithres, iprob) in enumerate(grid_indices):
                state = states[i]
                length = backtrack_lengths[i]
                end = start + length
                backtrack = backtrack_data[start:end]
                re_states[istage, ithres, iprob] = SaveState(
                    state["success_h0"],
                    state["success_h1"],
                    state["complexity"],
                    state["complexity_cumul"],
                    state["success_h1_cumul"],
                    state["nbranches"],
                    list(backtrack),
                )
                start = end
        return cls(
            re_states,
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
) -> StatesInfo:
    if len(survive_probs) != len(branching_pattern):
        msg = "Number of survive_probs must match the number of stages"
        raise ValueError(msg)

    var_init = 1.0
    nstages = len(branching_pattern)
    template = profile / np.sqrt(np.sum(profile**2))
    bias_snr = snr_final / np.sqrt(nstages + 1)
    rng = np.random.default_rng()
    states: list[State] = []
    fold_states: list[Folds] = []
    folds = np.zeros((ntrials, len(template)), dtype=np.float32)
    folds_h0, _ = simulate_folds(folds, 0, template, rng, 0, var_init, ntrials)
    folds_h1, _ = simulate_folds(folds, 0, template, rng, bias_snr, var_init, ntrials)
    initial_state = State()
    initial_fold_state = Folds(folds_h0, folds_h1, var_init)
    for istage in range(nstages):
        prev_state = initial_state if istage == 0 else states[istage - 1]
        prev_fold_state = initial_fold_state if istage == 0 else fold_states[istage - 1]
        if istage > 0 and prev_fold_state.is_empty:
            logger.info("Path not viable, No trials survived, stopping")
            break
        cur_state, cur_fold_state = prev_state.gen_next_using_surv_prob(
            prev_fold_state,
            survive_probs[istage],
            branching_pattern[istage],
            bias_snr,
            template,
            rng,
            ntrials,
            ducy_max,
        )
        states.append(cur_state)
        fold_states.append(cur_fold_state)
    return StatesInfo(states)


def evaluate_scheme(
    thresholds: np.ndarray,
    branching_pattern: np.ndarray,
    profile: np.ndarray,
    ntrials: int = 2048,
    snr_final: float = 8,
    ducy_max: float = 0.2,
) -> StatesInfo:
    var_init = 1.0
    nstages = len(branching_pattern)
    template = profile / np.sqrt(np.sum(profile**2))
    bias_snr = snr_final / np.sqrt(nstages + 1)
    rng = np.random.default_rng()
    if len(thresholds) != nstages:
        msg = "Number of thresholds must match the number of stages"
        raise ValueError(msg)
    states: list[State] = []
    fold_states: list[Folds] = []
    folds = np.zeros((ntrials, len(template)), dtype=np.float32)
    folds_h0, _ = simulate_folds(folds, 0, template, rng, 0, var_init, ntrials)
    folds_h1, _ = simulate_folds(folds, 0, template, rng, bias_snr, var_init, ntrials)
    initial_state = State()
    initial_fold_state = Folds(folds_h0, folds_h1, var_init)
    for istage in range(nstages):
        prev_state = initial_state if istage == 0 else states[istage - 1]
        prev_fold_state = initial_fold_state if istage == 0 else fold_states[istage - 1]
        if istage > 0 and prev_fold_state.is_empty:
            logger.info("Path not viable, No trials survived, stopping")
            break
        cur_state, cur_fold_state = prev_state.gen_next_using_thresh(
            prev_fold_state,
            thresholds[istage],
            branching_pattern[istage],
            bias_snr,
            template,
            rng,
            ntrials,
            ducy_max,
        )
        states.append(cur_state)
        fold_states.append(cur_fold_state)
    return StatesInfo(states)
