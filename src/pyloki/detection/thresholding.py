# ruff: noqa: ARG001

from __future__ import annotations

from pathlib import Path
from typing import Self

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
    profile: npt.NDArray[np.float32],
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
                    cur_state = cur_state[0]  # Get record from array
                    iprob_val = np.digitize(cur_state["success_h1_cumul"], probs) - 1
                    iprob = int(iprob_val.item())  # Extract scalar
                    if iprob < 0 or iprob >= nprobs:  # Clamp to valid range
                        continue
                    existing_state = states[istage, ithres, iprob]
                    if (
                        existing_state["is_empty"]
                        or cur_state["complexity_cumul"]
                        < existing_state["complexity_cumul"]
                    ):
                        states[istage, ithres, iprob] = cur_state
                        folds_out[ithres * nprobs + iprob] = cur_fold_state


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
    ) -> None:
        self.rng = np.random.default_rng()
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
        return len(self.profile)

    def get_current_thresholds_idx(self, istage: int) -> np.ndarray:
        guess = self.guess_path[istage]
        half_extent = self.beam_width
        lower_bound = max(0, guess - half_extent)
        upper_bound = min(self.snr_final, guess + half_extent)
        return np.where(
            (self.thresholds >= lower_bound) & (self.thresholds <= upper_bound),
        )[0]

    @Timer(name="DynamicThresholdScheme init", logger=logger.info)
    def init(self) -> None:
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
            iprob = np.digitize(cur_state["success_h1_cumul"], self.probs) - 1
            if iprob < 0:
                continue
            self.states[0, ithres, iprob] = cur_state
            folds_idx = int(ithres * self.nprobs + iprob)
            self.folds_in[folds_idx] = cur_fold_state

    @Timer(name="DynamicThresholdScheme run", logger=logger.info)
    def run(self, thres_neigh: int = 11) -> None:
        logger.info("Running dynamic threshold scheme")
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
                self.profile,
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


def determine_scheme(
    survive_probs: np.ndarray,
    branching_pattern: np.ndarray,
    ref_ducy: float,
    nbins: int = 64,
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
    profile = generate_folded_profile(nbins=nbins, ducy=ref_ducy)
    bias_snr = snr_final / np.sqrt(nstages + 1)
    rng = np.random.default_rng()
    states: list[np.recarray] = []
    fold_states: list[Folds] = []
    folds = np.zeros((ntrials, len(profile)), dtype=np.float32)
    folds_h0, _ = simulate_folds(folds, 0, profile, rng, 0, var_init, ntrials)
    folds_h1, _ = simulate_folds(folds, 0, profile, rng, bias_snr, var_init, ntrials)
    initial_state = np.ones(1, dtype=state_dtype)[0]
    initial_state["threshold"] = -1
    initial_state["threshold_prev"] = -1
    initial_fold_state = Folds(folds_h0, folds_h1, var_init)
    for istage in range(nstages):
        prev_state = initial_state if istage == 0 else states[istage - 1]
        prev_fold_state = initial_fold_state if istage == 0 else fold_states[istage - 1]
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
        fold_states.append(cur_fold_state)
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
) -> StatesInfo:
    var_init = 1.0
    nstages = len(branching_pattern)
    profile = generate_folded_profile(nbins=nbins, ducy=ref_ducy)
    bias_snr = snr_final / np.sqrt(nstages + 1)
    rng = np.random.default_rng()
    if len(thresholds) != nstages:
        msg = "Number of thresholds must match the number of stages"
        raise ValueError(msg)
    states: list[np.recarray] = []
    fold_states: list[Folds] = []
    folds = np.zeros((ntrials, len(profile)), dtype=np.float32)
    folds_h0, _ = simulate_folds(folds, 0, profile, rng, 0, var_init, ntrials)
    folds_h1, _ = simulate_folds(folds, 0, profile, rng, bias_snr, var_init, ntrials)
    initial_state = np.ones(1, dtype=state_dtype)[0]
    initial_state["threshold"] = -1
    initial_state["threshold_prev"] = -1
    initial_fold_state = Folds(folds_h0, folds_h1, var_init)
    for istage in range(nstages):
        prev_state = initial_state if istage == 0 else states[istage - 1]
        prev_fold_state = initial_fold_state if istage == 0 else fold_states[istage - 1]
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
        fold_states.append(cur_fold_state)
    return StatesInfo([StateInfo.from_record(state) for state in states])
