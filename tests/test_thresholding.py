"""Regression coverage for the dynamic threshold scheme.

`DynamicThresholdScheme.run()` used to complete without error while writing
**zero-filled** state records from stage 1 onward. Because `is_empty` is part of the
record, zeroing it set the flag to `False`, so the states looked populated while every
number in them -- threshold, complexity, survival probability -- was 0. The scheme then
discarded them all at the next stage (`np.digitize(0.0, probs) - 1 == -1`), leaving
nothing to backtrack and making the whole optimiser silently useless.

The cause is a numba miscompilation of whole-record assignment when the record comes
from an array returned by another jitted function; `TestRecordAssignment` below pins
the construct itself, and the scheme tests pin the consequence.
"""

from __future__ import annotations

import numpy as np
import pytest
from numba import njit
from scipy import stats

from pyloki.detection import schemes, thresholding
from pyloki.detection.thresholding import (
    DynamicThresholdScheme,
    schedule_survive_probs,
)

RECORD_DTYPE = np.dtype([("a", np.float32), ("b", np.float32), ("flag", np.bool_)])


@njit(cache=False)
def _make_record(value: float) -> np.ndarray:
    out = np.zeros(1, dtype=RECORD_DTYPE)
    out[0]["a"] = value
    out[0]["b"] = 2.0 * value
    out[0]["flag"] = True
    return out


@njit(cache=False)
def _store_via_slice(dst: np.ndarray, idx: int, value: float) -> None:
    dst[idx : idx + 1] = _make_record(value)


class TestRecordAssignment:
    """The numba construct underneath the bug.

    Whole-record assignment from a callee-returned array (`dst[i] = make()[0]`) silently
    writes zeros, and so does copying field by field for the boolean. Assigning the
    one-element array to a slice is the only form that round-trips, which is why the
    scheme code is written that way.
    """

    def test_slice_assignment_round_trips(self) -> None:
        dst = np.zeros(3, dtype=RECORD_DTYPE)
        _store_via_slice(dst, 1, 3.0)
        assert float(dst[1]["a"]) == pytest.approx(3.0)
        assert float(dst[1]["b"]) == pytest.approx(6.0)
        assert bool(dst[1]["flag"]) is True
        # neighbours untouched
        assert float(dst[0]["a"]) == 0.0
        assert float(dst[2]["a"]) == 0.0


SEED = 42


def _small_scheme(mode: str, seed: int = SEED) -> DynamicThresholdScheme:
    """Small enough for CI, large enough to reach the stage the bug appeared at."""
    return DynamicThresholdScheme(
        np.array([4.0, 2.0, 2.0, 2.0, 2.0]),
        ref_ducy=0.1,
        nbins=32,
        ntrials=256,
        nprobs=8,
        prob_min=0.05,
        snr_final=8.0,
        nthresholds=30,
        ducy_max=0.3,
        wtsp=1.5,
        beam_width=2.5,
        mode=mode,
        seed=seed,
    )


@pytest.fixture(scope="class", params=["legacy", "improved"])
def ran_scheme(request: pytest.FixtureRequest) -> DynamicThresholdScheme:
    """Run the seeded scheme once per mode; the tests below only read its states."""
    scheme = _small_scheme(request.param)
    scheme.run(thres_neigh=5)
    return scheme


class TestDynamicThresholdScheme:
    def test_states_survive_past_the_first_stage(
        self, ran_scheme: DynamicThresholdScheme
    ) -> None:
        """The headline symptom: everything vanished from stage 2 onward."""
        scheme = ran_scheme
        populated = [
            int((~scheme.states[i]["is_empty"]).sum()) for i in range(scheme.nstages)
        ]
        assert all(n > 0 for n in populated), (
            f"states emptied partway through: non-empty per stage = {populated}"
        )

    def test_stored_states_are_not_zero_filled(
        self, ran_scheme: DynamicThresholdScheme
    ) -> None:
        """The cause, distinct from the symptom.

        A zeroed record reads as `is_empty == False`, so counting non-empty states is
        not enough -- the numbers inside have to be real. Every stored state must carry
        the threshold it was built with, and no threshold in the grid is 0.
        """
        scheme = ran_scheme
        for istage in range(scheme.nstages):
            stage = scheme.states[istage]
            stored = stage[~stage["is_empty"]]
            assert len(stored) > 0
            assert np.all(stored["threshold"] > 0.0), (
                f"stage {istage}: zero-filled records "
                f"({int(np.sum(stored['threshold'] == 0.0))} of {len(stored)})"
            )
            assert np.all(stored["success_h1_cumul"] > 0.0)
            # `complexity` (= nleaves_next * success_h0) can legitimately be 0.0 if
            # 0 of ntrials H0 draws pass a high threshold; `complexity_cumul` cannot.
            assert np.all(stored["complexity_cumul"] > 0.0)
            assert np.all(stored["nbranches"] > 0.0)

    def test_survival_decreases_monotonically(
        self, ran_scheme: DynamicThresholdScheme
    ) -> None:
        """Cumulative survival is a product of probabilities, so it cannot grow."""
        scheme = ran_scheme
        best = []
        for istage in range(scheme.nstages):
            stage = scheme.states[istage]
            best.append(float(stage[~stage["is_empty"]]["success_h1_cumul"].max()))
        assert all(b <= 1.0 + 1e-6 for b in best)
        assert best == sorted(best, reverse=True), f"survival is not monotone: {best}"


class TestTrialsScheme:
    """`trials_scheme` used to return -inf before the search had branched.

    `norm.isf(1 / trials)` is `-inf` when the cumulative trial count is 1, which happens
    whenever a branching pattern starts with `B(s) = 1` -- a perfectly ordinary pattern
    for a strategy that cannot resolve anything at the earliest stages.
    `DynamicThresholdScheme` centres its threshold beam on this path, so an infinite
    guess empties the beam and the optimiser returns nothing at all.
    """

    def test_is_finite_when_the_search_has_not_branched(self) -> None:
        pattern = np.array([1.0, 1.0, 1.0, 8.0, 2.0])
        path = schemes.trials_scheme(pattern, trials_start=1)
        assert np.all(np.isfinite(path)), f"non-finite guess path: {path}"
        assert np.all(path >= 0.0)

    def test_unbranched_stages_need_no_threshold(self) -> None:
        path = schemes.trials_scheme(np.array([1.0, 1.0, 4.0]), trials_start=1)
        assert path[0] == pytest.approx(0.0)
        assert path[1] == pytest.approx(0.0)
        assert path[2] > 0.0

    def test_branching_patterns_are_unchanged(self) -> None:
        """The floor must not perturb a pattern that already branches."""
        pattern = np.array([4.0, 2.0, 3.0])
        expected = stats.norm.isf(1.0 / np.cumprod(pattern))
        assert np.all(expected > 0.0), "test is vacuous unless all entries are positive"
        np.testing.assert_allclose(
            schemes.trials_scheme(pattern, trials_start=1),
            expected,
        )

    def test_scheme_runs_on_a_pattern_that_starts_unbranched(self) -> None:
        """End to end: the configuration that used to produce an empty beam."""
        scheme = DynamicThresholdScheme(
            np.array([1.0, 1.0, 4.0, 2.0, 2.0]),
            ref_ducy=0.1,
            nbins=32,
            ntrials=256,
            nprobs=8,
            prob_min=0.05,
            snr_final=8.0,
            nthresholds=30,
            ducy_max=0.3,
            wtsp=1.5,
            beam_width=2.5,
            mode="improved",
            seed=SEED,
        )
        assert np.all(np.isfinite(scheme.guess_path))
        scheme.run(thres_neigh=5)
        populated = [
            int((~scheme.states[i]["is_empty"]).sum()) for i in range(scheme.nstages)
        ]
        assert all(n > 0 for n in populated), f"non-empty per stage = {populated}"


# A pruning pattern shaped like generate_branching_pattern: x4, x9 early, x3 later.
BP = np.array([4, 9, 1, 1, 3, 3, 1, 3, 1, 1, 3, 1, 3, 1, 1], dtype=float)
# A pattern whose branching stages after the fourth are far apart.
BP_SPARSE = np.array(
    [4, 3, 4, 1, 1, 9, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1], dtype=float
)


class TestScheduleSurviveProbs:
    def test_protect_zero_is_the_plain_vector(self) -> None:
        probs, stages, factor = schedule_survive_probs(BP, protect=0)
        np.testing.assert_allclose(probs[BP > 1], 1.0 / BP[BP > 1])
        assert np.all(probs[BP == 1] == 1.0)
        assert stages == []
        assert factor == 1.0
        probs_nb, _, _ = schedule_survive_probs(BP, protect=0, survive_nonbranch=0.95)
        assert np.all(probs_nb[BP == 1] == 0.95)

    def test_schedule_arithmetic_and_return_to_size(self) -> None:
        probs, stages, factor = schedule_survive_probs(BP)
        assert stages == [5, 6, 8]
        assert abs(factor - 36.0 ** (1.0 / 3.0)) < 1e-12
        assert np.all(probs[:4] == 1.0)
        for st in stages:
            assert abs(probs[st - 1] - 1.0 / (BP[st - 1] * factor)) < 1e-12
        np.testing.assert_allclose(probs[8:][BP[8:] > 1], 1.0 / BP[8:][BP[8:] > 1])
        # Noise tree vs the plain ladder: 36x through the protected stages, 1x from 8.
        rel = np.cumprod(BP * probs)
        assert abs(rel[1] - 36.0) < 1e-9
        assert abs(rel[3] - 36.0) < 1e-9
        assert np.allclose(rel[7:], 1.0)

    def test_window_is_in_branching_stages_not_levels(self) -> None:
        probs, stages, factor = schedule_survive_probs(BP_SPARSE)
        assert stages == [6, 10, 18]
        assert abs(factor - 48.0 ** (1.0 / 3.0)) < 1e-12
        rel = np.cumprod(BP_SPARSE * probs)
        assert abs(rel[2] - 48.0) < 1e-9
        assert np.allclose(rel[17:], 1.0)

    def test_fewer_branching_stages_than_asked(self) -> None:
        short = np.array([4.0, 9.0, 1.0, 1.0, 3.0, 1.0])
        probs, stages, factor = schedule_survive_probs(short)
        assert stages == [5]
        assert abs(factor - 36.0) < 1e-12
        assert abs(probs[4] - 1.0 / 108.0) < 1e-12
        probs_p, stages_p, factor_p = schedule_survive_probs(BP, return_stages=0)
        assert stages_p == []
        assert factor_p == 1.0
        assert np.all(probs_p[:4] == 1.0)

    def test_determine_scheme_accepts_the_vector(self) -> None:
        probs, _, _ = schedule_survive_probs(BP)
        states = thresholding.determine_scheme(
            probs, BP, ref_ducy=0.1, nbins=32, ntrials=128, snr_final=9.0, seed=3
        )
        thresholds = np.asarray(states.thresholds)
        assert len(thresholds) == len(BP)
        # No cut at the protected stages: thresholds there are the minimum over trials,
        # below the later cutting stages' thresholds.
        assert thresholds[:4].max() < thresholds[4]
