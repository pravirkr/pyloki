"""Regression coverage for the dynamic threshold scheme.

`DynamicThresholdScheme.run()` used to complete without error while writing
**zero-filled** state records from stage 1 onward. Because `is_empty` is part of the
record, zeroing it set the flag to `False`, so the states looked populated while every
number in them -- threshold, complexity, survival probability -- was 0. The scheme then
discarded them all at the next stage (`np.digitize(0.0, probs) - 1 == -1`), leaving
nothing to backtrack and making the whole optimiser silently useless.

The cause is a numba miscompilation of whole-record assignment when the record comes
from an array returned by another jitted function; `test_record_assignment_*` below pins
the construct itself, and the scheme tests pin the consequence.
"""

from __future__ import annotations

import numpy as np
import pytest
from numba import njit

from pyloki.detection.thresholding import DynamicThresholdScheme

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


def _small_scheme(mode: str) -> DynamicThresholdScheme:
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
    )


@pytest.mark.parametrize("mode", ["legacy", "improved"])
class TestDynamicThresholdScheme:
    def test_states_survive_past_the_first_stage(self, mode: str) -> None:
        """The headline symptom: everything vanished from stage 2 onward."""
        scheme = _small_scheme(mode)
        scheme.run(thres_neigh=5)
        populated = [
            int((~scheme.states[i]["is_empty"]).sum()) for i in range(scheme.nstages)
        ]
        assert all(n > 0 for n in populated), (
            f"states emptied partway through: non-empty per stage = {populated}"
        )

    def test_stored_states_are_not_zero_filled(self, mode: str) -> None:
        """The cause, distinct from the symptom.

        A zeroed record reads as `is_empty == False`, so counting non-empty states is
        not enough -- the numbers inside have to be real. Every stored state must carry
        the threshold it was built with, and no threshold in the grid is 0.
        """
        scheme = _small_scheme(mode)
        scheme.run(thres_neigh=5)
        for istage in range(scheme.nstages):
            stage = scheme.states[istage]
            stored = stage[~stage["is_empty"]]
            assert len(stored) > 0
            assert np.all(stored["threshold"] > 0.0), (
                f"stage {istage}: zero-filled records "
                f"({int(np.sum(stored['threshold'] == 0.0))} of {len(stored)})"
            )
            assert np.all(stored["success_h1_cumul"] > 0.0)
            assert np.all(stored["complexity"] > 0.0)

    def test_survival_decreases_monotonically(self, mode: str) -> None:
        """Cumulative survival is a product of probabilities, so it cannot grow."""
        scheme = _small_scheme(mode)
        scheme.run(thres_neigh=5)
        best = []
        for istage in range(scheme.nstages):
            stage = scheme.states[istage]
            best.append(float(stage[~stage["is_empty"]]["success_h1_cumul"].max()))
        assert all(b <= 1.0 + 1e-6 for b in best)
        assert best == sorted(best, reverse=True), f"survival is not monotone: {best}"
