"""Unit tests for `pyloki.utils.psr_utils`, the search-grid geometry kernels.

Why this module first
---------------------
`psr_utils` decides where the search looks. It sets the step size of every parameter
axis (`poly_taylor_step_*`), converts a parameter offset into a phase drift in bins
(`poly_taylor_shift_*`), tiles a parameter range into cells (`range_param`), subdivides
a parent cell into children (`branch_param`), and maps a time sample onto a profile bin
(`get_phase_idx`). Those five things are the search's resolution. A defect here does
not raise; it widens a cell or leaves a gap between two, and the pipeline goes on
returning candidates with quietly reduced sensitivity. That is the failure mode the
notebook-derived tests are worst at: they assert a peak lands in the right cell, which
stays true while the cells themselves are subtly wrong.

It is also a leaf. It imports only `utils.maths` and two constants from `utils.misc`,
so a failure here localises to here, and the tests run in milliseconds rather than
sharing a multi-second pipeline fixture.

Finally it spans the whole numba taxonomy in one module -- plain `@njit`, `@njit` that
calls a `@vectorize` ufunc (`poly_taylor_step_f` -> `maths.fact`), and `@vectorize`
itself (`get_phase_idx`) -- so the pattern set here transfers to every other module.

What is asserted
----------------
Invariants and cross-checks, not golden numbers. Almost every contract below is stated
by the source itself and can be checked without a reference implementation:

- `range_param` documents "Exact Outset Gridding": the cells must tile ``[vmin, vmax]``
  with no gap at either end. That is checkable as ``grid[0] - dv/2 == vmin``, and it is
  a regression guard -- the docstring records that an older `np.linspace(n+2)[1:-1]`
  left exactly such gaps unsearched.
- `branch_param` documents "zero overlap and zero gaps" against the parent cell, which
  is the same statement one level down.
- `range_param_count` and `range_param` are written separately and must agree on the
  count; `branch_param`, `branch_param_padded` and `branch_dparam_crackle` are three
  copies of one calculation and must agree on the spacing.
- The step/shift duality is the deepest one: a step size is *defined* as the parameter
  change that drifts the profile by `eta` bins over `tobs`, so feeding a step straight
  back into the matching shift function must return `eta`. It ties two independently
  written families of functions to a single physical definition, and it holds for every
  parameter order and for both the Chebyshev and plain Taylor coarsening.
- Every scalar kernel has a `_vec` twin, near-duplicated by hand. They must agree.

Coverage of the kernels, and the `NUMBA_DISABLE_JIT` question
-------------------------------------------------------------
Tests are parametrised over ``jit_variants(...)`` (see `tests/jit_utils`), running the
same assertion against the compiled dispatcher and against `.py_func` so the kernel
body is executed under the interpreter where `coverage` can see it. Measured on this
module: 22% -> 26% from the two ufunc bodies alone, against a baseline in which the
jitted fraction of the whole library sits at exactly 0.0% covered.

``NUMBA_DISABLE_JIT=1`` was measured as the alternative and does not work here:

1. It does not disable `@vectorize`. `maths.fact` is a `DUFunc` built by
   `fact_factory`, so it is still compiled while the `_fact` it closes over is not, and
   numba cannot type a plain Python global. `maths.fact` therefore raises `TypingError`
   under the flag, taking 14 of the 24 tests in `tests/test_maths.py` with it -- and
   `psr_utils` calls `maths.fact` from five kernels.
2. It hides the signature contracts. `core.common.shift_add` is declared
   ``f4[:,::1](f4[:,::1], ...)``; compiled it rejects a float64 array, un-jitted it
   accepts one and returns float64. A suite run under the flag would not notice.
3. It is ~110x slower on `core.fold.brutefold` (1.0 ms -> 112 ms at 2**18 samples),
   which rules out running the pipeline-level tests under it.

`.py_func` has its own limit, which is why `jit_variants` is opt-in per test rather
than applied everywhere: it un-jits one level only, so a kernel that passes a
first-class function into a still-jitted helper fails on the Python path
(`np_utils.nb_max.py_func` raises `TypingError`). Nothing in `psr_utils` hits that.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from pyloki.utils import psr_utils
from pyloki.utils.misc import C_VAL
from tests.jit_utils import jit_variants

NBINS = 64
ETA = 1.0
TOBS = 1000.0


class TestGetPhaseIdx:
    """`get_phase_idx` maps an arrival time onto a fractional profile bin.

    The contract is that the result is always a valid bin position: in ``[0, nbins)``
    for any time, any delay and any sign, because a phase is defined modulo one turn.
    An off-by-one at the top of the range indexes past the end of the fold array, which
    is why the wrap at exactly `nbins` is pinned separately.
    """

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.get_phase_idx))
    @pytest.mark.parametrize("proper_time", [-3.0, -0.25, 0.0, 0.25, 2.5, 1e4])
    def test_lies_in_the_profile(self, impl, proper_time: float) -> None:
        iphase = impl(proper_time, 2.0, NBINS, 0.1)
        assert 0.0 <= iphase < NBINS

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.get_phase_idx))
    def test_exact_quarter_turn(self, impl) -> None:
        """A quarter of a 1 Hz period is a quarter of the way through the profile."""
        assert impl(0.25, 1.0, 16, 0.0) == pytest.approx(4.0)

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.get_phase_idx))
    def test_is_periodic_in_one_turn(self, impl) -> None:
        freq = 2.0
        for t in (-1.3, 0.0, 0.4, 7.9):
            assert impl(t, freq, NBINS, 0.0) == pytest.approx(
                impl(t + 1.0 / freq, freq, NBINS, 0.0),
            )

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.get_phase_idx))
    def test_delay_is_a_shift_of_the_arrival_time(self, impl) -> None:
        """`delay` exists to absorb binary motion; it must act as ``t - delay``."""
        assert impl(1.7, 3.0, NBINS, 0.3) == pytest.approx(impl(1.4, 3.0, NBINS, 0.0))

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.get_phase_idx))
    def test_negative_time_wraps_forward(self, impl) -> None:
        """`math.floor` (not truncation) is what makes a negative phase come back."""
        assert impl(-0.25, 1.0, 16, 0.0) == pytest.approx(12.0)

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.get_phase_idx))
    @pytest.mark.parametrize(
        ("freq", "nbins", "match"),
        [
            (0.0, NBINS, "Frequency must be positive"),
            (-1.0, NBINS, "Frequency must be positive"),
            (1.0, 0, "Number of bins must be positive"),
            (1.0, -8, "Number of bins must be positive"),
        ],
    )
    def test_rejects_degenerate_input(
        self,
        impl,
        freq: float,
        nbins: int,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            impl(0.25, freq, nbins, 0.0)


class TestGetPhaseIdxInt:
    """`get_phase_idx_int` rounds the fractional bin half-up, and must never hit nbins.

    Half-up rounding of a value in ``[nbins - 0.5, nbins)`` lands on `nbins`, which is
    one past the end of the fold array. The kernel folds that back to 0 and this pins
    the case, because it is reachable from ordinary input rather than exotic input.
    """

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.get_phase_idx_int))
    @pytest.mark.parametrize("proper_time", [-2.0, -0.01, 0.0, 0.49, 0.99, 3.7])
    def test_is_a_valid_bin_index(self, impl, proper_time: float) -> None:
        ibin = impl(proper_time, 1.0, 16, 0.0)
        assert 0 <= int(ibin) < 16

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.get_phase_idx_int))
    def test_the_last_half_bin_wraps_to_zero(self, impl) -> None:
        """0.99 turn of 16 bins is 15.84, which rounds half-up to 16, i.e. to 0."""
        assert psr_utils.get_phase_idx(0.99, 1.0, 16, 0.0) == pytest.approx(15.84)
        assert int(impl(0.99, 1.0, 16, 0.0)) == 0

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.get_phase_idx_int))
    def test_rounds_half_up_not_down(self, impl) -> None:
        # 0.28125 turn of 16 bins is exactly 4.5 -> 5, not 4.
        assert psr_utils.get_phase_idx(0.28125, 1.0, 16, 0.0) == pytest.approx(4.5)
        assert int(impl(0.28125, 1.0, 16, 0.0)) == 5

    def test_indexes_a_fold_array_as_a_ufunc(self) -> None:
        """Compiled-path contract: it is a ufunc, and `core.fold` indexes with it.

        `brutefold` does ``fold[..., phase_map[isamp]]``, so the output has to be an
        integer array over an array input, not a float one. `.__wrapped__` is scalar
        Python and cannot show this, so it is asserted on the compiled ufunc only.
        """
        proper_time = np.linspace(-1.0, 5.0, 257)
        phase_map = psr_utils.get_phase_idx_int(proper_time, 3.0, NBINS, 0.0)
        assert phase_map.shape == proper_time.shape
        assert np.issubdtype(phase_map.dtype, np.integer)
        assert phase_map.min() >= 0
        assert phase_map.max() < NBINS


class TestRangeParam:
    """`range_param` tiles ``[vmin, vmax]`` with cells; the tiling must be exact.

    The docstring records that an earlier `np.linspace(..., n + 2)[1:-1]` produced an
    "inset" grid leaving unsearched gaps at both ends. These assertions are the
    statement that replaced it: the outer edges of the outermost cells sit exactly on
    `vmin` and `vmax`.
    """

    CASES: ClassVar = [
        (0.0, 1.0, 0.3),
        (1.0, 2.0, 0.25),
        (-5.0, 5.0, 0.7),
        (1e3, 1e3 + 1, 0.01),
    ]

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.range_param))
    @pytest.mark.parametrize(("vmin", "vmax", "dv"), CASES)
    def test_cells_tile_the_range_exactly(
        self,
        impl,
        vmin: float,
        vmax: float,
        dv: float,
    ) -> None:
        grid = impl(vmin, vmax, dv)
        assert len(grid) > 1, "test is vacuous on a one-cell grid"
        dv_actual = grid[1] - grid[0]
        np.testing.assert_allclose(grid[0] - dv_actual / 2.0, vmin, rtol=1e-12)
        np.testing.assert_allclose(grid[-1] + dv_actual / 2.0, vmax, rtol=1e-12)

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.range_param))
    @pytest.mark.parametrize(("vmin", "vmax", "dv"), CASES)
    def test_spacing_is_uniform_and_no_coarser_than_asked(
        self,
        impl,
        vmin: float,
        vmax: float,
        dv: float,
    ) -> None:
        grid = impl(vmin, vmax, dv)
        steps = np.diff(grid)
        # Differencing values of size |grid| loses absolute precision eps*|grid|, which
        # for the 1e3-offset case is 1e-11 of a 0.01 step; the tolerance is on the
        # difference, not on the ratio, for exactly that reason.
        np.testing.assert_allclose(
            steps,
            (vmax - vmin) / len(grid),
            atol=8 * np.finfo(np.float64).eps * np.abs(grid).max(),
            rtol=0,
        )
        assert steps[0] <= dv * (1 + 1e-9), "grid is coarser than requested"

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.range_param))
    @pytest.mark.parametrize("dv", [1.0, 2.0, 1e6])
    def test_a_step_covering_the_range_gives_one_centred_cell(
        self,
        impl,
        dv: float,
    ) -> None:
        np.testing.assert_allclose(impl(0.0, 1.0, dv), [0.5])

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.range_param))
    def test_returns_float64(self, impl) -> None:
        assert impl(0.0, 1.0, 0.3).dtype == np.float64

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.range_param))
    @pytest.mark.parametrize(
        ("vmin", "vmax", "dv"),
        [(1.0, 1.0, 0.1), (2.0, 1.0, 0.1), (0.0, 1.0, 0.0), (0.0, 1.0, -0.1)],
    )
    def test_rejects_an_empty_or_unstepped_range(
        self,
        impl,
        vmin: float,
        vmax: float,
        dv: float,
    ) -> None:
        with pytest.raises(ValueError, match="ensure vmin < vmax and dv > 0"):
            impl(vmin, vmax, dv)


class TestRangeParamCount:
    """`range_param_count` predicts `len(range_param)` without building the grid.

    It is used to size allocations, so a disagreement is a buffer mismatch rather than
    a wrong answer. The two are written separately, which is exactly why they are
    cross-checked rather than each compared to a hand-computed number.
    """

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.range_param_count))
    @pytest.mark.parametrize(
        ("vmin", "vmax", "dv"),
        [*TestRangeParam.CASES, (0.0, 1.0, 1.0), (0.0, 1.0, 2.0), (-1.0, 1.0, 0.125)],
    )
    def test_agrees_with_the_grid_it_describes(
        self,
        impl,
        vmin: float,
        vmax: float,
        dv: float,
    ) -> None:
        assert impl(vmin, vmax, dv) == len(psr_utils.range_param(vmin, vmax, dv))

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.range_param_count))
    @pytest.mark.parametrize(
        ("vmin", "vmax", "dv"),
        [(1.0, 1.0, 0.1), (2.0, 1.0, 0.1), (0.0, 1.0, 0.0), (0.0, 1.0, -0.1)],
    )
    def test_rejects_an_empty_or_unstepped_range(
        self,
        impl,
        vmin: float,
        vmax: float,
        dv: float,
    ) -> None:
        with pytest.raises(ValueError, match=r"ensure vmin < vmax and dv > 0\.0"):
            impl(vmin, vmax, dv)


class TestBranchParam:
    """`branch_param` subdivides one parent cell into contiguous children.

    Its docstring promises the children's outer edges are flush with the parent's and
    that there is "zero overlap and zero gaps" between siblings. In the pruning tree
    that promise is what makes the union of a node's children equal to the node: a gap
    is signal the search can no longer reach, an overlap is duplicated work.
    """

    CASES: ClassVar = [
        (10.0, 4.0, 1.0),
        (0.0, 1.0, 0.3),
        (-3.0, 0.5, 0.2),
        (1e-6, 1e-7, 3e-8),
    ]

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.branch_param))
    @pytest.mark.parametrize(("param_cur", "dparam_cur", "dparam_new"), CASES)
    def test_children_exactly_fill_the_parent_cell(
        self,
        impl,
        param_cur: float,
        dparam_cur: float,
        dparam_new: float,
    ) -> None:
        values, dparam_actual = impl(param_cur, dparam_cur, dparam_new)
        np.testing.assert_allclose(
            values[0] - dparam_actual / 2.0,
            param_cur - dparam_cur / 2.0,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            values[-1] + dparam_actual / 2.0,
            param_cur + dparam_cur / 2.0,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            len(values) * dparam_actual,
            dparam_cur,
            rtol=1e-12,
        )

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.branch_param))
    @pytest.mark.parametrize(("param_cur", "dparam_cur", "dparam_new"), CASES)
    def test_siblings_abut_without_gap_or_overlap(
        self,
        impl,
        param_cur: float,
        dparam_cur: float,
        dparam_new: float,
    ) -> None:
        values, dparam_actual = impl(param_cur, dparam_cur, dparam_new)
        assert len(values) > 1, "test is vacuous unless the cell actually split"
        np.testing.assert_allclose(np.diff(values), dparam_actual, rtol=1e-12)

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.branch_param))
    @pytest.mark.parametrize(("param_cur", "dparam_cur", "dparam_new"), CASES)
    def test_refined_spacing_is_no_coarser_than_requested(
        self,
        impl,
        param_cur: float,
        dparam_cur: float,
        dparam_new: float,
    ) -> None:
        _, dparam_actual = impl(param_cur, dparam_cur, dparam_new)
        assert dparam_actual <= dparam_new * (1 + 1e-12)

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.branch_param))
    @pytest.mark.parametrize("dparam_new", [2.0, 5.0, 1e3])
    def test_a_coarser_target_does_not_split(self, impl, dparam_new: float) -> None:
        """Asking for a spacing wider than the parent returns the parent unchanged."""
        values, dparam_actual = impl(5.0, 2.0, dparam_new)
        np.testing.assert_allclose(values, [5.0])
        assert dparam_actual == pytest.approx(2.0)

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.branch_param))
    def test_returns_float64(self, impl) -> None:
        values, _ = impl(10.0, 4.0, 1.0)
        assert values.dtype == np.float64

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.branch_param))
    @pytest.mark.parametrize(
        ("dparam_cur", "dparam_new"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)]
    )
    def test_rejects_non_positive_spacing(
        self,
        impl,
        dparam_cur: float,
        dparam_new: float,
    ) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            impl(1.0, dparam_cur, dparam_new)


class TestBranchParamVariants:
    """The three copies of the branching arithmetic must stay one calculation.

    `branch_param`, `branch_param_padded` and `branch_dparam_crackle` each recompute
    ``num_points = max(1, ceil(dparam_cur / dparam_new - eps))`` and the spacing that
    follows from it. They are near-duplicated source, so they are cross-checked against
    each other; editing one and not the others is the realistic way this breaks.
    """

    BRANCH_MAX = 32

    @pytest.mark.parametrize(
        ("param_cur", "dparam_cur", "dparam_new"),
        TestBranchParam.CASES,
    )
    def test_padded_matches_unpadded(
        self,
        param_cur: float,
        dparam_cur: float,
        dparam_new: float,
    ) -> None:
        expected, dparam_expected = psr_utils.branch_param(
            param_cur,
            dparam_cur,
            dparam_new,
        )
        out = np.full(self.BRANCH_MAX, np.nan)
        dparam_actual, num_points = psr_utils.branch_param_padded(
            out,
            param_cur,
            dparam_cur,
            dparam_new,
        )
        assert num_points == len(expected)
        assert dparam_actual == pytest.approx(dparam_expected)
        np.testing.assert_allclose(out[:num_points], expected, rtol=1e-12)
        assert np.all(np.isnan(out[num_points:])), "wrote past the reported count"

    @pytest.mark.parametrize(
        ("param_cur", "dparam_cur", "dparam_new"),
        TestBranchParam.CASES,
    )
    def test_crackle_matches_the_spacing(
        self,
        param_cur: float,
        dparam_cur: float,
        dparam_new: float,
    ) -> None:
        _, dparam_expected = psr_utils.branch_param(param_cur, dparam_cur, dparam_new)
        dparam_actual = psr_utils.branch_dparam_crackle(
            dparam_cur,
            dparam_new,
            self.BRANCH_MAX,
        )
        assert dparam_actual == pytest.approx(dparam_expected)

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.branch_param_padded))
    def test_padded_rejects_more_children_than_the_buffer_holds(self, impl) -> None:
        """The buffer is `branch_max` long; overrunning it must raise, not corrupt."""
        out = np.zeros(4)
        with pytest.raises(ValueError, match="increase branch_max"):
            impl(out, 0.0, 1.0, 0.1)

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.branch_param_padded))
    @pytest.mark.parametrize(
        ("dparam_cur", "dparam_new"),
        [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)],
    )
    def test_padded_rejects_non_positive_spacing(
        self,
        impl,
        dparam_cur: float,
        dparam_new: float,
    ) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            impl(np.zeros(self.BRANCH_MAX), 1.0, dparam_cur, dparam_new)

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.branch_dparam_crackle))
    def test_crackle_rejects_more_children_than_branch_max(self, impl) -> None:
        with pytest.raises(ValueError, match="increase branch_max"):
            impl(1.0, 0.1, 4)

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.branch_dparam_crackle))
    @pytest.mark.parametrize(
        ("dparam_cur", "dparam_new"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)]
    )
    def test_crackle_rejects_non_positive_spacing(
        self,
        impl,
        dparam_cur: float,
        dparam_new: float,
    ) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            impl(dparam_cur, dparam_new, self.BRANCH_MAX)


class TestStepShiftDuality:
    """A step size is the parameter change that drifts the profile by `eta` bins.

    That is the definition the whole grid rests on, and it makes the step and shift
    families each other's inverse. Feeding a step straight into the matching shift
    function must give back `eta`, for every parameter order and for both the
    Chebyshev-coarsened and plain Taylor grids. If either family drifts from the
    definition the search is mistuned -- too fine and it is needlessly slow, too coarse
    and it loses signal -- and nothing else in the suite would notice.
    """

    @pytest.mark.parametrize("nparams", [1, 2, 3, 4])
    @pytest.mark.parametrize("use_cheby", [True, False])
    def test_a_step_drifts_the_profile_by_eta_bins(
        self,
        nparams: int,
        use_cheby: bool,
    ) -> None:
        step = psr_utils.poly_taylor_step_f(nparams, TOBS, NBINS, ETA, 0.0, use_cheby)
        shift = psr_utils.poly_taylor_shift_d(
            np.zeros(nparams),
            step,
            TOBS,
            NBINS,
            C_VAL,
            0.0,
            use_cheby,
        )
        np.testing.assert_allclose(shift, ETA, rtol=1e-9)

    @pytest.mark.parametrize("nparams", [2, 3, 4])
    @pytest.mark.parametrize("use_cheby", [True, False])
    def test_split_f_fires_exactly_at_one_step(
        self,
        nparams: int,
        use_cheby: bool,
    ) -> None:
        """`split_f` is the same duality as a boolean, so it must turn over at `eta`."""
        step = psr_utils.poly_taylor_step_f(nparams, TOBS, NBINS, ETA, 0.0, use_cheby)
        for i in range(nparams):
            k = nparams - 1 - i  # step is returned in reverse derivative order
            assert not psr_utils.split_f(
                0.0,
                step[i] * 0.99,
                TOBS,
                k,
                NBINS,
                ETA,
                0.0,
                use_cheby,
            )
            assert psr_utils.split_f(
                0.0,
                step[i] * 1.01,
                TOBS,
                k,
                NBINS,
                ETA,
                0.0,
                use_cheby,
            )

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.poly_taylor_step_f))
    @pytest.mark.parametrize("nparams", [1, 2, 3, 4, 5])
    def test_steps_are_positive_and_ordered_by_derivative(
        self,
        impl,
        nparams: int,
    ) -> None:
        """Reverse order: the highest derivative -- the finest step -- comes first."""
        step = impl(nparams, TOBS, NBINS, ETA)
        assert step.shape == (nparams,)
        assert step.dtype == np.float64
        assert np.all(step > 0)
        assert np.all(np.diff(step) > 0), f"not ascending towards f0: {step}"

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.poly_taylor_step_f))
    @pytest.mark.parametrize("nparams", [1, 2, 3, 4])
    def test_chebyshev_coarsens_by_two_per_derivative(
        self,
        impl,
        nparams: int,
    ) -> None:
        plain = impl(nparams, TOBS, NBINS, ETA, 0.0, use_cheby=False)
        cheby = impl(nparams, TOBS, NBINS, ETA, 0.0, use_cheby=True)
        # steps are reversed, so k counts down across the array
        k = np.arange(nparams)[::-1]
        np.testing.assert_allclose(cheby / plain, 2.0**k, rtol=1e-12)

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.poly_taylor_step_f))
    def test_eta_scales_the_grid_linearly(self, impl) -> None:
        """`eta` is a bin tolerance, so doubling it doubles every step."""
        np.testing.assert_allclose(
            impl(4, TOBS, NBINS, 2 * ETA),
            2 * impl(4, TOBS, NBINS, ETA),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.poly_taylor_step_f))
    def test_a_longer_observation_needs_a_finer_grid(self, impl) -> None:
        assert np.all(impl(4, 2 * TOBS, NBINS, ETA) < impl(4, TOBS, NBINS, ETA))


class TestScalarVectorAgreement:
    """Every scalar kernel here has a hand-written `_vec` twin; they must agree.

    The batched forms are not generated from the scalar ones, they are copies with the
    broadcasting written out by hand, so the two can disagree silently. Running the
    scalar form over the batch and stacking is the cheapest statement that they do not.
    """

    F_MAX = np.array([100.0, 200.0, 412.5])

    def test_step_d_f_vec_matches_the_scalar_form(self) -> None:
        expected = np.stack(
            [psr_utils.poly_taylor_step_d_f(3, TOBS, NBINS, ETA, f) for f in self.F_MAX]
        )
        actual = psr_utils.poly_taylor_step_d_f_vec(3, TOBS, NBINS, ETA, self.F_MAX)
        assert actual.shape == (len(self.F_MAX), 3)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_step_d_vec_matches_the_scalar_form(self) -> None:
        expected = np.stack(
            [psr_utils.poly_taylor_step_d(3, TOBS, NBINS, ETA, f) for f in self.F_MAX]
        )
        actual = psr_utils.poly_taylor_step_d_vec(3, TOBS, NBINS, ETA, self.F_MAX)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_shift_d_vec_matches_the_scalar_form(self) -> None:
        old = np.zeros((len(self.F_MAX), 3))
        new = np.tile(np.array([1e-10, 1e-8, 1e-5]), (len(self.F_MAX), 1))
        expected = np.stack(
            [
                psr_utils.poly_taylor_shift_d(old[i], new[i], TOBS, NBINS, f)
                for i, f in enumerate(self.F_MAX)
            ]
        )
        actual = psr_utils.poly_taylor_shift_d_vec(old, new, TOBS, NBINS, self.F_MAX)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.poly_taylor_shift_d_f_vec))
    def test_shift_d_f_vec_leaves_the_frequency_axis_unscaled(self, impl) -> None:
        """The last column is `f` itself, so it must not pick up the `f/c` factor."""
        old = np.zeros((2, 3))
        new = np.tile(np.array([1e-10, 1e-8, 1e-5]), (2, 1))
        f_cur = np.array([100.0, 400.0])
        shift = impl(old, new, TOBS, NBINS, f_cur)
        assert shift.shape == (2, 3)
        # rows differ only through f_cur, and only on the non-frequency columns
        np.testing.assert_allclose(shift[1, -1], shift[0, -1], rtol=1e-12)
        np.testing.assert_allclose(
            shift[1, :-1] / shift[0, :-1],
            f_cur[1] / f_cur[0],
            rtol=1e-12,
        )

    @pytest.mark.parametrize("impl", jit_variants(psr_utils.poly_cheb_shift_vec))
    def test_cheb_shift_is_linear_in_the_parameter_offset(self, impl) -> None:
        old = np.zeros((2, 3))
        new = np.tile(np.array([1e-10, 1e-8, 1e-5]), (2, 1))
        f_cur = np.array([100.0, 400.0])
        np.testing.assert_allclose(
            impl(old, 2 * new, NBINS, f_cur),
            2 * impl(old, new, NBINS, f_cur),
            rtol=1e-12,
        )


class TestGetNearestIndices:
    """Mapping a parameter value back to its grid cell must be the real nearest cell.

    The kernel does it analytically -- ``int(n * (v - vmin) / range)`` -- rather than by
    searching, which is fast but only correct if it agrees with an argmin over the grid
    `range_param` actually built. The two are written from the same convention but not
    from the same code, so they are checked against each other over random values.
    """

    LIMITS = np.array([[-5.0, 5.0], [100.0, 200.0]])
    COUNTS = np.array([15, 7])

    def _grids(self) -> list[np.ndarray]:
        return [
            psr_utils.range_param(lo, hi, (hi - lo) / n)
            for (lo, hi), n in zip(self.LIMITS, self.COUNTS, strict=True)
        ]

    @pytest.mark.parametrize(
        "impl", jit_variants(psr_utils.get_nearest_indices_analytical)
    )
    def test_agrees_with_an_argmin_over_the_real_grid(self, impl) -> None:
        grids = self._grids()
        assert [len(g) for g in grids] == list(self.COUNTS)
        rng = np.random.default_rng(20240917)
        for _ in range(200):
            value = np.array(
                [rng.uniform(lo, hi) for lo, hi in self.LIMITS],
            )
            actual = impl(value, self.COUNTS, self.LIMITS)
            expected = np.array(
                [
                    int(np.argmin(np.abs(g - v)))
                    for g, v in zip(grids, value, strict=True)
                ]
            )
            np.testing.assert_array_equal(actual, expected, err_msg=f"value={value}")

    @pytest.mark.parametrize(
        "impl", jit_variants(psr_utils.get_nearest_indices_analytical)
    )
    def test_out_of_range_values_clamp_to_the_edge_cells(self, impl) -> None:
        np.testing.assert_array_equal(
            impl(np.array([-1e9, -1e9]), self.COUNTS, self.LIMITS), [0, 0]
        )
        np.testing.assert_array_equal(
            impl(np.array([1e9, 1e9]), self.COUNTS, self.LIMITS),
            self.COUNTS - 1,
        )

    @pytest.mark.parametrize(
        "impl", jit_variants(psr_utils.get_nearest_indices_analytical)
    )
    def test_degenerate_axes_collapse_to_index_zero(self, impl) -> None:
        """A 1-cell or 0-cell axis has no index to pick; it must not divide by zero."""
        np.testing.assert_array_equal(
            impl(np.array([3.0, 150.0]), np.array([1, 0]), self.LIMITS), [0, 0]
        )

    @pytest.mark.parametrize(
        "impl", jit_variants(psr_utils.get_nearest_indices_analytical)
    )
    def test_returns_int64_indices(self, impl) -> None:
        assert impl(np.array([1.0, 150.0]), self.COUNTS, self.LIMITS).dtype == np.int64

    @pytest.mark.parametrize(
        "impl", jit_variants(psr_utils.get_nearest_indices_2d_batch)
    )
    def test_the_2d_batch_form_matches_the_general_one(self, impl) -> None:
        rng = np.random.default_rng(20240917)
        accel = rng.uniform(-5.0, 5.0, 64)
        freq = rng.uniform(100.0, 200.0, 64)
        actual = impl(accel, freq, self.COUNTS, self.LIMITS)
        assert actual.shape == (64, 2)
        assert actual.dtype == np.int64
        expected = np.stack(
            [
                psr_utils.get_nearest_indices_analytical(
                    np.array([a, f]), self.COUNTS, self.LIMITS
                )
                for a, f in zip(accel, freq, strict=True)
            ]
        )
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize(
        "impl", jit_variants(psr_utils.get_nearest_indices_2d_batch)
    )
    def test_a_collapsed_axis_gives_index_zero(self, impl) -> None:
        """A zero-width limit makes the inverse step 0; that axis collapses to 0.

        Each axis is guarded separately, so collapsing one must not disturb the other.
        """
        collapsed_accel = np.array([[3.0, 3.0], [100.0, 200.0]])
        out = impl(np.array([3.0]), np.array([150.0]), self.COUNTS, collapsed_accel)
        assert out[0, -2] == 0
        assert out[0, -1] == 3, "the live frequency axis was disturbed"

        collapsed_freq = np.array([[-5.0, 5.0], [150.0, 150.0]])
        out = impl(np.array([3.0]), np.array([150.0]), self.COUNTS, collapsed_freq)
        assert out[0, -1] == 0
        assert out[0, -2] == 12, "the live acceleration axis was disturbed"

    @pytest.mark.parametrize(
        "impl",
        jit_variants(psr_utils.get_nearest_indices_2d_batch),
    )
    def test_the_2d_batch_form_clamps_out_of_range_values(self, impl) -> None:
        accel = np.array([-1e9, 1e9, 0.0])
        freq = np.array([-1e9, 1e9, 150.0])
        out = impl(accel, freq, self.COUNTS, self.LIMITS)
        np.testing.assert_array_equal(out[0], [0, 0])
        np.testing.assert_array_equal(out[1], self.COUNTS - 1)
        assert np.all(out >= 0)
        assert np.all(out < self.COUNTS)


class TestCompiledAndPythonDoNotDiverge:
    """The compiled kernel and its `.py_func` must return the same numbers.

    `jit_variants` asserts the same *properties* on both paths, which catches a
    divergence a test happens to look at. This checks the stronger thing directly:
    identical output for identical input, across the module. It is the guard that makes
    it safe to attribute coverage recorded on the Python path to the compiled kernel.

    `fastmath=True` permits reassociation, so equality is to a tight relative
    tolerance rather than bitwise.
    """

    F_VEC = np.array([100.0, 200.0])
    OLD = np.zeros((2, 3))
    NEW = np.tile(np.array([1e-10, 1e-8, 1e-5]), (2, 1))

    CALLS: ClassVar = [
        (psr_utils.get_phase_idx, (0.3, 2.0, NBINS, 0.05)),
        (psr_utils.get_phase_idx_int, (0.3, 2.0, NBINS, 0.05)),
        (psr_utils.poly_taylor_step_f, (4, TOBS, NBINS, ETA)),
        (psr_utils.poly_taylor_step_d_f, (4, TOBS, NBINS, ETA, 150.0)),
        (psr_utils.poly_taylor_step_d, (4, TOBS, NBINS, ETA, 150.0)),
        (psr_utils.poly_taylor_step_d_vec, (3, TOBS, NBINS, ETA, F_VEC)),
        (psr_utils.poly_taylor_step_d_f_vec, (3, TOBS, NBINS, ETA, F_VEC)),
        (psr_utils.poly_taylor_shift_d, (OLD[0], NEW[0], TOBS, NBINS, 150.0)),
        (psr_utils.poly_taylor_shift_d_vec, (OLD, NEW, TOBS, NBINS, F_VEC)),
        (psr_utils.poly_taylor_shift_d_f_vec, (OLD, NEW, TOBS, NBINS, F_VEC)),
        (psr_utils.split_f, (0.0, 1e-9, TOBS, 1, NBINS, ETA)),
        (psr_utils.period_step, (TOBS, NBINS, 0.01, 1.0)),
        (psr_utils.poly_cheb_step_vec, (3, NBINS, ETA, F_VEC)),
        (psr_utils.poly_cheb_shift_vec, (OLD, NEW, NBINS, F_VEC)),
        (psr_utils.branch_dparam_crackle, (4.0, 1.0, 16)),
        (psr_utils.range_param_count, (0.0, 1.0, 0.3)),
        (psr_utils.range_param, (0.0, 1.0, 0.3)),
        (
            psr_utils.get_nearest_indices_analytical,
            (np.array([1.0, 150.0]), np.array([15, 7]), TestGetNearestIndices.LIMITS),
        ),
        (
            psr_utils.get_nearest_indices_2d_batch,
            (
                np.array([1.0, -2.0]),
                np.array([150.0, 180.0]),
                np.array([15, 7]),
                TestGetNearestIndices.LIMITS,
            ),
        ),
    ]

    @pytest.mark.parametrize(
        ("kernel", "args"),
        CALLS,
        ids=[f.__name__ for f, _ in CALLS],
    )
    def test_same_result_on_both_paths(self, kernel, args: tuple) -> None:
        compiled = kernel(*args)
        python = jit_variants(kernel)[1].values[0](*args)
        np.testing.assert_allclose(compiled, python, rtol=1e-12)

    def test_branch_param_agrees_on_both_paths(self) -> None:
        """Returns a tuple, so it is unpacked rather than compared whole."""
        args = (10.0, 4.0, 1.0)
        values, dparam = psr_utils.branch_param(*args)
        values_py, dparam_py = psr_utils.branch_param.py_func(*args)
        np.testing.assert_allclose(values, values_py, rtol=1e-12)
        assert dparam == pytest.approx(dparam_py)

    def test_branch_param_padded_agrees_on_both_paths(self) -> None:
        """Writes through its first argument, so each path gets its own buffer."""
        out, out_py = np.zeros(16), np.zeros(16)
        dparam, count = psr_utils.branch_param_padded(out, 10.0, 4.0, 1.0)
        dparam_py, count_py = psr_utils.branch_param_padded.py_func(
            out_py,
            10.0,
            4.0,
            1.0,
        )
        assert count == count_py
        assert dparam == pytest.approx(dparam_py)
        np.testing.assert_allclose(out, out_py, rtol=1e-12)
