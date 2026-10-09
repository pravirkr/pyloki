"""Unit tests for `pyloki.utils.np_utils`, the array helpers under the search kernels.

Every kernel here is a small, general array operation with an obvious NumPy or
brute-force reference, so the contracts are cross-checks rather than golden numbers:
`cartesian_prod` against `itertools.product` and its two pure-NumPy siblings,
`find_nearest_sorted_idx` against an argmin over the whole array, the scalar kernels
against their `_vect`/batched twins, and `lstsq_weighted` against an exactly solvable
system. Tests are parametrised over ``jit_variants(...)`` (see `tests/jit_utils`) so the
same assertion runs against the compiled dispatcher and its Python source.

The `nb_max` limitation, and how it is handled
----------------------------------------------
`nb_max` and `np_mean` pass a NumPy builtin (`np.max`, `np.mean`) into the still-jitted
`np_apply_along_axis`. Their ``.py_func`` therefore raises `TypingError`: numba cannot
type a NumPy builtin arriving from the Python boundary. The same holds for calling the
compiled `np_apply_along_axis` from Python with `np.max`. So:

- `nb_max` and `np_mean` are tested on the compiled path only
  (`TestReductionsCompiledOnly`). Their one-line bodies stay uncovered, and that is the
  cost of the limitation, accepted rather than worked around.
- `np_apply_along_axis`, which does the real work, is covered through its own
  ``.py_func``, called with a plain Python callable. The same contract is checked on
  the compiled side through `nb_max`, which is the only way the compiled
  `np_apply_along_axis` can be reached, error paths included.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numba.core.errors import TypingError

from pyloki.utils import np_utils
from tests.jit_utils import jit_variants

# Library defects found while writing these tests. Each is pinned with a strict xfail
# so that the fix, when it lands, turns the pin into a failure and forces its removal.


def _brute_nearest(array: np.ndarray, value: float) -> int:
    """Index of the closest entry; the first (smallest) index on an exact tie."""
    return int(np.argmin(np.abs(array - value)))


class TestFindNearestSortedIdx:
    RNG_SEED = 20260929

    @pytest.mark.parametrize("impl", jit_variants(np_utils.find_nearest_sorted_idx))
    def test_matches_brute_force(self, impl) -> None:
        rng = np.random.default_rng(self.RNG_SEED)
        for _ in range(50):
            array = np.sort(rng.uniform(-10, 10, rng.integers(1, 40)))
            # Include values outside the array on both sides.
            for value in rng.uniform(-12, 12, 20):
                idx = impl(array, value)
                best = np.abs(array - value).min()
                # Within the documented tolerance of the true minimum distance.
                assert abs(value - array[idx]) <= best * (1 + 1e-5) + 1e-8

    @pytest.mark.parametrize("impl", jit_variants(np_utils.find_nearest_sorted_idx))
    def test_tie_returns_smaller_index(self, impl) -> None:
        array = np.array([0.0, 1.0, 2.0, 4.0])
        assert impl(array, 0.5) == 0
        assert impl(array, 1.5) == 1
        assert impl(array, 3.0) == 2

    @pytest.mark.parametrize("impl", jit_variants(np_utils.find_nearest_sorted_idx))
    def test_outside_range_clamps_to_ends(self, impl) -> None:
        """Clamp to the first and last entries.

        Above the last entry the ``idx == len(array)`` branch sets ``diff_curr = inf``;
        under ``fastmath`` that comparison is the one most at risk.
        """
        array = np.array([0.0, 1.0, 2.0])
        assert impl(array, -5.0) == 0
        assert impl(array, 5.0) == 2
        assert impl(array, np.float64(2.0)) == 2

    @pytest.mark.parametrize("impl", jit_variants(np_utils.find_nearest_sorted_idx))
    def test_single_element(self, impl) -> None:
        array = np.array([3.0])
        for value in (-1.0, 3.0, 7.0):
            assert impl(array, value) == 0

    @pytest.mark.parametrize(
        "impl",
        jit_variants(np_utils.find_nearest_sorted_idx)
        + jit_variants(np_utils.find_nearest_sorted_idx_vect),
    )
    def test_empty_array_raises(self, impl) -> None:
        value = np.array([1.0]) if "vect" in impl.__name__ else 1.0
        with pytest.raises(ValueError, match="must not be empty"):
            impl(np.array([], dtype=np.float64), value)

    @pytest.mark.parametrize(
        "impl", jit_variants(np_utils.find_nearest_sorted_idx_vect)
    )
    def test_vect_matches_scalar(self, impl) -> None:
        rng = np.random.default_rng(self.RNG_SEED + 1)
        array = np.sort(rng.uniform(-10, 10, 33))
        # Exact midpoints as well as random values, so the tie branch is exercised.
        values = np.concatenate(
            [rng.uniform(-12, 12, 200), (array[:-1] + array[1:]) / 2],
        )
        expected = [np_utils.find_nearest_sorted_idx(array, v) for v in values]
        np.testing.assert_array_equal(impl(array, values), expected)


class TestNpApplyAlongAxis:
    ARR = np.arange(12.0).reshape(3, 4) ** 1.5

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("func1d", [np.max, np.sum, lambda row: row[-1] - row[0]])
    def test_py_func_matches_numpy(self, axis: int, func1d) -> None:
        got = np_utils.np_apply_along_axis.py_func(func1d, axis, self.ARR)
        np.testing.assert_allclose(got, np.apply_along_axis(func1d, axis, self.ARR))

    def test_py_func_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            np_utils.np_apply_along_axis.py_func(np.max, 0, np.zeros((2, 2, 2)))

    def test_py_func_rejects_bad_axis(self) -> None:
        with pytest.raises(ValueError, match="axis"):
            np_utils.np_apply_along_axis.py_func(np.max, 2, self.ARR)

    def test_compiled_error_paths_via_nb_max(self) -> None:
        """The compiled kernel is reachable only from jitted code (see module docs)."""
        with pytest.raises(ValueError, match="2D"):
            np_utils.nb_max(np.zeros((2, 2, 2)), 0)
        with pytest.raises(ValueError, match="axis"):
            np_utils.nb_max(self.ARR, 2)


class TestReductionsCompiledOnly:
    """`nb_max`/`np_mean` have no usable ``.py_func``; see the module docstring."""

    ARR = np.random.default_rng(7).standard_normal((5, 9))

    @pytest.mark.parametrize("axis", [0, 1])
    def test_nb_max(self, axis: int) -> None:
        np.testing.assert_array_equal(
            np_utils.nb_max(self.ARR, axis),
            self.ARR.max(axis=axis),
        )

    @pytest.mark.parametrize("axis", [0, 1])
    def test_np_mean(self, axis: int) -> None:
        np.testing.assert_allclose(
            np_utils.np_mean(self.ARR, axis),
            self.ARR.mean(axis=axis),
            rtol=1e-12,
        )

    def test_py_func_limitation_is_real(self) -> None:
        """Pin the reason for this class.

        If this starts failing, fold these tests into `jit_variants` and delete it.
        """
        with pytest.raises(TypingError):
            np_utils.nb_max.py_func(self.ARR, 0)


class TestDownsample1d:
    @pytest.mark.parametrize("impl", jit_variants(np_utils.downsample_1d))
    @pytest.mark.parametrize("factor", [1, 2, 3, 6])
    def test_is_block_mean(self, impl, factor: int) -> None:
        arr = np.random.default_rng(11).standard_normal(36)
        np.testing.assert_allclose(
            impl(arr, factor),
            arr.reshape(-1, factor).mean(axis=1),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("impl", jit_variants(np_utils.downsample_1d))
    def test_preserves_total(self, impl) -> None:
        arr = np.random.default_rng(12).standard_normal(64)
        np.testing.assert_allclose(impl(arr, 4).sum() * 4, arr.sum(), rtol=1e-12)

    @pytest.mark.parametrize("impl", jit_variants(np_utils.downsample_1d))
    def test_non_divisible_length_raises(self, impl) -> None:
        # The message is NumPy's (py_func) or numba's (compiled); they differ.
        with pytest.raises(ValueError, match=r"reshape|size"):
            impl(np.arange(7.0), 2)


class TestNbRoll:
    ARR = np.arange(24.0).reshape(2, 3, 4)

    @pytest.mark.parametrize("impl", jit_variants(np_utils.nb_roll))
    @pytest.mark.parametrize("shift", [0, 1, -3, 25])
    def test_flat(self, impl, shift: int) -> None:
        np.testing.assert_array_equal(impl(self.ARR, shift), np.roll(self.ARR, shift))

    @pytest.mark.parametrize("impl", jit_variants(np_utils.nb_roll))
    @pytest.mark.parametrize("axis", [0, 1, 2, -1])
    def test_along_axis(self, impl, axis: int) -> None:
        np.testing.assert_array_equal(
            impl(self.ARR, 2, axis),
            np.roll(self.ARR, 2, axis),
        )

    @pytest.mark.parametrize("impl", jit_variants(np_utils.nb_roll))
    def test_tuple_shift_and_axis(self, impl) -> None:
        np.testing.assert_array_equal(
            impl(self.ARR, (1, -2), (1, 2)),
            np.roll(self.ARR, (1, -2), (1, 2)),
        )


class TestCartesianProd:
    CASES = (
        [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])],
        [np.array([0.5]), np.array([1.0, 2.0]), np.array([-1.0, 0.0, 1.0, 2.0])],
        [np.array([7.0, 8.0, 9.0])],
        [np.array([1.0]), np.array([2.0])],
    )

    @pytest.mark.parametrize("impl", jit_variants(np_utils.cartesian_prod))
    @pytest.mark.parametrize("arrays", CASES)
    def test_matches_itertools(self, impl, arrays: list[np.ndarray]) -> None:
        expected = np.array(list(itertools.product(*arrays)))
        np.testing.assert_array_equal(impl(arrays), expected)

    @pytest.mark.parametrize("arrays", CASES)
    def test_three_implementations_agree(self, arrays: list[np.ndarray]) -> None:
        """Cross-check three independent implementations of one operation.

        `cartesian_prod`, `cartesian_prod_np` and `cartesian_prod_st` must agree,
        row order included.
        """
        ref = np_utils.cartesian_prod(arrays)
        np.testing.assert_array_equal(np_utils.cartesian_prod_np(arrays), ref)
        np.testing.assert_array_equal(np_utils.cartesian_prod_st(arrays), ref)


class TestCartesianProdPadded:
    @staticmethod
    def _padded(
        batches: list[list[np.ndarray]],
        width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_batch, nparams = len(batches), len(batches[0])
        padded = np.full((n_batch, nparams, width), np.nan)
        counts = np.zeros((n_batch, nparams), dtype=np.int64)
        for i, params in enumerate(batches):
            for j, vals in enumerate(params):
                padded[i, j, : len(vals)] = vals
                counts[i, j] = len(vals)
        return padded, counts

    @pytest.mark.parametrize("impl", jit_variants(np_utils.cartesian_prod_padded))
    def test_matches_per_batch_product(self, impl) -> None:
        rng = np.random.default_rng(31)
        batches = [
            [rng.standard_normal(rng.integers(1, 5)) for _ in range(3)]
            for _ in range(6)
        ]
        padded, counts = self._padded(batches, width=5)
        cart, origins = impl(padded, counts, len(batches), 3)

        expected = [row for params in batches for row in itertools.product(*params)]
        expected_origins = [
            i for i, params in enumerate(batches) for _ in itertools.product(*params)
        ]
        np.testing.assert_array_equal(cart, np.array(expected))
        np.testing.assert_array_equal(origins, expected_origins)
        # The NaN padding must never leak into the product.
        assert np.isfinite(cart).all()

    @pytest.mark.parametrize("impl", jit_variants(np_utils.cartesian_prod_padded))
    def test_single_param_single_value(self, impl) -> None:
        padded, counts = self._padded([[np.array([4.0])], [np.array([5.0, 6.0])]], 3)
        cart, origins = impl(padded, counts, 2, 1)
        np.testing.assert_array_equal(cart, [[4.0], [5.0], [6.0]])
        np.testing.assert_array_equal(origins, [0, 1, 1])


class TestRowVectorUnique:
    @pytest.mark.parametrize("impl", jit_variants(np_utils.numba_bf_row_vector_unique))
    def test_first_occurrence_order(self, impl) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [5.0, 6.0], [3.0, 4.0]])
        np.testing.assert_array_equal(impl(arr), [[1, 2], [3, 4], [5, 6]])

    @pytest.mark.parametrize("impl", jit_variants(np_utils.numba_bf_row_vector_unique))
    def test_tolerance_merges_near_duplicates(self, impl) -> None:
        arr = np.array([[1.0, 2.0], [1.0 + 1e-10, 2.0], [1.0 + 1e-6, 2.0]])
        # 1e-10 is inside the 1e-8 tolerance, 1e-6 is outside.
        np.testing.assert_array_equal(impl(arr), arr[[0, 2]])

    @pytest.mark.parametrize("impl", jit_variants(np_utils.numba_bf_row_vector_unique))
    def test_matches_brute_force(self, impl) -> None:
        rng = np.random.default_rng(41)
        arr = rng.integers(0, 3, (60, 3)).astype(np.float64)
        _, first = np.unique(arr, axis=0, return_index=True)
        np.testing.assert_array_equal(impl(arr), arr[np.sort(first)])

    @pytest.mark.parametrize("impl", jit_variants(np_utils.numba_bypass_all_close))
    def test_all_close(self, impl) -> None:
        a = np.array([1.0, 2.0, 3.0])
        assert impl(a, a + 5e-9)
        assert not impl(a, a + np.array([0.0, 2e-8, 0.0]))
        assert impl(a, a + 0.05, 0.1)


class TestPadding:
    @pytest.mark.parametrize("impl", jit_variants(np_utils.cpadpow2))
    @pytest.mark.parametrize("nbins", [1, 2, 3, 5, 8, 9, 31, 64, 100])
    def test_cpadpow2(self, impl, nbins: int) -> None:
        arr = np.random.default_rng(nbins).standard_normal((3, nbins))
        out = impl(arr)
        n_out = out.shape[-1]
        assert n_out >= nbins
        assert n_out & (n_out - 1) == 0
        assert n_out < 2 * nbins or nbins == 1
        np.testing.assert_array_equal(out[:, :nbins], arr)
        # The padding is circular: it continues the profile from its start.
        np.testing.assert_array_equal(out[:, nbins:], arr[:, : n_out - nbins])

    @pytest.mark.parametrize("impl", jit_variants(np_utils.cpad2len))
    @pytest.mark.parametrize("size", [4, 7, 16])
    def test_cpad2len(self, impl, size: int) -> None:
        arr = np.random.default_rng(size).standard_normal((2, 3, 4))
        out = impl(arr, size)
        assert out.shape == (2, 3, size)
        np.testing.assert_array_equal(out[..., :4], arr)
        np.testing.assert_array_equal(out[..., 4:], 0.0)

    def test_pad_with_inf(self) -> None:
        params = [np.array([1.0, 2.0, 3.0]), np.array([4.0]), np.array([5.0, 6.0])]
        out = np_utils.pad_with_inf(params)
        assert out.shape == (3, 3)
        for row, arr in zip(out, params, strict=True):
            np.testing.assert_array_equal(row[: len(arr)], arr)
            assert np.isposinf(row[len(arr) :]).all()


class TestInterpolateMissing:
    @pytest.mark.parametrize("impl", jit_variants(np_utils.interpolate_missing))
    def test_fills_gaps_linearly(self, impl) -> None:
        profile = np.array([0.0, -1.0, -1.0, 3.0, -1.0, 5.0, -1.0])
        count = np.array([1, 0, 0, 1, 0, 1, 0])
        out = impl(profile.copy(), count)
        # Interior gaps are linear; the trailing gap holds the last value.
        np.testing.assert_allclose(out, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0])

    @pytest.mark.parametrize("impl", jit_variants(np_utils.interpolate_missing))
    def test_filled_bins_untouched(self, impl) -> None:
        rng = np.random.default_rng(51)
        profile = rng.standard_normal(40)
        count = rng.integers(0, 3, 40)
        out = impl(profile.copy(), count)
        np.testing.assert_array_equal(out[count > 0], profile[count > 0])

    @pytest.mark.parametrize("impl", jit_variants(np_utils.interpolate_missing))
    def test_all_empty_is_unchanged(self, impl) -> None:
        profile = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(
            impl(profile.copy(), np.zeros(3, np.int64)), profile
        )


class TestLstsqWeighted:
    """Contracts run on ``.py_func`` only: the compiled kernel does not compile.

    It passes ``rcond=None`` to `np.linalg.lstsq`, which numba's overload rejects for
    every input (`TypingError`); nothing in the library calls it, so nothing noticed.
    """

    @pytest.mark.xfail(strict=True, reason="#23", raises=TypingError)
    def test_compiled_kernel_compiles(self) -> None:
        design = np.vstack([np.ones(5), np.arange(5.0)]).T
        np_utils.lstsq_weighted(design, np.arange(5.0), np.ones(5))

    @pytest.mark.parametrize("impl", [np_utils.lstsq_weighted.py_func])
    def test_recovers_exact_solution(self, impl) -> None:
        rng = np.random.default_rng(61)
        t = np.linspace(-1, 1, 30)
        design = np.vstack([np.ones_like(t), t, t**2]).T
        x_true = np.array([0.3, -1.2, 2.5])
        errors = rng.uniform(0.5, 2.0, t.size)
        x_hat, cov, fitted = impl(design, design @ x_true, errors)
        np.testing.assert_allclose(x_hat, x_true, rtol=1e-10)
        np.testing.assert_allclose(fitted, design @ x_true, rtol=1e-10)
        w = np.diag(errors**-2)
        np.testing.assert_allclose(
            cov, np.linalg.inv(design.T @ w @ design), rtol=1e-10
        )

    @pytest.mark.parametrize("impl", [np_utils.lstsq_weighted.py_func])
    def test_weights_downweight_outlier(self, impl) -> None:
        t = np.linspace(0, 1, 11)
        design = np.vstack([np.ones_like(t), t]).T
        data = 2.0 + 3.0 * t
        data[5] += 10.0
        errors = np.ones_like(t)
        errors[5] = 1e6
        x_hat, _, _ = impl(design, data, errors)
        np.testing.assert_allclose(x_hat, [2.0, 3.0], atol=1e-9)

    @pytest.mark.parametrize("impl", [np_utils.lstsq_weighted.py_func])
    def test_rank_deficient_uses_pinv(self, impl) -> None:
        t = np.linspace(-1, 1, 20)
        design = np.vstack([t, 2 * t]).T  # rank 1
        errors = np.ones_like(t)
        x_hat, cov, fitted = impl(design, 3.0 * t, errors)
        assert np.isfinite(cov).all()
        np.testing.assert_allclose(cov, np.linalg.pinv(design.T @ design), atol=1e-10)
        np.testing.assert_allclose(fitted, 3.0 * t, atol=1e-10)
        # Minimum-norm solution: x ∝ (1, 2) with x1 + 2 x2 = 3.
        np.testing.assert_allclose(x_hat, [0.6, 1.2], atol=1e-10)


class TestDetermineRefSegs:
    @pytest.mark.parametrize(
        "func",
        [
            np_utils.determine_ref_segs,
            np_utils.determine_ref_segs_pareto,
        ],
    )
    @pytest.mark.parametrize(("nsegments", "n_runs"), [(8, 0), (8, 9), (1, 2)])
    def test_rejects_out_of_range(self, func, nsegments: int, n_runs: int) -> None:
        with pytest.raises(ValueError, match="n_runs"):
            func(nsegments, n_runs)

    def test_plain_spans_both_ends(self) -> None:
        for nsegments in range(1, 70):
            for n_runs in range(1, nsegments + 1):
                segs = np_utils.determine_ref_segs(nsegments, n_runs)
                assert len(segs) == n_runs
                assert segs[0] == 0
                assert np.all(np.diff(segs) > 0)
                if n_runs > 1:
                    assert segs[-1] == nsegments - 1

    def test_pareto_special_cases(self) -> None:
        """Both special cases are stated in the docstring."""
        for nsegments in range(1, 70):
            assert np_utils.determine_ref_segs_pareto(nsegments, 1) == [nsegments // 2]
            assert np_utils.determine_ref_segs_pareto(nsegments, nsegments) == list(
                range(nsegments),
            )

    def test_pareto_anchors_in_range_and_sorted(self) -> None:
        for nsegments in range(1, 70):
            for n_runs in range(2, nsegments + 1):
                segs = np_utils.determine_ref_segs_pareto(nsegments, n_runs)
                assert len(segs) == n_runs
                assert segs[0] >= 0
                assert segs[-1] <= nsegments - 1
                assert np.all(np.diff(segs) >= 0)

    def test_pareto_anchors_distinct_below_nsegments_minus_one(self) -> None:
        """No two runs share a reference segment.

        Holds for every ``n_runs`` except ``nsegments - 1``; that case is pinned below.
        """
        for nsegments in range(1, 200):
            for n_runs in range(1, nsegments - 1):
                segs = np_utils.determine_ref_segs_pareto(nsegments, n_runs)
                assert len(set(segs)) == n_runs, (nsegments, n_runs, segs)

    @pytest.mark.xfail(strict=True, reason="#25", raises=AssertionError)
    @pytest.mark.parametrize("nsegments", [3, 4, 16, 64])
    def test_pareto_anchors_distinct_at_nsegments_minus_one(
        self, nsegments: int
    ) -> None:
        """Pin two runs landing on one segment at ``n_runs == nsegments - 1``.

        ``margin = round((M - 1) / (2 (M - 1)))`` rounds 1/2 up to 1, so the anchors
        are squeezed into ``M - 2`` slots: e.g. ``(3, 2) -> [1, 1]``,
        ``(4, 3) -> [1, 2, 2]``. `prune.py` feeds a user's ``n_runs`` straight in.
        """
        segs = np_utils.determine_ref_segs_pareto(nsegments, nsegments - 1)
        assert len(set(segs)) == nsegments - 1, segs
