"""Unit tests for `pyloki.core.common`, the fold-combining kernels.

These are the innermost operations of both the FFA and the pruning search: build the
leaf parameter grid (`get_leaves*`), pick a fold out of the dynamic-programming
array (`load_folds_*`, `load_prune_folds_*`), and add two folds after rotating each by
a phase shift (`shift_add*`, `shift_3d*`). A wrong rotation does not raise; it smears
a pulse across bins and costs sensitivity silently.

Contracts, each against an independent reference:

- every real-space shift equals `np.roll` by the shift rounded half up. Phase shifts
  reach these kernels from `psr_utils.get_phase_idx`, so they lie in ``[0, nbins]``, and
  a shift that rounds to ``nbins`` wraps to 0;
- every complex kernel acts on an ``rfft`` spectrum as the Fourier twin of that roll:
  multiplication by ``exp(-2 pi i k s / nbins)``, computed directly. For
  `shift_add_complex` at integer shifts, the result is also compared with the ``rfft``
  of the rolled sum;
- the batch kernels equal their scalar counterparts row by row;
- the loaders index the fold arrays the FFA and pruning pass them. The prune loaders
  take every axis above acceleration at index 0, because pruning resolves only
  ``(accel, freq)`` in the initial grid (`core.taylor.poly_taylor_resolve_batch`);
- the explicit numba signatures are part of the contract: the explicitly typed kernels
  refuse float64 and non-contiguous input, which their ``py_func`` would accept. That
  contract is compiled-only by construction; every behavioural one runs on both paths.

Tolerances for the complex kernels are relative to the spectrum's scale. They
accumulate the phase by complex64 recurrence, so they are good to a few parts in 1e6
of the largest coefficient, not to machine precision.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from numba import typed

from pyloki.core import common
from tests.jit_utils import jit_variants

NBINS = 64
NBINS_F = NBINS // 2 + 1

# The loaders are chosen at run time by `set_*_load_func`, so they are parametrised by
# path rather than through `jit_variants`.
PATHS = [
    pytest.param(lambda f: f, id="compiled"),
    pytest.param(lambda f: f.py_func, id="py_func"),
]


def _round_shift(shift: float, nbins: int = NBINS) -> int:
    """Round a shift as the kernels do: half up in float32, ``nbins`` wraps to 0."""
    return math.floor(np.float32(shift) + np.float32(0.5)) % nbins


def _profiles(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 2, NBINS)).astype(np.float32)


def _spectra(real: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.fft.rfft(real, axis=-1).astype(np.complex64))


def _phase(shift: float) -> np.ndarray:
    return np.exp(-2j * np.pi * np.arange(NBINS_F) * shift / NBINS)


def _complex_atol(spectrum: np.ndarray) -> float:
    return 3e-5 * float(np.abs(spectrum).max())


# Shifts spanning the domain: integers, halves either side, the top edge that wraps.
SHIFTS = [
    (0.0, 0.0),
    (3.2, 7.5),
    (63.6, 0.49),
    (64.0, 12.5),
    (10.4999, 20.5001),
    (0.5, 63.5),
]


class TestGetLeaves:
    PARAMS = (np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0]), np.array([6.0]))
    DPARAMS = np.array([0.1, 0.2, 0.3])

    def _expected(self) -> np.ndarray:
        return np.array(
            [
                [[value, step] for value, step in zip(row, self.DPARAMS, strict=True)]
                for row in itertools.product(*self.PARAMS)
            ]
        )

    @pytest.mark.parametrize("impl", jit_variants(common.get_leaves))
    def test_matches_itertools(self, impl) -> None:
        np.testing.assert_array_equal(
            impl(typed.List(self.PARAMS), self.DPARAMS),
            self._expected(),
        )

    @pytest.mark.parametrize("impl", jit_variants(common.get_leaves_opt))
    def test_opt_has_the_same_leaves(self, impl) -> None:
        """Same leaf set as `get_leaves`, but not in the same order.

        `get_leaves_opt` varies the first parameter fastest and `get_leaves` the
        last, so only the set is compared.
        """
        got = impl(typed.List(self.PARAMS), self.DPARAMS)
        assert got.shape == self._expected().shape

        def rows(leaves: np.ndarray) -> list[tuple[float, ...]]:
            return sorted(map(tuple, leaves.reshape(len(leaves), -1)))

        assert rows(got) == rows(self._expected())

    @pytest.mark.parametrize("impl", jit_variants(common.get_leaves_opt))
    def test_opt_empty_axis(self, impl) -> None:
        # The library passes a numba typed.List (see ffa.py and the dyn_* modules).
        params = typed.List([np.array([1.0]), np.array([], dtype=np.float64)])
        assert impl(params, np.array([0.1, 0.2])).shape == (0, 2, 2)


class TestLoadFolds:
    @staticmethod
    def _fold(nparams: int, *, with_segments: bool) -> np.ndarray:
        shape = (3,) * nparams + (2, 5)
        if with_segments:
            shape = (4, *shape)
        return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    @pytest.mark.parametrize("path", PATHS)
    @pytest.mark.parametrize("nparams", [1, 2, 3, 4, 5])
    def test_ffa_loader_indexes_segment_and_params(self, path, nparams: int) -> None:
        fold = self._fold(nparams, with_segments=True)
        param_idx = np.array([2, 0, 1, 2, 1][:nparams])
        load = path(common.set_ffa_load_func(nparams))
        np.testing.assert_array_equal(load(fold, 3, param_idx), fold[(3, *param_idx)])

    @pytest.mark.parametrize("path", PATHS)
    @pytest.mark.parametrize("nparams", [1, 2, 3, 4, 5])
    def test_prune_loader_single(self, path, nparams: int) -> None:
        """Axes above acceleration are taken at index 0 (see the module docstring)."""
        fold = self._fold(nparams, with_segments=False)
        param_idx = np.array([2, 0, 1, 2, 1][:nparams])
        leading = (0,) * max(nparams - 2, 0)
        expected = fold[(*leading, *param_idx[-min(nparams, 2) :])]
        load = path(common.set_prune_load_func(nparams))
        np.testing.assert_array_equal(load(fold, param_idx), expected)

    @pytest.mark.parametrize("path", PATHS)
    @pytest.mark.parametrize("nparams", [1, 2, 3, 4, 5])
    def test_prune_loader_batch_matches_single(self, path, nparams: int) -> None:
        fold = self._fold(nparams, with_segments=False)
        rng = np.random.default_rng(nparams)
        batch_idx = rng.integers(0, 3, (6, nparams))
        load = path(common.set_prune_load_func(nparams))
        got = load(fold, batch_idx)
        assert got.shape == (6, 2, 5)
        for row, idx in zip(got, batch_idx, strict=True):
            np.testing.assert_array_equal(row, load(fold, idx))

    @pytest.mark.parametrize(
        "setter",
        [common.set_ffa_load_func, common.set_prune_load_func],
    )
    @pytest.mark.parametrize("nparams", [0, 6])
    def test_setters_reject_unsupported_dimension(self, setter, nparams: int) -> None:
        with pytest.raises(KeyError):
            setter(nparams)


class TestTrivialKernels:
    @pytest.mark.parametrize("impl", jit_variants(common.add))
    def test_add(self, impl) -> None:
        a, b = _profiles(1, 1)[0], _profiles(1, 2)[0]
        np.testing.assert_array_equal(impl(a, b), a + b)

    @pytest.mark.parametrize("impl", jit_variants(common.pack))
    def test_pack_is_identity(self, impl) -> None:
        a = _profiles(1, 3)[0]
        assert impl(a) is a or np.array_equal(impl(a), a)

    @pytest.mark.parametrize("impl", jit_variants(common.shift))
    @pytest.mark.parametrize("phase_shift", [0, 1, 17, -5, 70])
    def test_shift_is_roll_on_last_axis(self, impl, phase_shift: int) -> None:
        a = _profiles(3, 4)
        np.testing.assert_array_equal(impl(a, phase_shift), np.roll(a, phase_shift, -1))


class TestShiftAdd:
    @pytest.mark.parametrize("impl", jit_variants(common.shift_add))
    @pytest.mark.parametrize(("shift_tail", "shift_head"), SHIFTS)
    def test_is_rolled_sum(self, impl, shift_tail: float, shift_head: float) -> None:
        tail, head = _profiles(2, 5)
        expected = np.roll(tail, _round_shift(shift_tail), -1) + np.roll(
            head,
            _round_shift(shift_head),
            -1,
        )
        np.testing.assert_array_equal(
            impl(tail, head, shift_tail, shift_head), expected
        )

    @pytest.mark.parametrize("impl", jit_variants(common.shift_add_complex))
    @pytest.mark.parametrize(("shift_tail", "shift_head"), [(3, 7), (0, 63), (12, 40)])
    def test_complex_integer_shift_is_rolled_sum(
        self,
        impl,
        shift_tail: int,
        shift_head: int,
    ) -> None:
        tail, head = _profiles(2, 6)
        got = impl(_spectra(tail), _spectra(head), float(shift_tail), float(shift_head))
        # An integer roll is exact, so its spectrum is an exact reference.
        rolled = np.roll(tail, shift_tail, -1) + np.roll(head, shift_head, -1)
        expected = np.fft.rfft(rolled.astype(np.float64), axis=-1)
        np.testing.assert_allclose(got, expected, rtol=0, atol=_complex_atol(expected))

    @pytest.mark.parametrize(
        "impl",
        jit_variants(common.shift_add_complex)
        + jit_variants(common.shift_add_complex_direct),
    )
    @pytest.mark.parametrize(("shift_tail", "shift_head"), SHIFTS)
    def test_complex_matches_exponential(
        self,
        impl,
        shift_tail: float,
        shift_head: float,
    ) -> None:
        tail, head = _spectra(_profiles(1, 7)[0]), _spectra(_profiles(1, 8)[0])
        expected = tail * _phase(np.float32(shift_tail)) + head * _phase(
            np.float32(shift_head),
        )
        np.testing.assert_allclose(
            impl(tail, head, shift_tail, shift_head),
            expected,
            rtol=0,
            atol=_complex_atol(expected),
        )


class TestSignatureContract:
    """Explicit signatures are enforced by the compiled kernel only."""

    def test_real_kernels_refuse_float64(self) -> None:
        tail, head = _profiles(2, 9).astype(np.float64)
        with pytest.raises(TypeError, match="No matching definition"):
            common.shift_add(tail, head, 1.0, 2.0)
        batch = _profiles(3, 9).astype(np.float64)
        with pytest.raises(TypeError, match="No matching definition"):
            common.shift_3d_batch(batch, np.zeros(3))

    def test_real_kernels_refuse_non_contiguous(self) -> None:
        wide = np.zeros((2, 2 * NBINS), dtype=np.float32)
        with pytest.raises(TypeError, match="No matching definition"):
            common.shift_add(wide[:, ::2], wide[:, ::2], 1.0, 2.0)

    def test_complex_kernels_refuse_complex128(self) -> None:
        spec = _spectra(_profiles(1, 10)[0]).astype(np.complex128)
        with pytest.raises(TypeError, match="No matching definition"):
            common.shift_add_complex(spec, spec, 1.0, 2.0)

    def test_py_func_accepts_what_compiled_refuses(self) -> None:
        tail, head = _profiles(2, 11).astype(np.float64)
        out = common.shift_add.py_func(tail, head, 1.0, 2.0)
        assert out.dtype == np.float64


class TestBatchKernels:
    N_BATCH = 7

    def _inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(12)
        segments = _profiles(self.N_BATCH, 13)
        folds = _profiles(3, 14)
        shifts = np.concatenate([[0.0, NBINS - 0.2, 0.5], rng.uniform(0, NBINS, 4)])
        isuggest = rng.integers(0, 3, self.N_BATCH)
        return segments, shifts, folds, isuggest

    @pytest.mark.parametrize("impl", jit_variants(common.shift_add_batch))
    def test_shift_add_batch(self, impl) -> None:
        segments, shifts, folds, isuggest = self._inputs()
        got = impl(segments, shifts, folds, isuggest)
        for i in range(self.N_BATCH):
            expected = folds[isuggest[i]] + np.roll(
                segments[i],
                _round_shift(shifts[i]),
                -1,
            )
            np.testing.assert_array_equal(got[i], expected)

    @pytest.mark.parametrize("impl", jit_variants(common.shift_3d_batch))
    def test_shift_3d_batch(self, impl) -> None:
        segments, shifts, _, _ = self._inputs()
        got = impl(segments, shifts)
        for i in range(self.N_BATCH):
            np.testing.assert_array_equal(
                got[i],
                np.roll(segments[i], _round_shift(shifts[i]), -1),
            )

    @pytest.mark.parametrize("impl", jit_variants(common.shift_add_complex_batch))
    def test_shift_add_complex_batch(self, impl) -> None:
        segments, shifts, folds, isuggest = self._inputs()
        seg_f, folds_f = _spectra(segments), _spectra(folds)
        got = impl(seg_f, shifts, folds_f, isuggest)
        for i in range(self.N_BATCH):
            expected = folds_f[isuggest[i]] + seg_f[i] * _phase(np.float32(shifts[i]))
            np.testing.assert_allclose(
                got[i],
                expected,
                rtol=0,
                atol=_complex_atol(expected),
            )

    @pytest.mark.parametrize("impl", jit_variants(common.shift_3d_complex_batch))
    def test_shift_3d_complex_batch(self, impl) -> None:
        segments, shifts, _, _ = self._inputs()
        seg_f = _spectra(segments)
        got = impl(seg_f, shifts)
        for i in range(self.N_BATCH):
            expected = seg_f[i] * _phase(np.float32(shifts[i]))
            np.testing.assert_allclose(
                got[i],
                expected,
                rtol=0,
                atol=_complex_atol(expected),
            )


class TestAscendBatch:
    """Sum, over segments, of each segment's loaded fold rotated by its shift."""

    N_BATCH, N_SEG_TOTAL, N_ACC, N_FREQ = 4, 5, 3, 6

    def _inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(21)
        shape = (self.N_SEG_TOTAL, self.N_ACC, self.N_FREQ, 2, NBINS)
        dyp = rng.standard_normal(shape).astype(np.float32)
        idx_segments = np.array([0, 2, 3])
        param_idx = np.stack(
            [
                rng.integers(0, self.N_ACC, (self.N_BATCH, 3)),
                rng.integers(0, self.N_FREQ, (self.N_BATCH, 3)),
            ],
            axis=-1,
        )
        shifts = rng.uniform(0, NBINS, (self.N_BATCH, 3))
        shifts[0, 0] = NBINS - 0.1
        return dyp, idx_segments, param_idx, shifts

    @pytest.mark.parametrize("impl", jit_variants(common.shift_add_ascend_batch))
    def test_real(self, impl) -> None:
        dyp, idx_segments, param_idx, shifts = self._inputs()
        got = impl(dyp, common.load_prune_folds_2d, idx_segments, param_idx, shifts)
        for i in range(self.N_BATCH):
            expected = np.zeros((2, NBINS), dtype=np.float32)
            for s, iseg in enumerate(idx_segments):
                row = dyp[iseg, param_idx[i, s, 0], param_idx[i, s, 1]]
                expected += np.roll(row, _round_shift(shifts[i, s]), -1)
            np.testing.assert_allclose(got[i], expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize(
        "impl",
        jit_variants(common.shift_add_ascend_complex_batch),
    )
    def test_complex(self, impl) -> None:
        dyp, idx_segments, param_idx, shifts = self._inputs()
        dyp_f = _spectra(dyp)
        got = impl(dyp_f, common.load_prune_folds_2d, idx_segments, param_idx, shifts)
        for i in range(self.N_BATCH):
            expected = np.zeros((2, NBINS_F), dtype=np.complex128)
            for s, iseg in enumerate(idx_segments):
                row = dyp_f[iseg, param_idx[i, s, 0], param_idx[i, s, 1]]
                expected += row * _phase(np.float32(shifts[i, s]))
            np.testing.assert_allclose(
                got[i],
                expected,
                rtol=0,
                atol=_complex_atol(expected),
            )
