"""Unit tests for `pyloki.core.fold`, the brute-force folding kernels.

These build the initial fold every search starts from: each segment of the time
series folded at each trial frequency, in the time domain (`brutefold*`) or directly
as Fourier coefficients (`brutefold_start_complex*`). Everything downstream is sums and
rotations of these folds, so an error here propagates through the whole search.

Contracts, each against an independent reference:

- the time-domain folds equal an ``np.add.at`` accumulation into the phase bins that
  `psr_utils.get_phase_idx_int` assigns (itself tested in `test_psr_utils.py`), and
  conserve each segment's sum;
- `brutefold_bucketed` equals `brutefold_start`, which it reorders for speed;
- the complex folds equal a float64 DFT of each segment at the harmonics of each trial
  frequency, ``sum_t x[t] exp(-2 pi i m f (t tsamp - t_ref))``;
- `brutefold_complex_oversampled` equals ``rfft`` of the finely binned bucketed fold;
- `ffa_taylor_init*` pick the reference time the FFA levels assume, and
  `ffa_taylor_resolve` the closed-form phase and grid cell of a shifted parameter set.

Parallel kernels
----------------
Four kernels use ``parallel=True``/``prange``. Their ``py_func`` runs the loop serially,
so it cannot see a race. Each is therefore also run compiled at one thread and at every
available thread, and the two outputs must be identical: every ``prange`` iteration
writes only its own segment's (or frequency's) slice and there are no cross-iteration
reductions, so a correct kernel is deterministic regardless of thread count. The check
is repeated on inputs with many more segments than threads, with and without a partial
last segment where the kernel accepts one, and skips (visibly) with fewer than two
threads. It is evidence, not a proof: it catches a race only if one happens during the
test, at the thread counts and sizes used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np
import pytest
from numba import typed

from pyloki.core import fold
from pyloki.utils import psr_utils
from pyloki.utils.misc import C_VAL
from tests.jit_utils import jit_variants

if TYPE_CHECKING:
    from collections.abc import Iterator

TSAMP, SEGLEN, NBINS, NSEG = 1e-3, 512, 32, 6
FREQS = np.array([7.3, 11.05, 20.0])
T_REFS = [0.0, SEGLEN * TSAMP / 2]


def _series(nsamples: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ts_e = rng.standard_normal(nsamples).astype(np.float32)
    ts_v = rng.uniform(0.5, 1.5, nsamples).astype(np.float32)
    return ts_e, ts_v


def _reference_fold(
    ts: np.ndarray,
    freq: float,
    seglen: int,
    nbins: int,
    t_ref: float,
) -> np.ndarray:
    """Return shape (nsegments, nbins), with the last segment possibly partial."""
    nseg = -(-len(ts) // seglen)
    phases = psr_utils.get_phase_idx_int(
        np.arange(seglen) * TSAMP - t_ref, freq, nbins, 0
    )
    out = np.zeros((nseg, nbins), dtype=np.float64)
    for s in range(nseg):
        chunk = ts[s * seglen : (s + 1) * seglen]
        np.add.at(out[s], phases[: len(chunk)], chunk)
    return out


def _reference_dft(ts: np.ndarray, freq: float, t_ref: float) -> np.ndarray:
    """Return the DFT of each segment at harmonics of freq, (nsegments, nbins_f)."""
    nbins_f = NBINS // 2 + 1
    times = np.arange(SEGLEN) * TSAMP - t_ref
    basis = np.exp(-2j * np.pi * np.outer(np.arange(nbins_f), freq * times))
    return ts.reshape(-1, SEGLEN).astype(np.float64) @ basis.T


def _assert_matches_dft(
    out: np.ndarray,
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    t_ref: float,
) -> None:
    for j, freq in enumerate(FREQS):
        for comp, ts in enumerate((ts_e, ts_v)):
            expected = _reference_dft(ts, freq, t_ref)
            scale = np.abs(ts).reshape(-1, SEGLEN).sum(axis=1).max()
            np.testing.assert_allclose(
                out[:, j, comp],
                expected,
                rtol=0,
                atol=1e-5 * scale,
            )


@pytest.fixture
def restore_threads() -> Iterator[None]:
    yield
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)


class TestBrutefold:
    @pytest.mark.parametrize("impl", jit_variants(fold.brutefold))
    @pytest.mark.parametrize("nsegments", [1, 4, 5])
    def test_matches_reference(self, impl, nsegments: int) -> None:
        """Samples past ``nsegments * (nsamples // nsegments)`` are dropped."""
        ts_e, ts_v = _series(2003)
        proper_time = np.arange(2003) * TSAMP
        out = impl(ts_e, ts_v, proper_time, 9.7, nsegments, NBINS)
        seglen = 2003 // nsegments
        kept = nsegments * seglen
        for comp, ts in enumerate((ts_e, ts_v)):
            phases = psr_utils.get_phase_idx_int(proper_time[:kept], 9.7, NBINS, 0)
            expected = np.zeros((nsegments, NBINS))
            np.add.at(expected, (np.arange(kept) // seglen, phases), ts[:kept])
            np.testing.assert_allclose(out[:, comp], expected, rtol=1e-5, atol=1e-4)

    @pytest.mark.parametrize("impl", jit_variants(fold.brutefold_single))
    def test_single_matches_brutefold(self, impl) -> None:
        ts_e, ts_v = _series(2048)
        proper_time = np.arange(2048) * TSAMP
        np.testing.assert_allclose(
            impl(ts_e, proper_time, 9.7, 4, NBINS),
            fold.brutefold(ts_e, ts_v, proper_time, 9.7, 4, NBINS)[:, 0],
            rtol=1e-6,
        )

    @pytest.mark.parametrize(
        ("impl", "args"),
        [(p.values[0], "full") for p in jit_variants(fold.brutefold)]
        + [(p.values[0], "single") for p in jit_variants(fold.brutefold_single)],
    )
    def test_rejects_length_mismatch(self, impl, args: str) -> None:
        ts_e, ts_v = _series(100)
        short_time = np.arange(99) * TSAMP
        call = (ts_e, ts_v, short_time, 9.7, 2, NBINS)
        if args == "single":
            call = (ts_e, short_time, 9.7, 2, NBINS)
        with pytest.raises(ValueError, match="same length"):
            impl(*call)

    def test_signature_refuses_float64_series(self) -> None:
        ts_e, ts_v = _series(64)
        with pytest.raises(TypeError, match="No matching definition"):
            fold.brutefold(
                ts_e.astype(np.float64),
                ts_v,
                np.arange(64) * TSAMP,
                9.7,
                2,
                NBINS,
            )


class TestBrutefoldStart:
    @pytest.mark.parametrize("impl", jit_variants(fold.brutefold_start))
    @pytest.mark.parametrize("t_ref", T_REFS)
    @pytest.mark.parametrize("nsamples", [NSEG * SEGLEN, NSEG * SEGLEN - 100])
    def test_matches_reference(self, impl, t_ref: float, nsamples: int) -> None:
        """A partial last segment is folded over the samples it has."""
        ts_e, ts_v = _series(nsamples)
        out = impl(ts_e, ts_v, FREQS, SEGLEN, NBINS, TSAMP, t_ref)
        assert out.shape == (NSEG, len(FREQS), 2, NBINS)
        for j, freq in enumerate(FREQS):
            for comp, ts in enumerate((ts_e, ts_v)):
                np.testing.assert_allclose(
                    out[:, j, comp],
                    _reference_fold(ts, freq, SEGLEN, NBINS, t_ref),
                    rtol=1e-5,
                    atol=1e-4,
                )

    @pytest.mark.parametrize("impl", jit_variants(fold.brutefold_start))
    def test_conserves_segment_sums(self, impl) -> None:
        ts_e, ts_v = _series(NSEG * SEGLEN)
        out = impl(ts_e, ts_v, FREQS, SEGLEN, NBINS, TSAMP, 0.0)
        sums = ts_e.reshape(NSEG, SEGLEN).astype(np.float64).sum(axis=1)
        np.testing.assert_allclose(
            out[:, :, 0].sum(axis=-1),
            np.repeat(sums[:, None], len(FREQS), 1),
            atol=1e-3,
        )

    @pytest.mark.parametrize("impl", jit_variants(fold.brutefold_bucketed))
    @pytest.mark.parametrize("t_ref", T_REFS)
    def test_bucketed_matches_start(self, impl, t_ref: float) -> None:
        ts_e, ts_v = _series(NSEG * SEGLEN)
        np.testing.assert_allclose(
            impl(ts_e, ts_v, FREQS, SEGLEN, NBINS, TSAMP, t_ref),
            fold.brutefold_start(ts_e, ts_v, FREQS, SEGLEN, NBINS, TSAMP, t_ref),
            rtol=1e-5,
            atol=1e-4,
        )

    @pytest.mark.parametrize("impl", jit_variants(fold.brutefold_bucketed))
    def test_bucketed_rejects_partial_segment(self, impl) -> None:
        ts_e, ts_v = _series(NSEG * SEGLEN - 100)
        with pytest.raises(ValueError, match="multiple of the segment length"):
            impl(ts_e, ts_v, FREQS, SEGLEN, NBINS, TSAMP, 0.0)


class TestComplexFolds:
    @pytest.mark.parametrize(
        "impl",
        jit_variants(fold.brutefold_start_complex)
        + jit_variants(fold.brutefold_start_complex_opt),
    )
    @pytest.mark.parametrize("t_ref", T_REFS)
    def test_matches_dft(self, impl, t_ref: float) -> None:
        ts_e, ts_v = _series(NSEG * SEGLEN)
        out = impl(ts_e, ts_v, FREQS, SEGLEN, NBINS, TSAMP, t_ref)
        assert out.shape == (NSEG, len(FREQS), 2, NBINS // 2 + 1)
        _assert_matches_dft(out, ts_e, ts_v, t_ref)

    @pytest.mark.parametrize("impl", jit_variants(fold.brutefold_start_complex))
    def test_partial_segment_uses_the_samples_it_has(self, impl) -> None:
        ts_e, ts_v = _series(NSEG * SEGLEN - 100)
        out = impl(ts_e, ts_v, FREQS, SEGLEN, NBINS, TSAMP, 0.0)
        padded_e = np.concatenate([ts_e, np.zeros(100, np.float32)])
        padded_v = np.concatenate([ts_v, np.zeros(100, np.float32)])
        _assert_matches_dft(out, padded_e, padded_v, 0.0)

    @pytest.mark.parametrize("impl", jit_variants(fold.brutefold_start_complex_opt))
    def test_opt_rejects_partial_segment(self, impl) -> None:
        """Reject a partial segment; a search never passes one.

        The config makes ``bseg_brute`` and ``nsamps`` powers of two.
        """
        ts_e, ts_v = _series(NSEG * SEGLEN - 100)
        with pytest.raises(ValueError):  # noqa: PT011 -- numba's reshape message
            impl(ts_e, ts_v, FREQS, SEGLEN, NBINS, TSAMP, 0.0)

    @pytest.mark.parametrize("impl", jit_variants(fold.brutefold_complex_oversampled))
    @pytest.mark.parametrize("oversample", [1, 4])
    def test_oversampled_is_rfft_of_fine_fold(self, impl, oversample: int) -> None:
        ts_e, ts_v = _series(NSEG * SEGLEN)
        out = impl(ts_e, ts_v, FREQS, SEGLEN, NBINS, TSAMP, 0.0, oversample)
        fine = fold.brutefold_bucketed(
            ts_e,
            ts_v,
            FREQS,
            SEGLEN,
            NBINS * oversample,
            TSAMP,
            0.0,
        )
        expected = np.fft.rfft(fine.astype(np.float64), axis=-1)[..., : NBINS // 2 + 1]
        np.testing.assert_allclose(
            out,
            expected,
            rtol=0,
            atol=1e-5 * np.abs(expected).max(),
        )


PARALLEL_KERNELS = [
    pytest.param(fold.brutefold_start, (), id="brutefold_start"),
    pytest.param(fold.brutefold_bucketed, (), id="brutefold_bucketed"),
    pytest.param(fold.brutefold_start_complex, (), id="brutefold_start_complex"),
    pytest.param(fold.brutefold_complex_oversampled, (4,), id="complex_oversampled"),
]
# The two kernels that accept a partial last segment are also run with one.
THREAD_CASES = [
    pytest.param(fold.brutefold_start, (), 0, id="brutefold_start-full"),
    pytest.param(fold.brutefold_start, (), 50, id="brutefold_start-partial"),
    pytest.param(fold.brutefold_bucketed, (), 0, id="brutefold_bucketed-full"),
    pytest.param(fold.brutefold_start_complex, (), 0, id="start_complex-full"),
    pytest.param(fold.brutefold_start_complex, (), 50, id="start_complex-partial"),
    pytest.param(fold.brutefold_complex_oversampled, (4,), 0, id="oversampled-full"),
]


@pytest.mark.usefixtures("restore_threads")
class TestParallelKernels:
    """One thread and every thread must agree exactly; see the module docstring."""

    N_SEGMENTS = 8 * numba.config.NUMBA_NUM_THREADS

    @pytest.mark.parametrize(("kernel", "extra", "trim"), THREAD_CASES)
    def test_thread_count_does_not_change_the_result(
        self,
        kernel,
        extra,
        trim: int,
    ) -> None:
        if numba.config.NUMBA_NUM_THREADS < 2:
            pytest.skip("needs >= 2 numba threads to compare against one")
        ts_e, ts_v = _series(self.N_SEGMENTS * 128 - trim, seed=5)
        freqs = np.linspace(5.0, 40.0, 7)
        args = (ts_e, ts_v, freqs, 128, NBINS, TSAMP, 0.0, *extra)
        numba.set_num_threads(1)
        assert numba.get_num_threads() == 1
        serial = kernel(*args)
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
        assert numba.get_num_threads() >= 2
        for _ in range(3):
            np.testing.assert_array_equal(kernel(*args), serial)

    @pytest.mark.parametrize(("kernel", "extra"), PARALLEL_KERNELS)
    def test_parallel_matches_py_func(self, kernel, extra) -> None:
        ts_e, ts_v = _series(self.N_SEGMENTS * 128, seed=6)
        freqs = np.linspace(5.0, 40.0, 7)
        args = (ts_e, ts_v, freqs, 128, NBINS, TSAMP, 0.0, *extra)
        compiled = kernel(*args)
        python = kernel.py_func(*args)
        np.testing.assert_allclose(
            compiled,
            python,
            rtol=0,
            atol=1e-5 * float(np.abs(python).max()),
        )


class TestFfaInit:
    @pytest.mark.parametrize("impl", jit_variants(fold.ffa_taylor_init))
    @pytest.mark.parametrize(("nparams", "t_ref"), [(1, 0.0), (2, SEGLEN * TSAMP / 2)])
    def test_reference_time(self, impl, nparams: int, t_ref: float) -> None:
        """One parameter folds from the segment start; more fold about its middle."""
        ts_e, ts_v = _series(NSEG * SEGLEN)
        param_arr = typed.List([np.array([0.0])] * (nparams - 1) + [FREQS])
        np.testing.assert_array_equal(
            impl(ts_e, ts_v, param_arr, SEGLEN, NBINS, TSAMP),
            fold.brutefold_start(ts_e, ts_v, FREQS, SEGLEN, NBINS, TSAMP, t_ref),
        )

    @pytest.mark.parametrize("impl", jit_variants(fold.ffa_taylor_init_complex))
    @pytest.mark.parametrize(("nparams", "t_ref"), [(1, 0.0), (2, SEGLEN * TSAMP / 2)])
    def test_complex_reference_time(self, impl, nparams: int, t_ref: float) -> None:
        ts_e, ts_v = _series(NSEG * SEGLEN)
        param_arr = typed.List([np.array([0.0])] * (nparams - 1) + [FREQS])
        out = impl(ts_e, ts_v, param_arr, SEGLEN, NBINS, TSAMP)
        _assert_matches_dft(out, ts_e, ts_v, t_ref)


class TestFfaResolve:
    """Closed forms for the phase and grid cell of a parameter set one level down."""

    TSEG, LEVEL = 0.4, 3
    FREQ_LIMITS = np.array([[90.0, 110.0]])
    FREQ_COUNT = np.array([64])
    ACC_FREQ_LIMITS = np.array([[-20.0, 20.0], [90.0, 110.0]])
    ACC_FREQ_COUNT = np.array([16, 64])

    @staticmethod
    def _cell(value: float, limits: np.ndarray, count: int) -> int:
        lo, hi = limits
        return int(np.clip(np.floor(count * (value - lo) / (hi - lo)), 0, count - 1))

    @pytest.mark.parametrize("impl", jit_variants(fold.ffa_taylor_resolve))
    @pytest.mark.parametrize("latter", [0, 1])
    def test_frequency_only(self, impl, latter: int) -> None:
        freq = 101.37
        pindex, phase = impl(
            np.array([freq]),
            self.FREQ_COUNT,
            self.FREQ_LIMITS,
            self.LEVEL,
            latter,
            self.TSEG,
            NBINS,
        )
        delta_t = latter * 2 ** (self.LEVEL - 1) * self.TSEG
        np.testing.assert_allclose(phase, (delta_t * freq) % 1 * NBINS, atol=1e-9)
        assert pindex[0] == self._cell(freq, self.FREQ_LIMITS[0], 64)

    @pytest.mark.parametrize("impl", jit_variants(fold.ffa_taylor_resolve))
    @pytest.mark.parametrize("latter", [0, 1])
    # 101.5625 is a cell edge of the 64-cell grid on [90, 110]. Just above it, the
    # Doppler-shifted frequency (about 2e-6 Hz lower for latter=1) falls in the cell
    # below, so the test tells the shifted frequency from the unshifted one.
    @pytest.mark.parametrize("freq", [101.37, 101.5625 + 1e-6])
    def test_acceleration_and_frequency(self, impl, latter: int, freq: float) -> None:
        """The reference time moves by ``(latter - 1/2)`` of the previous segment."""
        accel = 7.5
        pindex, phase = impl(
            np.array([accel, freq]),
            self.ACC_FREQ_COUNT,
            self.ACC_FREQ_LIMITS,
            self.LEVEL,
            latter,
            self.TSEG,
            NBINS,
        )
        delta_t = (latter - 0.5) * 2 ** (self.LEVEL - 1) * self.TSEG
        velocity = accel * delta_t
        delay = accel * delta_t**2 / 2 / C_VAL
        freq_new = freq * (1 - velocity / C_VAL)
        np.testing.assert_allclose(
            phase,
            ((delta_t - delay) * freq) % 1 * NBINS,
            atol=1e-9,
        )
        assert pindex[0] == self._cell(accel, self.ACC_FREQ_LIMITS[0], 16)
        assert pindex[1] == self._cell(freq_new, self.ACC_FREQ_LIMITS[1], 64)
