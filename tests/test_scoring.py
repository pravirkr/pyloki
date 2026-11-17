import numpy as np
import pytest

from pyloki.detection import scoring
from pyloki.detection.scoring import MatchedFilter
from tests.jit_utils import jit_variants

NBINS = 64
WIDTHS = np.array([1, 2, 4, 8])


def _templates(nbins: int, width: int) -> np.ndarray:
    """Every circular shift of the zero-mean, unit-norm boxcar of `width` bins."""
    height = np.sqrt((nbins - width) / (nbins * width))
    b = width * height / (nbins - width)
    base = np.full(nbins, -b)
    base[:width] = height
    return np.array([np.roll(base, i) for i in range(nbins)])


def _boxcar_snr_ref(profile: np.ndarray, widths: np.ndarray, std: float = 1.0):
    """Brute force: best correlation with any shifted template, per width."""
    x = profile.astype(np.float64)
    return np.array([(_templates(len(x), w) @ x).max() / std for w in widths])


def _profiles(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, NBINS)).astype(np.float32)


class TestBoxWidthTrials:
    @pytest.mark.parametrize("impl", jit_variants(scoring.generate_box_width_trials))
    def test_known_ladder(self, impl) -> None:
        # wmax = 0.2 * 64 = 12; each step is max(w + 1, int(1.5 w)); 13 > 12 stops.
        np.testing.assert_array_equal(impl(64, 0.2, 1.5), [1, 2, 3, 4, 6, 9])

    @pytest.mark.parametrize("impl", jit_variants(scoring.generate_box_width_trials))
    @pytest.mark.parametrize(
        ("nbins", "ducy_max", "wtsp"), [(32, 0.5, 1.2), (8, 0.01, 2.0)]
    )
    def test_ladder_invariants(self, impl, nbins: int, ducy_max: float, wtsp: float):
        widths = impl(nbins, ducy_max, wtsp)
        wmax = max(1, int(ducy_max * nbins))
        assert widths[0] == 1
        assert np.all(np.diff(widths) > 0)
        assert widths[-1] <= wmax
        next_width = max(widths[-1] + 1, int(wtsp * widths[-1]))
        assert next_width > wmax, "the ladder stopped before reaching wmax"


class TestCircularPrefixSum:
    @pytest.mark.parametrize("impl", jit_variants(scoring.circular_prefix_sum))
    @pytest.mark.parametrize("nsum", [5, 16, 21, 48, 50])  # <, ==, >, k*n, k*n+r
    def test_matches_cumsum_of_the_tiled_profile(self, impl, nsum: int) -> None:
        x = np.random.default_rng(0).standard_normal(16).astype(np.float32)
        out = np.full(nsum, np.nan, dtype=np.float32)
        impl(x, out)
        want = np.cumsum(np.tile(x.astype(np.float64), 4))[:nsum]
        np.testing.assert_allclose(out, want, rtol=1e-5, atol=1e-5)


class TestDiffMax:
    @pytest.mark.parametrize(
        "impl",
        jit_variants(scoring.diff_max) + jit_variants(scoring.diff_max_buffer),
    )
    def test_is_the_max_of_the_difference(self, impl) -> None:
        rng = np.random.default_rng(1)
        x, y = rng.standard_normal((2, 37)).astype(np.float32)
        assert impl(x, y) == pytest.approx(float(np.max(x - y)), rel=1e-6)


class TestBoxcarSnr:
    @pytest.mark.parametrize("impl", jit_variants(scoring.boxcar_snr_1d))
    @pytest.mark.parametrize("std", [1.0, 2.5])
    def test_1d_matches_brute_force(self, impl, std: float) -> None:
        widths = scoring.generate_box_width_trials(NBINS, 0.2, 1.5)
        for profile in _profiles(5, seed=2):
            np.testing.assert_allclose(
                impl(profile, widths, std),
                _boxcar_snr_ref(profile, widths, std),
                rtol=1e-4,
                atol=1e-4,
            )

    @pytest.mark.parametrize("impl", jit_variants(scoring.boxcar_snr_1d))
    @pytest.mark.parametrize("start", [0, 30, 61])  # 61: the pulse wraps around
    def test_noiseless_pulse_scores_analytically(self, impl, start: int) -> None:
        """A width-w top hat of amplitude A scores A * sqrt(w (n - w) / n) at w."""
        amp, width = 3.0, 4
        profile = np.zeros(NBINS, dtype=np.float32)
        profile[(start + np.arange(width)) % NBINS] = amp
        snr = impl(profile, WIDTHS, 1.0)
        want = amp * np.sqrt(width * (NBINS - width) / NBINS)
        assert snr[list(WIDTHS).index(width)] == pytest.approx(want, rel=1e-5)
        assert np.argmax(snr) == list(WIDTHS).index(width)

    @pytest.mark.parametrize(
        "impl",
        jit_variants(scoring.boxcar_snr_2d)
        + jit_variants(scoring.boxcar_snr_2d_serial),
    )
    def test_2d_is_row_by_row_1d(self, impl) -> None:
        folds = _profiles(9, seed=3)
        want = np.array([scoring.boxcar_snr_1d(f, WIDTHS, 1.5) for f in folds])
        np.testing.assert_array_equal(impl(folds, WIDTHS, 1.5), want)

    @pytest.mark.parametrize("impl", jit_variants(scoring.boxcar_snr_nd))
    def test_nd_keeps_the_leading_shape(self, impl) -> None:
        data = _profiles(6, seed=4).reshape(2, 3, NBINS)
        got = impl(data, WIDTHS, 1.0)
        assert got.shape == (2, 3, len(WIDTHS))
        want = scoring.boxcar_snr_2d(data.reshape(-1, NBINS), WIDTHS, 1.0)
        np.testing.assert_array_equal(got.reshape(-1, len(WIDTHS)), want)

    @pytest.mark.parametrize("impl", jit_variants(scoring.boxcar_snr_beta_1d))
    def test_beta_matches_brute_force_with_weights(self, impl) -> None:
        rng = np.random.default_rng(5)
        ts_e = rng.standard_normal(NBINS).astype(np.float32)
        ts_v = rng.uniform(0.5, 2.0, NBINS).astype(np.float32)
        want = [
            np.max((t @ ts_e) / np.sqrt(t**2 @ ts_v))
            for t in (_templates(NBINS, w) for w in WIDTHS)
        ]
        np.testing.assert_allclose(impl(ts_e, ts_v, WIDTHS), want, rtol=1e-4)

    @pytest.mark.parametrize("impl", jit_variants(scoring.boxcar_snr_beta_1d))
    def test_beta_with_unit_weights_is_the_plain_boxcar(self, impl) -> None:
        ts_e = _profiles(1, seed=6)[0]
        ones = np.ones(NBINS, dtype=np.float32)
        np.testing.assert_allclose(
            impl(ts_e, ones, WIDTHS),
            scoring.boxcar_snr_1d(ts_e, WIDTHS, 1.0),
            rtol=1e-5,
            atol=1e-5,
        )


class TestSnrScoreBatch:
    @staticmethod
    def _batch(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        batch = rng.standard_normal((7, 2, NBINS)).astype(np.float32)
        batch[:, 1] = np.abs(batch[:, 1]) + 0.5  # variance channel
        return batch

    @pytest.mark.parametrize("impl", jit_variants(scoring.snr_score_batch_func))
    def test_is_the_best_boxcar_of_the_normalised_fold(self, impl) -> None:
        batch = self._batch(7)
        want = [np.max(_boxcar_snr_ref(e / np.sqrt(v), WIDTHS)) for e, v in batch]
        np.testing.assert_allclose(impl(batch, WIDTHS), want, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("impl", jit_variants(scoring.snr_score_func))
    def test_single_matches_batch(self, impl) -> None:
        batch = self._batch(8)
        want = scoring.snr_score_batch_func(batch, WIDTHS)
        got = [impl(np.ascontiguousarray(b), WIDTHS) for b in batch]
        np.testing.assert_allclose(got, want, rtol=1e-6)

    @pytest.mark.parametrize(
        "impl",
        jit_variants(scoring.snr_score_batch_func_complex),
    )
    def test_complex_scores_the_inverse_fft(self, impl) -> None:
        batch = self._batch(9)
        spectra = np.fft.rfft(batch, axis=-1).astype(np.complex64)
        np.testing.assert_allclose(
            impl(spectra, WIDTHS),
            scoring.snr_score_batch_func(batch, WIDTHS),
            rtol=1e-4,
            atol=1e-4,
        )


class TestMatchedFilterNoise:
    """False-alarm property of the public `MatchedFilter.compute_dot_double` score.

    Pure noise must not score as a detection. `norm_isf_func` once wrapped a negative
    index to the tail of its table and scored noise at ~28 sigma. The live
    search scores through `snr_score_batch_func` instead, which is tested above.
    """

    @pytest.mark.parametrize("seed", [42, 1234, 20260923])
    def test_pure_noise_does_not_score_as_a_detection(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        profiles = rng.normal(size=(200, NBINS)).astype(np.float32)
        scores = np.asarray(
            MatchedFilter(widths=WIDTHS, nbins=NBINS).compute_dot_double(profiles),
            dtype=float,
        )
        assert np.isfinite(scores).all(), (
            f"{int((~np.isfinite(scores)).sum())}/200 non-finite scores"
        )
        assert scores.max() < 6.0, (
            f"{int((scores > 6).sum())}/200 pure-noise profiles scored above "
            f"6 sigma, max {scores.max():.2f}"
        )
