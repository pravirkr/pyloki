import numpy as np
import pytest

from pyloki.detection import scoring
from pyloki.detection.scoring import MatchedFilter
from pyloki.utils import maths
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

    @pytest.mark.parametrize("impl", jit_variants(scoring.generate_box_width_trials))
    @pytest.mark.parametrize("ducy_max", [1.0, 1.5])
    def test_caps_below_nbins(self, impl, ducy_max: float) -> None:
        widths = impl(16, ducy_max, 1.5)
        assert widths[-1] < 16

    @pytest.mark.parametrize("impl", jit_variants(scoring.generate_box_width_trials))
    def test_rejects_nbins_below_two(self, impl) -> None:
        with pytest.raises(ValueError, match="nbins"):
            impl(1, 0.2, 1.5)


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
        e_tot = float(np.sum(ts_e))
        v_tot = float(np.sum(ts_v))
        want = []
        for w in WIDTHS:
            best = -np.inf
            for i in range(NBINS):
                idx = (i + np.arange(int(w))) % NBINS
                ew = float(np.sum(ts_e[idx]))
                vw = float(np.sum(ts_v[idx]))
                val = (ew - (vw / v_tot) * e_tot) / np.sqrt(vw * (1.0 - vw / v_tot))
                best = max(best, val)
            want.append(best)
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

    @pytest.mark.parametrize("impl", jit_variants(scoring.boxcar_snr_1d))
    def test_1d_full_width_returns_zero_instead_of_nan(self, impl) -> None:
        profile = _profiles(1, seed=10)[0]
        bad_widths = np.array([NBINS, NBINS + 1], dtype=np.int64)
        np.testing.assert_array_equal(impl(profile, bad_widths, 1.0), [0.0, 0.0])

    @pytest.mark.parametrize("impl", jit_variants(scoring.boxcar_snr_beta_1d))
    def test_beta_full_width_returns_zero_instead_of_nan(self, impl) -> None:
        profile = _profiles(1, seed=10)[0]
        ones = np.ones(NBINS, dtype=np.float32)
        bad_widths = np.array([NBINS, NBINS + 1], dtype=np.int64)
        np.testing.assert_array_equal(impl(profile, ones, bad_widths), [0.0, 0.0])

    @pytest.mark.parametrize("impl", jit_variants(scoring.snr_score_batch_func))
    def test_batch_full_width_returns_zero_instead_of_nan(self, impl) -> None:
        profile = _profiles(1, seed=10)[0]
        ones = np.ones(NBINS, dtype=np.float32)
        bad_widths = np.array([NBINS, NBINS + 1], dtype=np.int64)
        batch = np.stack([profile, ones])[None, :, :]
        np.testing.assert_array_equal(impl(batch, bad_widths), [0.0])


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


class TestMatchedFilterAndFFT:
    @pytest.mark.parametrize("shift", [0, -1])
    def test_rejects_non_positive_shift(self, shift: int) -> None:
        with pytest.raises(ValueError, match="Shift"):
            MatchedFilter(widths=WIDTHS, nbins=NBINS, shift=shift)

    @pytest.mark.parametrize(("nbins", "shift"), [(64, 3), (50, 3), (51, 4)])
    def test_non_divisible_shift_e_mat_and_compute_dot(
        self, nbins: int, shift: int
    ) -> None:
        widths = np.array([1, 2, 5], dtype=np.int64)
        mf = MatchedFilter(widths=widths, nbins=nbins, shift=shift)
        expected_rows = int(np.sum((nbins + mf.shifts - 1) // mf.shifts))
        assert mf.e_mat.shape == (expected_rows, nbins)

        rng = np.random.default_rng(11)
        x_3d = rng.standard_normal((2, 3, nbins)).astype(np.float32)
        want_3d = np.empty((2, 3, len(widths)), dtype=np.float32)
        for i in range(2):
            for j in range(3):
                for k in range(len(widths)):
                    want_3d[i, j, k] = max(
                        float(np.dot(x_3d[i, j], np.roll(mf.templates[k], s)))
                        for s in range(0, nbins, shift)
                    )

        np.testing.assert_allclose(mf.compute_dot(x_3d), want_3d, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            mf.compute_dot(x_3d[0]), want_3d[0], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            mf.compute_dot(x_3d[0, 0]), want_3d[0, 0], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            scoring.compute_snr_e_mat(x_3d[0, 0], mf.e_mat, mf.shifts, 1.5),
            want_3d[0, 0] / 1.5,
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.parametrize("nbins", [64, 50, 51])
    def test_fft_matches_dot_across_lengths(self, nbins: int) -> None:
        widths = np.array([1, 3, 7], dtype=np.int64)
        mf = MatchedFilter(widths=widths, nbins=nbins, shift=1)
        rng = np.random.default_rng(12)
        for shape in [(nbins,), (4, nbins), (2, 3, nbins)]:
            data = rng.standard_normal(shape).astype(np.float32)
            np.testing.assert_allclose(
                mf.compute_fft(data),
                mf.compute_dot(data),
                rtol=1e-4,
                atol=1e-4,
            )

    @pytest.mark.parametrize(
        "impl",
        jit_variants(scoring._compute_snr_fft),  # noqa: SLF001
    )
    @pytest.mark.parametrize("nbins", [64, 50, 51])
    def test_fft_asymmetric_template_circular_cross_correlation(
        self, impl, nbins: int
    ) -> None:
        rng = np.random.default_rng(13)
        data = rng.standard_normal((3, nbins)).astype(np.float32)
        asym = np.zeros((2, nbins), dtype=np.float32)
        asym[0, :5] = np.array([3.0, 1.5, 0.2, -1.0, -2.5], dtype=np.float32)
        asym[1, :4] = np.array([-2.0, 4.0, -0.5, 1.0], dtype=np.float32)
        for idx in range(2):
            asym[idx] = scoring._normalise(asym[idx])  # noqa: SLF001

        got = impl(data, asym)
        want = np.empty((3, 2), dtype=np.float32)
        for i in range(3):
            for j in range(2):
                want[i, j] = max(
                    float(np.dot(data[i], np.roll(asym[j], shift)))
                    for shift in range(nbins)
                )
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "impl",
        jit_variants(scoring._compute_snr_double),  # noqa: SLF001
    )
    def test_double_handles_float64_and_small_filter_bank(self, impl) -> None:
        rng = np.random.default_rng(14)
        data_f64 = rng.standard_normal((5, NBINS))
        for n_filters in [1, 3]:
            e_mat = np.zeros((n_filters, NBINS), dtype=np.float32)
            for k in range(n_filters):
                e_mat[k, k * 8 : k * 8 + 2] = 1.0
                e_mat[k] = scoring._normalise(e_mat[k])  # noqa: SLF001
            scores = impl(data_f64, e_mat, n_corr=10)
            assert scores.shape == (5,)
            assert np.isfinite(scores).all()

    @pytest.mark.parametrize(
        "impl",
        jit_variants(scoring._compute_snr_double),  # noqa: SLF001
    )
    def test_double_peak_uses_natural_log_lee(self, impl) -> None:
        mf = MatchedFilter(widths=np.array([1]), nbins=NBINS, shift=8)
        t0, t32 = mf.e_mat[0], mf.e_mat[4]
        profile = (5.0 * t0 + 5.0 * t32).astype(np.float32)
        got = float(impl(profile[None, :], mf.e_mat, n_corr=10)[0])

        scores = mf.e_mat @ profile
        gamma = float(mf.e_mat[0] @ mf.e_mat[4])
        s2 = (scores[0] ** 2 + scores[4] ** 2 - 2 * gamma * scores[0] * scores[4]) / (
            1.0 - gamma**2
        )
        n_filters = len(mf.e_mat)
        x_double = 0.5 * s2 - np.log(n_filters * (n_filters - 1) // 2)
        want = float(maths.norm_isf_func(x_double))
        assert got == pytest.approx(want, rel=1e-4, abs=1e-3)

    @pytest.mark.parametrize(
        "impl",
        jit_variants(scoring._compute_snr_double),  # noqa: SLF001
    )
    def test_double_solved_amplitude_positivity_guard(self, impl) -> None:
        base = scoring._gen_boxcar_templates(np.array([4], dtype=np.int64), NBINS)[0]  # noqa: SLF001
        e_ovl = np.stack([base, np.roll(base, 1)])
        prof_trough = (8.0 * e_ovl[0] - 3.0 * e_ovl[1]).astype(np.float32)
        scores_ovl = e_ovl @ prof_trough
        assert scores_ovl[0] > 0.0
        assert scores_ovl[1] > 0.0
        want_single = float(
            maths.norm_isf_func(
                maths.chi_sq_minus_logsf_func(float(scores_ovl.max() ** 2), 1)
                - np.log(2)
            )
        )
        got_trough = float(impl(prof_trough[None, :], e_ovl, n_corr=10)[0])
        assert got_trough == pytest.approx(want_single, rel=1e-4, abs=1e-3)

        wide = scoring._gen_boxcar_templates(np.array([16], dtype=np.int64), NBINS)[0]  # noqa: SLF001
        e_disjoint = np.stack([wide, np.roll(wide, 32)])
        prof_pos = (10.0 * e_disjoint[0] + 2.0 * e_disjoint[1]).astype(np.float32)
        scores_disj = e_disjoint @ prof_pos
        gamma = float(e_disjoint[0] @ e_disjoint[1])
        assert gamma < 0.0
        assert scores_disj[1] < 0.0  # negative raw projection despite +2 amplitude
        s2 = (
            scores_disj[0] ** 2
            + scores_disj[1] ** 2
            - 2.0 * gamma * scores_disj[0] * scores_disj[1]
        ) / (1.0 - gamma**2)
        want_double = float(maths.norm_isf_func(0.5 * s2 - np.log(1)))
        got_pos = float(impl(prof_pos[None, :], e_disjoint, n_corr=10)[0])
        assert got_pos == pytest.approx(want_double, rel=1e-4, abs=1e-3)


class TestHarmonicSumming:
    @pytest.mark.parametrize("impl", jit_variants(scoring.harmonic_summing_score_func))
    def test_pure_noise_does_not_false_trigger(self, impl) -> None:
        rng = np.random.default_rng(15)
        profiles = rng.standard_normal((200, NBINS)).astype(np.float32)
        ones = np.ones(NBINS, dtype=np.float32)
        scores = np.array([impl(np.stack([p, ones]), 8) for p in profiles], dtype=float)
        assert np.isfinite(scores).all()
        assert scores.max() < 6.0

    @pytest.mark.parametrize("impl", jit_variants(scoring.harmonic_summing_score_func))
    def test_clamps_nyquist_and_validates_bounds(self, impl) -> None:
        rng = np.random.default_rng(16)
        combined = np.stack([rng.standard_normal(NBINS), np.ones(NBINS)]).astype(
            np.float32
        )
        # n_harmonics > (nbins - 1) // 2 and > 32 clamps cleanly without crashing.
        res = impl(combined, 100)
        assert np.isfinite(res)

        with pytest.raises(ValueError, match="harmonic"):
            impl(combined, 0)
        small = np.stack([np.ones(2), np.ones(2)]).astype(np.float32)
        with pytest.raises(ValueError, match="harmonic"):
            impl(small, 4)


class TestBoxcarGlrt:
    @pytest.mark.parametrize("impl", jit_variants(scoring.boxcar_snr_glrt_1d))
    def test_uniform_weights_matches_boxcar_1d(self, impl) -> None:
        ts_e = _profiles(1, seed=21)[0]
        v0 = np.full(NBINS, 3.5, dtype=np.float32)
        got = impl(ts_e, v0, WIDTHS)
        want = scoring.boxcar_snr_1d(
            (ts_e / np.sqrt(v0)).astype(np.float32), WIDTHS, 1.0
        )
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("impl", jit_variants(scoring.boxcar_snr_glrt_1d))
    def test_heteroscedastic_with_zeros_matches_exact_glrt_and_dc_invariant(
        self, impl
    ) -> None:
        rng = np.random.default_rng(22)
        ts_v = rng.uniform(0.2, 5.0, NBINS).astype(np.float32)
        ts_v[5:10] = 0.0  # RFI-masked bins
        ts_e = (rng.standard_normal(NBINS) * np.sqrt(ts_v)).astype(np.float32)
        ts_e[ts_v == 0.0] = 0.0

        got = impl(ts_e, ts_v, WIDTHS)
        # Adding constant DC baseline mu_0 * ts_v leaves GLRT S/N invariant
        got_dc = impl((ts_e + 7.5 * ts_v).astype(np.float32), ts_v, WIDTHS)
        np.testing.assert_allclose(got, got_dc, rtol=1e-4, atol=1e-4)

        # Brute-force exact GLRT check
        v_tot = float(np.sum(ts_v))
        e_tot = float(np.sum(ts_e))
        want = []
        for w in WIDTHS:
            best = -np.inf
            for s in range(NBINS):
                idx = (np.arange(s, s + int(w)) % NBINS).astype(int)
                vw = float(np.sum(ts_v[idx]))
                ew = float(np.sum(ts_e[idx]))
                if vw > 1e-12 * v_tot and v_tot - vw > 1e-12 * v_tot:
                    frac = vw / v_tot
                    val = (ew - frac * e_tot) / np.sqrt(vw * (1.0 - frac))
                    best = max(best, val)
            want.append(best)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    def test_glrt_null_calibration_and_batch_wrappers(self) -> None:
        rng = np.random.default_rng(23)
        n = 4000
        v_row = rng.uniform(0.5, 4.0, NBINS).astype(np.float32)
        v_row[0] = 0.0  # masked zero-weight bin
        ts_v = np.broadcast_to(v_row, (n, NBINS)).copy()
        ts_e = (rng.standard_normal((n, NBINS)) * np.sqrt(ts_v)).astype(np.float32)

        # Fixed window [1, 5) GLRT is N(0, 1) under H0 even with skewed/zero V
        v_tot = float(np.sum(v_row))
        v_w = float(np.sum(v_row[1:5]))
        frac = v_w / v_tot
        e_tot = np.sum(ts_e, axis=1, dtype=np.float64)
        e_w = np.sum(ts_e[:, 1:5], axis=1, dtype=np.float64)
        z_fixed = (e_w - frac * e_tot) / np.sqrt(v_w * (1.0 - frac))
        assert abs(float(np.mean(z_fixed))) < 0.05
        assert abs(float(np.std(z_fixed)) - 1.0) < 0.05

        # Max-over-phase null distribution matches uniform-variance baseline
        w1 = np.array([4], dtype=np.int64)
        s_par = scoring.boxcar_snr_glrt_2d(ts_e, ts_v, w1)
        s_ser = scoring.boxcar_snr_glrt_2d_serial(ts_e, ts_v, w1)
        s_beta = scoring.boxcar_snr_beta_2d(ts_e, ts_v, w1)
        np.testing.assert_allclose(s_par, s_ser, rtol=1e-6)
        np.testing.assert_allclose(s_beta, s_par, rtol=1e-6)

        # Compare against uniform-variance null of same size
        uni_e = rng.standard_normal((n, NBINS)).astype(np.float32)
        uni_v = np.ones((n, NBINS), dtype=np.float32)
        s_uni = scoring.boxcar_snr_glrt_2d(uni_e, uni_v, w1)
        assert abs(float(np.mean(s_par)) - float(np.mean(s_uni))) < 0.1
        assert abs(float(np.std(s_par)) - float(np.std(s_uni))) < 0.1

        folds_ev = np.stack([ts_e[:16], ts_v[:16]], axis=1)
        s_3d = scoring.boxcar_snr_glrt_3d(folds_ev, WIDTHS)
        s_beta_3d = scoring.boxcar_snr_beta_3d(folds_ev, WIDTHS)
        s_batch = scoring.snr_glrt_score_batch_func(folds_ev, WIDTHS)
        np.testing.assert_allclose(s_beta_3d, s_3d, rtol=1e-6)
        np.testing.assert_allclose(s_batch, np.max(s_3d, axis=1), rtol=1e-6)


class TestDoubleBoxcarCombinedAndKadane:
    @pytest.mark.parametrize("impl", jit_variants(scoring.double_boxcar_snr_1d))
    def test_double_boxcar_boosts_interpulse_over_single(self, impl) -> None:
        prof = np.zeros(NBINS, dtype=np.float32)
        prof[4:7] = 5.0
        prof[20:23] = 5.0
        w = np.array([1, 2, 3, 4], dtype=np.int64)
        s_double = impl(prof, w)
        # Single-peak trial-penalized score
        z1 = float(np.max(scoring.boxcar_snr_1d(prof, w, 1.0)))
        n_trials = len(w) * NBINS
        s_single_lee = float(
            maths.norm_isf_func(
                maths.chi_sq_minus_logsf_func(z1 * z1, 1) - np.log(n_trials),
            ),
        )
        assert s_double > s_single_lee + 2.0

        # Pure white noise stays finite
        rng = np.random.default_rng(30)
        noise = rng.standard_normal(NBINS).astype(np.float32)
        assert np.isfinite(impl(noise, w))
        assert np.isfinite(impl(np.zeros(NBINS, dtype=np.float32), w))

    @pytest.mark.parametrize("impl", jit_variants(scoring.combined_score_func))
    def test_combined_score_is_max_of_boxcar_and_harmonic(self, impl) -> None:
        rng = np.random.default_rng(31)
        ev = np.stack([rng.standard_normal(NBINS), np.ones(NBINS)]).astype(np.float32)
        got = impl(ev, WIDTHS, 8)
        want = max(
            scoring.snr_score_func(ev, WIDTHS),
            scoring.harmonic_summing_score_func(ev, 8),
        )
        assert got == pytest.approx(want, rel=1e-6)

    @pytest.mark.parametrize(
        "mode",
        [
            "boxcar",
            "boxcar_glrt",
            "double_boxcar",
            "double_matched",
            "harmonic_z2",
            "combined",
            "kadane",
        ],
    )
    def test_score_profiles_all_modes(self, mode: str) -> None:
        rng = np.random.default_rng(32)
        folds_ev = np.stack(
            [rng.standard_normal((8, NBINS)), np.ones((8, NBINS))],
            axis=1,
        ).astype(np.float32)
        scores = scoring.score_profiles(
            folds_ev,
            WIDTHS,
            mode=mode,  # type: ignore[arg-type]
            n_harmonics=8,
        )
        assert scores.shape == (8,)
        assert np.all(np.isfinite(scores))

    def test_score_profiles_validation(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            scoring.score_profiles(np.zeros((4, NBINS), dtype=np.float32), WIDTHS)
        with pytest.raises(ValueError, match="Unsupported score mode"):
            scoring.score_profiles(
                np.ones((2, 2, NBINS), dtype=np.float32),
                WIDTHS,
                mode="unknown",  # type: ignore[arg-type]
            )


class TestMarkovNullingHMM:
    @pytest.mark.parametrize(
        "fn",
        jit_variants(scoring.markov_hmm_log_llr),
    )
    def test_markov_hmm_stability_and_null_calibration(self, fn) -> None:
        rng = np.random.default_rng(101)
        # 10,000 segments with extreme transition probabilities: must not overflow
        z_long = rng.standard_normal(10_000)
        llr_h0 = fn(z_long, 1.0, 1e-4, 1e-4)
        assert np.isfinite(llr_h0)
        assert llr_h0 < 10.0

    @pytest.mark.parametrize(
        "fn",
        jit_variants(scoring.viterbi_null_mask),
    )
    def test_viterbi_and_posterior_recover_on_burst(self, fn) -> None:
        from pyloki.simulation.pulse import generate_markov_nulling_envelope  # noqa: PLC0415

        rng = np.random.default_rng(202)
        n_seg = 200
        env = generate_markov_nulling_envelope(n_seg, 0.08, 0.02, rng=rng)
        assert env.shape == (n_seg,)
        assert set(np.unique(env)).issubset({0.0, 1.0})

        # Deterministic 20% ON block to verify HMM beats full-span unweighted sum
        true_on = np.zeros(n_seg, dtype=np.bool_)
        true_on[60:100] = True
        z = rng.standard_normal(n_seg) + np.where(true_on, 2.5, 0.0)

        mask = fn(z, 2.5, 0.025, 0.006)
        assert mask.shape == (n_seg,)
        # High overlap with the true ON window
        overlap = float(np.mean(mask == true_on))
        assert overlap > 0.85

        post = scoring.markov_hmm_posterior_on(z, 2.5, 0.025, 0.006)
        assert np.all((post >= 0.0) & (post <= 1.0))
        assert float(np.mean(post[true_on])) > 0.7
        assert float(np.mean(post[~true_on])) < 0.2

        best_llr, *_ = scoring.markov_hmm_grid_score(z)
        # Equivalent Gaussian S/N from 2 * log_llr exceeds unweighted sum S/N
        snr_unweighted = float(np.sum(z) / np.sqrt(n_seg))
        snr_hmm_eq = float(np.sqrt(max(2.0 * best_llr, 0.0)))
        assert snr_hmm_eq > snr_unweighted + 2.0

    def test_markov_hmm_posterior_bounded_long_high_snr(self) -> None:
        """Forward-backward posterior stays in [0, 1] on long arrays."""
        rng = np.random.default_rng(77)
        n = 10_000
        z = rng.normal(0.0, 1.0, size=n)
        z[2000:5000] += 5.0
        post = scoring.markov_hmm_posterior_on(z, amp=5.0, p_on_off=0.01, p_off_on=0.01)
        assert np.all(np.isfinite(post))
        assert float(np.min(post)) >= 0.0
        assert float(np.max(post)) <= 1.0


class TestScoreProfilesMaskedBins:
    @pytest.mark.parametrize(
        "mode",
        [
            "boxcar",
            "boxcar_glrt",
            "harmonic_z2",
            "combined",
            "double_boxcar",
            "double_matched",
            "kadane",
        ],
    )
    def test_zero_variance_bins_do_not_produce_nan_or_sentinel(
        self,
        mode: str,
    ) -> None:
        """Masked ts_v == 0 bins must return finite valid S/N for all modes."""
        nbins = 32
        folds_ev = np.zeros((2, 2, nbins), dtype=np.float32)
        folds_ev[:, 1, :] = 1.0
        # Inject a clean pulse at bin 10
        folds_ev[0, 0, 10] = 12.0
        # Mask bins 0, 1, 2 (ts_v == 0, ts_e == 0)
        folds_ev[:, 0, :3] = 0.0
        folds_ev[:, 1, :3] = 0.0
        widths = np.array([1, 2, 4], dtype=np.int64)
        scores = scoring.score_profiles(folds_ev, widths, mode=mode)  # type: ignore[arg-type]
        assert np.all(np.isfinite(scores))
        assert float(scores[0]) > 3.0
        assert float(scores[1]) > -100.0


class TestPhase1ScoringUpgrades:
    def test_boxcar_snr_beta_heteroscedastic_null_calibration(self) -> None:
        """Heteroscedastic GLRT SNR_beta is calibrated under H0 with skewed ts_v."""
        rng = np.random.default_rng(123)
        nbins = 32
        n_trials = 2000
        # Strongly skewed bin weights including zero-weight masked bins
        ts_v = np.linspace(0.2, 5.0, nbins, dtype=np.float32)
        ts_v[0] = 0.0
        ts_v[15] = 0.0
        dc_offset = 3.5
        # Under H0: E_b ~ N(dc_offset * V_b, V_b)
        ts_e = (
            dc_offset * ts_v[None, :]
            + rng.standard_normal((n_trials, nbins)).astype(np.float32)
            * np.sqrt(ts_v)[None, :]
        )
        widths = np.array([2], dtype=np.int64)
        snrs = scoring.boxcar_snr_beta_2d(
            np.ascontiguousarray(ts_e, dtype=np.float32),
            np.ascontiguousarray(
                np.broadcast_to(ts_v, (n_trials, nbins)),
                dtype=np.float32,
            ),
            widths,
        )[:, 0]
        assert np.all(np.isfinite(snrs))
        # Max over 32 circular windows of unit-normal scores has mean ~2.0 and std ~0.6
        assert 1.5 < float(np.mean(snrs)) < 2.6
        assert 0.35 < float(np.std(snrs)) < 0.9

    def test_matched_filter_per_width_shift_jump_list(self) -> None:
        widths = np.array([1, 2, 4, 8], dtype=np.int64)
        shifts = [1, 1, 2, 4]
        mf = scoring.MatchedFilter(widths, nbins=32, shift=shifts, kind="gaussian")
        np.testing.assert_array_equal(mf.shifts, np.array(shifts, dtype=np.int64))
        # Fewer templates than shift=1 reduces look-elsewhere trials
        mf_dense = scoring.MatchedFilter(widths, nbins=32, shift=1, kind="gaussian")
        assert mf.e_mat.shape[0] < mf_dense.e_mat.shape[0]

    def test_kadane_circular_accuracy_on_wrapped_pulse(self) -> None:
        nbins = 64
        prof = np.zeros(nbins, dtype=np.float32)
        # Pulse wrapping across bins 62, 63, 0, 1 (width 4)
        prof[62:] = 4.0
        prof[:2] = 4.0
        widths = np.array([1, 2, 4, 8], dtype=np.int64)
        exact_box = float(np.max(scoring.boxcar_snr_1d(prof, widths, 1.0)))
        biases = scoring.generate_kadane_biases(nbins, ducy_max=0.2)
        kad_snr, kad_w, _ = scoring.kadane_circular_normalized_1d(prof, biases)
        assert kad_w == 4
        assert kad_snr == pytest.approx(exact_box, rel=1e-4)

    @pytest.mark.parametrize(
        "mode",
        [
            "boxcar",
            "boxcar_glrt",
            "double_boxcar",
            "harmonic_z2",
            "combined",
            "kadane",
        ],
    )
    def test_ep_search_wired_with_score_mode(self, mode: str) -> None:
        from pyloki.config import PulsarSearchConfig  # noqa: PLC0415
        from pyloki.search import ep_search, ffa_search  # noqa: PLC0415
        from pyloki.simulation.pulse import PulseSignalConfig  # noqa: PLC0415

        sig = PulseSignalConfig(
            period=0.02,
            dt=5e-4,
            nsamps=2**13,
            snr=25.0,
            ducy=0.1,
            seed=7,
        )
        ts = sig.generate(shape="gaussian")
        cfg = PulsarSearchConfig(
            nsamps=ts.nsamps,
            tsamp=ts.dt,
            nbins=32,
            eta=1.0,
            param_limits=np.array([[-10.0, 10.0], [49.8, 50.2]]),
            prune_poly_order=2,
            bseg_brute=2**10,
            bseg_ffa=2**11,
            use_fourier=False,
            score_mode=mode,  # type: ignore[arg-type]
        )
        # End-to-end EP search with configured score_mode
        ep_res = ep_search(
            ts,
            cfg,
            threshold_scheme=np.full(3, 1.0, dtype=np.float32),
            n_runs=1,
            ref_segs=[1],
            quiet=True,
        )
        assert np.isfinite(ep_res.best_score)
        assert ep_res.best_score > 6.0

        # End-to-end FFA search with configured score_mode
        cfg_ffa = PulsarSearchConfig(
            nsamps=ts.nsamps,
            tsamp=ts.dt,
            nbins=32,
            eta=1.0,
            param_limits=np.array([[49.8, 50.2]]),
            bseg_brute=2**10,
            bseg_ffa=ts.nsamps,
            use_fourier=False,
            score_mode=mode,  # type: ignore[arg-type]
        )
        _, pgram = ffa_search(ts, cfg_ffa, quiet=True)
        assert float(np.max(pgram.snrs)) > 6.0
