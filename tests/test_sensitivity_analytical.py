from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate

from pyloki.sensitivity import analytical, heavy_tail


class TestAnalyticalSensitivity:
    @pytest.mark.parametrize(
        "fn",
        [analytical.gaussian_profile, analytical.von_mises_profile],
    )
    def test_profiles_zero_mean_unit_energy_and_parseval(self, fn) -> None:
        prof = fn(0.05, n_grid=2048)
        assert float(np.mean(prof)) == pytest.approx(0.0, abs=1e-12)
        assert float(np.mean(prof * prof)) == pytest.approx(1.0, rel=1e-12)
        # Sum of 2|c_k|^2 across many harmonics approaches 1.0
        hp = analytical.profile_fourier_powers(prof, n_harmonics=256)
        assert hp.size == 256
        assert float(np.sum(hp)) == pytest.approx(1.0, abs=1e-4)

    def test_fft_noncentral_chi2_matches_monte_carlo(self) -> None:
        rng = np.random.default_rng(7)
        n_samp = 1024
        n_cycles = 32  # exact integer Fourier bin => scalloping = 1.0
        n_harm = 8
        fap = 1e-3
        snr_opt = 6.5

        t_phase = np.mod(np.arange(n_samp, dtype=np.float64) * n_cycles / n_samp, 1.0)
        sigma = 0.05 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        dphi = ((t_phase - 0.5 + 0.5) % 1.0) - 0.5
        raw = np.exp(-0.5 * (dphi / sigma) ** 2)
        raw -= float(np.mean(raw))
        unit_sig = raw / float(np.linalg.norm(raw))
        sig = snr_opt * unit_sig

        prof = analytical.gaussian_profile(0.05, n_grid=2048)
        hp = analytical.profile_fourier_powers(prof, n_harm)
        p_theory = analytical.fft_detection_prob(snr_opt, hp, fap, scalloping=1.0)
        thresh = analytical.fft_harmonic_threshold(n_harm, fap)

        n_trials = 600
        noise = rng.standard_normal((n_trials, n_samp))
        spec = np.fft.rfft(noise + sig[None, :], axis=1)
        bins = np.arange(1, n_harm + 1) * n_cycles
        r_stat = (2.0 / float(n_samp)) * np.sum(np.abs(spec[:, bins]) ** 2, axis=1)
        p_emp = float(np.mean(r_stat >= thresh))
        # Binomial 4-sigma tolerance
        tol = 4.0 * math.sqrt(p_theory * (1.0 - p_theory) / n_trials) + 0.02
        assert abs(p_emp - p_theory) < tol

    def test_ffa_beats_fft_on_narrow_pulses_and_z2m_htest(self) -> None:
        comp = analytical.compare_ffa_vs_fft_flux_ratio(
            duty_cycle=0.02,
            n_harmonics=16,
            fap=1e-7,
            target_pdet=0.5,
        )
        # Coherent FFA requires strictly less flux than incoherent 16-harmonic FFT
        assert comp.snr_min_ffa < comp.snr_min_fft
        assert comp.flux_ratio_ffa_over_fft < 0.85
        assert comp.fft_gain_factor > 1.15

        prof = analytical.gaussian_profile(0.03, n_grid=128)
        sig_prof = 10.0 * prof / math.sqrt(128.0)
        z2 = analytical.z2m_statistic(sig_prof, m=16)
        h_val, best_m = analytical.h_test_statistic(sig_prof, m_max=20)
        assert z2 > 50.0
        assert h_val > 25.0
        assert 1 <= best_m <= 20


class TestHeavyTailSensitivity:
    @pytest.mark.parametrize("beta", [2.5, 3.0, 4.0])
    def test_pareto_pdf_exact_normalization_and_mean(self, beta: float) -> None:
        mean_s = 2.0
        a0 = heavy_tail.pareto_a0(mean_s, beta)
        area, _ = integrate.quad(
            lambda s: float(heavy_tail.pareto_signal_pdf(s, mean_s, beta)),
            a0,
            np.inf,
            limit=200,
        )
        mean_est, _ = integrate.quad(
            lambda s: s * float(heavy_tail.pareto_signal_pdf(s, mean_s, beta)),
            a0,
            np.inf,
            limit=200,
        )
        assert area == pytest.approx(1.0, rel=1e-5)
        assert mean_est == pytest.approx(mean_s, rel=1e-4)

    def test_lognormal_pdf_normalization_and_mean(self) -> None:
        mean_s = 1.5
        sigma_ln = 1.0
        area, _ = integrate.quad(
            lambda s: float(heavy_tail.lognormal_signal_pdf(s, mean_s, sigma_ln)),
            0.0,
            np.inf,
            limit=200,
        )
        mean_est, _ = integrate.quad(
            lambda s: s * float(heavy_tail.lognormal_signal_pdf(s, mean_s, sigma_ln)),
            0.0,
            np.inf,
            limit=200,
        )
        assert area == pytest.approx(1.0, rel=1e-5)
        assert mean_est == pytest.approx(mean_s, rel=1e-4)

    @pytest.mark.parametrize(
        ("dist", "shape_param"),
        [("pareto", 2.8), ("lognormal", 1.2)],
    )
    def test_heavy_tail_likelihood_beats_linear_baseline(
        self,
        dist: str,
        shape_param: float,
    ) -> None:
        model = heavy_tail.HeavyTailLikelihood(
            distribution=dist,  # type: ignore[arg-type]
            mean_s=0.5,
            shape_param=shape_param,
            sigma_n=1.0,
            x_range=(-6.0, 35.0),
            n_grid=512,
        )
        # Convolved density integrates to ~1
        area = float(integrate.trapezoid(model.p_h1_grid, model.x_grid))
        assert area == pytest.approx(1.0, abs=0.02)
        # Non-linear Neyman-Pearson KL divergence strictly exceeds linear baseline
        assert model.kl_divergence() > model.linear_kl_baseline()
        # Positive tail samples yield positive LLR
        assert model.score_samples(np.array([4.0, 5.0, 6.0])) > 0.0

    @pytest.mark.parametrize("sigma_n", [1.0, 0.5])
    def test_heavy_tail_default_range_strictly_monotonic_no_underflow(
        self,
        sigma_n: float,
    ) -> None:
        """Default x_range=(-8, 40) must not underflow/invert LLR."""
        model = heavy_tail.HeavyTailLikelihood(
            distribution="pareto",
            mean_s=2.0,
            shape_param=2.5,
            sigma_n=sigma_n,
        )
        assert np.all(np.isfinite(model.llr_grid))
        # LLR must remain strictly increasing past x=37.14 up to x=40.0
        tail_mask = model.x_grid >= 5.0
        diffs = np.diff(model.llr_grid[tail_mask])
        assert np.all(diffs > 0.0)
        # Extrapolation beyond x_max must not be clipped flat
        assert float(model.llr(45.0)) > float(model.llr(40.0))


def test_even_n_z2m_excludes_nyquist_and_compare_handles_target_le_fap() -> None:
    """Exclude Nyquist on even-n & handle target_pdet <= fap."""
    # Even n=16 has 7 independent complex harmonics (k=1..7), not 8 (k=8 is Nyquist)
    rng = np.random.default_rng(1234)
    noise = rng.standard_normal((4000, 16))
    z2_vals = np.array([analytical.z2m_statistic(row, m=8) for row in noise])
    # With k=1..7 (14 DOF), mean=14, var=28 (vs doubled Nyquist giving mean=16, var=36)
    assert float(np.mean(z2_vals)) == pytest.approx(14.0, abs=0.4)
    assert float(np.var(z2_vals)) == pytest.approx(28.0, abs=2.5)

    # profile_fourier_powers must also not double Nyquist bin
    alt = np.array([1.0, -1.0] * 8, dtype=np.float64)
    hp = analytical.profile_fourier_powers(alt, n_harmonics=8)
    assert float(np.sum(hp)) == 0.0

    # target_pdet <= fap must return 0.0 without crashing inside optimize.brentq
    cmp_res = analytical.compare_ffa_vs_fft_flux_ratio(
        duty_cycle=0.05,
        n_harmonics=8,
        fap=0.5,
        target_pdet=0.25,
    )
    assert cmp_res.snr_min_ffa == 0.0
    assert cmp_res.snr_min_fft == 0.0
