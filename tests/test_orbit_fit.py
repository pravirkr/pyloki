from __future__ import annotations

import math

import numpy as np
import pytest

from pyloki import kepler, orbit_fit
from pyloki.utils.misc import C_VAL


def _jit_variants(fn):
    return [pytest.param(fn, id="njit"), pytest.param(fn.py_func, id="py")]


class TestFourierPhaseCurveFit:
    def test_circular_exact_recovery(self) -> None:
        t = np.linspace(0.0, 4000.0, 80, dtype=np.float64)
        p_orb_true = 2400.0
        om_true = 2.0 * math.pi / p_orb_true
        x_true = 1.5
        phi_true = 1.2
        df_over_f_true = 2.5e-6
        delays = (
            0.3
            - df_over_f_true * (t - np.mean(t))
            + x_true * np.sin(om_true * t + phi_true)
        )

        fit = orbit_fit.fit_phase_curve_fourier(
            t,
            delays,
            p_orb_range=(1800.0, 3200.0),
            fit_eccentric=False,
        )
        assert fit.p_orb == pytest.approx(p_orb_true, rel=1e-4)
        assert fit.x_orb == pytest.approx(x_true, rel=1e-4)
        assert fit.phi_orb == pytest.approx(phi_true, abs=1e-3)
        assert fit.df_over_f == pytest.approx(df_over_f_true, rel=1e-3)
        assert fit.ecc_approx == 0.0
        assert fit.rms_residual < 1e-6
        assert fit.amplitudes.shape == (1,)
        assert fit.phases.shape == (1,)
        assert np.all(np.isfinite(fit.coeff_uncertainties))

    def test_eccentric_with_eclipse_mask(self) -> None:
        t = np.linspace(0.0, 6000.0, 120, dtype=np.float64)
        p_orb_true = 3000.0
        x_true = 2.0
        ecc_true = 0.25
        aop_true = 0.7
        phi_true = 0.4
        z_sec = kepler.keplerian_z(
            t,
            p_orb_true,
            ecc_true,
            phi_true,
            x_true,
            aop_true,
            0.5 * math.pi,
        )
        delays = 0.15 + 1.0e-6 * t + z_sec

        # Drop 15% of segments as an eclipse and corrupt those dropped points
        mask = np.ones(t.size, dtype=np.bool_)
        mask[40:58] = False
        delays_corrupted = delays.copy()
        delays_corrupted[~mask] += 50.0

        fit = orbit_fit.fit_phase_curve_fourier(
            t,
            delays_corrupted,
            p_orb_range=(2200.0, 4000.0),
            fit_eccentric=True,
            n_harmonics=6,
            fit_mask=mask,
        )
        assert fit.p_orb == pytest.approx(p_orb_true, rel=2e-3)
        assert fit.x_orb == pytest.approx(x_true, rel=0.05)
        assert fit.ecc_approx > 0.1
        assert fit.amplitudes.shape == (6,)
        assert fit.phases.shape == (6,)
        assert fit.rms_residual < 1e-3
        assert np.all(np.isfinite(fit.coeff_uncertainties))

    def test_validation_errors(self) -> None:
        t = np.linspace(0.0, 100.0, 10)
        y = np.zeros_like(t)
        with pytest.raises(ValueError, match="matching shape"):
            orbit_fit.fit_phase_curve_fourier(t, y[:-1], (10.0, 50.0))
        with pytest.raises(ValueError, match="Invalid p_orb_range"):
            orbit_fit.fit_phase_curve_fourier(t, y, (50.0, 10.0))
        with pytest.raises(ValueError, match="Need at least"):
            orbit_fit.fit_phase_curve_fourier(
                t[:3],
                y[:3],
                (10.0, 50.0),
                fit_eccentric=False,
            )


class TestOrbitKernels:
    @pytest.mark.parametrize(
        "fn", _jit_variants(orbit_fit.resolve_keplerian_trajectory)
    )
    def test_resolve_keplerian_trajectory(self, fn) -> None:
        seg_times = np.linspace(0.0, 1000.0, 8, dtype=np.float64)
        a_grid = np.linspace(-50.0, 50.0, 21, dtype=np.float64)
        f_grid = np.linspace(99.9, 100.1, 41, dtype=np.float64)
        a_idx, f_idx, shifts = fn(
            seg_times,
            100.0,
            3600.0,
            1.2,
            0.1,
            0.5,
            0.3,
            a_grid,
            f_grid,
            64,
        )
        assert a_idx.shape == (8,)
        assert f_idx.shape == (8,)
        assert shifts.shape == (8,)
        assert np.all((a_idx >= 0) & (a_idx < a_grid.size))
        assert np.all((f_idx >= 0) & (f_idx < f_grid.size))
        assert np.all((shifts >= 0) & (shifts < 64))
        assert int(shifts[0]) == 0

    @pytest.mark.parametrize("fn", _jit_variants(orbit_fit.winsorized_squared_loss))
    def test_winsorized_squared_loss_caps_outliers(self, fn) -> None:
        clean = np.array([0.1, -0.1, 0.05, -0.05, 0.08, -0.08, 0.02, -0.02])
        with_outlier = clean.copy()
        with_outlier[0] = 100.0
        loss_clean = fn(clean, 3.0)
        loss_outlier = fn(with_outlier, 3.0)
        # Raw sum of squares would be > 10000; Winsorization caps it strongly
        assert loss_clean > 0.0
        assert loss_outlier < 10.0


def _make_synthetic_pffa_cube(
    seg_times: np.ndarray,
    a_grid: np.ndarray,
    f_grid: np.ndarray,
    nbins: int,
    true_params: tuple[float, float, float, float, float, float],
    amp: float = 4.0,
    seed: int = 0,
) -> np.ndarray:
    """Create a synthetic P-FFA fold cube with a Gaussian pulse on the true orbit."""
    p_pul, p_orb, x_orb, ecc, aop, phi = true_params
    rng = np.random.default_rng(seed)
    nseg = seg_times.size
    cube = np.empty((nseg, a_grid.size, f_grid.size, 2, nbins), dtype=np.float32)
    cube[..., 0, :] = rng.normal(0.0, 1.0, size=(nseg, a_grid.size, f_grid.size, nbins))
    cube[..., 1, :] = 1.0

    a_idx, f_idx, shifts = orbit_fit.resolve_keplerian_trajectory(
        seg_times,
        1.0 / p_pul,
        p_orb,
        x_orb,
        ecc,
        aop,
        phi,
        a_grid,
        f_grid,
        nbins,
    )
    bins = np.arange(nbins, dtype=np.float64)
    center = 0.5 * nbins
    pulse = amp * np.exp(-0.5 * ((bins - center) / 1.5) ** 2)
    for iseg in range(nseg):
        # Place pulse shifted by -shift so rolling by +shift aligns all segments
        shifted_pulse = np.roll(pulse, -int(shifts[iseg])).astype(np.float32)
        cube[iseg, a_idx[iseg], f_idx[iseg], 0, :] += shifted_pulse
    return cube


class TestParameterRecovery:
    def test_multi_epoch_grid_recovery_and_inspect(self) -> None:
        true_params = (0.01, 3600.0, 1.0, 0.0, 0.0, 0.5)
        a_grid = np.linspace(-30.0, 30.0, 13, dtype=np.float64)
        f_grid = np.linspace(99.7, 100.3, 13, dtype=np.float64)
        nbins = 32

        t_epoch1 = np.linspace(0.0, 1200.0, 6, dtype=np.float64)
        t_epoch2 = np.linspace(5000.0, 6200.0, 6, dtype=np.float64)
        cube1 = _make_synthetic_pffa_cube(
            t_epoch1,
            a_grid,
            f_grid,
            nbins,
            true_params,
            amp=5.0,
            seed=1,
        )
        cube2 = _make_synthetic_pffa_cube(
            t_epoch2,
            a_grid,
            f_grid,
            nbins,
            true_params,
            amp=5.0,
            seed=2,
        )

        rec = orbit_fit.ParameterRecovery.from_pffa_cubes(
            [cube1, cube2],
            [t_epoch1, t_epoch2],
            a_grid,
            f_grid,
        )
        assert rec.nsegments == 12
        assert rec.nbins == nbins

        limits = kepler.KeplerianParamLimits(
            p_pul=(0.01, 0.01),
            p_orb=(3550.0, 3650.0),
            x_orb=(0.95, 1.05),
            ecc=(0.0, 0.0),
            phi_plus_om=(0.45, 0.55),
            phi_minus_om=(0.5, 0.5),
        )
        grid = limits.generate_grid(tol=1.0, tsamp=0.05, tobs=6200.0)
        flat_grid = np.vstack([grid.reshape(-1, 6), np.array([true_params])])
        top_scores, top_params = rec.solve_keplerian_grid(flat_grid, top_k=5)
        assert top_scores.shape == (5,)
        assert top_params.shape == (5, 6)
        assert np.all(np.diff(top_scores) <= 0.0)
        assert top_params[0, 1] == pytest.approx(true_params[1], rel=0.02)
        assert top_params[0, 2] == pytest.approx(true_params[2], rel=0.05)
        assert float(top_scores[0]) > 10.0

        insp = rec.inspect_orbit(top_params[0])
        assert insp.snr == pytest.approx(float(top_scores[0]), rel=1e-5)
        assert insp.integrated_fold.shape == (2, nbins)
        assert insp.segment_folds.shape == (12, 2, nbins)


class TestNonlinearKeplerianFits:
    def test_winsorized_doppler_fit_with_outliers(self) -> None:
        t = np.linspace(0.0, 6000.0, 80, dtype=np.float64)
        f0_true = 200.0
        p_orb_true = 3000.0
        x_true = 1.5
        ecc_true = 0.25
        aop_true = 1.1
        phi_true = 0.8

        _, v_arr, _ = kepler.keplerian_z_derivatives_exact(
            t,
            p_orb_true,
            ecc_true,
            phi_true,
            x_true * C_VAL,
            aop_true,
            0.5 * math.pi,
        )
        freqs = f0_true * (1.0 - v_arr / C_VAL)

        # Corrupt 15% of segments with huge RFI/eclipse frequency outliers
        rng = np.random.default_rng(42)
        bad_idx = rng.choice(t.size, size=12, replace=False)
        freqs_corrupted = freqs.copy()
        freqs_corrupted[bad_idx] += rng.choice([-0.5, 0.5], size=12)

        seed_guess = [
            f0_true * 1.00001,
            p_orb_true * 1.01,
            x_true * 0.96,
            ecc_true + 0.03,
            aop_true + 0.1,
            phi_true - 0.1,
        ]
        res = orbit_fit.fit_keplerian_periods_winsorized(
            t,
            freqs_corrupted,
            seed_guess,
            n_restarts=4,
            seed=7,
        )
        assert res.f0 == pytest.approx(f0_true, rel=1e-4)
        assert res.p_orb == pytest.approx(p_orb_true, rel=5e-3)
        assert res.x_orb == pytest.approx(x_true, rel=2e-2)
        assert res.ecc == pytest.approx(ecc_true, abs=0.05)

    def test_keplerian_multistart_from_fourier_seed(self) -> None:
        t = np.linspace(0.0, 6000.0, 100, dtype=np.float64)
        p_orb_true = 3000.0
        x_true = 1.8
        ecc_true = 0.25
        aop_true = 0.9
        phi_true = 0.6

        z_sec = kepler.keplerian_z(
            t,
            p_orb_true,
            ecc_true,
            phi_true,
            x_true,
            aop_true,
            0.5 * math.pi,
        )
        delays = 0.25 + 2.0e-6 * t + z_sec

        fourier_fit = orbit_fit.fit_phase_curve_fourier(
            t,
            delays,
            p_orb_range=(2400.0, 3800.0),
            fit_eccentric=True,
            n_harmonics=6,
        )
        res = orbit_fit.fit_keplerian_multistart(
            t,
            delays,
            fourier_seed=fourier_fit,
            n_restarts=6,
            seed=11,
        )
        assert res.p_orb == pytest.approx(p_orb_true, rel=1e-3)
        assert res.x_orb == pytest.approx(x_true, rel=5e-3)
        assert res.ecc == pytest.approx(ecc_true, abs=0.02)
        assert res.rms_residual < 1e-3
