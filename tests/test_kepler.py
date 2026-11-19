from __future__ import annotations

import math

import numpy as np
import pytest
from astropy import constants

from pyloki import kepler
from pyloki.utils.misc import C_VAL
from tests.jit_utils import jit_variants


class TestOrbitalScalars:
    def test_semi_major_axis_kepler_third_law(self) -> None:
        mass = 1.4
        p_orb = 8834.5
        omega = 2.0 * np.pi / p_orb
        expected = (
            (mass * constants.M_sun * constants.G / omega**2) ** (1.0 / 3.0)
            / constants.c
        ).value
        np.testing.assert_allclose(
            kepler.semi_major_axis(mass, p_orb),
            expected,
            rtol=1e-12,
        )

    @pytest.mark.parametrize("ecc", [0.0, 0.25, 0.6])
    def test_mass_function_exact_identity(self, ecc: float) -> None:
        """Verify f(M) = (m2 sin i)^3 / (m1 + m2)^2 in solar masses."""
        m1 = 1.338
        m2 = 1.249
        sin_i = 0.999
        p_orb = 8834.5
        total_mass = m1 + m2
        expected_mf = (m2 * sin_i) ** 3 / (total_mass**2)

        # Semi-major axis of pulsar orbit (a_1 sin i) in light-seconds
        a_rel_ls = kepler.semi_major_axis(total_mass, p_orb)
        x_orb = a_rel_ls * (m2 / total_mass) * sin_i
        omega = 2.0 * np.pi / p_orb
        k_over_c = (omega * x_orb) / math.sqrt(1.0 - ecc**2)

        got = kepler.mass_function(k_over_c, p_orb, ecc=ecc)
        np.testing.assert_allclose(got, expected_mf, rtol=1e-12)

    @pytest.mark.parametrize(
        ("p_orb", "ecc"),
        [(0.0, 0.0), (-10.0, 0.0), (100.0, 1.0), (100.0, -0.1)],
    )
    def test_mass_function_rejects_invalid_inputs(
        self,
        p_orb: float,
        ecc: float,
    ) -> None:
        with pytest.raises(ValueError, match=r"positive|0 <= ecc < 1"):
            kepler.mass_function(1e-4, p_orb, ecc=ecc)


class TestKeplerianKinematics:
    @pytest.mark.parametrize("impl", jit_variants(kepler.keplerian_ecc_anomaly))
    @pytest.mark.parametrize("ecc", [0.0, 0.3, 0.85])
    def test_kepler_equation_residual(self, impl, ecc: float) -> None:
        p_orb = 3600.0
        phi = 0.7
        t_arr = np.linspace(0.0, 2.5 * p_orb, 64)
        e_arr = impl(t_arr, p_orb, ecc, phi, n_iters=10)
        m_expected = 2.0 * np.pi * ((t_arr % p_orb) / p_orb) + phi
        resid = (e_arr - ecc * np.sin(e_arr) - m_expected + np.pi) % (
            2.0 * np.pi
        ) - np.pi
        np.testing.assert_allclose(resid, 0.0, atol=1e-12)

    @pytest.mark.parametrize("impl", jit_variants(kepler.keplerian_nu))
    def test_true_anomaly_smooth_across_pi(self, impl) -> None:
        p_orb = 100.0
        ecc = 0.5
        t_arr = np.array([0.0, p_orb / 2.0, p_orb - 1e-6])
        nu = impl(t_arr, p_orb, ecc, 0.0, n_iters=10)
        assert np.all(np.isfinite(nu))
        np.testing.assert_allclose(nu[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(nu[1], np.pi, atol=1e-12)

    @pytest.mark.parametrize("impl", jit_variants(kepler.compute_nu_derivatives))
    def test_nu_second_derivative_sign(self, impl) -> None:
        """Moving away from periastron (0 < E < pi), angular speed decreases."""
        e_arr = np.linspace(0.1, np.pi - 0.1, 10)
        dnu_dt, d2nu_dt2 = impl(e_arr, 0.4, 1000.0)
        assert np.all(dnu_dt > 0.0)
        assert np.all(d2nu_dt2 < 0.0)

    @pytest.mark.parametrize("impl", jit_variants(kepler.keplerian_z_derivatives_exact))
    @pytest.mark.parametrize("ecc", [0.0, 0.25, 0.7])
    def test_z_derivatives_exact_matches_finite_diff(self, impl, ecc: float) -> None:
        p_orb = 4200.0
        phi = 1.1
        a = 2.5 * C_VAL
        aop = 0.8
        inc = 1.2
        t_arr = np.linspace(0.0, p_orb * 0.9, 32)
        z_ex, vz_ex, az_ex = impl(t_arr, p_orb, ecc, phi, a, aop, inc)
        # Step size eps=1e-4 balances O(eps^2) truncation vs O(1/eps^2) float64 noise
        fd = kepler.keplerian_z_derivatives(
            t_arr,
            p_orb,
            ecc,
            phi,
            a,
            aop,
            inc,
            eps=1e-4,
        )
        np.testing.assert_allclose(z_ex, fd[0], rtol=1e-10, atol=1e-6)
        np.testing.assert_allclose(vz_ex, fd[1], rtol=1e-7, atol=1e-4)
        np.testing.assert_allclose(az_ex, fd[2], rtol=1e-5, atol=1e-2)


class TestDerivativeBoundsAndConnections:
    @pytest.mark.parametrize("impl", jit_variants(kepler.poly_projection_matrix))
    def test_poly_projection_recovers_polynomial(self, impl) -> None:
        t_arr = np.linspace(-2.0, 2.0, 50)
        coeffs_true = np.array([1.5, -0.4, 0.25, -0.05, 0.01])
        y_vec = sum(c * t_arr**k for k, c in enumerate(coeffs_true))
        p_mat = impl(t_arr, 4)
        coeffs_fit = kepler.mat_vec_product(p_mat, y_vec)
        err = kepler.polyfit_error(t_arr, y_vec, coeffs_fit)
        np.testing.assert_allclose(coeffs_fit, coeffs_true, rtol=1e-10, atol=1e-12)
        assert err < 1e-20

    @pytest.mark.parametrize("impl", jit_variants(kepler.find_derivative_connections))
    def test_find_derivative_connections_circular_exact(self, impl) -> None:
        fits, errs = impl(1.0, np.pi / 4.0, 0.0, 8, n_samples=50, res=0.5)
        assert fits.ndim == 2
        assert fits.shape[1] == 9
        assert errs.shape == (fits.shape[0],)
        assert np.max(errs) < 1e-14
        # For normalized circular orbit (p_orb = 2 pi => omega = 1), z^(k) = k! c_k
        d2 = 2.0 * fits[:, 2]
        d4 = 24.0 * fits[:, 4]
        mask = np.abs(d2) > 0.1
        np.testing.assert_allclose(-d4[mask] / d2[mask], 1.0, rtol=1e-7)

    def test_find_max_deriv_bounds_circular(self) -> None:
        x_orb = 1.5
        p_orb_min = 3600.0
        omega = 2.0 * np.pi / p_orb_min
        bounds = kepler.find_max_deriv_bounds(
            x_orb,
            np.pi / 4.0,
            p_orb_min,
            0.0,
            4,
            res=0.1,
        )
        expected = np.array(
            [x_orb * C_VAL * (omega**k) for k in range(5)],
            dtype=np.float64,
        )
        np.testing.assert_allclose(bounds, expected, rtol=2e-2)

    @pytest.mark.parametrize(("ideriv", "ecc_max"), [(0, 0.2), (2, 0.2), (4, 0.5)])
    def test_keplerian_derivative_bounds_matches_series(
        self,
        ideriv: int,
        ecc_max: float,
    ) -> None:
        a_max = 2.0
        omega_max = 1e-3
        k = np.arange(500)
        direct_sum = np.sum((ecc_max**k) * ((k + 1) ** ideriv))
        expected = a_max * (omega_max**ideriv) * direct_sum * C_VAL
        got = kepler.keplerian_derivative_bounds(ideriv, a_max, omega_max, ecc_max)
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_keplerian_derivative_bounds_high_ecc_exact_closed_form(self) -> None:
        """At ecc=0.99, closed-form sum (k+1)^4 e^k matches Eulerian formula."""
        a_max = 1.0
        omega_max = 1e-3
        ecc = 0.99
        eulerian_poly = 1.0 + 11.0 * ecc + 11.0 * ecc**2 + ecc**3
        expected = a_max * (omega_max**4) * (eulerian_poly / ((1.0 - ecc) ** 5)) * C_VAL
        got = kepler.keplerian_derivative_bounds(4, a_max, omega_max, ecc)
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    @pytest.mark.parametrize(("ideriv", "ecc_max"), [(-1, 0.1), (2, -0.1), (2, 1.0)])
    def test_keplerian_derivative_bounds_rejects_invalid(
        self,
        ideriv: int,
        ecc_max: float,
    ) -> None:
        with pytest.raises(ValueError, match=r"non-negative|0 <= ecc_max < 1"):
            kepler.keplerian_derivative_bounds(ideriv, 1.0, 1e-3, ecc_max)


class TestSimulateAndSolveOrbit:
    def test_simulate_keplerian_orbit_uses_orbital_semi_major_axis(self) -> None:
        rng = np.random.default_rng(42)
        t_arr = np.linspace(0.0, 500.0, 50)
        examples = kepler.simulate_keplerian_orbit(
            t_arr,
            n_orbits=8,
            rng=rng,
            p_pul_min=0.01,
            delta_p=0.0,
            p_dot_pul_max=0.0,
            p_orb_range=(3600.0, 3600.0),
            ecc_range=(0.1, 0.1),
            a_range=(1.5, 1.5),
            phi_range=(0.0, 2.0 * np.pi),
            om_range=(0.0, 2.0 * np.pi),
        )
        assert examples.shape == (8, 3, 50)
        # Position excursions must reflect a_orb = 1.5 lt-s (~4.5e8 m), not spin_down=0.
        excursion = np.ptp(examples[:, 0, :], axis=1)
        assert np.all(excursion > 0.1 * C_VAL)
        assert np.all(excursion < 3.5 * C_VAL)

    @pytest.mark.parametrize("use_default_guess", [True, False])
    def test_solve_keplerian_phases_short_array(
        self,
        use_default_guess: bool,
    ) -> None:
        """Recovers orbit on len(t_arr) = 20 (<= 25) without hardcoded slice failure."""
        p_orb_true = 4000.0
        x_true = 1.5
        ecc_true = 0.25
        aop_true = 0.7
        phi_true = 1.1
        const_true = 0.12
        lin_true = 2e-5

        t_arr = np.linspace(0.0, 3600.0, 20)
        t_centered = t_arr - np.mean(t_arr)
        z_true = kepler.keplerian_z(
            t_arr,
            p_orb_true,
            ecc_true,
            phi_true,
            x_true * C_VAL,
            aop_true,
            np.pi / 2,
        )
        recovered_phases = z_true / C_VAL + const_true + lin_true * t_centered

        orbit = kepler.KeplerianOrbit(
            semi_major_axis=x_true - 0.05,
            eccentricity=ecc_true - 0.03,
            omega_ecc=aop_true - 0.04,
            phi=phi_true - 0.04,
            p_orb=p_orb_true + 40.0,
        )
        if use_default_guess:
            res = orbit.solve_keplerian_phases(t_arr, recovered_phases)
        else:
            x0 = np.array(
                [
                    phi_true - 0.04,
                    p_orb_true + 40.0,
                    x_true - 0.05,
                    ecc_true - 0.03,
                    aop_true - 0.04,
                    const_true + 0.01,
                    lin_true,
                ],
                dtype=np.float64,
            )
            res = orbit.solve_keplerian_phases(
                t_arr,
                recovered_phases,
                initial_guess=x0,
            )

        assert res.fun < 1e-2
        np.testing.assert_allclose(res.x[1], p_orb_true, rtol=2e-2)
        np.testing.assert_allclose(res.x[2], x_true, rtol=2e-2)
        np.testing.assert_allclose(res.x[3], ecc_true, atol=3e-2)

    def test_solve_keplerian_phases_fit_mask_ignores_outliers(self) -> None:
        t_arr = np.linspace(0.0, 3600.0, 20)
        t_centered = t_arr - np.mean(t_arr)
        z_true = kepler.keplerian_z(
            t_arr,
            4000.0,
            0.2,
            0.9,
            1.4 * C_VAL,
            0.5,
            np.pi / 2,
        )
        phases = z_true / C_VAL + 0.05 + 1e-5 * t_centered
        phases[3] += 50.0
        phases[14] -= 50.0

        mask = np.ones(len(t_arr), dtype=bool)
        mask[[3, 14]] = False
        orbit = kepler.KeplerianOrbit(
            semi_major_axis=1.38,
            eccentricity=0.18,
            omega_ecc=0.48,
            phi=0.88,
            p_orb=4010.0,
        )
        res = orbit.solve_keplerian_phases(t_arr, phases, fit_mask=mask)
        assert res.fun < 1e-2

    def test_solve_keplerian_phases_rejects_bad_inputs(self) -> None:
        orbit = kepler.KeplerianOrbit(
            semi_major_axis=1.0,
            eccentricity=0.1,
            omega_ecc=0.2,
            phi=0.3,
            p_orb=1000.0,
        )
        t_arr = np.linspace(0.0, 100.0, 10)
        phases = np.zeros_like(t_arr)
        with pytest.raises(ValueError, match="Unsupported initial_guess"):
            orbit.solve_keplerian_phases(t_arr, phases, initial_guess="invalid")
        with pytest.raises(ValueError, match="fit_mask selects zero"):
            orbit.solve_keplerian_phases(
                t_arr,
                phases,
                fit_mask=np.zeros(10, dtype=bool),
            )


class TestPredictionTableAndGrid:
    def test_prediction_table_circular_zero_ecc_instantiation(self) -> None:
        omega_max = 2.0 * np.pi / 3600.0
        ptg = kepler.PredictionTableGenerator(
            x_orb_max=1.5,
            ecc_max=0.0,
            omega_max=omega_max,
            poly_deg=8,
            n_rad=np.pi / 2.0,
        )
        assert ptg.data.ndim == 2
        assert ptg.data.shape[1] == 9
        # Check d_k = k! c x w^k c_k scaling on d2 and d4 across om_ratios in [0.7, 1].
        d2 = ptg.data[:, 2]
        d4 = ptg.data[:, 4]
        mask = np.abs(d2) > 0.1 * np.max(np.abs(d2))
        ratios = -d4[mask] / d2[mask]
        assert np.all(ratios <= (omega_max**2) * (1.0 + 1e-5))
        assert np.all(ratios >= (0.7 * omega_max) ** 2 * (1.0 - 1e-5))

    def test_prediction_table_range_lookup_and_custom_keys(self) -> None:
        ptg = kepler.PredictionTableGenerator(
            x_orb_max=1.0,
            ecc_max=0.1,
            omega_max=2.0 * np.pi / 3600.0,
            poly_deg=6,
            n_rad=np.pi / 2.0,
        )
        # Default (2, 3, 4) -> compiled coord_function
        d_bins_3 = ptg.std_vals[[2, 3, 4]] / 5.0
        dic, coord_fn = ptg.generate_range_table(d_bins_3)
        for row in ptg.data[::50]:
            bounds = dic[coord_fn(row)]
            for col in range(ptg.poly_deg + 1):
                lo, hi = bounds[col]
                assert lo <= row[col] <= hi

        # Verify floor binning does not merge (-b, 0) into bin 0
        row_neg = np.zeros(ptg.poly_deg + 1, dtype=np.float64)
        row_neg[2] = -0.5 * d_bins_3[0]
        row_pos = np.zeros(ptg.poly_deg + 1, dtype=np.float64)
        row_pos[2] = 0.5 * d_bins_3[0]
        assert coord_fn(row_neg)[0] == -1
        assert coord_fn(row_pos)[0] == 0

        # Non-default key_indices (2, 3) previously raised IndexError
        d_bins_2 = ptg.std_vals[[2, 3]] / 5.0
        dic2, coord_fn2 = ptg.generate_range_table(d_bins_2, key_indices=(2, 3))
        sample_key = coord_fn2(ptg.data[0])
        assert len(sample_key) == 2
        assert sample_key in dic2

    def test_keplerian_param_limits_generate_grid_and_circular_edge(self) -> None:
        limits_ecc = kepler.KeplerianParamLimits(
            p_pul=(0.01, 0.010003),
            p_orb=(3600.0, 3602.0),
            x_orb=(1.0, 1.04),
            ecc=(0.1, 0.15),
            phi_plus_om=(0.0, 0.05),
            phi_minus_om=(0.0, 0.3),
        )
        grid = limits_ecc.generate_grid(tol=1.0, tsamp=0.02, tobs=100.0)
        assert grid.ndim == 7
        assert grid.shape[-1] == 6
        assert grid.size > 0
        # Output order: [p_pul, p_orb, x_orb, ecc, aop, phi]
        assert np.all((grid[..., 3] >= 0.1) & (grid[..., 3] < 0.15))

        # Circular limit (ecc == 0) and equal bounds avoid div-by-zero and empty grid
        limits_circ = kepler.KeplerianParamLimits(
            p_pul=(0.01, 0.01),
            p_orb=(3600.0, 3600.0),
            x_orb=(1.0, 1.0),
            ecc=(0.0, 0.0),
            phi_plus_om=(0.0, 0.05),
            phi_minus_om=(0.0, 0.0),
        )
        grid_circ = limits_circ.generate_grid(tol=1.0, tsamp=0.02, tobs=100.0)
        assert grid_circ.shape == (1, 1, 1, 1, grid_circ.shape[4], 1, 6)
        assert np.all(grid_circ[..., 3] == 0.0)

        limits_bad = kepler.KeplerianParamLimits(
            p_pul=(0.02, 0.01),
            p_orb=(3600.0, 3600.0),
            x_orb=(1.0, 1.0),
            ecc=(0.0, 0.0),
            phi_plus_om=(0.0, 0.1),
            phi_minus_om=(0.0, 0.1),
        )
        with pytest.raises(ValueError, match="exceeds max"):
            limits_bad.generate_grid(tol=1.0, tsamp=0.02, tobs=100.0)
