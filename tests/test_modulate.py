"""Unit tests for `pyloki.simulation.modulate`."""

from __future__ import annotations

import numpy as np
import pytest

from pyloki.simulation.modulate import CircularModulating, CircularT0Modulating
from pyloki.utils.misc import C_VAL


class TestCircularModulating:
    def test_requires_x_orb_or_m_c(self) -> None:
        with pytest.raises(ValueError, match="Either x_orb or m_c"):
            CircularModulating(p_orb=1000.0, psi=0.5)

    @pytest.mark.parametrize("psi", [0.0, np.pi])
    def test_to_derivatives_series_nodal_odd_terms_nonzero(self, psi: float) -> None:
        p_orb = 1000.0
        omega = 2.0 * np.pi / p_orb
        orbit = CircularModulating(p_orb=p_orb, psi=psi, x_orb=0.2)
        derivs = orbit.to_derivatives()
        coeffs = orbit.to_derivatives_series(7)["coeffs"]
        assert abs(coeffs[5]) > 0.0
        np.testing.assert_allclose(coeffs[5], derivs["crackle"], rtol=1e-12)
        np.testing.assert_allclose(coeffs[7], -(omega**2) * coeffs[5], rtol=1e-12)


class TestCircularT0Modulating:
    def test_matches_circular_modulating_with_a_in_meters(self) -> None:
        p_orb = 5000.0
        t_ref = 120.0
        t0 = 30.0
        x_orb = 0.15
        phi = 2.0 * np.pi * ((t_ref - t0) % p_orb) / p_orb
        t_arr = np.linspace(0.0, 1000.0, 64)

        m_t0 = CircularT0Modulating(a=x_orb * C_VAL, p_orb=p_orb, t0=t0)
        m_psi = CircularModulating(p_orb=p_orb, psi=phi, x_orb=x_orb)
        np.testing.assert_allclose(
            m_t0.generate(t_arr, t_ref),
            m_psi.generate(t_arr, t_ref),
            rtol=1e-12,
        )
