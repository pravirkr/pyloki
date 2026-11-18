"""Unit tests for `pyloki.config` parameter limits and search configuration."""

from __future__ import annotations

import numpy as np
import pytest

from pyloki.config import ParamLimits, PulsarSearchConfig
from pyloki.utils import transforms
from pyloki.utils.misc import C_VAL


class TestParamLimits:
    def test_get_cheby_limits_2param_order_and_values(self) -> None:
        tobs = 100.0
        t_s = tobs / 2
        limits = ParamLimits.from_taylor(freq=(99.0, 101.0), accel=(-10.0, 20.0))
        got = limits.get_cheby_limits(tobs)
        # Output layout: freq first, then ascending alpha_0, alpha_1, alpha_2.
        assert len(got) == 4
        assert got[0] == pytest.approx((99.0, 101.0))
        # x^2 = (T_0 + T_2)/2 => alpha_0 = alpha_2 = a * t_s^2 / 4, alpha_1 = 0
        expected_a = (-10.0 * t_s**2 / 4, 20.0 * t_s**2 / 4)
        assert got[1] == pytest.approx(expected_a)
        assert got[2] == pytest.approx((0.0, 0.0))
        assert got[3] == pytest.approx(expected_a)

    def test_get_cheby_limits_3param_matches_reversed_taylor_to_cheby(self) -> None:
        tobs = 60.0
        t_s = tobs / 2
        limits = ParamLimits.from_taylor(
            freq=(99.5, 100.5),
            accel=(-5.0, 15.0),
            jerk=(-0.2, 0.4),
        )
        got = limits.get_cheby_limits(tobs)
        # Output layout: freq first, then ascending alpha_0..alpha_3.
        assert len(got) == 5
        assert got[0] == pytest.approx((99.5, 100.5))
        sample = transforms.taylor_to_cheby(np.array([0.4, 15.0, 0.0, 0.0]), t_s)
        # sample has descending coefficients, so compare reversed against got[1:].
        for idx, alpha_val in enumerate(sample[::-1], start=1):
            lo, hi = got[idx]
            assert lo <= alpha_val <= hi
        assert got[4] == pytest.approx((-0.2 * t_s**3 / 24, 0.4 * t_s**3 / 24))

    def test_from_upper_negative_accel_and_negative_upper_limits(self) -> None:
        t_obs = 100.0
        limits_neg_acc = ParamLimits.from_upper(
            100.0, [0.0, -300.0], (-1.0, 1.0), t_obs
        )
        assert np.all(limits_neg_acc.limits[:, 0] < limits_neg_acc.limits[:, 1])

        limits_neg_upper = ParamLimits.from_upper(
            100.0, [-5.0, 100.0], (-6.0, -4.0), t_obs
        )
        assert np.all(limits_neg_upper.limits[:, 0] < limits_neg_upper.limits[:, 1])

        # Every shifted derivative in [u_min, u_max] x [-t_obs/2, t_obs/2] is bounded.
        for u in (-6.0, -5.0, -4.0):
            for dt in (-t_obs / 2, 0.0, t_obs / 2):
                shifted = transforms.shift_taylor_params(
                    np.array([u, 100.0, 0.0, 0.0]), dt
                )
                assert limits_neg_upper.limits[1, 0] <= shifted[1]
                assert shifted[1] <= limits_neg_upper.limits[1, 1]
                f_obs = 100.0 * (1.0 - shifted[2] / C_VAL)
                assert limits_neg_upper.limits[2, 0] <= f_obs
                assert f_obs <= limits_neg_upper.limits[2, 1]

    def test_from_upper_captures_interior_quadratic_accel_extremum(self) -> None:
        # a(dt) = 500 + 100*dt - 5*dt^2 on [-20, 20] is -3500 and +500 at the
        # endpoints, but reaches +1000 at interior vertex dt = +10.
        limits = ParamLimits.from_upper(
            100.0, [-10.0, 100.0, 500.0], (-10.01, -10.0), 40.0
        )
        assert limits.limits[2, 1] == pytest.approx(1000.0)


class TestPulsarSearchConfigValidation:
    @pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32])
    def test_accepts_numeric_ndarrays(self, dtype) -> None:
        cfg = PulsarSearchConfig(
            nsamps=1024,
            tsamp=1e-3,
            nbins=16,
            eta=1.0,
            param_limits=np.array([[99, 101]], dtype=dtype),
        )
        assert cfg.nparams == 1

    def test_rejects_invalid_param_limits_shapes_and_values(self) -> None:
        common_kw = {"nsamps": 1024, "tsamp": 1e-3, "nbins": 16, "eta": 1.0}
        with pytest.raises(TypeError, match="numpy ndarray"):
            PulsarSearchConfig(param_limits=[(99.0, 101.0)], **common_kw)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="shape"):
            PulsarSearchConfig(param_limits=np.array([99.0, 101.0]), **common_kw)
        with pytest.raises(ValueError, match="1-5 elements"):
            PulsarSearchConfig(
                param_limits=np.tile([0.0, 1.0], (6, 1)),
                **common_kw,
            )
        with pytest.raises(ValueError, match="min < max"):
            PulsarSearchConfig(
                param_limits=np.array([[np.nan, 101.0]]),
                **common_kw,
            )

    def test_fourier_rejects_odd_nbins_but_time_domain_allows_it(self) -> None:
        with pytest.raises(ValueError, match="must be even when use_fourier=True"):
            PulsarSearchConfig(
                nsamps=1024,
                tsamp=1e-3,
                nbins=15,
                eta=1.0,
                param_limits=np.array([[99.0, 101.0]]),
                use_fourier=True,
            )
        cfg = PulsarSearchConfig(
            nsamps=1024,
            tsamp=1e-3,
            nbins=15,
            eta=1.0,
            param_limits=np.array([[99.0, 101.0]]),
            use_fourier=False,
        )
        assert cfg.nbins == 15

    def test_bseg_brute_validation_and_default_clamping(self) -> None:
        with pytest.raises(ValueError, match="must be < nsamps"):
            PulsarSearchConfig(
                nsamps=1024,
                tsamp=1e-3,
                nbins=16,
                eta=1.0,
                param_limits=np.array([[99.0, 101.0]]),
                bseg_brute=1024,
            )

        # Short observation: nsamps * tsamp * f_min < 2**init_levels -> levels <= 0
        cfg_short = PulsarSearchConfig(
            nsamps=1024,
            tsamp=1e-4,
            nbins=16,
            eta=1.0,
            param_limits=np.array([[-1.0, 1.0], [10.0, 12.0]]),
        )
        assert 1 <= cfg_short.bseg_brute < cfg_short.nsamps

        # Very long period/fast sampling edge where levels >= log2(nsamps)
        cfg_many_cycles = PulsarSearchConfig(
            nsamps=64,
            tsamp=1.0,
            nbins=16,
            eta=1.0,
            param_limits=np.array([[500.0, 501.0]]),
        )
        assert 1 <= cfg_many_cycles.bseg_brute < cfg_many_cycles.nsamps
