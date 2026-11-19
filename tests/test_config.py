"""Unit tests for `pyloki.config` parameter limits and search configuration."""

from __future__ import annotations

import numpy as np
import pytest

from pyloki.config import (
    ParamLimits,
    PulsarSearchConfig,
    compute_max_coherent_subarc,
    compute_min_nsegments,
)
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
        # Numba bundles declare param_limits as f8[:, ::1]; anything else
        # would pass here and fail later inside FFA/pruning setup.
        assert cfg.param_limits.dtype == np.float64
        assert cfg.param_limits.flags.c_contiguous

    def test_non_contiguous_param_limits_are_made_contiguous(self) -> None:
        limits = np.asfortranarray([[-5.0, 5.0], [99.0, 101.0]])
        assert not limits.flags.c_contiguous
        cfg = PulsarSearchConfig(
            nsamps=1024,
            tsamp=1e-3,
            nbins=16,
            eta=1.0,
            param_limits=limits,
        )
        assert cfg.param_limits.flags.c_contiguous
        np.testing.assert_array_equal(cfg.param_limits, limits)

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


class TestSubArcGuardsAndSerialization:
    def test_compute_min_nsegments_and_max_subarc(self) -> None:
        nseg_cheb = compute_min_nsegments(
            tobs=3600.0,
            p_orb_min=7200.0,
            x_max=2.0,
            p_min=0.002,
            nbins=64,
            eta=1.0,
            use_chebyshev=True,
        )
        nseg_tay = compute_min_nsegments(
            tobs=3600.0,
            p_orb_min=7200.0,
            x_max=2.0,
            p_min=0.002,
            nbins=64,
            eta=1.0,
            use_chebyshev=False,
        )
        assert nseg_cheb >= 1
        assert (nseg_cheb & (nseg_cheb - 1)) == 0
        assert nseg_tay >= nseg_cheb

        t2 = compute_max_coherent_subarc(2, 7200.0, 2.0, 0.002, 64)
        t3 = compute_max_coherent_subarc(3, 7200.0, 2.0, 0.002, 64)
        t4 = compute_max_coherent_subarc(4, 7200.0, 2.0, 0.002, 64)
        assert 0.0 < t2 < t3 < t4

        with pytest.raises(ValueError, match="poly_order"):
            compute_max_coherent_subarc(1, 7200.0, 2.0, 0.002, 64)

    def test_config_roundtrip_and_estimate_resources(self, tmp_path) -> None:
        cfg = PulsarSearchConfig(
            nsamps=2**16,
            tsamp=1e-3,
            nbins=32,
            eta=1.0,
            param_limits=np.array([[-5.0, 5.0], [10.0, 10.1]]),
            bseg_brute=2**10,
            bseg_ffa=2**12,
        )
        d = cfg.to_dict()
        cfg2 = PulsarSearchConfig.from_dict(d)
        assert cfg2.nsamps == cfg.nsamps
        assert cfg2.nbins == cfg.nbins
        np.testing.assert_allclose(cfg2.param_limits, cfg.param_limits)

        toml_path = tmp_path / "search.toml"
        toml_path.write_text(
            "[search]\n"
            f"nsamps = {cfg.nsamps}\n"
            f"tsamp = {cfg.tsamp}\n"
            f"nbins = {cfg.nbins}\n"
            f"eta = {cfg.eta}\n"
            "param_limits = [[-5.0, 5.0], [10.0, 10.1]]\n"
            f"bseg_brute = {cfg.bseg_brute}\n"
            f"bseg_ffa = {cfg.bseg_ffa}\n",
        )
        cfg3 = PulsarSearchConfig.from_toml(toml_path, section="search")
        np.testing.assert_allclose(cfg3.param_limits, cfg.param_limits)

        res = cfg.estimate_resources()
        assert res["nsegments_ffa"] == 16
        assert res["nparam_vol_ffa"] > 0
        assert res["ffa_tensor_bytes"] > 0
        assert res["total_ffa_evals"] >= res["nsegments_ffa"] * res["nparam_vol_ffa"]
