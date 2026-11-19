"""Unit tests for `pyloki.ffa`: `FFAPlan` and `DynamicProgramming`."""

from __future__ import annotations

import numpy as np

from pyloki.config import PulsarSearchConfig
from pyloki.core.fold import ffa_taylor_resolve
from pyloki.ffa import DynamicProgramming, FFAPlan
from pyloki.io.timeseries import TimeSeries
from pyloki.utils import np_utils


def test_ffa_plan_dtypes_and_resolve_coordinates_single_param() -> None:
    cfg = PulsarSearchConfig(
        nsamps=1024,
        tsamp=1e-3,
        nbins=16,
        eta=1.0,
        param_limits=np.array([[99.0, 101.0]]),
        bseg_brute=128,
        bseg_ffa=1024,
    )
    plan = FFAPlan(cfg)
    assert plan.ncoords.dtype == np.int64
    assert plan.param_counts.dtype == np.int64

    coords = plan.resolve_coordinates_freq()
    assert len(coords) == plan.n_levels
    assert len(coords[0]) == 0
    for lvl in range(1, plan.n_levels):
        cart = np_utils.cartesian_prod_st(plan.params[lvl])
        assert len(coords[lvl]) == len(cart)
        p_idx0, shift0 = ffa_taylor_resolve(
            cart[0],
            plan.param_counts[lvl - 1],
            cfg.param_limits,
            lvl,
            1,
            cfg.tseg_brute,
            cfg.nbins,
        )
        assert int(coords[lvl][0]["idx"]) == int(p_idx0[0])
        assert coords[lvl][0]["shift"] == np.float32(shift0)


def test_ffa_plan_resolve_coordinates_multi_param() -> None:
    cfg = PulsarSearchConfig(
        nsamps=1024,
        tsamp=1e-3,
        nbins=16,
        eta=1.0,
        param_limits=np.array([[-10.0, 10.0], [99.0, 101.0]]),
        bseg_brute=128,
        bseg_ffa=1024,
    )
    plan = FFAPlan(cfg)
    coords = plan.resolve_coordinates()
    assert len(coords) == plan.n_levels
    assert len(coords[0]) == 0
    for lvl in range(1, plan.n_levels):
        cart = np_utils.cartesian_prod_st(plan.params[lvl])
        assert len(coords[lvl]) == len(cart)
        prev_counts = plan.param_counts[lvl - 1]
        p_idx_t, shift_t = ffa_taylor_resolve(
            cart[0],
            prev_counts,
            cfg.param_limits,
            lvl,
            0,
            cfg.tseg_brute,
            cfg.nbins,
        )
        p_idx_h, shift_h = ffa_taylor_resolve(
            cart[0],
            prev_counts,
            cfg.param_limits,
            lvl,
            1,
            cfg.tseg_brute,
            cfg.nbins,
        )
        flat_t = int(p_idx_t[0] * prev_counts[1] + p_idx_t[1])
        flat_h = int(p_idx_h[0] * prev_counts[1] + p_idx_h[1])
        assert int(coords[lvl][0]["i_tail"]) == flat_t
        assert coords[lvl][0]["shift_tail"] == np.float32(shift_t)
        assert int(coords[lvl][0]["i_head"]) == flat_h
        assert coords[lvl][0]["shift_head"] == np.float32(shift_h)


def test_get_fold_norm_fourier_respects_cfg_nbins() -> None:
    rng = np.random.default_rng(42)
    nsamps, dt, nbins = 4096, 1e-4, 30
    ts = TimeSeries(
        rng.standard_normal(nsamps).astype(np.float32),
        np.ones(nsamps, np.float32),
        dt,
    )
    cfg = PulsarSearchConfig(
        nsamps=nsamps,
        tsamp=dt,
        nbins=nbins,
        eta=1.0,
        param_limits=np.array([[99.0, 101.0]]),
        bseg_brute=512,
        bseg_ffa=4096,
        use_fourier=True,
    )
    dyp = DynamicProgramming(ts, cfg)
    dyp.initialize()
    dyp.execute(show_progress=False)
    norm = dyp.get_fold_norm()
    assert norm.shape[-1] == nbins
    expected = np.fft.irfft(dyp.fold, n=nbins).astype(np.float32)[0]
    np.testing.assert_allclose(norm, expected[..., 0, :] / np.sqrt(expected[..., 1, :]))
