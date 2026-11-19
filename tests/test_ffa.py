"""Unit tests for `pyloki.ffa`: `FFAPlan` and `DynamicProgramming`."""

from __future__ import annotations

import numpy as np
import pytest

from pyloki.config import PulsarSearchConfig
from pyloki.core.fold import ffa_taylor_resolve
from pyloki.detection import scoring
from pyloki.ffa import DynamicProgramming, FFAPlan
from pyloki.io.timeseries import TimeSeries
from pyloki.simulation.pulse import PulseSignalConfig
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


@pytest.mark.parametrize("use_fourier", [False, True])
def test_execute_scored_matches_full_execute(use_fourier: bool) -> None:
    sig = PulseSignalConfig(
        period=0.05,
        dt=1e-3,
        nsamps=2**13,
        snr=18.0,
        ducy=0.1,
        seed=42,
    )
    ts = sig.generate(shape="gaussian")
    cfg = PulsarSearchConfig(
        nsamps=ts.nsamps,
        tsamp=ts.dt,
        nbins=32,
        eta=1.0,
        param_limits=np.array([[19.8, 20.2]]),
        bseg_brute=2**10,
        bseg_ffa=ts.nsamps,
        use_fourier=use_fourier,
    )
    dyp_full = DynamicProgramming(ts, cfg)
    dyp_full.initialize()
    dyp_full.execute(show_progress=False)
    folds_norm = dyp_full.get_fold_norm()
    snrs_full = np.max(
        scoring.boxcar_snr_nd(folds_norm, cfg.score_widths, 1.0), axis=-1
    )
    flat_full = snrs_full.reshape(-1)
    thresh = float(np.percentile(flat_full, 85))
    want_idx = np.flatnonzero(flat_full >= thresh)
    want_scores = flat_full[want_idx]

    dyp_scored = DynamicProgramming(ts, cfg)
    dyp_scored.initialize()
    got_idx, got_scores = dyp_scored.execute_scored(thresh, tile_size=17)

    np.testing.assert_array_equal(got_idx, want_idx)
    np.testing.assert_allclose(got_scores, want_scores, rtol=1e-4, atol=1e-4)


def test_slice_segments_shares_buffer() -> None:
    rng = np.random.default_rng(7)
    n = 2**12
    ts = TimeSeries(rng.standard_normal(n), np.ones(n), 1e-3)
    cfg = PulsarSearchConfig(
        nsamps=n,
        tsamp=1e-3,
        nbins=16,
        eta=1.0,
        param_limits=np.array([[10.0, 10.2]]),
        bseg_brute=2**9,
        bseg_ffa=2**10,
    )
    dyp = DynamicProgramming(ts, cfg)
    dyp.initialize()
    dyp.execute(show_progress=False)
    assert dyp.nsegments == 4
    sub = dyp.slice_segments(1, 3)
    assert sub.nsegments == 2
    assert np.shares_memory(sub.fold, dyp.fold)
    with pytest.raises(ValueError, match="Invalid segment slice"):
        dyp.slice_segments(2, 2)


def test_execute_scored_advances_state_and_rejects_multi_segment() -> None:
    """execute_scored must advance metadata to final_level & guard nsegments."""
    rng = np.random.default_rng(13)
    n = 2**12
    ts = TimeSeries(rng.standard_normal(n), np.ones(n), 1e-3)

    # Full-obs FFA (n_seg_out == 1) must advance metadata so hit_idx fits param_arr
    cfg_full = PulsarSearchConfig(
        nsamps=n,
        tsamp=1e-3,
        nbins=16,
        eta=1.0,
        param_limits=np.array([[10.0, 10.2]]),
        bseg_brute=2**9,
        bseg_ffa=n,
    )
    dyp_scored = DynamicProgramming(ts, cfg_full)
    dyp_scored.initialize()
    hit_idx, hit_scores = dyp_scored.execute_scored(threshold=-100.0, tile_size=13)
    assert dyp_scored.ffa_level == cfg_full.niters_ffa
    expected_vol = int(np.prod([len(a) for a in dyp_scored.param_arr]))
    assert hit_idx.size == expected_vol
    assert int(np.max(hit_idx)) == expected_vol - 1
    assert hit_scores.size == expected_vol

    # Multi-segment P-FFA (bseg_ffa < nsamps => n_seg_out > 1) must raise ValueError
    cfg_multi = PulsarSearchConfig(
        nsamps=n,
        tsamp=1e-3,
        nbins=16,
        eta=1.0,
        param_limits=np.array([[10.0, 10.2]]),
        bseg_brute=2**9,
        bseg_ffa=2**10,
    )
    dyp_multi = DynamicProgramming(ts, cfg_multi)
    dyp_multi.initialize()
    with pytest.raises(ValueError, match="nsegments"):
        dyp_multi.execute_scored(threshold=0.0)
