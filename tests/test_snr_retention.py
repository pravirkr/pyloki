"""Directed EP injection, exact-ephemeris refold, and S/N retention tests."""

from __future__ import annotations

import numpy as np
import pytest

from pyloki.config import PulsarSearchConfig
from pyloki.search import ep_search, measure_snr_retention, refold_candidates
from pyloki.simulation.pulse import PulseSignalConfig


@pytest.mark.parametrize("poly_basis", ["taylor", "chebyshev"])
def test_snr_retention_accel_smoke(poly_basis: str) -> None:
    true_period = 0.02
    true_acc = 12.0
    sig_cfg = PulseSignalConfig(
        period=true_period,
        dt=2e-4,
        nsamps=2**15,
        snr=30.0,
        ducy=0.08,
        mod_type="derivative",
        mod_kwargs={"acc": true_acc},
        seed=101,
    )
    tseries = sig_cfg.generate(shape="gaussian", phi0=0.37)

    search_cfg = PulsarSearchConfig(
        nsamps=tseries.nsamps,
        tsamp=tseries.dt,
        nbins=32,
        eta=1.0,
        param_limits=np.array([[-25.0, 25.0], [49.7, 50.3]]),
        bseg_brute=2**11,
        bseg_ffa=2**12,
        ducy_max=0.2,
        wtsp=1.5,
    )
    nseg = search_cfg.nsamps // search_cfg.bseg_ffa
    thresholds = np.full(nseg - 1, 2.0, dtype=np.float32)

    ret = measure_snr_retention(
        tseries,
        search_cfg,
        true_freq=1.0 / true_period,
        mod_type="derivative",
        mod_kwargs={"acc": true_acc},
        threshold_scheme=thresholds,
        ref_segs=[nseg // 2],
        poly_basis=poly_basis,
        max_sugg=2**14,
        seed=42,
    )
    assert ret.snr_exact > 20.0
    assert ret.rho_mid >= 0.80
    assert ret.rho_ascend >= 0.85
    assert ret.rho_ts >= 0.85


def test_ep_search_and_refold_candidates_with_arc_slice() -> None:
    true_period = 0.025
    sig_cfg = PulseSignalConfig(
        period=true_period,
        dt=2e-4,
        nsamps=2**15,
        snr=25.0,
        ducy=0.1,
        mod_type="derivative",
        mod_kwargs={"acc": -8.0},
        seed=7,
    )
    tseries = sig_cfg.generate(shape="gaussian")
    search_cfg = PulsarSearchConfig(
        nsamps=tseries.nsamps,
        tsamp=tseries.dt,
        nbins=32,
        eta=1.0,
        param_limits=np.array([[-20.0, 20.0], [39.7, 40.3]]),
        bseg_brute=2**11,
        bseg_ffa=2**12,
    )
    res = ep_search(
        tseries,
        search_cfg,
        arc_slice=(1, 7),
        n_runs=1,
        stop_snr=10.0,
        seed=42,
        quiet=True,
    )
    assert res.dyp.nsegments == 6
    assert res.best_score > 10.0
    assert res.best_params.shape == (len(res.param_names),)

    snrs, profs = refold_candidates(
        tseries,
        res.best_params[np.newaxis, :],
        search_cfg.nbins,
        param_names=res.param_names,
        widths=search_cfg.score_widths,
    )
    assert snrs.shape == (1,)
    assert profs.shape == (1, 2, search_cfg.nbins)
    assert snrs[0] > 10.0

    # Empty candidate batch handled cleanly
    empty_snrs, empty_profs = refold_candidates(
        tseries,
        np.empty((0, 2), dtype=np.float64),
        search_cfg.nbins,
        param_names=search_cfg.param_names,
    )
    assert empty_snrs.size == 0
    assert empty_profs.shape == (0, 2, search_cfg.nbins)


def test_ep_search_asymmetric_arc_slice_preserves_t_ref() -> None:
    """Asymmetric arc_slice exposes t_ref so refold stays coherent."""
    nsamps = 2**15
    dt = 1e-3
    tobs = nsamps * dt
    true_freq = 1.0 / 0.015
    true_accel = -45.0
    sig = PulseSignalConfig(
        period=1.0 / true_freq,
        dt=dt,
        nsamps=nsamps,
        snr=30.0,
        ducy=0.1,
        mod_type="derivative",
        mod_kwargs={"acc": true_accel},
        seed=99,
    )
    tseries = sig.generate(shape="gaussian")
    search_cfg = PulsarSearchConfig(
        nsamps=nsamps,
        tsamp=dt,
        nbins=32,
        eta=1.0,
        param_limits=np.array([[-80.0, 80.0], [true_freq - 0.15, true_freq + 0.15]]),
        bseg_brute=2**10,
        bseg_ffa=2**12,
        use_fourier=False,
    )
    res = ep_search(
        tseries,
        search_cfg,
        n_runs=1,
        ref_segs=[2],
        arc_slice=(0, 4),
        poly_basis="taylor",
        use_moving_grid=True,
        snr_final=8.0,
        seed=11,
        quiet=True,
    )
    assert res.arc_slice == (0, 4)
    # First 4 of 8 segments -> midpoint is tobs / 4 (not tobs / 2)
    assert res.t_ref == pytest.approx(0.25 * tobs, rel=1e-6)
    snrs_good, _ = refold_candidates(
        tseries,
        res.best_params[np.newaxis, :],
        search_cfg.nbins,
        param_names=res.param_names,
        mod_tref=res.t_ref,
        widths=search_cfg.score_widths,
    )
    assert float(snrs_good[0]) > 15.0
