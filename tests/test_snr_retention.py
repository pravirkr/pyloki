"""Directed EP injection, exact-ephemeris refold, and S/N retention tests."""

from __future__ import annotations

import numpy as np
import pytest

from pyloki.config import PulsarSearchConfig
from pyloki.search import (
    decompose_fold_quality,
    ep_search,
    ep_sliding_arc_search,
    measure_snr_retention,
    refold_candidates,
)
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


def test_ep_sliding_arc_search_and_jerk_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-P-FFA sliding-arc EP shares base folds and jerk guard warns cleanly."""
    from pyloki import config as cfg_mod  # noqa: PLC0415

    nsamps = 2**14
    dt = 1e-3
    true_freq = 50.0
    sig = PulseSignalConfig(
        period=1.0 / true_freq,
        dt=dt,
        nsamps=nsamps,
        snr=25.0,
        ducy=0.1,
        mod_type="derivative",
        mod_kwargs={"acc": 10.0},
        seed=42,
    )
    ts = sig.generate(shape="gaussian")
    # Trigger jerk-smearing warning via aggressive compact-binary parameters
    warnings_seen: list[str] = []
    monkeypatch.setattr(
        cfg_mod.logger,
        "warning",
        lambda msg, *args: warnings_seen.append(msg % args if args else str(msg)),
    )
    cfg_warn = PulsarSearchConfig(
        nsamps=nsamps,
        tsamp=dt,
        nbins=32,
        eta=1.0,
        param_limits=np.array([[-30.0, 30.0], [49.8, 50.2]]),
        prune_poly_order=2,
        bseg_brute=2**10,
        bseg_ffa=2**11,
        use_fourier=False,
        p_orb_min=30.0,
        x_max=5.0,
    )
    assert any("unmodeled orbital jerk" in m for m in warnings_seen)

    results = ep_sliding_arc_search(
        ts,
        cfg_warn,
        arc_len=4,
        stride=2,
        threshold_scheme=np.full(3, 1.0, dtype=np.float32),
        n_runs=1,
        quiet=True,
    )
    assert len(results) == 3  # windows: (0, 4), (2, 6), (4, 8)
    assert [r.arc_slice for r in results] == [(0, 4), (2, 6), (4, 8)]
    # All windows share memory with the single base P-FFA fold tensor
    assert np.shares_memory(results[0].dyp.fold, results[1].dyp.fold)
    assert all(r.best_score > 10.0 for r in results)


def test_fold_quality_decomposition_and_randomized_subbin() -> None:
    """Decompose finite-binning vs FFA loss and check randomized sub-bin retention."""
    rng = np.random.default_rng(73)
    nsamps = 2**15
    dt = 2e-4
    tobs = nsamps * dt
    nbins = 32
    # Randomize fractional frequency bin offset in (0, 1)
    frac_pos = float(rng.uniform(0.15, 0.85))
    df_bin = 1.0 / (nbins * tobs)
    true_freq = 50.0 + frac_pos * df_bin
    sig = PulseSignalConfig(
        period=1.0 / true_freq,
        dt=dt,
        nsamps=nsamps,
        snr=35.0,
        ducy=0.08,
        mod_type="derivative",
        mod_kwargs={"acc": 8.0},
        seed=73,
    )
    ts = sig.generate(shape="gaussian", phi0=float(rng.uniform(0.0, 1.0)))
    cfg_ffa = PulsarSearchConfig(
        nsamps=nsamps,
        tsamp=dt,
        nbins=nbins,
        eta=1.0,
        param_limits=np.array([[-15.0, 15.0], [49.7, 50.3]]),
        bseg_brute=2**11,
        bseg_ffa=nsamps,
        use_fourier=True,
    )
    qual = decompose_fold_quality(
        ts,
        cfg_ffa,
        true_freq=true_freq,
        mod_type="derivative",
        mod_kwargs={"acc": 8.0},
    )
    assert 0.8 <= qual["binning_retention"] <= 1.05
    assert 0.85 <= qual["ffa_retention_vs_nbins"] <= 1.1
    assert qual["total_retention"] > 0.8

    cfg_ep = PulsarSearchConfig(
        nsamps=nsamps,
        tsamp=dt,
        nbins=nbins,
        eta=1.0,
        param_limits=np.array([[-15.0, 15.0], [49.7, 50.3]]),
        prune_poly_order=2,
        bseg_brute=2**11,
        bseg_ffa=2**12,
        use_fourier=False,
    )
    ret = measure_snr_retention(
        ts,
        cfg_ep,
        true_freq=true_freq,
        mod_type="derivative",
        mod_kwargs={"acc": 8.0},
        threshold_scheme=np.full(7, 2.0, dtype=np.float32),
        n_runs=1,
        ref_segs=[4],
        seed=73,
    )
    assert ret.rho_ascend >= 0.90
    assert np.isfinite(ret.delta_f_over_df)
    assert np.isfinite(ret.delta_a_over_da)


@pytest.mark.slow
@pytest.mark.parametrize("ecc", [0.0, 0.1, 0.2])
def test_snr_retention_eccentric_sweep_slow(ecc: float) -> None:
    """Slow multi-eccentricity EP S/N retention sweep over e in [0, 0.2]."""
    from pyloki.utils.misc import C_VAL  # noqa: PLC0415

    nsamps = 2**16
    dt = 2e-4
    true_period = 0.02
    kep_kwargs = {
        "p_orb": 300.0,
        "a": 0.05 * C_VAL,
        "ecc": ecc,
        "om": 0.4,
        "phi": 0.7,
        "inc": 0.5 * np.pi,
    }
    sig = PulseSignalConfig(
        period=true_period,
        dt=dt,
        nsamps=nsamps,
        snr=35.0,
        ducy=0.08,
        mod_type="keplerian",
        mod_kwargs=kep_kwargs,
        seed=55,
    )
    ts = sig.generate(shape="gaussian")
    cfg = PulsarSearchConfig(
        nsamps=nsamps,
        tsamp=dt,
        nbins=32,
        eta=1.0,
        param_limits=np.array([[-1.0, 1.0], [-40.0, 40.0], [49.7, 50.3]]),
        prune_poly_order=3,
        bseg_brute=2**11,
        bseg_ffa=2**13,
        use_fourier=False,
    )
    ret = measure_snr_retention(
        ts,
        cfg,
        true_freq=1.0 / true_period,
        mod_type="keplerian",
        mod_kwargs=kep_kwargs,
        threshold_scheme=np.full(7, 2.0, dtype=np.float32),
        n_runs=1,
        ref_segs=[4],
        seed=55,
    )
    assert ret.rho_ascend >= 0.80
