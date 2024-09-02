from __future__ import annotations

import numpy as np
from numba import typed

from pyloki import ffa, prune, scores, thresholds
from pyloki.base import SearchConfig
from pyloki.periodogram import Periodogram
from pyloki.timeseries import SignalConfig, TimeSeries
from pyloki.utils import get_logger

logger = get_logger(__name__)


def search_pulsar(
    tim_data: TimeSeries,
    search_cfg: SearchConfig,
    true_params: np.ndarray,
) -> tuple:
    dyp = ffa.DynamicProgramming(tim_data, search_cfg)
    dyp.initialize()
    dyp.do_iterations()
    folds = dyp.fold[0][..., 0, :] / np.sqrt(dyp.fold[0][..., 1, :])

    true_period_idx, _ = dyp.dp_funcns.resolve(
        true_params,
        dyp.param_arr,
        dyp.ffa_level,
        0,
    )

    widths = scores.generate_width_trials(search_cfg.nbins, ducy_max=0.1, wtsp=1)
    snrs = scores.boxcar_snr(folds, widths)
    periods = dyp.param_arr[-1]
    best_indices = np.unravel_index(np.nanargmax(snrs), snrs.shape)

    logger.info("\n*** Search results ***")
    logger.info(f"True period index: {true_period_idx}")
    logger.info(f"Search best index: {best_indices}")
    if len(best_indices) > 2:
        accels = dyp.param_arr[-2]
        logger.info(f"Best acceleration = {accels[best_indices[-3]]}")
    logger.info(f"Best period = {periods[best_indices[-2]]}")
    logger.info(f"Best width = {widths[best_indices[-1]]}")
    logger.info(f"Best S/N = {np.nanmax(snrs)}")

    if len(best_indices) > 2:
        snrs = snrs[best_indices[-3]]
    pgram = Periodogram(widths, periods, snrs, tim_data.tobs)
    return dyp, pgram


def pruning_search(
    dyp: ffa.DynamicProgramming,
    target_snr: float,
    snr_margin: float,
) -> prune.Pruning:
    bound_scheme = thresholds.threshold_scheme_bound(
        dyp.nchunks,
        target_snr,
        snr_margin,
    )
    trials_scheme = thresholds.threshold_scheme_trials(dyp.nchunks, dyp.nparams)
    threshold_scheme = np.minimum(bound_scheme, trials_scheme)
    prn = prune.Pruning(dyp, threshold_scheme, max_sugg=131072)
    prn.initialize(seg_ref=12)
    return prn


def search_pulsar_pruning(
    tim_data: TimeSeries,
    search_cfg: SearchConfig,
    true_params: np.ndarray,
) -> tuple:
    dyp = ffa.DynamicProgramming(tim_data, search_cfg)
    dyp.initialize()
    dyp.do_iterations("prune")
    folds = dyp.fold[0][..., 0, :] / np.sqrt(dyp.fold[0][..., 1, :])

    true_period_idx, _ = dyp.dp_funcns.resolve(
        true_params,
        dyp.param_arr,
        dyp.ffa_level,
        0,
    )

    widths = scores.generate_width_trials(search_cfg.nbins, ducy_max=0.1, wtsp=1)
    snrs = scores.boxcar_snr(folds, widths)
    periods = dyp.param_arr[-1]
    best_indices = np.unravel_index(np.nanargmax(snrs), snrs.shape)

    logger.info("\n*** Search results ***")
    logger.info(f"True period index: {true_period_idx}")
    logger.info(f"Search best index: {best_indices}")
    if len(best_indices) > 2:
        accels = dyp.param_arr[-2]
        logger.info(f"Best acceleration = {accels[best_indices[-3]]}")
    logger.info(f"Best period = {periods[best_indices[-2]]}")
    logger.info(f"Best width = {widths[best_indices[-1]]}")
    logger.info(f"Best S/N = {np.nanmax(snrs)}")

    if len(best_indices) > 2:
        snrs = snrs[best_indices[-3]]
    pgram = Periodogram(widths, periods, snrs, tim_data.tobs)
    return dyp, pgram


def test_pulsar_search_periodicity(
    psr_type: str = "msp",
    tol: float | None = None,
    fold_bins: int | None = None,
) -> ffa.DynamicProgramming:
    if psr_type == "msp":
        period = 1.012345678910111213 * 1e-2  # (s)
        dt = 8.192e-5
    elif psr_type == "regular":
        period = 1.212345678910111213  # (s)
        dt = 8.192e-3
    cfg = SignalConfig(period=period, dt=dt)
    tim_data = TimeSeries.generate_signal(cfg)

    param_limits = typed.List([(cfg.period * 0.5, cfg.period * 1.6)])
    if tol is None:
        tol = cfg.tol
    if fold_bins is None:
        fold_bins = cfg.fold_bins
    logger.info(f"Using tolerance: {tol}, fold_bins: {fold_bins}")
    search_cfg = SearchConfig(cfg.nsamps, cfg.dt, tol, fold_bins, param_limits)
    true_params = np.array([cfg.period])
    dyp, pgram = search_pulsar(tim_data, search_cfg, true_params)

    fig1 = tim_data.plot_fold(cfg.freq, cfg.fold_bins)
    fig2 = pgram.plot(figsize=(10, 5))
    fig1.show()
    fig2.show()
    return dyp


def test_pulsar_search_accn(
    psr_type: str = "msp",
    accel: float = 100,
    tol: float | None = None,
    fold_bins: int | None = None,
) -> ffa.DynamicProgramming:
    if psr_type == "msp":
        period = 1.012345678910111213 * 1e-2  # (s)
        dt = 8.192e-5
    elif psr_type == "regular":
        period = 1.212345678910111213  # (s)
        dt = 8.192e-2
    cfg = SignalConfig(period=period, dt=dt, mod_kwargs={"acc": accel})
    tim_data = TimeSeries.generate_signal(cfg)

    param_limits = typed.List(
        [
            (-3.1 * abs(accel), 3.1 * abs(accel)),
            (cfg.period * 0.95, cfg.period * 1.1),
        ],
    )
    if tol is None:
        tol = cfg.tol
    if fold_bins is None:
        fold_bins = cfg.fold_bins
    logger.info(f"Using tolerance: {tol}, fold_bins: {fold_bins}")
    search_cfg = SearchConfig(cfg.nsamps, cfg.dt, tol, fold_bins, param_limits)
    true_params = np.array([accel, cfg.period])
    dyp, pgram = search_pulsar(tim_data, search_cfg, true_params)

    fig1 = tim_data.plot_fold(cfg.freq, cfg.fold_bins)
    fig2 = pgram.plot(figsize=(10, 5))
    fig1.show()
    fig2.show()
    return dyp
