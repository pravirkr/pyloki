from __future__ import annotations

import numpy as np
from numba import typed

from pyloki.config import PulsarSearchConfig
from pyloki.detection import scoring, thresholding
from pyloki.ffa import DynamicProgramming
from pyloki.io.timeseries import TimeSeries
from pyloki.periodogram import Periodogram
from pyloki.prune import Pruning
from pyloki.search import ffa_search
from pyloki.simulation.pulse import PulseSignalConfig
from pyloki.utils import np_utils
from pyloki.utils.misc import get_logger

logger = get_logger(__name__)


def pruning_search(
    dyp: DynamicProgramming,
    target_snr: float,
) -> Pruning:
    bound_scheme = thresholding.bound_scheme(
        dyp.nchunks,
        target_snr,
    )
    trials_scheme = thresholding.trials_scheme(dyp.nchunks, dyp.nparams)
    threshold_scheme = np.minimum(bound_scheme, trials_scheme)
    prn = Pruning(dyp, threshold_scheme, max_sugg=131072)
    prn.initialize(ref_seg=12)
    return prn


def search_pulsar_pruning(
    tseries: TimeSeries,
    search_cfg: PulsarSearchConfig,
    true_params: np.ndarray,
) -> tuple:
    dyp = DynamicProgramming(tseries, search_cfg)
    dyp.initialize()
    dyp.do_iterations("prune")
    folds = dyp.fold[0][..., 0, :] / np.sqrt(dyp.fold[0][..., 1, :])

    true_period_idx, _ = dyp.dp_funcns.resolve(
        true_params,
        dyp.param_arr,
        dyp.ffa_level,
        0,
    )

    widths = scoring.generate_width_trials(search_cfg.nbins, ducy_max=0.1, wtsp=1)
    snrs = scoring.boxcar_snr(folds, widths)
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
) -> DynamicProgramming:
    if psr_type == "msp":
        period = 1.012345678910111213 * 1e-2  # (s)
        dt = 8.192e-5
    elif psr_type == "regular":
        period = 1.212345678910111213  # (s)
        dt = 8.192e-3
    cfg = PulseSignalConfig(period=period, dt=dt)
    tseries = TimeSeries.generate_signal(cfg)

    param_limits = typed.List([(cfg.period * 0.5, cfg.period * 1.6)])
    if tol is None:
        tol = cfg.tol
    if fold_bins is None:
        fold_bins = cfg.fold_bins
    logger.info(f"Using tolerance: {tol}, fold_bins: {fold_bins}")
    search_cfg = PulsarSearchConfig(cfg.nsamps, cfg.dt, fold_bins, tol, param_limits)
    true_params = np.array([cfg.freq])
    dyp, pgram = ffa_search(tseries, search_cfg)
    true_params_idx = np_utils.find_nearest_sorted_idx(dyp.param_arr[-1], true_params)
    logger.info("\n*** Search results ***")
    logger.info(f"True param index: {true_params_idx}")
    logger.info(f"Best param indices: {pgram.find_best_indices()}")
    logger.info(pgram.get_summary())
    fig = pgram.plot_1d("freq", figsize=(10, 5))
    fig.show()
    return dyp


def test_pulsar_search_accn(
    psr_type: str = "msp",
    accel: float = 100,
    tol: float | None = None,
    fold_bins: int | None = None,
) -> DynamicProgramming:
    if psr_type == "msp":
        period = 1.012345678910111213 * 1e-2  # (s)
        dt = 8.192e-5
    elif psr_type == "regular":
        period = 1.212345678910111213  # (s)
        dt = 8.192e-2
    cfg = PulseSignalConfig(period=period, dt=dt, mod_kwargs={"acc": accel})
    tseries = TimeSeries.generate_signal(cfg)

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
    search_cfg = PulsarSearchConfig(cfg.nsamps, cfg.dt, fold_bins, tol, param_limits)
    dyp, pgram = ffa_search(tseries, search_cfg)
    idx_freq = np_utils.find_nearest_sorted_idx(dyp.param_arr[-1], cfg.freq)
    idx_acc = np_utils.find_nearest_sorted_idx(dyp.param_arr[-2], cfg.mod_kwargs["acc"])
    true_params_idx = (idx_acc, idx_freq)
    logger.info("\n*** Search results ***")
    logger.info(f"True param index: {true_params_idx}")
    logger.info(pgram.get_summary())
    fig = pgram.plot_2d("freq", "accel", figsize=(10, 5))
    fig.show()
    return dyp
