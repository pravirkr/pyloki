from __future__ import annotations

from typing import TYPE_CHECKING

from pyloki.detection import scoring
from pyloki.ffa import DynamicProgramming
from pyloki.periodogram import Periodogram

if TYPE_CHECKING:
    from pyloki.config import PulsarSearchConfig
    from pyloki.io.timeseries import TimeSeries


def ffa_search(
    tseries: TimeSeries,
    search_cfg: PulsarSearchConfig,
) -> tuple[DynamicProgramming, Periodogram]:
    """
    Perform a Fast Folding Algorithm search on a time series.

    Parameters
    ----------
    tseries : TimeSeries
        The time series to search.
    search_cfg : PulsarSearchConfig
        The configuration object for the search.

    Returns
    -------
    tuple[DynamicProgramming, Periodogram]
        The DynamicProgramming object and the Periodogram object.
    """
    dyp = DynamicProgramming(tseries, search_cfg)
    dyp.initialize()
    dyp.execute()
    folds = dyp.get_fold_norm()
    widths = scoring.generate_box_width_trials(
        search_cfg.nbins,
        ducy_max=0.2,
        spacing_factor=1,
    )
    snrs = scoring.boxcar_snr(folds, widths)
    pgram = Periodogram(
        params={"width": widths, "freq": dyp.param_arr[-1]},
        snrs=snrs,
        tobs=tseries.tobs,
    )
    return dyp, pgram
