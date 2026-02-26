from __future__ import annotations

from typing import TYPE_CHECKING

from pyloki.detection import scoring
from pyloki.ffa import DynamicProgramming
from pyloki.periodogram import Periodogram
from pyloki.utils.misc import get_logger, quiet_logger

if TYPE_CHECKING:
    from pyloki.config import PulsarSearchConfig
    from pyloki.io.timeseries import TimeSeries

logger = get_logger(__name__)


def ffa_search(
    tseries: TimeSeries,
    search_cfg: PulsarSearchConfig,
    *,
    quiet: bool = False,
    show_progress: bool = False,
) -> tuple[DynamicProgramming, Periodogram]:
    """Perform a Fast Folding Algorithm search on a time series.

    Parameters
    ----------
    tseries : TimeSeries
        The time series to search.
    search_cfg : PulsarSearchConfig
        The configuration object for the search.
    quiet : bool, optional
        If True, suppresses logging output, by default False.
    show_progress : bool, optional
        Whether to show progress bar, by default False.

    Returns
    -------
    tuple[DynamicProgramming, Periodogram]
        The DynamicProgramming object and the Periodogram object.
    """
    with quiet_logger(quiet=quiet):
        dyp = DynamicProgramming(tseries, search_cfg)
        dyp.initialize()
        dyp.execute(show_progress=show_progress)
        folds = dyp.get_fold_norm()
        snrs = scoring.boxcar_snr_nd(folds, search_cfg.score_widths, 1.0)
    pgram = Periodogram(
        params={"width": search_cfg.score_widths, **dyp.param_arr_dict},
        snrs=snrs,
        tobs=tseries.tobs,
    )
    return dyp, pgram
