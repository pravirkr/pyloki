from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt

if TYPE_CHECKING:
    import numpy as np


class Periodogram:
    def __init__(
        self,
        widths: np.ndarray,
        periods: np.ndarray,
        snrs: np.ndarray,
        tobs: float,
    ) -> None:
        self.widths = widths
        self.periods = periods
        self.snrs = snrs
        self.tobs = tobs

    @property
    def freqs(self) -> np.ndarray:
        return 1.0 / self.periods

    def plot(
        self,
        iwidth: int | None = None,
        figsize: tuple[float, float] = (10, 5),
        dpi: int = 100,
    ) -> plt.Figure:
        snr = self.snrs.max(axis=1) if iwidth is None else self.snrs[:, iwidth]

        figure, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(self.periods, snr, marker="o", markersize=2, alpha=0.5)
        ax.set_xlim(self.periods.min(), self.periods.max())
        ax.set_xlabel("Trial Period (s)", fontsize=16)
        ax.set_ylabel("S/N", fontsize=16)

        if iwidth is None:
            plt.title("Best S/N at any trial width", fontsize=18)
        else:
            width_bins = self.widths[iwidth]
            plt.title(f"S/N at trial width = {width_bins:d}", fontsize=18)
        ax.grid(linestyle=":")
        return figure
