import numpy as np
from pruning.utils import Spyden
from pruning.scores import boxcar_snr_1d

from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText


def plot_text(text, ax, xv, yv, horizontalalignment="left", fontsize=15):
    ax.text(
        xv,
        yv,
        text,
        horizontalalignment=horizontalalignment,
        verticalalignment="center",
        family="monospace",
        transform=ax.transAxes,
        fontsize=fontsize,
    )


class Table(object):
    """ """

    def __init__(
        self,
        col_off: None,
        top_margin: float = 0.1,
        line_height: float = 0.1,
        **kwargs,
    ) -> None:
        self.col_off = col_off
        if self.col_off is None:
            self.col_off = [0.2, 0.5, 0.75, 0.95]
        self.rows = []
        self.num_col = len(self.col_off)
        self.top_margin = top_margin
        self.line_height = line_height
        self.plot_kwargs = kwargs

    def add_row(self, row):
        assert len(row) == self.num_col, "row length should be equal to number of columns"
        self.rows.append(row)

    def skip_row(self) -> None:
        self.rows.append(None)

    def plot(self, ax):
        ax.axis("off")
        yv = 1.0 - self.top_margin
        for row in self.rows:
            if row is None:
                continue
            for icol in range(self.num_col):
                plot_text(
                    row[icol],
                    ax,
                    self.col_off[icol],
                    yv,
                    "right",
                    **self.plot_kwargs,
                )
            yv -= self.line_height


def fold_ephemeris_plot(
    ephemeris_fold,
    ephemeris_fold_subints,
    freq,
    dt,
    tobs,
    mod_kwargs: dict | None = None,
    figsize=(10, 6.5),
    dpi=100,
    cmap="magma_r",
):
    fold_bins = len(ephemeris_fold)
    spyden_boxcar = Spyden(
        ephemeris_fold,
        tempwidth_max=fold_bins // 2,
        template_kind="boxcar",
    )
    match_boxcar = boxcar_snr_1d(ephemeris_fold, np.arange(1, fold_bins // 2))
    if mod_kwargs is None:
        mod_kwargs = {"acc": 0, "jerk": 0, "snap": 0}

    figure = plt.figure(figsize=figsize, dpi=dpi)
    grid = figure.add_gridspec(
        nrows=2, ncols=2, height_ratios=(1.5, 1), width_ratios=(1, 1.5)
    )
    grid.update(left=0.1, right=0.98, bottom=0.08, top=0.95, hspace=0.2)
    axtable = plt.subplot(grid[0, 0])
    axsubints = plt.subplot(grid[0, 1])
    axprofile = plt.subplot(grid[1, :])

    ducy = spyden_boxcar.best_width / len(ephemeris_fold)
    table = Table(col_off=[0.01, 0.75], top_margin=0.1, line_height=0.12, fontsize=12)
    table.add_row(["Tsamp", f"{dt*1e3:.3f}"])
    table.add_row(["Period", 1 / freq])
    table.add_row(["Accel", mod_kwargs["acc"]])
    table.add_row(["Jerk", mod_kwargs["jerk"]])
    table.add_row(["Snap", mod_kwargs["snap"]])
    table.add_row(["Width", spyden_boxcar.best_width])
    table.add_row(["Ducy", f"{ducy:.3f}"])
    table.plot(axtable)

    axsubints.imshow(
        ephemeris_fold_subints,
        aspect="auto",
        interpolation="none",
        extent=[0, fold_bins, 0, tobs],
        cmap=plt.get_cmap(cmap),
        origin="lower",
    )
    axsubints.set_ylabel("Time (seconds)")
    axsubints.set_xlabel("Phase bin")
    axprofile.plot(
        range(fold_bins), ephemeris_fold, color="#404040", label="Folded Profile"
    )
    axprofile.plot(
        range(fold_bins),
        spyden_boxcar.best_model,
        color="#d62728",
        alpha=0.65,
        ls="--",
        label="Boxcar template",
    )
    axprofile.set_xlim(0, fold_bins)
    axprofile.set_xlabel("Phase bin")
    axprofile.set_ylabel("Normalised amplitude")

    textstr = (
        f"S/N (Boxcar) = {spyden_boxcar.snr:.2f}\n"
        f"S/N (Boxcar, 1D) = {match_boxcar.max():.2f}\n"
    )

    at = AnchoredText(textstr, loc="upper left", prop={"size": 10}, frameon=True)
    at.patch.set_boxstyle("round", pad=0, rounding_size=0.2)
    axprofile.add_artist(at)
    axprofile.legend(loc="upper right")
    return figure
