from __future__ import annotations

from typing import TYPE_CHECKING

import seaborn as sns

if TYPE_CHECKING:
    from matplotlib import pyplot as plt


def set_seaborn(**rc_kwargs) -> None:
    rc = {
        # Fontsizes
        "font.size": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.direction": "in",
        "xtick.minor.visible": True,
        "xtick.top": False,
        "ytick.direction": "in",
        "ytick.minor.visible": True,
        "ytick.right": False,
        # Set line widths
        "axes.axisbelow": "line",
        "axes.linewidth": 1,
        "lines.linewidth": 1.5,
        "lines.markersize": 3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.serif": "Times",
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        # Use LaTeX for math formatting
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath, amssymb}",  # {txfonts, mathptmx}
    }
    rc.update(rc_kwargs)
    sns.set_theme(
        context="paper",
        style="ticks",
        palette="colorblind",
        font_scale=1,
        rc=rc,
    )


def plot_text(
    text: str,
    ax: plt.Axes,
    xv: float,
    yv: float,
    horizontalalignment: str = "left",
    fontsize: int = 15,
) -> None:
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


class Table:
    def __init__(
        self,
        col_off: list[float],
        top_margin: float = 0.1,
        line_height: float = 0.1,
        **kwargs,
    ) -> None:
        self.col_off = col_off
        if self.col_off is None:
            self.col_off = [0.2, 0.5, 0.75, 0.95]
        self.rows: list[list[str | float | int] | None] = []
        self.num_col = len(self.col_off)
        self.top_margin = top_margin
        self.line_height = line_height
        self.plot_kwargs = kwargs

    def add_row(self, row: list[str | float | int]) -> None:
        if len(row) != self.num_col:
            msg = f"row length should be equal to number of columns: {self.num_col}"
            raise ValueError(msg)
        self.rows.append(row)

    def skip_row(self) -> None:
        self.rows.append(None)

    def plot(self, ax: plt.Axes) -> None:
        ax.axis("off")
        yv = 1.0 - self.top_margin
        for row in self.rows:
            if row is None:
                continue
            for icol in range(self.num_col):
                plot_text(
                    str(row[icol]),
                    ax,
                    self.col_off[icol],
                    yv,
                    "right",
                    **self.plot_kwargs,
                )
            yv -= self.line_height
