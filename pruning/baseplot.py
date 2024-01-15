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
