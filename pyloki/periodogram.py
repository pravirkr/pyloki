from __future__ import annotations

from typing import ClassVar

import attrs
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import pyplot as plt
from sigpyproc.viz.styles import set_seaborn

from pyloki.utils import np_utils


def get_precision(d: float) -> int:
    """Get the number of decimal places to display for a given number."""
    if d == 0:
        return 1
    return max(0, -int(np.floor(np.log10(abs(d))))) + 1


@attrs.define(auto_attribs=True, kw_only=True)
class Periodogram:
    REQUIRED_PARAMS: ClassVar[tuple[str, str]] = ("freq", "width")
    OPTIONAL_PARAMS: ClassVar[tuple[str, ...]] = ("crackle", "snap", "jerk", "accel")

    params: dict[str, np.ndarray]
    snrs: np.ndarray
    tobs: float
    data: xr.DataArray = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        self._validate_inputs()
        self._create_xarray()

    @property
    def valid_params(self) -> tuple[str, ...]:
        return self.OPTIONAL_PARAMS + self.REQUIRED_PARAMS

    @property
    def param_names(self) -> list[str]:
        return [p for p in self.valid_params if p in self.params]

    @property
    def ndim(self) -> int:
        return len(self.param_names)

    def get_slice(self, fixed_param_idx: dict[str, int]) -> xr.DataArray:
        return self.data.isel(fixed_param_idx)

    def find_best_indices(self) -> tuple:
        return np.unravel_index(self.data.argmax().to_numpy(), self.data.shape)  # type: ignore[union-attr]

    def find_best_params(self) -> dict[str, float]:
        best_snr = self.data.max().item()
        best_indices = self.find_best_indices()
        best_params = {
            str(name): self.data.coords[name].to_numpy()[idx]
            for name, idx in zip(self.data.dims, best_indices, strict=False)
        }
        return {"snr": best_snr, **best_params}

    def get_indices_summary(self, true_values: dict[str, float] | None = None) -> str:
        best_indices = self.find_best_indices()
        summary: list[str] = []
        if true_values:
            true_indices = [
                np_utils.find_nearest_sorted_idx(self.params[name], true_values[name])
                for name in self.param_names
                if name in true_values
            ]
            summary += [f"True param indices: {tuple(true_indices)}"]
        summary += [f"Best param indices: {best_indices}"]
        return "\n".join(summary)

    def get_summary(self) -> str:
        best_params = self.find_best_params()
        summary: list[str] = []
        summary += [f"Best S/N: {best_params['snr']:.2f}"]
        summary += [f"Best Period: {1 / best_params['freq']}"]
        for name in self.param_names:
            if name != "snr":
                summary += [f"Best {name}: {best_params[name]}"]
        return "\n".join(summary)

    def plot_1d(
        self,
        param: str,
        fixed_param_idx: dict[str, int] | None = None,
        x_lim: tuple[float, float] | None = None,
        figsize: tuple[float, float] = (10, 5),
        dpi: int = 100,
    ) -> plt.Figure:
        if param not in self.param_names:
            msg = f"Invalid parameter: {param}"
            raise ValueError(msg)
        fixed_param_idx = fixed_param_idx or {}
        fixed_params_str = ", ".join(
            [f"{key}={val}" for key, val in fixed_param_idx.items()],
        )
        title = f"Best S/N vs {param}"
        if fixed_params_str:
            title += f" for {fixed_params_str}"
        x = self.data[param].to_numpy()
        sliced_data = self.get_slice(fixed_param_idx)
        other_dims = [dim for dim in sliced_data.dims if dim != param]
        y = sliced_data.max(dim=other_dims).to_numpy()

        set_seaborn()
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(x, y, marker="o", markersize=2, alpha=0.5)
        if x_lim:
            ax.set_xlim(x_lim)
        else:
            ax.set_xlim(x.min(), x.max())
        ax.set_xlabel(f"Trial {param}", fontsize=16)
        ax.set_ylabel("S/N", fontsize=16)
        ax.set_title(title, fontsize=18)
        ax.grid(linestyle=":")
        return fig

    def plot_2d(
        self,
        param_x: str,
        param_y: str,
        fixed_param_idx: dict[str, int] | None = None,
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        figsize: tuple[float, float] = (10, 8),
        dpi: int = 100,
    ) -> plt.Figure:
        if param_x not in self.param_names or param_y not in self.param_names:
            msg = f"Invalid parameter: {param_x} or {param_y}"
            raise ValueError(msg)
        fixed_param_idx = fixed_param_idx or {}

        x = self.data[param_x].to_numpy()
        y = self.data[param_y].to_numpy()
        sliced_data = self.get_slice(fixed_param_idx)
        other_dims = [dim for dim in sliced_data.dims if dim not in [param_x, param_y]]
        z = sliced_data.max(dim=other_dims).to_numpy()

        set_seaborn()
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(
            z,
            aspect="auto",
            origin="lower",
            extent=(x.min(), x.max(), y.min(), y.max()),
            cmap="viridis",
        )
        ax.set_xlabel(f"Trial {param_x}", fontsize=16)
        ax.set_ylabel(f"Trial {param_y}", fontsize=16)
        ax.set_title(f"Best S/N: {param_x} vs {param_y}", fontsize=18)
        if x_lim:
            ax.set_xlim(x_lim)
        if y_lim:
            ax.set_ylim(y_lim)
        fig.colorbar(im, ax=ax, label="S/N")
        return fig

    def save(self, filename: str) -> None:
        self.data.to_netcdf(filename)

    @classmethod
    def load(cls, filename: str) -> Periodogram:
        data = xr.open_dataarray(filename)
        params = {str(dim): data[dim].to_numpy() for dim in data.dims}
        return cls(params=params, snrs=data.to_numpy(), tobs=float(data.attrs["tobs"]))

    def _validate_inputs(self) -> None:
        if not all(param in self.params for param in self.REQUIRED_PARAMS):
            msg = f"Missing required parameters: {self.REQUIRED_PARAMS}"
            raise ValueError(msg)
        for param in self.params:
            if param not in self.valid_params:
                msg = f"Invalid parameter: {param}"
                raise ValueError(msg)
        expected_shape = tuple(len(self.params[p]) for p in self.param_names)
        if self.snrs.shape != expected_shape:
            msg = (
                f"SNR shape {self.snrs.shape} does not match expected "
                f"shape {expected_shape}"
            )
            raise ValueError(msg)

    def _create_xarray(self) -> None:
        coords = {name: self.params[name] for name in self.param_names}
        self.data = xr.DataArray(self.snrs, coords=coords, dims=self.param_names)
        self.data.attrs["tobs"] = self.tobs

    def __str__(self) -> str:
        param_info = ", ".join(f"{k}: {len(v)}" for k, v in self.params.items())
        return f"Periodogram({param_info}, tobs: {self.tobs})"

    def __repr__(self) -> str:
        return self.__str__()


@attrs.define(auto_attribs=True, kw_only=True)
class ScatteredPeriodogram:
    param_names: list[str]  # length: nparams
    data: pd.DataFrame = attrs.field(init=False, factory=pd.DataFrame)

    @property
    def n_runs(self) -> int:
        return self.data["run_id"].nunique()

    def add_run(self, param_sets: np.ndarray, scores: np.ndarray, run_id: str) -> None:
        """Add a pruning run results to the existing dataframe.

        Parameters
        ----------
        param_sets : np.ndarray
            Parameter sets (n_sugg, n_params, 2).
        scores : np.ndarray
            Scores for each parameter set (n_sugg,).
        run_id : str
            Unique Identifier for the specific run being added.
        """
        self._validate_inputs(param_sets, scores)
        run_data = {}
        for i, name in enumerate(self.param_names):
            run_data[name] = param_sets[:, i, 0]
            run_data[f"d{name}"] = param_sets[:, i, 1]
        run_data["score"] = scores
        run_df = pd.DataFrame.from_dict(run_data)
        run_df["run_id"] = run_id
        self.data = pd.concat([self.data, run_df], ignore_index=True)

    def get_summary_cands(self, n: int = 10, run_id: str | None = None) -> str:
        """Return top N candidates, optionally for a specific run.

        Parameters
        ----------
        n : int, optional
            Number of top candidates to return, by default 10.
        run_id : str | None, optional
            Unique identifier for a specific run, by default None.

        Returns
        -------
        str
            Summary of top candidates.
        """
        data = self.data if run_id is None else self.data[self.data["run_id"] == run_id]
        top_df = data.nlargest(n, "score")
        summary: list[str] = []
        summary.append("Top candidates:")

        param_formatters = {}
        for p in self.param_names:
            d = self.data[f"d{p}"][0]
            decimals = get_precision(d)
            param_formatters[p] = f"{{:.{decimals}f}}"

        dparams_str = ", ".join(
            f"d{p}: {self.data[f'd{p}'][0]:.10g}" for p in self.param_names
        )
        summary.append(f"dparams: {dparams_str}")
        for row in top_df.itertuples(index=False):
            params_str = ", ".join(
                f"{p}: {param_formatters[p].format(getattr(row, p))}"
                for p in self.param_names
            )
            summary.append(
                f"Run: {row.run_id}, S/N: {row.score:.2f}, {params_str}",  # ty:ignore[possibly-missing-attribute]
            )
        return "\n".join(summary)

    def get_best_in_each_run(self) -> str:
        """Return the best candidate from each run.

        Returns
        -------
        str
            Summary of best candidates from each run.
        """
        # Group by run_id and find the row with maximum score in each group
        best_per_run = self.data.loc[self.data.groupby("run_id")["score"].idxmax()]

        summary: list[str] = []
        summary.append("Best candidate in each run:")

        # Get parameter formatters (same as in get_summary_cands)
        param_formatters = {}
        for p in self.param_names:
            d = self.data[f"d{p}"][0]  # Use first value as reference for precision
            decimals = get_precision(d)
            param_formatters[p] = f"{{:.{decimals}f}}"

        # Show dparams once at the top
        dparams_str = ", ".join(
            f"d{p}: {self.data[f'd{p}'][0]:.10g}" for p in self.param_names
        )
        summary.append(f"dparams: {dparams_str}")

        # Sort by score descending for consistent ordering
        best_per_run = best_per_run.sort_values("run_id", ascending=True)

        for row in best_per_run.itertuples(index=False):
            params_str = ", ".join(
                f"{p}: {param_formatters[p].format(getattr(row, p))}"
                for p in self.param_names
            )
            summary.append(
                f"Run: {row.run_id}, S/N: {row.score:.2f}, {params_str}",
            )
        return "\n".join(summary)

    def plot_correlation(
        self,
        param_x: str,
        param_y: str,
        run_id: str | None = None,
        true_values: dict[str, float] | None = None,
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        figsize: tuple[float, float] = (8, 6),
        dpi: int = 100,
        cmap: str = "magma_r",
    ) -> plt.Figure:
        """Plot correlation between two parameters with optional true values."""
        data = self.data if run_id is None else self.data[self.data["run_id"] == run_id]
        set_seaborn()
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        scatter = sns.scatterplot(
            data=data,
            x=param_x,
            y=param_y,
            hue="score",
            size="score",
            sizes=(20, 200),
            palette=cmap,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.2,
            legend="auto",
            ax=ax,
        )
        norm = plt.Normalize(data["score"].min(), data["score"].max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, pad=0.02, aspect=30, label="Score")
        scatter.legend_.remove()

        if true_values:
            if param_x in true_values:
                ax.axvline(
                    true_values[param_x],
                    c="r",
                    ls="--",
                    label=f"True {param_x}",
                )
            if param_y in true_values:
                ax.axhline(
                    true_values[param_y],
                    c="r",
                    ls="--",
                    label=f"True {param_y}",
                )
        if x_lim:
            ax.set_xlim(x_lim)
        if y_lim:
            ax.set_ylim(y_lim)
        ax.set_xlabel(f"Trial {param_x}")
        ax.set_ylabel(f"Trial {param_y}")
        ax.set_title(f"Best S/N: {param_x} vs {param_y}")
        return fig

    def plot_scores(
        self,
        kind: str = "scatter",
        run_id: str | None = None,
        figsize: tuple[float, float] = (7, 3.5),
        dpi: int = 100,
    ) -> plt.Figure:
        """Plot score distribution as scatter or histogram."""
        data = self.data if run_id is None else self.data[self.data["run_id"] == run_id]
        set_seaborn()
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if kind == "scatter":
            ax.scatter(
                range(len(data)),
                data["score"],
                s=12,
                edgecolor="tab:blue",
                facecolor="white",
                linewidth=0.6,
            )
            if run_id is None:
                # Highlight each run with a different background color and label
                prev_idx = 0
                for i, rid in enumerate(data["run_id"].unique()):
                    run_data = data[data["run_id"] == rid]
                    idx = prev_idx + len(run_data)
                    ax.axvspan(prev_idx, idx, color=f"C{i % 5}", alpha=0.08)
                    if idx < len(data):
                        ax.axvline(idx, color="gray", linestyle=":", alpha=0.5)
                    prev_idx = idx
                    ax.set_title("Score Distribution Across Runs")
            else:
                ax.set_title(f"Score Distribution for Run {run_id}")
            ax.set_xlabel("Candidate Index")
            ax.set_ylabel("Score (S/N)")

        else:
            if run_id is not None:
                sns.histplot(
                    data=data,
                    x="score",
                    bins=50,
                    ax=ax,
                    element="step",
                    linewidth=1.2,
                    edgecolor="black",
                )
                ax.set_title(f"Score Distribution for Run {run_id}")
            else:
                sns.histplot(
                    data=data,
                    x="score",
                    hue="run_id",
                    bins=50,
                    element="step",
                    palette=sns.color_palette("colorblind", n_colors=self.n_runs),
                    linewidth=1.2,
                    edgecolor="black",
                    ax=ax,
                )
                ax.set_title("Score Distribution Across Runs")
                sns.move_legend(
                    ax,
                    "upper right",
                    ncol=3,
                    fontsize="xx-small",
                    frameon=False,
                    title="run",
                    title_fontsize="xx-small",
                )
            ax.set_yscale("log")
            ax.set_xlabel("Score (S/N)")
            ax.set_ylabel("Count")
        return fig

    def get_stats(self) -> pd.DataFrame:
        """Get statistics for each parameter."""
        stats = []
        for param in self.param_names:
            param_stats = self.data[param].describe()
            unc_stats = self.data[f"d{param}"].describe()
            stats.append(
                {
                    "parameter": param,
                    "mean": param_stats["mean"],
                    "std": param_stats["std"],
                    "median": param_stats["50%"],
                    "mean_uncertainty": unc_stats["mean"],
                },
            )
        return pd.DataFrame(stats)

    @classmethod
    def load(cls, filename: str) -> ScatteredPeriodogram:
        """Load results from a HDF5 file."""
        with h5py.File(filename, "r") as f:
            if "pruning_version" not in f.attrs:
                msg = "Not a valid pruning results file"
                raise ValueError(msg)
            param_names = list(f.attrs["param_names"])
            pgram = cls(param_names=param_names)
            for run_id_str, run_group in f["runs"].items():
                run_id = run_id_str.split("seg_")[-1]
                param_sets = run_group["param_sets"][:]
                scores = run_group["scores"][:]
                pgram.add_run(param_sets, scores, run_id)
        return pgram

    def _validate_inputs(self, param_sets: np.ndarray, scores: np.ndarray) -> None:
        if param_sets.ndim != 3 or param_sets.shape[-1] != 2:
            msg = f"param_sets should be (n_sugg, n_params, 2), got {param_sets.shape}"
            raise ValueError(msg)
        expected_scores_shape = (param_sets.shape[0],)
        if scores.shape != expected_scores_shape:
            msg = (
                f"scores shape {scores.shape} does not match "
                f"expected shape {expected_scores_shape}"
            )
            raise ValueError(msg)
        nparams = param_sets.shape[1]
        if len(self.param_names) != nparams:
            msg = f"Got {len(self.param_names)} names for {nparams} parameters"
            raise ValueError(msg)

    def __str__(self) -> str:
        run_stats = self.data.groupby("run_id").agg(
            {
                "score": ["count", "min", "max"],
            },
        )

        summary = [
            "ScatteredPeriodogram Summary:",
            f"Parameters: {', '.join(self.param_names)}",
            f"Total runs: {len(run_stats)}",
            f"Total candidates: {len(self.data)}",
            "\nRun Statistics:",
        ]

        for rid, stats in run_stats.iterrows():
            summary.append(
                f"Run {rid}: {int(stats['score']['count'])} candidates, "
                f"max S/N: {stats['score']['max']:.2f}, "
                f"min S/N: {stats['score']['min']:.2f}",
            )
        return "\n".join(summary)

    def __repr__(self) -> str:
        return self.__str__()


@attrs.define(auto_attribs=True, kw_only=True)
class PruningStatsPlotter:
    data: pd.DataFrame = attrs.field(init=False, factory=pd.DataFrame)

    @property
    def n_runs(self) -> int:
        return self.data["run_id"].nunique()

    def add_run(self, level_stats: np.ndarray, run_id: str) -> None:
        """Add a pruning stats run results to the existing dataframe.

        Parameters
        ----------
        level_stats : np.ndarray
            Level statistics (n_levels, 9).
        run_id : str
            Unique Identifier for the specific run being added.
        """
        if not isinstance(level_stats, np.ndarray):
            msg = "level_stats should be a numpy array"
            raise TypeError(msg)
        if level_stats.dtype.names is None or len(level_stats.dtype.names) != 9:
            msg = "level_stats should have 9 fields"
            raise ValueError(msg)
        run_df = pd.DataFrame.from_records(level_stats)
        run_df["run_id"] = run_id
        self.data = pd.concat([self.data, run_df], ignore_index=True)

    def get_level_stats(self, run_id: str | None = None) -> pd.DataFrame:
        """Get level statistics for a specific run.

        Parameters
        ----------
        run_id : str | None, optional
            Unique identifier for a specific run, by default None.

        Returns
        -------
        pd.DataFrame
            Level statistics for the specified run.
        """
        return self.data if run_id is None else self.data[self.data["run_id"] == run_id]

    def plot_level_stats(
        self,
        run_id: int | None = None,
        figsize: tuple[float, float] = (10, 5),
        dpi: int = 100,
    ) -> plt.Figure:
        """Plot level statistics for a specific run."""
        data = self.data if run_id is None else self.data[self.data["run_id"] == run_id]
        n_runs = self.n_runs if run_id is None else 1
        set_seaborn()
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.lineplot(
            data=data,
            x="level",
            y="n_leaves_surv",
            hue="run_id",
            palette=sns.color_palette("colorblind", n_colors=n_runs),
            ax=ax,
        )
        sns.move_legend(
            ax,
            "lower center",
            ncol=3,
            fontsize="xx-small",
            frameon=False,
            title="run",
            title_fontsize="xx-small",
        )
        ax.set_yscale("log")
        ax.set_title("Pruning Complexity")
        ax.set_xlabel("Level")
        ax.set_ylabel("Number of Leaves Surviving")
        return fig

    @classmethod
    def load(cls, filename: str) -> PruningStatsPlotter:
        """Load results from a HDF5 file."""
        with h5py.File(filename, "r") as f:
            if "pruning_version" not in f.attrs:
                msg = "Not a valid pruning results file"
                raise ValueError(msg)
            pstats = cls()
            for run_id_str, run_group in f["runs"].items():
                level_stats = run_group["level_stats"][:]
                pstats.add_run(level_stats, run_id_str)
        return pstats
