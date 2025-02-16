from __future__ import annotations

from typing import ClassVar

import attrs
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt


@attrs.define(auto_attribs=True, kw_only=True)
class Periodogram:
    REQUIRED_PARAMS: ClassVar[tuple[str, str]] = ("freq", "width")
    OPTIONAL_PARAMS: ClassVar[tuple[str, ...]] = ("snap", "jerk", "accel")

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

    def get_summary(self) -> str:
        best_params = self.find_best_params()
        summary: list[str] = []
        summary += [f"Best S/N: {best_params['snr']:.2f}"]
        summary += [f"Best Period: {1/best_params['freq']}"]
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
