from __future__ import annotations

from typing import ClassVar

import numpy as np
from matplotlib import pyplot as plt


class Periodogram:
    REQUIRED_PARAMS: ClassVar[list[str]] = ["widths", "periods"]
    OPTIONAL_PARAMS: ClassVar[list[str]] = ["accels", "jerks", "snaps"]

    def __init__(
        self,
        params: dict[str, np.ndarray],
        snrs: np.ndarray,
        tobs: float,
    ) -> None:
        self._validate_inputs(params, snrs)
        self.params = params
        self.snrs = snrs
        self.tobs = tobs

    @property
    def param_names(self) -> list[str]:
        return list(self.params.keys())

    @property
    def ndim(self) -> int:
        return len(self.param_names)

    @property
    def freqs(self) -> np.ndarray:
        return 1.0 / self.params["periods"]

    def get_slice(self, fixed_params: dict[str, int]) -> np.ndarray:
        slice_indices: list[slice | int] = [slice(None)] * self.ndim
        for param, idx in fixed_params.items():
            if param not in self.param_names:
                msg = f"Invalid parameter: {param}"
                raise ValueError(msg)
            dim_idx = self.param_names.index(param)
            slice_indices[dim_idx] = idx
        return self.snrs[tuple(slice_indices)]

    def find_best_indices(self) -> tuple:
        return np.unravel_index(np.argmax(self.snrs), self.snrs.shape)

    def find_best_parameters(self) -> dict[str, float | np.ndarray]:
        best_snr = np.max(self.snrs)
        best_indices = self.find_best_indices()
        best_params = {
            name: self.params[name][idx]
            for name, idx in zip(self.param_names, best_indices)
        }
        return {"snr": best_snr, **best_params}

    def plot_1d(
        self,
        param: str,
        fixed_params: dict[str, int] | None = None,
        figsize: tuple[float, float] = (10, 5),
        dpi: int = 100,
    ) -> plt.Figure:
        if param not in self.param_names:
            msg = f"Invalid parameter: {param}"
            raise ValueError(msg)
        fixed_params = fixed_params or {}
        fixed_params_str = ", ".join(
            [f"{key}={val}" for key, val in fixed_params.items()],
        )
        title = f"Best S/N vs {param}"
        if fixed_params_str:
            title += f" for {fixed_params_str}"
        x = self.params[param]
        y = self.get_slice(fixed_params).max(axis=tuple(range(self.ndim - 1)))

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(x, y, marker="o", markersize=2, alpha=0.5)
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
        fixed_params: dict[str, int] | None = None,
        figsize: tuple[float, float] = (10, 8),
        dpi: int = 100,
    ) -> plt.Figure:
        if param_x not in self.param_names or param_y not in self.param_names:
            msg = f"Invalid parameter: {param_x} or {param_y}"
            raise ValueError(msg)
        fixed_params = fixed_params or {}

        x = self.params[param_x]
        y = self.params[param_y]
        z = self.get_slice(fixed_params).max(axis=tuple(range(self.ndim - 2)))

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(
            z.T,
            aspect="auto",
            origin="lower",
            extent=(x.min(), x.max(), y.min(), y.max()),
            cmap="viridis",
        )
        ax.set_xlabel(f"Trial {param_x}", fontsize=16)
        ax.set_ylabel(f"Trial {param_y}", fontsize=16)
        ax.set_title(f"Best S/N: {param_x} vs {param_y}", fontsize=18)
        fig.colorbar(im, ax=ax, label="S/N")
        return fig

    def save(self, filename: str) -> None:
        np.savez(
            filename,
            **self.params,
            snrs=self.snrs,
            tobs=self.tobs,
            param_names=self.param_names,
        )

    @classmethod
    def load(cls, filename: str) -> Periodogram:
        data = np.load(filename, allow_pickle=True)
        params = {name: data[name] for name in data["param_names"]}
        return cls(params=params, snrs=data["snrs"], tobs=float(data["tobs"]))

    def _validate_inputs(self, params: dict[str, np.ndarray], snrs: np.ndarray) -> None:
        if "periods" not in params or "widths" not in params:
            msg = "Periodogram requires 'periods' and 'widths' to be in params"
            raise ValueError(msg)

        valid_params = set(self.REQUIRED_PARAMS + self.OPTIONAL_PARAMS)
        for param in params:
            if param not in valid_params:
                msg = f"Invalid parameter: {param}"
                raise ValueError(msg)

        expected_shape = tuple(len(params[p]) for p in self.param_names)
        if snrs.shape != expected_shape:
            msg = (
                f"SNR shape {snrs.shape} does not match expected shape {expected_shape}"
            )
            raise ValueError(msg)

    def __str__(self) -> str:
        param_info = ", ".join(f"{k}: {len(v)}" for k, v in self.params.items())
        return f"Periodogram({param_info}, tobs: {self.tobs})"

    def __repr__(self) -> str:
        return self.__str__()
