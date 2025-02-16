from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from matplotlib import pyplot as plt
from rich.progress import track
from sigpyproc.viz.styles import set_seaborn

from pyloki.config import PulsarSearchConfig
from pyloki.detection import scoring
from pyloki.search import ffa_search
from pyloki.utils import np_utils
from pyloki.utils.misc import get_logger

if TYPE_CHECKING:
    from numba import types

    from pyloki.io.timeseries import TimeSeries
    from pyloki.simulation.pulse import PulseSignalConfig

logger = get_logger(__name__)

nparam_to_str = {1: "freq", 2: "acc", 3: "jerk", 4: "snap"}


def plot_sensitivity(filename: str | Path) -> plt.Figure:
    with h5py.File(filename, "r") as f:
        ducy_arr = f["ducy_arr"][:]
        tol_bins_arr = f["tol_bins_arr"][:]
        losses = f["losses"][:]
    set_seaborn(**{"text.usetex": True})
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    titles = ["Folding loss", "Shifting loss", "Dynamic FFA loss", "Empirical FFA loss"]
    for iax, ax in enumerate(axs.flat):
        for itol_bin in range(len(tol_bins_arr)):
            ax.plot(
                ducy_arr,
                losses[iax, itol_bin],
                "o-",
                lw=2,
                markersize=8,
                label=f"tol = {tol_bins_arr[itol_bin]}",
            )
            ax.set_xlabel("Duty Cycle")
            ax.set_ylabel("Recovered significance")
            ax.set_title(titles[iax])
            ax.legend()
            if iax == 0:
                ax.set_ylim(0.2, 1.2)
    fig.tight_layout()
    return fig


def test_sensitivity_ffa(
    tim_data: TimeSeries,
    signal_cfg: PulseSignalConfig,
    search_cfg: PulsarSearchConfig,
) -> tuple[float, float, float]:
    dyp, pgram = ffa_search(tim_data, search_cfg)
    snr_shifted = get_shifted_snr(dyp.dparams, tim_data, signal_cfg, search_cfg)

    nparams = len(search_cfg.param_limits)
    true_params_idx = [
        np_utils.find_nearest_sorted_idx(dyp.param_arr[-1], signal_cfg.freq),
    ]
    for deriv in range(2, nparams + 1):
        idx = np_utils.find_nearest_sorted_idx(
            dyp.param_arr[-deriv],
            signal_cfg.mod_kwargs[nparam_to_str[deriv]],
        )
        true_params_idx.insert(0, idx)
    snr_dynamic = float(pgram.data[tuple(true_params_idx)].max())
    snr_empirical = pgram.find_best_params()["snr"]
    logger.info(f"snr_dynamic: {snr_dynamic}, snr_empirical: {snr_empirical}")
    return snr_shifted, snr_dynamic, snr_empirical


def get_shifted_snr(
    dparams: np.ndarray,
    tim_data: TimeSeries,
    signal_cfg: PulseSignalConfig,
    search_cfg: PulsarSearchConfig,
) -> float:
    # Check the params grid around +- dparam
    nparams = len(dparams)
    shift_snr = []
    grid = [
        [num * sign for num, sign in zip(dparams, signs, strict=False)]
        for signs in product([-1, 1], repeat=nparams)
    ]
    for diff_params in grid:
        freq_shifted = signal_cfg.freq + diff_params[-1]
        fold = tim_data.fold_ephem(
            freq_shifted,
            signal_cfg.fold_bins,
            mod_kwargs={
                nparam_to_str[deriv]: signal_cfg.mod_kwargs[nparam_to_str[deriv]]
                + diff_params[-deriv]
                for deriv in range(2, nparams + 1)
            },
        )
        shift_snr.append(get_best_snr(fold, search_cfg))
    return np.max(shift_snr)


def get_best_snr(fold: np.ndarray, search_cfg: PulsarSearchConfig) -> float:
    widths = scoring.generate_box_width_trials(
        len(fold),
        ducy_max=search_cfg.ducy_max,
        spacing_factor=search_cfg.wtsp,
    )
    return scoring.boxcar_snr_1d(fold, widths).max()


class TestFFASensitivity:
    def __init__(
        self,
        cfg: PulseSignalConfig,
        param_limits: types.ListType[types.Tuple[float, float]],
        ducy_arr: np.ndarray | None = None,
        tol_bins_arr: np.ndarray | None = None,
        ducy_max: float = 0.5,
        wtsp: float = 1,
    ) -> None:
        self.cfg = cfg
        self.param_limits = param_limits
        if ducy_arr is None:
            ducy_arr = np.linspace(0.01, 0.3, 15)
        self.ducy_arr = ducy_arr
        if tol_bins_arr is None:
            tol_bins_arr = np.array([1, 2, 4, 8])
        self.tol_bins_arr = tol_bins_arr
        self.ducy_max = ducy_max
        self.wtsp = wtsp
        self.rng = np.random.default_rng()
        self.losses = np.zeros((4, self.ntols, self.nducy), dtype=float)

    @property
    def nprams(self) -> int:
        return len(self.param_limits)

    @property
    def ntols(self) -> int:
        return len(self.tol_bins_arr)

    @property
    def nducy(self) -> int:
        return len(self.ducy_arr)

    @property
    def file_id(self) -> str:
        return (
            f"{nparam_to_str[self.nprams]}_nsamps_{int(np.log2(self.cfg.nsamps)):02d}_"
            f"period_{self.cfg.period:.3f}_os_{self.cfg.os:.1f}"
        )

    def run(self, outdir: str | Path) -> str:
        outpath = Path(outdir)
        if not outpath.is_dir():
            msg = f"Output directory {outdir} does not exist"
            raise FileNotFoundError(msg)
        for idu in track(range(self.nducy), description="Processing ducy..."):
            cfg_update = self.cfg.get_updated({"ducy": self.ducy_arr[idu]})
            self.losses[:, :, idu] = self._execute(cfg_update)
        return self._save(outpath)

    def _execute(self, cfg: PulseSignalConfig) -> np.ndarray:
        phi0 = 0.5 + self.rng.uniform(-cfg.dt, cfg.dt)
        tim_data = cfg.generate(phi0=phi0)
        fold_perfect = tim_data.fold_ephem(
            cfg.freq,
            cfg.fold_bins_ideal,
            mod_kwargs=cfg.mod_kwargs,
        )
        fold_os = tim_data.fold_ephem(
            cfg.freq,
            cfg.fold_bins,
            mod_kwargs=cfg.mod_kwargs,
        )
        losses = np.zeros((4, self.ntols), dtype=float)
        for itol in range(self.ntols):
            search_cfg = PulsarSearchConfig(
                cfg.nsamps,
                cfg.dt,
                cfg.fold_bins,
                self.tol_bins_arr[itol],
                self.param_limits,
                ducy_max=self.ducy_max,
                wtsp=self.wtsp,
            )
            losses[1:, itol] = test_sensitivity_ffa(
                tim_data,
                cfg,
                search_cfg,
            )
        snr_os = get_best_snr(fold_os, search_cfg)
        snr_desired = get_best_snr(fold_perfect, search_cfg)
        losses[0] = snr_os
        return (losses**2) / (snr_desired**2)

    def _save(self, outpath: Path) -> str:
        outfile = outpath / f"ffa_sensitivity_{self.file_id}.h5"
        with h5py.File(outfile, "w") as f:
            for attr in ["period", "os", "nsamps"]:
                f.attrs[attr] = getattr(self.cfg, attr)
            for arr in [
                "ducy_arr",
                "tol_bins_arr",
                "losses",
            ]:
                f.create_dataset(
                    arr,
                    data=getattr(self, arr),
                    compression="gzip",
                    compression_opts=9,
                )
            f.create_dataset(
                "param_limits",
                data=np.array(self.param_limits),
                compression="gzip",
                compression_opts=9,
            )

        return outfile.as_posix()
