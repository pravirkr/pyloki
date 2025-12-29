from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from rich.progress import track
from sigpyproc.viz.styles import set_seaborn

from pyloki.config import PulsarSearchConfig
from pyloki.detection import scoring
from pyloki.search import ffa_search
from pyloki.utils import np_utils
from pyloki.utils.misc import CONSOLE, get_logger

if TYPE_CHECKING:
    from numba import types

    from pyloki.io.timeseries import TimeSeries
    from pyloki.simulation.pulse import PulseSignalConfig

logger = get_logger(__name__)

nparam_to_str = {1: "freq", 2: "acc", 3: "jerk", 4: "snap"}


def plot_sensitivity(filename: str | Path) -> plt.Figure:
    with h5py.File(filename, "r") as f:
        nsamps = f.attrs["nsamps"]
        tsamp = f.attrs["dt"]
        snr_inj = f.attrs["snr"]
        freq = 1 / f.attrs["period"]
        ds = f.attrs["ds"]
        nbins_ideal = f.attrs["nbins_ideal"]
        ducy_arr = f["ducy_arr"][:]
        eta_arr = f["eta_arr"][:]
        ds_arr = f["ds_arr"][:]
        losses_real = f["losses_real"][:]
        losses_complex = f["losses_complex"][:]
        losses_ds = f["losses_ds"][:]
    set_seaborn(use_latex=True, font_size=16)
    fig, (axs) = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    for ids in range(len(ds_arr)):
        nbins = int(nbins_ideal / ds_arr[ids])
        axs[0, 0].plot(
            ducy_arr,
            losses_ds[ids],
            "o-",
            lw=2,
            markersize=8,
            label=f"ds = {ds_arr[ids]:.2f}, " + r"$N_{b}$ = " + f"{nbins}",
        )
        axs[0, 0].set_xlabel("Duty Cycle")
        axs[0, 0].set_ylabel("Recovered significance")
        axs[0, 0].set_title("Folding loss")
        axs[0, 0].legend()

    palette = sns.color_palette("colorblind", n_colors=len(eta_arr))
    titles = ["Shifting loss", "Dynamic FFA loss", "Empirical FFA loss"]
    for iax, ax in enumerate(axs.flat[1:]):
        for itol_bin in range(len(eta_arr)):
            color = palette[itol_bin]
            ax.plot(
                ducy_arr,
                losses_real[iax, itol_bin],
                "o-",
                lw=2,
                markersize=8,
                color=color,
                label=r"$\eta$ = " + f"{eta_arr[itol_bin]:.1f}",
            )
            if iax != 0:
                ax.plot(
                    ducy_arr,
                    losses_complex[iax, itol_bin],
                    "o--",
                    lw=2,
                    markersize=8,
                    color=color,
                )
        ax.set_xlabel("Duty Cycle")
        ax.set_ylabel("Recovered significance")
        ax.set_title(titles[iax])

        h1, l1 = ax.get_legend_handles_labels()
        style_handles = [
            Line2D([0], [0], color="black", lw=2, linestyle="-", label="Real"),
            Line2D([0], [0], color="black", lw=2, linestyle="--", label="Complex"),
        ]
        handles = h1 + style_handles
        labels = l1 + [h.get_label() for h in style_handles]
        ax.legend(
            handles,
            labels,
            loc="lower right",
            ncol=2,
            frameon=True,
        )
    fig.suptitle(
        f"Sensitivity to: nsamps (lb) = {np.log2(nsamps):.0f}, freq = {freq:.3f}, "
        f"ds = {ds:.2f}, snr_inj = {snr_inj:.2f}, dt = {tsamp:.3f}",
    )
    return fig


def test_sensitivity_ffa(
    tim_data: TimeSeries,
    signal_cfg: PulseSignalConfig,
    search_cfg: PulsarSearchConfig,
    *,
    quiet: bool,
) -> tuple[float, float, float]:
    dyp, pgram = ffa_search(tim_data, search_cfg, quiet=quiet, show_progress=False)
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
    if not quiet:
        logger.info(
            f"snr_dynamic: {snr_dynamic:.2f}, snr_empirical: {snr_empirical:.2f}",
        )
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
        shift_snr.append(get_best_snr(fold, signal_cfg.ducy, search_cfg.wtsp))
    return np.max(shift_snr)


def get_best_snr(fold: np.ndarray, ducy_max: float, wtsp: float) -> float:
    widths = scoring.generate_box_width_trials(len(fold), ducy_max=ducy_max, wtsp=wtsp)
    return scoring.boxcar_snr_1d(fold, widths, 1.0).max()


class TestFFASensitivity:
    def __init__(
        self,
        cfg: PulseSignalConfig,
        param_limits: types.ListType[types.Tuple[float, float]],
        ducy_arr: np.ndarray | None = None,
        eta_arr: np.ndarray | None = None,
        ds_arr: np.ndarray | None = None,
        ducy_max: float = 0.5,
        wtsp: float = 1,
        *,
        quiet: bool = False,
    ) -> None:
        self.cfg = cfg
        self.param_limits = param_limits
        if ducy_arr is None:
            ducy_arr = np.linspace(0.01, 0.3, 15)
        if eta_arr is None:
            eta_arr = np.array([1, 2, 4, 8])
        if ds_arr is None:
            ds_arr = np.array([1, 1.25, 1.5, 1.75, 2])
        self.ducy_arr = ducy_arr
        self.eta_arr = eta_arr
        self.ds_arr = ds_arr
        self.ducy_max = ducy_max
        self.wtsp = wtsp
        self.quiet = quiet
        self.rng = np.random.default_rng()
        self.losses_real = np.zeros((3, self.ntols, self.nducy), dtype=float)
        self.losses_complex = np.zeros((3, self.ntols, self.nducy), dtype=float)
        self.losses_ds = np.zeros((self.nds, self.nducy), dtype=float)

    @property
    def nparams(self) -> int:
        return len(self.param_limits)

    @property
    def ntols(self) -> int:
        return len(self.eta_arr)

    @property
    def nducy(self) -> int:
        return len(self.ducy_arr)

    @property
    def nds(self) -> int:
        return len(self.ds_arr)

    @property
    def file_id(self) -> str:
        return (
            f"{nparam_to_str[self.nparams]}_nsamps_{int(np.log2(self.cfg.nsamps)):02d}_"
            f"period_{self.cfg.period:.3f}_ds_{self.cfg.ds:.1f}"
        )

    def run(self, outdir: str | Path) -> str:
        outpath = Path(outdir)
        if not outpath.is_dir():
            msg = f"Output directory {outdir} does not exist"
            raise FileNotFoundError(msg)
        for idu in track(
            range(self.nducy),
            description="Processing ducy...",
            console=CONSOLE,
            transient=True,
        ):
            cfg_update = self.cfg.get_updated({"ducy": self.ducy_arr[idu]})
            losses_total = self._execute(cfg_update)
            self.losses_ds[:, idu] = losses_total[0]
            self.losses_real[:, :, idu] = losses_total[1]
            self.losses_complex[:, :, idu] = losses_total[2]
        return self._save(outpath)

    def _execute(
        self,
        cfg: PulseSignalConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phi0 = 0.5 + self.rng.uniform(-cfg.dt, cfg.dt)
        tim_data = cfg.generate(phi0=phi0)
        fold_perfect = tim_data.fold_ephem(
            cfg.freq,
            cfg.fold_bins_ideal,
            mod_kwargs=cfg.mod_kwargs,
        )
        losses_real = np.zeros((3, self.ntols), dtype=float)
        losses_complex = np.zeros((3, self.ntols), dtype=float)
        losses_ds = np.zeros((self.nds), dtype=float)
        # Compute signal strength over different oversampling factors
        for ids in range(self.nds):
            nbins = int(cfg.fold_bins_ideal / self.ds_arr[ids])
            fold_ds = tim_data.fold_ephem(
                cfg.freq,
                nbins,
                mod_kwargs=cfg.mod_kwargs,
            )
            losses_ds[ids] = get_best_snr(fold_ds, self.ducy_max, self.wtsp)
        # Compute signal strength over different eta
        for itol in range(self.ntols):
            search_cfg_real = PulsarSearchConfig(
                nsamps=cfg.nsamps,
                tsamp=cfg.dt,
                nbins=cfg.fold_bins,
                eta=self.eta_arr[itol],
                param_limits=self.param_limits,
                ducy_max=self.ducy_max,
                wtsp=self.wtsp,
                bseg_brute=cfg.nsamps // 16384,
                use_fourier=False,
            )
            losses_real[:, itol] = test_sensitivity_ffa(
                tim_data,
                cfg,
                search_cfg_real,
                quiet=self.quiet,
            )
            search_cfg_complex = PulsarSearchConfig(
                nsamps=cfg.nsamps,
                tsamp=cfg.dt,
                nbins=cfg.fold_bins,
                eta=self.eta_arr[itol],
                param_limits=self.param_limits,
                ducy_max=self.ducy_max,
                wtsp=self.wtsp,
                bseg_brute=cfg.nsamps // 16384,
                use_fourier=True,
            )
            losses_complex[:, itol] = test_sensitivity_ffa(
                tim_data,
                cfg,
                search_cfg_complex,
                quiet=self.quiet,
            )
        snr_desired = get_best_snr(fold_perfect, self.ducy_max, self.wtsp)
        losses_real_signi = (losses_real**2) / (snr_desired**2)
        losses_complex_signi = (losses_complex**2) / (snr_desired**2)
        losses_ds_signi = (losses_ds**2) / (snr_desired**2)
        return losses_ds_signi, losses_real_signi, losses_complex_signi

    def _save(self, outpath: Path) -> str:
        outfile = outpath / f"pyloki_ffa_sensitivity_{self.file_id}.h5"
        with h5py.File(outfile, "w") as f:
            for attr in ["period", "ds", "nsamps", "nbins_ideal", "dt", "snr"]:
                f.attrs[attr] = getattr(self.cfg, attr)
            for arr in [
                "ducy_arr",
                "eta_arr",
                "ds_arr",
                "losses_real",
                "losses_complex",
                "losses_ds",
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
