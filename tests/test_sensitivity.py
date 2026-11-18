from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np

from pyloki.sensitivity import sim_ffa
from pyloki.simulation.pulse import PulseSignalConfig

if TYPE_CHECKING:
    from pathlib import Path


def test_ffa_run_writes_the_attrs_the_plot_reads(monkeypatch, tmp_path: Path) -> None:
    # The FFA searches are slow and not under test here: stub them out.
    monkeypatch.setattr(
        sim_ffa, "test_sensitivity_ffa", lambda *_, **__: (1.0, 1.0, 1.0)
    )
    cfg = PulseSignalConfig(
        period=0.01, dt=64e-6, nsamps=2**14, snr=20, ducy=0.1, seed=1
    )
    sim = sim_ffa.TestFFASensitivity(
        cfg=cfg,
        param_limits=np.array([[-10.0, 10.0], [99.0, 101.0]]),
        ducy_arr=np.array([0.1]),
        eta_arr=np.array([1]),
        ds_arr=np.array([1.0, 1.5]),
        seed=0,
        quiet=True,
    )
    outfile = sim.run(tmp_path)
    with h5py.File(outfile, "r") as f:
        # Every attr `plot_sensitivity` reads must be present.
        for attr in ["nsamps", "dt", "snr", "period", "ds", "nbins_ideal"]:
            assert attr in f.attrs, attr
        assert f.attrs["nbins_ideal"] == cfg.fold_bins_ideal
        assert f["losses_ds"].shape == (2, 1)
