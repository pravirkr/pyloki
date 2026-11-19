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


def test_ffa_sensitivity_unmodulated_odd_fold_bins_end_to_end(tmp_path: Path) -> None:
    """End-to-end un-mocked sensitivity run handles omitted derivs and odd bins."""
    # period / dt = 0.007 / 64e-6 -> fold_bins == 109 (odd).
    cfg = PulseSignalConfig(
        period=0.007,
        dt=64e-6,
        nsamps=2**12,
        snr=15,
        ducy=0.1,
        mod_kwargs={},
        seed=1,
    )
    assert cfg.fold_bins % 2 == 1
    sim = sim_ffa.TestFFASensitivity(
        cfg=cfg,
        param_limits=np.array([[-10.0, 10.0], [142.0, 144.0]]),
        ducy_arr=np.array([0.1]),
        eta_arr=np.array([2]),
        ds_arr=np.array([1.0]),
        ducy_max=0.3,
        wtsp=1.5,
        seed=0,
        quiet=True,
    )
    outfile = sim.run(tmp_path)
    with h5py.File(outfile, "r") as f:
        for key in ("losses_ds", "losses_real", "losses_complex"):
            vals = f[key][:]
            assert np.all(np.isfinite(vals))
            assert np.all(vals > 0.0)


def test_file_id_supports_5param_crackle() -> None:
    cfg = PulseSignalConfig(
        period=0.01, dt=64e-6, nsamps=2**14, snr=20, ducy=0.1, seed=1
    )
    sim = sim_ffa.TestFFASensitivity(
        cfg=cfg,
        param_limits=np.tile([-1.0, 1.0], (5, 1)),
        seed=0,
    )
    assert sim.file_id.startswith("crackle_nsamps_")
