from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyloki import kepler
from pyloki.core import circular, epicyclic, svd_kepler
from pyloki.utils.misc import C_VAL

if TYPE_CHECKING:
    from pathlib import Path


def _jit_variants(fn):
    return [pytest.param(fn, id="njit"), pytest.param(fn.py_func, id="py")]


class TestKEP02PhysicalCircular4D:
    @pytest.mark.parametrize(
        "branch_fn",
        _jit_variants(circular.cir_physical_branch_batch),
    )
    def test_physical_circular_branches_only_omega_and_freq(self, branch_fn) -> None:
        # Leaf slot layout: [omega, freq, x_sin_phi, x_cos_phi] + 2 auxiliary slots
        leaf = np.zeros((1, 6, 2), dtype=np.float64)
        leaf[0, 0, :] = [2.0 * math.pi / 3600.0, 2.0e-4]  # omega, d_omega
        leaf[0, 1, :] = [100.0, 5.0e-3]  # freq, d_freq
        leaf[0, 2, :] = [0.6, 0.05]  # x_sin_phi
        leaf[0, 3, :] = [0.8, 0.05]  # x_cos_phi

        out_leaves, origins = branch_fn(
            leaf,
            (0.0, 5400.0),  # 1.5 P_orb half-span => 3 P_orb total
            (0.0, 1800.0),
            64,
            1.0,
        )
        assert out_leaves.shape[0] > 1
        assert np.all(origins == 0)
        # x_sin_phi and x_cos_phi NEVER branch
        assert np.all(out_leaves[:, 2, 0] == pytest.approx(0.6))
        assert np.all(out_leaves[:, 3, 0] == pytest.approx(0.8))

        # Resolve across 3 orbits using poly_circular_resolve_batch
        param_limits = np.array([[-100.0, 100.0], [99.0, 101.0]], dtype=np.float64)
        grid_counts = np.array([201, 201], dtype=np.int64)
        pidx, pshift = circular.poly_circular_resolve_batch(
            out_leaves[:4],
            (10800.0, 60.0),
            (0.0, 60.0),
            grid_counts,
            param_limits,
            64,
        )
        assert pidx.shape == (4, 2)
        assert pshift.shape == (4,)
        assert np.all(np.isfinite(pshift))


class TestKEP01SVDKepEngine:
    def test_svd_basis_build_save_load_and_reconstruction(
        self,
        tmp_path: Path,
    ) -> None:
        span_axes = [
            np.linspace(0.0, 600.0, 16),
            np.linspace(0.0, 1200.0, 32),
            np.linspace(0.0, 2400.0, 64),
        ]
        tau_tol = 1e-3
        basis = svd_kepler.SVDOrbitalBasis.build(
            span_axes,
            p_orb_range=(2800.0, 3600.0),
            x_orb_range=(0.5, 2.0),
            ecc_range=(0.0, 0.3),
            tau_tol=tau_tol,
            n_examples=200,
            seed=42,
        )
        assert len(basis.levels) == 3
        assert basis.levels[0].trans_matrix is None
        assert basis.levels[1].trans_matrix is not None
        assert basis.levels[2].novelties is not None

        # Round-trip save/load
        npz_path = tmp_path / "svd_basis.npz"
        basis.save_npz(npz_path)
        loaded = svd_kepler.SVDOrbitalBasis.load_npz(npz_path)
        assert len(loaded.levels) == 3
        np.testing.assert_allclose(loaded.levels[2].v_phase, basis.levels[2].v_phase)

        # Held-out eccentric orbit projection residual is bounded by ~tau_tol
        lvl = loaded.levels[2]
        z_held = kepler.keplerian_z(
            lvl.t_axis,
            3200.0,
            0.2,
            0.5,
            1.2,
            0.8,
            0.5 * math.pi,
        )
        z_norm = (z_held - float(np.mean(z_held))) / lvl.tau_tol
        coeffs = lvl.v_phase @ z_norm
        recon = (coeffs @ lvl.v_phase) * lvl.tau_tol
        rms_err = float(np.sqrt(np.mean((recon - (z_held - np.mean(z_held))) ** 2)))
        assert rms_err < 2.0 * tau_tol

    @pytest.mark.parametrize("branch_fn", _jit_variants(svd_kepler.svd_branch_batch))
    def test_svd_branch_and_resolve_kernels(self, branch_fn) -> None:
        span_axes = [
            np.linspace(0.0, 600.0, 16),
            np.linspace(0.0, 1200.0, 32),
        ]
        basis = svd_kepler.SVDOrbitalBasis.build(
            span_axes,
            p_orb_range=(3000.0, 3500.0),
            x_orb_range=(0.8, 1.5),
            ecc_range=(0.0, 0.2),
            tau_tol=2e-3,
            n_examples=80,
            seed=7,
        )
        lvl0, lvl1 = basis.levels[0], basis.levels[1]
        assert lvl1.trans_matrix is not None
        assert lvl1.novelties is not None
        parents = np.zeros((2, lvl0.rank), dtype=np.float64)
        children, origins = branch_fn(
            parents,
            lvl1.trans_matrix,
            lvl1.novelties,
            lvl1.delta,
        )
        assert children.shape[1] == lvl1.rank
        assert children.shape[0] > 0
        # Identical parents should deduplicate cleanly
        assert children.shape[0] <= lvl1.novelties.shape[0]
        assert np.all((origins >= 0) & (origins < 2))

        acc, freq, phase = svd_kepler.svd_resolve_batch(
            children[:4],
            np.full(min(4, children.shape[0]), 100.0, dtype=np.float64),
            0.0,
            600.0,
            lvl1.t_axis,
            lvl1.v_phase,
            lvl1.v_vel,
            lvl1.v_acc,
            lvl1.tau_tol,
            64,
        )
        assert np.all(np.isfinite(acc))
        assert np.all(np.isfinite(freq))
        assert np.all((phase >= 0.0) & (phase <= 64.0))


class TestKEP03EpicyclicValidator:
    @pytest.mark.parametrize(
        "res_fn",
        _jit_variants(epicyclic.epicyclic_min_residual_batch),
    )
    def test_epicyclic_passes_eccentric_orbit_and_rejects_oscillatory_poly(
        self,
        res_fn,
    ) -> None:
        cfg = epicyclic.ValidatorConfig(
            enabled=True,
            omega_min=2.0 * math.pi / 4000.0,
            omega_max=2.0 * math.pi / 2000.0,
            x_max=2.0,
            ecc_max=0.25,
            n_colloc=20,
        )
        proj = epicyclic.EpicyclicProjector.build(0.0, 2000.0, 50.0, cfg)

        # True eccentric (e=0.2) Keplerian orbit in meters
        z_true = kepler.keplerian_z(
            proj.t_colloc,
            3000.0,
            0.2,
            0.5,
            1.5 * C_VAL,
            0.7,
            0.5 * math.pi,
        )
        # Unphysical high-frequency ripple well above omega_max
        z_bad = 1.5 * C_VAL * np.sin(10.0 * cfg.omega_max * proj.t_colloc)
        batch = np.vstack([z_true, z_bad])

        residuals = res_fn(batch, proj.projectors)
        assert residuals[0] <= cfg.max_residual_m
        assert residuals[1] > cfg.max_residual_m

        mask = proj.filter_trajectories(batch)
        assert mask.tolist() == [True, False]


class TestKEP04PackedVoxelAndDimensionlessKey:
    @pytest.mark.parametrize("fn", _jit_variants(kepler.dimensionless_kepler_key))
    def test_dimensionless_key_is_scale_invariant_and_has_fallback(self, fn) -> None:
        # Pure circular orbit: z(t) = A sin(om t + phi)
        # z2 = -om^2 z, z3 = -om^2 z', z4 = +om^4 z, z5 = +om^4 z', z6 = -om^6 z
        def _circ_derivs(amp: float, om: float, phi: float) -> tuple[float, ...]:
            s, c = math.sin(phi), math.cos(phi)
            return (
                -amp * (om**2) * s,
                -amp * (om**3) * c,
                amp * (om**4) * s,
                amp * (om**5) * c,
                -amp * (om**6) * s,
            )

        ok1, a1, z5_1, z6_1 = fn(*_circ_derivs(1.0 * C_VAL, 2e-3, 0.7))
        ok2, a2, z5_2, z6_2 = fn(*_circ_derivs(3.5 * C_VAL, 5e-3, 0.7))
        assert ok1
        assert ok2
        assert a1 == pytest.approx(a2, rel=1e-12)
        assert z5_1 == pytest.approx(z5_2, rel=1e-12)
        assert z6_1 == pytest.approx(z6_2, rel=1e-12)

        # Fallback when z2 == 0 (phi = 0)
        ok_fb, *_ = fn(*_circ_derivs(2.0 * C_VAL, 3e-3, 0.0))
        assert ok_fb

    @pytest.mark.parametrize("fn", _jit_variants(kepler.lookup_voxel_bounds))
    def test_packed_voxel_table_lookup(self, fn) -> None:
        gen = kepler.PredictionTableGenerator(
            x_orb_max=1.5,
            ecc_max=0.1,
            omega_max=2.0 * math.pi / 3600.0,
            poly_deg=6,
        )
        bins = (5.0, 0.05, 1e-4)
        table = gen.build_packed_voxel_table((2, 3, 4), target_index=5, bin_sizes=bins)
        assert table.keys.size > 0
        # Query an actual row from gen.data
        row = gen.data[0]
        found, lo, hi = table.query(float(row[2]), float(row[3]), float(row[4]))
        assert found
        assert lo <= float(row[5]) <= hi

        # Direct @njit lookup_voxel_bounds call
        found2, lo2, hi2 = fn(table.keys, table.d_min, table.d_max, table.keys[0])
        assert found2
        assert lo2 <= hi2

        # Upper-bound 21-bit aliasing check in PackedVoxelTable.query
        aliased_v0 = float(row[2]) + (1 << 21) * bins[0]
        found_alias, _, _ = table.query(aliased_v0, float(row[3]), float(row[4]))
        assert not found_alias

    def test_dimensionless_key_near_z2_zero_crossing(self) -> None:
        """Near z2 == 0, dimensionless_kepler_key must fall back to -z5/z3."""
        om = 2.0e-3
        amp = 1.5 * C_VAL
        # Tiny phase angle -> z2 is tiny (~1e-10), z3 is large (~1e-4)
        phi = 1e-6
        s, c = math.sin(phi), math.cos(phi)
        z2 = -amp * (om**2) * s
        z3 = -amp * (om**3) * c
        # Slightly perturb z4 as happens in eccentric/noisy trajectories
        z4 = amp * (om**4) * s * 10.0
        z5 = amp * (om**5) * c
        z6 = -amp * (om**6) * s
        ok, angle, z5_hat, _ = kepler.dimensionless_kepler_key(z2, z3, z4, z5, z6)
        assert ok
        assert z5_hat == pytest.approx(1.0, rel=1e-4)
        assert abs(angle - math.atan2(-s, -c)) < 1e-4

    def test_svd_doppler_gauge_and_group_preservation(self) -> None:
        """Verify svd_resolve_batch Doppler gauge & svd_branch_batch groups."""
        # Constant non-zero velocity v_ref = v_cur = 3e6 m/s (0.01 c)
        t_axis = np.linspace(0.0, 100.0, 11, dtype=np.float64)
        v_const = 3.0e6
        # d(t) = (v_const / C_VAL) * t -> d'(t) = v_const / C_VAL
        tau_tol = 1.0
        v_phase = ((v_const / C_VAL) * t_axis)[None, :]
        v_vel = np.full((1, t_axis.size), v_const, dtype=np.float64)
        v_acc = np.zeros((1, t_axis.size), dtype=np.float64)
        coords = np.array([[1.0]], dtype=np.float64)
        f0 = np.array([100.0], dtype=np.float64)

        _, freq_out, phase_out = svd_kepler.svd_resolve_batch(
            coords,
            f0,
            0.0,
            1.003,
            t_axis,
            v_phase,
            v_vel,
            v_acc,
            tau_tol,
            1000,
        )
        # Constant velocity means f_obs(t) == f0 identically!
        assert float(freq_out[0]) == pytest.approx(100.0, rel=1e-12)
        # Over dt=1.003s at constant observed freq 100Hz: 100.3 cycles -> bin 300
        assert float(phase_out[0]) == pytest.approx(300.0, abs=1e-6)

        # Cross-parent f0 group preservation in svd_branch_batch
        parents = np.zeros((2, 1), dtype=np.float64)
        trans = np.ones((1, 1), dtype=np.float64)
        novs = np.array([[0.0], [1.0]], dtype=np.float64)
        delta = np.array([1.0], dtype=np.float64)
        groups = np.array([10, 20], dtype=np.int64)
        children, origins = svd_kepler.svd_branch_batch(
            parents, trans, novs, delta, groups
        )
        assert children.shape == (4, 1)
        assert set(origins.tolist()) == {0, 1}

    def test_core_exports_and_layering(self) -> None:
        """Verify core/__init__.py exports & core/svd_kepler layering."""
        from pyloki import core  # noqa: PLC0415

        assert "svd_kepler" in core.__all__
        assert "epicyclic" in core.__all__
        assert "cir_physical_branch_batch" in core.__all__
        assert not hasattr(svd_kepler, "kepler")
