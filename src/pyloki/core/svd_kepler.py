"""Empirical SVD basis and manifold novelty-cloud orbital pruning engine.

Compresses a bounded Keplerian orbital prior onto the minimum effective linear
basis dimension at each partial integration span, constructs offline subspace
transition matrices and integer-lattice deduplicated novelty clouds, and
provides Numba-accelerated runtime branching and resolution kernels.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import attrs
import numpy as np
from numba import njit
from scipy.interpolate import CubicSpline

from pyloki.utils.kepler_math import keplerian_z
from pyloki.utils.misc import C_VAL

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy import typing as npt


@attrs.frozen(auto_attribs=True, kw_only=True)
class SVDBasisLevel:
    """Immutable SVD basis and transition data for one partial integration span."""

    t_axis: npt.NDArray[np.float64]
    v_phase: npt.NDArray[np.float64]
    v_vel: npt.NDArray[np.float64]
    v_acc: npt.NDArray[np.float64]
    delta: npt.NDArray[np.float64]
    tau_tol: float
    trans_matrix: npt.NDArray[np.float64] | None = None
    novelties: npt.NDArray[np.float64] | None = None

    @property
    def rank(self) -> int:
        return int(self.v_phase.shape[0])


def _simulate_delay_ensemble(
    t_axis: np.ndarray,
    *,
    p_orb_range: tuple[float, float],
    x_orb_range: tuple[float, float],
    ecc_range: tuple[float, float],
    n_examples: int,
    v_res_max: float = 0.0,
    a_spin_max: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Sample Keplerian arrival-time delays (seconds) over the prior box."""
    rng = np.random.default_rng(seed)
    p_orbs = rng.uniform(p_orb_range[0], p_orb_range[1], size=n_examples)
    x_orbs = rng.uniform(x_orb_range[0], x_orb_range[1], size=n_examples)
    eccs = rng.uniform(ecc_range[0], ecc_range[1], size=n_examples)
    aops = rng.uniform(0.0, 2.0 * math.pi, size=n_examples)
    phis = rng.uniform(0.0, 2.0 * math.pi, size=n_examples)

    t_mid = float(np.mean(t_axis))
    t_rel = t_axis - t_mid
    v_res = (
        rng.uniform(-v_res_max, v_res_max, size=n_examples)
        if v_res_max > 0.0
        else np.zeros(n_examples, dtype=np.float64)
    )
    a_spin = (
        rng.uniform(-a_spin_max, a_spin_max, size=n_examples)
        if a_spin_max > 0.0
        else np.zeros(n_examples, dtype=np.float64)
    )

    delays = np.empty((n_examples, t_axis.size), dtype=np.float64)
    for m in range(n_examples):
        z_sec = keplerian_z(
            t_axis,
            float(p_orbs[m]),
            float(eccs[m]),
            float(phis[m]),
            float(x_orbs[m]),
            float(aops[m]),
            0.5 * math.pi,
        )
        delays[m] = (
            z_sec
            + (v_res[m] / C_VAL) * t_rel
            + 0.5 * (a_spin[m] / C_VAL) * (t_rel * t_rel)
        )
    return delays


def build_svd_level(
    t_axis: npt.ArrayLike,
    delays_sec: npt.ArrayLike,
    *,
    tau_tol: float,
    minimax_tol: float = 0.5,
    max_rank: int | None = None,
) -> SVDBasisLevel:
    """Extract an empirical minimax-rank SVD basis strictly from phase delays.

    Velocity and acceleration basis vectors are computed via exact cubic-spline
    differentiation of each retained right-singular vector ``v_j(t)``.
    """
    t = np.ascontiguousarray(t_axis, dtype=np.float64)
    if t.ndim != 1 or t.size < 4:
        msg = "t_axis must be 1D with at least 4 time points"
        raise ValueError(msg)
    if tau_tol <= 0.0:
        msg = f"tau_tol must be positive, got {tau_tol}"
        raise ValueError(msg)

    z = np.asarray(delays_sec, dtype=np.float64)
    z_centered = (z - np.mean(z, axis=1, keepdims=True)) / tau_tol
    u, s, vt = np.linalg.svd(z_centered, full_matrices=False)

    # Minimax effective rank: max_m |U_mj * S_j| * max_t |V_jt| > minimax_tol
    max_c = np.max(np.abs(u), axis=0) * s
    max_v = np.max(np.abs(vt), axis=1)
    active = np.where((max_c * max_v) > minimax_tol)[0]
    rank = int(active[-1]) + 1 if active.size > 0 else 1
    if max_rank is not None:
        rank = min(rank, max(1, max_rank))

    v_phase = np.ascontiguousarray(vt[:rank, :], dtype=np.float64)
    v_vel = np.empty_like(v_phase)
    v_acc = np.empty_like(v_phase)
    for j in range(rank):
        cs = CubicSpline(t, v_phase[j])
        v_vel[j] = cs(t, 1) * tau_tol * C_VAL
        v_acc[j] = cs(t, 2) * tau_tol * C_VAL

    stds = np.std(v_phase, axis=1)
    delta = np.where(stds > 0.0, 1.0 / stds, 1.0).astype(np.float64)
    return SVDBasisLevel(
        t_axis=t,
        v_phase=v_phase,
        v_vel=v_vel,
        v_acc=v_acc,
        delta=delta,
        tau_tol=float(tau_tol),
    )


def build_svd_transition(
    parent: SVDBasisLevel,
    child: SVDBasisLevel,
    delays_child_sec: npt.ArrayLike,
) -> SVDBasisLevel:
    """Compute least-squares transition matrix and deduplicated novelty cloud."""
    # Restrict child basis onto parent time axis via cubic spline interpolation
    v_child_on_parent = np.empty((child.rank, parent.t_axis.size), dtype=np.float64)
    for j in range(child.rank):
        cs = CubicSpline(child.t_axis, child.v_phase[j])
        vals = cs(parent.t_axis)
        v_child_on_parent[j] = vals - float(np.mean(vals))

    a_prime = v_child_on_parent.T  # (n_parent, d_child)
    trans = np.linalg.pinv(a_prime) @ parent.v_phase.T  # (d_child, d_parent)

    # Project simulated trajectories onto parent & child bases to build novelty cloud
    z_child = np.asarray(delays_child_sec, dtype=np.float64)
    z_child_c = (z_child - np.mean(z_child, axis=1, keepdims=True)) / child.tau_tol

    # Interpolate trajectories onto parent time axis
    z_parent_c = np.empty((z_child.shape[0], parent.t_axis.size), dtype=np.float64)
    for m in range(z_child.shape[0]):
        vals = np.interp(parent.t_axis, child.t_axis, z_child[m])
        z_parent_c[m] = (vals - float(np.mean(vals))) / parent.tau_tol

    c_parent = z_parent_c @ parent.v_phase.T
    c_parent_q = np.rint(c_parent / parent.delta[None, :]) * parent.delta[None, :]
    c_child = z_child_c @ child.v_phase.T
    raw_nov = c_child - (c_parent_q @ trans.T)

    # Deduplicate novelty cloud on integer child lattice offline
    k_nov = np.rint(raw_nov / child.delta[None, :]).astype(np.int64)
    _, uniq_idx = np.unique(k_nov, axis=0, return_index=True)
    novelties = np.ascontiguousarray(raw_nov[np.sort(uniq_idx)], dtype=np.float64)

    return SVDBasisLevel(
        t_axis=child.t_axis,
        v_phase=child.v_phase,
        v_vel=child.v_vel,
        v_acc=child.v_acc,
        delta=child.delta,
        tau_tol=child.tau_tol,
        trans_matrix=np.ascontiguousarray(trans, dtype=np.float64),
        novelties=novelties,
    )


@attrs.frozen(auto_attribs=True, kw_only=True)
class SVDOrbitalBasis:
    """Chain of ``SVDBasisLevel`` objects across growing partial integration spans."""

    levels: tuple[SVDBasisLevel, ...]

    @classmethod
    def build(
        cls,
        span_axes: Sequence[npt.ArrayLike],
        *,
        p_orb_range: tuple[float, float],
        x_orb_range: tuple[float, float],
        ecc_range: tuple[float, float] = (0.0, 0.3),
        tau_tol: float,
        n_examples: int = 1000,
        v_res_max: float = 0.0,
        a_spin_max: float = 0.0,
        minimax_tol: float = 0.5,
        max_rank: int | None = None,
        seed: int | None = None,
    ) -> SVDOrbitalBasis:
        """Build an empirical SVD basis chain for a sequence of growing time spans."""
        if not span_axes:
            msg = "span_axes must be non-empty"
            raise ValueError(msg)
        axes = [np.ascontiguousarray(a, dtype=np.float64) for a in span_axes]
        # Simulate on the widest span and interpolate/slice for shorter spans
        t_full = axes[-1]
        full_delays = _simulate_delay_ensemble(
            t_full,
            p_orb_range=p_orb_range,
            x_orb_range=x_orb_range,
            ecc_range=ecc_range,
            n_examples=n_examples,
            v_res_max=v_res_max,
            a_spin_max=a_spin_max,
            seed=seed,
        )

        built: list[SVDBasisLevel] = []
        for idx, t_ax in enumerate(axes):
            if idx == len(axes) - 1:
                del_k = full_delays
            else:
                del_k = np.empty((n_examples, t_ax.size), dtype=np.float64)
                for m in range(n_examples):
                    del_k[m] = np.interp(t_ax, t_full, full_delays[m])
            lvl = build_svd_level(
                t_ax,
                del_k,
                tau_tol=tau_tol,
                minimax_tol=minimax_tol,
                max_rank=max_rank,
            )
            if built:
                lvl = build_svd_transition(built[-1], lvl, del_k)
            built.append(lvl)
        return cls(levels=tuple(built))

    def save_npz(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {
            "n_levels": np.array([len(self.levels)], dtype=np.int64),
        }
        for i, lvl in enumerate(self.levels):
            arrays[f"t_axis_{i}"] = lvl.t_axis
            arrays[f"v_phase_{i}"] = lvl.v_phase
            arrays[f"v_vel_{i}"] = lvl.v_vel
            arrays[f"v_acc_{i}"] = lvl.v_acc
            arrays[f"delta_{i}"] = lvl.delta
            arrays[f"tau_tol_{i}"] = np.array([lvl.tau_tol], dtype=np.float64)
            if lvl.trans_matrix is not None:
                arrays[f"trans_{i}"] = lvl.trans_matrix
            if lvl.novelties is not None:
                arrays[f"nov_{i}"] = lvl.novelties
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **arrays)  # ty: ignore[invalid-argument-type]

    @classmethod
    def load_npz(cls, path: str | Path) -> SVDOrbitalBasis:
        with np.load(Path(path)) as data:
            n_levels = int(data["n_levels"][0])
            levels: list[SVDBasisLevel] = []
            for i in range(n_levels):
                trans = data.get(f"trans_{i}")
                nov = data.get(f"nov_{i}")
                levels.append(
                    SVDBasisLevel(
                        t_axis=data[f"t_axis_{i}"],
                        v_phase=data[f"v_phase_{i}"],
                        v_vel=data[f"v_vel_{i}"],
                        v_acc=data[f"v_acc_{i}"],
                        delta=data[f"delta_{i}"],
                        tau_tol=float(data[f"tau_tol_{i}"][0]),
                        trans_matrix=trans,
                        novelties=nov,
                    ),
                )
        return cls(levels=tuple(levels))


@njit(cache=True)
def _hash_int_row(row: np.ndarray) -> np.uint64:
    """Fast 64-bit FNV-1a-style hash for integer lattice coordinate rows."""
    h = np.uint64(14695981039346656037)
    prime = np.uint64(1099511628211)
    for j in range(row.size):
        h ^= np.uint64(row[j] + 2147483648)
        h *= prime
    return h


@njit(cache=True)
def svd_branch_batch(
    parent_coords: np.ndarray,
    trans_matrix: np.ndarray,
    novelties: np.ndarray,
    delta_next: np.ndarray,
    parent_group_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand parent SVD coordinates along the novelty cloud with O(N log N) dedup.

    When ``parent_group_ids`` (e.g. quantized ``f0`` bucket IDs per parent) is
    provided, deduplication is scoped per group so distinct-``f0`` parents that
    share orbital coordinates never collapse into one another.

    Returns ``(child_coords, parent_origins)`` with unique lattice-snapped rows.
    """
    n_parents = parent_coords.shape[0]
    n_nov = novelties.shape[0]
    d_next = trans_matrix.shape[0]
    d_prev = trans_matrix.shape[1]
    total = n_parents * n_nov
    if total == 0:
        return (
            np.empty((0, d_next), dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )

    raw_keys = np.empty((total, d_next), dtype=np.int64)
    raw_groups = np.empty(total, dtype=np.int64)
    hashes = np.empty(total, dtype=np.uint64)
    raw_origins = np.empty(total, dtype=np.int64)
    base_c = np.empty(d_next, dtype=np.float64)

    out_idx = 0
    for p in range(n_parents):
        grp = int(parent_group_ids[p]) if parent_group_ids is not None else 0
        for r in range(d_next):
            acc = 0.0
            for c in range(d_prev):
                acc += trans_matrix[r, c] * parent_coords[p, c]
            base_c[r] = acc
        for nv in range(n_nov):
            for r in range(d_next):
                raw_keys[out_idx, r] = round(
                    (base_c[r] + novelties[nv, r]) / delta_next[r],
                )
            raw_groups[out_idx] = grp
            h = _hash_int_row(raw_keys[out_idx])
            h ^= np.uint64(grp + 2147483648)
            h *= np.uint64(1099511628211)
            hashes[out_idx] = h
            raw_origins[out_idx] = p
            out_idx += 1

    # Sort by hash for O(N log N) deduplication (verify exact group + row match)
    order = np.argsort(hashes)
    keep = np.zeros(total, dtype=np.bool_)
    n_keep = 0
    for pos in range(total):
        cur = order[pos]
        is_dup = False
        back = pos - 1
        while back >= 0 and hashes[order[back]] == hashes[cur]:
            prev = order[back]
            if raw_groups[cur] == raw_groups[prev]:
                same = True
                for r in range(d_next):
                    if raw_keys[cur, r] != raw_keys[prev, r]:
                        same = False
                        break
                if same:
                    is_dup = True
                    break
            back -= 1
        if not is_dup:
            keep[cur] = True
            n_keep += 1

    children = np.empty((n_keep, d_next), dtype=np.float64)
    origins = np.empty(n_keep, dtype=np.int64)
    w = 0
    for i in range(total):
        if keep[i]:
            for r in range(d_next):
                children[w, r] = raw_keys[i, r] * delta_next[r]
            origins[w] = raw_origins[i]
            w += 1
    return children, origins


@njit(cache=True)
def svd_eval_point(
    coords: np.ndarray,
    t_eval: float,
    t_axis: np.ndarray,
    v_phase: np.ndarray,
    v_vel: np.ndarray,
    v_acc: np.ndarray,
    tau_tol: float,
) -> tuple[float, float, float]:
    """Evaluate ``(delay_sec, vel_ms, acc_mss)`` at ``t_eval`` by linear interp."""
    n = t_axis.size
    if t_eval <= t_axis[0]:
        i0, i1, w1 = 0, 0, 0.0
    elif t_eval >= t_axis[n - 1]:
        i0, i1, w1 = n - 1, n - 1, 0.0
    else:
        idx = int(np.searchsorted(t_axis, t_eval))
        i0 = idx - 1
        i1 = idx
        w1 = (t_eval - t_axis[i0]) / (t_axis[i1] - t_axis[i0])
    w0 = 1.0 - w1

    delay = 0.0
    vel = 0.0
    acc = 0.0
    rank = coords.size
    for j in range(rank):
        cj = coords[j]
        delay += cj * (w0 * v_phase[j, i0] + w1 * v_phase[j, i1]) * tau_tol
        vel += cj * (w0 * v_vel[j, i0] + w1 * v_vel[j, i1])
        acc += cj * (w0 * v_acc[j, i0] + w1 * v_acc[j, i1])
    return delay, vel, acc


@njit(cache=True)
def svd_resolve_batch(
    coords_batch: np.ndarray,
    f0_batch: np.ndarray,
    t_ref: float,
    t_eval: float,
    t_axis: np.ndarray,
    v_phase: np.ndarray,
    v_vel: np.ndarray,
    v_acc: np.ndarray,
    tau_tol: float,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve SVD leaves at ``t_eval`` to ``(accel_mss, freq_hz, phase_bin)``."""
    n_batch = coords_batch.shape[0]
    acc_out = np.empty(n_batch, dtype=np.float64)
    freq_out = np.empty(n_batch, dtype=np.float64)
    phase_out = np.empty(n_batch, dtype=np.float64)

    dt = t_eval - t_ref
    for i in range(n_batch):
        d_ref, v_ref, _ = svd_eval_point(
            coords_batch[i],
            t_ref,
            t_axis,
            v_phase,
            v_vel,
            v_acc,
            tau_tol,
        )
        d_cur, v_cur, a_cur = svd_eval_point(
            coords_batch[i],
            t_eval,
            t_axis,
            v_phase,
            v_vel,
            v_acc,
            tau_tol,
        )
        f0 = f0_batch[i]
        # f0 is the observed spin frequency at t_ref: f0 = f_rest * (1 - v_ref / c)
        f_rest = f0 / (1.0 - v_ref / C_VAL)
        acc_out[i] = a_cur
        freq_out[i] = f_rest * (1.0 - v_cur / C_VAL)
        proper_dt = dt - (d_cur - d_ref)
        cyc = (proper_dt * f_rest) % 1.0
        if cyc < 0.0:
            cyc += 1.0
        pbin = cyc * nbins
        if pbin >= nbins:
            pbin = 0.0
        phase_out[i] = pbin
    return acc_out, freq_out, phase_out
