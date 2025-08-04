from __future__ import annotations

import numpy as np
from numba import njit, types

from pyloki.utils import np_utils, psr_utils
from pyloki.utils.misc import C_VAL


@njit(cache=True, fastmath=True)
def poly_circular_resolve_batch(
    leaf_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
    fold_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
    n_leaves = len(leaf_batch)
    nparams = len(param_arr)
    delta_t = coord_add[0] - coord_init[0]

    # Extract parameters leaf_batch[:, :-2]
    omega_batch = leaf_batch[:, 0, 0]
    x_cos_phi_batch = leaf_batch[:, 1, 0]
    x_sin_phi_batch = leaf_batch[:, 2, 0]
    freq_old_batch = leaf_batch[:, 3, 0]

    # x is already in light seconds (no division by C_VAL)
    new_x_cos_om_t_plus_phi_batch = x_cos_phi_batch * np.cos(
        omega_batch * delta_t,
    ) - x_sin_phi_batch * np.sin(omega_batch * delta_t)
    new_x_sin_om_t_plus_phi_batch = x_sin_phi_batch * np.cos(
        omega_batch * delta_t,
    ) + x_cos_phi_batch * np.sin(omega_batch * delta_t)
    delay_batch = new_x_cos_om_t_plus_phi_batch - x_cos_phi_batch
    new_v_over_c_batch = -new_x_sin_om_t_plus_phi_batch * omega_batch
    new_freq_batch = freq_old_batch * (1 - new_v_over_c_batch)
    new_a_batch = (
        -(omega_batch**2) * (new_x_cos_om_t_plus_phi_batch - x_cos_phi_batch) * C_VAL
    )

    relative_phase_batch = psr_utils.get_phase_idx(
        delta_t,
        freq_old_batch,
        fold_bins,
        delay_batch,
    )
    param_idx_batch = np.zeros((n_leaves, nparams), dtype=np.int64)
    param_idx_batch[:, -1] = np_utils.find_nearest_sorted_idx_vect(
        param_arr[-1],
        new_freq_batch,
    )
    param_idx_batch[:, -2] = np_utils.find_nearest_sorted_idx_vect(
        param_arr[-2],
        new_a_batch,
    )
    return param_idx_batch, relative_phase_batch


@njit(cache=True, fastmath=True)
def poly_circular_branch_batch(
    param_set_batch: np.ndarray,
    coord_cur: tuple[float, float],
    fold_bins: int,
    tol_bins: float,
    poly_order: int,
    param_limits: types.ListType[types.Tuple[float, float]],
    branch_max: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Branch a batch of parameter sets to leaves."""
    n_batch = len(param_set_batch)
    # Only two parameters to branch: omega and frequency; x_cos_phi, x_sin_phi params
    # never get refined (their accuracy does not increase with time).
    nparams = 2
    _, scale_cur = coord_cur
    param_cur_batch = param_set_batch[:, :nparams, 0]
    dparam_cur_batch = param_set_batch[:, :nparams, 1]
    x_cos_phi_cur_batch = param_set_batch[:, 2, 0]
    dx_cos_phi_cur_batch = param_set_batch[:, 2, 1]
    x_sin_phi_cur_batch = param_set_batch[:, 3, 0]
    dx_sin_phi_cur_batch = param_set_batch[:, 3, 1]
    f0_batch = param_set_batch[:, -2, 0]
    t0_batch = param_set_batch[:, -1, 0]
    scale_batch = param_set_batch[:, -1, 1]
    x_cur_batch = np.sqrt(x_cos_phi_cur_batch**2 + x_sin_phi_cur_batch**2)

    tseg_cur = 2 * scale_cur
    dparam_opt_batch = np.empty((n_batch, nparams), dtype=np.float64)
    domega_opt_batch = psr_utils.poly_taylor_step_d_vec(
        nparams,
        tseg_cur,
        fold_bins,
        tol_bins,
        x_cur_batch,
    )
    dfreq_opt_batch = psr_utils.poly_taylor_step_f(
        1,
        tseg_cur,
        fold_bins,
        tol_bins,
        t_ref=tseg_cur / 2,
    )
    dparam_opt_batch[:, 0] = domega_opt_batch
    dparam_opt_batch[:, 1] = dfreq_opt_batch

    shift_bins_batch = psr_utils.poly_taylor_shift_d_vec(
        dparam_cur_batch,
        dparam_opt_batch,
        tseg_cur,
        fold_bins,
        x_cur_batch,
        t_ref=tseg_cur / 2,
    )

    # --- Vectorized Padded Branching ---
    pad_branched_params = np.empty((n_batch, nparams, branch_max), dtype=np.float64)
    pad_branched_dparams = np.empty((n_batch, nparams), dtype=np.float64)
    branched_counts = np.empty((n_batch, nparams), dtype=np.int64)
    for i in range(n_batch):
        for j in range(nparams):
            p_min, p_max = param_limits[j]
            dparam_act, count = psr_utils.branch_param_padded(
                pad_branched_params[i, j],
                param_cur_batch[i, j],
                dparam_cur_batch[i, j],
                dparam_opt_batch[i, j],
                p_min,
                p_max,
            )
            pad_branched_dparams[i, j] = dparam_act
            branched_counts[i, j] = count

    # --- Vectorized Selection ---
    # Select based on mask: shape (n_batch, nparams, 1)
    mask_2d = shift_bins_batch > tol_bins  # Shape (n_batch, nparams)
    for i in range(n_batch):
        for j in range(nparams):
            if not mask_2d[i, j]:
                pad_branched_params[i, j, :] = 0
                pad_branched_params[i, j, 0] = param_cur_batch[i, j]
                pad_branched_dparams[i, j] = dparam_cur_batch[i, j]
                branched_counts[i, j] = 1

    # --- Optimized Padded Cartesian Product ---
    batch_leaves_circular, batch_origins = np_utils.cartesian_prod_padded(
        pad_branched_params,
        branched_counts,
        n_batch,
        nparams,
    )
    total_leaves = len(batch_origins)
    batch_leaves = np.zeros((total_leaves, poly_order + 2, 2), dtype=np.float64)
    batch_leaves[:, :-2, 0] = batch_leaves_circular
    batch_leaves[:, :-2, 1] = pad_branched_dparams[batch_origins]
    batch_leaves[:, 2, 0] = x_cos_phi_cur_batch[batch_origins]
    batch_leaves[:, 2, 1] = dx_cos_phi_cur_batch[batch_origins]
    batch_leaves[:, 3, 0] = x_sin_phi_cur_batch[batch_origins]
    batch_leaves[:, 3, 1] = dx_sin_phi_cur_batch[batch_origins]
    batch_leaves[:, -2, 0] = f0_batch[batch_origins]
    batch_leaves[:, -1, 0] = t0_batch[batch_origins]
    batch_leaves[:, -1, 1] = scale_batch[batch_origins]
    return batch_leaves, batch_origins
