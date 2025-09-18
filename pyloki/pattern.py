from __future__ import annotations

from typing import TYPE_CHECKING

from pyloki.core import (
    generate_bp_chebyshev,
    generate_bp_chebyshev_approx,
    generate_bp_taylor,
    generate_bp_taylor_approx,
    generate_bp_taylor_circular,
    generate_bp_taylor_fixed,
    generate_bp_taylor_fixed_circular,
)

if TYPE_CHECKING:
    import numpy as np
    from numba import types


def generate_branching_pattern_approx(
    param_arr: types.ListType,
    dparams_lim: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tseg_ffa: float,
    nsegments: int,
    fold_bins: int,
    tol_bins: float,
    use_conservative_errors: bool = False,  # noqa: FBT002
    kind: str = "taylor",
) -> np.ndarray:
    """Generate the approximate branching pattern for the pruning search.

    This is a simplified version of the branching pattern that only tracks the
    worst-case branching factor.

    Returns
    -------
    np.ndarray
        Branching pattern for the pruning search.
    """
    ref_seg = 0
    isuggest = 0
    if kind == "taylor":
        return generate_bp_taylor_approx(
            param_arr,
            dparams_lim,
            param_limits,
            tseg_ffa,
            nsegments,
            fold_bins,
            tol_bins,
            ref_seg,
            isuggest,
            use_conservative_errors,
        )
    if kind == "chebyshev":
        return generate_bp_chebyshev_approx(
            param_arr,
            dparams_lim,
            param_limits,
            tseg_ffa,
            nsegments,
            fold_bins,
            tol_bins,
            ref_seg,
            isuggest,
            use_conservative_errors,
        )
    msg = f"Invalid kind: {kind}"
    raise ValueError(msg)


def generate_branching_pattern(
    param_arr: types.ListType,
    dparams_lim: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tseg_ffa: float,
    nsegments: int,
    fold_bins: int,
    tol_bins: float,
    ref_seg: int = 0,
    use_conservative_errors: bool = False,  # noqa: FBT002
    kind: str = "taylor",
) -> np.ndarray:
    """Generate the exact branching pattern for the pruning search.

    This tracks the exact number of branches per node to compute the average
    branching factor.

    Parameters
    ----------
    param_arr : types.ListType
        Parameter array for each dimension.
    dparams_lim : np.ndarray
        Parameter step (grid) sizes for each dimension in a 1D array.
    param_limits : types.ListType[types.Tuple[float, float]]
        Parameter limits for each dimension.
    tseg_ffa : float
        The duration of the starting segment.
    nsegments : int
        The number of segments to generate the branching pattern for.
    fold_bins : int
        The number of bins in the frequency array.
    tol_bins : float
        The tolerance for the branching pattern.
    ref_seg : int
        The reference segment to generate the branching pattern for.
    use_conservative_errors : bool
        Whether to use a conservative grid.
    kind : str
        The kind of branching pattern to generate.

    Returns
    -------
    np.ndarray
        The branching pattern for the pruning search.
    """
    if kind == "taylor":
        return generate_bp_taylor(
            param_arr,
            dparams_lim,
            param_limits,
            tseg_ffa,
            nsegments,
            fold_bins,
            tol_bins,
            ref_seg,
            use_conservative_errors,
        )
    if kind == "chebyshev":
        return generate_bp_chebyshev(
            param_arr,
            dparams_lim,
            param_limits,
            tseg_ffa,
            nsegments,
            fold_bins,
            tol_bins,
            ref_seg,
            use_conservative_errors,
        )
    if kind == "taylor_fixed":
        return generate_bp_taylor_fixed(
            param_arr,
            dparams_lim,
            param_limits,
            tseg_ffa,
            nsegments,
            fold_bins,
            tol_bins,
            ref_seg,
        )
    msg = f"Invalid kind: {kind}"
    raise ValueError(msg)

def generate_branching_pattern_circular(
    param_arr: types.ListType,
    dparams_lim: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tseg_ffa: float,
    nsegments: int,
    fold_bins: int,
    tol_bins: float,
    ref_seg: int = 0,
    use_conservative_errors: bool = False,  # noqa: FBT002
    kind: str = "taylor",
) -> np.ndarray:
    """Generate the exact branching pattern for the circular pruning search.

    This tracks the exact number of branches per node to compute the average
    branching factor.

    Parameters
    ----------
    param_arr : types.ListType
        Parameter array for each dimension.
    dparams_lim : np.ndarray
        Parameter step (grid) sizes for each dimension in a 1D array.
    param_limits : types.ListType[types.Tuple[float, float]]
        Parameter limits for each dimension.
    tseg_ffa : float
        The duration of the starting segment.
    nsegments : int
        The number of segments to generate the branching pattern for.
    fold_bins : int
        The number of bins in the frequency array.
    tol_bins : float
        The tolerance for the branching pattern.
    ref_seg : int
        The reference segment to generate the branching pattern for.
    use_conservative_errors : bool
        Whether to use a conservative grid.
    kind : str
        The kind of branching pattern to generate.

    Returns
    -------
    np.ndarray
        The branching pattern for the pruning search.
    """
    if kind == "taylor":
        return generate_bp_taylor_circular(
            param_arr,
            dparams_lim,
            param_limits,
            tseg_ffa,
            nsegments,
            fold_bins,
            tol_bins,
            ref_seg,
            use_conservative_errors,
        )
    if kind == "taylor_fixed":
        return generate_bp_taylor_fixed_circular(
            param_arr,
            dparams_lim,
            param_limits,
            tseg_ffa,
            nsegments,
            fold_bins,
            tol_bins,
            ref_seg,
        )
    msg = f"Invalid kind: {kind}"
    raise ValueError(msg)
