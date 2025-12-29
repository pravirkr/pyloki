# ruff: noqa: ARG001

from __future__ import annotations

from typing import Self

import numpy as np
from numba import njit, types
from numba.experimental import structref
from numba.extending import overload, overload_method


@structref.register
class MiddleOutSchemeTemplate(types.StructRef):
    pass


class MiddleOutScheme(structref.StructRefProxy):
    """A utility class for indexing segments in a hierarchical pruning algorithm.

    The scheme allow for "middle-out" enumeration of the segments.

    Parameters
    ----------
    nsegments : int
        The total number of segments (M) in the hierarchical search scheme.
    ref_idx : int
        The reference (starting) segment index (q) for pruning.
    tsegment : float, optional
        The duration of each segment in seconds, default is 1.0.
    """

    def __new__(cls, nsegments: int, ref_idx: int, tsegment: float = 1.0) -> Self:
        """Create a new instance of MiddleOutScheme."""
        return middle_out_scheme_init_func(nsegments, ref_idx, tsegment)

    @property
    @njit(cache=True, fastmath=True)
    def nsegments(self) -> int:
        return self.nsegments

    @property
    @njit(cache=True, fastmath=True)
    def ref_idx(self) -> int:
        return self.ref_idx

    @property
    @njit(cache=True, fastmath=True)
    def tsegment(self) -> float:
        return self.tsegment

    @property
    @njit(cache=True, fastmath=True)
    def data(self) -> np.ndarray:
        return self.data

    @property
    @njit(cache=True, fastmath=True)
    def ref_time(self) -> float:
        """Reference time at the middle of the reference segment in seconds."""
        return (self.ref_idx + 0.5) * self.tsegment

    @njit(cache=True, fastmath=True)
    def get_segment_idx(self, level: int) -> int:
        """Get the segment index at the specified hierarchical level.

        Parameters
        ----------
        level : int
            The hierarchical level, where 0 is the reference segment.

        Returns
        -------
        int
            The segment index at the given level.
        """
        return get_segment_idx_func(self, level)

    @njit(cache=True, fastmath=True)
    def get_coord(self, level: int) -> tuple[float, float]:
        """Get the current coord (ref and scale) at the given level.

        The reference time is the center of the time interval covered by all segments
        from level 0 to the specified level. The scale is the half-width of this
        interval.

        Parameters
        ----------
        level : int
            The current hierarchical level, where 0 is the reference segment.

        Returns
        -------
        tuple[float, float]
            The reference and scale for the current level in seconds.
        """
        return get_coord_func(self, level)

    @njit(cache=True, fastmath=True)
    def get_segment_coord(self, level: int) -> tuple[float, float]:
        """Get the ref and scale for the segment (to be added) at the given level.

        Parameters
        ----------
        level : int
            The hierarchical level, where 0 is the reference segment.

        Returns
        -------
        tuple[float, float]
            The reference and scale for the segment at the given level in seconds.
        """
        return get_segment_coord_func(self, level)

    @njit(cache=True, fastmath=True)
    def get_current_coord(self, level: int) -> tuple[float, float]:
        """Get current ref, scale for an adaptive grid."""
        return get_current_coord_func(self, level)

    @njit(cache=True, fastmath=True)
    def get_current_coord_fixed(self, level: int) -> tuple[float, float]:
        """Get fixed ref, scale for a fixed grid."""
        return get_current_coord_fixed_func(self, level)

    @njit(cache=True, fastmath=True)
    def get_valid(self, prune_level: int) -> tuple[float, float]:
        return get_valid_func(self, prune_level)

    @njit(cache=True, fastmath=True)
    def get_delta(self, level: int) -> float:
        """Get the difference between the current coord and the reference.

        This measures the shift of the current coord from the reference coord.

        Parameters
        ----------
        level : int
            The hierarchical level, where 0 is the reference segment.

        Returns
        -------
        float
            The difference between the current coord and the reference in seconds.
        """
        return get_delta_func(self, level)


fields_middle_out_scheme = [
    ("nsegments", types.i8),
    ("ref_idx", types.i8),
    ("tsegment", types.f8),
    ("data", types.i8[:]),
]

structref.define_boxing(MiddleOutSchemeTemplate, MiddleOutScheme)
MiddleOutSchemeType = MiddleOutSchemeTemplate(fields_middle_out_scheme)


@njit(cache=True, fastmath=True)
def middle_out_scheme_init_func(
    nsegments: int,
    ref_idx: int,
    tsegment: float,
) -> MiddleOutScheme:
    self = structref.new(MiddleOutSchemeType)
    if nsegments <= 0:
        msg = f"nsegments must be greater than 0, got {nsegments}."
        raise ValueError(msg)
    if ref_idx < 0 or ref_idx >= nsegments:
        msg = f"ref_idx must be in [0, {nsegments}), got {ref_idx}."
        raise ValueError(msg)
    if tsegment <= 0:
        msg = f"tsegment must be greater than 0, got {tsegment}."
        raise ValueError(msg)
    self.nsegments = nsegments
    self.ref_idx = ref_idx
    self.tsegment = tsegment
    self.data = np.argsort(np.abs(np.arange(nsegments) - ref_idx), kind="mergesort")
    return self


@njit(cache=True, fastmath=True)
def get_segment_idx_func(self: MiddleOutScheme, level: int) -> int:
    if level < 0 or level >= self.nsegments:
        msg = f"level must be in [0, {self.nsegments}), got {level}."
        raise ValueError(msg)
    return self.data[level]


@njit(cache=True, fastmath=True)
def get_coord_func(self: MiddleOutScheme, level: int) -> tuple[float, float]:
    if level < 0 or level >= self.nsegments:
        msg = f"level must be in [0, {self.nsegments - 1}], got {level}."
        raise ValueError(msg)
    scheme_till_now = self.data[: level + 1]
    ref = (np.min(scheme_till_now) + np.max(scheme_till_now) + 1) / 2
    scale = ref - np.min(scheme_till_now)
    return ref * self.tsegment, scale * self.tsegment


@njit(cache=True, fastmath=True)
def get_segment_coord_func(self: MiddleOutScheme, level: int) -> tuple[float, float]:
    if level < 0 or level >= self.nsegments:
        msg = f"level must be in [0, {self.nsegments - 1}], got {level}."
        raise ValueError(msg)
    ref = (self.get_segment_idx(level) + 0.5) * self.tsegment
    scale = 0.5 * self.tsegment
    return ref, scale


@njit(cache=True, fastmath=True)
def get_current_coord_func(self: MiddleOutScheme, level: int) -> tuple[float, float]:
    if level < 0 or level >= self.nsegments:
        msg = f"level must be in [0, {self.nsegments - 1}], got {level}."
        raise ValueError(msg)

    if level == 0:
        return self.get_coord(level)
    prev_ref, _ = self.get_coord(level - 1)
    _, cur_scale = self.get_coord(level)
    return prev_ref, cur_scale


@njit(cache=True, fastmath=True)
def get_current_coord_fixed_func(
    self: MiddleOutScheme,
    level: int,
) -> tuple[float, float]:
    if level < 0 or level >= self.nsegments:
        msg = f"level must be in [0, {self.nsegments - 1}], got {level}."
        raise ValueError(msg)
    if level == 0:
        return self.get_coord(level)
    t0_init, scale_init = self.get_coord(0)
    t0_next, scale_next = self.get_coord(level)
    left_edge = t0_next - scale_next
    right_edge = t0_next + scale_next
    scale_cur = max(abs(left_edge - t0_init), abs(right_edge - t0_init))
    return t0_init, scale_cur - scale_init


@njit(cache=True, fastmath=True)
def get_valid_func(self: MiddleOutScheme, prune_level: int) -> tuple[float, float]:
    scheme_till_now = self.data[:prune_level]
    return np.min(scheme_till_now), np.max(scheme_till_now)


@njit(cache=True, fastmath=True)
def get_delta_func(self: MiddleOutScheme, level: int) -> float:
    return self.get_coord(level)[0] - self.ref_time


@overload(MiddleOutScheme)
def overload_middle_out_scheme_construct(
    nsegments: int,
    ref_idx: int,
    tsegment: float,
) -> types.FunctionType:
    def impl(nsegments: int, ref_idx: int, tsegment: float) -> MiddleOutScheme:
        return middle_out_scheme_init_func(nsegments, ref_idx, tsegment)

    return impl


@overload_method(MiddleOutSchemeTemplate, "get_segment_idx")
def ol_get_segment_idx_func(self: MiddleOutScheme, level: int) -> types.FunctionType:
    def impl(self: MiddleOutScheme, level: int) -> int:
        return get_segment_idx_func(self, level)

    return impl


@overload_method(MiddleOutSchemeTemplate, "get_coord")
def ol_get_coord_func(self: MiddleOutScheme, level: int) -> types.FunctionType:
    def impl(self: MiddleOutScheme, level: int) -> tuple[float, float]:
        return get_coord_func(self, level)

    return impl


@overload_method(MiddleOutSchemeTemplate, "get_segment_coord")
def ol_get_segment_coord_func(self: MiddleOutScheme, level: int) -> types.FunctionType:
    def impl(self: MiddleOutScheme, level: int) -> tuple[float, float]:
        return get_segment_coord_func(self, level)

    return impl


@overload_method(MiddleOutSchemeTemplate, "get_current_coord")
def ol_get_current_coord_func(self: MiddleOutScheme, level: int) -> types.FunctionType:
    def impl(self: MiddleOutScheme, level: int) -> tuple[float, float]:
        return get_current_coord_func(self, level)

    return impl


@overload_method(MiddleOutSchemeTemplate, "get_current_coord_fixed")
def ol_get_current_coord_fixed_func(
    self: MiddleOutScheme,
    level: int,
) -> types.FunctionType:
    def impl(self: MiddleOutScheme, level: int) -> tuple[float, float]:
        return get_current_coord_fixed_func(self, level)

    return impl


@overload_method(MiddleOutSchemeTemplate, "get_valid")
def ol_get_valid_func(self: MiddleOutScheme, prune_level: int) -> types.FunctionType:
    def impl(self: MiddleOutScheme, prune_level: int) -> tuple[float, float]:
        return get_valid_func(self, prune_level)

    return impl


@overload_method(MiddleOutSchemeTemplate, "get_delta")
def ol_get_delta_func(self: MiddleOutScheme, level: int) -> types.FunctionType:
    def impl(self: MiddleOutScheme, level: int) -> float:
        return get_delta_func(self, level)

    return impl
