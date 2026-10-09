"""Unit tests for `pyloki.utils.snail`, the middle-out segment scheme.

`MiddleOutScheme` fixes the order in which the pruning search adds segments: outward
from a reference segment ``q``, nearest first. Every pruning level asks it for the time
coordinates (reference time and half-width) of the data seen so far, of the segment
being added, and of the grid the parameters are expressed on. A wrong coordinate does
not raise; it shifts every parameter transform in the search.

The references are computed independently of the kernel:

- the order is ``sorted(range(M), key=lambda i: (abs(i - q), i))``: nearest first, the
  left segment first on a tie. At every level the covered segments must stay
  contiguous;
- each coordinate follows from the covered time interval ``[a, b]``: the centre and the
  half-width of ``[a, b]`` for the scheme's coordinate, and, for a grid fixed at the
  reference segment, the furthest edge from ``t0 = (q + 1/2) tsegment`` less the
  reference segment's own half-width.

The kernels take the ``structref`` itself, so their ``py_func`` runs against the Python
proxy. The ``structref.new`` constructor cannot run in pure Python and is tested
compiled-only.
"""

from __future__ import annotations

import numpy as np
import pytest
from numba.core.errors import TypingError

from pyloki.utils import snail
from tests.jit_utils import jit_variants

# Library defects found while writing these tests. Each is pinned with a strict xfail
# so that the fix, when it lands, turns the pin into a failure and forces its removal.

SCHEMES = [
    (7, 2, 0.5, 1),
    (8, 0, 1.0, 1),
    (8, 7, 2.0, 1),
    (1, 0, 1.0, 1),
    (6, 3, 1.5, 1),
]
STRIDED = [(9, 4, 1.0, 3), (10, 1, 0.5, 2), (6, 5, 2.0, 4)]


def _order(nsegments: int, ref_idx: int) -> list[int]:
    return sorted(range(nsegments), key=lambda i: (abs(i - ref_idx), i))


def _interval(
    nsegments: int,
    ref_idx: int,
    tseg: float,
    level: int,
) -> tuple[float, float]:
    """Return the time interval [a, b] covered by levels 0..level."""
    seen = _order(nsegments, ref_idx)[: level + 1]
    return min(seen) * tseg, (max(seen) + 1) * tseg


def _coord(
    nsegments: int,
    ref_idx: int,
    tseg: float,
    level: int,
) -> tuple[float, float]:
    a, b = _interval(nsegments, ref_idx, tseg, level)
    return (a + b) / 2, (b - a) / 2


def _fixed_grid_scale(nsegments: int, ref_idx: int, tseg: float, level: int) -> float:
    """Return the furthest covered edge from t0, less the reference half-width."""
    t0 = (ref_idx + 0.5) * tseg
    a, b = _interval(nsegments, ref_idx, tseg, level)
    return max(abs(a - t0), abs(b - t0)) - tseg / 2


def _scheme(spec: tuple) -> snail.MiddleOutScheme:
    return snail.MiddleOutScheme(*spec)


class TestConstruction:
    @pytest.mark.parametrize("spec", SCHEMES + STRIDED)
    def test_fields(self, spec) -> None:
        _, ref_idx, tseg, _ = spec
        s = _scheme(spec)
        assert (s.nsegments, s.ref_idx, s.tsegment, s.stride) == spec
        assert s.ref_time == pytest.approx((ref_idx + 0.5) * tseg)
        props = type(s)
        for name in ("nsegments", "ref_idx", "tsegment", "stride", "ref_time"):
            getter = getattr(props, name).fget
            assert getter.py_func(s) == getter(s)
        np.testing.assert_array_equal(props.data.fget.py_func(s), s.data)

    @pytest.mark.parametrize(
        ("args", "match"),
        [
            ((0, 0, 1.0, 1), "nsegments"),
            ((5, -1, 1.0, 1), "ref_idx"),
            ((5, 5, 1.0, 1), "ref_idx"),
            ((5, 2, 0.0, 1), "tsegment"),
            ((5, 2, 1.0, 0), "stride"),
        ],
    )
    def test_rejects_bad_input(self, args, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            snail.MiddleOutScheme(*args)

    def test_order_is_middle_out_and_contiguous(self) -> None:
        for nsegments in range(1, 30):
            for ref_idx in range(nsegments):
                data = snail.MiddleOutScheme(nsegments, ref_idx, 1.0, 1).data
                assert list(data) == _order(nsegments, ref_idx), (nsegments, ref_idx)
                for level in range(nsegments):
                    seen = data[: level + 1]
                    assert set(seen) == set(range(seen.min(), seen.max() + 1))


class TestCoordinates:
    @pytest.mark.parametrize("impl", jit_variants(snail.get_segment_idx_func))
    @pytest.mark.parametrize("spec", SCHEMES)
    def test_segment_idx(self, impl, spec) -> None:
        s = _scheme(spec)
        order = _order(spec[0], spec[1])
        assert [impl(s, level) for level in range(spec[0])] == order

    @pytest.mark.parametrize("impl", jit_variants(snail.get_coord_func))
    @pytest.mark.parametrize("spec", SCHEMES)
    def test_coord_is_centre_and_half_width(self, impl, spec) -> None:
        s = _scheme(spec)
        for level in range(spec[0]):
            assert impl(s, level) == pytest.approx(_coord(*spec[:3], level))
        assert impl(s, 0) == pytest.approx((s.ref_time, spec[2] / 2))

    @pytest.mark.parametrize("impl", jit_variants(snail.get_segment_coord_func))
    @pytest.mark.parametrize("spec", SCHEMES)
    def test_segment_coord(self, impl, spec) -> None:
        s = _scheme(spec)
        tseg = spec[2]
        for level, idx in enumerate(_order(spec[0], spec[1])):
            assert impl(s, level) == pytest.approx(((idx + 0.5) * tseg, tseg / 2))

    @pytest.mark.parametrize("impl", jit_variants(snail.get_segment_coords_so_far_func))
    @pytest.mark.parametrize("spec", SCHEMES)
    def test_segment_coords_so_far(self, impl, spec) -> None:
        s = _scheme(spec)
        level = spec[0] - 1
        seg_idx, coords = impl(s, level)
        np.testing.assert_array_equal(seg_idx, _order(spec[0], spec[1]))
        for i, idx in enumerate(seg_idx):
            np.testing.assert_allclose(coords[i], ((idx + 0.5) * spec[2], spec[2] / 2))

    @pytest.mark.parametrize("spec", SCHEMES)
    def test_delta(self, spec) -> None:
        """Contract on ``py_func`` only; the compiled kernel does not compile."""
        impl = snail.get_delta_func.py_func
        s = _scheme(spec)
        t0 = (spec[1] + 0.5) * spec[2]
        for level in range(spec[0]):
            assert impl(s, level) == pytest.approx(_coord(*spec[:3], level)[0] - t0)

    @pytest.mark.xfail(strict=True, reason="#53", raises=TypingError)
    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda s: snail.get_delta_func(s, 1), id="kernel"),
            pytest.param(lambda s: s.get_delta(1), id="method"),
            pytest.param(lambda s: type(s).get_delta.py_func(s, 1), id="method_py"),
        ],
    )
    def test_delta_compiles(self, call) -> None:
        """Pin `get_delta_func` reading ``self.ref_time``.

        ``ref_time`` is a property of the Python proxy, not a field of the
        ``structref``, so compiled code has no such attribute (`TypingError`). The
        proxy's `get_delta` method is itself ``@njit`` and fails the same way.
        """
        call(_scheme((7, 2, 0.5, 1)))

    @pytest.mark.parametrize(
        "impl",
        [
            p.values[0]
            for func in (
                snail.get_segment_idx_func,
                snail.get_coord_func,
                snail.get_segment_coord_func,
                snail.get_segment_coords_so_far_func,
                snail.get_anchor_level_func,
                snail.do_transform_func,
                snail.get_current_coord_stride_func,
            )
            for p in jit_variants(func)
        ],
    )
    @pytest.mark.parametrize("level", [-1, 7])
    def test_rejects_out_of_range_level(self, impl, level: int) -> None:
        with pytest.raises(ValueError, match="level must be in"):
            impl(_scheme((7, 2, 0.5, 1)), level)


class TestGridCoordinates:
    """The coordinates the pruning search expresses its parameters in."""

    @pytest.mark.parametrize("impl", jit_variants(snail.get_current_coord_func))
    @pytest.mark.parametrize("spec", SCHEMES)
    def test_current_coord(self, impl, spec) -> None:
        s = _scheme(spec)
        nsegments = spec[0]
        assert impl(s, 0, moving_grid=True) == pytest.approx(_coord(*spec[:3], 0))
        assert impl(s, 0, moving_grid=False) == pytest.approx(_coord(*spec[:3], 0))
        for level in range(1, nsegments):
            # Moving grid: the previous level's centre, the current level's width.
            moving = (_coord(*spec[:3], level - 1)[0], _coord(*spec[:3], level)[1])
            assert impl(s, level, moving_grid=True) == pytest.approx(moving)
            # Fixed grid: anchored at the reference segment's centre.
            fixed = (s.ref_time, _fixed_grid_scale(*spec[:3], level))
            assert impl(s, level, moving_grid=False) == pytest.approx(fixed)

    @pytest.mark.parametrize("impl", jit_variants(snail.get_previous_coord_func))
    @pytest.mark.parametrize("spec", [s for s in SCHEMES if s[0] > 1])
    def test_previous_coord(self, impl, spec) -> None:
        s = _scheme(spec)
        for level in range(1, spec[0]):
            assert impl(s, level, moving_grid=True) == pytest.approx(
                _coord(*spec[:3], level - 1)
            )
            expected = (
                _coord(*spec[:3], 0)
                if level == 1
                else (s.ref_time, _fixed_grid_scale(*spec[:3], level - 1))
            )
            assert impl(s, level, moving_grid=False) == pytest.approx(expected)
        with pytest.raises(ValueError, match="level must be in"):
            impl(s, 0, moving_grid=True)

    @pytest.mark.parametrize("impl", jit_variants(snail.get_report_coord_func))
    @pytest.mark.parametrize("spec", SCHEMES)
    def test_report_coord(self, impl, spec) -> None:
        """Report at the centre of the full span.

        With a fixed grid, the width stays the one measured from the reference
        segment, as in `get_current_coord`.
        """
        s = _scheme(spec)
        last = spec[0] - 1
        full = _coord(*spec[:3], last)
        assert impl(s, moving_grid=True) == pytest.approx(full)
        fixed = (full[0], _fixed_grid_scale(*spec[:3], last))
        assert impl(s, moving_grid=False) == pytest.approx(fixed)


class TestStride:
    @pytest.mark.parametrize("impl", jit_variants(snail.get_anchor_level_func))
    @pytest.mark.parametrize("spec", STRIDED)
    def test_anchor_is_last_multiple_of_stride_before_level(self, impl, spec) -> None:
        s = _scheme(spec)
        stride = spec[3]
        assert impl(s, 0) == 0
        for level in range(1, spec[0]):
            assert impl(s, level) == ((level - 1) // stride) * stride

    @pytest.mark.parametrize("impl", jit_variants(snail.get_anchor_level_func))
    def test_anchor_with_unit_stride_is_previous_level(self, impl) -> None:
        s = _scheme((7, 2, 0.5, 1))
        assert [impl(s, level) for level in range(1, 7)] == list(range(6))

    @pytest.mark.parametrize("impl", jit_variants(snail.do_transform_func))
    @pytest.mark.parametrize("spec", [*STRIDED, (7, 2, 0.5, 1)])
    def test_do_transform_on_multiples_of_stride(self, impl, spec) -> None:
        s = _scheme(spec)
        stride = spec[3]
        got = [impl(s, level) for level in range(spec[0])]
        assert got == [level > 0 and level % stride == 0 for level in range(spec[0])]

    @pytest.mark.parametrize("impl", jit_variants(snail.get_current_coord_stride_func))
    @pytest.mark.parametrize("spec", [*STRIDED, (7, 2, 0.5, 1)])
    def test_current_coord_stride(self, impl, spec) -> None:
        """Anchored at the last grid update; width reaches the furthest edge."""
        s = _scheme(spec)
        nsegments, _, tseg, stride = spec
        assert impl(s, 0) == pytest.approx(_coord(*spec[:3], 0))
        for level in range(1, nsegments):
            anchor_level = ((level - 1) // stride) * stride
            anchor_ref = _coord(*spec[:3], anchor_level)[0]
            a, b = _interval(*spec[:3], level)
            width = max(abs(a - anchor_ref), abs(b - anchor_ref)) - tseg / 2
            assert impl(s, level) == pytest.approx((anchor_ref, width))


class TestValidAndMethods:
    @pytest.mark.parametrize("impl", jit_variants(snail.get_valid_func))
    @pytest.mark.parametrize("spec", SCHEMES)
    def test_valid_is_the_covered_segment_range(self, impl, spec) -> None:
        s = _scheme(spec)
        order = _order(spec[0], spec[1])
        for prune_level in range(1, spec[0] + 1):
            seen = order[:prune_level]
            assert impl(s, prune_level) == (min(seen), max(seen))

    @pytest.mark.parametrize("impl", jit_variants(snail.get_current_coord_func))
    @pytest.mark.parametrize("level", [-1, 7])
    def test_current_coord_rejects_out_of_range_level(self, impl, level: int) -> None:
        with pytest.raises(ValueError, match="level must be in"):
            impl(_scheme((7, 2, 0.5, 1)), level, moving_grid=True)

    def test_method_wrappers_on_both_paths(self) -> None:
        """Run the Python body of each ``@njit`` method wrapper on the proxy too."""
        s = _scheme((9, 4, 1.0, 3))
        cls = type(s)
        for name, args in [
            ("get_segment_idx", (1,)),
            ("get_coord", (3,)),
            ("get_anchor_level", (5,)),
            ("get_segment_coord", (2,)),
            ("get_current_coord", (4, True)),
            ("get_previous_coord", (4, False)),
            ("get_report_coord", (True,)),
            ("get_current_coord_stride", (5,)),
            ("do_transform", (3,)),
            ("get_valid", (4,)),
        ]:
            method = getattr(cls, name)
            assert method.py_func(s, *args) == method(s, *args), name
        seg_idx, _ = cls.get_segment_coords_so_far.py_func(s, 3)
        np.testing.assert_array_equal(seg_idx, s.data[:4])

    def test_methods_route_to_the_kernels(self) -> None:
        """The proxy methods are thin wrappers; one call each, for the wiring."""
        s = _scheme((9, 4, 1.0, 3))
        assert s.get_segment_idx(1) == snail.get_segment_idx_func(s, 1)
        assert s.get_coord(3) == snail.get_coord_func(s, 3)
        assert s.get_segment_coord(2) == snail.get_segment_coord_func(s, 2)
        assert s.get_anchor_level(5) == snail.get_anchor_level_func(s, 5)
        assert s.get_current_coord(4, moving_grid=True) == snail.get_current_coord_func(
            s,
            4,
            moving_grid=True,
        )
        assert s.get_previous_coord(
            4, moving_grid=False
        ) == snail.get_previous_coord_func(
            s,
            4,
            moving_grid=False,
        )
        assert s.get_report_coord(moving_grid=True) == snail.get_report_coord_func(
            s,
            moving_grid=True,
        )
        assert s.get_current_coord_stride(5) == snail.get_current_coord_stride_func(
            s, 5
        )
        assert s.do_transform(3) == snail.do_transform_func(s, 3)
        assert s.get_valid(4) == snail.get_valid_func(s, 4)
        seg_idx, _ = s.get_segment_coords_so_far(3)
        np.testing.assert_array_equal(seg_idx, s.data[:4])
