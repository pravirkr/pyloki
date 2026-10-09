"""Wiring tests for `pyloki.dynamic`, the search-facing function bundles.

Each ``dyn_*`` module packs a search configuration into a numba ``structref`` and
exposes the operations the FFA and pruning loops call (``seed``, ``branch``,
``validate``, ``resolve``, ``transform``, ``report``, ``shift_add``, ``score``,
``ascend``, ...). The kernels behind them are tested in `test_taylor.py`,
`test_chebyshev.py`, `test_circular.py`, `test_fold.py` and `test_common.py`. What is
left to test here is the routing: each operation must call the right kernel with the
bundle's configuration and the right coordinates, and switch correctly on
``use_moving_grid``.

These run on the compiled path only. The bundles' Python proxies expose methods but not
fields, so a ``py_func`` cannot read ``self.nbins`` and the rest, and the method bodies
are compiled by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
from numba import typed

from pyloki.core import chebyshev, circular, common, fold, taylor
from pyloki.detection import scoring
from pyloki.dynamic import (
    dyn_circular_taylor,
    dyn_pffa,
    dyn_poly_cheby,
    dyn_poly_taylor,
)
from pyloki.utils import transforms
from pyloki.utils.misc import C_VAL
from pyloki.utils.world_tree import WorldTree

# Library defects found while writing these tests. Each is pinned with a strict xfail
# so that the fix, when it lands, turns the pin into a failure and forces its removal.
REPORT_ISSUE = "#34: the fixed-grid report discards its re-centring transform"

NBINS = 32
COUNTS = np.array([4, 8])
SCORE_WIDTHS = np.array([1, 2, 4])
T_INIT, T_ADD, T_CUR = 10.0, 30.0, 18.3719


def _param_arr(order: int) -> typed.List:
    higher = [np.array([0.0])] * (order - 2)
    return typed.List([*higher, np.array([-1.0, 1.0]), np.array([99.5, 100.5])])


def _limits(order: int) -> np.ndarray:
    higher = [[-1e-3, 1e-3]] * (order - 2)
    return np.array([*higher, [-20.0, 20.0], [99.0, 101.0]])


def _poly_funcs(init: object, order: int, *, moving: bool) -> object:
    return init(
        _param_arr(order),
        np.array([1e-4] * (order - 2) + [0.05, 0.01]),
        COUNTS,
        10.0,
        NBINS,
        1.0,
        _limits(order),
        128,
        SCORE_WIDTHS,
        order,
        64,
        "aggressive",
        moving,
    )


def _taylor(*, moving: bool, complex_: bool = False) -> object:
    init = (
        dyn_poly_taylor.prune_poly_taylor_complex_dp_functs_init
        if complex_
        else dyn_poly_taylor.prune_poly_taylor_dp_functs_init
    )
    return _poly_funcs(init, 3, moving=moving)


def _cheby(*, moving: bool, complex_: bool = False) -> object:
    init = (
        dyn_poly_cheby.prune_chebyshev_complex_dp_functs_init
        if complex_
        else dyn_poly_cheby.prune_chebyshev_dp_functs_init
    )
    return _poly_funcs(init, 3, moving=moving)


CIRC_EXTRA = (100.0, 1e12, 2.0, 5.0)  # p_orb_min, x_mass_const, prop. and valid. sig.


def _circ(*, moving: bool, complex_: bool = False) -> object:
    init = (
        dyn_circular_taylor.prune_circ_taylor_complex_dp_functs_init
        if complex_
        else dyn_circular_taylor.prune_circ_taylor_dp_functs_init
    )
    return init(
        _param_arr(5),
        np.array([1e-9, 1e-6, 1e-4, 0.05, 0.01]),
        COUNTS,
        10.0,
        NBINS,
        1.0,
        _limits(5),
        128,
        SCORE_WIDTHS,
        5,
        64,
        "aggressive",
        *CIRC_EXTRA,
        moving,
    )


def _taylor_leaves(order: int) -> np.ndarray:
    rng = np.random.default_rng(order)
    leaves = np.zeros((3, order + 2, 2))
    leaves[:, order - 2, 0] = rng.uniform(-5, 5, 3)  # accel
    leaves[:, order - 1, 0] = rng.uniform(-3e4, 3e4, 3)  # velocity
    leaves[:, :order, 1] = [*([1e-6] * (order - 2)), 0.05, 10.0]
    leaves[:, -1, 0] = [99.8, 100.1, 100.4]
    return leaves


def _cheby_leaves() -> np.ndarray:
    leaves = _taylor_leaves(3)
    leaves[:, :-1] = transforms.taylor_to_cheby_full(leaves[:, :-1], 5.0)
    return leaves


class TestTaylorRouting:
    @pytest.mark.parametrize("moving", [True, False])
    def test_resolve(self, moving: bool) -> None:
        leaves = _taylor_leaves(3)
        got = _taylor(moving=moving).resolve(
            leaves, (T_ADD, 1.0), (T_CUR, 1.0), (T_INIT, 1.0)
        )
        if moving:
            want = taylor.poly_taylor_resolve_batch(
                leaves,
                (T_ADD, 1.0),
                (T_CUR, 1.0),
                (T_INIT, 1.0),
                COUNTS,
                _limits(3),
                NBINS,
            )
        else:
            want = taylor.poly_taylor_fixed_resolve_batch(
                leaves,
                (T_ADD, 1.0),
                (T_INIT, 1.0),
                COUNTS,
                _limits(3),
                NBINS,
            )
        for g, w in zip(got, want, strict=True):
            np.testing.assert_array_equal(g, w)

    @pytest.mark.parametrize("moving", [True, False])
    def test_transform(self, moving: bool) -> None:
        leaves = _taylor_leaves(3)
        got = _taylor(moving=moving).transform(leaves, (T_ADD, 1.0), (T_CUR, 1.0))
        if moving:
            want = taylor.poly_taylor_transform_batch(
                leaves,
                (T_ADD, 1.0),
                (T_CUR, 1.0),
                "aggressive",
            )
            np.testing.assert_array_equal(got, want)
        else:
            np.testing.assert_array_equal(got, leaves)

    def test_branch_and_validate(self) -> None:
        funcs = _taylor(moving=True)
        leaves = _taylor_leaves(3)
        got, origins = funcs.branch(leaves, (20.0, 12.0), (18.0, 6.0))
        want, want_origins = taylor.poly_taylor_branch_batch(
            leaves, (20.0, 12.0), NBINS, 1.0, 3, 64
        )
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(origins, want_origins)
        kept, kept_origins = funcs.validate(got, origins, (20.0, 12.0))
        np.testing.assert_array_equal(kept, got)
        np.testing.assert_array_equal(kept_origins, origins)

    def test_report_moving(self) -> None:
        leaves = _taylor_leaves(3)
        got = _taylor(moving=True).report(leaves.copy(), (T_ADD, 20.0), (T_INIT, 5.0))
        np.testing.assert_array_equal(got, taylor.poly_taylor_report_batch(leaves))

    @pytest.mark.xfail(strict=True, reason=REPORT_ISSUE, raises=AssertionError)
    def test_report_fixed_grid_recentres(self) -> None:
        """Pin the fixed-grid report discarding its re-centring transform.

        `report_func` calls `poly_taylor_transform_batch(leaves, coord_report,
        coord_end, ...)` and ignores the result: that kernel returns a new array and
        leaves its input alone. On a fixed grid the leaves are expanded about
        ``t_init``, so the reported frequency is ``f(t_init)`` rather than ``f`` at the
        report time: 100.0 instead of ``100 (1 - 5 * 20 / c)`` for the leaf below.
        """
        leaf = np.zeros((1, 5, 2))
        leaf[0, 1, 0] = 5.0
        leaf[0, :4, 1] = [1e-6, 0.05, 10.0, 0.0]
        leaf[0, -1, 0] = 100.0
        got = _taylor(moving=False).report(leaf.copy(), (T_ADD, 20.0), (T_INIT, 5.0))
        recentred = taylor.poly_taylor_transform_batch(
            leaf,
            (T_ADD, 20.0),
            (T_INIT, 5.0),
            "aggressive",
        )
        np.testing.assert_allclose(got, taylor.poly_taylor_report_batch(recentred))
        assert got[0, 2, 0] == pytest.approx(
            100.0 * (1 - 5.0 * (T_ADD - T_INIT) / C_VAL)
        )


class TestChebyshevRouting:
    @pytest.mark.parametrize("moving", [True, False])
    def test_resolve(self, moving: bool) -> None:
        leaves = _cheby_leaves()
        got = _cheby(moving=moving).resolve(
            leaves, (T_ADD, 1.0), (T_CUR, 5.0), (T_INIT, 1.0)
        )
        if moving:
            want = chebyshev.poly_chebyshev_resolve_batch(
                leaves,
                (T_ADD, 1.0),
                (T_CUR, 5.0),
                (T_INIT, 1.0),
                COUNTS,
                _limits(3),
                NBINS,
            )
        else:
            want = chebyshev.poly_chebyshev_fixed_resolve_batch(
                leaves,
                (T_ADD, 1.0),
                (T_CUR, 5.0),
                (T_INIT, 1.0),
                COUNTS,
                _limits(3),
                NBINS,
            )
        for g, w in zip(got, want, strict=True):
            np.testing.assert_array_equal(g, w)

    @pytest.mark.parametrize("moving", [True, False])
    def test_transform(self, moving: bool) -> None:
        leaves = _cheby_leaves()
        got = _cheby(moving=moving).transform(leaves, (T_ADD, 7.0), (T_CUR, 5.0))
        if moving:
            want = chebyshev.poly_chebyshev_transform_batch(
                leaves,
                (T_ADD, 7.0),
                (T_CUR, 5.0),
                "aggressive",
            )
            np.testing.assert_array_equal(got, want)
        else:
            np.testing.assert_array_equal(got, leaves)

    def test_branch_uses_both_coordinates(self) -> None:
        leaves = _cheby_leaves()
        got, origins = _cheby(moving=True).branch(leaves, (20.0, 12.0), (18.0, 6.0))
        want, want_origins = chebyshev.poly_chebyshev_branch_batch(
            leaves,
            (20.0, 12.0),
            (18.0, 6.0),
            NBINS,
            1.0,
            3,
            64,
            "aggressive",
        )
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(origins, want_origins)

    def test_report_moving(self) -> None:
        leaves = _cheby_leaves()
        got = _cheby(moving=True).report(leaves.copy(), (T_ADD, 5.0), (T_INIT, 5.0))
        np.testing.assert_array_equal(
            got,
            chebyshev.poly_chebyshev_report_batch(leaves, (T_ADD, 5.0)),
        )

    @pytest.mark.xfail(strict=True, reason=REPORT_ISSUE, raises=AssertionError)
    def test_report_fixed_grid_recentres(self) -> None:
        """Pin the same discarded re-centring transform as `dyn_poly_taylor`."""
        leaves = _cheby_leaves()
        got = _cheby(moving=False).report(leaves.copy(), (T_ADD, 7.0), (T_INIT, 5.0))
        recentred = chebyshev.poly_chebyshev_transform_batch(
            leaves,
            (T_ADD, 7.0),
            (T_INIT, 5.0),
            "aggressive",
        )
        np.testing.assert_allclose(
            got,
            chebyshev.poly_chebyshev_report_batch(recentred, (T_ADD, 7.0)),
        )


class TestCircularRouting:
    @staticmethod
    def _leaves() -> np.ndarray:
        return _taylor_leaves(5)

    @pytest.mark.parametrize("moving", [True, False])
    def test_resolve(self, moving: bool) -> None:
        leaves = self._leaves()
        got = _circ(moving=moving).resolve(
            leaves.copy(), (T_ADD, 1.0), (T_CUR, 1.0), (T_INIT, 1.0)
        )
        fn = (
            circular.circ_taylor_resolve_batch
            if moving
            else circular.circ_taylor_fixed_resolve_batch
        )
        want = fn(
            leaves.copy(),
            (T_ADD, 1.0),
            (T_CUR, 1.0),
            (T_INIT, 1.0),
            COUNTS,
            _limits(5),
            NBINS,
            2.0,
        )
        for g, w in zip(got, want, strict=True):
            np.testing.assert_array_equal(g, w)

    @pytest.mark.parametrize("moving", [True, False])
    def test_transform(self, moving: bool) -> None:
        leaves = self._leaves()
        got = _circ(moving=moving).transform(leaves, (T_ADD, 1.0), (T_CUR, 1.0))
        if moving:
            want = circular.circ_taylor_transform_batch(
                leaves,
                (T_ADD, 1.0),
                (T_CUR, 1.0),
                "aggressive",
                2.0,
            )
            np.testing.assert_array_equal(got, want)
        else:
            np.testing.assert_array_equal(got, leaves)

    def test_branch_and_validate(self) -> None:
        funcs = _circ(moving=True)
        leaves = self._leaves()
        got, origins = funcs.branch(leaves, (20.0, 12.0), (18.0, 6.0))
        want, want_origins = circular.circ_taylor_branch_batch(
            leaves,
            (20.0, 12.0),
            NBINS,
            1.0,
            5,
            64,
            2.0,
        )
        # circ_taylor_branch_batch builds its output with np.empty and never writes
        # the d0 error slot [:, -2, 1], so that slot holds whatever was in memory
        # and two calls need not agree there. Compare every slot it writes.
        written = np.ones(got.shape, dtype=bool)
        written[:, -2, 1] = False
        np.testing.assert_array_equal(got[written], want[written])
        np.testing.assert_array_equal(origins, want_origins)
        kept, kept_origins = funcs.validate(got, origins, (20.0, 12.0))
        want_kept, want_kept_origins = circular.circ_taylor_validate_batch(
            got,
            origins,
            CIRC_EXTRA[0],
            CIRC_EXTRA[1],
            CIRC_EXTRA[3],
        )
        np.testing.assert_array_equal(kept, want_kept)
        np.testing.assert_array_equal(kept_origins, want_kept_origins)

    def test_validate_uses_the_validation_significance(self) -> None:
        """A 3-sigma unphysical snap is kept at 5 sigma and rejected at 2 sigma."""
        leaves = np.zeros((2, 7, 2))
        leaves[:, 1] = [3.0, 1.0]  # snap at 3 sigma
        leaves[:, 3] = [3.0, 1.0]  # accel of the same sign: -snap * accel < 0
        leaves[1, 1] = [30.0, 1.0]  # 30-sigma snap and accel: rejected at either
        leaves[1, 3] = [30.0, 1.0]
        origins = np.array([0, 1])
        kept, _ = _circ(moving=True).validate(leaves, origins, (20.0, 12.0))
        np.testing.assert_array_equal(kept, leaves[:1])
        strict, _ = circular.circ_taylor_validate_batch(
            leaves,
            origins,
            CIRC_EXTRA[0],
            CIRC_EXTRA[1],
            CIRC_EXTRA[2],
        )
        assert len(strict) == 0  # the propagator's 2 sigma would reject both

    @pytest.mark.xfail(strict=True, reason=REPORT_ISSUE, raises=AssertionError)
    def test_report_fixed_grid_recentres(self) -> None:
        """Pin the same discarded re-centring transform as `dyn_poly_taylor`."""
        leaves = self._leaves()
        got = _circ(moving=False).report(leaves.copy(), (T_ADD, 20.0), (T_INIT, 5.0))
        recentred = taylor.poly_taylor_transform_batch(
            leaves,
            (T_ADD, 20.0),
            (T_INIT, 5.0),
            "aggressive",
        )
        np.testing.assert_allclose(got, taylor.poly_taylor_report_batch(recentred))


PRUNE_BUNDLES = [
    pytest.param(_taylor, taylor.poly_taylor_seed, 3, id="taylor"),
    pytest.param(_cheby, chebyshev.poly_chebyshev_seed, 3, id="chebyshev"),
    pytest.param(_circ, None, 5, id="circular"),
]


class TestSharedOperations:
    @staticmethod
    def _folds(n: int, *, complex_: bool) -> np.ndarray:
        rng = np.random.default_rng(n)
        real = rng.standard_normal((n, 2, NBINS)).astype(np.float32)
        # A broad pulse, so that the widest scoring width decides the score.
        real[:, 0, 5:9] += 3.0
        real[:, 1] = np.abs(real[:, 1]) + 1.0
        if not complex_:
            return real
        return np.ascontiguousarray(np.fft.rfft(real, axis=-1).astype(np.complex64))

    @pytest.mark.parametrize(("make", "_seed", "_order"), PRUNE_BUNDLES)
    @pytest.mark.parametrize("complex_", [False, True])
    def test_score_and_shift_add(
        self, make, _seed, _order: int, complex_: bool
    ) -> None:
        funcs = make(moving=True, complex_=complex_)
        folds = self._folds(6, complex_=complex_)
        score_fn = (
            scoring.snr_score_batch_func_complex
            if complex_
            else scoring.snr_score_batch_func
        )
        np.testing.assert_array_equal(funcs.score(folds), score_fn(folds, SCORE_WIDTHS))
        segs = self._folds(4, complex_=complex_)
        shifts = np.array([0.0, 3.4, 17.5, 31.6])
        isuggest = np.array([0, 5, 2, 2])
        add_fn = common.shift_add_complex_batch if complex_ else common.shift_add_batch
        np.testing.assert_array_equal(
            funcs.shift_add(segs, shifts, folds, isuggest),
            add_fn(segs, shifts, folds, isuggest),
        )

    @pytest.mark.parametrize(("make", "seed_fn", "order"), PRUNE_BUNDLES[:2])
    def test_seed(self, make, seed_fn, order: int) -> None:
        funcs = make(moving=True)
        leaves = seed_fn(
            _param_arr(order), np.array([1e-4, 0.05, 0.01]), order, (5.0, 5.0)
        )
        fold_seg = self._folds(len(leaves), complex_=False).reshape(
            1, 2, len(leaves) // 2, 2, NBINS
        )
        tree = funcs.seed(fold_seg, (5.0, 5.0))
        np.testing.assert_array_equal(tree.leaves, leaves)
        np.testing.assert_array_equal(tree.scores, funcs.score(tree.folds))

    @pytest.mark.parametrize(("make", "_seed", "_order"), PRUNE_BUNDLES)
    def test_load_and_pack(self, make, _seed, _order: int) -> None:
        funcs = make(moving=True)
        data = self._folds(5, complex_=False)
        np.testing.assert_array_equal(funcs.load(data, 3), data[3])
        np.testing.assert_array_equal(funcs.pack(data), data)


class TestAscend:
    def test_batch_size_does_not_change_the_result(self) -> None:
        """Rescoring a tree: scores move to scores_ep; folds and scores are rebuilt."""
        funcs = _taylor(moving=False)
        rng = np.random.default_rng(9)
        n = 5
        leaves = _taylor_leaves(3)
        leaves = np.concatenate([leaves, leaves[:2]])
        dyp = rng.standard_normal((4, 4, 8, 2, NBINS)).astype(np.float32)
        idx_segments = np.array([0, 2, 3])
        coord_segments = np.array([[5.0, 5.0], [25.0, 5.0], [35.0, 5.0]])

        def run(batch_size: int) -> WorldTree:
            tree = WorldTree(
                leaves.copy(),
                np.zeros((n, 2, NBINS), np.float32),
                rng.standard_normal(n).astype(np.float32),
                np.zeros((n, 5), np.int32),
            )
            before = tree.scores.copy()
            funcs.ascend(
                tree,
                dyp,
                common.load_prune_folds_2d,
                idx_segments,
                coord_segments,
                (T_INIT, 5.0),
                batch_size,
            )
            np.testing.assert_array_equal(tree.scores_ep, before)
            return tree

        one, many = run(1), run(n)
        np.testing.assert_allclose(one.folds, many.folds, rtol=1e-6)
        np.testing.assert_allclose(one.scores, many.scores, rtol=1e-6)
        idx, phase = taylor.poly_taylor_ascend_resolve_batch(
            leaves,
            coord_segments,
            (T_INIT, 5.0),
            COUNTS,
            _limits(3),
            NBINS,
        )
        combined = common.shift_add_ascend_batch(
            dyp,
            common.load_prune_folds_2d,
            idx_segments,
            idx,
            phase,
        )
        np.testing.assert_allclose(many.folds, combined, rtol=1e-6)
        np.testing.assert_allclose(
            many.scores,
            scoring.snr_score_batch_func(combined, SCORE_WIDTHS),
            rtol=1e-6,
        )


class TestFFARouting:
    TSAMP, BSEG = 1e-3, 64

    def _funcs(self, *, complex_: bool = False) -> object:
        init = (
            dyn_pffa.ffa_taylor_complex_dp_functs_init
            if complex_
            else dyn_pffa.ffa_taylor_dp_functs_init
        )
        return init(_limits(2), self.TSAMP, NBINS, self.BSEG)

    @pytest.mark.parametrize("complex_", [False, True])
    def test_init_folds_the_series(self, complex_: bool) -> None:
        rng = np.random.default_rng(1)
        ts_e = rng.standard_normal(4 * self.BSEG).astype(np.float32)
        ts_v = np.ones(4 * self.BSEG, np.float32)
        param_arr = typed.List([np.array([0.0]), np.array([99.5, 100.5])])
        got = self._funcs(complex_=complex_).init(ts_e, ts_v, param_arr)
        fn = fold.ffa_taylor_init_complex if complex_ else fold.ffa_taylor_init
        np.testing.assert_array_equal(
            got, fn(ts_e, ts_v, param_arr, self.BSEG, NBINS, self.TSAMP)
        )

    @pytest.mark.parametrize("complex_", [False, True])
    @pytest.mark.parametrize("latter", [0, 1])
    def test_resolve_uses_the_brute_segment_duration(
        self, complex_: bool, latter: int
    ) -> None:
        pset = np.array([3.0, 100.2])
        got = self._funcs(complex_=complex_).resolve(pset, COUNTS, 3, latter)
        want = fold.ffa_taylor_resolve(
            pset,
            COUNTS,
            _limits(2),
            3,
            latter,
            self.BSEG * self.TSAMP,
            NBINS,
        )
        np.testing.assert_array_equal(got[0], want[0])
        assert got[1] == want[1]

    def test_shift_add_and_pack(self) -> None:
        funcs = self._funcs()
        rng = np.random.default_rng(2)
        tail, head = rng.standard_normal((2, 2, NBINS)).astype(np.float32)
        np.testing.assert_array_equal(
            funcs.shift_add(tail, head, 3.4, 17.6),
            common.shift_add(tail, head, 3.4, 17.6),
        )
        np.testing.assert_array_equal(funcs.pack(tail, 2), tail)

    def test_complex_shift_add(self) -> None:
        funcs = self._funcs(complex_=True)
        rng = np.random.default_rng(3)
        tail, head = (
            np.ascontiguousarray(np.fft.rfft(a, axis=-1).astype(np.complex64))
            for a in rng.standard_normal((2, 2, NBINS)).astype(np.float32)
        )
        np.testing.assert_array_equal(
            funcs.shift_add(tail, head, 3.4, 17.6),
            common.shift_add_complex(tail, head, 3.4, 17.6),
        )
