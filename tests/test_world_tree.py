"""Unit tests for `pyloki.utils.world_tree`, the candidate buffer of the pruning search.

`WorldTree` (and `WorldTreeComplex`, with complex folds) is a numba ``structref``: a
fixed-capacity buffer of leaves, folds, scores and backtracks, of which the first
``valid_size`` rows are live. Pruning adds survivors to it in batches. When a batch
overflows the buffer, `prune_on_overload` raises the threshold and keeps only the
candidates at or above it. A defect here silently drops or duplicates candidates.

Contracts:

- adding appends rows in order, refuses once full, and keeps every array aligned: after
  any operation each live row's leaf, fold and backtrack are those that arrived with its
  score;
- the overflow threshold has a closed form,
  ``max(current, nextafter(k-th largest), nextafter(median element))`` over the old and
  new scores, where ``k`` is the capacity. The kept rows are exactly the candidates at
  or above it, compared in float64 as the kernel does;
- `get_best`/`get_best_k` against a sort; `trim_empty`/`get_new` for shape and content.

The kernels take the ``structref`` itself, so their ``py_func`` runs against the Python
proxy. That works for all of them except the two ``structref.new`` constructors, which
numba cannot run in pure Python; those are tested compiled-only. Each test runs on both
the float and the complex tree.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numba import njit
from numba.core.errors import TypingError

from pyloki.utils import world_tree as wt
from tests.jit_utils import jit_variants

# Library defects found while writing these tests. Each is pinned with a strict xfail
# so that the fix, when it lands, turns the pin into a failure and forces its removal.

NPARAMS, NBINS = 2, 8

TREES = [
    pytest.param(wt.WorldTree, np.float32, id="float"),
    pytest.param(wt.WorldTreeComplex, np.complex64, id="complex"),
]


def _rows(n: int, dtype: type, seed: int) -> tuple[np.ndarray, ...]:
    """Return n candidates whose scores are distinct, so each row can be traced."""
    rng = np.random.default_rng(seed)
    leaves = rng.standard_normal((n, NPARAMS + 2, 2))
    folds = rng.standard_normal((n, 2, NBINS))
    if dtype is np.complex64:
        folds = folds + 1j * rng.standard_normal((n, 2, NBINS))
    folds = folds.astype(dtype)
    scores = rng.permutation(np.linspace(-3.0, 3.0, n)).astype(np.float32)
    backtracks = rng.integers(0, 1000, (n, NPARAMS + 2)).astype(np.int32)
    return leaves, folds, scores, backtracks


def _tree(
    cls: type,
    dtype: type,
    size: int,
    valid: int,
    seed: int = 0,
) -> wt.WorldTree | wt.WorldTreeComplex:
    tree = cls(*_rows(size, dtype, seed))
    tree.valid_size = valid
    return tree


def _live(tree) -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Map each live score to a copy of its (leaf, fold, backtrack) row.

    Copies, because the kernels move rows in place.
    """
    n = tree.valid_size
    return {
        float(s): (
            tree.leaves[i].copy(),
            tree.folds[i].copy(),
            tree.backtracks[i].copy(),
        )
        for i, s in enumerate(tree.scores[:n])
    }


def _assert_scores_ep_tracks_scores(tree) -> None:
    """Every add writes the score to ``scores_ep`` as well."""
    n = tree.valid_size
    np.testing.assert_array_equal(tree.scores_ep[:n], tree.scores[:n])


def _assert_rows_match(tree, source: dict) -> None:
    for score, (leaf, fold, bt) in _live(tree).items():
        src_leaf, src_fold, src_bt = source[score]
        np.testing.assert_array_equal(leaf, src_leaf)
        np.testing.assert_array_equal(fold, src_fold)
        np.testing.assert_array_equal(bt, src_bt)


@pytest.mark.parametrize(("cls", "dtype"), TREES)
class TestConstructionAndProperties:
    def test_init_fields(self, cls, dtype) -> None:
        leaves, folds, scores, backtracks = _rows(5, dtype, 1)
        tree = cls(leaves, folds, scores, backtracks)
        assert (tree.size, tree.valid_size, tree.nparams) == (5, 5, NPARAMS)
        np.testing.assert_array_equal(tree.scores_ep, scores)
        # Compare memory, not identity: an aliased field would share memory whatever
        # array objects the proxy hands back.
        assert not np.shares_memory(tree.scores_ep, tree.scores)

    def test_summary_properties(self, cls, dtype) -> None:
        tree = _tree(cls, dtype, 10, 4)
        live = tree.scores[:4]
        props = type(tree)
        for get in (lambda p: p.fget, lambda p: p.fget.py_func):
            assert get(props.size_lb)(tree) == pytest.approx(2.0)
            assert get(props.score_max)(tree) == live.max()
            assert get(props.score_min)(tree) == live.min()

    def test_field_getters_on_both_paths(self, cls, dtype) -> None:
        tree = _tree(cls, dtype, 6, 4)
        props = type(tree)
        for name in ("leaves", "folds", "scores", "scores_ep", "backtracks"):
            getter = getattr(props, name).fget
            np.testing.assert_array_equal(getter.py_func(tree), getter(tree))
        for name, value in [("valid_size", 4), ("size", 6), ("nparams", NPARAMS)]:
            getter = getattr(props, name).fget
            assert getter(tree) == getter.py_func(tree) == value
        props.valid_size.fset.py_func(tree, 2)
        assert tree.valid_size == 2

    def test_empty_tree_properties_are_zero(self, cls, dtype) -> None:
        tree = _tree(cls, dtype, 6, 0)
        assert tree.size_lb == 0.0
        assert tree.score_max == 0.0
        assert tree.score_min == 0.0


@pytest.mark.parametrize(("cls", "dtype"), TREES)
class TestAdd:
    @pytest.mark.parametrize("impl", jit_variants(wt.add_func))
    def test_appends_until_full(self, cls, dtype, impl) -> None:
        tree = _tree(cls, dtype, 4, 2)
        leaves, folds, scores, backtracks = _rows(3, dtype, 7)
        assert impl(tree, leaves[0], folds[0], scores[0], backtracks[0])
        assert impl(tree, leaves[1], folds[1], scores[1], backtracks[1])
        assert tree.valid_size == 4
        assert not impl(tree, leaves[2], folds[2], scores[2], backtracks[2])
        assert tree.valid_size == 4
        np.testing.assert_array_equal(tree.scores[2:4], scores[:2])
        np.testing.assert_array_equal(tree.scores_ep[2:4], scores[:2])
        np.testing.assert_array_equal(tree.folds[3], folds[1])

    @pytest.mark.parametrize("impl", jit_variants(wt.add_batch_func))
    def test_batch_fast_path(self, cls, dtype, impl) -> None:
        tree = _tree(cls, dtype, 10, 3)
        batch = _rows(5, dtype, 8)
        before = _live(tree)
        assert impl(tree, *batch, -1.5) == -1.5
        assert tree.valid_size == 8
        _assert_scores_ep_tracks_scores(tree)
        source = before | {
            float(s): (batch[0][i], batch[1][i], batch[3][i])
            for i, s in enumerate(batch[2])
        }
        _assert_rows_match(tree, source)

    @pytest.mark.parametrize("impl", jit_variants(wt.add_batch_func))
    def test_batch_that_exactly_fills_takes_the_fast_path(
        self, cls, dtype, impl
    ) -> None:
        """A batch equal to the space left is added whole, with no pruning."""
        tree = _tree(cls, dtype, 8, 3)
        batch = _rows(5, dtype, 13)
        before = set(_live(tree))
        assert impl(tree, *batch, -9.0) == -9.0
        assert tree.valid_size == 8
        assert set(_live(tree)) == before | {float(s) for s in batch[2]}
        _assert_scores_ep_tracks_scores(tree)

    @pytest.mark.parametrize("impl", jit_variants(wt.add_batch_func))
    def test_empty_batch_is_a_no_op(self, cls, dtype, impl) -> None:
        tree = _tree(cls, dtype, 5, 2)
        empty = tuple(a[:0] for a in _rows(1, dtype, 9))
        assert impl(tree, *empty, 0.25) == 0.25
        assert tree.valid_size == 2


@pytest.mark.parametrize(("cls", "dtype"), TREES)
class TestOverflow:
    SIZE = 8

    @staticmethod
    def _expected_threshold(scores: np.ndarray, size: int, current: float) -> float:
        ordered = np.sort(scores.astype(np.float64))
        total = len(ordered)
        kth = ordered[-size]
        mid = ordered[-(total // 2) - 1]
        return max(
            current,
            math.nextafter(kth, math.inf),
            math.nextafter(mid, math.inf),
        )

    @pytest.mark.parametrize("impl", jit_variants(wt.add_batch_func))
    @pytest.mark.parametrize(("valid", "n_batch"), [(6, 10), (8, 3), (2, 30)])
    @pytest.mark.parametrize("current", [-np.inf, 0.5])
    def test_keeps_exactly_the_candidates_above_threshold(
        self,
        cls,
        dtype,
        impl,
        valid: int,
        n_batch: int,
        current: float,
    ) -> None:
        tree = _tree(cls, dtype, self.SIZE, valid, seed=valid)
        batch = _rows(n_batch, dtype, seed=100 + n_batch)
        # Keep old and new scores disjoint so every row can be traced.
        batch[2][:] += 10.0 * (np.arange(n_batch) % 2) + 0.001
        source = _live(tree) | {
            float(s): (batch[0][i], batch[1][i], batch[3][i])
            for i, s in enumerate(batch[2])
        }
        all_scores = np.array(list(source))
        threshold = impl(tree, *batch, current)

        assert threshold == self._expected_threshold(all_scores, self.SIZE, current)
        assert tree.valid_size <= self.SIZE
        # The threshold is a Python float just above a float32 score. Compiled, numba
        # compares the float32 scores with it in float64, so that score is dropped,
        # and the kept set is exact. The Python source differs at that one boundary
        # value: NumPy compares a float32 array with a Python float in float32, which
        # rounds the threshold back onto that score, while the existing rows are
        # filtered by the compiled `prune_on_overload` it calls through the proxy. So
        # py_func is held to the bounds that hold either way.
        kept = np.sort(tree.scores[: tree.valid_size].astype(np.float64))
        above = np.sort(all_scores[all_scores >= threshold])
        if impl is wt.add_batch_func:
            np.testing.assert_array_equal(kept, above)
        else:
            assert set(above) <= set(kept)
            assert (kept.astype(np.float32) >= np.float32(threshold)).all()
        _assert_scores_ep_tracks_scores(tree)
        _assert_rows_match(tree, source)

    @pytest.mark.parametrize("impl", jit_variants(wt.add_batch_func))
    def test_overflow_where_no_newcomer_clears_the_threshold(
        self,
        cls,
        dtype,
        impl,
    ) -> None:
        tree = _tree(cls, dtype, self.SIZE, self.SIZE)
        before = _live(tree)
        batch = _rows(4, dtype, seed=12)
        batch[2][:] = -100.0 - np.arange(4, dtype=np.float32)
        threshold = impl(tree, *batch, -np.inf)
        assert threshold > -100.0
        assert set(_live(tree)) <= set(before)
        _assert_rows_match(tree, before)

    @pytest.mark.parametrize("impl", jit_variants(wt.prune_on_overload_func))
    def test_prune_on_overload_keeps_only_existing_rows(self, cls, dtype, impl) -> None:
        """It only removes rows; adding the batch is `add_batch`'s job."""
        tree = _tree(cls, dtype, self.SIZE, self.SIZE)
        before = _live(tree)
        incoming = np.linspace(-2.0, 2.0, 5).astype(np.float32)
        threshold = impl(tree, incoming, -np.inf)
        assert all(s >= threshold for s in _live(tree))
        assert set(_live(tree)) == {s for s in before if s >= threshold}
        _assert_rows_match(tree, before)

    @pytest.mark.parametrize("impl", jit_variants(wt.prune_on_overload_func))
    def test_prune_on_overload_empty_batch(self, cls, dtype, impl) -> None:
        tree = _tree(cls, dtype, self.SIZE, 5)
        assert impl(tree, np.zeros(0, np.float32), 1.25) == 1.25
        assert tree.valid_size == 5


@pytest.mark.parametrize(("cls", "dtype"), TREES)
class TestSelection:
    @pytest.mark.parametrize("impl", jit_variants(wt.get_best_func))
    def test_get_best(self, cls, dtype, impl) -> None:
        tree = _tree(cls, dtype, 10, 6)
        leaf, fold, score = impl(tree)
        i = int(np.argmax(tree.scores[:6]))
        assert score == tree.scores[:6].max()
        np.testing.assert_array_equal(leaf, tree.leaves[i])
        np.testing.assert_array_equal(fold, tree.folds[i])

    @pytest.mark.parametrize("impl", jit_variants(wt.get_best_k_func))
    @pytest.mark.parametrize("k", [0, 1, 3, 6, 9])
    def test_get_best_k(self, cls, dtype, impl, k: int) -> None:
        """Descending by score, zero-padded when ``k`` exceeds the live rows."""
        tree = _tree(cls, dtype, 10, 6)
        leaves, scores = impl(tree, k)
        assert leaves.shape == (k, NPARAMS + 2, 2)
        order = np.argsort(tree.scores[:6])[::-1][: min(k, 6)]
        np.testing.assert_array_equal(scores[: len(order)], tree.scores[order])
        np.testing.assert_array_equal(leaves[: len(order)], tree.leaves[order])
        assert not scores[len(order) :].any()
        assert not leaves[len(order) :].any()

    @pytest.mark.parametrize("impl", jit_variants(wt.keep_func))
    def test_keep_moves_rows_to_the_front(self, cls, dtype, impl) -> None:
        tree = _tree(cls, dtype, 8, 6)
        # Make scores_ep distinct from scores so the test can see it move with its row.
        tree.scores_ep[:] = -np.arange(8, dtype=np.float32)
        before = _live(tree)
        mask = np.array([True, False, False, True, True, False])
        chosen = tree.scores[:6][mask].copy()
        chosen_ep = tree.scores_ep[:6][mask].copy()
        impl(tree, mask)
        assert tree.valid_size == 3
        np.testing.assert_array_equal(tree.scores[:3], chosen)
        np.testing.assert_array_equal(tree.scores_ep[:3], chosen_ep)
        _assert_rows_match(tree, before)

    @pytest.mark.parametrize("impl", jit_variants(wt.keep_func))
    def test_keep_nothing(self, cls, dtype, impl) -> None:
        tree = _tree(cls, dtype, 8, 6)
        impl(tree, np.zeros(6, dtype=np.bool_))
        assert tree.valid_size == 0


class TestNewAndTrim:
    @pytest.mark.parametrize(
        ("impl", "cls", "dtype"),
        [(p.values[0], wt.WorldTree, np.float32) for p in jit_variants(wt.get_new_func)]
        + [
            (p.values[0], wt.WorldTreeComplex, np.complex64)
            for p in jit_variants(wt.get_new_func_complex)
        ],
    )
    def test_get_new_is_empty_with_same_row_shape(self, impl, cls, dtype) -> None:
        tree = _tree(cls, dtype, 5, 5)
        new = impl(tree, 12)
        assert isinstance(new, cls)
        assert (new.size, new.valid_size, new.nparams) == (12, 0, NPARAMS)
        assert new.folds.dtype == dtype
        assert new.leaves.shape[1:] == tree.leaves.shape[1:]

    @pytest.mark.parametrize(
        ("impl", "cls", "dtype"),
        [
            (p.values[0], wt.WorldTree, np.float32)
            for p in jit_variants(wt.trim_empty_func)
        ]
        + [
            (p.values[0], wt.WorldTreeComplex, np.complex64)
            for p in jit_variants(wt.trim_empty_func_complex)
        ],
    )
    def test_trim_empty_keeps_the_live_rows(self, impl, cls, dtype) -> None:
        tree = _tree(cls, dtype, 9, 4)
        trimmed = impl(tree)
        assert (trimmed.size, trimmed.valid_size) == (4, 4)
        np.testing.assert_array_equal(trimmed.scores, tree.scores[:4])
        np.testing.assert_array_equal(trimmed.folds, tree.folds[:4])

    @pytest.mark.parametrize(("cls", "dtype"), TREES)
    def test_methods_route_to_the_kernels(self, cls, dtype) -> None:
        """The proxy methods are thin wrappers; one call each, for the wiring."""
        tree = _tree(cls, dtype, 8, 3)
        assert tree.get_new(4).size == 4
        assert tree.trim_empty().size == 3
        assert tree.get_best()[2] == tree.scores[:3].max()
        assert tree.get_best_k(2)[1][0] == tree.scores[:3].max()
        leaves, folds, scores, backtracks = _rows(1, dtype, 11)
        assert tree.add(leaves[0], folds[0], scores[0], backtracks[0])
        assert tree.add_batch(leaves, folds, scores, backtracks, 0.0) == 0.0
        tree._keep(np.ones(tree.valid_size, dtype=np.bool_))  # noqa: SLF001
        assert tree.prune_on_overload(np.zeros(0, np.float32), 0.5) == 0.5


@njit(cache=False)
def _jit_methods(tree, leaves, folds, scores, backtracks):  # noqa: ANN202
    """Call each method from compiled code, which dispatches to the @overload_method."""
    new = tree.get_new(4)
    best = tree.get_best()
    best_k = tree.get_best_k(2)
    added = tree.add(leaves[0], folds[0], scores[0], backtracks[0])
    threshold = tree.add_batch(leaves, folds, scores, backtracks, -1e30)
    pruned = tree.prune_on_overload(scores[:0], 0.5)
    tree._keep(np.ones(tree.valid_size, dtype=np.bool_))  # noqa: SLF001
    trimmed = tree.trim_empty()
    return new.size, best[2], best_k[1][0], added, threshold, pruned, trimmed.size


class TestCompiledMethodDispatch:
    """The search calls these methods from compiled code; check that path forwards."""

    @pytest.mark.parametrize(("cls", "dtype"), TREES)
    def test_methods_from_compiled_code(self, cls, dtype) -> None:
        tree = _tree(cls, dtype, 12, 3)
        expected_best = tree.scores[:3].max()
        leaves, folds, scores, backtracks = _rows(2, dtype, 17)
        size_new, best, best_k0, added, threshold, pruned, size_trim = _jit_methods(
            tree,
            leaves,
            folds,
            scores,
            backtracks,
        )
        assert size_new == 4
        assert best == best_k0 == expected_best
        assert added
        assert threshold == -1e30
        assert pruned == 0.5
        assert tree.valid_size == size_trim == 6


class TestUniqueIndices:
    @pytest.mark.parametrize("impl", jit_variants(wt.get_unique_indices))
    def test_first_occurrence_of_each_frequency(self, impl) -> None:
        """Keys on the last row's value (``f0``) to 1e-9."""
        leaves = np.zeros((5, 4, 2))
        leaves[:, -1, 0] = [100.0, 101.0, 100.0 + 1e-12, 102.0, 101.0]
        np.testing.assert_array_equal(impl(leaves), [0, 1, 3])

    @staticmethod
    def _leaves(pairs: list[tuple[float, float]]) -> np.ndarray:
        """Leaves whose last two rows, ``d0`` and ``f0``, hold the given pairs."""
        leaves = np.zeros((len(pairs), 3, 2))
        for i, (d0, f0) in enumerate(pairs):
            leaves[i, -2, 0] = d0
            leaves[i, -1, 0] = f0
        return leaves

    @pytest.mark.parametrize("impl", jit_variants(wt.get_unique_indices_scores))
    def test_scores_without_repeats(self, impl) -> None:
        leaves = self._leaves([(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)])
        scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(impl(leaves, scores), [0, 1, 2])

    @pytest.mark.xfail(strict=True, reason="#51", raises=AssertionError)
    @pytest.mark.parametrize("impl", jit_variants(wt.get_unique_indices_scores))
    def test_scores_keep_best_of_each_repeat(self, impl) -> None:
        """Pin the running count being reset by a better-scoring repeat.

        On a repeat with a higher score the kernel does ``count = count_dict[key]``,
        which rewinds the output position, so later unique entries overwrite earlier
        ones: ``[A, B, A (better), C]`` returns ``[3]`` instead of ``[2, 1, 3]``.
        """
        leaves = self._leaves([(1.0, 10.0), (2.0, 20.0), (1.0, 10.0), (3.0, 30.0)])
        scores = np.array([1.0, 2.0, 5.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(np.sort(impl(leaves, scores)), [1, 2, 3])

    @pytest.mark.xfail(strict=True, reason="#51", raises=AssertionError)
    @pytest.mark.parametrize("impl", jit_variants(wt.get_unique_indices_scores))
    def test_scores_distinct_leaves_are_distinct(self, impl) -> None:
        """Pin the key being the sum of rows ``d0`` and ``f0``.

        ``(1, 10)`` and ``(2, 9)`` share the key 11e9, so they count as repeats: with
        scores ``[2, 1]`` the result is ``[0]`` (the collision alone), and with
        ``[1, 2]`` it is empty (collision plus the reset above).
        """
        leaves = self._leaves([(1.0, 10.0), (2.0, 9.0)])
        scores = np.array([1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(np.sort(impl(leaves, scores)), [0, 1])

    @pytest.mark.xfail(strict=True, reason="#51", raises=TypingError)
    @pytest.mark.parametrize(("cls", "dtype"), TREES)
    @pytest.mark.parametrize(
        "impl",
        jit_variants(wt.trim_repeats_func)
        + jit_variants(wt.trim_repeats_threshold_func)
        + [
            pytest.param(lambda t: t.trim_repeats(), id="method"),
            pytest.param(lambda t: t.trim_repeats_threshold(), id="method_threshold"),
        ],
    )
    def test_trim_repeats_compiles(self, cls, dtype, impl) -> None:
        """Pin ``trim_repeats*`` passing a 2-D slice where the helper indexes 3-D.

        They hand ``leaves[:n, :nparams, 0]`` (2-D) to `get_unique_indices_scores`,
        which reads ``leaves[ii][-2:, 0]``; numba finds no ``getitem`` for that.
        """
        impl(_tree(cls, dtype, 6, 6))
