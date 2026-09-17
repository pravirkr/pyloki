"""Helpers for unit-testing numba kernels so that coverage can see them.

A ``@njit`` kernel is executed as machine code, so ``coverage`` never records a line
inside its body. Measured on this repo, that is not a rounding error: 3384 of the
library's 8309 statements (40.7%) live inside ``@njit``/``@vectorize`` bodies, and the
suite as it stands covers **0** of them. Every kernel is invisible.

Numba keeps the undecorated Python function on the decorated object, and calling it
runs the same source under the interpreter, where coverage does see it:

- ``@njit``/``@jit`` produce a ``Dispatcher`` exposing ``.py_func``;
- ``@vectorize``/``@guvectorize`` produce a ``DUFunc``/ufunc exposing ``.__wrapped__``.

`python_impl` hides that difference; `jit_variants` turns it into a parametrisation.

Why run both paths rather than only the Python one
--------------------------------------------------
The compiled kernel is what ships, and it is not merely a faster copy of the source.
It enforces the declared signature, which the Python source does not: given the
explicit ``@njit(["f4[:,::1](f4[:,::1],...)"])`` on ``core.common.shift_add``, the
compiled kernel rejects a float64 array with ``TypeError``, while ``.py_func`` accepts
it and returns float64. A py_func-only test would pass on input the library refuses.
So a test that asserted only against ``.py_func`` would be testing a program that is
not the one being shipped.

`jit_variants` therefore runs *the same assertions* against both, which buys coverage
of the kernel body without letting the two implementations drift apart unnoticed:
any divergence in behaviour that a test can see fails that test on one path only.
Contracts that belong to the compiled path alone -- signature enforcement, ufunc
broadcasting, dtype coercion -- are asserted directly on the dispatcher instead, since
they are properties of the compilation and not of the source.

The alternative, ``NUMBA_DISABLE_JIT=1``, was measured against this codebase and
rejected; see ``tests/test_psr_utils.py`` for the findings.

Limits worth knowing before reaching for `jit_variants`
-------------------------------------------------------
``.py_func`` un-jits exactly one level. Calls the body makes to *other* jitted
functions still enter the compiled dispatcher, so a kernel whose helper takes a
first-class function argument breaks at that boundary: ``np_utils.nb_max.py_func``
raises ``TypingError``, because it hands ``np.max`` to the still-jitted
``np_apply_along_axis``, and numba cannot type a NumPy builtin arriving from Python.
Such kernels should be tested on the compiled path only, with a comment saying so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


def python_impl(func: Callable[..., Any]) -> Callable[..., Any]:
    """Return the pure-Python function behind a numba-compiled object.

    Parameters
    ----------
    func : Callable[..., Any]
        A ``@njit``/``@jit`` dispatcher or a ``@vectorize``/``@guvectorize`` ufunc.

    Returns
    -------
    Callable[..., Any]
        The undecorated Python implementation.

    Raises
    ------
    TypeError
        If `func` exposes no pure-Python implementation.
    """
    for attr in ("py_func", "__wrapped__"):
        impl = getattr(func, attr, None)
        if callable(impl):
            return impl
    msg = (
        f"{getattr(func, '__name__', func)!r} exposes neither .py_func nor "
        f".__wrapped__; it cannot be exercised under coverage."
    )
    raise TypeError(msg)


def jit_variants(func: Callable[..., Any]) -> list[Any]:
    """Parametrise a test over the compiled kernel and its Python source.

    Use as ``@pytest.mark.parametrize("impl", jit_variants(kernel))``. The test body
    then runs twice, with ids ``compiled`` and ``py_func``, asserting the same
    contract against the code that ships and the code coverage can measure.

    Parameters
    ----------
    func : Callable[..., Any]
        The numba-compiled object to parametrise over.

    Returns
    -------
    list[Any]
        Two ``pytest.param`` entries, compiled first.
    """
    return [
        pytest.param(func, id="compiled"),
        pytest.param(python_impl(func), id="py_func"),
    ]
