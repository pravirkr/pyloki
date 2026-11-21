# PyLoki architecture

This page records the frozen layout of the library and the rules that keep it
that way. This is descriptive guidance, not enforced by import checks in CI.

## Package layout and dependency direction

Modules under [`src/pyloki/`](../src/pyloki) follow a strict layered layout
(lower layers never import higher layers):

- `utils/` (`maths`, `kepler_math`, `transforms`, `psr_utils`, `np_utils`, `snail`,
  `world_tree`, `timing`, `misc`): stateless mathematical helpers, low-level
  Kepler anomaly/delay kernels, Taylor / Chebyshev / circular-Taylor coordinate
  transforms, `MiddleOutScheme` segment traversal, Numba `structref` candidate
  trees (`WorldTree`, `WorldTreeComplex`), and logging, timing, and
  multiprocessing utilities. Must not import from `core`, `dynamic`, or top-level
  search modules.
- `core/` (`fold`, `common`, `taylor`, `chebyshev`, `circular`, `svd_kepler`,
  `epicyclic`): stateless `@njit` kernels for brute-force folding, FFA
  shift-add, tree seeding, branching, validation, resolution, basis re-centring,
  branching-pattern generation, and Keplerian/epicyclic prototype kernels. Must
  not hold search state or import `dynamic` or higher-level modules.
- `detection/` (`scoring`, `thresholding`, `schemes`): boxcar, heteroscedastic
  GLRT, double-peak, harmonic-$Z^2_m$, Kadane, and 2-state Markov HMM S/N
  kernels and dynamic pruning threshold scheme generation and analysis
  (`DynamicThresholdScheme`, `schedule_survive_probs`).
- `dynamic/` (`dyn_pffa`, `dyn_poly_taylor`, `dyn_poly_cheby`,
  `dyn_circular_taylor`): Numba `structref` function bundles (`*DPFuncts`)
  binding `PulsarSearchConfig` parameters to stateless `core` and `detection`
  kernels.
- Top-level and supporting modules: `config.py` (`ParamLimits`,
  `PulsarSearchConfig`), `ffa.py` (`FFAPlan`, `DynamicProgramming`), `prune.py`
  (`Pruning`, `prune_dyp_tree`, `trace_ep_hypothesis`,
  `verify_resolve_stability`, `prune_suggestion`, `MidTreeCheckpoint`), `search.py` (`ffa_search`, `ep_search`,
  `ep_sliding_arc_search`, `refold_candidates`, `measure_snr_retention`,
  `decompose_fold_quality`), `periodogram.py` (`Periodogram`,
  `ScatteredPeriodogram`), `kepler.py`, `orbit_fit.py`, `io/` (`timeseries`,
  `preprocess`, `preprocess_kernels`, `cands`), `simulation/` (`pulse`,
  `modulate`), `sensitivity/` (`sim_ffa`, `analytical`, `heavy_tail`), and
  `apps/` (`pyloki_viewer`, including `WaterfallLabState` / `WaterfallLabApp`).

## Public API

- Keep [`src/pyloki/__init__.py`](../src/pyloki/__init__.py) lightweight
  (`__version__` and root logger setup only — no heavy eager imports or Numba
  compilation at import time).
- Import domain classes and functions from their canonical modules
  (`pyloki.config`, `pyloki.ffa`, `pyloki.prune`, `pyloki.search`,
  `pyloki.periodogram`, `pyloki.io.timeseries`, `pyloki.detection.*`,
  `pyloki.simulation.pulse`), and keep `__all__` accurate in `pyloki.core` and
  `pyloki.dynamic`.
- Preserve existing public signatures and defaults; new optional parameters
  (such as `seed: int | None = None`) must keep default behavior
  backward-compatible.

## Dispatch

- Hot FFA and EP loops never branch on Python configuration objects; search
  configuration is bound once into a Numba `structref` bundle in `pyloki.dynamic`
  and wrapped in `PicklableStructRefWrapper` for `ProcessPoolExecutor` workers.
- `DynamicProgramming` in [`ffa.py`](../src/pyloki/ffa.py) dispatches on
  `cfg.use_fourier` (`FFATaylorComplexDPFuncts` for `complex64` vs
  `FFATaylorDPFuncts` for `float32`) and uses rank-specialized loaders from
  `core.set_ffa_load_func(nparams)`.
- `Pruning._PRUNE_DISPATCH` in [`prune.py`](../src/pyloki/prune.py) dispatches on
  `(prune_poly_order == 5, poly_basis, use_fourier)` to select
  `PruneCircTaylor*DPFuncts`, `PrunePolyTaylor*DPFuncts`, or
  `PrunePolyChebyshev*DPFuncts`; `use_moving_grid` selects moving- vs fixed-grid
  `resolve`, `transform`, and `report` kernels inside the bundle, and
  `core.set_prune_load_func` supplies rank-specialized segment loaders.
- Adding a new basis or parameterization follows the 3-layer pattern: stateless
  `@njit` kernels in `core/` -> `structref` bundle in `dynamic/` implementing
  the uniform method contract -> registration in `Pruning._PRUNE_DISPATCH` and
  `PulsarSearchConfig.generate_branching_pattern`.

## Numba and tests

- JIT kernels: follow [`tests/jit_utils.py`](../tests/jit_utils.py) for
  coverage-friendly dual-path tests (`jit_variants`).
- Unwritten `np.empty` slots: see [`tests/test_unwritten_slots.py`](../tests/test_unwritten_slots.py).
- Parallel RNG reproducibility: kernels that draw randomness inside `prange`
  must derive per-iteration generators deterministically from `(entropy, idx)`
  so seeded results do not depend on Numba thread count.
- Full-pipeline integration tests that repeat per noise realisation: mark
  `@pytest.mark.slow` and run with `pytest --runslow`.
