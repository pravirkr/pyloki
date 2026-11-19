# PyLoki Utility Snippets

Self-contained Jupyter notebooks demonstrating individual `pyloki` utilities
(no external data files required):

- `01_preprocessing_and_rfi.ipynb`: robust preprocessing (`PreprocessOptions`), birdie
  zapping, and `DM=0` RFI mask generation/application.
- `02_scoring_and_thresholds.ipynb`: heteroscedastic GLRT boxcar, double-boxcar,
  combined/Kadane scorers, and empirical `threshold_simulation`.
- `03_orbit_fitting.ipynb`: global Fourier phase-curve fitting with eclipse masking,
  multi-start exact Keplerian polishing, and Winsorized Doppler fitting.
- `04_candidate_sifting_and_export.ipynb`: single-DM phase-track clustering and
  self-harmonic flagging, multi-DM `SurveyCandidateGroups` with `DM=0` RFI filtering,
  rational `HarmonicMatcher`, segment ephemeris export, and TEMPO2 `t2pred.dat` export.
- `05_waterfalls_and_subbands.ipynb`: 2D `(segment x phase)` waterfall extraction,
  S/N-preserving `bin2d`, 4-panel diagnostic card, headless `WaterfallLabState`, and
  non-wrapping dispersive `fold_subbands`.
- `06_sensitivity_and_nulling.ipynb`: analytical FFA/EP vs. FFT harmonic sensitivity
  comparator, $Z^2_m$ & $H$-test, 2-state Markov HMM nulling/eclipse scorer, and
  Pareto/lognormal `HeavyTailLikelihood` bounds.
