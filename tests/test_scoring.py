import numpy as np
import pytest

from pyloki.detection.scoring import MatchedFilter

NBINS = 64
WIDTHS = np.array([1, 2, 4, 8])


class TestMatchedFilterNoise:
    """Regression guard on the score the public scoring API returns for noise.

    ``_compute_snr_double`` ends in ``maths.norm_isf_func(max(x_single, x_double))``
    where each ``x`` is ``chi_sq_minus_logsf_func(...)`` minus a look-elsewhere
    penalty, and so is routinely negative on noise. ``norm_isf_func`` used to wrap
    a negative index to the tail of its table and return ~+28 sigma for exactly
    those inputs: on this input, 193 of 200 pure-noise profiles scored above 20
    sigma, with a median of 28.02 and one non-finite.

    Read the predicate. This is a false-alarm property of the *scoring function*,
    not a measurement that a search emits false candidates. Neither
    ``compute_dot_double`` nor ``harmonic_summing_score_func`` is called anywhere
    in ``src/pyloki``; the live search scores through ``snr_score_batch_func``,
    which never reaches ``norm_isf_func``. A user reaches this by calling the
    public scoring API directly.
    """

    @pytest.mark.parametrize("seed", [42, 1234, 20260923])
    def test_pure_noise_does_not_score_as_a_detection(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        profiles = rng.normal(size=(200, NBINS)).astype(np.float32)
        scores = np.asarray(
            MatchedFilter(widths=WIDTHS, nbins=NBINS).compute_dot_double(profiles),
            dtype=float,
        )
        assert np.isfinite(scores).all(), (
            f"{int((~np.isfinite(scores)).sum())}/200 non-finite scores"
        )
        assert scores.max() < 6.0, (
            f"{int((scores > 6).sum())}/200 pure-noise profiles scored above "
            f"6 sigma, max {scores.max():.2f}"
        )
