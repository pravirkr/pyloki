import numpy as np

from pyloki.detection import thresholding
from pyloki.detection.thresholding import schedule_survive_probs

# a pruning pattern of the kind generate_branching_pattern returns: x4, x9 at the start, x3 later
BP = np.array([4.0, 9.0, 1.0, 1.0, 3.0, 3.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 3.0, 1.0, 1.0])
# a pattern whose branching stages after the fourth are far apart
BP_SPARSE = np.array(
    [4.0, 3.0, 4.0, 1.0, 1.0, 9.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0, 1.0, 1.0]
)


def test_protect_zero_is_the_plain_vector() -> None:
    probs, stages, factor = schedule_survive_probs(BP, protect=0)
    np.testing.assert_allclose(probs[BP > 1], 1.0 / BP[BP > 1])
    assert np.all(probs[BP == 1] == 1.0)
    assert stages == [] and factor == 1.0
    probs_nb, _, _ = schedule_survive_probs(BP, protect=0, survive_nonbranch=0.95)
    assert np.all(probs_nb[BP == 1] == 0.95)


def test_schedule_arithmetic_and_return_to_size() -> None:
    probs, stages, factor = schedule_survive_probs(BP)
    assert stages == [5, 6, 8]
    assert abs(factor - 36.0 ** (1.0 / 3.0)) < 1e-12
    assert np.all(probs[:4] == 1.0)
    for st in stages:
        assert abs(probs[st - 1] - 1.0 / (BP[st - 1] * factor)) < 1e-12
    np.testing.assert_allclose(probs[8:][BP[8:] > 1], 1.0 / BP[8:][BP[8:] > 1])
    # the noise tree relative to the plain ladder's: 36x through the protected stages, 1x from stage 8 on
    rel = np.cumprod(BP * probs)
    assert abs(rel[1] - 36.0) < 1e-9 and abs(rel[3] - 36.0) < 1e-9
    assert np.allclose(rel[7:], 1.0)


def test_window_is_in_branching_stages_not_levels() -> None:
    probs, stages, factor = schedule_survive_probs(BP_SPARSE)
    assert stages == [6, 10, 18]
    assert abs(factor - 48.0 ** (1.0 / 3.0)) < 1e-12
    rel = np.cumprod(BP_SPARSE * probs)
    assert abs(rel[2] - 48.0) < 1e-9 and np.allclose(rel[17:], 1.0)


def test_fewer_branching_stages_than_asked() -> None:
    short = np.array([4.0, 9.0, 1.0, 1.0, 3.0, 1.0])
    probs, stages, factor = schedule_survive_probs(short)
    assert stages == [5] and abs(factor - 36.0) < 1e-12 and abs(probs[4] - 1.0 / 108.0) < 1e-12
    probs_p, stages_p, factor_p = schedule_survive_probs(BP, return_stages=0)
    assert stages_p == [] and factor_p == 1.0 and np.all(probs_p[:4] == 1.0)


def test_determine_scheme_accepts_the_vector() -> None:
    probs, _, _ = schedule_survive_probs(BP)
    states = thresholding.determine_scheme(
        probs, BP, ref_ducy=0.1, nbins=32, ntrials=128, snr_final=9.0, seed=3
    )
    thresholds = np.asarray(states.thresholds)
    assert len(thresholds) == len(BP)
    # no cut at the protected stages: the thresholds there are the minimum over trials, below the
    # later cutting stages' thresholds
    assert thresholds[:4].max() < thresholds[4]
