import numpy as np
import pytest

from pyloki.detection.scoring import boxcar_snr_1d
from pyloki.simulation.pulse import (
    PulseShape,
    PulseSignalConfig,
    build_cdf_lut,
    generate_pulse_template,
    von_mises_pulse_shape,
)
from tests.jit_utils import jit_variants

SHAPES = ["boxcar", "gaussian", "von_mises"]
PHASES = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.98]
DUCY = 0.1
# The template is accumulated in float32, so one period of ~nbins samples
# carries a rounding error of order nbins * eps(float32).
FLUX_TOL = 1e-5


def make_template(
    shape: str,
    phi0: float,
    nbins: int = 256,
    ducy: float = DUCY,
    period: float = 1.0,
    nperiods: int = 1,
    impl=generate_pulse_template,
) -> np.ndarray:
    """Sample one (or several) periods of the pulse template at nbins/period.

    `impl` is the kernel or its ``py_func``; the parallel kernel's Python source runs
    on a representative subset of the tests below.
    """
    dt = period / nbins
    ngrid = max(4096, int(100 / ducy))
    cdf_lut = build_cdf_lut(shape, ducy, phi0, ngrid=ngrid)
    proper_time = np.arange(nbins * nperiods) * dt
    return impl(proper_time, dt, period, cdf_lut)


class TestBuildCdfLut:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("phi0", PHASES)
    def test_lut_is_a_periodic_cdf(self, shape: str, phi0: float) -> None:
        ngrid = 4096
        lut = build_cdf_lut(shape, DUCY, phi0, ngrid=ngrid)
        assert lut.shape == (ngrid + 1,)
        # non-decreasing, so that no differenced sample can be negative
        assert np.all(np.diff(lut) >= 0)
        # the wrap element is the periodic continuation: a CDF gains 1/period
        np.testing.assert_allclose(lut[-1], lut[0] + 1.0, atol=1e-6)

    def test_unknown_shape(self) -> None:
        with pytest.raises(ValueError, match="Unknown shape"):
            build_cdf_lut("triangle", DUCY, 0.5)

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("ducy", [0.05, 0.1, 0.3])
    def test_width_is_fwtm(self, shape: str, ducy: float) -> None:
        """`width` is the full width at a tenth of the maximum, for all shapes."""
        nbins = 4096
        prof = make_template(shape, 0.5, nbins=nbins, ducy=ducy)
        above = np.flatnonzero(prof >= 0.1 * prof.max())
        fwtm = (above[-1] - above[0] + 1) / nbins
        # the edges are resolved to one sample and to one LUT cell
        np.testing.assert_allclose(fwtm, ducy, atol=3.0 / nbins)


class TestGeneratePulseTemplate:
    @pytest.mark.parametrize("impl", jit_variants(generate_pulse_template))
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("phi0", PHASES)
    def test_flux_per_period(self, impl, shape: str, phi0: float) -> None:
        """One period of the template integrates to exactly one, at any phase."""
        prof = make_template(shape, phi0, impl=impl)
        np.testing.assert_allclose(prof.sum(), 1.0, atol=FLUX_TOL)

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("phi0", [0.0, 0.03, 0.5, 0.97])
    def test_flux_over_many_periods(self, shape: str, phi0: float) -> None:
        nperiods = 8
        prof = make_template(shape, phi0, nperiods=nperiods)
        np.testing.assert_allclose(prof.sum(), nperiods, atol=nperiods * FLUX_TOL)

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("phi0", PHASES)
    def test_no_negative_samples(self, shape: str, phi0: float) -> None:
        prof = make_template(shape, phi0)
        assert prof.min() >= 0.0

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("nbins", [8192, 16384])
    def test_no_negative_samples_finer_than_lut(
        self,
        shape: str,
        nbins: int,
    ) -> None:
        """period/dt well above ngrid: samples can land inside the wrap cell."""
        ngrid = max(4096, int(100 / DUCY))
        assert nbins > ngrid
        prof = make_template(shape, 0.5, nbins=nbins)
        assert prof.min() >= 0.0
        np.testing.assert_allclose(prof.sum(), 1.0, atol=nbins * 1e-7)

    @pytest.mark.parametrize("impl", jit_variants(generate_pulse_template))
    @pytest.mark.parametrize("shape", SHAPES)
    def test_boundary_sample_is_not_dropped(self, impl, shape: str) -> None:
        """The sample straddling the period boundary carries the wrapped mass."""
        nbins = 64
        # Peak on the boundary: the last sample, [1 - 1/nbins, 1), wraps onto the
        # first, and by symmetry the two carry equal, maximal flux.
        prof = make_template(shape, 0.0, nbins=nbins, impl=impl)
        assert prof[-1] > 0
        np.testing.assert_allclose(prof[-1], prof[0], atol=1e-6)
        np.testing.assert_allclose(prof[-1], prof.max(), atol=1e-6)
        np.testing.assert_allclose(prof.sum(), 1.0, atol=FLUX_TOL)

    @pytest.mark.parametrize("impl", jit_variants(generate_pulse_template))
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("phi0", PHASES)
    def test_profile_is_a_circular_shift(self, impl, shape: str, phi0: float) -> None:
        """Moving the peak by half a period rolls the profile by half a period.

        A template that silently truncates the mass falling outside [0, 1) is
        not translation invariant on the circle, so this fails long before the
        flux deficit becomes obvious.
        """
        nbins = 256
        prof = make_template(shape, phi0, nbins=nbins, impl=impl)
        shifted = make_template(shape, (phi0 + 0.5) % 1.0, nbins=nbins, impl=impl)
        np.testing.assert_allclose(
            prof,
            np.roll(shifted, -nbins // 2),
            atol=1e-6,
        )

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("phi0", PHASES)
    def test_peak_is_at_phi0(self, shape: str, phi0: float) -> None:
        """`phi0` is the centre of the pulse, for all shapes."""
        nbins = 256
        prof = make_template(shape, phi0, nbins=nbins)
        # circular centroid: argmax is ambiguous on the boxcar plateau
        phase = (np.arange(nbins) + 0.5) / nbins
        centroid = np.angle(np.sum(prof * np.exp(2j * np.pi * phase))) / (2 * np.pi)
        offset = (centroid - phi0 + 0.5) % 1.0 - 0.5
        assert abs(offset) < 1.0 / nbins

    @pytest.mark.parametrize("shape", SHAPES)
    def test_incommensurate_sampling(self, shape: str) -> None:
        """period/dt need not be an integer."""
        period = 1.234567
        nperiods = 16
        dt = period / 300.7
        # peak on the period boundary, so every pulse straddles a wrap
        cdf_lut = build_cdf_lut(shape, DUCY, 0.0, ngrid=4096)
        nsamps = int(nperiods * period / dt)
        # start half a period in, so the window holds exactly nperiods pulses
        proper_time = period / 2 + np.arange(nsamps) * dt
        prof = generate_pulse_template(proper_time, dt, period, cdf_lut)
        assert prof.min() >= 0.0
        np.testing.assert_allclose(prof.sum(), nperiods, atol=nperiods * FLUX_TOL)

    def test_sample_longer_than_period(self) -> None:
        """A sample covering a whole period collects all of the flux."""
        period = 1.0
        cdf_lut = build_cdf_lut("gaussian", DUCY, 0.3, ngrid=4096)
        proper_time = np.array([0.0, 1.0, 2.0])
        prof = generate_pulse_template(proper_time, period, period, cdf_lut)
        np.testing.assert_allclose(prof, np.ones(3), atol=FLUX_TOL)


class TestGenerateInitialScale:
    def test_generate_max_iter_zero_hits_target_snr(self) -> None:
        """Initial scale alone (max_iter=0) already lands near target S/N."""
        cfg = PulseSignalConfig(
            period=0.01,
            dt=1e-4,
            nsamps=2**16,
            snr=20.0,
            ducy=0.1,
            seed=7,
        )
        ts = cfg.generate(shape="gaussian", max_iter=0)
        fold = ts.fold_ephem(cfg.freq, cfg.fold_bins_ideal)
        widths = np.arange(1, cfg.fold_bins_ideal // 2)
        measured_snr = float(boxcar_snr_1d(fold, widths, 1.0).max())
        assert abs(measured_snr - 20.0) < 3.0


class TestPulseShape:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("phi0", [0.0, 0.25, 0.5, 0.95])
    def test_matches_cdf_lut_template_and_preserves_flux(
        self, shape: str, phi0: float
    ) -> None:
        nbins = 256
        period = 1.0
        dt = period / nbins
        proper_time = np.arange(nbins) * dt
        got = PulseShape(
            proper_time, dt, period, shape=shape, width=DUCY, pos=phi0
        ).generate()
        lut = build_cdf_lut(shape, DUCY, phi0, ngrid=16384)
        want = generate_pulse_template(proper_time, dt, period, lut)
        np.testing.assert_allclose(got.sum(), 1.0, atol=1e-9)
        np.testing.assert_allclose(got, want, atol=2e-4)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_width_is_fwtm(self, shape: str) -> None:
        nbins = 2048
        dt = 1.0 / nbins
        proper_time = np.arange(nbins) * dt
        prof = PulseShape(
            proper_time, dt, 1.0, shape=shape, width=DUCY, pos=0.5
        ).generate()
        above = np.flatnonzero(prof >= 0.1 * prof.max())
        fwtm = (above[-1] - above[0] + 1) / nbins
        np.testing.assert_allclose(fwtm, DUCY, atol=3.0 / nbins)


def test_von_mises_pulse_shape_uses_fwtm_convention() -> None:
    period, width, pos = 1.0, 0.1, 0.5
    t = np.array(
        [(pos - width / 2) * period, pos * period, (pos + width / 2) * period],
    )
    vals = von_mises_pulse_shape(t, 0.01, period, width, pos=pos)
    np.testing.assert_allclose(vals, [0.1, 1.0, 0.1], rtol=1e-12)
