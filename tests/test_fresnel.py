"""Required tests for the Fresnel finite-distance entry of fourier_diffraction.

Every numerical expected value below is produced by an *independent* reference
implemented directly in this file: a brute-force O(n^4) two-dimensional direct
complex sum (forward DFT, pointwise transfer-function multiply, inverse direct
sum).  The module under test never generates its own expected values; the
forward/backward round trip appears only as a supplementary consistency test.
Complex fields are compared as complex numbers, so phase is checked, not just
intensity.
"""

import cmath
import math
import unittest

import fourier_diffraction as fd


# ---------------------------------------------------------------------------
# Independent reference (must not call the module's transforms)
# ---------------------------------------------------------------------------


def reference_fresnel_field(aperture, dx, wavelength, z):
    """Brute-force direct-sum Fresnel transfer-function propagation.

    Forward 2-D DFT by explicit double sum, pointwise multiply by the
    separable kernel exp(-i pi lambda z (fx^2 + fy^2)), inverse by explicit
    double sum with 1/n**2 scaling.  Returns the centered complex field.
    """
    n = len(aperture)
    # Centered -> unshifted indices.
    un = [[0j] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            un[(i - n // 2) % n][(j - n // 2) % n] = complex(aperture[i][j])
    # Forward transform, direct definition.
    spec = [[0j] * n for _ in range(n)]
    for k in range(n):
        for l in range(n):
            s = 0j
            for m in range(n):
                for p in range(n):
                    s += un[m][p] * cmath.exp(-2j * math.pi * (k * m + l * p) / n)
            spec[k][l] = s
    # Pointwise separable kernel on centered frequency bins.
    for k in range(n):
        fx = ((k + n // 2) % n - n // 2) / (n * dx)
        for l in range(n):
            fy = ((l + n // 2) % n - n // 2) / (n * dx)
            spec[k][l] *= cmath.exp(-1j * math.pi * wavelength * z * (fx * fx + fy * fy))
    # Inverse transform, direct definition with 1/n**2 scaling.
    out = [[0j] * n for _ in range(n)]
    for m in range(n):
        for p in range(n):
            s = 0j
            for k in range(n):
                for l in range(n):
                    s += spec[k][l] * cmath.exp(2j * math.pi * (k * m + l * p) / n)
            out[m][p] = s / (n * n)
    # Unshifted -> centered layout.
    field = [[0j] * n for _ in range(n)]
    for k in range(n):
        for l in range(n):
            field[(k + n // 2) % n][(l + n // 2) % n] = out[k][l]
    return field


def make_aperture(n):
    """Deterministic non-trivial complex aperture (no randomness)."""
    return [
        [cmath.rect(1.0 + 0.1 * ((i + 2 * j) % 3), 0.31 * i - 0.17 * j) for j in range(n)]
        for i in range(n)
    ]


def assert_complex_grid_close(case, got, want, tol):
    n = len(want)
    scale = max(1.0, max(abs(want[i][j]) for i in range(n) for j in range(n)))
    for i in range(n):
        for j in range(n):
            case.assertLessEqual(
                abs(got[i][j] - want[i][j]) / scale,
                tol,
                f"field mismatch at ({i}, {j}): {got[i][j]!r} vs {want[i][j]!r}",
            )


# ---------------------------------------------------------------------------
# Numerical agreement with the independent reference
# ---------------------------------------------------------------------------


class TestFresnelAgainstReference(unittest.TestCase):
    def test_complex_field_matches_direct_sum(self):
        n, dx, wavelength, z = 8, 12.0e-6, 0.6328e-6, 0.035
        aperture = make_aperture(n)
        result = fd.fresnel(aperture, dx, wavelength, z)
        want = reference_fresnel_field(aperture, dx, wavelength, z)
        assert_complex_grid_close(self, result.field, want, 1e-11)

    def test_single_point_aperture_matches_direct_sum(self):
        # Delta at the array centre: phase as well as amplitude must agree.
        n, dx, wavelength, z = 8, 10.0e-6, 0.5e-6, 0.02
        aperture = [[0.0] * n for _ in range(n)]
        aperture[n // 2][n // 2] = 1.0
        result = fd.fresnel(aperture, dx, wavelength, z)
        want = reference_fresnel_field(aperture, dx, wavelength, z)
        assert_complex_grid_close(self, result.field, want, 1e-11)
        # Impulse response of a unit-modulus kernel has equal total energy.
        self.assertLessEqual(abs(result.power_error), 1e-12)

    def test_negative_distance_matches_direct_sum(self):
        n, dx, wavelength, z = 8, 12.0e-6, 0.6328e-6, -0.035
        aperture = make_aperture(n)
        result = fd.fresnel(aperture, dx, wavelength, z)
        want = reference_fresnel_field(aperture, dx, wavelength, z)
        assert_complex_grid_close(self, result.field, want, 1e-11)


# ---------------------------------------------------------------------------
# Exact / structural behaviour
# ---------------------------------------------------------------------------


class TestFresnelStructure(unittest.TestCase):
    def test_zero_distance_returns_input(self):
        n = 6
        aperture = make_aperture(n)
        result = fd.fresnel(aperture, 10.0e-6, 0.5e-6, 0.0)
        for i in range(n):
            for j in range(n):
                self.assertEqual(result.field[i][j], complex(aperture[i][j]))
        self.assertEqual(result.du, 10.0e-6)
        self.assertEqual(result.power_error, 0.0)
        self.assertEqual(result.output_power, result.input_power)

    def test_plane_wave_is_unchanged(self):
        # Constant aperture -> spectrum is a delta at the zero-frequency bin,
        # where H = 1, so the field must come back unchanged.
        n = 8
        amp = 0.7 - 0.2j
        aperture = [[amp] * n for _ in range(n)]
        result = fd.fresnel(aperture, 10.0e-6, 0.5e-6, 0.031)
        for i in range(n):
            for j in range(n):
                self.assertLessEqual(abs(result.field[i][j] - amp), 1e-15)

    def test_plane_phase_ramp_gains_exact_kernel_phase(self):
        # Aperture whose unshifted layout is exp(2 pi i (k0 m + l0 p) / n)
        # has a single spectrum bin at (k0, l0); propagation multiplies the
        # whole field by H[k0, l0] exactly.
        n, k0, l0 = 8, 2, 5  # l0 = 5 aliases to frequency bin -3
        dx, wavelength, z = 10.0e-6, 0.5e-6, 0.025
        aperture = [
            [
                cmath.exp(2j * math.pi * (k0 * ((i - n // 2) % n) + l0 * ((j - n // 2) % n)) / n)
                for j in range(n)
            ]
            for i in range(n)
        ]
        fk = k0 / (n * dx)
        fl = ((l0 + n // 2) % n - n // 2) / (n * dx)
        h = cmath.exp(-1j * math.pi * wavelength * z * (fk * fk + fl * fl))
        result = fd.fresnel(aperture, dx, wavelength, z)
        for i in range(n):
            for j in range(n):
                self.assertLessEqual(
                    abs(result.field[i][j] - h * aperture[i][j]), 1e-12,
                    f"phase-ramp mismatch at ({i}, {j})",
                )

    def test_forward_backward_round_trip(self):
        # Supplementary consistency check only: |H| = 1 and H(-z) = conj(H(z))
        # make the round trip the identity up to round-off.
        n = 12
        aperture = make_aperture(n)
        dx, wavelength, z = 8.0e-6, 0.55e-6, 0.04
        forward = fd.fresnel(aperture, dx, wavelength, z)
        back = fd.fresnel(forward.field, dx, wavelength, -z)
        assert_complex_grid_close(self, back.field, aperture, 1e-12)

    def test_output_window_and_coordinates(self):
        n, dx = 7, 9.0e-6
        result = fd.fresnel(make_aperture(n), dx, 0.5e-6, 0.01)
        self.assertEqual(result.du, dx)
        self.assertEqual(result.center, (n // 2, n // 2))
        for p in range(n):
            self.assertAlmostEqual(result.x[p], (p - n // 2) * dx, delta=1e-20)
            self.assertAlmostEqual(result.y[p], (p - n // 2) * dx, delta=1e-20)
        self.assertEqual(result.x[n // 2], 0.0)

    def test_parseval_energy_conservation(self):
        n = 16
        result = fd.fresnel(make_aperture(n), 10.0e-6, 0.5e-6, 0.05)
        self.assertLessEqual(abs(result.power_error), 1e-12)

    def test_zero_input_has_zero_power_error(self):
        n = 4
        result = fd.fresnel([[0.0] * n for _ in range(n)], 1e-5, 0.5e-6, 0.01)
        self.assertEqual(result.input_power, 0.0)
        self.assertEqual(result.power_error, 0.0)


# ---------------------------------------------------------------------------
# Error semantics
# ---------------------------------------------------------------------------


class TestFresnelErrors(unittest.TestCase):
    def setUp(self):
        self.aperture = make_aperture(4)

    def test_z_type_and_finiteness(self):
        for bad, exc in (
            (True, TypeError),
            ("0.1", TypeError),
            (complex(0.1), TypeError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (float("-inf"), ValueError),
        ):
            with self.assertRaises(exc, msg=f"z={bad!r}"):
                fd.fresnel(self.aperture, 1e-5, 0.5e-6, bad)

    def test_physical_parameters(self):
        with self.assertRaises(ValueError):
            fd.fresnel(self.aperture, 0.0, 0.5e-6, 0.1)      # dx <= 0
        with self.assertRaises(ValueError):
            fd.fresnel(self.aperture, 1e-5, -0.5e-6, 0.1)    # wavelength <= 0
        with self.assertRaises(TypeError):
            fd.fresnel(self.aperture, True, 0.5e-6, 0.1)     # bool dx

    def test_aperture_validation_shared(self):
        with self.assertRaises(ValueError):
            fd.fresnel([[1.0, float("nan")], [1.0, 1.0]], 1e-5, 0.5e-6, 0.1)
        with self.assertRaises(ValueError):
            fd.fresnel([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], 1e-5, 0.5e-6, 0.1)
        with self.assertRaises(TypeError):
            fd.fresnel([[True, 1.0], [1.0, 1.0]], 1e-5, 0.5e-6, 0.1)
        with self.assertRaises(ValueError):
            fd.fresnel([[1.0] * 33 for _ in range(33)], 1e-5, 0.5e-6, 0.1)

    def test_error_message_names_parameter(self):
        with self.assertRaises(ValueError) as ctx:
            fd.fresnel(self.aperture, 1e-5, 0.5e-6, float("nan"))
        self.assertIn("z", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
