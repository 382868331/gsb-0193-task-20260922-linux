"""Required tests for the Fresnel finite-distance entry of fourier_diffraction.

Every numerical expected value below is produced by an *independent*
reference implemented directly in this file: a brute-force O(N^4) 2-D
complex sum of the fixed discrete Fresnel kernel (direct_fresnel), and a
closed-form chirp for the single-pixel aperture.  The module under test is
never used to generate its own expected values; the forward/backward
round trip appears only as a supplementary consistency check with an
explicit, documented finite-window error bound.
"""

import cmath
import math
import unittest

import fourier_diffraction as fd


# ---------------------------------------------------------------------------
# Independent references (must not call the module's propagation code)
# ---------------------------------------------------------------------------


def direct_fresnel(aperture, dx, wavelength, z):
    """Brute-force 2-D complex sum of the fixed discrete Fresnel kernel.

    U[p, q] = dx**2 / (i * wavelength * z)
              * sum_{t,u} A[t, u]
                * exp(i pi (((p-t)dx)**2 + ((q-u)dx)**2) / (wavelength z))

    Computed element by element; no transform routine involved.
    """
    n = len(aperture)
    scale = dx * dx / (1j * wavelength * z)
    out = [[0j] * n for _ in range(n)]
    for p in range(n):
        for q in range(n):
            s = 0j
            for t in range(n):
                for u in range(n):
                    phase = math.pi * (((p - t) * dx) ** 2 + ((q - u) * dx) ** 2) / (
                        wavelength * z
                    )
                    s += complex(aperture[t][u]) * cmath.exp(1j * phase)
            out[p][q] = scale * s
    return out


def max_field(grid):
    return max(abs(v) for row in grid for v in row)


def grid_diff(a, b):
    n = len(a)
    return max(abs(a[i][j] - b[i][j]) for i in range(n) for j in range(n))


DX = 10.0e-6
WAVELENGTH = 0.5e-6


def deterministic_aperture(n):
    """Fixed complex aperture (no randomness, no module helpers)."""
    return [
        [complex(((i * 7 + j * 3) % 5) - 2, ((i + 2 * j) % 3) - 1) / 2 for j in range(n)]
        for i in range(n)
    ]


class TestAgainstDirectSum(unittest.TestCase):
    """Small grids: module output vs the brute-force 2-D complex sum.

    The complex field itself is compared (phase included), not only |U|**2.
    """

    def check(self, n, z):
        aperture = deterministic_aperture(n)
        result = fd.fresnel(aperture, DX, WAVELENGTH, z)
        want = direct_fresnel(aperture, DX, WAVELENGTH, z)
        scale = max(1.0, max_field(want))
        self.assertLess(grid_diff(result.field, want), 1e-11 * scale, msg=f"n={n}, z={z}")
        # Intensity must be consistent with the returned complex field.
        for i in range(n):
            for j in range(n):
                u = result.field[i][j]
                self.assertAlmostEqual(
                    result.intensity[i][j], u.real * u.real + u.imag * u.imag, places=30
                )

    def test_forward_positive_distance(self):
        for n in (1, 2, 3, 5, 6):
            self.check(n, 2.0e-3)

    def test_backward_negative_distance(self):
        # Negative z must use the conjugate kernel; verified against the
        # same independent sum with a signed denominator.
        for n in (2, 4, 6):
            self.check(n, -3.0e-3)

    def test_forward_backward_are_kernel_conjugates(self):
        # |U(-z)| equals |U(z)| mirrored only through the kernel sign; the
        # direct sums already pin the phases, so here we only check the
        # module's two signs produce conjugate-related intensities for a
        # real symmetric aperture.
        n = 6
        aperture = [[1.0 if (i - n // 2) ** 2 + (j - n // 2) ** 2 <= 4 else 0.0
                     for j in range(n)] for i in range(n)]
        fwd = fd.fresnel(aperture, DX, WAVELENGTH, 2.5e-3)
        bwd = fd.fresnel(aperture, DX, WAVELENGTH, -2.5e-3)
        for i in range(n):
            for j in range(n):
                scale = max(1.0, fwd.intensity[i][j], bwd.intensity[i][j])
                self.assertLess(abs(fwd.intensity[i][j] - bwd.intensity[i][j]), 1e-12 * scale)


class TestSinglePixel(unittest.TestCase):
    def test_delta_gives_closed_form_chirp(self):
        # A single lit pixel at the centre must produce
        # U[p, q] = dx**2/(i lambda z) * exp(i pi ((px)**2+(qy)**2)/(lambda z))
        # exactly (closed form, no transform involved).
        n = 8
        z = 1.5e-3
        aperture = [[0j] * n for _ in range(n)]
        aperture[n // 2][n // 2] = 2.0 - 0.5j
        result = fd.fresnel(aperture, DX, WAVELENGTH, z)
        scale = DX * DX / (1j * WAVELENGTH * z)
        c = n // 2
        amp = abs(2.0 - 0.5j)
        for p in range(n):
            for q in range(n):
                phase = math.pi * (((p - c) * DX) ** 2 + ((q - c) * DX) ** 2) / (
                    WAVELENGTH * z
                )
                want = scale * (2.0 - 0.5j) * cmath.exp(1j * phase)
                self.assertLess(
                    abs(result.field[p][q] - want),
                    1e-12 * amp * abs(scale),
                    msg=f"p={p} q={q}",
                )


class TestPlanePhase(unittest.TestCase):
    def test_linear_phase_aperture_matches_direct_sum(self):
        # Plane wave with a linear phase ramp (tilt): complex field,
        # phase included, against the independent 2-D sum.
        n = 6
        z = 2.0e-3
        aperture = [
            [cmath.exp(1j * 2 * math.pi * (3 * t + u) / n) for u in range(n)]
            for t in range(n)
        ]
        result = fd.fresnel(aperture, DX, WAVELENGTH, z)
        want = direct_fresnel(aperture, DX, WAVELENGTH, z)
        scale = max(1.0, max_field(want))
        self.assertLess(grid_diff(result.field, want), 1e-11 * scale)

    def test_tilt_shifts_intensity_centroid(self):
        # A tilted plane wave must displace the diffracted intensity
        # towards the tilt direction; sign of the displacement is checked
        # on the intensity centroid along the tilted axis.
        n = 16
        z = 4.0e-3
        aperture = [
            [cmath.exp(1j * 2 * math.pi * 2 * t / n) for _ in range(n)]
            for t in range(n)
        ]
        result = fd.fresnel(aperture, DX, WAVELENGTH, z)
        total = sum(sum(row) for row in result.intensity)
        centroid = sum(
            i * sum(result.intensity[i]) for i in range(n)
        ) / total
        self.assertGreater(centroid, n // 2)


class TestZeroDistance(unittest.TestCase):
    def test_z_zero_returns_input_exactly(self):
        aperture = deterministic_aperture(5)
        result = fd.fresnel(aperture, DX, WAVELENGTH, 0.0)
        self.assertEqual(result.z, 0.0)
        self.assertEqual(result.power_error, 0.0)
        self.assertEqual(result.input_power, result.output_power)
        for i in range(5):
            for j in range(5):
                self.assertEqual(result.field[i][j], aperture[i][j])
                v = aperture[i][j]
                self.assertAlmostEqual(
                    result.intensity[i][j], v.real * v.real + v.imag * v.imag, places=30
                )

    def test_z_zero_does_not_mutate_input(self):
        aperture = deterministic_aperture(4)
        snapshot = [list(row) for row in aperture]
        fd.fresnel(aperture, DX, WAVELENGTH, 0.0)
        self.assertEqual(aperture, snapshot)


class TestRoundTrip(unittest.TestCase):
    def test_forward_backward_error_within_documented_bound(self):
        # Supplementary consistency check.  Same-window Fresnel cannot be
        # exactly invertible: energy diffracted outside the finite window
        # is truncated.  For a compact central aperture at a distance
        # where the chirp is adequately sampled over the lit support
        # (lambda*|z|/(2 dx**2) >= lit radius in samples) the recovery
        # error is small; the bound below is the documented contract.
        n = 32
        z = 6.0e-3
        aperture = [[0j] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if abs(i - n // 2) <= 3 and abs(j - n // 2) <= 3:
                    aperture[i][j] = 1.0 + 0.3j * (i - j)
        fwd = fd.fresnel(aperture, DX, WAVELENGTH, z)
        back = fd.fresnel(fwd.field, DX, WAVELENGTH, -z)
        scale = max_field(aperture)
        err = grid_diff(back.field, aperture) / scale
        self.assertLess(err, 0.15)
        self.assertLess(abs(fwd.power_error), 0.05)


class TestSamplingWindowAndPower(unittest.TestCase):
    def test_output_window_matches_input_window(self):
        n = 12
        z = 3.0e-3
        result = fd.fresnel(deterministic_aperture(n), DX, WAVELENGTH, z)
        self.assertEqual(result.n, n)
        self.assertEqual(result.dx, DX)
        self.assertEqual(result.center, (n // 2, n // 2))
        self.assertEqual(len(result.x), n)
        self.assertEqual(len(result.y), n)
        for p in range(n):
            self.assertAlmostEqual(result.x[p], (p - n // 2) * DX, places=25)
            self.assertAlmostEqual(result.y[p], (p - n // 2) * DX, places=25)
        self.assertEqual(result.x[n // 2], 0.0)

    def test_power_accounting_fields(self):
        n = 8
        z = 4.0e-3
        aperture = deterministic_aperture(n)
        result = fd.fresnel(aperture, DX, WAVELENGTH, z)
        expect_in = DX * DX * sum(
            abs(v) ** 2 for row in aperture for v in row
        )
        expect_out = DX * DX * sum(sum(row) for row in result.intensity)
        self.assertLess(abs(result.input_power - expect_in), 1e-12 * expect_in)
        self.assertLess(abs(result.output_power - expect_out), 1e-12 * expect_out)
        self.assertAlmostEqual(
            result.power_error,
            (expect_out - expect_in) / expect_in,
            places=12,
        )

    def test_zero_aperture_power_error_is_zero(self):
        n = 4
        aperture = [[0.0] * n for _ in range(n)]
        result = fd.fresnel(aperture, DX, WAVELENGTH, 2.0e-3)
        self.assertEqual(result.input_power, 0.0)
        self.assertEqual(result.output_power, 0.0)
        self.assertEqual(result.power_error, 0.0)


class TestErrors(unittest.TestCase):
    def setUp(self):
        self.aperture = deterministic_aperture(4)

    def test_z_type_and_finiteness(self):
        for bad in (True, "1.0", None, 1j):
            with self.assertRaises(TypeError, msg=f"z={bad!r}"):
                fd.fresnel(self.aperture, DX, WAVELENGTH, bad)
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError, msg=f"z={bad!r}"):
                fd.fresnel(self.aperture, DX, WAVELENGTH, bad)

    def test_dx_and_wavelength_validation(self):
        with self.assertRaises(ValueError):
            fd.fresnel(self.aperture, 0.0, WAVELENGTH, 1e-3)
        with self.assertRaises(ValueError):
            fd.fresnel(self.aperture, -DX, WAVELENGTH, 1e-3)
        with self.assertRaises(TypeError):
            fd.fresnel(self.aperture, True, WAVELENGTH, 1e-3)
        with self.assertRaises(ValueError):
            fd.fresnel(self.aperture, DX, 0.0, 1e-3)
        with self.assertRaises(ValueError):
            fd.fresnel(self.aperture, DX, float("nan"), 1e-3)
        with self.assertRaises(TypeError):
            fd.fresnel(self.aperture, DX, "5e-7", 1e-3)

    def test_aperture_validation_still_applies(self):
        with self.assertRaises(ValueError):
            fd.fresnel([[1.0, 2.0], [3.0]], DX, WAVELENGTH, 1e-3)  # ragged
        with self.assertRaises(ValueError):
            fd.fresnel([], DX, WAVELENGTH, 1e-3)  # empty
        with self.assertRaises(ValueError):
            fd.fresnel([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], DX, WAVELENGTH, 1e-3)
        with self.assertRaises(ValueError):
            fd.fresnel([[float("nan")]], DX, WAVELENGTH, 1e-3)
        with self.assertRaises(TypeError):
            fd.fresnel([[True]], DX, WAVELENGTH, 1e-3)
        with self.assertRaises(ValueError):
            fd.fresnel(
                [[0.0] * 33 for _ in range(33)], DX, WAVELENGTH, 1e-3
            )  # above the 32 limit

    def test_error_messages_name_parameter(self):
        with self.assertRaises(ValueError) as ctx:
            fd.fresnel(self.aperture, DX, float("inf"), 1e-3)
        self.assertIn("wavelength", str(ctx.exception))
        with self.assertRaises(TypeError) as ctx:
            fd.fresnel(self.aperture, DX, WAVELENGTH, "z")
        self.assertIn("z", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
