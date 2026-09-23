"""Input contract and error-semantics tests."""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import unittest

from fourier_diffraction import (
    DiffractionError,
    circular_aperture,
    rectangular_aperture,
    fraunhofer_propagate,
)


def _ones(n):
    return [[1.0 + 0.0j for _ in range(n)] for _ in range(n)]


class ScalarValidationTests(unittest.TestCase):
    def test_rejects_nan_and_infinity(self):
        for bad, label in ((float("nan"), "NaN"), (float("inf"), "inf"),
                          (float("-inf"), "-inf")):
            with self.subTest(value=label, where="dx"):
                with self.assertRaises(DiffractionError) as cm:
                    fraunhofer_propagate(_ones(2), bad, 1.0, 1.0)
                self.assertEqual(cm.exception.parameter, "dx")
            with self.subTest(value=label, where="wavelength"):
                with self.assertRaises(DiffractionError):
                    fraunhofer_propagate(_ones(2), 1.0, bad, 1.0)
            with self.subTest(value=label, where="z"):
                with self.assertRaises(DiffractionError):
                    fraunhofer_propagate(_ones(2), 1.0, 1.0, bad)

    def test_rejects_non_positive_scalars(self):
        for bad in (0.0, -1e-9):
            with self.assertRaises(DiffractionError) as cm:
                fraunhofer_propagate(_ones(2), bad, 1.0, 1.0)
            self.assertEqual(cm.exception.parameter, "dx")
        with self.assertRaises(DiffractionError):
            fraunhofer_propagate(_ones(2), 1.0, 0.0, 1.0)
        with self.assertRaises(DiffractionError):
            fraunhofer_propagate(_ones(2), 1.0, 1.0, -1.0)

    def test_rejects_wrong_scalar_types(self):
        with self.assertRaises(DiffractionError):
            fraunhofer_propagate(_ones(2), "1.0", 1.0, 1.0)
        with self.assertRaises(DiffractionError):
            fraunhofer_propagate(_ones(2), 1.0, 1.0, None)

    def test_integer_parameters_reject_bools(self):
        with self.assertRaises(DiffractionError) as cm:
            circular_aperture(True, 1.0, 1.0)
        self.assertEqual(cm.exception.parameter, "n")
        with self.assertRaises(DiffractionError) as cm:
            fraunhofer_propagate(_ones(2), 1.0, 1.0, 1.0, pad_to=False)
        self.assertEqual(cm.exception.parameter, "pad_to")
        with self.assertRaises(DiffractionError) as cm:
            rectangular_aperture(4, 1.0, True, 1.0)
        self.assertEqual(cm.exception.parameter, "width")

    def test_rejects_nan_complex_aperture_samples(self):
        aperture = [[1.0 + 0.0j, 2.0], [0.0, float("nan") * 1j]]
        with self.assertRaises(DiffractionError) as cm:
            fraunhofer_propagate(aperture, 1.0, 1.0, 1.0)
        self.assertEqual(cm.exception.parameter, "aperture[1][1]")
        aperture[1][1] = complex(float("inf"), 0.0)
        with self.assertRaises(DiffractionError) as cm:
            fraunhofer_propagate(aperture, 1.0, 1.0, 1.0)
        self.assertEqual(cm.exception.parameter, "aperture[1][1]")

    def test_rejects_boolean_aperture_samples(self):
        aperture = [[1, 1], [True, 0]]
        with self.assertRaises(DiffractionError) as cm:
            fraunhofer_propagate(aperture, 1.0, 1.0, 1.0)
        self.assertEqual(cm.exception.parameter, "aperture[1][0]")


class ApertureStructureTests(unittest.TestCase):
    def test_rejects_empty_and_non_square(self):
        with self.assertRaises(DiffractionError) as cm:
            fraunhofer_propagate([], 1.0, 1.0, 1.0)
        self.assertEqual(cm.exception.parameter, "aperture")
        with self.assertRaises(DiffractionError):
            fraunhofer_propagate([[1.0, 0.0]], 1.0, 1.0, 1.0)
        with self.assertRaises(DiffractionError):
            fraunhofer_propagate([[1.0], [0.0, 1.0]], 1.0, 1.0, 1.0)

    def test_rejects_oversized_aperture(self):
        with self.assertRaises(DiffractionError) as cm:
            fraunhofer_propagate(_ones(33), 1.0, 1.0, 1.0)
        self.assertEqual(cm.exception.parameter, "aperture")
        with self.assertRaises(DiffractionError):
            circular_aperture(0, 1.0, 1.0)
        with self.assertRaises(DiffractionError):
            circular_aperture(33, 1.0, 1.0)

    def test_rejects_non_numeric_samples_and_rows(self):
        with self.assertRaises(DiffractionError):
            fraunhofer_propagate([["1.0", 0.0], [0.0, 1.0]], 1.0, 1.0, 1.0)
        with self.assertRaises(DiffractionError):
            fraunhofer_propagate([1.0, 0.0], 1.0, 1.0, 1.0)

    def test_pad_bounds(self):
        aperture = _ones(8)
        with self.assertRaises(DiffractionError) as cm:
            fraunhofer_propagate(aperture, 1.0, 1.0, 1.0, pad_to=7)
        self.assertEqual(cm.exception.parameter, "pad_to")
        with self.assertRaises(DiffractionError) as cm:
            fraunhofer_propagate(aperture, 1.0, 1.0, 1.0, pad_to=65)
        self.assertEqual(cm.exception.parameter, "pad_to")
        with self.assertRaises(DiffractionError):
            fraunhofer_propagate(aperture, 1.0, 1.0, 1.0, pad_to=16.0)

    def test_generator_physical_parameter_validation(self):
        with self.assertRaises(DiffractionError):
            circular_aperture(8, 1.0, 0.0)
        with self.assertRaises(DiffractionError):
            rectangular_aperture(8, 1.0, -1.0, 1.0)
        with self.assertRaises(DiffractionError):
            circular_aperture(8, 0.0, 1.0)
        with self.assertRaises(DiffractionError):
            circular_aperture(8, float("nan"), 1.0)

    def test_error_message_is_locatable(self):
        try:
            fraunhofer_propagate(_ones(2), 1.0, -5.0, 1.0)
        except DiffractionError as exc:
            self.assertIn("wavelength", str(exc))
            self.assertEqual(exc.parameter, "wavelength")
            self.assertIn("-5", str(exc))
        else:
            self.fail("DiffractionError not raised")

    def test_no_partial_output_on_failure(self):
        # The call either returns a complete result or raises; there is no
        # mutable batch/state API to leave partially modified.
        with self.assertRaises(DiffractionError):
            fraunhofer_propagate(_ones(4), 1.0, 1.0, 1.0, pad_to=3)
        # A follow-up valid call must be unaffected.
        result = fraunhofer_propagate(_ones(2), 1.0, 1.0, 1.0)
        self.assertEqual(result.padded_size, 2)


if __name__ == "__main__":
    unittest.main()
