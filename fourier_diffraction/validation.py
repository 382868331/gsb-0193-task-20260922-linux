"""Validation helpers and error semantics for the Fourier diffraction package.

All invalid input raises :class:`DiffractionError` carrying the offending
parameter name so that errors are locatable.  The package never silently
clamps, casts or repairs an illegal value.
"""

from __future__ import annotations

import math

__all__ = ["DiffractionError", "checked_real", "checked_int", "checked_complex"]


class DiffractionError(ValueError):
    """Raised for any contract violation on the public API.

    Attributes:
        parameter: dotted/indexed name of the offending parameter, e.g.
            ``"wavelength"`` or ``"aperture[3][2]"``.
    """

    def __init__(self, parameter: str, message: str):
        self.parameter = parameter
        super().__init__(f"invalid value for {parameter!r}: {message}")


def _reject_bool(value, name: str) -> None:
    # bool is a subclass of int; the contract forbids booleans wherever an
    # integer/number parameter is expected.
    if isinstance(value, bool):
        raise DiffractionError(name, "boolean is not accepted here")


def checked_real(value, name: str, *, positive: bool = False) -> float:
    """Validate a finite real scalar (int or float, but not bool)."""
    _reject_bool(value, name)
    if not isinstance(value, (int, float)):
        raise DiffractionError(
            name, f"expected a real number, got {type(value).__name__}"
        )
    result = float(value)
    if math.isnan(result):
        raise DiffractionError(name, "NaN is not accepted")
    if math.isinf(result):
        raise DiffractionError(name, "infinite value is not accepted")
    if positive and result <= 0.0:
        raise DiffractionError(name, f"expected a strictly positive value, got {result}")
    return result


def checked_int(value, name: str, *, minimum=None, maximum=None) -> int:
    """Validate an integer parameter (booleans rejected)."""
    _reject_bool(value, name)
    if not isinstance(value, int):
        raise DiffractionError(
            name, f"expected an integer, got {type(value).__name__}"
        )
    if minimum is not None and value < minimum:
        raise DiffractionError(
            name, f"value {value} is below the minimum {minimum}"
        )
    if maximum is not None and value > maximum:
        raise DiffractionError(
            name, f"value {value} is above the maximum {maximum}"
        )
    return value


def checked_complex(value, name: str):
    """Validate a finite complex/real scalar (bool rejected)."""
    _reject_bool(value, name)
    if not isinstance(value, (int, float, complex)):
        raise DiffractionError(
            name, f"expected a complex/real number, got {type(value).__name__}"
        )
    result = complex(value)
    if math.isnan(result.real) or math.isnan(result.imag):
        raise DiffractionError(name, "NaN is not accepted")
    if math.isinf(result.real) or math.isinf(result.imag):
        raise DiffractionError(name, "infinite value is not accepted")
    return result
