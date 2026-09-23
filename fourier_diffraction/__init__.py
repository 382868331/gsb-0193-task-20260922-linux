"""fourier_diffraction — sampled Fraunhofer diffraction for complex apertures.

Pure Python 3.14 standard library; the 2-D DFT is implemented in this
package (``fourier_diffraction.dft``) and no FFT package is used.

Public API:
    fraunhofer_propagate  -- fixed Fraunhofer model propagator
    FraunhoferResult      -- result container
    circular_aperture     -- deterministic circular aperture
    rectangular_aperture  -- deterministic rectangular aperture
    grid_coordinates      -- input-plane pixel coordinates
    DiffractionError      -- locatable contract-violation error
"""

from .apertures import (
    MAX_APERTURE_N,
    circular_aperture,
    grid_coordinates,
    rectangular_aperture,
)
from .propagator import MAX_PADDED_M, FraunhoferResult, fraunhofer_propagate
from .validation import DiffractionError

__all__ = [
    "fraunhofer_propagate",
    "FraunhoferResult",
    "circular_aperture",
    "rectangular_aperture",
    "grid_coordinates",
    "DiffractionError",
    "MAX_APERTURE_N",
    "MAX_PADDED_M",
]
