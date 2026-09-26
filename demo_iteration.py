"""Fixed-input demonstration of finite-distance Fresnel propagation.

Propagates a 32x32 circular aperture over a finite distance with the Fresnel
transfer-function entry, checks the forward/backward round trip, and triggers
one genuine input-rejection boundary.  Everything is computed here and now:
no sleep, no network, no prerecorded results.  Runs in well under 90 seconds
(the whole computation is milliseconds).
"""

import time

import fourier_diffraction as fd


def main() -> None:
    # ---- Fixed physical parameters ------------------------------------------------
    n = 32
    dx = 10.0e-6          # 10 micrometre sample pitch (input AND output window)
    wavelength = 0.5e-6   # 500 nm
    z = 0.05              # 5 cm, finite distance (Fraunhofer would need ~ metres)
    radius = 6.0 * dx

    aperture = fd.circular_aperture(n, radius, dx)
    lit = int(sum(sum(row) for row in aperture))

    t0 = time.perf_counter()
    forward = fd.fresnel(aperture, dx, wavelength, z)
    backward = fd.fresnel(forward.field, dx, wavelength, -z)
    at_zero = fd.fresnel(aperture, dx, wavelength, 0.0)
    elapsed = time.perf_counter() - t0

    ci, cj = forward.center
    peak = max(max(row) for row in forward.intensity)
    roundtrip = max(
        abs(backward.field[i][j] - complex(aperture[i][j]))
        for i in range(n) for j in range(n)
    )
    identity = max(
        abs(at_zero.field[i][j] - complex(aperture[i][j]))
        for i in range(n) for j in range(n)
    )

    print("Fresnel finite-distance propagation (transfer-function model)")
    print(f"  aperture        : {n}x{n} circular, radius {radius * 1e6:.1f} um, {lit} lit pixels")
    print(f"  dx (= du)       : {dx * 1e6:.2f} um   wavelength: {wavelength * 1e9:.1f} nm")
    print(f"  distance z      : {z * 1e3:.1f} mm (signed; negative z back-propagates)")
    print(f"  window          : x in [{forward.x[0] * 1e3:+.3f}, {forward.x[-1] * 1e3:+.3f}] mm")
    print(f"  centre intensity: {forward.intensity[ci][cj]:.6e}  (index {forward.center})")
    print(f"  peak intensity  : {peak:.6e}")
    print(f"  input power     : {forward.input_power:.12e}")
    print(f"  output power    : {forward.output_power:.12e}")
    print(f"  power error     : {forward.power_error:+.3e} (Parseval, |H| = 1)")
    print(f"  round-trip error: {roundtrip:.3e} (z -> -z returns the input)")
    print(f"  z = 0 identity  : {identity:.3e} (input returned, no division by zero)")
    print(f"  elapsed         : {elapsed * 1e3:.1f} ms (computed live)")

    # ---- Genuinely triggered rejection boundary -----------------------------------
    try:
        fd.fresnel(aperture, dx, wavelength, float("nan"))
    except ValueError as exc:
        print(f"  rejected as designed: fresnel(..., z=nan) -> ValueError: {exc}")


if __name__ == "__main__":
    main()

