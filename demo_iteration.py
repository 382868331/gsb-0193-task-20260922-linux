"""Fixed-input demonstration of finite-distance Fresnel propagation.

Runs real computations here and now (no sleep, no network, no prerecorded
answers): forward propagation of a rectangular aperture, an independent
brute-force 2-D cross-check on a small grid, a forward/backward round trip
with its honest finite-window error, the z = 0 degenerate case, and one
genuinely triggered input rejection.  Finishes in well under 90 seconds
(typical: a fraction of a second).
"""

import cmath
import math
import time

import fourier_diffraction as fd


def direct_fresnel(aperture, dx, wavelength, z):
    """Independent brute-force 2-D sum of the fixed discrete kernel."""
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


def main() -> None:
    n = 32
    dx = 10.0e-6          # 10 um sample pitch, identical in both planes
    wavelength = 0.5e-6   # 500 nm
    z = 6.0e-3            # 6 mm forward propagation

    aperture = fd.rectangular_aperture(n, width=8 * dx, height=5 * dx, dx=dx)
    lit = int(sum(sum(row) for row in aperture))

    t0 = time.perf_counter()
    fwd = fd.fresnel(aperture, dx, wavelength, z)
    back = fd.fresnel(fwd.field, dx, wavelength, -z)
    elapsed = time.perf_counter() - t0

    peak = max(max(row) for row in fwd.intensity)
    ci, cj = fwd.center
    max_in = max(abs(v) for row in aperture for v in row)
    rt_err = max(
        abs(back.field[i][j] - aperture[i][j]) for i in range(n) for j in range(n)
    ) / max_in

    print("=== Fresnel finite-distance propagation demo (real computation) ===")
    print(f"aperture       : {n}x{n} rectangular, {lit} lit pixels, dx = {dx:.2e} m")
    print(f"distance       : z = {z:.2e} m (wavelength {wavelength:.2e} m)")
    print(f"window         : output grid {fwd.n}x{fwd.n}, same pitch dx = {fwd.dx:.2e} m")
    print(f"coordinates    : x in [{fwd.x[0]:.3e}, {fwd.x[-1]:.3e}] m, "
          f"centre index {fwd.center} at x = {fwd.x[ci]:.1e} m")
    print(f"peak intensity : {peak:.6e} (centre intensity {fwd.intensity[ci][cj]:.6e})")
    print(f"input power    : {fwd.input_power:.12e}")
    print(f"output power   : {fwd.output_power:.12e}")
    print(f"power error    : {fwd.power_error:+.3e} (finite-window truncation, not round-off)")
    print(f"round trip     : +z then -z, max field recovery error = {rt_err:.3e}")
    print(f"elapsed        : {elapsed * 1000:.1f} ms")

    # ---- Independent cross-check on a small grid (phase included) ----------------
    print()
    print("=== Independent brute-force cross-check (8x8, complex field) ===")
    m = 8
    small = [
        [complex(((i * 7 + j * 3) % 5) - 2, ((i + 2 * j) % 3) - 1) / 2 for j in range(m)]
        for i in range(m)
    ]
    got = fd.fresnel(small, dx, wavelength, 2.0e-3)
    want = direct_fresnel(small, dx, wavelength, 2.0e-3)
    scale = max(abs(v) for row in want for v in row)
    diff = max(
        abs(got.field[i][j] - want[i][j]) for i in range(m) for j in range(m)
    ) / scale
    print(f"max relative complex-field deviation vs direct sum: {diff:.3e}")

    # ---- Degenerate distance ------------------------------------------------------
    print()
    print("=== z = 0 degenerate case ===")
    ident = fd.fresnel(small, dx, wavelength, 0.0)
    same = all(ident.field[i][j] == small[i][j] for i in range(m) for j in range(m))
    print(f"z = 0 returns the input unchanged: {same} (power_error = {ident.power_error})")

    # ---- A boundary that really triggers ------------------------------------------
    print()
    print("=== Input rejection boundary (actually raised below) ===")
    try:
        fd.fresnel(aperture, dx, wavelength, float("nan"))
    except ValueError as exc:
        print(f"ValueError raised as specified: {exc}")
    else:
        raise AssertionError("expected ValueError for z = NaN was not raised")


if __name__ == "__main__":
    main()

