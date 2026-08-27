#!/usr/bin/env python3
"""
Meridional sections of the Descartes (Cartesian) ovoids.

Reproduces figure 2.25 of *Lectures on Optics*, chapter 2, as a fully
vectorised PDF.  The surface is drawn from the explicit sag derived in
section 2.7,

                    (1 + S rho^2) +- sqrt(1 + (2S - O^2 G) rho^2)
        z(rho)  =  ----------------------------------------------
                                     O G

        r(rho)  =  +- sqrt(rho^2 - z(rho)^2)

with rho the polar radius measured from the vertex and G, O, T, S the form
parameters of the ovoid.  The minus sign is the physical branch, the one
that passes through the vertex; the plus sign is the outer sheet of the
quartic.

Usage:  python3 cartesian_surfaces.py [-o OUTPUT.pdf]
"""

import argparse

import numpy as np
import matplotlib as mpl

mpl.use("pdf")                      # vector backend, no rasterisation
import matplotlib.pyplot as plt

# Keep text as real vector glyphs rather than outlines or bitmaps.
mpl.rcParams.update({
    "pdf.fonttype":      42,        # embed TrueType, text stays selectable
    "pdf.compression":   9,
    "font.family":       "serif",
    "mathtext.fontset":  "cm",      # Computer Modern, matches the book
    "axes.linewidth":    0.6,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
})


def form_parameters(no, ni, zo, zi):
    """Form parameters G, O, T, S of the ovoid conjugating zo and zi."""
    G = (ni**2 * zi - no**2 * zo) ** 2 / (
        ni * no * (ni * zi - no * zo) * (ni * zo - no * zi)
    )
    O = (ni * zo - no * zi) / (zi * zo * (ni - no))
    T = (ni - no) * (ni + no) ** 2 / (4 * ni * no * zi * zo * (ni * zi - no * zo))
    S = (ni + no) * (ni**2 * zi - no**2 * zo) / (
        2 * ni * no * zi * zo * (ni * zi - no * zo)
    )
    return G, O, T, S


def branch(no, ni, zo, zi, sign, rho_max, n=200_000):
    """
    Return (z, r) for one branch, as a closed curve.

    `sign` is +1 or -1 and selects the root of the quadratic.  Points where
    the discriminant or rho^2 - z^2 turn negative are dropped, which is what
    closes the curve on itself.
    """
    G, O, T, S = form_parameters(no, ni, zo, zi)

    rho = np.linspace(0.0, rho_max, n)
    disc = 1.0 + (2.0 * S - O**2 * G) * rho**2
    ok = disc >= 0.0
    rho, disc = rho[ok], disc[ok]

    z = ((1.0 + S * rho**2) + sign * np.sqrt(disc)) / (O * G)

    r2 = rho**2 - z**2
    ok = r2 >= 0.0
    z, r2 = z[ok], r2[ok]
    r = np.sqrt(r2)

    # upper half traced outwards, lower half traced back: one closed path
    z_closed = np.concatenate([z, z[::-1]])
    r_closed = np.concatenate([r, -r[::-1]])
    return z_closed, r_closed


def draw_panel(ax, no, ni, zo, zi, label, rho_max):
    for sign, style, width, name in (
        (-1, "-",  1.1, "minus sign"),
        (+1, "--", 1.0, "plus sign"),
    ):
        z, r = branch(no, ni, zo, zi, sign, rho_max)
        if z.size:
            ax.plot(z, r, style, color="black", lw=width, label=name)

    ax.plot([zo, zi], [0.0, 0.0], "o", color="black", ms=3.2, zorder=5)

    # axes drawn through the origin, in the style of the book
    ax.axhline(0.0, color="0.35", lw=0.5, zorder=0)
    ax.axvline(0.0, color="0.35", lw=0.5, zorder=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    ax.set_title(f"({label})", fontsize=10, pad=8)
    ax.set_xlabel("$z$", fontsize=9, labelpad=1)
    ax.set_ylabel("$r$", fontsize=9, rotation=0, labelpad=6)
    ax.tick_params(length=2.5, width=0.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=6.2, loc="upper right", framealpha=1.0,
              borderpad=0.3, handlelength=1.6, borderaxespad=0.2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output", default="CartesianSurfaces.pdf")
    p.add_argument("--ni", type=float, default=1.7)
    p.add_argument("--no", type=float, default=1.0)
    args = p.parse_args()

    #        zo    zi    label  rho_max
    cases = [(-10,  20,  "a",   300),
             (-10, -20,  "b",   300),
             ( 10,  20,  "c",   300),
             ( 10, -20,  "d",   300)]

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.0))
    for ax, (zo, zi, lab, rmax) in zip(axes, cases):
        draw_panel(ax, args.no, args.ni, zo, zi, lab, rmax)

    fig.tight_layout(pad=0.6, w_pad=1.6)
    fig.savefig(args.output)          # vector PDF
    print(f"wrote {args.output}")

    for zo, zi, lab, _ in cases:
        G, O, T, S = form_parameters(args.no, args.ni, zo, zi)
        print(f"  ({lab}) zo={zo:+4d} zi={zi:+4d}  "
              f"G={G:+9.5f} O={O:+9.5f} T={T:+11.3e} S={S:+11.3e}  "
              f"S^2-GOT={S**2 - G*O*T:+.1e}")


if __name__ == "__main__":
    main()
