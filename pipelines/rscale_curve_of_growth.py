#!/usr/bin/env python3
"""
rscale_curve_of_growth.py
==========================

Re-run Extract1dStep on an ALREADY background-subtracted MIRI MRS cube
(the "*_bgsub_s3d.fits" that miri_mrs_refine_extract.py already produced)
at several ifu_rscale values, and report the flux near 5 um at each.

Why: a real point source's flux should climb steeply at small radius, then
flatten once the aperture correction (apply_apcorr=True) catches the rest
of the encircled energy -- the classic curve of growth. If the flux instead
keeps climbing roughly linearly with aperture area well past that point,
that's the signature of extended, spatially structured light in the
aperture (host continuum, an HII region, a dust clump) rather than pure
point-source contamination -- exactly the "different aperture picks up a
different amount of a nearby real feature" idea, just without requiring a
literal bound companion.

Run this on both a bright epoch (June) and the faint one (December) and
compare the two curves -- a point source should look similar in *shape*
(after normalizing to e.g. the rscale=1.25 value) regardless of epoch; a
growing excess at large radius specifically in the faint epoch is the
extended-light signature.

Usage
-----
    python rscale_curve_of_growth.py \\
        --bgsub-cube refined/28122025_abi_f5_rfix/SN2024aecx_28122025_abi_f5_rfix_bgsub_s3d.fits \\
        --center 45.674 50.238 --wave-range 4.9 5.2 \\
        --rscale-grid 1.25 1.5 1.75 2.0 2.5 3.0

    python rscale_curve_of_growth.py \\
        --bgsub-cube refined/01062025/SN2024aecx_01062025_bgsub_s3d.fits \\
        --center <June x> <June y> --wave-range 4.9 5.2 \\
        --rscale-grid 1.25 1.5 1.75 2.0 2.5 3.0
"""
import argparse

import numpy as np
from jwst.extract_1d import Extract1dStep
from jwst.datamodels import IFUCubeModel
from astropy.table import Table


def extract_at_rscale(cube_path, center_xy, rscale):
    model = IFUCubeModel(cube_path)
    step = Extract1dStep()
    step.extraction_type = "box"
    step.ifu_set_srctype = "POINT"
    step.ifu_autocen = True
    step.center_xy = list(center_xy)
    step.ifu_rscale = float(rscale)
    step.ifu_rfcorr = True
    step.apply_apcorr = True
    step.subtract_background = False
    step.save_results = False
    result = step.run(model)
    # result is a MultiSpecModel-like; grab the first spectrum table
    spec = result.spec[0].spec_table if hasattr(result, "spec") else result[0].spec_table
    return Table(spec)


def band_median_flux(tbl, wave_range):
    w = np.asarray(tbl["WAVELENGTH"], float)
    f = np.asarray(tbl["FLUX"], float)
    m = (w >= wave_range[0]) & (w <= wave_range[1]) & np.isfinite(f)
    if m.sum() == 0:
        return np.nan
    return float(np.nanmedian(f[m]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bgsub-cube", required=True, help="the *_bgsub_s3d.fits already produced for this epoch")
    ap.add_argument("--center", nargs=2, type=float, required=True, metavar=("X", "Y"))
    ap.add_argument("--wave-range", nargs=2, type=float, default=[4.9, 5.2], metavar=("LO", "HI"))
    ap.add_argument("--rscale-grid", nargs="+", type=float, default=[1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
    a = ap.parse_args()

    print(f"{'rscale':>8} {'median flux (Jy)':>18} {'ratio to rscale[0]':>20}")
    fluxes = []
    for r in a.rscale_grid:
        tbl = extract_at_rscale(a.bgsub_cube, tuple(a.center), r)
        f = band_median_flux(tbl, tuple(a.wave_range))
        fluxes.append(f)
        ratio = f / fluxes[0] if fluxes[0] else np.nan
        print(f"{r:8.2f} {f:18.6g} {ratio:20.3f}")

    print()
    print("A point source (with apcorr already applied) should flatten to ~1.0-1.1x by "
          "the middle of this grid. Still climbing steadily at the largest radii "
          "indicates extended flux in the aperture, not just point-source PSF wings.")