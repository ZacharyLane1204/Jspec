#!/usr/bin/env python3
"""
check_source_position_consistency.py
=====================================

Convert the "adopted" MIRI MRS extraction position (from each epoch's
<name>_report.json, position.adopted = [x, y]) to sky coordinates using
that epoch's own Stage-3 cube WCS, and report the angular separation
between epochs. A fixed target should land on the same RA/Dec every
epoch (to well under an arcsec); a jump of an arcsec or more means the
centroiding locked onto a different spot in the field that epoch (a host
knot, a bad region, etc.), not the same physical source.

Usage
-----
    python check_source_position_consistency.py \\
        --epoch Jan=report:refined/17012025/SN2024aecx_17012025_report.json,cube:.../stage3/..._s3d.fits \\
        --epoch Jun=report:refined/01062025/SN2024aecx_01062025_report.json,cube:.../stage3/..._s3d.fits \\
        --epoch Dec=report:refined/28122025/SN2024aecx_28122025_report.json,cube:.../stage3/..._s3d.fits

Notes
-----
- Uses the SCI extension header's WCS, dropped to the celestial (RA/Dec)
  axes only, with `origin=0` (adopted x,y are 0-based array indices, as
  used throughout miri_mrs_refine_extract.py).
- If the report's `position.adopted` list is [x, y] in the *cube* frame
  used at extraction time, this directly answers "same sky position?".
  If your cubes were reprojected to a shared grid before this stage, the
  raw pixel values alone would already have been comparable -- run this
  anyway; the RA/Dec should agree either way if the source is genuinely
  the same object every epoch.
"""
import argparse
import json

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u


def adopted_pix(report_path):
    rep = json.load(open(report_path))
    xy = rep.get("position", {}).get("adopted")
    if xy is None:
        raise KeyError(f"no position.adopted in {report_path}")
    return float(xy[0]), float(xy[1])


def pix_to_sky(cube_path, x, y):
    with fits.open(cube_path) as h:
        hdr = h["SCI"].header if "SCI" in h else h[1].header
        # MIRI MRS cubes use a WAVE-TAB (lookup-table) wavelength axis; astropy
        # needs the full open HDUList (fobj=h) to resolve that table extension,
        # even though we only want the celestial (RA/Dec) axes out of it.
        w = WCS(hdr, fobj=h).celestial
        ra, dec = w.wcs_pix2world(x, y, 0)
    return SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg)


def parse_epoch_arg(item):
    label, rest = item.split("=", 1)
    parts = dict(p.split(":", 1) for p in rest.split(","))
    return label, parts["report"], parts["cube"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epoch", action="append", required=True,
                    help="LABEL=report:PATH,cube:PATH  (repeatable)")
    a = ap.parse_args()

    coords = []
    for item in a.epoch:
        label, report_path, cube_path = parse_epoch_arg(item)
        x, y = adopted_pix(report_path)
        sc = pix_to_sky(cube_path, x, y)
        coords.append((label, x, y, sc))
        print(f"{label}: pixel ({x:.2f}, {y:.2f}) -> RA={sc.ra.deg:.6f} deg, Dec={sc.dec.deg:.6f} deg")

    print()
    ref_label, _, _, ref_sc = coords[0]
    for label, x, y, sc in coords[1:]:
        sep = sc.separation(ref_sc)
        print(f"{label} vs {ref_label}: {sep.arcsec:.3f} arcsec "
              f"({'consistent with the same source' if sep.arcsec < 0.3 else 'LIKELY A DIFFERENT POSITION'})")