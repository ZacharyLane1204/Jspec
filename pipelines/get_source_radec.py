#!/usr/bin/env python3
"""
get_source_radec.py
=====================

Derive a fixed sky position (RA, Dec) for a target from a channel/band cube
where auto-centroiding is known to be reliable -- e.g. ch1-short, which has
the largest MRS field of view and the best S/N for a source that's much
fainter by ch4-long. Use that RA/Dec as --ra/--dec in abi_reextract_pfpc.py
and compare_channel_extraction.py for EVERY channel and epoch, instead of
letting each channel auto-centroid independently. This avoids the failure
mode seen at ch4-long: photutils detection failing on a faint,
thermal-background-dominated slice and silently falling back to the
brightest nearby (wrong) source.

A fixed sky position resolves to a different PIXEL position in each
channel's own cube (different pixel scale, different ifualign grid), which
is exactly what resolve_source_pixel()/extract_spectrum_manual() already
handle via WCS -- so this is a drop-in replacement for per-channel
auto-centroiding, not a workaround.

Usage
-----
    # Auto-centroid the given cube and report its sky position:
    python get_source_radec.py --cube SN2024aecx_2025-06-01_uncal/SN2024aecx_2025-06-01_uncal_level3_ch1-short_s3d.fits

    # Or convert a pixel position you already trust:
    python get_source_radec.py --cube ..._ch1-short_s3d.fits --x 22.87 --y 21.39
"""
import argparse

from astropy.io import fits
from astropy.wcs import WCS

import jspec.pipelines.miri_mrs_auto_pipeline as mp


def pixel_to_radec(cube_file, x, y):
    """Inverse of miri_mrs_auto_pipeline.resolve_source_pixel: pixel (x, y)
    in `cube_file`'s spatial frame -> (ra, dec) in degrees, using the same
    full-WCS (all_pix2world, SIP/distortion-aware) approach."""
    with fits.open(cube_file) as hdul:
        wcs2d = WCS(hdul["SCI"].header, fobj=hdul, naxis=2)
    ra, dec = wcs2d.all_pix2world(x, y, 0)
    return float(ra), float(dec)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", required=True,
                     help="A channel/band cube where you trust the source position "
                          "(ideally one with a clean single detection, e.g. ch1-short)")
    ap.add_argument("--x", type=float, default=None,
                     help="Pixel x to convert directly, skipping auto-centroiding")
    ap.add_argument("--y", type=float, default=None)
    args = ap.parse_args()

    if (args.x is None) != (args.y is None):
        raise SystemExit("Give both --x and --y, or neither.")

    if args.x is not None:
        x, y = args.x, args.y
        print(f"Using supplied pixel position ({x:.3f}, {y:.3f})")
    else:
        centroid = mp.find_source_centroid(args.cube)
        x, y = centroid["x"], centroid["y"]
        print(f"Auto-centroid on {args.cube}: ({x:.3f}, {y:.3f}) "
              f"+/- ({centroid['x_err']:.3f}, {centroid['y_err']:.3f}), "
              f"SNR={centroid['snr']:.1f}")
        if max(centroid["x_scatter"], centroid["y_scatter"]) > 1.0:
            print("  WARNING: sub-band positions disagree by more than a pixel -- "
                  "double check this centroid before trusting it as your fixed "
                  "reference position (pick a different / brighter cube instead).")

    ra, dec = pixel_to_radec(args.cube, x, y)
    print(f"\nRA, Dec = {ra:.6f}, {dec:.6f}")
    print("\nUse this everywhere:")
    print(f"  --ra {ra:.6f} --dec {dec:.6f}")


if __name__ == "__main__":
    main()