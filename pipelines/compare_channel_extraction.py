#!/usr/bin/env python3
"""
compare_channel_extraction.py
================================

Diagnostic for a uniform flux offset between abi_reextract_pfpc.py's output
and PFPC's own extraction. Compares three things for ONE channel/band, side
by side, to isolate where the offset is introduced:

  1. PFPC's own x1d.fits for that channel/band (the reference).
  2. extract_spectrum_manual() run DIRECTLY on PFPC's un-subtracted Level-3
     cube, bypassing AstroBkgInterp entirely. If this alone is already off
     from (1) by the same factor, the discrepancy has nothing to do with
     AstroBkgInterp -- it's in how extract_spectrum_manual() is configured
     (aperture correction, source position, srctype) relative to whatever
     PFPC's own Spec3 Extract1dStep call used.
  3. abi_reextract_pfpc.py's actual output x1d for that channel/band (the
     AstroBkgInterp-subtracted extraction).

If (2) and (3) agree with each other but not with (1): the offset is in
extract_spectrum_manual()'s settings, not AstroBkgInterp.
If (2) matches (1) but (3) doesn't: AstroBkgInterp's subtraction is the
actual problem.
If both (2) and (3) match (1): the offset is downstream, in
combine_epoch_spectrum()'s overlap rescaling -- check its printed
scale_log for this epoch instead.

Usage
-----
    python compare_channel_extraction.py \\
        --pfpc-x1d SN2024aecx_2025-06-01_uncal/SN2024aecx_uncal_level3_ch1-short_x1d.fits \\
        --pfpc-cube SN2024aecx_2025-06-01_uncal/SN2024aecx_uncal_level3_ch1-short_s3d.fits \\
        --abi-x1d SN2024aecx_2025-06-01_abi/SN2024aecx_2025-06-01_ch1-short_abi_x1d.fits \\
        --plot compare_ch1_short.png
"""
import argparse

import numpy as np
from astropy.io import fits

import pipelines.miri_mrs_auto_pipeline as mp


def read_x1d_simple(path):
    with fits.open(path) as h:
        tab = h["EXTRACT1D"].data
    wave = np.asarray(tab["WAVELENGTH"], dtype=float)
    flux = np.asarray(tab["FLUX"], dtype=float)
    order = np.argsort(wave)
    return wave[order], flux[order]


def median_ratio(wave_ref, flux_ref, wave_test, flux_test):
    """Median of flux_test / flux_ref, both interpolated onto wave_test,
    restricted to their shared wavelength range."""
    lo = max(wave_ref.min(), wave_test.min())
    hi = min(wave_ref.max(), wave_test.max())
    f_ref_i = np.interp(wave_test, wave_ref, flux_ref)
    m = ((wave_test >= lo) & (wave_test <= hi) & np.isfinite(flux_test)
         & np.isfinite(f_ref_i) & (f_ref_i != 0))
    if not m.any():
        return np.nan
    return float(np.median(flux_test[m] / f_ref_i[m]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pfpc-x1d", required=True,
                     help="PFPC's own x1d.fits for this channel/band (the reference)")
    ap.add_argument("--pfpc-cube", required=True,
                     help="PFPC's own un-subtracted s3d.fits for this channel/band")
    ap.add_argument("--abi-x1d", required=True,
                     help="abi_reextract_pfpc.py's output x1d for this channel/band")
    ap.add_argument("--ra", type=float, default=None,
                     help="Source RA (deg), to match what abi_reextract_pfpc.py used. "
                          "If omitted, auto-centroids the PFPC cube directly.")
    ap.add_argument("--dec", type=float, default=None)
    ap.add_argument("--plot", default=None, help="Optional PNG comparing all three")
    args = ap.parse_args()

    wave_ref, flux_ref = read_x1d_simple(args.pfpc_x1d)
    print(f"(1) PFPC's own extraction: {args.pfpc_x1d}")
    print(f"    {wave_ref.min():.2f}-{wave_ref.max():.2f} um, "
          f"median flux {np.nanmedian(flux_ref):.4g}")

    if args.ra is not None and args.dec is not None:
        src_x, src_y = mp.resolve_source_pixel(args.pfpc_cube, ra=args.ra, dec=args.dec)
    else:
        centroid = mp.find_source_centroid(args.pfpc_cube)
        src_x, src_y = centroid["x"], centroid["y"]
    print(f"    re-extracting directly from PFPC's OWN un-subtracted cube at "
          f"({src_x:.2f}, {src_y:.2f}) -- no AstroBkgInterp involved...")

    wave_direct, flux_direct, err_direct, _, meta = mp.extract_spectrum_manual(
        args.pfpc_cube, x=src_x, y=src_y, return_meta=True)
    ratio_direct = median_ratio(wave_ref, flux_ref, wave_direct, flux_direct)
    print(f"(2) extract_spectrum_manual() on PFPC's own cube (no AstroBkgInterp): "
          f"median flux {np.nanmedian(flux_direct):.4g}, ratio to (1) = {ratio_direct:.3f}")

    wave_abi, flux_abi = read_x1d_simple(args.abi_x1d)
    ratio_abi = median_ratio(wave_ref, flux_ref, wave_abi, flux_abi)
    print(f"(3) abi_reextract_pfpc.py's output (AstroBkgInterp + extraction): "
          f"median flux {np.nanmedian(flux_abi):.4g}, ratio to (1) = {ratio_abi:.3f}")

    print()
    close = lambda r: np.isfinite(r) and abs(r - 1) <= 0.3
    if not close(ratio_direct) and not close(ratio_abi):
        print("Both (2) and (3) disagree with PFPC's own extraction -- the offset is in "
              "extract_spectrum_manual()'s configuration (aperture correction, source "
              "position, or srctype), not in AstroBkgInterp. Compare its apply_apcorr, "
              "srctype and center_xy settings against whatever PFPC's own Spec3 "
              "Extract1dStep call used.")
    elif close(ratio_direct) and not close(ratio_abi):
        print("(2) matches PFPC's own extraction but (3) doesn't -- the offset is "
              "introduced by AstroBkgInterp's background subtraction itself (check the "
              "source mask size / fwhm_scale, and whether the background model is "
              "removing real source flux or adding a spurious pedestal).")
    elif not close(ratio_direct) and close(ratio_abi):
        print("(2) disagrees with PFPC but (3) doesn't -- inconsistent; double check both "
              "runs used the exact same source position (x, y).")
    else:
        print("Both (2) and (3) agree with PFPC's own extraction here -- if the *combined* "
              "spectrum still shows the offset, it's downstream, in "
              "combine_epoch_spectrum()'s overlap rescaling. Check its printed scale_log "
              "for this epoch for a factor far from 1.0.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(wave_ref, flux_ref, label="(1) PFPC's own x1d", lw=1.5)
        ax.plot(wave_direct, flux_direct, label="(2) direct re-extract, no ABI", lw=1)
        ax.plot(wave_abi, flux_abi, label="(3) abi_reextract_pfpc.py output", lw=1)
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel("Flux")
        ax.legend()
        plt.tight_layout()
        plt.savefig(args.plot)
        plt.show()
        print(f"\nSaved comparison plot: {args.plot}")


if __name__ == "__main__":
    main()