#!/usr/bin/env python3
"""
kde_bkg_correct.py
====================

FITS convenience wrapper around `background_subtraction.determine_background`
(automatic star + nebula/galaxy masking, tiled SExtractor-style background
estimation with KDE-mode rescue for heavily masked tiles) for use on JWST
rate/cal files.

WHY THIS IS A BETTER FIT THAN AstroBkgInterp FOR A TARGET IN A GALAXY
------------------------------------------------------------------------
AstroBkgInterp (see astrobkginterp_thermal.py) masks a single aperture
around one named source position. If the real target -- a supernova, say
-- sits inside a host galaxy that fills a meaningful fraction of the
frame, that single small aperture leaves most of the galaxy's real light
unmasked, so it gets absorbed into the fitted "background" and then
subtracted out. That's exactly the failure mode this module is built to
avoid: `build_source_and_nebula_masks` detects extended emission
automatically (via multi-scale contrast tests, plus a large-scale excess
test for objects too big to show local contrast), grows the mask outward
until the object's own flux actually reaches the noise floor (avoiding a
faint over-subtracted ring at the mask edge), and only then hands the
masked image to a standard tiled background estimator -- falling back to
a KDE-mode estimate, tile by tile, wherever too little clean sky survives
the masking for the standard estimator to be trusted.

IMPORTANT: run this on individual Stage-2 (cal) or Stage-1 (rate)
exposures, BEFORE Stage-3 (Image3) combination -- not on an already
drizzled/combined mosaic (i2d). A combined mosaic has a ragged footprint
and non-uniform coverage near its edges (dither/overlap seams), which
this module's extended-emission tests can partially misread as
"nebulosity" alongside the real target. Individual exposures have a
clean rectangular footprint and uniform coverage, which is what this
approach assumes.

SCALE NOTE: `background_subtraction.py`'s docstring says it was built for
a ~25 arcmin FOV at 0.72"/px. MIRI imaging has a much smaller field
(roughly 74"x113" for the F1130W-and-shorter filters) at ~0.11"/px, so
its frames are typically a few hundred to ~1000+ pixels on a side --
broadly comparable in *pixel count* to a fraction of the reference wide
field, so the pixel-based defaults (box_size=100, etc.) are a reasonable
starting point, but not guaranteed optimal. Start with the defaults,
check the diagnostic plots (`save_diagnostics=True`) on one exposure, and
scale `box_size` down if the galaxy fills most of the frame and few
tiles are getting a reliable direct estimate (many/most needing KDE
rescue isn't itself a problem -- that's what the rescue path is for --
but if *every* tile needs rescuing there may be no clean sky left at all
in that exposure, which no spatial background estimator can fix).

USAGE
-----
    from kde_bkg_correct import apply_to_fits_file
    apply_to_fits_file("jw01040001001_02101_00001_mirimage_rate.fits",
                        outdir="kde_corrected", box_size=80,
                        save_diagnostics=True)
"""

import contextlib
import os

from astropy.io import fits

from jspec.pipelines.background_subtraction import determine_background


@contextlib.contextmanager
def _chdir(path):
    """determine_background's diagnostic plots are written with hardcoded
    filenames to the current working directory; temporarily chdir into
    `path` so they land next to the corrected FITS file instead of
    wherever this script happened to be run from."""
    prev = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def apply_to_fits_file(fname, outdir, suffix="_kdebkg", save_diagnostics=False,
                        **kwargs):
    """Run `determine_background` on a rate/cal FITS file's SCI extension
    and write the corrected file to `outdir`.

    Parameters
    ----------
    fname : str
        Path to the input FITS file (must have a SCI extension).
    outdir : str
        Directory to write the corrected file (and, if requested,
        diagnostic plots) into.
    suffix : str
        Appended (before the extension) to the output filename.
    save_diagnostics : bool
        If True, save `determine_background`'s diagnostic PNGs (data,
        star mask, nebula mask, source mask, final background surface)
        into `outdir`, prefixed with this file's stem so multiple
        exposures don't overwrite each other's diagnostics.
    **kwargs
        Passed through to `determine_background` (e.g. box_size,
        min_clean_frac, large_scale_sigma_thresh, verbose, seed).

    Returns
    -------
    outpath : str
        Path to the corrected FITS file.
    """
    os.makedirs(outdir, exist_ok=True)
    base = os.path.basename(fname)
    stem, ext = os.path.splitext(base)

    with fits.open(fname) as hdul:
        data = hdul["SCI"].data.astype(float)

        if save_diagnostics:
            diag_dir = os.path.join(outdir, f"{stem}{suffix}_diag")
            with _chdir(diag_dir):
                background, background_rms = determine_background(
                    data, testing_plot=True, **kwargs
                )
        else:
            background, background_rms = determine_background(data, **kwargs)

        corrected = data - background
        hdul["SCI"].data = corrected.astype(hdul["SCI"].data.dtype)

        outpath = os.path.join(outdir, f"{stem}{suffix}{ext}")
        hdul.writeto(outpath, overwrite=True)

    print(f"[kde-bkg] {base} -> {os.path.basename(outpath)}")
    return outpath