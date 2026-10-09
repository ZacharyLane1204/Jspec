#!/usr/bin/env python3
"""
astrobkginterp_thermal.py
==========================

Thermal-noise-only background correction for JWST imaging, built on top of
AstroBkgInterp (https://github.com/brynickson/AstroBkgInterp).

WHY THIS EXISTS
----------------
AstroBkgInterp fits a background model (in `bkg_mode="polynomial"`) by
sliding a square tile of size `bin_size` across the source-masked image,
fitting a 2D polynomial of order `k` inside each tile, and median-combining
all the overlapping fits. Whatever that fitted surface contains is what
gets *subtracted*. `bin_size` sets the smallest spatial scale the model is
allowed to vary on:

    - small bin_size  -> the background model can (and will) follow
      small-scale / pixel-to-pixel structure, e.g. thermal noise, faint
      striping, local drift.
    - large bin_size  -> the background model can only capture smooth
      trends across large regions -- real large-scale structure (e.g.
      diffuse emission, a scattered-light gradient) you want to keep.

Used on its own, AstroBkgInterp has no "keep the large-scale stuff, only
kill the noise" mode -- subtracting its fit always removes everything at
and above that fit's spatial resolution.

THE FIX: a two-scale decomposition
-----------------------------------
We run AstroBkgInterp *twice* on the same (source-masked) image:

  1. COARSE pass -- large `bin_size`, low polynomial order, low
     `cube_resolution`. This is deliberately too coarse to represent
     thermal noise; it converges on the large-scale structure you want
     preserved. Call this `bkg_coarse`.

  2. FINE pass -- small `bin_size`. This tracks everything down to the
     thermal-noise scale (including the large-scale trend, since a fine
     fit reproduces coarse structure too). Call this `bkg_fine`.

The thermal-noise-only component is the *difference*:

    noise_map = bkg_fine - bkg_coarse

and the corrected image is

    corrected = data - noise_map
              = (data - bkg_fine) + bkg_coarse

i.e. "fully background-subtract the image, then add the large-scale
structure back in." This removes only the part of the background that
varies on scales finer than `bin_size_coarse`, and leaves everything
smoother than that untouched.

USAGE
-----
As a library:

    from astrobkginterp_thermal import correct_thermal_background
    corrected, diag = correct_thermal_background(
        data, src_x=512, src_y=512, err=err_array,
    )

On a JWST rate/cal file (single SCI extension, 2D):

    from astrobkginterp_thermal import apply_to_fits_file
    apply_to_fits_file("jw01040001001_02101_00001_mirimage_rate.fits",
                        outdir="thermal_corrected")

To mask a specific sky position instead of the header's TARG_RA/TARG_DEC
(e.g. a source embedded in extended structure, or a source other than the
nominal proposal target), pass src_ra/src_dec in decimal degrees:

    apply_to_fits_file("jw01040001001_02101_00001_mirimage_rate.fits",
                        outdir="thermal_corrected",
                        src_ra=83.8221, src_dec=-5.3911)

If neither pixel coordinates nor RA/Dec are given, the target's pixel
position is derived from the file's WCS and the TARG_RA/TARG_DEC header
keywords, falling back to the image center if those are absent.

CAVEAT -- read before trusting this on a crowded/extended field
-----------------------------------------------------------------
AstroBkgInterp's source-masking only masks *one* aperture (optionally
elliptical), dithered by +/-1 pixel, centered on a single (src_x, src_y).
It was built for images dominated by one (or a few, isolated) sources on
top of a complex background -- e.g. exoplanet/disk imaging, coronagraphy,
isolated point-source photometry -- not for automatically masking many
unrelated sources scattered across a deep mosaic field. If your MIRI
field has multiple science targets or a crowded background, this
correction will only mask the one source you tell it about; other
sources in the field will be (incorrectly) treated as background and can
bias the fit. Inspect `bkg_coarse`/`noise_map` on at least one exposure
before trusting this on a whole proposal.

If your target itself sits inside real extended/large-scale structure
(a nebula, disk, diffuse emission, etc.), make sure `aper_rad`/`ann_width`
are large enough to clear the source's visible extent -- otherwise the
annulus used to estimate "background" is actually still the source's
wings, and that gets partially interpolated back into the aperture. Also
make sure `bin_size_fine` is smaller than the smallest real structure you
want to preserve (or that structure will get partially treated as noise
and removed) but larger than the spatial scale of the noise you're
targeting -- if those two scales aren't well separated, this two-pass
decomposition can't cleanly split them.
"""

import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from AstroBkgInterp import AstroBkgInterp


# ---------------------------------------------------------------------------
# Core two-scale correction
# ---------------------------------------------------------------------------

def _make_abi(src_x, src_y, bin_size, k, cube_resolution, aper_rad, ann_width,
              mask_type, semi_major, semi_minor, angle, kernel, pool_size):
    abi = AstroBkgInterp()
    abi.src_x, abi.src_y = src_x, src_y
    abi.bkg_mode = "polynomial"
    abi.k = k
    abi.bin_size = bin_size
    abi.cube_resolution = cube_resolution
    abi.aper_rad = aper_rad
    abi.ann_width = ann_width
    abi.mask_type = mask_type
    abi.semi_major = semi_major
    abi.semi_minor = semi_minor
    abi.angle = angle
    abi.kernel = kernel
    abi.pool_size = pool_size
    abi.combine_fits = False
    return abi


def correct_thermal_background(
    data,
    src_x,
    src_y,
    err=None,
    bin_size_coarse=None,
    bin_size_fine=25,
    k_coarse=2,
    k_fine=1,
    cube_resolution_coarse="low",
    cube_resolution_fine="high",
    aper_rad=5,
    ann_width=5,
    mask_type="circular",
    semi_major=6,
    semi_minor=4,
    angle=0,
    kernel=None,
    pool_size=1,
    verbose=True,
):
    """Remove only the small-scale (thermal-noise-scale) background,
    preserving structure larger than `bin_size_coarse`.

    Parameters
    ----------
    data : 2D ndarray
        Science image (e.g. the SCI extension of a _rate.fits file).
    src_x, src_y : int
        Pixel coordinates of the source to mask while fitting the
        background (AstroBkgInterp masks exactly one source region).
    err : 2D ndarray, optional
        Uncertainty array matching `data`.
    bin_size_coarse : int, optional
        Tile size (pixels) for the large-scale pass. Larger = more of the
        background structure is preserved (fewer/larger tiles => coarser,
        smoother model). Default: ~1/3 of the smaller image dimension.
    bin_size_fine : int, optional
        Tile size (pixels) for the small-scale pass. This should be
        comparable to the spatial scale of the thermal noise/striping you
        want removed -- smaller values chase finer structure. Default 25.
    k_coarse, k_fine : int
        Polynomial order for each pass. Keep `k_fine` low (1-2) so the
        fine pass models local level/slope rather than trying to fit
        sources that slipped past the mask.
    cube_resolution_coarse, cube_resolution_fine : {"low","medium","high"}
        Sliding-window sampling density for each pass (see AstroBkgInterp
        docs). "high" is slow but most accurate; "low" is fine for the
        coarse pass since it's deliberately a smooth fit.
    aper_rad, ann_width, mask_type, semi_major, semi_minor, angle :
        Passed through to AstroBkgInterp's source-masking (see its
        docstring). Only one source is masked.
    kernel : ndarray or None
        Optional smoothing kernel applied to the source-masked data
        before fitting (see AstroBkgInterp).
    pool_size : int
        Multiprocessing pool size (only used for 3D/cube inputs in
        AstroBkgInterp; harmless to leave at 1 for 2D images).

    Returns
    -------
    corrected : 2D ndarray
        `data` with the small-scale (thermal-noise) background removed.
    diagnostics : dict
        {"bkg_coarse", "bkg_fine", "noise_map", "err_fine"} -- inspect
        these (especially `noise_map`) before trusting the correction.
    """
    ny, nx = data.shape
    if bin_size_coarse is None:
        bin_size_coarse = max(int(min(ny, nx) / 3), bin_size_fine + 5)

    if bin_size_coarse <= bin_size_fine:
        raise ValueError(
            f"bin_size_coarse ({bin_size_coarse}) must be larger than "
            f"bin_size_fine ({bin_size_fine}) -- the coarse pass has to "
            f"be coarser than the fine pass, or the decomposition is "
            f"meaningless."
        )

    if verbose:
        print(f"[thermal-bkg] coarse pass: bin_size={bin_size_coarse}, "
              f"k={k_coarse}, resolution={cube_resolution_coarse}")
    abi_coarse = _make_abi(src_x, src_y, bin_size_coarse, k_coarse,
                            cube_resolution_coarse, aper_rad, ann_width,
                            mask_type, semi_major, semi_minor, angle,
                            kernel, pool_size)
    _, bkg_coarse, _, _ = abi_coarse.run(data, err=err)

    if verbose:
        print(f"[thermal-bkg] fine pass:   bin_size={bin_size_fine}, "
              f"k={k_fine}, resolution={cube_resolution_fine}")
    abi_fine = _make_abi(src_x, src_y, bin_size_fine, k_fine,
                          cube_resolution_fine, aper_rad, ann_width,
                          mask_type, semi_major, semi_minor, angle,
                          kernel, pool_size)
    _, bkg_fine, _, err_fine = abi_fine.run(data, err=err)

    noise_map = bkg_fine - bkg_coarse
    corrected = data - noise_map

    diagnostics = {
        "bkg_coarse": bkg_coarse,
        "bkg_fine": bkg_fine,
        "noise_map": noise_map,
        "err_fine": err_fine,
    }
    return corrected, diagnostics


# ---------------------------------------------------------------------------
# FITS convenience wrapper
# ---------------------------------------------------------------------------

def radec_to_pixel(hdul, ra, dec):
    """Convert a source RA/Dec (deg) to pixel coordinates using the SCI
    extension's WCS. Raises ValueError if the position falls outside the
    image."""
    sci_hdr = hdul["SCI"].header
    ny, nx = hdul["SCI"].data.shape
    w = WCS(sci_hdr)
    x, y = w.all_world2pix(ra, dec, 0)
    if not (np.isfinite(x) and np.isfinite(y)):
        raise ValueError(f"WCS conversion of RA={ra}, Dec={dec} produced a "
                          f"non-finite pixel position -- check the coordinates "
                          f"and that this exposure actually covers that sky position.")
    if not (0 <= x < nx and 0 <= y < ny):
        raise ValueError(f"RA={ra}, Dec={dec} -> pixel ({x:.1f}, {y:.1f}) falls "
                          f"outside this exposure ({nx}x{ny}); it isn't covered "
                          f"by this file's footprint.")
    return int(round(float(x))), int(round(float(y)))


def get_source_pixel_coords(hdulist):
    """Derive (src_x, src_y) from TARG_RA/TARG_DEC + the SCI WCS, falling
    back to the image center if unavailable."""
    hdr0 = hdulist[0].header

    ra = hdr0.get("TARG_RA")
    dec = hdr0.get("TARG_DEC")
    if ra is not None and dec is not None:
        try:
            return radec_to_pixel(hdulist, ra, dec)
        except Exception as exc:
            print(f"[thermal-bkg] WCS lookup failed ({exc}); "
                  f"falling back to image center.")

    ny, nx = hdulist["SCI"].data.shape
    print("[thermal-bkg] no usable TARG_RA/TARG_DEC/WCS; "
          "using image center as source position.")
    return nx // 2, ny // 2


def apply_to_fits_file(fname, outdir, src_x=None, src_y=None, src_ra=None, src_dec=None,
                        suffix="_thermcorr", save_diagnostics=False, **kwargs):
    """Run `correct_thermal_background` on a rate/cal FITS file's SCI
    extension and write the corrected file to `outdir`.

    Parameters
    ----------
    fname : str
        Path to the input FITS file (must have a SCI extension; ERR is
        used if present).
    outdir : str
        Directory to write the corrected file into.
    src_x, src_y : int, optional
        Source pixel position, if you already know it.
    src_ra, src_dec : float, optional
        Source position in decimal degrees (ICRS). Takes precedence over
        src_x/src_y if both are given, and over the file's own
        TARG_RA/TARG_DEC. Converted to this exposure's pixel frame via its
        WCS -- use this when you want to mask a specific sky position
        (e.g. a source embedded in extended structure, or one that isn't
        the nominal proposal target) rather than trusting the header.
    suffix : str
        Appended (before the extension) to the output filename.
    save_diagnostics : bool
        If True, also write a `<name>_thermdiag.fits` file with
        bkg_coarse / bkg_fine / noise_map as extra extensions, so you can
        visually confirm the split before trusting it.
    **kwargs
        Passed through to `correct_thermal_background`.

    Returns
    -------
    outpath : str
        Path to the corrected FITS file.
    """
    os.makedirs(outdir, exist_ok=True)

    with fits.open(fname) as hdul:
        data = hdul["SCI"].data.astype(float)
        err = hdul["ERR"].data.astype(float) if "ERR" in hdul else None

        if src_ra is not None and src_dec is not None:
            src_x, src_y = radec_to_pixel(hdul, src_ra, src_dec)
        elif src_x is None or src_y is None:
            src_x, src_y = get_source_pixel_coords(hdul)

        corrected, diag = correct_thermal_background(
            data, src_x, src_y, err=err, **kwargs
        )

        hdul["SCI"].data = corrected.astype(hdul["SCI"].data.dtype)

        base = os.path.basename(fname)
        stem, ext = os.path.splitext(base)
        outpath = os.path.join(outdir, f"{stem}{suffix}{ext}")
        hdul.writeto(outpath, overwrite=True)

        if save_diagnostics:
            diagpath = os.path.join(outdir, f"{stem}{suffix}_diag.fits")
            fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(diag["bkg_coarse"], name="BKG_COARSE"),
                fits.ImageHDU(diag["bkg_fine"], name="BKG_FINE"),
                fits.ImageHDU(diag["noise_map"], name="NOISE_MAP"),
            ]).writeto(diagpath, overwrite=True)

    print(f"[thermal-bkg] {base} -> {os.path.basename(outpath)} "
          f"(source at x={src_x}, y={src_y})")
    return outpath