#!/usr/bin/env python3
"""
combine_nirspec_orders.py
==========================

Helper for stitching two or more NIRSpec Fixed-Slit x1d spectra (typically
different grating/filter settings of the *same* slit/target, e.g. G235M +
G395M) into one continuous 1D spectrum, blending the wavelength overlap
region with an inverse-variance weighted average rather than just picking
one spectrum or the other there.

Usage
-----
    from combine_nirspec_orders import combine_overlapping_spectra

    wave, flux, err = combine_overlapping_spectra([
        "f170lp_g235m_s400a1_s400a1_s000000001_x1d.fits",
        "f290lp_g395m_s400a1_s400a1_s000000001_x1d.fits",
    ])

Works for any number of input files -- they're sorted by starting
wavelength and merged pairwise, so e.g. G140M + G235M + G395M all combine
in one call.

Also includes `group_files_by_epoch()`, which clusters a file list by
exposure-start MJD (splitting wherever consecutive files are more than
`max_gap_days` apart). Use it to split a file list into epochs *before*
calling `combine_overlapping_spectra`, so spectra taken on different days
don't get blended into one "final" spectrum.
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time


def read_x1d(path, extname=None, dq_reject=True):
    """Read WAVELENGTH/FLUX/FLUX_ERROR out of a JWST x1d (EXTRACT1D) or c1d
    (COMBINE1D) file, dropping non-finite points and (by default) anything
    flagged bad in DQ. Output is sorted by wavelength.

    extname=None picks whichever of EXTRACT1D / COMBINE1D the file has. The
    error column is FLUX_ERROR in x1d files and ERROR in c1d files."""
    with fits.open(path) as hdul:
        if extname is None:
            names = [h.name.upper() for h in hdul]
            extname = next((n for n in ("EXTRACT1D", "COMBINE1D") if n in names), None)
            if extname is None:
                raise KeyError(f"No EXTRACT1D or COMBINE1D extension in {path}; "
                               f"HDUs: {[h.name for h in hdul]}")
        d = hdul[extname].data
        cols = {n.upper(): n for n in d.columns.names}

        wave = d[cols["WAVELENGTH"]].astype(float)
        flux = d[cols["FLUX"]].astype(float)
        err = d[cols.get("FLUX_ERROR", cols.get("ERROR"))].astype(float)
        dq = d[cols["DQ"]] if "DQ" in cols else np.zeros(len(wave), int)

    good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    if dq_reject:
        good &= (dq == 0)

    wave, flux, err = wave[good], flux[good], err[good]
    order = np.argsort(wave)
    return wave[order], flux[order], err[order]


def read_ecsv_spectrum(path):
    """Read a *final* spectrum saved as ECSV (e.g. the NIRSpec S400A1,
    MIRI LRS and MIRI MRS products), whatever the column capitalisation:
    WAVELENGTH/FLUX/FLUX_ERROR (MRS) or wavelength/flux/flux_err (NIRSpec,
    LRS). Non-finite points and err <= 0 are dropped; output is sorted by
    wavelength (LRS is stored descending in places)."""
    t = Table.read(path, format="ascii.ecsv")
    cols = {n.upper(): n for n in t.colnames}
    err_key = next((k for k in ("FLUX_ERROR", "FLUX_ERR", "ERROR", "ERR")
                    if k in cols), None)
    if err_key is None:
        raise KeyError(f"No error column in {path}; columns: {t.colnames}")

    wave = np.asarray(t[cols["WAVELENGTH"]], dtype=float)
    flux = np.asarray(t[cols["FLUX"]], dtype=float)
    err = np.asarray(t[cols[err_key]], dtype=float)

    good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    wave, flux, err = wave[good], flux[good], err[good]
    order = np.argsort(wave)
    return wave[order], flux[order], err[order]


def read_spectrum(path, dq_reject=True):
    """Read a spectrum from either a JWST x1d/c1d FITS file or a final
    .ecsv spectrum (dispatches on the file extension)."""
    if str(path).lower().endswith(".ecsv"):
        return read_ecsv_spectrum(path)
    return read_x1d(path, dq_reject=dq_reject)


def get_mjd(path):
    """Return the MJD of a file's exposure (MJD-OBS if present, else
    EXPSTART), read from the primary header. Returns None if neither is
    present. For .ecsv files the MJD_MEAN (else MJD_MIN) table-metadata
    keyword is used instead."""
    if str(path).lower().endswith(".ecsv"):
        meta = Table.read(path, format="ascii.ecsv").meta
        for key in ("MJD_MEAN", "MJD_MIN"):
            if meta.get(key) is not None:
                return float(meta[key])
        return None
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        for key in ("MJD-OBS", "EXPSTART"):
            if key in hdr and hdr[key] is not None:
                return float(hdr[key])
    return None


def get_mjd_info(x1d_files):
    """Collect MJD-MIN/MJD-MAX/MJD-MEAN across a set of x1d files (start-of-
    exposure MJD for each file). Returns a dict with None values if no file
    had a usable MJD keyword."""
    mjds = [m for m in (get_mjd(f) for f in x1d_files) if m is not None]
    if not mjds:
        return {"MJD_MIN": None, "MJD_MAX": None, "MJD_MEAN": None}
    return {
        "MJD_MIN": min(mjds),
        "MJD_MAX": max(mjds),
        "MJD_MEAN": float(np.mean(mjds)),
    }


def _merge_pair(wave1, flux1, err1, wave2, flux2, err2, scale_to_overlap=False):
    """Merge two spectra (in wavelength), blending their overlap region
    with an inverse-variance weighted average. Returns (wave, flux, err),
    sorted and with the overlap collapsed onto the union of both grids."""
    # Make spectrum 1 the bluer one.
    if wave1.min() > wave2.min():
        wave1, flux1, err1, wave2, flux2, err2 = wave2, flux2, err2, wave1, flux1, err1

    ov_lo, ov_hi = wave2.min(), wave1.max()

    # No overlap: just concatenate and sort.
    if ov_hi <= ov_lo:
        wave = np.concatenate([wave1, wave2])
        flux = np.concatenate([flux1, flux2])
        err = np.concatenate([err1, err2])
        order = np.argsort(wave)
        return wave[order], flux[order], err[order]

    blue_mask = wave1 < ov_lo
    red_mask = wave2 > ov_hi
    ov1_mask = (wave1 >= ov_lo) & (wave1 <= ov_hi)
    ov2_mask = (wave2 >= ov_lo) & (wave2 <= ov_hi)

    # Optional: correct for a small overall flux offset between the two
    # gratings before blending (uses the median ratio in the overlap).
    if scale_to_overlap:
        f1_ov_med = np.nanmedian(flux1[ov1_mask])
        f2_ov_med = np.nanmedian(flux2[ov2_mask])
        if f2_ov_med != 0:
            scale = f1_ov_med / f2_ov_med
            flux2 = flux2 * scale
            err2 = err2 * abs(scale)

    # Blend on the union of both native grids within the overlap band.
    ov_grid = np.unique(np.concatenate([wave1[ov1_mask], wave2[ov2_mask]]))

    f1_i = np.interp(ov_grid, wave1[ov1_mask], flux1[ov1_mask])
    e1_i = np.interp(ov_grid, wave1[ov1_mask], err1[ov1_mask])
    f2_i = np.interp(ov_grid, wave2[ov2_mask], flux2[ov2_mask])
    e2_i = np.interp(ov_grid, wave2[ov2_mask], err2[ov2_mask])

    w1, w2 = 1.0 / e1_i**2, 1.0 / e2_i**2
    flux_ov = (f1_i * w1 + f2_i * w2) / (w1 + w2)
    err_ov = np.sqrt(1.0 / (w1 + w2))

    wave_out = np.concatenate([wave1[blue_mask], ov_grid, wave2[red_mask]])
    flux_out = np.concatenate([flux1[blue_mask], flux_ov, flux2[red_mask]])
    err_out = np.concatenate([err1[blue_mask], err_ov, err2[red_mask]])

    order = np.argsort(wave_out)
    return wave_out[order], flux_out[order], err_out[order]


def combine_overlapping_spectra(x1d_files, dq_reject=True, scale_to_overlap=False,
                                wave_ranges=None):
    """Read and combine any number of (possibly overlapping) x1d spectra
    into one continuous spectrum, sorted by wavelength.

    Parameters
    ----------
    x1d_files : list of str
        Paths to x1d FITS files (e.g. different gratings of the same slit).
    dq_reject : bool
        Drop pixels with DQ != 0 before combining (default True).
    scale_to_overlap : bool
        If True, rescale each spectrum to match the median flux level of
        the previous one in their shared overlap before blending. Useful
        if there's a systematic flux offset between gratings; leave off
        if the orders already agree (check `check_overlap_offset` below).
    wave_ranges : dict, optional
        {path: (lo, hi)} wavelength cuts (um) applied to individual input
        files *before* combining; either bound may be None. E.g. keep only
        MIRI LRS >= 4.9 um: {lrs_path: (4.9, None)}.

    Returns
    -------
    wave, flux, err : ndarray
        The combined, wavelength-sorted spectrum.
    mjd_info : dict
        {"MJD_MIN", "MJD_MAX", "MJD_MEAN"} spanning all the input files'
        exposure start times (None values if no file had a usable MJD
        keyword). Pass this straight through to `save_combined_spectrum`.
    """
    wave_ranges = wave_ranges or {}
    specs = []
    for f in x1d_files:
        w, fl, e = read_spectrum(f, dq_reject=dq_reject)
        lo, hi = wave_ranges.get(f, (None, None))
        keep = np.ones(len(w), bool)
        if lo is not None:
            keep &= w >= lo
        if hi is not None:
            keep &= w <= hi
        specs.append((w[keep], fl[keep], e[keep]))
    specs = sorted(specs, key=lambda s: s[0].min())  # bluest first

    wave, flux, err = specs[0]
    for w2, f2, e2 in specs[1:]:
        wave, flux, err = _merge_pair(wave, flux, err, w2, f2, e2,
                                       scale_to_overlap=scale_to_overlap)

    # Always enforce ascending wavelength order, even for a single input
    # file -- some extract_1d outputs (MIRI LRS in particular) are stored
    # in descending order, and without this the "combined" spectrum for a
    # single-file case would silently come out unsorted.
    order = np.argsort(wave)
    wave, flux, err = wave[order], flux[order], err[order]

    mjd_info = get_mjd_info(x1d_files)
    return wave, flux, err, mjd_info


def check_overlap_offset(file_blue, file_red, dq_reject=True):
    """Quick diagnostic: report the overlap wavelength range and the
    median flux ratio between two spectra in that range, so you can decide
    whether `scale_to_overlap=True` is warranted."""
    w1, f1, _ = read_spectrum(file_blue, dq_reject=dq_reject)
    w2, f2, _ = read_spectrum(file_red, dq_reject=dq_reject)
    ov_lo, ov_hi = w2.min(), w1.max()
    if ov_hi <= ov_lo:
        print("No wavelength overlap between these two files.")
        return None
    m1 = (w1 >= ov_lo) & (w1 <= ov_hi)
    m2 = (w2 >= ov_lo) & (w2 <= ov_hi)
    med1, med2 = np.nanmedian(f1[m1]), np.nanmedian(f2[m2])
    ratio = med1 / med2 if med2 else np.nan
    print(f"Overlap: {ov_lo:.3f}-{ov_hi:.3f} um | "
          f"median flux blue={med1:.4g}, red={med2:.4g}, ratio={ratio:.3f}")
    return ratio


def mjd_to_datestr(mjd):
    """Human-readable UTC date (YYYY-MM-DD) for an MJD, for use in epoch
    filenames. Returns 'unknown-date' if mjd is None."""
    if mjd is None:
        return "unknown-date"
    return Time(mjd, format="mjd").iso[:10]


def date_from_filenames(files):
    """Earliest YYYYMMDD / YYYY-MM-DD token found in the file names,
    returned as 'YYYY-MM-DD' (None if no name has one). Preferred over the
    MJD-derived UTC date for labelling epochs, because it matches the date
    you already use in your file names (an exposure near 0h UTC can
    otherwise land on the neighbouring day)."""
    import re
    found = []
    for f in files:
        m = re.search(r"(20\d{2})-?(\d{2})-?(\d{2})", str(f).split("/")[-1])
        if m:
            found.append("-".join(m.groups()))
    return min(found) if found else None


def group_files_by_epoch(files, max_gap_days=1.0):
    """Cluster files into epochs by exposure-start MJD: sort ascending, and
    start a new cluster whenever the gap to the previous file's MJD exceeds
    `max_gap_days`. This is what enforces "don't combine spectra taken more
    than a day apart" -- call this before `combine_overlapping_spectra` and
    combine each cluster separately, rather than combining everything at
    once.

    Files with no usable MJD keyword are each put in their own trailing
    single-file cluster rather than silently lumped into a real epoch.

    Returns a list of file-lists, in time order.
    """
    tagged = [(get_mjd(f), f) for f in files]
    with_mjd = sorted((t for t in tagged if t[0] is not None), key=lambda t: t[0])
    without_mjd = [t for t in tagged if t[0] is None]

    clusters = []
    current = []
    prev_mjd = None
    for mjd, f in with_mjd:
        if prev_mjd is not None and (mjd - prev_mjd) > max_gap_days:
            clusters.append(current)
            current = []
        current.append(f)
        prev_mjd = mjd
    if current:
        clusters.append(current)

    for _, f in without_mjd:
        clusters.append([f])

    return clusters


def save_combined_spectrum(wave, flux, err, out_basename, mjd_info=None):
    """Save the combined spectrum to both FITS and ECSV. If `mjd_info` (as
    returned by `combine_overlapping_spectra`) is given, MJD_MIN/MJD_MAX/
    MJD_MEAN are written into the table metadata, which astropy carries
    through to the FITS extension header and the ECSV header block."""
    t = Table({"WAVELENGTH": wave, "FLUX": flux, "FLUX_ERROR": err})
    if mjd_info:
        for key, val in mjd_info.items():
            if val is not None:
                t.meta[key] = val
    t.write(out_basename + ".fits", overwrite=True)
    t.write(out_basename + ".ecsv", overwrite=True)
    print(f"Saved combined spectrum to {out_basename}.fits / .ecsv")


def save_spectrum_txt(wave, flux, err, path):
    """Write a 3-column text spectrum in the same layout as the plotting
    notebook's inputs (e.g. 4217_p1.txt): a '# Wavelength Flux Flux_Error'
    header line, then scientific-notation columns."""
    np.savetxt(path, np.column_stack([wave, flux, err]), fmt="%.6e",
               header="Wavelength Flux Flux_Error")


def combine_by_epoch(files, max_gap_days=1.0, wave_ranges=None,
                     scale_to_overlap=False, out_dir=None, prefix="spec"):
    """One call for the whole workflow: take a flat list of NIRSpec / MIRI
    LRS / MIRI MRS spectra (.ecsv or x1d .fits; ``None`` entries are
    ignored), split them into epochs (files <= `max_gap_days` apart share
    an epoch), and combine each epoch into a single spectrum.

    If `out_dir` is given, each epoch is written to
    ``<out_dir>/<prefix>_<YYYY-MM-DD>.txt`` (plus .ecsv/.fits with MJD
    metadata via `save_combined_spectrum`).

    Returns a list of dicts, in time order, with keys
    date, files, wave, flux, err, mjd_info, txt_path.
    """
    from pathlib import Path
    files = [str(f) for f in files if f is not None]
    results = []
    for group in group_files_by_epoch(files, max_gap_days=max_gap_days):
        wave, flux, err, mjd_info = combine_overlapping_spectra(
            group, scale_to_overlap=scale_to_overlap, wave_ranges=wave_ranges)
        date = date_from_filenames(group) or mjd_to_datestr(mjd_info["MJD_MEAN"])
        txt_path = None
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            base = str(out_dir / f"{prefix}_{date}")
            save_combined_spectrum(wave, flux, err, base, mjd_info=mjd_info)
            txt_path = base + ".txt"
            save_spectrum_txt(wave, flux, err, txt_path)
        results.append(dict(date=date, files=group, wave=wave, flux=flux,
                            err=err, mjd_info=mjd_info, txt_path=txt_path))
    return results


if __name__ == "__main__":
    # Example: combine the two S400A1 gratings.
    files = [
        "f170lp_g235m_s400a1_s400a1_s000000001_x1d.fits",
        "f290lp_g395m_s400a1_s400a1_s000000001_x1d.fits",
    ]
    check_overlap_offset(files[0], files[1])
    wave, flux, err, mjd_info = combine_overlapping_spectra(files, scale_to_overlap=False)
    save_combined_spectrum(wave, flux, err, "s400a1_combined", mjd_info=mjd_info)