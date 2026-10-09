"""
jspec.columns
=============

One spelling for spectral table columns.

The pipelines grew apart on this: most write ``WAVELENGTH`` / ``FLUX`` /
``FLUX_ERROR``, but ``nirspec_fs_refining`` and
``nirspec_miri_calibration_utils`` wrote ``wavelength`` / ``flux`` /
``flux_err``. Anything consuming both then needs a case-insensitive
lookup, and the mismatch is invisible until a downstream ``KeyError``.

The canonical spelling is upper case, matching the ``x1d`` FITS tables
the JWST pipeline itself produces (``WAVELENGTH``, ``FLUX``,
``FLUX_ERROR``, ``SURF_BRIGHT``, ``DQ`` ...). Agreeing with the format
that already exists beats inventing a nicer one.

Reading is deliberately *not* made strict. ``jspec.spectrum`` accepts
both spellings and will keep doing so, because files written by older
versions of these scripts are still on disk and still perfectly good.
``normalize_columns`` is how those are brought into line when they pass
through anything that rewrites them.
"""
from __future__ import annotations

__all__ = ["WAVELENGTH", "FLUX", "FLUX_ERROR", "CANONICAL",
           "canonical_name", "normalize_columns", "spectrum_table"]

WAVELENGTH = "WAVELENGTH"
FLUX = "FLUX"
FLUX_ERROR = "FLUX_ERROR"

#: Lower-case alias -> canonical name. Only covers columns that were
#: actually written by some version of these pipelines; an unrecognised
#: column is upper-cased rather than dropped or renamed.
CANONICAL = {
    "wavelength": WAVELENGTH, "wave": WAVELENGTH, "lam": WAVELENGTH,
    "lambda": WAVELENGTH, "wl": WAVELENGTH,
    "flux": FLUX, "fnu": FLUX, "f_nu": FLUX, "flux_jy": FLUX,
    "flux_err": FLUX_ERROR, "flux_error": FLUX_ERROR, "fluxerr": FLUX_ERROR,
    "err": FLUX_ERROR, "error": FLUX_ERROR, "sigma": FLUX_ERROR,
    "e_flux": FLUX_ERROR, "unc": FLUX_ERROR, "uncertainty": FLUX_ERROR,
    # Variants that carry extra meaning: keep them distinct from
    # FLUX_ERROR so a total error is never mistaken for a formal one.
    "flux_err_total": "FLUX_ERR_TOTAL",
    "flux_err_formal": "FLUX_ERR_FORMAL",
    "flux_sys": "FLUX_SYS",
    "surf_bright": "SURF_BRIGHT",
    "sb_error": "SB_ERROR",
    "background": "BACKGROUND",
    "bkgd_error": "BKGD_ERROR",
    "npixels": "NPIXELS",
    "dq": "DQ", "flag": "FLAG", "mask": "MASK",
}


def canonical_name(name: str) -> str:
    """Canonical spelling for one column name."""
    return CANONICAL.get(str(name).strip().lower(), str(name).upper())


def normalize_columns(table, inplace: bool = True):
    """Rename a table's columns to the canonical spellings.

    Parameters
    ----------
    table : astropy.table.Table
        Modified in place by default.
    inplace : bool
        Set False to work on a copy and leave the input untouched.

    Returns
    -------
    astropy.table.Table
        The table with renamed columns.

    Notes
    -----
    A rename that would collide with an existing column is skipped, so a
    table that somehow holds both ``flux_err`` and ``FLUX_ERROR`` keeps
    both rather than silently losing one.
    """
    if not inplace:
        table = table.copy()
    for name in list(table.colnames):
        new = canonical_name(name)
        if new != name and new not in table.colnames:
            table.rename_column(name, new)
    return table


def spectrum_table(wave, flux, err=None, unit=None, wave_unit="um",
                   meta=None, **extra):
    """Build a spectral table with canonical columns and units.

    The single place a 1-D spectrum table is constructed, so every
    pipeline writing one ends up with the same column names, the same
    order, and units actually set.

    Parameters
    ----------
    wave, flux : array_like
        Wavelength (microns by default) and flux.
    err : array_like, optional
        Flux uncertainty, written as ``FLUX_ERROR``.
    unit : str, optional
        Flux unit, applied to ``FLUX`` and ``FLUX_ERROR``.
    wave_unit : str
        Wavelength unit, default "um".
    meta : dict, optional
        Table metadata. ``None`` values are dropped, since they survive
        a round trip through ECSV as the string "None".
    **extra
        Additional columns, named canonically on the way in.
    """
    import numpy as np
    from astropy.table import Table

    cols = {WAVELENGTH: np.asarray(wave, dtype=float),
            FLUX: np.asarray(flux, dtype=float)}
    if err is not None:
        cols[FLUX_ERROR] = np.asarray(err, dtype=float)
    for name, values in extra.items():
        if values is not None:
            cols[canonical_name(name)] = np.asarray(values)

    table = Table(cols)
    table[WAVELENGTH].unit = wave_unit
    if unit is not None:
        table[FLUX].unit = unit
        if FLUX_ERROR in table.colnames:
            table[FLUX_ERROR].unit = unit
    if meta:
        table.meta.update({k: v for k, v in meta.items() if v is not None})
    return table
