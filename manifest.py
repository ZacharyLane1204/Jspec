"""
jspec.manifest
==============

``products.csv`` -- the index of final data products under an output
directory.

The problem it solves: after a few proposals and a few epochs, an output
tree holds thousands of files, and the handful you actually want to
analyse are indistinguishable by path alone from Stage-2 intermediates.
Worse, "the final product" means different things per instrument -- for
MIRI MRS it is the ``abi_reextract_pfpc`` re-extraction, not the Stage-3
``x1d``, and nothing in the filename says so.

So each pipeline records what it produced, and ``products.csv`` collects
it:

===============  =========================================================
``instrument``   Canonical label, e.g. ``MIRI-MRS``
``target``       Target name (the tag part of the epoch directory)
``date``         Epoch date, ``YYYY-MM-DD``
``epoch``        Epoch directory name, ``<target>_<date>``
``level``        ``stage3`` or ``final``
``product``      What produced it, e.g. ``abi_combined``, ``x1d``
``recommended``  ``yes`` for the product to analyse for this instrument
``path``         Path to the ECSV, relative to the output directory
``fits_path``    Matching FITS, if one was written
``n_rows``       Number of spectral samples
``wave_min_um``  Shortest wavelength, microns
``wave_max_um``  Longest wavelength, microns
``mjd``          Mean MJD of the epoch, if known
``updated``      When the row was written (UTC, ISO 8601)
===============  =========================================================

``recommended`` is the column worth explaining. Both the Stage-3 ``x1d``
and the re-extraction are recorded, because comparing them is how you
check the re-extraction did something sensible -- but only one of them
is the product to feed into the analysis, and that is the one flagged.

Rows are keyed on ``(instrument, epoch, product)``, so re-running a
pipeline updates its row instead of appending a duplicate.
"""
from __future__ import annotations

import csv
import datetime as _dt
import os

from .layout import find_epoch_dirs, instrument_label, parse_epoch_dirname

__all__ = ["MANIFEST_NAME", "FIELDS", "record_product", "read_manifest",
           "write_manifest", "rebuild_manifest", "RECOMMENDED_PRODUCT"]

MANIFEST_NAME = "products.csv"

FIELDS = ("instrument", "target", "date", "epoch", "level", "product",
          "recommended", "path", "fits_path", "n_rows", "wave_min_um",
          "wave_max_um", "mjd", "updated")

#: The product each instrument's analysis should actually start from.
#: MIRI MRS is the interesting case: the Stage-3 x1d is extracted from a
#: cube with no background handling beyond the pipeline's own, so on a
#: target sitting in host light the AstroBkgInterp re-extraction is the
#: one to use.
RECOMMENDED_PRODUCT = {
    "MIRI-MRS": "abi_combined",
    "MIRI-LRS": "final",
    "NIRSpec-FS": "final",
    "NIRSpec-IFU": "abi_combined",
}

_KEY = ("instrument", "epoch", "product")


def _utcnow():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _manifest_path(outdir):
    return os.path.join(outdir, MANIFEST_NAME)


def _relpath(path, outdir):
    """Store paths relative to outdir so a tree stays valid if moved."""
    if not path:
        return ""
    try:
        return os.path.relpath(os.path.abspath(path), os.path.abspath(outdir))
    except ValueError:            # different drive on Windows
        return str(path)


def _spectrum_summary(path):
    """(n_rows, wave_min, wave_max, mjd) for an ECSV, best effort.

    Never raises: the manifest is a convenience, and a malformed or
    half-written file should leave a row with blanks rather than crash
    the pipeline run that produced it.
    """
    blank = ("", "", "", "")
    if not path or not os.path.exists(path):
        return blank
    try:
        from astropy.table import Table
        import numpy as np
        table = Table.read(path, format="ascii.ecsv")
        name = next((c for c in table.colnames
                      if c.upper() in ("WAVELENGTH", "WAVE")), None)
        if name is None:
            return blank
        wave = np.asarray(table[name], dtype=float)
        wave = wave[np.isfinite(wave)]
        if wave.size == 0:
            return (len(table), "", "", "")
        mjd = table.meta.get("MJD_MEAN", table.meta.get("MJD_MIN", ""))
        return (len(table), f"{wave.min():.4f}", f"{wave.max():.4f}",
                f"{float(mjd):.5f}" if mjd not in ("", None) else "")
    except Exception:
        return blank


def read_manifest(outdir):
    """Existing rows as a list of dicts; empty list if there is none."""
    path = _manifest_path(outdir)
    if not os.path.exists(path):
        return []
    try:
        with open(path, newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def write_manifest(outdir, rows):
    """Write `rows`, sorted, to ``<outdir>/products.csv``."""
    os.makedirs(outdir, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r.get("instrument", ""),
                                        r.get("date", ""),
                                        r.get("product", "")))
    path = _manifest_path(outdir)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS,
                                 extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})
    return path


def record_product(outdir, instrument, target, date, ecsv_path,
                   product="final", level="final", fits_path=None,
                   recommended=None, quiet=False):
    """Add or update one row in ``<outdir>/products.csv``.

    Safe to call on every run: the row is replaced in place, keyed on
    (instrument, epoch, product). Failures are reported but never raised
    -- losing a manifest row should not lose a reduction.
    """
    try:
        label = instrument_label(instrument)
        epoch = f"{target}_{date}"
        if recommended is None:
            recommended = RECOMMENDED_PRODUCT.get(label) == product
        n_rows, wmin, wmax, mjd = _spectrum_summary(ecsv_path)

        row = {
            "instrument": label, "target": target, "date": date,
            "epoch": epoch, "level": level, "product": product,
            "recommended": "yes" if recommended else "",
            "path": _relpath(ecsv_path, outdir),
            "fits_path": _relpath(fits_path, outdir),
            "n_rows": n_rows, "wave_min_um": wmin, "wave_max_um": wmax,
            "mjd": mjd, "updated": _utcnow(),
        }

        rows = [r for r in read_manifest(outdir)
                if tuple(r.get(k, "") for k in _KEY) != tuple(row[k] for k in _KEY)]
        rows.append(row)
        path = write_manifest(outdir, rows)
        if not quiet:
            print(f"  recorded in {os.path.relpath(path, outdir)}: "
                  f"{label} {epoch} {product}")
        return path
    except Exception as exc:
        if not quiet:
            print(f"  WARNING: could not update {MANIFEST_NAME}: {exc}")
        return None


def rebuild_manifest(outdir, quiet=False):
    """Rebuild ``products.csv`` by scanning the output tree.

    For recovering an index after moving files around, or for adopting a
    tree produced before the manifest existed. Only looks in the
    ``final``, ``reextract`` and ``stage3`` directories of each epoch,
    since those are where products live.
    """
    found = []
    for label, tag, date, epoch_path in find_epoch_dirs(outdir):
        for stage, level in (("final", "final"), ("reextract", "final"),
                              ("stage3", "stage3")):
            stage_path = os.path.join(epoch_path, stage)
            if not os.path.isdir(stage_path):
                continue
            for name in sorted(os.listdir(stage_path)):
                if not name.endswith(".ecsv"):
                    continue
                ecsv = os.path.join(stage_path, name)
                product = _product_from_name(name, tag, date) or stage
                fits_path = ecsv[:-5] + ".fits"
                n_rows, wmin, wmax, mjd = _spectrum_summary(ecsv)
                found.append({
                    "instrument": label, "target": tag, "date": date,
                    "epoch": f"{tag}_{date}", "level": level,
                    "product": product,
                    "recommended": "yes" if RECOMMENDED_PRODUCT.get(label) == product else "",
                    "path": _relpath(ecsv, outdir),
                    "fits_path": _relpath(fits_path, outdir)
                    if os.path.exists(fits_path) else "",
                    "n_rows": n_rows, "wave_min_um": wmin,
                    "wave_max_um": wmax, "mjd": mjd, "updated": _utcnow(),
                })
    path = write_manifest(outdir, found)
    if not quiet:
        print(f"{len(found)} product(s) indexed in {path}")
    return found


def _product_from_name(name, tag, date):
    """Recover the product token from ``<tag>_<date>_<product>.ecsv``."""
    stem = name[:-5] if name.endswith(".ecsv") else name
    prefix = f"{tag}_{date}"
    return stem[len(prefix):].strip("_") if stem.startswith(prefix) else ""
