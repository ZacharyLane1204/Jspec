#!/usr/bin/env python
"""
import_pfpc_products.py
=========================

Sort MRS-PFPC's output into the standard jspec layout.

Why this exists
---------------
``pfpc_proc`` writes every product it makes -- rate files, cal files,
Level-3 cubes, spectra, association files -- back into the same flat
directory it read the uncal files from. That is fine for PFPC's own
purposes, but it means an epoch directory ends up holding several
hundred files at four different processing stages with nothing but the
filename suffix to tell them apart.

``prepare_mrs_pfpc_epochs`` deliberately points PFPC at the epoch
directory anyway, because PFPC names every Level-3 product after that
directory and pointing it anywhere else would give products named after
the wrong thing. This script cleans up afterwards::

    <outdir>/MIRI-MRS/<target>_<date>/
        uncal/     jw*_uncal.fits
        stage1/    jw*_rate.fits, jw*_rateints.fits, jw*_trapsfilled.fits
        stage2/    jw*_cal.fits, jw*_s3d.fits (per-exposure), jw*_x1d.fits
        stage3/    <stem>_level3_ch*-*_s3d.fits, *_x1d.fits
        pfpc/      <stem>_pfpc_ch*-*_x1d.fits, <stem>_pfpc_sn.fits
        asn/       *_asn.json, *.json
        reextract/ (filled by abi_reextract_pfpc)
        final/     (filled by abi_reextract_pfpc)

Files are *moved*, not copied, and anything already in the right place
is left alone, so running this twice is harmless. The flat uncal files
that ``prepare_mrs_pfpc_epochs`` placed for PFPC's benefit are removed
once they are confirmed to also exist in ``uncal/`` -- re-running
``pfpc_proc`` afterwards needs them put back, which
``prepare_mrs_pfpc_epochs`` will do.

The Stage-3 and PFPC spectra are recorded in ``<outdir>/products.csv``.
Neither is flagged as the recommended product: for MIRI MRS that is the
AstroBkgInterp re-extraction, which ``abi_reextract_pfpc`` writes next.

Usage
-----
    python -m jspec.pipelines.import_pfpc_products --outdir SN2024aecx/

    # one epoch only
    python -m jspec.pipelines.import_pfpc_products --outdir SN2024aecx/ \\
        --date 2025-06-01

    # see what would move, without moving it
    python -m jspec.pipelines.import_pfpc_products --outdir SN2024aecx/ --dry-run
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

from jspec.layout import ensure_epoch_tree, find_epoch_dirs
from jspec.manifest import MANIFEST_NAME, record_product

#: (stage, glob patterns) in priority order. The first pattern a file
#: matches decides where it goes, so the more specific Level-3 and PFPC
#: patterns must come before the generic per-exposure ones.
ROUTING = [
    ("pfpc",   ("*_pfpc_ch*_x1d.fits", "*_pfpc_sn.fits", "*_pfpc*.fits")),
    ("stage3", ("*_level3_*_s3d.fits", "*_level3_*_x1d.fits",
                "*_level3_*_cal.fits", "*_level3*.fits")),
    ("stage1", ("*_rate.fits", "*_rateints.fits", "*_trapsfilled.fits",
                "*_dark*.fits")),
    ("stage2", ("*_cal.fits", "*_calints.fits", "*_s3d.fits", "*_x1d.fits",
                "*_s2d.fits", "*_flat*.fits")),
    ("asn",    ("*_asn.json", "*.json")),
    ("uncal",  ("*_uncal.fits",)),
]


def classify(name):
    """Which stage directory a filename belongs in, or None."""
    import fnmatch
    for stage, patterns in ROUTING:
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            return stage
    return None


def _same_file(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def organize_epoch(outdir, tag, date, dry_run=False, quiet=False):
    """Move one epoch's flat PFPC output into its stage directories.

    Returns a dict of stage -> number of files moved.
    """
    paths = ensure_epoch_tree(outdir, "MIRI-MRS", tag, date)
    epoch_dir = paths["epoch"]
    # pfpc/ is not part of the standard tree -- only the MRS-PFPC route
    # produces these -- so it is created here rather than sitting empty
    # in every NIRSpec epoch directory.
    paths["pfpc"] = os.path.join(epoch_dir, "pfpc")
    os.makedirs(paths["pfpc"], exist_ok=True)
    moved = {}

    for entry in sorted(os.listdir(epoch_dir)):
        src = os.path.join(epoch_dir, entry)
        if os.path.isdir(src):
            continue
        stage = classify(entry)
        if stage is None:
            continue

        dst = os.path.join(paths[stage], entry)

        # An uncal file here is the flat copy prepare_mrs_pfpc_epochs
        # placed for pfpc_proc to find. It is only safe to drop once the
        # real one is confirmed present in uncal/.
        if stage == "uncal" and os.path.exists(dst):
            if not dry_run:
                os.remove(src)
            moved["uncal (dropped duplicate)"] = moved.get(
                "uncal (dropped duplicate)", 0) + 1
            continue

        if os.path.exists(dst):
            if _same_file(src, dst):
                continue
            # Same name, different file: keep both rather than guess.
            dst = os.path.join(paths[stage], f"dup_{entry}")
            if os.path.exists(dst):
                continue

        if not dry_run:
            shutil.move(src, dst)
        moved[stage] = moved.get(stage, 0) + 1

    if not quiet:
        stem = paths["stem"]
        if moved:
            summary = ", ".join(f"{n} -> {s}" for s, n in sorted(moved.items()))
            print(f"  {stem}: {summary}" + ("  [dry run]" if dry_run else ""))
        else:
            print(f"  {stem}: nothing to move (already organised)")
    return moved


def record_epoch_products(outdir, tag, date, quiet=False):
    """Note this epoch's PFPC and Level-3 spectra in products.csv.

    Both are recorded but neither is flagged as recommended: comparing
    them against the AstroBkgInterp re-extraction is the point, and that
    is the product the analysis should actually use.
    """
    paths = ensure_epoch_tree(outdir, "MIRI-MRS", tag, date)
    paths["pfpc"] = os.path.join(paths["epoch"], "pfpc")
    n = 0
    for stage, product, level in (("pfpc", "pfpc_x1d", "stage3"),
                                   ("stage3", "x1d", "stage3")):
        files = sorted(glob.glob(os.path.join(paths[stage], "*_x1d.fits")))
        if not files:
            continue
        for path in files:
            ecsv = _ensure_ecsv(path)
            record_product(outdir, "MIRI-MRS", tag, date, ecsv or path,
                            product=product, level=level, fits_path=path,
                            recommended=False, quiet=True)
            n += 1
    if n and not quiet:
        print(f"    {n} spectrum file(s) recorded in {MANIFEST_NAME}")
    return n


def _ensure_ecsv(fits_path):
    """Write an ECSV beside an x1d FITS, with canonical column names.

    Returns the ECSV path, or "" if the file could not be read -- a
    PFPC product that is not a simple spectral table is not worth
    failing the whole import over.
    """
    ecsv = fits_path[:-5] + ".ecsv"
    if os.path.exists(ecsv):
        return ecsv
    try:
        from astropy.table import Table
        from jspec.columns import normalize_columns
        table = normalize_columns(Table.read(fits_path))
        table.write(ecsv, format="ascii.ecsv", overwrite=True)
        return ecsv
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True,
                     help="Output directory containing MIRI-MRS/<target>_<date>/")
    ap.add_argument("--tag", default=None,
                     help="Only process epochs with this target tag")
    ap.add_argument("--date", default=None,
                     help="Only process this epoch date (YYYY-MM-DD)")
    ap.add_argument("--dry-run", action="store_true",
                     help="Report what would move without moving anything")
    ap.add_argument("--no-record", action="store_true",
                     help="Skip updating products.csv")
    args = ap.parse_args()

    epochs = [(tag, date) for _l, tag, date, _p
              in find_epoch_dirs(args.outdir, "MIRI-MRS")
              if not (args.tag and tag != args.tag)
              and not (args.date and date != args.date)]

    if not epochs:
        raise SystemExit(
            f"No MIRI-MRS epoch directories found under {args.outdir}.\n"
            f"Expected <outdir>/MIRI-MRS/<target>_<date>/, which is what "
            f"prepare_mrs_pfpc_epochs creates.")

    print(f"{len(epochs)} epoch(s) under {args.outdir}")
    for tag, date in epochs:
        organize_epoch(args.outdir, tag, date, dry_run=args.dry_run)
        if not args.dry_run and not args.no_record:
            record_epoch_products(args.outdir, tag, date)

    if args.dry_run:
        print("\nDry run -- nothing was moved.")
    else:
        print(f"\nNext, re-extract with AstroBkgInterp for the final spectra:")
        print(f"  python -m jspec.pipelines.abi_reextract_pfpc --outdir {args.outdir}")


if __name__ == "__main__":
    main()
