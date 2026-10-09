#!/usr/bin/env python3
"""
nirspec_refine_extract.py
=========================

Look at -- and, only if you ask, move -- the extraction box of a Stage-3
NIRSpec FS spectrum. This is section 9 of the STScI notebook
(nirspec_x1d_process.ipynb) as a command:

    Stage-3 *_s2d.fits  --Extract1dStep-->  *_x1d_mod.fits   (flux-calibrated, Jy)

Input is the Stage-3 **s2d** from nirspec_fs_auto_pipeline.py: rectified and
combined, so the source sits on ONE row at every wavelength and the
pipeline's own Extract1dStep can be used directly. (If you pass a *_crf /
*_cal / *_x1d file, the matching *_s2d in the same folder is used instead.)

Default action: LOOK. Nothing is re-extracted unless you pass --y / --y-shift.
  * reports the rows the pipeline extracted (read from its x1d)
  * estimates where the source sits (median cross-slit profile, plus the same
    in wavelength sub-bands) and shows the --y-shift that would imply. The
    search is limited to a window around the pipeline box (--y0 / --search-radius)
    so a bright host elsewhere in the slit can't win. Agreeing sub-bands mean the
    peak is STABLE, not that it is your target -- always confirm on the QA plot.
    It never fails on a faint source; it tells you when the sub-bands disagree.
  * saves a QA plot (s2d + boxes, profile, spectra) and the pipeline's
    spectrum as <name>.ecsv / .fits

Two different "y" options -- don't mix them up:
  --y0 ROW      WHERE TO LOOK. Centre of the window used for the position
                estimate (default: the pipeline box centre). Changes nothing.
  --y ROW       WHERE TO EXTRACT. Re-run Extract1dStep with the box centred on
                this s2d row (0-based).
  --y-shift DY  Same, but relative to the pipeline's own box.
The box WIDTH is never changed (6 px; 5 for S1600A1): there is no aperture
correction, so a different width would break the flux calibration. --y-shift 0
should closely reproduce the pipeline's own x1d, which makes a good sanity check.

With --stitch, several s2d files (e.g. the gratings of ONE epoch) are joined
into <name>_final.*; it refuses if they come from different epochs.

Usage
-----
    # look first
    python nirspec_refine_extract.py --spec-file stage3/<product>_s2d.fits \\
        --out-dir refined/g395m --name sn2024aecx_g395m

    # look for the source in a band you know has signal, near row 24
    python nirspec_refine_extract.py --spec-file ... --y0 24 --wave-range 3.0 3.4

    # apply a shift you have decided on (or --y 33.5 for an absolute row)
    python nirspec_refine_extract.py --spec-file ... --y-shift -1.5

    # one epoch, all gratings, stitched
    python nirspec_refine_extract.py --spec-file stage3/*_20250131_*_s2d.fits \\
        --out-dir refined/epoch1 --name sn2024aecx_20250131 --y-shift -1.5 --stitch
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from pipelines.nirspec_fs_refining import (
    read_x1d, read_s2d, cross_profile, estimate_position, pipeline_extraction_box, slit_name,
    reextract_at_row, stitch, save_spectrum, plot_qa, plot_spectrum, check_same_epoch, mjd_to_datestr,
)


def log(*a):
    print(*a, flush=True)


def resolve_s2d(path):
    """Return the s2d to use. crf/cal/x1d inputs are swapped for their
    sibling *_s2d.fits."""
    d, base = os.path.dirname(path), os.path.basename(path)
    if base.endswith("_s2d.fits"):
        return path
    for suf in ("_x1d_mod.fits", "_x1d.fits", "_crf.fits", "_cal.fits"):
        if base.endswith(suf):
            cand = os.path.join(d, base[: -len(suf)] + "_s2d.fits")
            if os.path.exists(cand):
                log(f"  NOTE: {base} is not an s2d; using {os.path.basename(cand)} instead.")
                return cand
            break
    raise SystemExit(f"{base} is not a *_s2d.fits file and no matching s2d was found next to it. "
                     f"Point --spec-file at the Stage-3 s2d.")


def sibling_x1d(s2d_path):
    cand = s2d_path[: -len("_s2d.fits")] + "_x1d.fits"
    return cand if os.path.exists(cand) else None


def process_one(s2d_path, out_dir, base, y=None, y_shift=None, y0=None, radius=6.0, n_sub=4,
                ref=None, slit=None, wave_range=None, plots=True):
    log(f"\n== {os.path.basename(s2d_path)}")
    s2d = read_s2d(s2d_path)
    hdr = s2d["header"]
    slit = slit or slit_name(s2d_path)
    mjd = hdr.get("EXPSTART")
    log(f"  target={hdr.get('TARGNAME')}  slit={slit}  {hdr.get('GRATING')}/{hdr.get('FILTER')}  "
        f"epoch={mjd_to_datestr(mjd) if mjd else 'unknown'}  image={s2d['sci'].shape[0]} rows x {s2d['sci'].shape[1]} cols")

    x1d_pipe = sibling_x1d(s2d_path)
    box_pipe = pipeline_extraction_box(x1d_pipe) if x1d_pipe else None
    pieces_pipe = read_x1d(x1d_pipe, label="pipeline x1d", slit=slit) if x1d_pipe else []
    c_pipe = 0.5 * (box_pipe[0] + box_pipe[1]) if box_pipe else None
    if box_pipe:
        log(f"  pipeline box: rows {box_pipe[0]:.1f}-{box_pipe[1]:.1f} (centre {c_pipe:.2f})")
    elif x1d_pipe is None:
        log("  no pipeline x1d next to this s2d")

    # ---- look: where is the source? (a hint, never applied) ------------------
    centre_guess = y0 if y0 is not None else c_pipe
    est = estimate_position(s2d["sci"], s2d["wave"], wave_range, y0=centre_guess, radius=radius, n_sub=n_sub)
    where = f"rows {centre_guess - radius:.0f}-{centre_guess + radius:.0f}" if centre_guess is not None else "whole slit"
    bands = ", ".join("nan" if not np.isfinite(b) else f"{b:.1f}" for b in est["bands"])
    log(f"  profile peak ({where}): row {est['y']:.2f}   sub-bands: {bands}   scatter {est['scatter']:.2f} px")
    if est["consistent"]:
        log("    sub-bands agree -> a STABLE peak at this row (check on the QA plot that it is your target, "
            "not a host/neighbour)" +
            (f"; that would be --y-shift {est['y'] - c_pipe:+.2f}" if c_pipe is not None else ""))
    else:
        log("    sub-bands DISAGREE (or too few) -> too faint to locate this way here. Try --wave-range on a "
            "band with real signal, a different --y0, or judge from the QA plot.")

    # ---- act: only if asked ----------------------------------------------------
    center = None
    if y is not None:
        center = float(y)
    elif y_shift is not None:
        if c_pipe is None:
            raise SystemExit("--y-shift is relative to the pipeline box, which could not be read; use --y instead.")
        center = c_pipe + float(y_shift)

    box_new, pieces_new = None, []
    if center is not None:
        if c_pipe is not None and abs(center - c_pipe) > 2.0:
            log(f"  CAUTION: this moves the box {center - c_pipe:+.1f} px from the pipeline's. The width is fixed, "
                f"so a large move can clip the source -- check the QA plot.")
        x1d_mod, box_new = reextract_at_row(s2d_path, center, out_dir, ref_path=ref, slit=slit)
        pieces_new = read_x1d(x1d_mod, label=f"re-extracted (row {center:.1f})", slit=slit, strict=False)
        log(f"  wrote {os.path.basename(x1d_mod)}")

    final = pieces_new or pieces_pipe
    unit = final[0]["unit"] if final else "Jy"
    if final:
        wave, flux, err, unit = stitch(final, verbose=False)   # joins NRS1/NRS2 pieces of one file
        meta = dict(TARGET=hdr.get("TARGNAME"), SLIT=slit, MJD_MEAN=mjd,
                    EXTRACT=(f"re-extracted, box centre row {center:.2f}" if center is not None
                             else "pipeline Stage-3 x1d"))
        save_spectrum(os.path.join(out_dir, base), wave, flux, err, unit, meta)
        log(f"  saved {base}.ecsv / .fits")
    else:
        log("  nothing extracted (no pipeline x1d and no --y/--y-shift) -- QA plot only")

    if plots:
        spectra = [(p["label"], p["wave"], p["flux"]) for p in (pieces_pipe if center is not None else []) + final]
        profile = cross_profile(s2d["sci"], s2d["wave"], wave_range)
        plot_qa(os.path.join(out_dir, base + "_qa.png"), s2d, profile, box_pipe, box_new, spectra, unit,
                title=os.path.basename(s2d_path), est_row=est["y"])
        log(f"  saved {base}_qa.png")
    return final


def run(spec_files, out_dir="refined", name=None, y=None, y_shift=None, y0=None, search_radius=6.0,
        n_sub=4, extract1d_ref=None, slit=None, wave_range=None, stitch_files=False,
        max_epoch_gap_days=1.0, plots=True):
    """Same as the command line; call this from a notebook."""
    os.makedirs(out_dir, exist_ok=True)
    s2ds = [resolve_s2d(f) for f in spec_files]
    name = name or os.path.basename(s2ds[0])[: -len("_s2d.fits")]

    all_pieces = []
    for s2d in s2ds:
        base = name if len(s2ds) == 1 else os.path.basename(s2d)[: -len("_s2d.fits")]
        all_pieces += process_one(s2d, out_dir, base, y, y_shift, y0, search_radius, n_sub,
                                  extract1d_ref, slit, wave_range, plots)

    if stitch_files:
        if len(s2ds) < 2:
            log("\n--stitch needs two or more files; skipped.")
        elif all_pieces:
            mjd_min, mjd_max = check_same_epoch(s2ds, max_epoch_gap_days)   # raises on mixed epochs
            log(f"\n== stitching {len(s2ds)} files")
            wave, flux, err, unit = stitch(all_pieces)
            meta = dict(TARGET=name, MJD_MIN=mjd_min, MJD_MAX=mjd_max,
                        MJD_MEAN=None if mjd_min is None else 0.5 * (mjd_min + mjd_max))
            out = os.path.join(out_dir, f"{name}_final")
            save_spectrum(out, wave, flux, err, unit, meta)
            if plots:
                plot_spectrum(out + ".png", [(p["label"], p["wave"], p["flux"]) for p in all_pieces]
                              + [("stitched", wave, flux)], unit, title=name)
            log(f"  saved {os.path.basename(out)}.ecsv / .fits / .png")


def main():
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec-file", nargs="+", required=True, help="Stage-3 *_s2d.fits file(s)")
    ap.add_argument("--out-dir", default="refined")
    ap.add_argument("--name", default=None, help="Output base name (default: from the s2d file name)")
    ap.add_argument("--slit", default=None, help="Slit name (default: from the s2d header)")

    ap.add_argument("--y0", type=float, default=None,
                    help="WHERE TO LOOK: centre of the window for the position estimate (default: pipeline box centre)")
    ap.add_argument("--search-radius", type=float, default=6.0, help="Half-width of that window in rows (default 6)")
    ap.add_argument("--wave-range", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                    help="Only use this wavelength range (um) for the position estimate / profile")
    ap.add_argument("--n-sub", type=int, default=4, help="Wavelength sub-bands for the consistency check (default 4)")

    ap.add_argument("--y", type=float, default=None,
                    help="WHERE TO EXTRACT: re-run Extract1dStep with the box centred on this s2d row (0-based)")
    ap.add_argument("--y-shift", "--yshift", dest="y_shift", type=float, default=None,
                    help="Re-run Extract1dStep with the pipeline's box shifted by this many rows")
    ap.add_argument("--extract1d-ref", default=None, help="Path to an EXTRACT1D reference JSON (default: looked up from CRDS)")

    ap.add_argument("--stitch", action="store_true", help="Also stitch all given files into <name>_final.*")
    ap.add_argument("--max-epoch-gap-days", type=float, default=1.0, help="--stitch refuses inputs further apart than this")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    if a.y is not None and a.y_shift is not None:
        ap.error("use --y or --y-shift, not both")
    run(a.spec_file, a.out_dir, a.name, a.y, a.y_shift, a.y0, a.search_radius, a.n_sub, a.extract1d_ref,
        a.slit, tuple(a.wave_range) if a.wave_range else None, a.stitch, a.max_epoch_gap_days, not a.no_plots)


if __name__ == "__main__":
    main()