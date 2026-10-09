#!/usr/bin/env python3
"""
abi_reextract_pfpc.py
=======================

Alternative to PFPC's own point-source extraction: run AstroBkgInterp on
each of MRS-PFPC's per-channel/band Level-3 cubes ("<name>_ch{1..4}-
{short,medium,long}_s3d.fits") *before* extracting a 1D spectrum, instead
of using PFPC's own Extract1dStep call on the un-subtracted cube. Produces
one background-subtracted x1d per channel/band, then combines all of them
into a single final spectrum (FITS + ECSV) for this epoch, using the same
adaptive, sigma-clipped overlap-rescaling merge as combine_epoch_spectra.py.

Rationale: PFPC corrects a systematic point-source flux-calibration /
throughput pattern -- it does not remove host or extended background. If
the target sits on host light, subtracting that background with
AstroBkgInterp before extraction removes contamination PFPC was never
designed to touch.

What this does NOT do: apply pfpc_cor's flux correction. pfpc_cor almost
certainly discovers its inputs by the standard "<name>_level3_ch{N}-
{band}_x1d.fits" naming pattern; this script's outputs are named
"<name>_ch{N}-{band}_abi_x1d.fits" specifically so they will NOT be picked
up by pfpc_cor and silently mixed with PFPC's own products. Whether you
should still apply PFPC's flux correction on top of this is worth
resolving directly against pfpc_cor's own source before doing it by hand;
it was not verified here.

Reuses, rather than reimplements:
  - miri_mrs_auto_pipeline.find_source_centroid / resolve_source_pixel for
    locating the source in each cube
  - miri_mrs_auto_pipeline.run_astrobkginterp_background for the ABI call
  - miri_mrs_auto_pipeline.extract_spectrum_manual for the flux-calibrated
    Extract1dStep call (subtract_background=False, since ABI already
    subtracted it -- extract_spectrum_manual's own docstring warns about
    the double-subtraction this avoids)
  - combine_epoch_spectra.combine_epoch_spectrum for the final multi-
    channel/band stitch
  - combine_nirspec_orders.save_combined_spectrum for the FITS/ECSV output

Not every channel is guaranteed to be a good candidate for AstroBkgInterp.
Its own printed diagnostics (the "background-model term adds a median X in
quadrature" line) can show the fit becoming order-of-magnitude more
uncertain in the smaller-FOV channels (3, 4) -- a sign of too little
source/background contrast for a 2D fit to constrain, not something
poly_k/bin_size tuning fixes. `--skip-abi-channels` bypasses AstroBkgInterp
entirely for the channels you name, extracting straight from the raw cube
with `extract_pfpc_style()` instead -- PFPC's own recipe (pipeline-default
local annulus background), just with a fixed center_xy for consistency
with the ABI-processed channels alongside it.

miri_mrs_auto_pipeline.py, combine_nirspec_orders.py and
combine_epoch_spectra.py all need to be importable (same directory or on
PYTHONPATH). Run this once per epoch's Level-3 directory -- it does not
do its own epoch splitting (see prepare_mrs_pfpc_epochs.py for that).

Usage
-----
    python abi_reextract_pfpc.py \\
        --level3-dir SN2024aecx_2024-06-01_uncal/ \\
        --outdir SN2024aecx_2024-06-01_abi/ \\
        --product-name SN2024aecx_2024-06-01

    # with a known sky position (skips per-band auto-centroiding -- resolves
    # this RA/Dec directly in each cube's own WCS instead):
    python abi_reextract_pfpc.py --level3-dir ... --outdir ... \\
        --product-name ... --ra 123.4567 --dec -12.3456

    # channel 4's AstroBkgInterp fit is unreliable (huge background-model
    # uncertainty term) -- use PFPC's own extraction for it instead:
    python abi_reextract_pfpc.py --level3-dir ... --outdir ... \\
        --product-name ... --ra 123.4567 --dec -12.3456 \\
        --skip-abi-channels 4
"""
import argparse
import glob
import json
import os
import re
import shutil

import numpy as np

import jspec.pipelines.miri_mrs_auto_pipeline as mp
from jspec.pipelines.combine_epoch_spectra import combine_epoch_spectrum
from jspec.pipelines.combine_nirspec_orders import save_combined_spectrum
from jspec.layout import ensure_epoch_tree, find_epoch_dirs
from jspec.manifest import MANIFEST_NAME, record_product


_CHBAND_RE = re.compile(r"_ch(\d)-(short|medium|long)_s3d\.fits$", re.IGNORECASE)


def find_level3_cubes(level3_dir, pattern="*_ch*-*_s3d.fits"):
    """All per-channel/band Stage-3 (PFPC) cubes in `level3_dir`, sorted by
    (channel, band) for readable log output -- the actual combination
    order downstream is by wavelength, not this."""
    files = sorted(glob.glob(os.path.join(level3_dir, pattern)))
    band_order = {"short": 0, "medium": 1, "long": 2}

    def sort_key(f):
        m = _CHBAND_RE.search(os.path.basename(f))
        if not m:
            return (99, 99)
        return (int(m.group(1)), band_order.get(m.group(2).lower(), 9))

    return sorted(files, key=sort_key)


def extract_pfpc_style(cube_file, out_file, x, y, ifu_rfcorr=True,
                        ifu_rscale=None, ifu_covar_scale=1.0, srctype="POINT",
                        apply_apcorr=True):
    """Extract straight from the raw (un-subtracted) Level-3 cube using
    PFPC's own recipe, for channels where AstroBkgInterp's background fit
    is unreliable (see the module docstring's "when to use this" note).

    Mirrors miri_mrs_auto_pipeline.extract_spectrum_manual() almost
    exactly, with the one deliberate difference: `subtract_background` is
    left UNSET rather than hardcoded to False, so it falls through to the
    pipeline's own default (None -> the MIRI MRS extract1d reference
    file's setting, which for a point source is local annulus background
    subtraction -- confirmed directly in the installed jwst==3.0.0 source,
    not assumed). This is exactly what PFPC's own Spec3 extract_1d call
    does (it only sets ifu_autocen and ifu_rfcorr, never touching
    subtract_background), so this function reproduces PFPC's own
    extraction, just with a caller-supplied fixed center_xy instead of
    letting ifu_autocen search for it independently in every band.

    Returns (wave, flux, err, meta), wavelength-sorted."""
    from stdatamodels.jwst import datamodels
    from jwst.extract_1d import Extract1dStep

    kwargs = dict(center_xy=[float(x), float(y)], ifu_autocen=True,
                  ifu_rfcorr=ifu_rfcorr, apply_apcorr=apply_apcorr,
                  ifu_covar_scale=ifu_covar_scale)
    if srctype is not None:
        kwargs["ifu_set_srctype"] = srctype
    if ifu_rscale is not None:
        kwargs["ifu_rscale"] = ifu_rscale

    cube = datamodels.open(cube_file)
    result = Extract1dStep.call(cube, **kwargs)
    if out_file:
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        result.save(out_file)

    spec = result.spec[0]
    tab = spec.spec_table
    wave = np.asarray(tab["WAVELENGTH"], dtype=float)
    flux = np.asarray(tab["FLUX"], dtype=float)
    err = np.asarray(tab["FLUX_ERROR"] if "FLUX_ERROR" in tab.dtype.names else tab["ERROR"],
                      dtype=float)
    order = np.argsort(wave)
    wave, flux, err = wave[order], flux[order], err[order]
    meta = {"n_good": int(np.isfinite(flux).sum()), "out_file": out_file,
            "method": "pfpc-style (no AstroBkgInterp)"}

    cube.close()
    result.close()
    return wave, flux, err, meta


def process_cube(cube_file, out_dir, product_name, ra=None, dec=None,
                  bkg_mode="polynomial", fwhm_scale=2.5, poly_k=3,
                  bin_size=9, min_aper_rad=5.0, aper_rad=7, ann_width=6,
                  wave_range=None, clip_sigma=5.0, n_sub=4, search_radius=8.0,
                  ifu_rfcorr=True, ifu_rscale=None, ifu_covar_scale=1.0,
                  channel_overrides=None, skip_abi_channels=None):
    """AstroBkgInterp on one channel/band cube, then extract its
    background-subtracted spectrum. Returns (x1d_path, centroid_or_None)
    -- centroid is None when a fixed (ra, dec) was supplied instead of
    auto-detection.

    `skip_abi_channels`, if given, is a set/list of channel numbers as
    strings ("1".."4"). For a cube in one of those channels, AstroBkgInterp
    is skipped ENTIRELY and extract_pfpc_style() runs directly on the raw
    cube instead -- use this for a channel where AstroBkgInterp's
    background-model uncertainty term (printed during the run) is orders
    of magnitude larger than in the other channels, which is a direct sign
    the fit has too little source/background contrast to trust, and no
    amount of poly_k/bin_size tuning will fix that.

    `channel_overrides`, if given, is a dict keyed by channel number as a
    string ("1".."4") -> dict of any run_astrobkginterp_background kwargs
    to override for that channel's cube specifically (bkg_mode, fwhm_scale,
    poly_k, bin_size, min_aper_rad, aper_rad, ann_width, use_abi_err). Use
    this when the background fit degrades in the smaller-FOV channels
    (3, 4) relative to channel 1/2 -- e.g. {"3": {"poly_k": 2}, "4":
    {"poly_k": 1, "bin_size": 5}} to use a lower-order fit where there's
    less field to constrain it."""
    m = _CHBAND_RE.search(os.path.basename(cube_file))
    tag = f"ch{m.group(1)}-{m.group(2)}" if m else os.path.splitext(
        os.path.basename(cube_file))[0]
    tag_name = f"{product_name}_{tag}"
    channel_num = m.group(1) if m else None

    if ra is not None and dec is not None:
        src_x, src_y = mp.resolve_source_pixel(cube_file, ra=ra, dec=dec)
        centroid = None
        print(f"[{tag}] using supplied sky position -> pixel ({src_x:.2f}, {src_y:.2f})")
    else:
        centroid = mp.find_source_centroid(
            cube_file, wave_range=wave_range, clip_sigma=clip_sigma, n_sub=n_sub,
            search_radius=search_radius,
            plot_file=os.path.join(out_dir, f"{tag_name}_centroid_qa.png"))
        src_x, src_y = centroid["x"], centroid["y"]
        print(f"[{tag}] auto-centroid -> ({src_x:.2f}, {src_y:.2f}) "
              f"+/- ({centroid['x_err']:.2f}, {centroid['y_err']:.2f}), "
              f"SNR={centroid['snr']:.1f}")
        if max(centroid["x_scatter"], centroid["y_scatter"]) > 1.0:
            print(f"  NOTE: sub-band positions disagree by more than a pixel in "
                  f"{tag} -- check {tag_name}_centroid_qa.png before trusting this.")

    x1d_file = os.path.join(out_dir, f"{tag_name}_abi_x1d.fits")

    if channel_num is not None and channel_num in (skip_abi_channels or ()):
        print(f"[{tag}] channel {channel_num} in skip_abi_channels -- using PFPC's own "
              f"extraction recipe on the raw cube (no AstroBkgInterp)")
        wave, flux, err, meta = extract_pfpc_style(
            cube_file, x1d_file, src_x, src_y, ifu_rfcorr=ifu_rfcorr,
            ifu_rscale=ifu_rscale, ifu_covar_scale=ifu_covar_scale)
        print(f"[{tag}] extracted {meta['n_good']}/{len(wave)} finite points "
              f"({wave[0]:.2f}-{wave[-1]:.2f} um) -> {x1d_file} (pfpc-style, no ABI)")
        return x1d_file, centroid

    bkg_kwargs = dict(bkg_mode=bkg_mode, fwhm_scale=fwhm_scale, poly_k=poly_k,
                       bin_size=bin_size, min_aper_rad=min_aper_rad,
                       aper_rad=aper_rad, ann_width=ann_width)
    override = (channel_overrides or {}).get(channel_num)
    if override:
        bkg_kwargs.update(override)
        print(f"[{tag}] channel {channel_num} override applied: {override}")

    bgsub_path = mp.run_astrobkginterp_background(
        cube_file, src_x, src_y, out_dir, tag_name, **bkg_kwargs)
    print(f"[{tag}] background-subtracted cube: {bgsub_path} "
          f"(QA plot: {tag_name}_abi_bgsub_quicklook.png)")

    wave, flux, err, (sx, sy), meta = mp.extract_spectrum_manual(
        bgsub_path, out_file=x1d_file, x=src_x, y=src_y, ifu_rfcorr=ifu_rfcorr,
        ifu_rscale=ifu_rscale, ifu_covar_scale=ifu_covar_scale, return_meta=True)
    print(f"[{tag}] extracted {meta['n_good']}/{len(wave)} finite points "
          f"({wave[0]:.2f}-{wave[-1]:.2f} um) -> {x1d_file}")

    return x1d_file, centroid


#: Where per-channel Level-3 cubes may live inside an epoch directory,
#: in the order they are preferred. "pfpc" wins because a PFPC-corrected
#: cube is a better starting point than the plain Stage-3 one.
_CUBE_SUBDIRS = ("pfpc", "stage3")


def discover_mrs_epochs(outdir, tag=None, date=None):
    """Find MIRI-MRS epochs under `outdir` that have Level-3 cubes.

    Returns (tag, date, level3_dir, paths) tuples, where `paths` is the
    epoch's layout dict. Epochs without cubes are skipped silently --
    they are usually just visits that have not been through Stage 3 yet.
    """
    found = []
    for _label, dir_tag, dir_date, epoch_path in find_epoch_dirs(outdir, "MIRI-MRS"):
        if (tag and dir_tag != tag) or (date and dir_date != date):
            continue
        for sub in _CUBE_SUBDIRS:
            candidate = os.path.join(epoch_path, sub)
            if glob.glob(os.path.join(candidate, "*_ch*-*_s3d.fits")):
                paths = ensure_epoch_tree(outdir, "MIRI-MRS", dir_tag, dir_date)
                found.append((dir_tag, dir_date, candidate, paths))
                break
    return found


def process_level3_dir_with_abi(level3_dir, out_dir, product_name, ra=None, dec=None,
                                 pattern="*_ch*-*_s3d.fits", auto_scale=True,
                                 clip_sigma_combine=3.0, min_scale_snr=3.0,
                                 max_scale_factor=2.0, max_rel_uncertainty=0.3,
                                 channel_overrides=None, skip_abi_channels=None,
                                 **abi_extract_kwargs):
    """Run AstroBkgInterp + re-extraction on every channel/band cube in
    `level3_dir`, then combine all of them into one final spectrum for
    this epoch. Returns (out_base, x1d_files, scale_log)."""
    os.makedirs(out_dir, exist_ok=True)
    cubes = find_level3_cubes(level3_dir, pattern=pattern)
    if not cubes:
        raise SystemExit(f"No cubes matching {pattern!r} found in {level3_dir}")
    print(f"{len(cubes)} channel/band cube(s) found in {level3_dir}")

    x1d_files = []
    for cube_file in cubes:
        x1d_file, _ = process_cube(cube_file, out_dir, product_name, ra=ra, dec=dec,
                                    channel_overrides=channel_overrides,
                                    skip_abi_channels=skip_abi_channels,
                                    **abi_extract_kwargs)
        x1d_files.append(x1d_file)

    print(f"\nCombining {len(x1d_files)} channel/band spectra into one final spectrum...")
    wave, flux, err, mjd_info, scale_log = combine_epoch_spectrum(
        x1d_files, auto_scale=auto_scale, clip_sigma=clip_sigma_combine,
        min_scale_snr=min_scale_snr, max_scale_factor=max_scale_factor,
        max_rel_uncertainty=max_rel_uncertainty)
    for s in scale_log:
        if s["scaled"]:
            print(f"  scaled new segment by {s['factor']:.4f} "
                  f"+/- {s['factor_std']:.4f} to match established composite "
                  f"(file: {s['file']})")
        else:
            print(f"  no rescale applied (file: {s['file']}) -- {s['reason']}")

    out_base = os.path.join(out_dir, f"{product_name}_abi_combined")
    save_combined_spectrum(wave, flux, err, out_base, mjd_info=mjd_info)
    print(f"\nFinal combined spectrum: {out_base}.fits / .ecsv")

    return out_base, x1d_files, scale_log


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True,
                     help="Output directory for this target. On its own, every "
                          "MIRI-MRS epoch found underneath is re-extracted in "
                          "turn and recorded in products.csv.")
    ap.add_argument("--level3-dir", default=None,
                     help="Process one specific directory of per-channel/band "
                          "*_ch{N}-{band}_s3d.fits cubes instead of scanning "
                          "--outdir. Use this for a PFPC working directory that "
                          "does not follow the jspec layout. Requires "
                          "--product-name.")
    ap.add_argument("--product-name", default=None,
                     help="Output filename stem; only needed with --level3-dir "
                          "(otherwise it is the epoch's <target>_<date>).")
    ap.add_argument("--tag", default=None,
                     help="Only process epochs whose directory tag matches this")
    ap.add_argument("--date", default=None,
                     help="Only process this epoch date (YYYY-MM-DD)")
    ap.add_argument("--ra", type=float, default=None,
                     help="Source RA (deg). With --dec, skips per-band "
                          "auto-centroiding and resolves this sky position "
                          "directly in each cube's own WCS instead.")
    ap.add_argument("--dec", type=float, default=None)
    ap.add_argument("--bkg-mode", default="polynomial")
    ap.add_argument("--fwhm-scale", type=float, default=2.5,
                     help="Source-mask radius = ceil(fwhm_scale * PSF FWHM) per "
                          "slice -- see run_astrobkginterp_background's docstring "
                          "for why 2.5, not the AstroBkgInterp default, is used")
    ap.add_argument("--poly-k", type=int, default=3)
    ap.add_argument("--bin-size", type=int, default=9)
    ap.add_argument("--min-aper-rad", type=float, default=5.0)
    ap.add_argument("--channel-overrides", default=None,
                     help="Per-channel background-fit overrides, as inline JSON or a "
                          "path to a JSON file, keyed by channel number: "
                          '\'{"3": {"poly_k": 2}, "4": {"poly_k": 1, "bin_size": 5}}\'. '
                          "Use this if the ABI-to-ABI overlap factors (see the "
                          "combine step's printed scale_log) drift smoothly with "
                          "channel -- lower poly_k for the smaller-FOV channels "
                          "(3, 4) first, and check each channel's "
                          "*_abi_bgsub_quicklook.png before and after.")
    ap.add_argument("--skip-abi-channels", default=None,
                     help="Comma-separated channel numbers (e.g. '4') to skip "
                          "AstroBkgInterp for entirely, using PFPC's own extraction "
                          "recipe (pipeline-default local background) directly on the "
                          "raw cube instead. Use this where the printed background-model "
                          "uncertainty term is orders of magnitude larger than in other "
                          "channels -- a sign the fit has too little source/background "
                          "contrast to trust, not something poly_k tuning will fix.")
    ap.add_argument("--no-rfcorr", action="store_true",
                     help="Disable the extra 1D residual-fringe correction in "
                          "Extract1dStep (on by default, matching PFPC's own "
                          "extraction)")
    ap.add_argument("--ifu-covar-scale", type=float, default=1.0)
    ap.add_argument("--no-auto-scale", action="store_true",
                     help="Disable adaptive overlap rescaling in the final combine")
    ap.add_argument("--clip-sigma", type=float, default=3.0,
                     help="Sigma-clip threshold for the overlap ratio in the final combine")
    ap.add_argument("--min-scale-snr", type=float, default=3.0)
    ap.add_argument("--max-scale-factor", type=float, default=2.0,
                     help="Skip rescaling a segment if the factor falls outside "
                          "[1/this, this] -- guards against noise-dominated overlaps "
                          "at faint, long-wavelength channels")
    ap.add_argument("--max-rel-uncertainty", type=float, default=0.3,
                     help="Skip rescaling a segment if factor_std / |factor| exceeds this")
    args = ap.parse_args()

    if (args.ra is None) != (args.dec is None):
        raise SystemExit("Give both --ra and --dec, or neither.")

    channel_overrides = None
    if args.channel_overrides:
        if os.path.isfile(args.channel_overrides):
            with open(args.channel_overrides) as f:
                channel_overrides = json.load(f)
        else:
            channel_overrides = json.loads(args.channel_overrides)
        print(f"Channel overrides: {channel_overrides}")

    skip_abi_channels = None
    if args.skip_abi_channels:
        skip_abi_channels = {c.strip() for c in args.skip_abi_channels.split(",") if c.strip()}
        print(f"Skipping AstroBkgInterp entirely for channel(s): {sorted(skip_abi_channels)}")

    common = dict(
        ra=args.ra, dec=args.dec,
        bkg_mode=args.bkg_mode, fwhm_scale=args.fwhm_scale, poly_k=args.poly_k,
        bin_size=args.bin_size, min_aper_rad=args.min_aper_rad,
        channel_overrides=channel_overrides, skip_abi_channels=skip_abi_channels,
        ifu_rfcorr=not args.no_rfcorr, ifu_covar_scale=args.ifu_covar_scale,
        auto_scale=not args.no_auto_scale, clip_sigma_combine=args.clip_sigma,
        min_scale_snr=args.min_scale_snr, max_scale_factor=args.max_scale_factor,
        max_rel_uncertainty=args.max_rel_uncertainty)

    if args.level3_dir:
        if not args.product_name:
            raise SystemExit("--level3-dir requires --product-name.")
        process_level3_dir_with_abi(args.level3_dir, args.outdir,
                                     args.product_name, **common)
        return

    epochs = discover_mrs_epochs(args.outdir, tag=args.tag, date=args.date)
    if not epochs:
        raise SystemExit(
            f"No MIRI-MRS epochs with Level-3 cubes found under {args.outdir}.\n"
            f"Expected <outdir>/MIRI-MRS/<target>_<date>/stage3/*_ch*-*_s3d.fits "
            f"(or pfpc/), which is what miri_mrs_auto_pipeline and "
            f"import_pfpc_products produce. To point at an arbitrary directory "
            f"of cubes instead, use --level3-dir with --product-name.")

    print(f"{len(epochs)} MIRI-MRS epoch(s) to re-extract\n")
    if (args.ra is not None) and len(epochs) > 1:
        print("  NOTE: an explicit --ra/--dec was given but more than one epoch "
              "will be processed -- the same sky position is applied to all of "
              "them. That is usually fine for a fixed target, but narrow the run "
              "with --date if the epochs need different positions.\n")

    for tag, date, level3_dir, paths in epochs:
        stem = paths["stem"]
        print(f"=== {stem}: cubes in {os.path.relpath(level3_dir, args.outdir)}")
        out_base, _x1ds, _log = process_level3_dir_with_abi(
            level3_dir, paths["reextract"], stem, **common)

        # The combined spectrum is the product to analyse, so it is
        # promoted out of reextract/ into final/ and flagged in the index.
        final_base = os.path.join(paths["final"], f"{stem}_abi_combined")
        for ext in (".ecsv", ".fits"):
            if os.path.exists(out_base + ext):
                shutil.copy2(out_base + ext, final_base + ext)
        record_product(args.outdir, "MIRI-MRS", tag, date,
                        final_base + ".ecsv", product="abi_combined",
                        level="final",
                        fits_path=final_base + ".fits"
                        if os.path.exists(final_base + ".fits") else None)

    print(f"\nProduct index: {os.path.join(args.outdir, MANIFEST_NAME)}")


if __name__ == "__main__":
    main()