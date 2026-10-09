#!/usr/bin/env python3
"""
miri_imaging_auto_pipeline.py
================================

Automated, proposal-level MIRI imaging reduction pipeline, built on top of
the logic in JWPipeNB-MIRI-imaging.ipynb.

Given only a JWST proposal number, this script:

  1. Queries MAST for *every* MIRI/IMAGE observation in that proposal -- no
     observation number or filter needs to be specified; every filter used
     in the proposal is downloaded automatically.
  2. Downloads every uncalibrated science exposure.
  3. Runs `calwebb_detector1` on every exposure, then classifies each rate
     file as science or background via the BKGDTARG header keyword (rather
     than requiring a separate, manually-specified background observation).
  4. Runs `calwebb_image2` on every science and background rate file, with
     each science exposure's background subtraction matched by FILTER
     (mirroring the notebook's `writel2asn`).
  5. Groups the Stage-2 calibrated exposures by (TARGNAME, FILTER) and runs
     `calwebb_image3` once per group, drizzle-combining every exposure that
     shares a target and filter into one mosaicked image + source catalog.
     (Grouping matters because a proposal can observe more than one target,
     and/or use more than one filter -- both would otherwise get blended
     into a single, meaningless mosaic.)

Unlike the spectroscopic pipelines, there's no "final combined spectrum"
step here, and no epoch/day restriction on what gets combined -- Stage 3
mosaicking multiple epochs of imaging together into one deep image is
normal, desired practice (this is different from spectra, where combining
across widely-separated epochs can hide real time variability). If you
want per-epoch mosaics instead, split `--proposal` runs by visit or filter
using `--tag`.

Usage
-----
    python miri_imaging_auto_pipeline.py --proposal 1040

    python miri_imaging_auto_pipeline.py --proposal 1040 --max-cores half --tag visit1

Notes
-----
- `--tag SOMESTRING` appends `_SOMESTRING` to the default output directory
  name and to every Stage-3 product name. Omitted by default.
- Re-running with `--skip-download` reuses any `_uncal.fits` files already
  present in `<outdir>/uncal/`.
- `--bkg-method astrobkginterp` replaces the dedicated-background-exposure
  subtraction with an AstroBkgInterp-based correction that removes only
  the small-scale (thermal-noise-scale) background from each science
  exposure directly, preserving large-scale background structure. See
  astrobkginterp_thermal.py for how this works and its caveats (it masks
  a single source per exposure, so it suits isolated-target proposals
  better than crowded/multi-target mosaics).
"""

import argparse
import glob
import os
import time
import warnings
from collections import defaultdict

warnings.simplefilter("ignore", RuntimeWarning)

# ---------------------------------------------------------------------------
# CRDS setup (must happen before importing jwst/crds-dependent modules)
# ---------------------------------------------------------------------------
if os.getenv("CRDS_PATH") is None:
    os.environ["CRDS_PATH"] = os.path.join(os.path.expanduser("~"), "crds_cache")
if os.getenv("CRDS_SERVER_URL") is None:
    os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

from astropy.io import fits                     # noqa: E402
from astroquery.mast import Observations        # noqa: E402

import jspec.pipelines.jwst_proposal_pipeline as jwst_proposal_pipeline                                      # noqa: E402
from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Image3Pipeline  # noqa: E402
from jwst.associations import asn_from_list as afl                        # noqa: E402
from jwst.associations.lib.rules_level2_base import DMSLevel2bBase        # noqa: E402
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base       # noqa: E402

from jspec.pipelines.astrobkginterp_thermal import apply_to_fits_file as abi_thermal_correct  # noqa: E402
from jspec.pipelines.kde_bkg_correct import apply_to_fits_file as kde_bkg_correct  # noqa: E402


# ===========================================================================
# Helpers ported from JWPipeNB-MIRI-imaging.ipynb
# ===========================================================================

def filter_list(files, key, value):
    return [f for f in files if fits.getheader(f).get(key) == value]


def writel2asn(onescifile, bgfiles, asnfile, prodname):
    """Level-2 association for one exposure, with FILTER-matched
    backgrounds attached."""
    asn = afl.asn_from_list([onescifile], rule=DMSLevel2bBase, product_name=prodname)
    this_filter = fits.getval(onescifile, "FILTER")
    for f in bgfiles:
        if fits.getval(f, "FILTER") == this_filter:
            asn["products"][0]["members"].append({"expname": f, "exptype": "background"})
    _, serialized = asn.dump()
    with open(asnfile, "w") as outfile:
        outfile.write(serialized)


# ===========================================================================
# Pipeline stages
# ===========================================================================

def query_and_download(proposal_padded, proposal_num, uncal_dir):
    print(f"Querying MAST for all MIRI/IMAGE observations in proposal "
          f"{proposal_padded} ...")
    obs_table = Observations.query_criteria(
        instrument_name=["MIRI/IMAGE"],
        provenance_name=["CALJWST"],
        proposal_id=[str(proposal_num)],
    )
    print(f"  Found {len(obs_table)} observation/visit entries.")

    file_criteria = {"productType": "SCIENCE", "productSubGroupDescription": "UNCAL", "calib_level": [1]}
    pfilter = Observations.filter_products

    downloads = set()
    for exposure in obs_table:
        products = Observations.get_product_list(exposure)
        if len(products) == 0:
            continue
        matched = pfilter(products, **file_criteria)
        if len(matched) > 0:
            downloads.update(matched["dataURI"])

    downloads = sorted(u for u in downloads if f"jw{proposal_padded}" in u)
    print(f"  {len(downloads)} uncalibrated exposure(s) selected for download "
          f"(all filters included).")

    for uri in downloads:
        Observations.download_file(uri, local_path=os.path.join(uncal_dir, os.path.basename(uri)))

    return sorted(glob.glob(os.path.join(uncal_dir, "*uncal.fits")))


def run_detector1(uncal_files, det1_dir, max_cores):
    os.makedirs(det1_dir, exist_ok=True)
    det1dict = defaultdict(dict)
    if max_cores != "none":
        det1dict["jump"]["maximum_cores"] = max_cores

    for f in uncal_files:
        print(f"[Detector1] {os.path.basename(f)}")
        Detector1Pipeline.call(f, steps=det1dict, save_results=True, output_dir=det1_dir)

    return sorted(glob.glob(os.path.join(det1_dir, "*_rate.fits")))


def run_kde_bkg_correction(sci_rate_files, kdebkg_dir, box_size=80, min_clean_frac=0.2,
                            large_scale_sigma_thresh=4, save_diagnostics=False):
    """Replace each science rate file's background with the automatic
    star/nebula-masked, KDE-rescued estimate from background_subtraction.py
    (see kde_bkg_correct.py). Unlike astrobkginterp mode, this doesn't
    need a source position at all -- it detects and masks extended
    structure (e.g. a host galaxy) automatically, growing the mask until
    the object's own flux reaches the noise floor.

    NOTE: run this on individual (pre-Stage-3) exposures only -- see
    kde_bkg_correct.py's module docstring for why a combined/drizzled
    mosaic isn't a valid input for this method.
    """
    os.makedirs(kdebkg_dir, exist_ok=True)
    corrected = []
    for f in sci_rate_files:
        print(f"[KdeBkg] {os.path.basename(f)}")
        outpath = kde_bkg_correct(
            f, kdebkg_dir,
            box_size=box_size,
            min_clean_frac=min_clean_frac,
            large_scale_sigma_thresh=large_scale_sigma_thresh,
            save_diagnostics=save_diagnostics,
        )
        corrected.append(outpath)
    return sorted(corrected)


def run_thermal_bkg_correction(sci_rate_files, thermcorr_dir, bin_size_coarse=None,
                                bin_size_fine=25, src_ra=None, src_dec=None,
                                save_diagnostics=False):
    """Replace each science rate file's background with a version that has
    only the small-scale (thermal-noise-scale) component removed, via
    AstroBkgInterp (see astrobkginterp_thermal.py). Large-scale background
    structure (bin_size_coarse and larger) is preserved.

    If src_ra/src_dec are given, that sky position is masked in every
    exposure (converted to each exposure's own pixel frame via its WCS),
    overriding the header's TARG_RA/TARG_DEC. Otherwise each exposure's
    own TARG_RA/TARG_DEC is used, or the image center as a last resort.

    NOTE: AstroBkgInterp masks a single source per exposure. This is
    appropriate for proposals with one isolated target per field, not
    for crowded/multi-target mosaics -- see astrobkginterp_thermal.py's
    module docstring. Inspect the diagnostics on at least one exposure
    (pass save_diagnostics=True) before trusting it across a full
    proposal.
    """
    os.makedirs(thermcorr_dir, exist_ok=True)
    corrected = []
    for f in sci_rate_files:
        print(f"[ThermalBkg] {os.path.basename(f)}")
        outpath = abi_thermal_correct(
            f, thermcorr_dir,
            src_ra=src_ra, src_dec=src_dec,
            bin_size_coarse=bin_size_coarse,
            bin_size_fine=bin_size_fine,
            save_diagnostics=save_diagnostics,
        )
        corrected.append(outpath)
    return sorted(corrected)


def run_image2(rate_files, bg_rate_files, asn_dir, image2_dir, do_bkg_subtract=True):
    os.makedirs(asn_dir, exist_ok=True)
    os.makedirs(image2_dir, exist_ok=True)
    image2dict = defaultdict(dict)

    if do_bkg_subtract:
        image2dict["bkg_subtract"]["sigma"] = 2
    else:
        # Background already handled upstream (e.g. AstroBkgInterp thermal
        # correction applied directly to the science rate files), so skip
        # the pipeline's own dedicated-background-exposure subtraction.
        image2dict["bkg_subtract"]["skip"] = True

    all_rate = list(rate_files) + list(bg_rate_files)
    for f in all_rate:
        base = os.path.basename(f)
        stem = base[:-5] if base.endswith(".fits") else os.path.splitext(base)[0]
        asnfile = os.path.join(asn_dir, f"{stem}_l2asn.json")
        writel2asn(f, bg_rate_files if do_bkg_subtract else [], asnfile, "Level2")
        print(f"[Image2] {os.path.basename(f)}")
        Image2Pipeline.call(asnfile, steps=image2dict, save_results=True, output_dir=image2_dir)

    return sorted(glob.glob(os.path.join(image2_dir, "*_cal.fits")))


def run_image3(cal_files, image3_dir, tag_suffix=""):
    """Group calibrated exposures by (TARGNAME, FILTER) and run Image3 once
    per group, producing one drizzled mosaic + source catalog per group."""
    os.makedirs(image3_dir, exist_ok=True)
    image3dict = defaultdict(dict)

    by_group = defaultdict(list)
    for f in cal_files:
        targname = fits.getval(f, "TARGNAME") or "UNKNOWN_TARGET"
        filt = fits.getval(f, "FILTER") or "UNKNOWN_FILTER"
        by_group[(targname, filt)].append(os.path.abspath(f))

    print(f"  {len(by_group)} target/filter group(s) found -> "
          f"{len(by_group)} Stage-3 association(s).")

    outputs = {}
    for (targname, filt), files in by_group.items():
        product_name = f"{targname}_{filt}{tag_suffix}"
        asnfile = os.path.join(image3_dir, f"{product_name}_l3asn.json")
        asn = afl.asn_from_list(files, rule=DMS_Level3_Base, product_name=product_name)
        asn.data["asn_type"] = "image3"
        _, serialized = asn.dump()
        with open(asnfile, "w") as fh:
            fh.write(serialized)

        print(f"[Image3] {targname} / {filt}: combining {len(files)} exposure(s)")
        Image3Pipeline.call(asnfile, steps=image3dict, save_results=True, output_dir=image3_dir)

        out_i2d = os.path.join(image3_dir, f"{product_name}_i2d.fits")
        if os.path.exists(out_i2d):
            outputs[(targname, filt)] = out_i2d
        else:
            print(f"  WARNING: expected output {out_i2d} not found.")

    return outputs


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proposal", required=True, help="JWST proposal/program number, e.g. 1040")
    parser.add_argument("--outdir", required=True,
                    help="Output directory for this target. Instrument and epoch subdirectories are created inside it: <outdir>/<INSTRUMENT>/<target>_<date>/{uncal,stage1,stage2,stage3,reextract,final}")
    parser.add_argument("--skip-download", action="store_true",
                         help="Reuse existing files in <outdir>/uncal/ instead of querying MAST")
    parser.add_argument("--max-cores", default="half", choices=["none", "quarter", "half", "all"],
                         help="Cores to use for the jump step (default: half)")
    parser.add_argument("--tag", default=None,
                         help="Optional string included in the output directory name and in every "
                              "Stage-3 product name. Omitted by default.")
    parser.add_argument("--bkg-method", default="pipeline", choices=["pipeline", "astrobkginterp", "kde"],
                         help="'pipeline' (default): subtract dedicated BKGDTARG background "
                              "exposures via calwebb_image2's own bkg_subtract step, as before. "
                              "'astrobkginterp': model and subtract only the small-scale "
                              "(thermal-noise-scale) background directly from each science exposure "
                              "using AstroBkgInterp (single-source masking), preserving large-scale "
                              "background structure. 'kde': automatically mask stars AND extended "
                              "structure (e.g. a host galaxy) via background_subtraction.py, growing "
                              "the mask until the object's own flux reaches the noise floor, then "
                              "estimate the background with a tiled estimator that falls back to a "
                              "KDE-mode estimate in heavily masked tiles. Better suited than "
                              "astrobkginterp when the target sits inside extended emission that "
                              "fills a meaningful fraction of the frame. Dedicated background "
                              "exposures are still downloaded/classified but are not used for the "
                              "correction in 'astrobkginterp' or 'kde' mode. See "
                              "astrobkginterp_thermal.py / kde_bkg_correct.py for details and "
                              "caveats.")
    parser.add_argument("--abi-bin-size-coarse", type=int, default=None,
                         help="[astrobkginterp mode] tile size (px) for the large-scale pass -- "
                              "structure at and above this scale is preserved. Default: ~1/3 of "
                              "the smaller image dimension.")
    parser.add_argument("--abi-bin-size-fine", type=int, default=25,
                         help="[astrobkginterp mode] tile size (px) for the small-scale pass -- "
                              "should match the spatial scale of the thermal noise you want "
                              "removed. Default: 25.")
    parser.add_argument("--bkg-src-ra", type=float, default=None,
                         help="[astrobkginterp mode] RA (decimal degrees) of the source to mask "
                              "while fitting the background, applied to every science exposure. "
                              "Overrides each exposure's TARG_RA header value. Use this to mask a "
                              "specific sky position (e.g. a source embedded in extended "
                              "structure, or something other than the nominal proposal target). "
                              "Must be given together with --bkg-src-dec.")
    parser.add_argument("--bkg-src-dec", type=float, default=None,
                         help="[astrobkginterp mode] Dec (decimal degrees) counterpart to "
                              "--bkg-src-ra. Must be given together with --bkg-src-ra.")
    parser.add_argument("--abi-save-diagnostics", action="store_true",
                         help="[astrobkginterp mode] also save *_thermdiag.fits files with the "
                              "coarse/fine background models and the subtracted noise map, so you "
                              "can visually confirm large-scale structure was preserved.")
    parser.add_argument("--kde-box-size", type=int, default=80,
                         help="[kde mode] tile size (px) for the tiled background estimator and "
                              "the scale other detection parameters in background_subtraction.py "
                              "derive from. Default 80 (this pipeline's default is scaled down from "
                              "background_subtraction.py's own default of 100, to better suit "
                              "MIRI's smaller frames -- see kde_bkg_correct.py).")
    parser.add_argument("--kde-min-clean-frac", type=float, default=0.2,
                         help="[kde mode] minimum unmasked pixel fraction a tile needs before the "
                              "standard tiled estimate is trusted; tiles below this get a KDE-mode "
                              "estimate instead. Default 0.2.")
    parser.add_argument("--kde-large-scale-sigma-thresh", type=float, default=4,
                         help="[kde mode] detection threshold (multiples of the coarse-floor RMS) "
                              "for the large-scale excess test that catches objects too big to show "
                              "local contrast (e.g. a galaxy filling much of the frame). Default 4.")
    parser.add_argument("--kde-save-diagnostics", action="store_true",
                         help="[kde mode] save background_subtraction.py's diagnostic PNGs (data, "
                              "star mask, nebula mask, source mask, background surface) per "
                              "exposure -- check these, especially the nebula mask, before trusting "
                              "this on a full proposal.")
    args = parser.parse_args()

    if (args.bkg_src_ra is None) != (args.bkg_src_dec is None):
        raise SystemExit("--bkg-src-ra and --bkg-src-dec must be given together.")

    proposal_num = int(args.proposal)
    proposal_padded = f"{proposal_num:05d}"
    tag_suffix = f"_{args.tag}" if args.tag else ""
    outdir = os.path.abspath(args.outdir)   # absolute: association files need it

    uncal_dir = os.path.join(outdir, "uncal")
    det1_dir = os.path.join(outdir, "stage1")
    asn_dir = os.path.join(outdir, "asn")
    image2_dir = os.path.join(outdir, "stage2")
    image3_dir = os.path.join(outdir, "stage3")
    thermcorr_dir = os.path.join(outdir, "thermcorr")
    kdebkg_dir = os.path.join(outdir, "kdebkg")
    dirs_to_make = [uncal_dir, det1_dir, asn_dir, image2_dir, image3_dir]
    if args.bkg_method == "astrobkginterp":
        dirs_to_make.append(thermcorr_dir)
    elif args.bkg_method == "kde":
        dirs_to_make.append(kdebkg_dir)
    for d in dirs_to_make:
        os.makedirs(d, exist_ok=True)

    print(f"JWST Calibration Pipeline version: {jwst_proposal_pipeline.__version__}")
    print(f"Output directory: {outdir}\n")

    t0 = time.perf_counter()

    if args.skip_download:
        uncal_files = sorted(glob.glob(os.path.join(uncal_dir, "*uncal.fits")))
        print(f"--skip-download: using {len(uncal_files)} existing uncal file(s).")
    else:
        uncal_files = query_and_download(proposal_padded, proposal_num, uncal_dir)

    if not uncal_files:
        raise SystemExit("No uncalibrated files found/downloaded -- nothing to process.")

    rate_files = run_detector1(uncal_files, det1_dir, args.max_cores)
    t1 = time.perf_counter()
    print(f"Detector1 complete for {len(rate_files)} exposure(s) "
          f"({(t1 - t0) / 60:.1f} min elapsed)\n")

    sci_rate = filter_list(rate_files, "BKGDTARG", False)
    bg_rate = filter_list(rate_files, "BKGDTARG", True)
    print(f"Classified {len(sci_rate)} science and {len(bg_rate)} background exposure(s).\n")

    if args.bkg_method == "astrobkginterp":
        if bg_rate:
            print(f"NOTE: {len(bg_rate)} exposure(s) are flagged BKGDTARG=True, but "
                  f"--bkg-method astrobkginterp doesn't use a dedicated background "
                  f"pointing -- it models the background directly from each exposure. "
                  f"All {len(rate_files)} exposure(s) will be treated as correctable "
                  f"science frames. If any of these are genuinely pointed at blank sky "
                  f"rather than your target, verify that before trusting the output "
                  f"(see header TARGNAME/TARG_RA/TARG_DEC).\n")
        print("--bkg-method astrobkginterp: correcting small-scale (thermal-noise-scale) "
              "background directly on each science exposure; dedicated background "
              "exposures will not be subtracted via calwebb_image2.\n")
        sci_rate = run_thermal_bkg_correction(
            rate_files, thermcorr_dir,
            bin_size_coarse=args.abi_bin_size_coarse,
            bin_size_fine=args.abi_bin_size_fine,
            src_ra=args.bkg_src_ra,
            src_dec=args.bkg_src_dec,
            save_diagnostics=args.abi_save_diagnostics,
        )
        bg_rate_for_image2 = []
        do_bkg_subtract = False
    elif args.bkg_method == "kde":
        if bg_rate:
            print(f"NOTE: {len(bg_rate)} exposure(s) are flagged BKGDTARG=True, but "
                  f"--bkg-method kde doesn't use a dedicated background pointing -- it "
                  f"models the background directly from each exposure via automatic "
                  f"star/nebula masking. All {len(rate_files)} exposure(s) will be "
                  f"treated as correctable science frames. If any of these are "
                  f"genuinely pointed at blank sky rather than your target, verify "
                  f"that before trusting the output (see header TARGNAME/TARG_RA/"
                  f"TARG_DEC).\n")
        print("--bkg-method kde: automatically masking stars and extended structure "
              "(e.g. a host galaxy) and correcting the background per exposure via "
              "background_subtraction.py; dedicated background exposures will not be "
              "subtracted via calwebb_image2.\n")
        sci_rate = run_kde_bkg_correction(
            rate_files, kdebkg_dir,
            box_size=args.kde_box_size,
            min_clean_frac=args.kde_min_clean_frac,
            large_scale_sigma_thresh=args.kde_large_scale_sigma_thresh,
            save_diagnostics=args.kde_save_diagnostics,
        )
        bg_rate_for_image2 = []
        do_bkg_subtract = False
    else:
        bg_rate_for_image2 = bg_rate
        do_bkg_subtract = True

    cal_files = run_image2(sci_rate, bg_rate_for_image2, asn_dir, image2_dir,
                            do_bkg_subtract=do_bkg_subtract)
    t2 = time.perf_counter()
    print(f"Image2 complete: {len(cal_files)} calibrated exposure(s) "
          f"({(t2 - t0) / 60:.1f} min elapsed)\n")

    if not cal_files:
        raise SystemExit("No Stage-2 calibrated products were produced -- stopping.")

    mosaics = run_image3(cal_files, image3_dir, tag_suffix=tag_suffix)
    t3 = time.perf_counter()
    print(f"Image3 complete: {len(mosaics)} mosaic(s) "
          f"({(t3 - t0) / 60:.1f} min elapsed)\n")

    print(f"\nAll done. Total runtime: {(time.perf_counter() - t0) / 60:.1f} min")
    print(f"  Calibrated exposures: {image2_dir}")
    print(f"  Final mosaics (one per target/filter): {image3_dir}")
    for key, path in mosaics.items():
        print(f"    {key}: {path}")


if __name__ == "__main__":
    main()