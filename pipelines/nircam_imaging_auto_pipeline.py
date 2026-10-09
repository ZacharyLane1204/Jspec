#!/usr/bin/env python3
"""
nircam_imaging_auto_pipeline.py
=================================

Automated, proposal-level NIRCam imaging reduction pipeline. Structurally
mirrors miri_imaging_auto_pipeline.py, adapted for NIRCam's imaging mode.

Given only a JWST proposal number, this script:

  1. Queries MAST for *every* NIRCAM/IMAGE observation in that proposal --
     no observation number, detector, module, or filter needs to be
     specified; every filter/pupil combination and every detector used in
     the proposal is downloaded automatically.
  2. Downloads every uncalibrated science exposure.
  3. Runs `calwebb_detector1` on every exposure, then classifies each rate
     file as science or background via the BKGDTARG header keyword.
  4. Runs `calwebb_image2` on every science and background rate file, with
     each science exposure's background subtraction matched by
     (FILTER, PUPIL, DETECTOR) so short- and long-wavelength channel data,
     and different detectors, are never cross-matched.
  5. Groups the Stage-2 calibrated exposures by (TARGNAME, FILTER, PUPIL)
     and runs `calwebb_image3` once per group, drizzle-combining every
     exposure (across all detectors/modules that share that target and
     filter/pupil combination) into one mosaicked image + source catalog.
     Grouping by PUPIL in addition to FILTER matters for NIRCam because
     many science filters are paired with specific pupil-wheel elements
     (e.g. narrow-band filters paired with a CLEAR or grism pupil
     position); combining across mismatched pupils would blend
     incompatible data.

As with the MIRI imaging pipeline, there's no "final combined spectrum"
step and no epoch restriction on what gets combined into a Stage-3
mosaic -- combining multiple epochs of imaging into one deep image is
normal practice here. If you want per-epoch mosaics instead, split
`--proposal` runs by visit using `--tag`.

Usage
-----
    python nircam_imaging_auto_pipeline.py --proposal 1040

    python nircam_imaging_auto_pipeline.py --proposal 1040 --max-cores half --tag visit1

Notes
-----
- `--tag SOMESTRING` appends `_SOMESTRING` to the default output directory
  name and to every Stage-3 product name. Omitted by default.
- Re-running with `--skip-download` reuses any `_uncal.fits` files already
  present in `<outdir>/uncal/`.
- NIRCam exposures come from up to 8 SW detectors (A1-A4, B1-B4) and 2 LW
  detectors (ALONG, BLONG) per exposure. All are downloaded and run
  through Detector1/Image2 individually, then combined at Stage 3 by
  (TARGNAME, FILTER, PUPIL) regardless of which detector(s) they came
  from -- this is standard practice for NIRCam mosaics.
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


# ===========================================================================
# Helpers
# ===========================================================================

def filter_list(files, key, value):
    return [f for f in files if fits.getheader(f).get(key) == value]


def writel2asn(onescifile, bgfiles, asnfile, prodname):
    """Level-2 association for one exposure, with FILTER+PUPIL+DETECTOR-
    matched backgrounds attached (so SW/LW channels and different
    detectors never get cross-matched)."""
    asn = afl.asn_from_list([onescifile], rule=DMSLevel2bBase, product_name=prodname)
    this_key = (fits.getval(onescifile, "FILTER"),
                fits.getval(onescifile, "PUPIL"),
                fits.getval(onescifile, "DETECTOR"))
    for f in bgfiles:
        f_key = (fits.getval(f, "FILTER"), fits.getval(f, "PUPIL"), fits.getval(f, "DETECTOR"))
        if f_key == this_key:
            asn["products"][0]["members"].append({"expname": f, "exptype": "background"})
    _, serialized = asn.dump()
    with open(asnfile, "w") as outfile:
        outfile.write(serialized)


# ===========================================================================
# Pipeline stages
# ===========================================================================

def query_and_download(proposal_padded, proposal_num, uncal_dir):
    print(f"Querying MAST for all NIRCam/IMAGE observations in proposal "
          f"{proposal_padded} ...")
    obs_table = Observations.query_criteria(
        instrument_name=["NIRCAM/IMAGE"],
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
          f"(all detectors/filters/pupils included).")

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


def run_image2(rate_files, bg_rate_files, asn_dir, image2_dir):
    os.makedirs(asn_dir, exist_ok=True)
    os.makedirs(image2_dir, exist_ok=True)
    image2dict = defaultdict(dict)
    image2dict["bkg_subtract"]["sigma"] = 2

    all_rate = list(rate_files) + list(bg_rate_files)
    for f in all_rate:
        base = os.path.basename(f)
        stem = base[:-5] if base.endswith(".fits") else os.path.splitext(base)[0]
        asnfile = os.path.join(asn_dir, f"{stem}_l2asn.json")
        writel2asn(f, bg_rate_files, asnfile, "Level2")
        print(f"[Image2] {os.path.basename(f)}")
        Image2Pipeline.call(asnfile, steps=image2dict, save_results=True, output_dir=image2_dir)

    return sorted(glob.glob(os.path.join(image2_dir, "*_cal.fits")))


def run_image3(cal_files, image3_dir, tag_suffix=""):
    """Group calibrated exposures by (TARGNAME, FILTER, PUPIL) and run
    Image3 once per group -- combining across all detectors/modules that
    share a target and filter/pupil combination -- producing one drizzled
    mosaic + source catalog per group."""
    os.makedirs(image3_dir, exist_ok=True)
    image3dict = defaultdict(dict)

    by_group = defaultdict(list)
    for f in cal_files:
        targname = fits.getval(f, "TARGNAME") or "UNKNOWN_TARGET"
        filt = fits.getval(f, "FILTER") or "UNKNOWN_FILTER"
        pupil = fits.getval(f, "PUPIL") or "UNKNOWN_PUPIL"
        by_group[(targname, filt, pupil)].append(os.path.abspath(f))

    print(f"  {len(by_group)} target/filter/pupil group(s) found -> "
          f"{len(by_group)} Stage-3 association(s).")

    outputs = {}
    for (targname, filt, pupil), files in by_group.items():
        product_name = f"{targname}_{filt}_{pupil}{tag_suffix}"
        asnfile = os.path.join(image3_dir, f"{product_name}_l3asn.json")
        asn = afl.asn_from_list(files, rule=DMS_Level3_Base, product_name=product_name)
        asn.data["asn_type"] = "image3"
        _, serialized = asn.dump()
        with open(asnfile, "w") as fh:
            fh.write(serialized)

        print(f"[Image3] {targname} / {filt} / {pupil}: combining {len(files)} exposure(s)")
        Image3Pipeline.call(asnfile, steps=image3dict, save_results=True, output_dir=image3_dir)

        out_i2d = os.path.join(image3_dir, f"{product_name}_i2d.fits")
        if os.path.exists(out_i2d):
            outputs[(targname, filt, pupil)] = out_i2d
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
    args = parser.parse_args()

    proposal_num = int(args.proposal)
    proposal_padded = f"{proposal_num:05d}"
    tag_suffix = f"_{args.tag}" if args.tag else ""
    outdir = os.path.abspath(args.outdir)   # absolute: association files need it

    uncal_dir = os.path.join(outdir, "uncal")
    det1_dir = os.path.join(outdir, "stage1")
    asn_dir = os.path.join(outdir, "asn")
    image2_dir = os.path.join(outdir, "stage2")
    image3_dir = os.path.join(outdir, "stage3")
    for d in (uncal_dir, det1_dir, asn_dir, image2_dir, image3_dir):
        os.makedirs(d, exist_ok=True)

    # print(f"JWST Calibration Pipeline version: {jwst_proposal_pipeline.__version__}")
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

    cal_files = run_image2(sci_rate, bg_rate, asn_dir, image2_dir)
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
    print(f"  Final mosaics (one per target/filter/pupil): {image3_dir}")
    for key, path in mosaics.items():
        print(f"    {key}: {path}")


if __name__ == "__main__":
    main()