#!/usr/bin/env python3
"""
nirspec_ifu_auto_pipeline.py
============================

Automated, proposal-level NIRSpec IFU (EXP_TYPE = NRS_IFU) reduction pipeline.
It is the NIRSpec counterpart of `miri_mrs_auto_pipeline.py` and keeps the same
overall design: download -> Detector1 -> Spec2 -> epoch grouping -> Spec3 ->
AstroBkgInterp background subtraction, with the final 1D extraction left as a
manual step (`extract_spectrum_manual()`), because the right extraction centre
is usually a target-specific position that the header RA/Dec does not capture.

What is the same as the MRS pipeline
------------------------------------
  * MAST download through `mast_uncal_download.download_uncal`.
  * Epoch (day-level) separation on the Stage-2 `_cal` files, before Stage 3;
    Stage 3 is run once per (target, epoch) group, so exposures more than
    `--max-epoch-gap-days` apart are never combined into the same cube.
  * `jwst`'s own background paths (`bkg_subtract`, `master_background`) are
    skipped; background estimation/subtraction is done on the Stage-3 cube with
    AstroBkgInterp (wavelength-dependent source mask + 2D polynomial fit per
    slice), including the same repair of AstroBkgInterp's error propagation
    (NaN-filled ERR, extra axis, and its broken background-error term).
  * No automatic Extract1dStep / stitched "final spectrum". The product is the
    background-subtracted cube for each (target, epoch, grating/filter).
  * The source position for AstroBkgInterp can be given as pixel x/y or RA/Dec,
    or found automatically (DAOStarFinder on the median-collapsed cube).

What is different, and why (NIRSpec IFU vs MIRI MRS)
-----------------------------------------------------
  * No fringe steps. `fringe`, `residual_fringe`, `straylight` and the 1D
    `ifu_rfcorr` are MIRI-only; they are not touched here. `ifu_rscale` in
    Extract1dStep is also MIRI-only in jwst (the NIRSpec extraction radius comes
    from the extract1d reference file), so it is not exposed.
  * Imprint (a.k.a. leakcal) subtraction. NIRSpec IFU programs can include
    imprint exposures (header IS_IMPRT = T, MSA closed), which measure light
    leaking through the MSA into the IFU. They are processed through Detector1
    but never treated as science; instead, for each science exposure the
    imprint exposures of the same detector/grating/filter (and not flagged as
    a background pointing) are attached to its Level-2 association as
    "imprint" members, and `imprint_subtract` is switched on. jwst's ImprintStep
    then picks the one that matches the science exposure's observation number
    and dither position (or uses the single one available). If a program has no
    imprint exposures the step is a no-op, and the coverage check prints a note.
  * Grating/filter cubes instead of channel/band cubes. NIRSpec gratings have
    little overlap and different calibrations, so the default is
    `cube_build.output_type='band'` (one cube per grating+filter; NRS1 and NRS2
    of the high-resolution gratings are merged by cube_build itself). Use
    `--output-type multi` to get a single cube across gratings, or `grating`.
    Stitch the per-cube spectra afterwards with your existing combine scripts.
  * Detectors: high-resolution gratings (G140H, G235H, G395H) fall on both NRS1
    and NRS2; the coverage check warns if one is missing. Medium-resolution
    gratings and the prism are not expected on NRS2.
  * PSF size for the source mask. The MRS relation 0.033*lambda + 0.106 arcsec
    does not apply. The NIRSpec IFU PSF is undersampled and roughly constant
    with wavelength, at ~0.18-0.22 arcsec (geometric mean of the along- and
    across-slice FWHM). `_nirspec_fwhm_pixels()` uses the functional form of
    D'Eugenio et al. (2024), sigma[arcsec] = a + b*lambda*exp(-c/lambda), with
    the constants a=0.076, b=0.013, c=7.6 from the GA-NIFS refit
    (arXiv:2603.00232), then FWHM = 2.3548*sigma. It is used only to size the
    mask, so an approximation is fine, but treat it as approximate; STPSF is the
    more careful choice if you want a real PSF model.
  * The default cube spaxel is 0.1 arcsec, close to the slice width, so the PSF
    is only ~2 pixels FWHM. `--scalexy 0.05` is a common choice for point
    sources. All mask sizes below are converted through the cube's own pixel
    scale, and the minimum mask radius is given in arcsec (`--min-aper-rad-arcsec`)
    rather than pixels, so the settings mean the same thing at either scale.
  * The NIRSpec IFU field of view is only ~3 x 3 arcsec, so a large mask leaves
    little sky. If the printed "background-model term" is large, reduce
    `--fwhm-scale`, or skip AstroBkgInterp for that cube (see notes).
  * Optional, off by default (STScI defaults are kept unless you ask):
    `--clean-flicker` (1/f noise; masks the IFU slices, so only the sky between
    slices is used to fit the noise), `--badpix-selfcal`, and `--pixel-replace`
    (Stage 3, interpolates over bad pixels before cube building).
  * Outlier detection uses the jwst defaults (`kernel_size='7 7'`,
    `threshold_percent=99.8`). The MRS "11 1" kernel is tuned to MRS's
    detector layout and is not carried over; override with `--outlier-kernel`
    and `--outlier-threshold` if needed.
  * Not ported from the MRS module: the second-pass centroid refinement, PSF
    deblending, and `estimate_empirical_flux_error()`. The last one relies on
    the extraction radius being a known multiple of the PSF FWHM, which is true
    for MRS but not for NIRSpec, where the radius comes from a reference file.

Pipeline
--------
  1. Query MAST for every NIRSPEC/IFU observation in the proposal and download
     every uncalibrated exposure (science, background and imprint alike).
  2. `calwebb_detector1` on every exposure.
  3. Classify the rate files by header: science (NRS_IFU, not background, not
     imprint), imprint (IS_IMPRT), dedicated background (BKGDTARG). Dedicated
     backgrounds are not processed further (AstroBkgInterp handles background).
  4. `calwebb_spec2` per science exposure, with the matching imprint members
     attached. `bkg_subtract`, `cube_build` and `extract_1d` are skipped.
  5. Group Stage-2 `_cal` files by (TARGNAME, epoch); an epoch is a cluster of
     exposures whose MJDs are all within `--max-epoch-gap-days` of neighbours.
  6. `calwebb_spec3` once per group: `cube_build` (default output_type='band'),
     outlier detection, `master_background` and `extract_1d` skipped.
  7. Per cube: locate the source (given, or DAOStarFinder), run AstroBkgInterp
     on SCI/ERR, and save `bgsub/<cube-name>_abi_bgsub_s3d.fits` plus a
     quick-look PNG.

Usage
-----
    python nirspec_ifu_auto_pipeline.py --proposal 1234

    python nirspec_ifu_auto_pipeline.py --proposal 1234 --scalexy 0.05 \\
        --src-x 30.2 --src-y 31.7 --tag visit1

    # iterate on background settings without redoing the reduction
    python nirspec_ifu_auto_pipeline.py --proposal 1234 --from-stage3 --fwhm-scale 2.0

Then extract by hand:
    from nirspec_ifu_auto_pipeline import extract_spectrum_manual
    wave, flux, err, (x, y) = extract_spectrum_manual(
        'bgsub/<cube>_abi_bgsub_s3d.fits', ra=166.45643, dec=-0.0348)

Notes
-----
- Requires `AstroBkgInterp`: pip install git+https://github.com/brynickson/AstroBkgInterp.git
- Needs `combine_nirspec_orders.py` and `mast_uncal_download.py` next to this
  script (the same two modules the MRS pipeline imports).
- `--src-x/--src-y` (pixel) or `--ra/--dec` (deg): give one pair, not both. With
  several cubes in one run (several epochs or gratings) the pixel position of a
  given source can differ between cubes, so for multi-cube runs prefer
  `--ra/--dec` (resolved in each cube's own WCS) or leave both unset.
- If auto-locate finds no source, that cube is skipped with a warning; re-run
  with `--from-stage3` and an explicit position.
- AstroBkgInterp uses `multiprocessing.Pool`, which on macOS spawns rather than
  forks: run from a real script (under `if __name__ == "__main__":`) or a
  notebook, not by piping into `python` on stdin.
- Re-running with `--skip-download` reuses `_uncal.fits` files already in
  `<outdir>/uncal/`. Detector1 and Spec2 skip exposures whose output already
  exists unless `--redo` is given.
"""

import argparse
import glob
import logging
import os
import re
import time
import warnings
from collections import defaultdict

import numpy as np
from tqdm import tqdm


warnings.simplefilter("ignore", RuntimeWarning)

# ---------------------------------------------------------------------------
# CRDS setup (must happen before importing jwst/crds-dependent modules)
# ---------------------------------------------------------------------------
if os.getenv("CRDS_PATH") is None:
    os.environ["CRDS_PATH"] = os.path.join(os.path.expanduser("~"), "crds_cache")
if os.getenv("CRDS_SERVER_URL") is None:
    os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

# jwst/stpipe/CRDS log verbosity is controlled by --verbose in main() below
# (default: WARNING-only, to keep the console to our own print()s/tqdm bars).
# NB: setLevel() alone is not enough -- stpipe resets these to INFO on the
# first Step.call(). See jwst_logging.configure_jwst_logging for why, and
# for the override-proof version applied in main().
_QUIET_LOGGERS = ("stpipe", "jwst", "stcal", "stdatamodels", "tweakwcs",
                  "CRDS", "asdf")
for _name in _QUIET_LOGGERS:
    logging.getLogger(_name).setLevel(logging.WARNING)

from jspec.pipelines.jwst_logging import (  # noqa: E402
    configure_jwst_logging, configure_from_env, set_env_level,
)

# Applied at import, not just in main(): on spawn platforms the
# multiprocessed jwst steps re-import this module in child processes
# that never reach main(), and those children share our stderr.
configure_from_env()

from astropy.io import fits                     # noqa: E402
from astropy.stats import sigma_clipped_stats    # noqa: E402

import jwst                                      # noqa: E402
from jwst.pipeline import Detector1Pipeline, Spec2Pipeline, Spec3Pipeline  # noqa: E402
from jwst.associations import asn_from_list as afl                        # noqa: E402
from jwst.associations.lib.rules_level2_base import DMSLevel2bBase        # noqa: E402
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base       # noqa: E402

from jspec.pipelines.combine_nirspec_orders import get_mjd, group_files_by_epoch, mjd_to_datestr  # noqa: E402
from jspec.pipelines.mast_uncal_download import download_uncal  # noqa: E402  (keep next to this script)
from jspec.layout import (  # noqa: E402
    ensure_epoch_tree, find_epoch_dirs, find_staged_uncal, stage_files)
from jspec.manifest import MANIFEST_NAME, record_product  # noqa: E402

# AstroBkgInterp -- https://github.com/brynickson/AstroBkgInterp
from AstroBkgInterp.AstroBkgInterp import AstroBkgInterp  # noqa: E402
from jspec.pipelines._multiproposal import main_per_proposal


NRS_IFU_EXP_TYPE = "NRS_IFU"
FAILURES = []            # (stage, file/product, message) -- printed at the end


# ===========================================================================
# Header helpers, classification and association writers
# ===========================================================================

def _hdr(path):
    return fits.getheader(path, 0)


def classify_rate_files(rate_files):
    """Split Detector1 outputs by what the headers say they are.

    Returns (science, imprint, background, other):
      * imprint    -- IS_IMPRT = T (MSA-closed leakcal exposures)
      * background -- BKGDTARG = T, and not an imprint (dedicated background pointing)
      * science    -- everything else with EXP_TYPE = NRS_IFU
      * other      -- not NRS_IFU at all (ignored, with a note in main())
    """
    sci, imprint, bkg, other = [], [], [], []
    for f in rate_files:
        h = _hdr(f)
        if h.get("EXP_TYPE") != NRS_IFU_EXP_TYPE:
            other.append(f)
        elif bool(h.get("IS_IMPRT")):
            imprint.append(f)
        elif bool(h.get("BKGDTARG")):
            bkg.append(f)
        else:
            sci.append(f)
    return sci, imprint, bkg, other


def _setup_key(h):
    """(detector, grating, filter): the configuration an imprint must share
    with the science exposure it is subtracted from."""
    return (h.get("DETECTOR"), h.get("GRATING"), h.get("FILTER"))


def match_imprints(sci_file, imprint_files, max_gap_days):
    """Imprint exposures to attach to one science exposure's Level-2 association.

    Candidates must share detector/grating/filter, must not be imprints of a
    dedicated background pointing (BKGDTARG), and must be within
    `max_gap_days` of the science exposure. Among those, the ones from the
    same observation number are preferred; if none share the observation
    number, all candidates are attached. jwst's ImprintStep then does the
    final match by observation number and dither position, and uses the single
    imprint available if there is only one (see its docstring). Returns a list
    of file paths, possibly empty."""
    hs = _hdr(sci_file)
    mjd_s = get_mjd(sci_file)
    cands = []
    for f in imprint_files:
        h = _hdr(f)
        if bool(h.get("BKGDTARG")) or _setup_key(h) != _setup_key(hs):
            continue
        mjd_i = get_mjd(f)
        if mjd_s is not None and mjd_i is not None and abs(mjd_s - mjd_i) > max_gap_days:
            continue
        cands.append((f, h))
    same_obs = [f for f, h in cands if h.get("OBSERVTN") == hs.get("OBSERVTN")]
    return same_obs or [f for f, _ in cands]


def writel2asn(sci_file, imprint_files, selfcal_files, asnfile, prodname="Level2"):
    """Level-2 association for one science exposure: imprint members (if any)
    and, optionally, selfcal members. No dedicated-background members --
    background handling is done with AstroBkgInterp on the Stage-3 cube."""
    asn = afl.asn_from_list([sci_file], rule=DMSLevel2bBase, product_name=prodname)
    members = asn["products"][0]["members"]
    for f in imprint_files:
        members.append({"expname": f, "exptype": "imprint"})
    for f in selfcal_files:
        members.append({"expname": f, "exptype": "selfcal"})
    _, serialized = asn.dump()
    with open(asnfile, "w") as fh:
        fh.write(serialized)


def writel3asn(sci_files, asnfile, prodname):
    asn = afl.asn_from_list(sci_files, rule=DMS_Level3_Base, product_name=prodname)
    _, serialized = asn.dump()
    with open(asnfile, "w") as fh:
        fh.write(serialized)


def _safe_name(s):
    """File-name-safe target name (spaces and punctuation -> underscores)."""
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", str(s)).strip("_") or "UNKNOWN_TARGET"


# ===========================================================================
# Download and pre-flight checks
# ===========================================================================

def query_and_download(proposal_padded, proposal_num, uncal_dir, dataset_ids=None):
    """Download every NIRSpec/IFU uncal file of the proposal -- see
    mast_uncal_download.py. Downloads are verified, retried and restart-safe,
    and <outdir>/mast_uncal_manifest.txt lists what MAST holds.
    (Exclusive-access data: export MAST_API_TOKEN first.)"""
    return download_uncal(["NIRSPEC/IFU"], proposal_padded, proposal_num, uncal_dir,
                          name_contains=("nrs1", "nrs2"), dataset_ids=dataset_ids)


def check_ifu_coverage(uncal_files):
    """Pre-flight, on the uncal headers, before the slow stages. Warns if:
      * a high-resolution grating (G140H/G235H/G395H) exposure is missing NRS1
        or NRS2 (both are needed for full wavelength coverage);
      * a setup has fewer dither positions than NUMDTHPT says.
    Also prints an informational note if science exposures exist but there are
    no imprint (leakcal) exposures at all. (Expected if you used --dataset to
    select a subset.)"""
    dets_by_stem = defaultdict(set)
    grating_by_stem = {}
    for f in uncal_files:
        parts = os.path.basename(f).split("_")
        det = next((p for p in parts if p.startswith("nrs")), "?")
        stem = "_".join(parts[:3])
        dets_by_stem[stem].add(det)
        grating_by_stem.setdefault(stem, str(_hdr(f).get("GRATING", "")).upper())

    notes = []
    for stem, dets in sorted(dets_by_stem.items()):
        if grating_by_stem.get(stem, "").endswith("H") and dets != {"nrs1", "nrs2"}:
            notes.append(f"{stem} ({grating_by_stem[stem]}): only {', '.join(sorted(dets))} present "
                         f"(high-resolution gratings need nrs1 AND nrs2)")

    dith, expected = defaultdict(set), {}
    n_sci = n_imprint = 0
    for f in uncal_files:
        h = _hdr(f)
        if h.get("EXP_TYPE") != NRS_IFU_EXP_TYPE:
            continue
        if bool(h.get("IS_IMPRT")):
            n_imprint += 1
            continue
        if bool(h.get("BKGDTARG")):
            continue
        n_sci += 1
        key = (h.get("TARGNAME"), h.get("OBSERVTN"), h.get("VISIT"),
               h.get("GRATING"), h.get("FILTER"), h.get("DETECTOR"))
        dith[key].add(h.get("PATT_NUM"))
        if h.get("NUMDTHPT"):
            try:
                expected[key] = int(h["NUMDTHPT"])
            except (TypeError, ValueError):
                pass
    for key, pts in sorted(dith.items(), key=lambda kv: str(kv[0])):
        if key in expected and len(pts) < expected[key]:
            notes.append(f"{key[0]} obs {key[1]} visit {key[2]} {key[3]}/{key[4]} {key[5]}: "
                         f"{len(pts)} of {expected[key]} dither positions present")

    if notes:
        print("\nCOVERAGE WARNINGS (a missing download? see <outdir>/mast_uncal_manifest.txt):")
        for n in notes:
            print(f"  ! {n}")
        print()
    else:
        print(f"Coverage check: {len(dets_by_stem)} exposure(s), no missing detectors or dither positions.")
    if n_sci and not n_imprint:
        print("  Note: no imprint (leakcal) exposures in this set, so imprint subtraction will not "
              "run. That is normal for programs that did not take them.")
    elif n_imprint:
        print(f"  {n_imprint} imprint exposure(s) found; they will be matched to science exposures "
              f"by detector/grating/filter.")
    return notes


# ===========================================================================
# Reduction stages: Detector1 -> Spec2
# ===========================================================================

def run_detector1(uncal_files, det1_dir, max_cores, redo=False):
    os.makedirs(det1_dir, exist_ok=True)
    det1dict = defaultdict(dict)
    if max_cores != "none":
        det1dict["jump"]["maximum_cores"] = max_cores

    for f in tqdm(uncal_files, desc="Detector1", unit="file"):
        rate = os.path.join(det1_dir, os.path.basename(f).replace("_uncal.fits", "_rate.fits"))
        if os.path.exists(rate) and not redo:
            continue                                   # already done (--redo to force)
        try:
            Detector1Pipeline.call(f, steps=det1dict, save_results=True, output_dir=det1_dir)
        except Exception as exc:
            tqdm.write(f"  Detector1 FAILED for {os.path.basename(f)}: {exc}")
            FAILURES.append(("Detector1", os.path.basename(f), str(exc)))

    return sorted(glob.glob(os.path.join(det1_dir, "*nrs[12]_rate.fits")))


def run_spec2(sci_rate, imprint_rate, asn_dir, spec2_dir, max_gap_days, redo=False,
              clean_flicker=False, badpix_selfcal=False):
    """Spec2 for every science exposure. Cube building and 1D extraction are
    left to Stage 3 / the manual extraction step; jwst's own background
    subtraction is off (AstroBkgInterp does that later)."""
    os.makedirs(asn_dir, exist_ok=True)
    os.makedirs(spec2_dir, exist_ok=True)

    spec2dict = defaultdict(dict)
    spec2dict["bkg_subtract"]["skip"] = True         # AstroBkgInterp handles background, later
    spec2dict["imprint_subtract"]["skip"] = False    # no-op unless imprint members are attached
    spec2dict["cube_build"]["skip"] = True           # Stage 3 does the cube building
    spec2dict["extract_1d"]["skip"] = True
    if clean_flicker:
        spec2dict["clean_flicker_noise"]["skip"] = False
        spec2dict["clean_flicker_noise"]["mask_science_regions"] = True   # mask the IFU slices
    if badpix_selfcal:
        spec2dict["badpix_selfcal"]["skip"] = False

    n_imprint_attached = 0
    for f in tqdm(sci_rate, desc="Spec2", unit="file"):
        cal = os.path.join(spec2_dir, os.path.basename(f).replace("_rate.fits", "_cal.fits"))
        if os.path.exists(cal) and not redo:
            continue                                   # already done (--redo to force)

        imprints = match_imprints(f, imprint_rate, max_gap_days)
        n_imprint_attached += bool(imprints)
        selfcal = []
        if badpix_selfcal:
            k = _setup_key(_hdr(f))
            selfcal = [g for g in sci_rate if _setup_key(_hdr(g)) == k]

        asnfile = os.path.join(asn_dir, os.path.basename(f).replace("_rate.fits", "_l2asn.json"))
        writel2asn(f, imprints, selfcal, asnfile)
        try:
            Spec2Pipeline.call(asnfile, steps=spec2dict, save_results=True, output_dir=spec2_dir)
        except Exception as exc:
            tqdm.write(f"  Spec2 FAILED for {os.path.basename(f)}: {exc}")
            FAILURES.append(("Spec2", os.path.basename(f), str(exc)))

    if imprint_rate:
        print(f"  Imprint members attached to {n_imprint_attached} of {len(sci_rate)} science exposure(s) "
              f"this run.")
    return sorted(glob.glob(os.path.join(spec2_dir, "*nrs[12]_cal.fits")))


# ===========================================================================
# Epoch grouping (on Stage-2 cal files, before Stage 3)
# ===========================================================================

def collect_existing_cubes(outdir, tag=None):
    """Find Stage-3 cubes already present in the epoch trees (--from-stage3).

    Returns (cube_name, cube_file, paths) tuples matching what the main
    loop builds, so the re-extraction half of the pipeline runs
    identically whether or not Stage 3 was just executed.
    """
    found = []
    for _label, dir_tag, date, epoch_path in find_epoch_dirs(outdir, "NIRSpec-IFU"):
        if tag and dir_tag != tag:
            continue
        paths = None
        for cube in sorted(glob.glob(os.path.join(epoch_path, "stage3", "*_s3d.fits"))):
            if paths is None:
                paths = ensure_epoch_tree(outdir, "NIRSpec-IFU", dir_tag, date)
            found.append((os.path.basename(cube).rsplit("_s3d.fits", 1)[0],
                           cube, paths))
    return found


def record_stage3_products(outdir, paths, tag, date):
    """Note this epoch's Stage-3 x1d spectra in the output product index.

    They are recorded but not flagged as recommended: for an IFU the
    product to analyse is the AstroBkgInterp re-extraction, and having
    both listed makes the two easy to compare.
    """
    for x1d in sorted(glob.glob(os.path.join(paths["stage3"], "*_x1d.fits"))):
        ecsv = x1d[:-5] + ".ecsv"
        if not os.path.exists(ecsv):
            try:
                from astropy.table import Table
                from jspec.columns import normalize_columns
                normalize_columns(Table.read(x1d)).write(
                    ecsv, format="ascii.ecsv", overwrite=True)
            except Exception:
                ecsv = ""
        record_product(outdir, "NIRSpec-IFU", tag, date, ecsv or x1d,
                       product="x1d", level="stage3", fits_path=x1d,
                       recommended=False, quiet=True)


def get_epoch_groups(cal_files, max_epoch_gap_days):
    """Group Stage-2 calibrated files by (TARGNAME, epoch), where an epoch is a
    cluster of exposures whose MJDs are all within max_epoch_gap_days of their
    neighbours. Returns {(targname, epoch_label): [cal_files]}."""
    by_target = defaultdict(list)
    for f in cal_files:
        by_target[fits.getval(f, "TARGNAME") or "UNKNOWN_TARGET"].append(f)

    groups = {}
    for targname, files in by_target.items():
        for cluster in group_files_by_epoch(files, max_gap_days=max_epoch_gap_days):
            mjds = [m for m in (get_mjd(f) for f in cluster) if m is not None]
            label = mjd_to_datestr(np.mean(mjds)) if mjds else "unknown-date"
            groups[(targname, label)] = cluster
    return groups


# ===========================================================================
# Stage 3: cubes per (target, epoch) group
# ===========================================================================

def _find_cubes(spec3_dir, product_name):
    """Every '<product_name>_*_s3d.fits' in spec3_dir. Matched case-insensitively,
    since the association machinery can change the case of product names."""
    prefix = product_name.lower() + "_"
    return sorted(os.path.join(spec3_dir, n) for n in os.listdir(spec3_dir)
                  if n.lower().startswith(prefix) and n.lower().endswith("_s3d.fits"))


def run_spec3_group(cal_files, asn_dir, spec3_dir, product_name, output_type="band",
                    scalexy=None, outlier_kernel="7 7", outlier_threshold=99.8,
                    pixel_replace=False):
    """Run Spec3 for one (target, epoch) group. With output_type='band' (default)
    this yields one cube per grating+filter, named
    '<product_name>_<grating>-<filter>_s3d.fits'; 'multi' gives one cube
    spanning all gratings. No background subtraction and no 1D extraction --
    both happen afterward (AstroBkgInterp + manual Extract1dStep call).
    Returns the list of cube paths (empty if none were produced)."""
    spec3dict = defaultdict(dict)
    spec3dict["cube_build"]["output_type"] = output_type
    if scalexy:
        spec3dict["cube_build"]["scalexy"] = float(scalexy)
    spec3dict["outlier_detection"]["skip"] = False
    spec3dict["outlier_detection"]["kernel_size"] = outlier_kernel
    spec3dict["outlier_detection"]["threshold_percent"] = float(outlier_threshold)
    spec3dict["master_background"]["skip"] = True    # AstroBkgInterp handles background, not jwst
    spec3dict["extract_1d"]["skip"] = True           # no automatic 1D product
    if pixel_replace:
        spec3dict["pixel_replace"]["skip"] = False

    asnfile = os.path.join(asn_dir, f"{product_name}_l3asn.json")
    writel3asn(cal_files, asnfile, product_name)
    try:
        Spec3Pipeline.call(asnfile, steps=spec3dict, save_results=True, output_dir=spec3_dir)
    except Exception as exc:
        tqdm.write(f"  Spec3 FAILED for {product_name}: {exc}")
        FAILURES.append(("Spec3", product_name, str(exc)))
        return []

    cubes = _find_cubes(spec3_dir, product_name)
    if not cubes:
        print(f"  WARNING: no s3d output found for {product_name}")
    return cubes


# ===========================================================================
# Cube geometry: wavelength, pixel scale, PSF FWHM
# ===========================================================================

def _wavelength_from_wcstab(cube_file):
    """Fallback: pull the wavelength array out of a WCS-TABLE ('-TAB') extension
    directly. Returns None if nothing usable is found (never raises)."""
    with fits.open(cube_file) as hdul:
        ordered = sorted(hdul, key=lambda h: h.name.upper() not in ("WCS-TABLE", "WCS-TAB"))
        for hdu in ordered:
            names = getattr(hdu.data, "names", None)
            if not names:
                continue
            match = next((n for n in names if n.lower() == "wavelength"), None)
            if match is None:
                match = next((n for n in names if "wave" in n.lower()), None)
            if match is None:
                continue
            col = np.asarray(hdu.data[match])
            candidates = [np.asarray(col.ravel()[0], dtype=float)] if col.dtype == object else []
            if col.ndim > 1:
                candidates.append(np.asarray(col[0], dtype=float))
                candidates.append(np.asarray(col).ravel().astype(float))
            else:
                candidates.append(col.astype(float))
            for cand in candidates:
                if cand.ndim == 1 and cand.size > 1 and np.all(np.diff(cand) != 0):
                    return cand
    return None


def get_cube_wavelength(cube_file):
    """Per-slice wavelength array (microns), one value per SCI slice along axis 0.

    Read from the cube's gwcs (via stdatamodels), the same source the jwst
    pipeline itself uses; falls back to a direct WCS-TABLE parse, then to a
    linear CRPIX3/CDELT3/CRVAL3 axis (only valid when CTYPE3 is not a '-TAB'
    axis). Raises RuntimeError if none of the three work."""
    wave = None
    try:
        from stdatamodels.jwst import datamodels
        with datamodels.open(cube_file) as model:
            nz, ny, nx = model.data.shape
            _, _, wave = model.meta.wcs(np.full(nz, nx // 2, dtype=float),
                                         np.full(nz, ny // 2, dtype=float),
                                         np.arange(nz, dtype=float))
            wave = np.asarray(wave, dtype=float)
            if not (wave.ndim == 1 and wave.size == nz and np.all(np.isfinite(wave))):
                wave = None
    except Exception as exc:
        warnings.warn(f"gwcs wavelength read failed for {cube_file} ({exc!r}); "
                       "falling back to WCS-TABLE / linear WCS.")

    if wave is None:
        wave = _wavelength_from_wcstab(cube_file)

    if wave is None:
        with fits.open(cube_file) as hdul:
            hdr = hdul["SCI"].header
            ctype3 = hdr.get("CTYPE3", "")
            if "TAB" in ctype3.upper():
                raise RuntimeError(
                    f"No usable wavelength found in {cube_file} via gwcs or WCS-TABLE, and "
                    f"CTYPE3={ctype3!r} is a tabular axis, so a linear fallback would be wrong.")
            nz = hdul["SCI"].data.shape[0]
            wave = (np.arange(nz) - hdr["CRPIX3"] + 1) * hdr["CDELT3"] + hdr["CRVAL3"]
            if hdr.get("CUNIT3", "").lower() in ("m", "meter", "metre"):
                wave = wave * 1e6  # m -> um
    return wave


def _pixel_scale_arcsec(cube_file):
    with fits.open(cube_file) as hdul:
        return abs(hdul["SCI"].header["CDELT1"]) * 3600.0


def _nirspec_fwhm_arcsec(wave):
    """Approximate NIRSpec IFU PSF FWHM (arcsec, geometric mean of the along-
    and across-slice values) vs wavelength (microns).

    Functional form from D'Eugenio et al. (2024):
        sigma = a + b * lambda * exp(-c / lambda)
    with a=0.076, b=0.013, c=7.6 (a refit reported in GA-NIFS, arXiv:2603.00232),
    and FWHM = 2.3548 * sigma. This gives ~0.18 arcsec at 1 micron and ~0.21 at
    5 microns. It is only used to size the source mask, where a rough number is
    enough; the true PSF is elliptical and depends on dither sampling."""
    wave = np.asarray(wave, dtype=float)
    sigma = 0.076 + 0.013 * wave * np.exp(-7.6 / wave)
    return 2.3548 * sigma


def _nirspec_fwhm_pixels(cube_file, wave=None):
    """PSF FWHM in cube pixels for every wavelength slice."""
    if wave is None:
        wave = get_cube_wavelength(cube_file)
    return _nirspec_fwhm_arcsec(wave) / _pixel_scale_arcsec(cube_file)


# ===========================================================================
# Source location (pixel space)
# ===========================================================================

def locate_source_pixel(cube_file, edge=2):
    """Auto-locate the brightest point source in the cube's median-collapsed
    frame, in pixel coordinates (DAOStarFinder, as in AstroBkgInterp's own
    tutorial). The detection FWHM follows the cube's pixel scale. Detections
    within `edge` pixels of the array edge are ignored (cube edges are prone
    to artefacts). Returns (src_x, src_y), or None if nothing was found."""
    from photutils.detection import DAOStarFinder

    with fits.open(cube_file) as hdul:
        data = np.asarray(hdul["SCI"].data, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        collapsed = np.nan_to_num(np.nanmedian(data, axis=0), nan=0.0)

    pix = _pixel_scale_arcsec(cube_file)
    fwhm_px = float(np.clip(0.19 / pix, 1.5, 8.0))

    _, median, std = sigma_clipped_stats(collapsed, sigma=3.0)
    sources = DAOStarFinder(fwhm=fwhm_px, threshold=3.0 * std)(collapsed - median)
    if sources is None or len(sources) == 0:
        return None

    # photutils renamed xcentroid/ycentroid -> x_centroid/y_centroid in v3.0
    xcol = "x_centroid" if "x_centroid" in sources.colnames else "xcentroid"
    ycol = "y_centroid" if "y_centroid" in sources.colnames else "ycentroid"

    ny, nx = collapsed.shape
    best, best_val = None, -np.inf
    for s in sources:
        x, y = float(s[xcol]), float(s[ycol])
        ix, iy = int(round(x)), int(round(y))
        if not (edge <= ix < nx - edge and edge <= iy < ny - edge):
            continue
        if collapsed[iy, ix] > best_val:
            best, best_val = (x, y), collapsed[iy, ix]
    return best


def resolve_source_pixel(cube_file, ra=None, dec=None, x=None, y=None):
    """Resolve a source position to pixel (x, y) in the cube's spatial frame,
    from RA/Dec (deg) or pixel x/y -- give one pair, not both. RA/Dec is
    converted with the cube's full WCS via `all_world2pix`, so any distortion
    terms in the WCS solution are accounted for."""
    have_xy = (x is not None) and (y is not None)
    have_radec = (ra is not None) and (dec is not None)
    if have_xy and have_radec:
        raise ValueError("Give either (x, y) or (ra, dec), not both.")
    if have_xy:
        return float(x), float(y)
    if have_radec:
        from astropy.wcs import WCS
        with fits.open(cube_file) as hdul:
            # naxis=2 keeps just the spatial axes; fobj=hdul lets astropy
            # resolve a '-TAB' spectral axis (multi-grating cubes) if present.
            wcs2d = WCS(hdul["SCI"].header, fobj=hdul, naxis=2)
        x_pix, y_pix = wcs2d.all_world2pix(ra, dec, 0)
        return float(x_pix), float(y_pix)
    raise ValueError("Provide either (x, y) or (ra, dec).")


# ===========================================================================
# Manual 1D extraction
# ===========================================================================

def extract_spectrum_manual(cube_file, out_file=None, ra=None, dec=None, x=None, y=None,
                             ifu_covar_scale=1.0, srctype="POINT", apply_apcorr=True,
                             return_meta=False):
    """Extract a flux-calibrated, background-subtracted 1D spectrum from a
    NIRSpec IFU cube at a given position.

    Accepts either (ra, dec) in degrees or (x, y) in pixels (see
    `resolve_source_pixel`) and runs `Extract1dStep` with:

      * `subtract_background=False`: AstroBkgInterp already removed the
        background, so pass the `*_abi_bgsub_s3d.fits` cube here, not the raw
        Stage-3 one; letting the step fit its own annulus background as well
        would subtract it twice.
      * `ifu_set_srctype='POINT'` and `apply_apcorr=True`: the step uses the
        reference-file point-source extraction radius and applies the
        wavelength-dependent aperture correction, so the result is the
        source's total flux in Jy rather than the flux inside a fixed
        aperture. Pass `srctype=None` to use the cube header's SRCTYPE.
      * `ifu_covar_scale`: inflates the errors for the spatial covariance from
        resampling. Nominal errors are optimistic without it.

    (`ifu_rfcorr` and `ifu_rscale` are MIRI-only in jwst and are not used.)

    Returns (wave, flux, err, (src_x, src_y)), wavelength-sorted, or
    (wave, flux, err, (src_x, src_y), meta) when `return_meta=True`."""
    from stdatamodels.jwst import datamodels
    from jwst.extract_1d import Extract1dStep

    src_x, src_y = resolve_source_pixel(cube_file, ra=ra, dec=dec, x=x, y=y)

    kwargs = dict(subtract_background=False, center_xy=[src_x, src_y],
                  apply_apcorr=apply_apcorr, ifu_covar_scale=ifu_covar_scale)
    if srctype is not None:
        kwargs["ifu_set_srctype"] = srctype

    cube = datamodels.open(cube_file)
    result = Extract1dStep.call(cube, **kwargs)
    if out_file:
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        result.save(out_file)

    tab = result.spec[0].spec_table
    wave = np.asarray(tab["WAVELENGTH"], dtype=float)
    flux = np.asarray(tab["FLUX"], dtype=float)
    err = np.asarray(tab["FLUX_ERROR"] if "FLUX_ERROR" in tab.dtype.names else tab["ERROR"],
                     dtype=float)
    order = np.argsort(wave)
    wave, flux, err = wave[order], flux[order], err[order]

    try:
        flux_unit = str(tab.columns["FLUX"].unit or "Jy")
    except Exception:
        flux_unit = "Jy"

    meta = {
        "flux_unit": flux_unit, "src_x": src_x, "src_y": src_y,
        "n_good": int(np.isfinite(flux).sum()), "srctype": srctype,
        "apply_apcorr": apply_apcorr, "ifu_covar_scale": ifu_covar_scale,
        "out_file": out_file,
    }

    cube.close()
    result.close()

    if return_meta:
        return wave, flux, err, (src_x, src_y), meta
    return wave, flux, err, (src_x, src_y)


# ===========================================================================
# AstroBkgInterp: error-array repair and the background-subtraction wrapper
# ===========================================================================

def _fill_bad_uncertainties(err):
    """Replace non-finite / non-positive uncertainties with that wavelength
    slice's median uncertainty (or the cube-wide median, for a slice with
    nothing usable at all).

    AstroBkgInterp only runs `interp_nans` on the *science* slice, never on the
    error slice, then propagates `sqrt(err**2 + bkg_err**2)`. A single NaN in
    ERR therefore survives to the output, and since a cube's ERR is NaN outside
    the IFU footprint, the propagated ERR comes back ~100% NaN. Filling first
    keeps the propagation meaningful; the originally-bad pixels are restored to
    NaN afterwards."""
    err = np.asarray(err, dtype=np.float32)
    bad = ~np.isfinite(err) | (err <= 0)
    if not bad.any():
        return err, bad

    good = err[~bad]
    global_med = float(np.median(good)) if good.size else 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
        axes = tuple(range(1, err.ndim)) if err.ndim > 1 else None
        slice_med = np.nanmedian(np.where(bad, np.nan, err), axis=axes)
    slice_med = np.where(np.isfinite(slice_med) & (slice_med > 0), slice_med, global_med)

    fill = (np.broadcast_to(slice_med.reshape((-1,) + (1,) * (err.ndim - 1)), err.shape)
            if err.ndim > 1 else np.full(err.shape, global_med, dtype=np.float32))
    err = np.where(bad, fill.astype(np.float32), err)
    return err, bad


def _finalize_err_cube(err_out, orig_err, bad_mask, shape):
    """Coerce AstroBkgInterp's propagated error back to the cube's shape and
    restore the input cube's own NaNs. AstroBkgInterp wraps the masked error in
    an extra leading axis, so a cube run returns (nz, 1, ny, nx) rather than
    (nz, ny, nx). Squeeze it, fall back to the input ERR if it is still
    unusable, and re-NaN the pixels that were only filled to keep the
    propagation alive."""
    arr = np.asarray(err_out, dtype=np.float32) if err_out is not None else None
    if arr is not None and arr.shape != shape:
        squeezed = np.squeeze(arr)
        arr = squeezed if squeezed.shape == shape else None

    if arr is None or not np.isfinite(arr).any():
        print("  WARNING: AstroBkgInterp returned an unusable propagated ERR "
              "(wrong shape or all-NaN) -- keeping the input cube's ERR instead.")
        return np.asarray(orig_err, dtype=np.float32)

    still_bad = ~np.isfinite(arr) | (arr <= 0)
    if still_bad.any():
        arr = np.where(still_bad, np.asarray(orig_err, dtype=np.float32), arr)
    if bad_mask is not None and bad_mask.any():
        arr = np.where(bad_mask, np.float32(np.nan), arr)
    return arr


def _background_model_error(diff, src_x, src_y, fwhm, bin_size, valid=None):
    """Per-slice uncertainty of the fitted background model.

    AstroBkgInterp's own `bkg_err` is unusable (it squares and sums the lstsq
    residuals and broadcasts that scalar through the polynomial basis, so it is
    an SSR over thousands of pixels rather than a per-pixel error). Estimate it
    from the actual residual scatter instead: the background is a smooth
    surface fitted to many pixels, so sigma_bkg ~ sigma_pixel / bin_size, with
    sigma_pixel taken from the robust scatter of the background-subtracted
    slice away from the source.

    `valid` marks real spaxels: the cube's padding outside the IFU footprint
    can be finite zeros rather than NaN, and including it would return the
    noise of the padding instead of the noise of the sky.

    Returns one value per slice, to be added in quadrature to the cube ERR."""
    nz, ny, nx = diff.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - float(src_x), yy - float(src_y))
    fwhm = np.atleast_1d(np.asarray(fwhm, dtype=float))

    out = np.zeros(nz, dtype=np.float64)
    for i in range(nz):
        f = fwhm[i] if fwhm.size > 1 else fwhm[0]
        sl = diff[i]
        if valid is not None:
            ok = valid[i] if valid.ndim == 3 else valid
        else:
            ok = np.isfinite(sl) & (sl != 0)
        m = ok & np.isfinite(sl) & (r > 3.0 * f)
        if m.sum() < 32:
            m = ok & np.isfinite(sl)
        if m.sum() < 8:
            continue
        v = sl[m]
        sigma = 1.4826 * np.median(np.abs(v - np.median(v)))
        out[i] = sigma / max(float(bin_size), 1.0)
    # A slice whose background fit failed outright leaves sigma=0; fall back to
    # the cube-wide median so it is not silently treated as noiseless.
    good = out > 0
    if good.any():
        out[~good] = np.median(out[good])
    return out


def run_astrobkginterp_background(cube_file, src_x, src_y, out_dir, product_name,
                                   bkg_mode="polynomial", aper_rad=7, ann_width=6,
                                   fwhm_scale=2.5, poly_k=3, bin_size=9,
                                   cube_resolution="high", pool_size=None,
                                   min_aper_rad_arcsec=0.3, use_abi_err=False):
    """Run AstroBkgInterp on the cube's SCI/ERR arrays: mask a wavelength-
    dependent aperture around (src_x, src_y) and fit/subtract a background per
    wavelength slice. Saves the background-subtracted cube (all other
    extensions carried over unchanged) plus a quick-look PNG, and returns the
    output cube path.

    Source-mask size: in cube mode AstroBkgInterp ignores `aper_rad` and
    recomputes it per slice as ceil(fwhm_scale * fwhm(lambda)), so `fwhm_scale`
    is the real knob. If the mask is too small, the PSF wings are fitted as
    background and subtracted off the source. `min_aper_rad_arcsec` sets a floor
    on the mask radius in arcsec (converted to pixels with the cube's pixel
    scale, so it behaves the same at 0.05 and 0.1 arcsec spaxels). The NIRSpec
    IFU field is only ~3 arcsec across, so watch the mask against the FOV.

    `use_abi_err` is False because AstroBkgInterp's background-error term is
    wrong (see `_background_model_error`); the model uncertainty is estimated
    from the residual scatter instead.

    The ERR extension is sanitized on the way in and repaired on the way out
    (`_fill_bad_uncertainties` / `_finalize_err_cube`)."""
    os.makedirs(out_dir, exist_ok=True)

    with fits.open(cube_file) as hdul:
        data = np.nan_to_num(hdul["SCI"].data, nan=0.0)
        has_err = "ERR" in hdul
        bad_err_mask = None
        if has_err:
            orig_err = np.asarray(hdul["ERR"].data, dtype=np.float32)
            err, bad_err_mask = _fill_bad_uncertainties(orig_err)
            if bad_err_mask.any():
                print(f"  ERR: filled {bad_err_mask.mean() * 100:.1f}% non-finite/non-positive "
                      f"uncertainties before background fitting (restored to NaN afterwards).")
        else:
            # AstroBkgInterp calls np.zeros_like(err) unconditionally in cube
            # mode and crashes on err=None -- use a uniform placeholder.
            print("  WARNING: no ERR extension in cube -- using a uniform placeholder array "
                  "(AstroBkgInterp requires a non-None err array for cube input); "
                  "propagated errors will not be meaningful.")
            orig_err = np.ones_like(data)
            err = orig_err

        wave = get_cube_wavelength(cube_file)
        pix_arcsec = _pixel_scale_arcsec(cube_file)
        fwhm = _nirspec_fwhm_pixels(cube_file, wave=wave)
        min_aper_rad_px = float(min_aper_rad_arcsec) / pix_arcsec

        # AstroBkgInterp sets aper_rad = ceil(fwhm_scale * fwhm[i]) per slice, so
        # a floor on the mask radius has to be applied through the FWHM array.
        fwhm_eff = np.maximum(fwhm, min_aper_rad_px / max(fwhm_scale, 1e-6))
        r_slice = np.ceil(fwhm_scale * fwhm_eff)
        ny, nx = data.shape[1:]
        print(f"  Cube {nx}x{ny} px at {pix_arcsec:.3f} arcsec/px; source mask radius "
              f"{r_slice.min():.0f}-{r_slice.max():.0f} px "
              f"({r_slice.min() * pix_arcsec:.2f}-{r_slice.max() * pix_arcsec:.2f} arcsec; "
              f"fwhm_scale={fwhm_scale}); PSF FWHM {fwhm.min():.1f}-{fwhm.max():.1f} px")
        if r_slice.max() * 2 > 0.6 * min(nx, ny):
            print("  WARNING: the source mask spans more than 60% of the cube's short side, so "
                  "little sky is left to constrain the fit. Consider a smaller --fwhm-scale.")

        bi = AstroBkgInterp()
        bi.src_x, bi.src_y = src_x, src_y
        bi.aper_rad = aper_rad
        bi.ann_width = ann_width
        bi.fwhm = fwhm_eff
        bi.fwhm_scale = fwhm_scale
        bi.bkg_mode = bkg_mode
        bi.k = poly_k
        bi.bin_size = bin_size
        bi.cube_resolution = cube_resolution
        bi.pool_size = pool_size if pool_size is not None else max(1, (os.cpu_count() or 2) - 1)

        diff, bkg, mask, err_out = bi.run(data, err=err)

        out_hdul = hdul.copy()
        out_hdul["SCI"].data = diff.astype(hdul["SCI"].data.dtype)
        if has_err:
            if use_abi_err:
                err_final = _finalize_err_cube(err_out, orig_err, bad_err_mask, data.shape)
            else:
                bkg_err = _background_model_error(
                    diff, src_x, src_y, fwhm_eff, bin_size,
                    valid=np.isfinite(orig_err) & (orig_err > 0))
                err_final = np.sqrt(orig_err.astype(np.float64) ** 2
                                    + bkg_err[:, None, None] ** 2).astype(np.float32)
                err_final[bad_err_mask] = np.nan
                print(f"  ERR: background-model term adds a median "
                      f"{np.nanmedian(bkg_err):.3g} in quadrature "
                      f"(AstroBkgInterp's own term ignored -- see docstring).")
            out_hdul["ERR"].data = err_final.astype(hdul["ERR"].data.dtype)

        out_path = os.path.join(out_dir, f"{product_name}_abi_bgsub_s3d.fits")
        out_hdul.writeto(out_path, overwrite=True)

    try:
        import matplotlib.pyplot as plt
        mid = data.shape[0] // 2
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        for a, img, title in zip(ax, (data[mid], bkg[mid], diff[mid]),
                                 ("Raw", "Background model", "Background-subtracted")):
            finite = img[np.isfinite(img) & (img != 0)]
            vmin, vmax = (np.percentile(finite, [2, 99.5]) if finite.size else (None, None))
            a.imshow(img, origin="lower", vmin=vmin, vmax=vmax)
            a.set_title(f"{title}  ({wave[mid]:.2f} um)")
            a.scatter(src_x, src_y, marker="+", c="red", s=60)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{product_name}_abi_bgsub_quicklook.png"))
        plt.close(fig)
    except Exception as e:
        print(f"  (skipped quick-look plot: {e})")

    return out_path


# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proposal", required=True, metavar="ID [ID ...]", help="JWST proposal/program number, e.g. 1234. Several may be given (--proposal A B C, or A,B,C); they are downloaded and reduced one after another into the same --outdir, and sort into epochs by observation date rather than by program.")
    parser.add_argument("--outdir", required=True,
                    help="Output directory for this target. Instrument and epoch subdirectories are created inside it: <outdir>/<INSTRUMENT>/<target>_<date>/{uncal,stage1,stage2,stage3,reextract,final}")
    parser.add_argument("--skip-download", action="store_true",
                         help="Reuse existing files in <outdir>/uncal/ instead of querying MAST")
    parser.add_argument("--from-stage3", action="store_true",
                         help="Skip download/Detector1/Spec2/Spec3 and reuse the '*_s3d.fits' cubes "
                              "already in <outdir>/stage3/ -- for iterating on background settings "
                              "(--src-x/--src-y/--ra/--dec, --fwhm-scale, --bkg-mode, ...) without "
                              "re-running the reduction.")
    parser.add_argument("--max-cores", default="half", choices=["none", "quarter", "half", "all"],
                         help="Cores to use for the jump step (default: half)")
    parser.add_argument("--log-file", default=None,
                         help="Also write the jwst/stpipe log to this file. Pairs well with "
                              "the default quiet console: clean terminal, full detail on "
                              "disk if a run needs debugging afterwards.")
    parser.add_argument("--verbose", action="store_true",
                         help="Show full jwst/stpipe/CRDS logging output instead of the default "
                              "WARNING-only console.")
    parser.add_argument("--redo", action="store_true",
                         help="Re-run Detector1 and Spec2 even if their outputs already exist")
    parser.add_argument("--dataset", nargs="+", default=None,
                         help="Restrict to specific dataset(s)/exposure(s), e.g. --dataset "
                              "jw01234001001_02101_00001. A file is kept if its name contains any of them.")

    # -- reduction options --
    parser.add_argument("--max-epoch-gap-days", type=float, default=1.0,
                         help="Exposures more than this many days apart (by MJD) are never combined "
                              "into the same Stage-3 cube; also the maximum time offset between an "
                              "imprint exposure and the science exposure it is matched to (default: 1.0)")
    parser.add_argument("--output-type", choices=["band", "grating", "multi"], default="band",
                         help="cube_build output_type: 'band' = one cube per grating+filter "
                              "(default), 'grating' = one per grating, 'multi' = one cube across "
                              "all gratings")
    parser.add_argument("--scalexy", type=float, default=None,
                         help="Cube spaxel size in arcsec. Default: the pipeline's (0.1). 0.05 is a "
                              "common choice for point sources.")
    parser.add_argument("--outlier-kernel", default="7 7",
                         help="outlier_detection kernel_size (default: '7 7', the jwst default)")
    parser.add_argument("--outlier-threshold", type=float, default=99.8,
                         help="outlier_detection threshold_percent (default: 99.8)")
    parser.add_argument("--clean-flicker", action="store_true",
                         help="Turn on 1/f noise cleaning (clean_flicker_noise) in Spec2, with the "
                              "IFU slices masked. Off by default.")
    parser.add_argument("--badpix-selfcal", action="store_true",
                         help="Turn on badpix_selfcal in Spec2, using the other exposures of the "
                              "same detector/grating/filter. Off by default.")
    parser.add_argument("--pixel-replace", action="store_true",
                         help="Turn on pixel_replace in Spec3 (interpolate over bad pixels before "
                              "cube building). Off by default.")
    parser.add_argument("--tag", default=None,
                         help="Display name for the target, used as the stem of the epoch "
                              "directory and of every product in it, e.g. --tag SN2024aecx "
                              "gives SN2024aecx_2025-06-01/. Default: the target name from "
                              "the file headers. Give this explicitly when the headers spell "
                              "the target inconsistently between visits.")

    # -- AstroBkgInterp background settings --
    parser.add_argument("--src-x", type=float, default=None,
                         help="Source x pixel for the AstroBkgInterp masking centre. If omitted, each "
                              "cube auto-locates its own brightest source via DAOStarFinder.")
    parser.add_argument("--src-y", type=float, default=None, help="Source y pixel; see --src-x.")
    parser.add_argument("--ra", type=float, default=None,
                         help="Source RA (deg), as an alternative to --src-x/--src-y, converted to "
                              "pixels through each cube's WCS. Give --ra/--dec or --src-x/--src-y, not both.")
    parser.add_argument("--dec", type=float, default=None, help="Source Dec (deg); see --ra.")
    parser.add_argument("--bkg-mode", choices=["simple", "polynomial"], default="polynomial",
                         help="AstroBkgInterp background-fitting mode (default: polynomial)")
    parser.add_argument("--aper-rad", type=float, default=7,
                         help="Source-mask radius in pixels. AstroBkgInterp ignores this for cubes "
                              "(it uses --fwhm-scale); kept for completeness (default: 7)")
    parser.add_argument("--ann-width", type=float, default=6,
                         help="Width in pixels of the annulus used to sample the background (default: 6)")
    parser.add_argument("--fwhm-scale", type=float, default=2.5,
                         help="Scales the wavelength-dependent PSF FWHM to set the per-slice "
                              "source-mask radius. Too small and the PSF wings are fitted as "
                              "background and subtracted off the source; too large and little sky "
                              "is left in the ~3 arcsec IFU field (default: 2.5)")
    parser.add_argument("--min-aper-rad-arcsec", type=float, default=0.3,
                         help="Floor on the source-mask radius, in arcsec (default: 0.3)")
    parser.add_argument("--use-abi-err", action="store_true",
                         help="Propagate AstroBkgInterp's own background-error term. Off by default "
                              "because it is buggy and inflates ERR by orders of magnitude.")
    parser.add_argument("--poly-k", type=int, default=3,
                         help="Degree of the 2D polynomial fit when --bkg-mode polynomial (default: 3)")
    parser.add_argument("--bin-size", type=int, default=9,
                         help="Sliding-window size (pixels) for the polynomial fit (default: 9)")
    parser.add_argument("--cube-resolution", choices=["low", "medium", "high"], default="high",
                         help="Sampling density for the polynomial background fit (default: high)")
    parser.add_argument("--pool-size", type=int, default=None,
                         help="Multiprocessing pool size for AstroBkgInterp (default: cpu count - 1; "
                              "1 forces single-threaded)")
    args = parser.parse_args()

    # Must happen before any pipeline step runs: whoever configures
    # logging first wins, and stpipe defaults to INFO.
    _level = "INFO" if args.verbose else "WARNING"
    set_env_level(_level)   # so spawned workers match the parent
    configure_jwst_logging(_level, log_file=getattr(args, "log_file", None))

    have_xy = (args.src_x is not None) and (args.src_y is not None)
    have_radec = (args.ra is not None) and (args.dec is not None)
    if have_xy and have_radec:
        raise SystemExit("Give either --src-x/--src-y or --ra/--dec, not both.")
    if (args.src_x is not None) != (args.src_y is not None):
        raise SystemExit("--src-x and --src-y must be given together.")
    if (args.ra is not None) != (args.dec is not None):
        raise SystemExit("--ra and --dec must be given together.")

    proposal_num = int(args.proposal)
    proposal_padded = f"{proposal_num:05d}"
    outdir = os.path.abspath(args.outdir)   # absolute: association files need it

    # Raw downloads stage here; epochs are only known once the headers
    # have been read, after which each gets its own directory tree.
    download_dir = os.path.join(outdir, "_downloaded_uncal")
    os.makedirs(download_dir, exist_ok=True)

    print(f"jwst version: {jwst.__version__}  |  output directory: {outdir}")
    t0 = time.perf_counter()

    # to_process entries are (cube_name, cube_file, paths), where paths is
    # the epoch's layout dict.
    if args.from_stage3:
        to_process = collect_existing_cubes(outdir, args.tag)
        if not to_process:
            raise SystemExit(
                f"--from-stage3: no existing '*_s3d.fits' cubes found under {outdir}.\n"
                f"Expected them in <outdir>/NIRSpec-IFU/<target>_<date>/stage3/.")
        print(f"--from-stage3: reusing {len(to_process)} existing cube(s) "
              f"(download/Detector1/Spec2/Spec3 skipped)")
    else:
        if args.skip_download:
            uncal_files = [f for f in find_staged_uncal(outdir, download_dir)
                            if re.search(r"nrs[12]_uncal\.fits$", os.path.basename(f))]
            if args.dataset:
                uncal_files = [f for f in uncal_files
                               if any(d in os.path.basename(f) for d in args.dataset)]
            print(f"--skip-download: using {len(uncal_files)} existing uncal file(s).")
        else:
            uncal_files = query_and_download(proposal_padded, proposal_num, download_dir,
                                             dataset_ids=args.dataset)
        if not uncal_files:
            raise SystemExit("No uncal files to process -- stopping.")

        check_ifu_coverage(uncal_files)

        # Epochs are split here, on the uncal headers, before any
        # calibration runs. Grouping later (after Stage 2, as this script
        # used to) left every epoch sharing one stage1/stage2 directory,
        # which makes a Level-3 association spanning two visits far too
        # easy to build by accident.
        epoch_groups = get_epoch_groups(uncal_files, args.max_epoch_gap_days)
        n_targets = len(set(k[0] for k in epoch_groups))
        print(f"{len(epoch_groups)} (target, epoch) group(s) across {n_targets} target(s) "
              f"(>{args.max_epoch_gap_days} day gap = new epoch)\n")

        to_process = []
        for (targname, epoch), group_uncal in epoch_groups.items():
            tag = args.tag or _safe_name(targname)
            paths = ensure_epoch_tree(outdir, "NIRSpec-IFU", tag, epoch)
            group_name = paths["stem"]
            print(f"=== {group_name}: {len(group_uncal)} exposure(s) -> "
                  f"{os.path.relpath(paths['epoch'], outdir)}")

            staged = stage_files(group_uncal, paths["uncal"])
            rate_files = run_detector1(staged, paths["stage1"], args.max_cores,
                                        redo=args.redo)
            if args.dataset:
                rate_files = [f for f in rate_files
                              if any(d in os.path.basename(f) for d in args.dataset)]

            sci_rate, imprint_rate, bkg_rate, other_rate = classify_rate_files(rate_files)
            print(f"  {len(sci_rate)} science, {len(imprint_rate)} imprint, "
                  f"{len(bkg_rate)} dedicated-background exposure(s)"
                  + (f"; ignoring {len(other_rate)} non-NRS_IFU file(s)" if other_rate else ""))
            if bkg_rate:
                print("  Dedicated-background exposures are not processed further -- "
                      "AstroBkgInterp masks the source on the science cube instead.")
            if not sci_rate:
                print(f"  WARNING: no NRS_IFU science exposures in {group_name} -- skipping.")
                continue

            cal_files = run_spec2(sci_rate, imprint_rate, paths["asn"], paths["stage2"],
                                   args.max_epoch_gap_days, redo=args.redo,
                                   clean_flicker=args.clean_flicker,
                                   badpix_selfcal=args.badpix_selfcal)
            if args.dataset:
                cal_files = [f for f in cal_files
                             if any(d in os.path.basename(f) for d in args.dataset)]
            if not cal_files:
                print(f"  WARNING: no Stage-2 products for {group_name} -- skipping.")
                continue

            for cube_file in run_spec3_group(
                    cal_files, paths["asn"], paths["stage3"], group_name,
                    output_type=args.output_type, scalexy=args.scalexy,
                    outlier_kernel=args.outlier_kernel,
                    outlier_threshold=args.outlier_threshold,
                    pixel_replace=args.pixel_replace):
                cube_name = os.path.basename(cube_file).rsplit("_s3d.fits", 1)[0]
                to_process.append((cube_name, cube_file, paths))
            record_stage3_products(outdir, paths, tag, epoch)

    if (have_xy or have_radec) and len(to_process) > 1:
        print(f"  NOTE: an explicit source position was given but there are {len(to_process)} cubes "
              f"in this run. A pixel position (--src-x/--src-y) is applied unchanged to every cube, "
              f"which is only right if they share a pixel grid; --ra/--dec is resolved separately in "
              f"each cube's own WCS and is the safer choice here.\n")

    bgsub_outputs = {}
    for cube_name, cube_file, paths in tqdm(to_process, desc="AstroBkgInterp", unit="cube"):
        bgsub_dir = paths["reextract"]
        try:
            if have_xy or have_radec:
                src_x, src_y = resolve_source_pixel(cube_file, ra=args.ra, dec=args.dec,
                                                     x=args.src_x, y=args.src_y)
            else:
                located = locate_source_pixel(cube_file)
                if located is None:
                    tqdm.write(f"  WARNING: auto-locate found no source for {cube_name} -- skipping "
                               f"(re-run with --from-stage3 and --src-x/--src-y or --ra/--dec).")
                    continue
                src_x, src_y = located
            tqdm.write(f"  {cube_name}: source at (x={src_x:.2f}, y={src_y:.2f})")

            bgsub_outputs[cube_name] = run_astrobkginterp_background(
                cube_file, src_x, src_y, bgsub_dir, cube_name,
                bkg_mode=args.bkg_mode, aper_rad=args.aper_rad, ann_width=args.ann_width,
                fwhm_scale=args.fwhm_scale, poly_k=args.poly_k, bin_size=args.bin_size,
                min_aper_rad_arcsec=args.min_aper_rad_arcsec, use_abi_err=args.use_abi_err,
                cube_resolution=args.cube_resolution, pool_size=args.pool_size,
            )
        except Exception as exc:
            tqdm.write(f"  AstroBkgInterp FAILED for {cube_name}: {exc}")
            FAILURES.append(("AstroBkgInterp", cube_name, str(exc)))

    t_end = time.perf_counter()
    print(f"\nDone in {(t_end - t0) / 60:.1f} min. Background-subtracted cubes:")
    for key, path in bgsub_outputs.items():
        print(f"  {key}: {os.path.relpath(path, outdir)}")
    print(f"\nProduct index: {os.path.join(outdir, MANIFEST_NAME)}")

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S):")
        for stage, what, msg in FAILURES:
            print(f"  [{stage}] {what}: {msg[:200]}")

    print("\nNo 1D spectrum was extracted automatically. Extract manually with "
          "extract_spectrum_manual() from this module, e.g.:\n"
          "    from jspec.pipelines.nirspec_ifu_auto_pipeline import extract_spectrum_manual\n"
          "    wave, flux, err, (x, y) = extract_spectrum_manual(\n"
          "        '<bgsub_cube>.fits', x=30.2, y=31.7)          # pixel coords, or:\n"
          "    wave, flux, err, (x, y) = extract_spectrum_manual(\n"
          "        '<bgsub_cube>.fits', ra=166.45643, dec=-0.0348)  # RA/Dec (deg)")


if __name__ == "__main__":
    main_per_proposal(main)