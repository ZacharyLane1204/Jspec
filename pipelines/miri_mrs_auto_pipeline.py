#!/usr/bin/env python3
"""
miri_mrs_auto_pipeline.py
===========================

Automated, proposal-level MIRI MRS (IFU) reduction pipeline, built on top of
JWPipeNB-MIRI-MRS.ipynb for the general Detector1/Spec2/Spec3 structure,
MRS_reductions.ipynb for the extraction-quality settings (fringe correction,
cube-building mode, outlier rejection), and AstroBkgInterp
(https://github.com/brynickson/AstroBkgInterp) for background estimation
and subtraction.

Rewrite history
---------------
  - `residual_fringe` was left at its pipeline default (OFF) in Stage 2.
    MRS is heavily fringed, especially in channels 3-4; without this step
    that shows up as high-frequency spikiness that doesn't appear in a
    properly-reduced spectrum. Now explicitly enabled, plus `ifu_rfcorr` at
    the 1D-extraction stage as a second, softer correction.
  - `cube_build.output_type` is `'multi'` (one cube spanning all
    channels/bands, built by the pipeline's own resampling) rather than
    `'band'` + post-hoc splicing, which showed visible flux jumps at band
    boundaries.
  - `outlier_detection` is tuned (`kernel_size='11 1'`,
    `threshold_percent=99.8`) for MRS's cosmic-ray/bad-pixel pattern.
  - `mrs_imatch` (band-to-band level matching) was fully removed from
    `jwst`'s `Spec3Pipeline` in v1.20.0 (Sept 2025) -- there's nothing to
    configure or skip anymore, so it's just gone from this script too. It
    isn't needed here regardless, since background handling is done
    explicitly (see below) rather than left to band-level normalization.
  - Background estimation/subtraction is now done entirely with
    AstroBkgInterp, run directly on the Stage-3 multi-band cube (SCI + ERR
    arrays): source masking (with a wavelength-dependent aperture that
    tracks the MRS PSF FWHM) followed by a polynomial background fit over
    the masked frame, per wavelength slice. `jwst`'s own
    `master_background`/`mrs_imatch` background paths are skipped entirely
    in favor of this -- there is no more `--background-mode`. This
    replaces an earlier version of this script that built its own
    reference background spectrum from an annulus or a set of boxes
    within the cube and fed it to `MasterBackgroundStep`; AstroBkgInterp
    does this same kind of job (and considerably more robustly -- 2D
    polynomial background modelling with wavelength-dependent source
    masking, rather than a 1D spectrum built from one region and force-
    subtracted uniformly) so that approach has been dropped.
  - Epoch (day-level) separation happens on the Stage-2 calibrated
    exposures *before* Stage 3, and Stage 3 is run once per (target,
    epoch) group, so exposures more than `--max-epoch-gap-days` apart
    never get combined into the same cube.
  - This script no longer builds a "final combined spectrum" -- there is
    no automatic `Extract1dStep` call, no per-slit/per-epoch overlap
    stitching, and no `final_spectrum/*_final_combined.{fits,ecsv}`
    output. The pipeline's product is now the AstroBkgInterp
    background-subtracted multi-band cube for each (target, epoch); the
    1D extraction (the actual 3D-cube-to-1D-spectrum step) is done
    manually afterward, via the `extract_spectrum_manual()` helper below,
    since the right extraction center is often a target-specific position
    that the header RA/Dec doesn't quite capture -- that decision is left
    to you, not automated here.
  - The second-pass centroid refinement (`refine_and_reprocess`) now
    collapses the entire cube into one inverse-variance weighted image and
    runs SExtractor-style detection on it (`find_source_centroid`), rather
    than fitting a centroid in every wavelength slice (or every bin of N
    slices) and combining the results. The per-slice approach needed each
    slice/bin to be an independent detection of the source, which a faint
    MRS target is not, and it leaned entirely on the cube's ERR array as
    its noise estimate -- so when AstroBkgInterp handed back an all-NaN
    ERR (see below), every slice failed its SNR test and the refinement
    raised rather than returning a position. Collapsing first puts all
    ~10,000 slices' signal into one frame, where the source is a solid
    detection, and falls back to a per-slice robust RMS when ERR is
    unusable. Detection runs after a `Background2D` mesh subtraction, so a
    supernova sitting on a bright extended host is found as its own
    compact source instead of being swallowed by a galaxy-sized segment.
  - AstroBkgInterp's error propagation is now repaired on the way out. It
    only NaN-interpolates the science slice, never the error slice, and
    then propagates `sqrt(err**2 + bkg_err**2)`; since an MRS multi-band
    cube's ERR is mostly NaN outside each channel's footprint, the
    propagated ERR came back 100% NaN. It also wraps the masked error in
    an extra axis, so a cube run returned (nz, 1, ny, nx) rather than
    (nz, ny, nx) -- a 4D ERR extension that no longer matched SCI.
    `run_astrobkginterp_background` now fills bad uncertainties before the
    fit and squeezes/repairs/re-NaNs the result afterward.
  - Wherever a source position is needed (AstroBkgInterp's masking center,
    and `extract_spectrum_manual()`), it can be given as RA/Dec (deg) *or*
    pixel x/y -- `resolve_source_pixel()` converts RA/Dec to pixel
    coordinates using the cube's full WCS via `all_world2pix` (which
    accounts for the WCS's distortion/SIP terms, if any), rather than the
    plain tangent-plane `wcs_world2pix`.

Pipeline
--------
  1. Query MAST for every MIRI/IFU (MRS) observation in the given proposal
     and download every uncalibrated exposure.
  2. `calwebb_detector1` on every exposure.
  3. `calwebb_spec2` with `residual_fringe` and `fringe` explicitly ON.
     Science files skip cube_build/extract_1d (Stage 3 handles that).
  4. Group Stage-2 calibrated science exposures by (TARGNAME, epoch), where
     an epoch is a cluster of exposures whose MJDs are all within
     `--max-epoch-gap-days` (default 1.0) of their neighbors.
  5. Run `calwebb_spec3` once per (target, epoch) group with
     `cube_build.output_type='multi'`, tuned `outlier_detection`,
     `master_background` skipped, and `extract_1d` skipped (no 1D
     product is made at this stage).
  6. Locate the source in the resulting cube -- either at a given
     `--src-x`/`--src-y` pixel position, or (default) automatically, via
     `DAOStarFinder` on the median-collapsed cube, matching
     AstroBkgInterp's own tutorial approach.
  7. Run `AstroBkgInterp` on the cube's SCI/ERR arrays: source-mask a
     wavelength-dependent aperture around that position (aperture radius
     scaled to the MRS PSF FWHM at each wavelength) and fit/subtract a 2D
     background per slice (`--bkg-mode polynomial` by default).
  8. Save the background-subtracted cube to
     `bgsub/<TARGNAME>_<epoch-date>_abi_bgsub_s3d.fits` (all other cube
     extensions -- WCS, weightmap, wavelength table -- are carried over
     unchanged from the Stage-3 cube), plus a quick-look PNG comparing the
     raw and background-subtracted collapsed images. No 1D spectrum is
     extracted automatically.

Usage
-----
    python miri_mrs_auto_pipeline.py --proposal 1523

    python miri_mrs_auto_pipeline.py --proposal 1523 --src-x 34.2 --src-y 51.7 \\
        --bkg-mode polynomial --poly-k 3 --bin-size 9 --tag visit1

Notes
-----
- Requires the `AstroBkgInterp` package: `pip install
  git+https://github.com/brynickson/AstroBkgInterp.git` (or clone + `pip
  install -e .`).
- `--src-x`/`--src-y` (pixel) or `--ra`/`--dec` (deg) give the source
  position for AstroBkgInterp's masking center -- give one pair, not both.
  Either way, this only makes sense for a single (target, epoch) group per
  run; for a whole-proposal run with multiple targets/epochs, leave both
  unset and let each group auto-locate its own source instead.
- If auto-locate finds no source (e.g. too faint, or a busy field), that
  group is skipped with a warning -- rerun with `--skip-download` and an
  explicit `--src-x`/`--src-y` or `--ra`/`--dec` for that target.
- To find a good x/y (with an uncertainty) for a target instead of relying
  on the header RA/Dec or eyeballing an image, use `find_source_centroid()`
  -- it collapses the cube into one weighted image, detects the source, and
  reports both a formal error and the scatter of independent sub-band
  measurements (the latter is the number to judge it by). See
  `refining.ipynb` for the full second-pass workflow.
- Note that `Extract1dStep`'s `ifu_rfcorr` (1D residual fringe correction)
  is silently disabled by `jwst` for multi-band cubes -- it only runs on a
  single IFU band. The Stage-2 `residual_fringe` step still applies.
- AstroBkgInterp uses `multiprocessing.Pool`, which on macOS spawns rather
  than forks. Run anything that calls `run_astrobkginterp_background` from
  a real script (under `if __name__ == "__main__":`) or a notebook -- not
  by piping a script into `python` on stdin, where the spawned workers
  cannot re-import `__main__` and the run falls back to crawling.
- Re-running with `--skip-download` reuses any `_uncal.fits` files already
  present in `<outdir>/uncal/`.
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

# Keep the jwst/stpipe/CRDS per-step chatter (CRDS reference lookups, step
# parameter dumps, etc.) out of the console -- only warnings/errors from
# these loggers get through. Our own print()s (and the tqdm bars below)
# are the only routine progress output.
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
from astroquery.mast import Observations        # noqa: E402

import jwst                                      # noqa: E402
from jwst.pipeline import Detector1Pipeline, Spec2Pipeline, Spec3Pipeline  # noqa: E402
from jwst.associations import asn_from_list as afl                        # noqa: E402
from jwst.associations.lib.rules_level2_base import DMSLevel2bBase        # noqa: E402
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base       # noqa: E402

from jspec.pipelines.combine_nirspec_orders import get_mjd, group_files_by_epoch, mjd_to_datestr  # noqa: E402
from jspec.pipelines.mast_uncal_download import download_uncal  # noqa: E402  (keep next to this script)
from jspec.layout import ensure_epoch_tree, find_epoch_dirs, product_basename  # noqa: E402
from jspec.manifest import MANIFEST_NAME, record_product  # noqa: E402

# AstroBkgInterp -- https://github.com/brynickson/AstroBkgInterp
# pip install git+https://github.com/brynickson/AstroBkgInterp.git
from AstroBkgInterp.AstroBkgInterp import AstroBkgInterp  # noqa: E402


# ===========================================================================
# Helpers ported from JWPipeNB-MIRI-MRS.ipynb
# ===========================================================================

def filter_list(files, key, value):
    return [f for f in files if fits.getheader(f).get(key) == value]


def writel2asn(onescifile, selfcalfiles, asnfile, prodname):
    """Level-2 association for one science exposure, with channel-matched
    selfcal files attached. No dedicated-background members here -- MRS
    background handling no longer uses jwst's own master_background/BKGDTARG
    path; see AstroBkgInterp below."""
    asn = afl.asn_from_list([onescifile], rule=DMSLevel2bBase, product_name=prodname)

    hdr = fits.getheader(onescifile)
    this_channel = hdr["CHANNEL"]

    for f in selfcalfiles:
        h = fits.getheader(f)
        if h.get("CHANNEL") == this_channel:
            asn["products"][0]["members"].append({"expname": f, "exptype": "selfcal"})

    _, serialized = asn.dump()
    with open(asnfile, "w") as outfile:
        outfile.write(serialized)


def writel3asn(scifiles, asnfile, prodname):
    asn = afl.asn_from_list(scifiles, rule=DMS_Level3_Base, product_name=prodname)
    _, serialized = asn.dump()
    with open(asnfile, "w") as outfile:
        outfile.write(serialized)


# ===========================================================================
# Reduction stages: download -> Detector1 -> Spec2
# ===========================================================================

def query_and_download(proposal_padded, proposal_num, uncal_dir, dataset_ids=None):
    """Download every MIRI/IFU (MRS) uncal file of the proposal -- see
    mast_uncal_download.py. Downloads are verified, retried and restart-safe,
    failures are reported, and <outdir>/mast_uncal_manifest.txt lists what
    MAST holds. (Exclusive-access data: export MAST_API_TOKEN first.)"""
    return download_uncal(["MIRI/IFU"], proposal_padded, proposal_num, uncal_dir,
                          name_contains=("mirifu",), dataset_ids=dataset_ids)


def check_mrs_coverage(uncal_files):
    """Pre-flight, on the uncal headers, before the slow stages: warn if an
    exposure is missing one of its two detectors (mirifushort = channels 1-2,
    mirifulong = channels 3-4) or a band has fewer dither positions than
    NUMDTHPT says. A partial download otherwise just gives a cube with
    missing channels / fewer dithers and no error. (Expected if you used
    --dataset to select a subset.)"""
    dets_by_stem = defaultdict(set)
    for f in uncal_files:
        parts = os.path.basename(f).split("_")
        det = next((p for p in parts if p.startswith("mirifu")), "?")
        dets_by_stem["_".join(parts[:3])].add(det)
    notes = [f"{stem}: only {', '.join(sorted(d))} present (expected mirifushort AND mirifulong)"
             for stem, d in sorted(dets_by_stem.items()) if d != {"mirifushort", "mirifulong"}]

    dith = defaultdict(set)
    expected = {}
    for f in uncal_files:
        h = fits.getheader(f)
        if h.get("BKGDTARG") is True:
            continue
        key = (h.get("TARGNAME"), h.get("OBSERVTN"), h.get("VISIT"), h.get("BAND"), h.get("DETECTOR"))
        dith[key].add(h.get("PATT_NUM"))
        if h.get("NUMDTHPT"):
            expected[key] = int(h["NUMDTHPT"])
    for key, pts in sorted(dith.items(), key=lambda kv: str(kv[0])):
        if key in expected and len(pts) < expected[key]:
            notes.append(f"{key[0]} obs {key[1]} visit {key[2]} band {key[3]} {key[4]}: "
                         f"{len(pts)} of {expected[key]} dither positions present")
    if notes:
        print("\nCOVERAGE WARNINGS (a missing download? see <outdir>/mast_uncal_manifest.txt):")
        for n in notes:
            print(f"  ! {n}")
        print()
    else:
        print(f"Coverage check: all {len(dets_by_stem)} exposure(s) have both MRS detectors and full dither sets.")
    return notes



# def query_and_download(proposal_padded, proposal_num, uncal_dir, dataset_ids=None):
#     print(f"Querying MAST for all MIRI/IMAGE observations in proposal "
#           f"{proposal_padded} ...")
#     obs_table = Observations.query_criteria(
#         instrument_name=["MIRI/IMAGE"],
#         provenance_name=["CALJWST"],
#         proposal_id=[str(proposal_num)],
#     )
#     print(f"  Found {len(obs_table)} observation/visit entries.")
 
#     file_criteria = {"productType": "SCIENCE", "productSubGroupDescription": "UNCAL", "calib_level": [1]}
#     pfilter = Observations.filter_products
 
#     downloads = set()
#     for exposure in obs_table:
#         products = Observations.get_product_list(exposure)
#         if len(products) == 0:
#             continue
#         matched = pfilter(products, **file_criteria)
#         if len(matched) > 0:
#             downloads.update(matched["dataURI"])
 
#     downloads = sorted(u for u in downloads if f"jw{proposal_padded}" in u)
 
#     if dataset_ids:
#         before = len(downloads)
#         downloads = sorted(u for u in downloads if any(d in u for d in dataset_ids))
#         print(f"  {before} uncalibrated exposure(s) found in the proposal; "
#               f"{len(downloads)} match --dataset {dataset_ids}.")
#         matched_ids = {d for d in dataset_ids if any(d in u for u in downloads)}
#         missing = [d for d in dataset_ids if d not in matched_ids]
#         if missing:
#             print(f"  WARNING: no matching files for --dataset entries: {missing}")
#     else:
#         print(f"  {len(downloads)} uncalibrated exposure(s) selected for download "
#               f"(all filters included).")
 
#     for uri in downloads:
#         Observations.download_file(uri, local_path=os.path.join(uncal_dir, os.path.basename(uri)))
 
#     uncal_files = sorted(glob.glob(os.path.join(uncal_dir, "*uncal.fits")))
#     if dataset_ids:
#         uncal_files = [f for f in uncal_files if any(d in os.path.basename(f) for d in dataset_ids)]
#     return uncal_files









def run_detector1(uncal_files, det1_dir, max_cores, redo=False):
    os.makedirs(det1_dir, exist_ok=True)
    det1dict = defaultdict(dict)
    if max_cores != "none":
        det1dict["jump"]["maximum_cores"] = max_cores

    for f in tqdm(uncal_files, desc="Detector1", unit="file"):
        rate = os.path.join(det1_dir, os.path.basename(f).replace("_uncal.fits", "_rate.fits"))
        if os.path.exists(rate) and not redo:
            continue                                   # already done (--redo to force)
        Detector1Pipeline.call(f, steps=det1dict, save_results=True, output_dir=det1_dir)

    return sorted(glob.glob(os.path.join(det1_dir, "*mirifu*rate.fits")))


def run_spec2(sci_rate, asn_dir, spec2_dir):
    os.makedirs(asn_dir, exist_ok=True)
    os.makedirs(spec2_dir, exist_ok=True)

    spec2dict = defaultdict(dict)
    spec2dict["bkg_subtract"]["skip"] = True   # AstroBkgInterp handles all background work, later
    spec2dict["fringe"]["skip"] = False
    spec2dict["residual_fringe"]["skip"] = False   # off by default in the pipeline; MRS needs it on
    spec2dict["cube_build"]["skip"] = True   # Stage 3 does the cube building
    spec2dict["extract_1d"]["skip"] = True

    for f in tqdm(sci_rate, desc="Spec2", unit="file"):
        asnfile = os.path.join(asn_dir, os.path.basename(f).replace("_rate.fits", "_l2asn.json"))
        writel2asn(f, sci_rate, asnfile, "Level2")
        Spec2Pipeline.call(asnfile, steps=spec2dict, save_results=True, output_dir=spec2_dir)

    return sorted(glob.glob(os.path.join(spec2_dir, "*mirifu*_cal.fits")))


# ===========================================================================
# Per-epoch directory staging (see jspec/layout.py for the layout)
# ===========================================================================

def stage_uncal(files, uncal_dir):
    """Link (or copy) an epoch's uncal files into its own uncal/ directory.

    Symlinks are used so that re-running does not duplicate tens of GB of
    raw data; if the filesystem refuses them (some network mounts do) the
    file is copied instead. Returns the staged paths, which are what the
    rest of the pipeline then works on, so Detector1 outputs land in the
    right epoch tree.
    """
    staged = []
    for src in files:
        dst = os.path.join(uncal_dir, os.path.basename(src))
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                import shutil
                shutil.copy2(src, dst)
        staged.append(dst)
    return sorted(staged)


def find_staged_uncal(outdir, download_dir):
    """Collect uncal files for --skip-download from anywhere sensible.

    Looks in the staging directory, in any existing epoch tree's uncal/,
    and in a plain <outdir>/uncal/ (the pre-layout location), so a tree
    built by an older version of this script, or by
    prepare_mrs_pfpc_epochs, still works. Duplicates are resolved by real
    path, since epoch directories hold symlinks back to the staging area.
    """
    candidates = glob.glob(os.path.join(download_dir, "*uncal.fits"))
    candidates += glob.glob(os.path.join(outdir, "uncal", "*uncal.fits"))
    candidates += glob.glob(os.path.join(outdir, "*", "*", "uncal", "*uncal.fits"))
    candidates += glob.glob(os.path.join(outdir, "*uncal.fits"))

    seen, unique = set(), []
    for path in sorted(candidates):
        key = os.path.realpath(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def collect_existing_cubes(outdir, tag=None):
    """Find Stage-3 cubes already present in the epoch trees.

    Used by --from-stage3. Returns (product_name, cube_file, paths)
    tuples matching what the main loop builds, so the re-extraction half
    of the pipeline is identical either way.
    """
    found = []
    for _label, dir_tag, date, epoch_path in find_epoch_dirs(outdir, "MIRI-MRS"):
        if tag and dir_tag != tag:
            continue
        stage3 = os.path.join(epoch_path, "stage3")
        cubes = sorted(glob.glob(os.path.join(stage3, "*_s3d.fits")))
        if not cubes:
            continue
        paths = ensure_epoch_tree(outdir, "MIRI-MRS", dir_tag, date)
        # One multi-band cube per epoch is the normal case; if a run left
        # several, take the one matching the epoch stem.
        stem = paths["stem"]
        preferred = [c for c in cubes if os.path.basename(c).startswith(stem)]
        for cube in (preferred or cubes):
            name = os.path.basename(cube).rsplit("_s3d.fits", 1)[0]
            found.append((name, cube, paths))
    return found


def record_stage3_products(outdir, paths, tag, date):
    """Note the Stage-3 x1d products in the output directory's index.

    Stage 3 is not the recommended product for MRS -- the AstroBkgInterp
    re-extraction is -- but it is recorded anyway, because comparing the
    two is the quickest check that the re-extraction behaved.
    """
    for x1d in sorted(glob.glob(os.path.join(paths["stage3"], "*_x1d.fits"))):
        ecsv = x1d[:-5] + ".ecsv"
        if not os.path.exists(ecsv):
            try:
                from astropy.table import Table
                from jspec.columns import normalize_columns
                table = normalize_columns(Table.read(x1d))
                table.write(ecsv, format="ascii.ecsv", overwrite=True)
            except Exception:
                ecsv = ""
        record_product(outdir, "MIRI-MRS", tag, date, ecsv or x1d,
                       product="x1d", level="stage3", fits_path=x1d,
                       recommended=False, quiet=True)


# ===========================================================================
# Epoch grouping (on uncal files, before any calibration runs)
# ===========================================================================

def get_epoch_groups(files, max_epoch_gap_days):
    """Group exposures by (TARGNAME, epoch).

    An epoch is a cluster of exposures whose MJDs are all within
    `max_epoch_gap_days` of their neighbours; the label is the mean date
    as YYYY-MM-DD. Returns {(targname, epoch_label): [files]}.

    Works on any JWST product with TARGNAME and a date in the header, but
    is called on the *uncal* files, so each epoch can be given its own
    directory tree before Detector1 runs. Grouping later (at Stage 3, as
    this script originally did) means all epochs share one stage1/stage2
    directory, which makes it far too easy to build a Level-3 association
    spanning two visits.
    """
    by_target = defaultdict(list)
    for f in cal_files:
        by_target[fits.getval(f, "TARGNAME") or "UNKNOWN_TARGET"].append(f)

    groups = {}
    for targname, files in by_target.items():
        clusters = group_files_by_epoch(files, max_gap_days=max_epoch_gap_days)
        for cluster in clusters:
            mjds = [m for m in (get_mjd(f) for f in cluster) if m is not None]
            label = mjd_to_datestr(np.mean(mjds)) if mjds else "unknown-date"
            groups[(targname, label)] = cluster
    return groups


# ===========================================================================
# Stage 3: one multi-band cube per (target, epoch) group
# ===========================================================================

def run_spec3_group(cal_files, asn_dir, spec3_dir, product_name):
    """Run Spec3 for one (target, epoch) group, producing a single cube
    spanning all channels/bands (`cube_build.output_type='multi'`), with no
    background subtraction and no 1D extraction -- both are done
    afterward, by AstroBkgInterp + a manual Extract1dStep call."""
    spec3dict = defaultdict(dict)
    spec3dict["cube_build"]["output_type"] = "multi"
    spec3dict["outlier_detection"]["skip"] = False
    spec3dict["outlier_detection"]["kernel_size"] = "11 1"
    spec3dict["outlier_detection"]["threshold_percent"] = 99.8
    spec3dict["master_background"]["skip"] = True   # AstroBkgInterp handles background, not jwst
    spec3dict["extract_1d"]["skip"] = True   # no automatic 1D product -- see module docstring

    asnfile = os.path.join(asn_dir, f"{product_name}_l3asn.json")
    writel3asn(cal_files, asnfile, product_name)

    Spec3Pipeline.call(asnfile, steps=spec3dict, save_results=True, output_dir=spec3_dir)

    out_files = sorted(glob.glob(os.path.join(spec3_dir, f"{product_name}_*_s3d.fits")))
    if not out_files:
        print(f"  WARNING: no s3d output found for {product_name}")
    return out_files[0] if out_files else None


# ===========================================================================
# Source location (pixel space) and background subtraction (AstroBkgInterp)
# ===========================================================================

def locate_source_pixel(cube_file):
    """Auto-locate the brightest point source in the cube's median-
    collapsed frame, in pixel coordinates -- matches AstroBkgInterp's own
    tutorial approach (DAOStarFinder on the collapsed cube), and gives
    AstroBkgInterp what it actually wants (src_x, src_y in pixels), rather
    than RA/Dec. Returns (src_x, src_y), or None if nothing was found."""
    from photutils.detection import DAOStarFinder

    with fits.open(cube_file) as hdul:
        data = np.nan_to_num(hdul["SCI"].data, nan=0.0)

    collapsed = np.zeros(data.shape[1:])
    for a in range(data.shape[1]):
        for b in range(data.shape[2]):
            collapsed[a, b] = np.median(data[:, a, b])

    mean, median, std = sigma_clipped_stats(collapsed, sigma=3.0)
    sources = DAOStarFinder(fwhm=2.0, threshold=3.0 * std)(collapsed - median)
    if sources is None or len(sources) == 0:
        return None

    # photutils renamed xcentroid/ycentroid -> x_centroid/y_centroid in v3.0;
    # support either so this doesn't break across photutils versions.
    xcol = "x_centroid" if "x_centroid" in sources.colnames else "xcentroid"
    ycol = "y_centroid" if "y_centroid" in sources.colnames else "ycentroid"

    peakvals = np.array([collapsed[int(round(s[ycol])), int(round(s[xcol]))]
                          for s in sources])
    best = sources[int(np.argmax(peakvals))]
    return float(best[xcol]), float(best[ycol])


def resolve_source_pixel(cube_file, ra=None, dec=None, x=None, y=None):
    """Resolve a source position to pixel (x, y) in the cube's spatial
    frame, accepting either RA/Dec (deg) or pixel x/y directly -- give one
    pair, not both. RA/Dec is converted with the cube's full WCS via
    `all_world2pix` rather than the simpler `wcs_world2pix`, so any
    distortion/SIP terms present in the WCS solution are accounted for
    rather than assuming a pure tangent-plane projection."""
    have_xy = (x is not None) and (y is not None)
    have_radec = (ra is not None) and (dec is not None)
    if have_xy and have_radec:
        raise ValueError("Give either (x, y) or (ra, dec), not both.")
    if have_xy:
        return float(x), float(y)
    if have_radec:
        from astropy.wcs import WCS
        with fits.open(cube_file) as hdul:
            # Multi-band cubes carry the wavelength axis as a FITS "-TAB" lookup
            # table (CTYPE3='WAVE-TAB' + a WCS-TABLE extension). astropy can only
            # build a WCS from such a header if it is handed the HDUList too --
            # WCS(header) alone raises "HDUList is required to retrieve -TAB
            # coordinates". naxis=2 keeps just the spatial (RA, Dec) axes.
            wcs2d = WCS(hdul["SCI"].header, fobj=hdul, naxis=2)
        x_pix, y_pix = wcs2d.all_world2pix(ra, dec, 0)
        return float(x_pix), float(y_pix)
    raise ValueError("Provide either (x, y) or (ra, dec).")


def extract_spectrum_manual(cube_file, out_file=None, ra=None, dec=None, x=None, y=None,
                             ifu_rfcorr=True, ifu_rscale=None, ifu_covar_scale=1.0,
                             srctype="POINT", apply_apcorr=True, return_meta=False):
    """Extract a flux-calibrated, background-subtracted 1D spectrum from an
    IFU cube at a given position.

    Accepts either (ra, dec) in degrees or (x, y) in pixels -- resolved via
    `resolve_source_pixel` -- and runs `Extract1dStep` with:

      * `subtract_background=False`, because AstroBkgInterp already removed
        the background upstream (pass the `*_abi_bgsub_s3d.fits` cube here,
        not the raw Stage-3 one -- letting the step also fit and subtract its
        own annulus background would subtract it twice).
      * `ifu_set_srctype='POINT'` and `apply_apcorr=True`, which together are
        what make the output properly flux calibrated: the step uses the
        wavelength-dependent point-source extraction aperture and applies the
        matching aperture correction, so the result is the source's *total*
        flux in Jy rather than whatever happened to fall inside a fixed
        aperture. Pass `srctype=None` to fall back to the cube header's own
        SRCTYPE.
      * `ifu_rfcorr=True`, a second, softer 1D residual-fringe correction on
        top of the Stage-2 one -- MRS channels 3-4 are otherwise visibly
        fringed.
      * `ifu_covar_scale`, which inflates the errors to account for the
        spatial covariance introduced when the cube was resampled. The
        nominal errors are optimistic without it; ~1 is the pipeline default
        but values of a few are commonly used for MRS.

    Returns (wave, flux, err, (src_x, src_y)), wavelength-sorted, or
    (wave, flux, err, (src_x, src_y), meta) when `return_meta=True`."""
    from stdatamodels.jwst import datamodels
    from jwst.extract_1d import Extract1dStep

    src_x, src_y = resolve_source_pixel(cube_file, ra=ra, dec=dec, x=x, y=y)

    kwargs = dict(subtract_background=False, center_xy=[src_x, src_y],
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

    try:
        flux_unit = str(tab.columns["FLUX"].unit or "Jy")
    except Exception:
        flux_unit = "Jy"

    meta = {
        "flux_unit": flux_unit,
        "src_x": src_x, "src_y": src_y,
        "n_good": int(np.isfinite(flux).sum()),
        "srctype": srctype, "apply_apcorr": apply_apcorr,
        "ifu_rfcorr": ifu_rfcorr, "ifu_covar_scale": ifu_covar_scale,
        "out_file": out_file,
    }

    cube.close()
    result.close()

    if return_meta:
        return wave, flux, err, (src_x, src_y), meta
    return wave, flux, err, (src_x, src_y)


def _annulus_aperture_positions(ny, nx, valid, src_x, src_y, r_ap, r_in, r_out,
                                 n_target, seed=0):
    """Up to `n_target` (x, y) centers for circular apertures of radius
    `r_ap`, placed on one or more concentric rings between `r_in` and
    `r_out` px of (src_x, src_y).

    Sampling a ring (rather than random pixels anywhere in the frame) keeps
    every sky measurement locally representative of the background/host
    structure actually near the source, instead of picking up unrelated
    structure elsewhere in the field (a different source, a channel edge,
    ...). Positions on each ring are spaced by the ring's circumference
    divided by `floor(pi * radius / r_ap)`, which is exactly the spacing at
    which neighbouring apertures of radius `r_ap` stop overlapping, so the
    sky measurements are as independent as the geometry allows.

    Only a candidate whose own centre pixel is finite (`valid`) and whose
    full disc stays clear of the array edge is kept -- a clipped aperture
    would silently undercount the sky flux inside it. Deterministic for a
    given `seed`, so calling this once per wavelength slice (only `r_ap`/
    `r_in`/`r_out` change slice to slice) gives comparable geometry."""
    rng = np.random.default_rng(seed)
    centers = []
    margin = r_ap + 1.0
    n_rings = max(1, int(round((r_out - r_in) / max(2.0 * r_ap, 1e-6))) + 1)
    for radius in np.linspace(r_in, r_out, n_rings):
        if len(centers) >= n_target:
            break
        n_on_ring = max(1, int(np.floor(np.pi * radius / max(r_ap, 1e-6))))
        phase = rng.uniform(0, 2 * np.pi)
        for k in range(n_on_ring):
            if len(centers) >= n_target:
                break
            ang = phase + 2 * np.pi * k / n_on_ring
            cx, cy = src_x + radius * np.cos(ang), src_y + radius * np.sin(ang)
            ix, iy = int(round(cx)), int(round(cy))
            if not (margin <= ix < nx - margin and margin <= iy < ny - margin):
                continue
            if not valid[iy, ix]:
                continue
            centers.append((cx, cy))
    return centers


def estimate_empirical_flux_error(bgsub_cube_file, wave, flux, src_x, src_y,
                                   fwhm_pix=None, rscale=2.0,
                                   ann_in_scale=2.5, ann_out_scale=6.0,
                                   n_apertures=24, seed=0, min_apertures=6):
    """Empirical, per-wavelength 1-sigma flux uncertainty, in the SAME
    calibrated units as `flux` (Jy) -- without needing to hand-replicate
    `Extract1dStep`'s surface-brightness -> Jy conversion or its
    wavelength-dependent point-source aperture correction.

    The trick: at a given wavelength slice, sum a circular aperture of
    radius `rscale * FWHM` on the source, and identical apertures on sky,
    directly in the cube's *native* (uncalibrated, MJy/sr) units. The ratio

        snr_native = source_aperture_sum / sigma_sky_native

    is dimensionless. Converting native cube units to a calibrated Jy flux
    (the surface-brightness -> flux conversion, and the aperture
    correction) multiplies signal and noise by the *same* factor at a given
    wavelength, so that ratio survives the conversion unchanged:

        snr_native ~= flux_jy / err_jy   =>   err_jy = flux_jy / snr_native

    This only requires that the aperture radius used here reasonably
    matches `Extract1dStep`'s own extraction aperture -- `rscale` defaults
    to the value used elsewhere in this pipeline for exactly that reason --
    never the absolute calibration constants. That is what makes it robust:
    a previous version of this check summed the cube's native-unit SCI
    directly against the calibrated Jy `err` and got a meaningless ratio
    (~1e-6) because the two were never in the same units to begin with.

    Sky apertures are placed on rings between `ann_in_scale` and
    `ann_out_scale` aperture-radii from the source (see
    `_annulus_aperture_positions`), not anywhere in the frame, so they
    sample locally representative background/host structure near the
    source. `ann_in_scale` must be > 2 so sky apertures of the same radius
    as the source aperture cannot overlap it.

    A slice is skipped (left out of the return) if fewer than
    `min_apertures` valid sky positions are available (e.g. very long
    wavelengths, where a large aperture radius leaves little room in a
    small IFU footprint) or if the source/sky sums aren't usable.

    Returns a dict with `wave` (native cube wavelength grid, restricted to
    slices where an estimate was obtained), `err` (the empirical 1-sigma
    error, same units as `flux`), `snr_native`, `n_apertures_used`, and
    `coverage` (fraction of the full cube wavelength grid with an
    estimate)."""
    if ann_in_scale <= 2.0:
        raise ValueError("ann_in_scale must be > 2 so sky apertures of radius "
                          "rscale*FWHM can't overlap the source aperture itself "
                          f"(got {ann_in_scale}).")
    from photutils.aperture import CircularAperture, aperture_photometry

    w_cube = get_mrs_wavelength(bgsub_cube_file)
    if fwhm_pix is None:
        fwhm_pix = _mrs_fwhm_pixels(bgsub_cube_file, wave=w_cube)

    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    order = np.argsort(wave)
    flux_on_cube = np.interp(w_cube, wave[order], flux[order],
                              left=np.nan, right=np.nan)

    with fits.open(bgsub_cube_file) as h:
        sci = h["SCI"].data
        err_cube = h["ERR"].data

    nz, ny, nx = sci.shape
    out_err = np.full(nz, np.nan)
    out_snr = np.full(nz, np.nan)
    n_used = np.zeros(nz, dtype=int)

    for i in range(nz):
        r_ap = float(rscale * fwhm_pix[i])
        r_in, r_out = ann_in_scale * r_ap, ann_out_scale * r_ap
        valid = np.isfinite(sci[i]) & np.isfinite(err_cube[i])
        centers = _annulus_aperture_positions(ny, nx, valid, src_x, src_y,
                                               r_ap, r_in, r_out, n_apertures,
                                               seed=seed)
        n_used[i] = len(centers)
        if len(centers) < min_apertures:
            continue

        d = np.nan_to_num(sci[i])
        sky_sums = np.asarray(
            aperture_photometry(d, CircularAperture(centers, r=r_ap))["aperture_sum"])
        sigma_sky = 1.4826 * np.median(np.abs(sky_sums - np.median(sky_sums)))
        src_sum = float(aperture_photometry(
            d, CircularAperture([(src_x, src_y)], r=r_ap))["aperture_sum"][0])

        if not (np.isfinite(sigma_sky) and sigma_sky > 0 and src_sum > 0):
            continue
        snr = src_sum / sigma_sky
        out_snr[i] = snr
        if np.isfinite(flux_on_cube[i]) and snr > 0:
            out_err[i] = abs(flux_on_cube[i]) / snr

    good = np.isfinite(out_err)
    return {"wave": w_cube[good], "err": out_err[good], "snr_native": out_snr[good],
            "n_apertures_used": n_used[good], "coverage": float(good.mean())}


def _mrs_wavelength_from_wcstab(cube_file):
    """Hand-parsed fallback: pull the wavelength array out of a WCS-TABLE
    (or WCS-TAB) FITS-WCS "-TAB" extension directly. Kept as a fallback for
    when stdatamodels/gwcs isn't available or the file has no ASDF
    extension -- but this can be brittle across jwst versions, since the
    exact row/column layout of the -TAB table has varied (single row with a
    TDIM-shaped array cell, one row per sample, object-dtype variable-length
    cells, etc.). Returns None if nothing usable is found (never raises)."""
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
            # Try every plausible layout rather than assuming row 0 holds
            # the whole array: a single row with a (possibly TDIM-shaped)
            # array cell, an object/variable-length cell, or one row per
            # sample (in which case the whole column, flattened, is the
            # wavelength grid).
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


def get_mrs_wavelength(cube_file):
    """The per-native-slice wavelength array (microns) for `cube_file`,
    one value per SCI slice along axis 0.

    This is the wavelength lookup that used to live inline in
    `_mrs_fwhm_pixels` -- pulled out into its own public function because
    it's generally useful (e.g. for binning/weighting by real wavelength
    elsewhere) and because it's already the robust version of the pandas/
    `WCS-TABLE`-parsing approach it's tempting to hand-roll: read from the
    cube's gwcs (via stdatamodels), the same source the jwst pipeline
    itself uses, which sidesteps having to hand-parse the FITS "-TAB"
    WCS-TABLE extension cube_build writes for combined multi-band cubes
    (that extension's exact row/column layout has changed across jwst
    versions, e.g. between jwst 2.x and 3.0.0, which is why a naive
    `Table(hdul["WCS-TABLE"].data).to_pandas().wavelength.values[0]` read
    can come back empty or wrong-shaped). Falls back to a direct
    `WCS-TABLE` parse (`_mrs_wavelength_from_wcstab`), then to the
    notebook's linear CRPIX/CDELT/CRVAL approach, only if the gwcs read
    fails. Raises RuntimeError if none of the three work."""
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
        wave = _mrs_wavelength_from_wcstab(cube_file)

    if wave is None:
        with fits.open(cube_file) as hdul:
            # Last resort: treat the spectral axis as linear, exactly like
            # extract_cube_manual() in extract_spectrum_from_cube.ipynb.
            # Only valid for single-band cubes with CTYPE3='WAVE' (not
            # 'WAVE-TAB'); raises if that's not the case, since a linear
            # read of a genuinely tabular axis would silently be wrong.
            hdr = hdul["SCI"].header
            ctype3 = hdr.get("CTYPE3", "")
            if "TAB" in ctype3.upper():
                raise RuntimeError(
                    f"No usable wavelength found in {cube_file} via gwcs or WCS-TABLE, and "
                    f"CTYPE3={ctype3!r} indicates a tabular (non-linear) spectral axis, so a "
                    "linear CRPIX3/CDELT3/CRVAL3 fallback would be wrong. Check that jwst/"
                    "stdatamodels can actually open this file (e.g. `datamodels.open(cube_file)`)."
                )
            nz = hdul["SCI"].data.shape[0]
            wave = (np.arange(nz) - hdr["CRPIX3"] + 1) * hdr["CDELT3"] + hdr["CRVAL3"]
            if hdr.get("CUNIT3", "").lower() in ("m", "meter", "metre"):
                wave = wave * 1e6  # m -> um

    return wave


def _mrs_fwhm_pixels(cube_file, wave=None):
    """Wavelength-dependent MRS PSF FWHM, in pixels, for every wavelength
    slice of the cube -- matches AstroBkgInterp's MRS tutorial: the
    empirical relation theta(lambda) = 0.033*lambda + 0.106 (arcsec, lambda
    in microns; from the MIRI MRS PSF/dithering docs), converted to pixels
    via the cube's spatial pixel scale (CDELT1).

    `wave` can be passed in (e.g. already fetched via `get_mrs_wavelength`)
    to avoid re-deriving it; otherwise it's looked up here."""
    if wave is None:
        wave = get_mrs_wavelength(cube_file)

    with fits.open(cube_file) as hdul:
        cdelt1_deg = hdul["SCI"].header["CDELT1"]

    cdelt1_arcsec = abs(cdelt1_deg) * 3600.0
    theta_arcsec = 0.033 * wave + 0.106
    return theta_arcsec / cdelt1_arcsec


def _fill_bad_uncertainties(err):
    """Replace non-finite / non-positive uncertainties with that wavelength
    slice's median uncertainty (or the cube-wide median, for a slice with
    nothing usable at all).

    This exists because AstroBkgInterp only runs `interp_nans` on the
    *science* slice, never on the error slice, and then propagates
    `sqrt(err**2 + bkg_err**2)`. A single NaN in ERR therefore survives
    straight through to the output, and since an MRS multi-band cube is
    mostly NaN outside each channel's footprint, the propagated ERR cube
    comes back ~100% NaN -- which silently poisons everything downstream
    that uses ERR as a noise estimate (centroid SNR tests, inverse-variance
    weighting, the extracted spectrum's errors). Filling first keeps the
    error propagation meaningful; the originally-bad pixels are restored to
    NaN afterwards by `_finalize_err_cube`."""
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
    restore the input cube's own NaNs.

    AstroBkgInterp wraps the masked error in an extra leading axis
    (`masked_err = np.array([masked_err])`), so a cube run returns an
    (nz, 1, ny, nx) array rather than (nz, ny, nx) -- writing that straight
    into the ERR extension produces a 4D ERR that no longer matches SCI.
    Squeeze it back, fall back to the input ERR if anything is still
    unusable, and re-NaN the pixels that were only filled in to keep the
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

    AstroBkgInterp's own `bkg_err` is unusable (it squares and sums the
    lstsq residuals, then broadcasts that scalar through the polynomial
    basis, so it is an SSR over thousands of pixels rather than a per-pixel
    error -- inflating the cube ERR by ~5 orders of magnitude). Estimate it
    properly instead.

    The background is a smooth surface fitted to many pixels at once, so its
    uncertainty is much *smaller* than the per-pixel noise: each sliding
    window constrains the local surface with ~bin_size^2 pixels, giving
    sigma_bkg ~ sigma_pixel / bin_size. Measure sigma_pixel from the robust
    scatter of the background-subtracted slice well away from the source, so
    it reflects what actually remains after the fit.

    `valid` marks which spaxels are real. This matters: an MRS cube's
    rectangular array is mostly padding outside the IFU footprint, and that
    padding is *finite zeros*, not NaN. Measuring the scatter without
    excluding it returns ~1e-10 -- the noise of the padding -- rather than
    the noise of the sky. Pass the ERR-based validity mask (finite and
    positive); it falls back to non-zero finite pixels if omitted.

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
    # A slice whose background fit failed outright leaves sigma=0; fall back
    # to the cube-wide median so it is not silently treated as noiseless.
    good = out > 0
    if good.any():
        out[~good] = np.median(out[good])
    return out


def run_astrobkginterp_background(cube_file, src_x, src_y, out_dir, product_name,
                                   bkg_mode="polynomial", aper_rad=7, ann_width=6,
                                   fwhm_scale=2.5, poly_k=3, bin_size=9,
                                   cube_resolution="high", pool_size=None,
                                   min_aper_rad=5.0, use_abi_err=False):
    """Run AstroBkgInterp on the cube's SCI/ERR arrays: mask a wavelength-
    dependent aperture around (src_x, src_y) and fit/subtract a background
    per wavelength slice. Saves the background-subtracted cube (all other
    extensions carried over from the input unchanged) plus a quick-look
    comparison PNG, and returns the output cube path.

    How big is the source mask, really?
    -----------------------------------
    In cube mode AstroBkgInterp **overrides** whatever `aper_rad` you pass,
    recomputing it per slice as `ceil(fwhm_scale * fwhm(lambda))`. So
    `aper_rad` only applies to 2D input, and `fwhm_scale` is the real knob.
    That matters for a bright target: the MRS PSF has broad wings, and at
    the old default of 1.5 FWHM a good fraction of the source still lands
    outside the mask, gets treated as background, and is fitted and
    subtracted -- eating into the very flux you are trying to measure.
    `fwhm_scale` now defaults to 2.5, and `min_aper_rad` (pixels) sets a
    floor so the mask cannot collapse to a couple of pixels at the blue end
    where the PSF is narrowest. Raise `fwhm_scale` further if the extracted
    spectrum still looks suppressed; the cost is a larger interpolated hole,
    so the background under the source is less well constrained.

    `use_abi_err` is False because AstroBkgInterp's background-error term is
    wrong: `polyfit2d_cube` takes the *sum of squared residuals* that
    `np.linalg.lstsq` returns and uses it as if it were the fitted
    coefficients, so `bkg_err` comes out as an SSR summed over thousands of
    pixels -- inflating the cube's ERR by ~5 orders of magnitude and making
    the extracted spectrum's error bars meaningless. Instead the background
    model's uncertainty is estimated here from the actual residual scatter
    (see `_background_model_error`). Set True only to reproduce the old
    (broken) behaviour.

    The ERR extension is sanitized on the way in and repaired on the way out
    (`_fill_bad_uncertainties` / `_finalize_err_cube`) so the saved cube has
    a real, correctly-shaped uncertainty array rather than the all-NaN
    (nz, 1, ny, nx) one AstroBkgInterp produces unaided."""
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
            # AstroBkgInterp.run() calls np.zeros_like(err) unconditionally in
            # cube mode and crashes on err=None -- fall back to a uniform
            # (unweighted) array rather than skip error propagation entirely.
            print("  WARNING: no ERR extension in cube -- using a uniform "
                  "placeholder array (AstroBkgInterp requires a non-None err "
                  "array for cube input); propagated errors will not be meaningful.")
            orig_err = np.ones_like(data)
            err = orig_err
        fwhm = _mrs_fwhm_pixels(cube_file)

        # AstroBkgInterp sets aper_rad = ceil(fwhm_scale * fwhm[i]) per slice,
        # so a floor on the mask radius has to be applied through the FWHM
        # array it reads. Inflating fwhm here is equivalent to, and simpler
        # than, patching its per-slice override.
        fwhm_eff = np.maximum(fwhm, float(min_aper_rad) / max(fwhm_scale, 1e-6))
        r_slice = np.ceil(fwhm_scale * fwhm_eff)
        print(f"  Source mask radius: {r_slice.min():.0f}-{r_slice.max():.0f} px "
              f"(fwhm_scale={fwhm_scale}, min_aper_rad={min_aper_rad}); "
              f"PSF FWHM {fwhm.min():.1f}-{fwhm.max():.1f} px")

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
        import matplotlib
        # matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mid = data.shape[0] // 2
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].set_title("Raw")
        ax[1].set_title("Background model")
        ax[2].set_title("Background-subtracted")
        ax[0].imshow(data[mid], origin="lower")
        ax[1].imshow(bkg[mid], origin="lower")
        ax[2].imshow(diff[mid], origin="lower")
        for a in ax:
            a.scatter(src_x, src_y, marker="+", c="red", s=60)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{product_name}_abi_bgsub_quicklook.png"))
        plt.close(fig)
    except Exception as e:
        print(f"  (skipped quick-look plot: {e})")

    return out_path


# ===========================================================================
# Centroid refinement on a background-subtracted cube, then reprocess
# ===========================================================================

def fit_slice_centroid(img_slice, err_slice, x0, y0, box_half_size=5,
                        snr_threshold=3.0, max_drift_pix=3.0, return_snr=False,
                        diagnostics=None):
    """Fit the centroid of a point source within a small box around
    (x0, y0) in a single 2D image (one wavelength slice, or a bin of
    several summed together -- see `refine_centroid_weighted`'s `n_bin`),
    rejecting it (returning NaNs) if the local peak doesn't clear an SNR
    threshold or the fit drifts implausibly far from the initial guess.
    `snr` (peak above local background, in local-noise units) is returned
    too when `return_snr=True`, for weighting slices against each other.

    If `diagnostics` is a dict, it's updated in place with counts of why
    slices were rejected (`no_finite_data`, `low_snr`, `fit_failed`,
    `drift`, `accepted`) and a running list of every achieved SNR
    (`diagnostics["all_snr"]`, including sub-threshold ones) -- so a
    caller that ends up with nothing usable can report *why*, and what
    SNR was actually achievable, rather than just failing blind."""
    from photutils.centroids import centroid_quadratic

    def reject(reason):
        if diagnostics is not None:
            diagnostics[reason] = diagnostics.get(reason, 0) + 1
        return (np.nan, np.nan, np.nan) if return_snr else (np.nan, np.nan)

    yi, xi = int(round(y0)), int(round(x0))
    ylo, yhi = max(yi - box_half_size, 0), min(yi + box_half_size + 1, img_slice.shape[0])
    xlo, xhi = max(xi - box_half_size, 0), min(xi + box_half_size + 1, img_slice.shape[1])
    cutout = img_slice[ylo:yhi, xlo:xhi]

    if not np.any(np.isfinite(cutout)):
        return reject("no_finite_data")

    local_bg = np.nanmedian(cutout)
    local_std = np.nanstd(cutout)
    if err_slice is not None:
        err_cut = err_slice[ylo:yhi, xlo:xhi]
        err_med = np.nanmedian(err_cut) if np.any(np.isfinite(err_cut)) else np.nan
        if np.isfinite(err_med) and err_med > 0:
            local_std = err_med
    peak = np.nanmax(cutout)
    snr = (peak - local_bg) / local_std if local_std > 0 else np.nan
    if diagnostics is not None and np.isfinite(snr):
        diagnostics.setdefault("all_snr", []).append(float(snr))
    if not np.isfinite(snr) or snr < snr_threshold:
        return reject("low_snr")

    try:
        xc, yc = centroid_quadratic(cutout - local_bg)
    except Exception:
        return reject("fit_failed")

    xc_full, yc_full = xc + xlo, yc + ylo
    if np.hypot(xc_full - x0, yc_full - y0) > max_drift_pix:
        return reject("drift")

    if diagnostics is not None:
        diagnostics["accepted"] = diagnostics.get("accepted", 0) + 1
    return (xc_full, yc_full, snr) if return_snr else (xc_full, yc_full)


def refine_centroid_weighted(cube_file, x0, y0, box_half_size=5, snr_threshold=3.0,
                              max_drift_pix=3.0, sigma_clip_final=3.0, n_bin=20):
    """Refine a source position using `cube_file` -- run this on a
    *background-subtracted* cube, where the point source stands out
    cleanly rather than sitting on top of MRS's own background, so the
    per-slice fits below are much cleaner than they'd be on a raw Stage-3
    cube.

    `n_bin` (default 20) sums this many adjacent wavelength slices
    together before fitting each bin, rather than fitting every native
    slice on its own. This matters a lot for a faint continuum source: a
    wavelength-collapsed image (like the one you picked your target from)
    gets its high SNR *because* it integrates over the whole cube -- a
    single native slice usually doesn't have anywhere near that SNR on
    its own, and requiring every slice to individually clear a real
    detection threshold can leave nothing usable. Binning trades
    wavelength resolution in the *centroid measurement* (which you don't
    need much of -- the source position shouldn't move fast with
    wavelength except possibly at channel boundaries) for a ~sqrt(n_bin)
    SNR boost per fit. Pass `n_bin=1` for the original per-native-slice
    behavior.

    Each bin is fit independently (`fit_slice_centroid`); the per-bin
    centroids are then combined with an inverse-variance weight based on
    that bin's SNR and the MRS PSF FWHM at that wavelength (centroiding
    precision scales roughly as FWHM/SNR, from `_mrs_fwhm_pixels`) -- so
    sharp, high-SNR bins pull the combined position much harder than
    faint or blurry ones, rather than every surviving bin counting
    equally as in a plain (sigma-clipped) mean.

    Returns a dict: x, y, x_err, y_err (formal, from the weights), x_scatter/
    y_scatter (the weighted RMS of the per-bin measurements around the
    combined position -- if this is much bigger than x_err/y_err, the
    bins disagree more than their nominal precision suggests, which is
    worth a look before trusting the result), n_slices_used/n_slices_total
    (in bins, if n_bin>1), the raw per-bin arrays for your own diagnostics,
    and `per_slice_wavelength`/`wavelength` -- the mean wavelength (microns)
    of each bin (from `get_mrs_wavelength`), aligned index-for-index with
    per_slice_x/per_slice_y/per_slice_snr, so you can e.g. plot the refined
    position vs. wavelength or spot a channel-boundary artifact. Raises
    RuntimeError with a breakdown of *why* (SNR achieved vs. required, how
    many failed on drift, etc.) if nothing survives -- read that message
    before just lowering snr_threshold."""
    
    from astropy.stats import sigma_clip as _sigma_clip

    hdul = fits.open(cube_file)
    data = hdul["SCI"].data
    err = hdul["ERR"].data if "ERR" in hdul else None
    if err is not None:
        err = np.squeeze(np.asarray(err))
        if err.shape != data.shape or not np.isfinite(err).any():
            err = None
    hdul.close()

    wave = get_mrs_wavelength(cube_file)
    fwhm = _mrs_fwhm_pixels(cube_file, wave=wave)
    
    print(f"Refining centroid for cube: {cube_file} with initial guess x0={x0}, y0={y0}")

    n_wave_native = data.shape[0]
    if n_bin > 1:
        n_bins = n_wave_native // n_bin
        if n_bins < 1:
            raise ValueError(f"n_bin={n_bin} is larger than the cube's {n_wave_native} slices.")
        trimmed = n_bins * n_bin
        data = data[:trimmed].reshape(n_bins, n_bin, *data.shape[1:]).sum(axis=1)
        if err is not None:
            err = np.sqrt((err[:trimmed] ** 2).reshape(n_bins, n_bin, *err.shape[1:]).sum(axis=1))
        fwhm = fwhm[:trimmed].reshape(n_bins, n_bin).mean(axis=1)
        wave = wave[:trimmed].reshape(n_bins, n_bin).mean(axis=1)
    n_wave = data.shape[0]

    xs = np.full(n_wave, np.nan)
    ys = np.full(n_wave, np.nan)
    snrs = np.full(n_wave, np.nan)
    diag = {}
    for i in range(n_wave):
        err_slice = err[i] if err is not None else None
        xs[i], ys[i], snrs[i] = fit_slice_centroid(
            data[i], err_slice, x0, y0, box_half_size=box_half_size,
            snr_threshold=snr_threshold, max_drift_pix=max_drift_pix,
            return_snr=True, diagnostics=diag)

    good = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(snrs) & (snrs > 0)
    if not np.any(good):
        all_snr = diag.get("all_snr", [])
        snr_summary = (f"achieved SNR ranged {np.min(all_snr):.2f}-{np.max(all_snr):.2f} "
                        f"(median {np.median(all_snr):.2f}) against snr_threshold={snr_threshold}"
                        if all_snr else "no bin even had finite data to measure an SNR from")
        raise RuntimeError(
            f"No usable bins for centroid refinement in {cube_file} (n_bin={n_bin}, "
            f"{n_wave} bins tried). Rejection breakdown: "
            f"{ {k: v for k, v in diag.items() if k != 'all_snr'} }. {snr_summary}. "
            f"If SNR is the limiter, try a larger n_bin (fewer, deeper bins) before lowering "
            f"snr_threshold; if drift dominates, the initial (x0, y0) guess may be off, or "
            f"max_drift_pix is too tight.")

    w_good = (snrs[good] / fwhm[good]) ** 2

    # A bin can pass the per-bin SNR/drift checks and still be a
    # systematic outlier (e.g. a cosmic-ray residual right at the fit box
    # edge) -- one more sigma-clip pass on the surviving centroids
    # themselves, before combining.
    x_masked = _sigma_clip(xs[good], sigma=sigma_clip_final, masked=True)
    y_masked = _sigma_clip(ys[good], sigma=sigma_clip_final, masked=True)
    keep = ~(np.ma.getmaskarray(x_masked) | np.ma.getmaskarray(y_masked))

    w_k, x_k, y_k = w_good[keep], xs[good][keep], ys[good][keep]
    if w_k.sum() == 0 or len(w_k) == 0:
        raise RuntimeError(f"All candidate bins were rejected by the final sigma-clip "
                            f"in {cube_file} -- try a larger sigma_clip_final.")

    x_ref = float(np.sum(w_k * x_k) / np.sum(w_k))
    y_ref = float(np.sum(w_k * y_k) / np.sum(w_k))
    x_err = float(np.sqrt(1.0 / np.sum(w_k)))
    y_err = float(np.sqrt(1.0 / np.sum(w_k)))
    x_scatter = float(np.sqrt(np.sum(w_k * (x_k - x_ref) ** 2) / np.sum(w_k)))
    y_scatter = float(np.sqrt(np.sum(w_k * (y_k - y_ref) ** 2) / np.sum(w_k)))

    return {
        "x": x_ref, "y": y_ref, "x_err": x_err, "y_err": y_err,
        "x_scatter": x_scatter, "y_scatter": y_scatter,
        "n_slices_used": int(keep.sum()), "n_slices_total": int(n_wave),
        "per_slice_x": xs, "per_slice_y": ys, "per_slice_snr": snrs,
        "per_slice_wavelength": wave, "wavelength": wave,
    }


# ===========================================================================
# Weighted cube collapse + SExtractor-style centroiding
# ===========================================================================

def _robust_slice_stats(cube_file, kmin, kmax, chunk=256):
    """Per-wavelength-slice robust median and sigma (1.4826 * MAD) over the
    selected slice range, read in contiguous wavelength chunks so the whole
    cube never has to be in memory at once. Slices with no finite data come
    back as NaN."""
    n = kmax - kmin
    med = np.full(n, np.nan)
    sig = np.full(n, np.nan)
    nfinite = np.zeros(n, dtype=int)

    with fits.open(cube_file, memmap=True) as hdul:
        sci = hdul["SCI"].data
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            block = np.asarray(sci[kmin + start:kmin + stop], dtype=np.float32)
            block = np.where(np.isfinite(block), block, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmedian(block, axis=(1, 2))
                mad = np.nanmedian(np.abs(block - m[:, None, None]), axis=(1, 2))
            med[start:stop] = m
            sig[start:stop] = 1.4826 * mad
            nfinite[start:stop] = np.isfinite(block).sum(axis=(1, 2))

    # A slice can be perfectly flat (MAD == 0) without being useless; give it
    # the run-of-the-mill noise level rather than an infinite weight.
    typical = np.nanmedian(sig[np.isfinite(sig) & (sig > 0)]) if np.any(sig > 0) else 1.0
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, typical)
    med = np.where(np.isfinite(med), med, 0.0)
    return med, sig, nfinite


def make_weighted_image(cube_file, wave_range=None, clip_sigma=5.0, n_sub=0,
                         row_block=16, use_err=True, use_dq=True, min_frac=0.1):
    """Collapse a whole IFU cube down to a single inverse-variance weighted
    2D image -- the "combined weighted image" a source-detection run is
    actually worth doing on.

    This replaces the per-slice (or per-bin) centroid fitting that
    `refine_centroid_weighted` does. Fitting individual wavelength slices
    only works if each slice independently detects the source, which a faint
    MRS target simply doesn't do; binning helps but still throws away most of
    the available signal and is exquisitely sensitive to the ERR array being
    sane (which, for an AstroBkgInterp output cube, it historically was not).
    Collapsing the *entire* cube with optimal weights puts every photon in
    all ~10,000 slices into one frame, which is where the source's SNR
    actually lives.

    Weighting, per pixel: `w = 1 / err**2` from the cube's ERR extension when
    that extension is usable, otherwise a per-slice scalar `w = 1 / sigma**2`
    with sigma the robust (MAD-based) noise of that slice. Each slice has its
    own robust median subtracted first, so slices sitting at different
    background levels (very much the case across MRS channels) combine
    without one pedestal dominating.

    Parameters
    ----------
    wave_range : (lo, hi) or None
        Restrict the collapse to this wavelength range in microns. Worth
        using: the MRS PSF is ~4x sharper at 5 um than at 28 um, so
        collapsing only channel 1 (e.g. `(4.9, 7.7)`) gives a tighter, more
        precisely centroidable PSF -- at the cost of the longer-wavelength
        signal. `None` (default) uses the whole cube.
    clip_sigma : float or None
        Sigma-clip each spatial pixel along the wavelength axis before
        combining, to kill cosmic-ray and outlier-detection residuals. Set to
        None to skip.
    n_sub : int
        Also build this many independent sub-band images by splitting the
        wavelength range into contiguous chunks. Centroiding each one gives
        an empirical position scatter (see `find_source_centroid`), which is
        a far more honest uncertainty than any formal error.
    row_block : int
        Number of image rows processed at a time. The sigma-clip needs a
        pixel's full wavelength history in memory, so the cube is walked in
        row slabs rather than all at once.
    min_frac : float
        Pixels with usable data in fewer than this fraction of slices are
        masked out (NaN) in the output -- these are the FOV edges, where one
        channel covers the sky and the others do not.

    Returns
    -------
    dict with `image` (weighted mean, MJy/sr), `err` (its propagated
    uncertainty), `snr` (image/err), `coverage`, `wave`, `wave_range`,
    `n_slices`, `weighting` ('err' or 'slice-rms'), and -- when `n_sub>0` --
    `sub_images`, `sub_errs`, `sub_wave_ranges`.
    """
    from astropy.stats import sigma_clip as _sigma_clip

    wave = get_mrs_wavelength(cube_file)
    nz_total = len(wave)

    if wave_range is not None:
        lo, hi = wave_range
        sel = np.where((wave >= lo) & (wave <= hi))[0]
        if sel.size < 2:
            raise ValueError(f"wave_range={wave_range} selects {sel.size} slices of "
                             f"{nz_total} (cube covers {wave.min():.2f}-{wave.max():.2f} um).")
        kmin, kmax = int(sel[0]), int(sel[-1]) + 1
    else:
        kmin, kmax = 0, nz_total
    wave_sel = wave[kmin:kmax]
    n = kmax - kmin

    with fits.open(cube_file, memmap=True) as hdul:
        ny, nx = hdul["SCI"].data.shape[1:]
        err_ok = False
        if use_err and "ERR" in hdul:
            err_shape = np.squeeze(np.asarray(hdul["ERR"].data[:1])).shape
            # An (nz, 1, ny, nx) ERR (AstroBkgInterp's unsqueezed output) still
            # squeezes down to (ny, nx) per slice, so shape alone isn't enough
            # -- check that some of it is actually finite before trusting it.
            probe_idx = np.linspace(kmin, kmax - 1, min(32, n)).astype(int)
            probe = np.squeeze(np.asarray(hdul["ERR"].data[probe_idx], dtype=np.float32))
            err_ok = err_shape == (ny, nx) and np.isfinite(probe).mean() > 0.01
        has_dq = use_dq and "DQ" in hdul

    slice_med, slice_sig, _ = _robust_slice_stats(cube_file, kmin, kmax)
    weighting = "err" if err_ok else "slice-rms"

    # Contiguous sub-band boundaries, used for both the total and (if asked)
    # the independent sub-images -- the total is just their sum.
    n_groups = max(int(n_sub), 1)
    edges = np.linspace(0, n, n_groups + 1).astype(int)

    sum_wd = np.zeros((n_groups, ny, nx))
    sum_w = np.zeros((n_groups, ny, nx))
    counts = np.zeros((ny, nx))

    with fits.open(cube_file, memmap=True) as hdul:
        sci, errext = hdul["SCI"].data, (hdul["ERR"].data if err_ok else None)
        # The DQ extension is written as int32 with BZERO/BSCALE (i.e. stored
        # unsigned), which astropy refuses to memory-map. Read it unscaled
        # instead -- BZERO is 2**31, whose bit 0 is 0, so testing the
        # DO_NOT_USE bit with `& 1` gives the same answer either way.
        dq_hdul = fits.open(cube_file, memmap=True, do_not_scale_image_data=True) if has_dq else None
        dq = dq_hdul["DQ"].data if dq_hdul is not None else None

        try:
            for y0b in range(0, ny, row_block):
                y1b = min(y0b + row_block, ny)
                d = np.asarray(sci[kmin:kmax, y0b:y1b, :], dtype=np.float32).astype(float)
                bad = ~np.isfinite(d)
                if dq is not None:
                    bad |= (np.asarray(dq[kmin:kmax, y0b:y1b, :]) & 1).astype(bool)

                d -= slice_med[:, None, None]

                if err_ok:
                    e = np.squeeze(np.asarray(errext[kmin:kmax, y0b:y1b, :],
                                               dtype=np.float32)).astype(float)
                    w = np.where(np.isfinite(e) & (e > 0), 1.0 / np.maximum(e, 1e-30) ** 2, 0.0)
                    bad |= ~(np.isfinite(e) & (e > 0))
                else:
                    w = np.broadcast_to((1.0 / slice_sig ** 2)[:, None, None], d.shape).copy()

                d[bad] = np.nan
                if clip_sigma is not None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        clipped = _sigma_clip(np.ma.masked_invalid(d), sigma=clip_sigma,
                                              axis=0, masked=True)
                    bad |= np.ma.getmaskarray(clipped)

                w = np.where(bad, 0.0, w)
                d = np.where(bad, 0.0, d)

                counts[y0b:y1b, :] = (~bad).sum(axis=0)
                for g in range(n_groups):
                    lo, hi = edges[g], edges[g + 1]
                    sum_wd[g, y0b:y1b, :] = (w[lo:hi] * d[lo:hi]).sum(axis=0)
                    sum_w[g, y0b:y1b, :] = w[lo:hi].sum(axis=0)
        finally:
            if dq_hdul is not None:
                dq_hdul.close()

    def _combine(swd, sw):
        with np.errstate(invalid="ignore", divide="ignore"):
            img = np.where(sw > 0, swd / sw, np.nan)
            erri = np.where(sw > 0, 1.0 / np.sqrt(np.maximum(sw, 1e-300)), np.nan)
        return img, erri

    tot_wd, tot_w = sum_wd.sum(axis=0), sum_w.sum(axis=0)
    image, err_image = _combine(tot_wd, tot_w)

    coverage = counts / float(n)
    poor = coverage < min_frac
    image[poor] = np.nan
    err_image[poor] = np.nan

    if weighting == "slice-rms":
        # Slice-RMS weights are *relative* -- one scalar per slice, with no
        # calibrated per-pixel scale -- so 1/sqrt(sum_w) is only correct up to
        # an arbitrary constant. Left alone it comes out ~10 orders of
        # magnitude below the actual scatter, which makes every downstream
        # chi2 meaningless and hands a few pixels absurd leverage in any
        # weighted fit. Renormalise to the empirical noise of the collapsed
        # image, which preserves the map's relative structure (that part does
        # correctly track per-pixel coverage) while fixing the scale.
        fin = np.isfinite(image)
        if fin.sum() > 32:
            emp = 1.4826 * np.median(np.abs(image[fin] - np.median(image[fin])))
            eg = np.isfinite(err_image) & (err_image > 0)
            if emp > 0 and eg.any():
                err_image = err_image * (emp / np.median(err_image[eg]))

    out = {
        "image": image, "err": err_image,
        "snr": np.where(np.isfinite(err_image) & (err_image > 0), image / err_image, np.nan),
        "coverage": coverage, "wave": wave_sel,
        "wave_range": (float(wave_sel[0]), float(wave_sel[-1])),
        "n_slices": int(n), "n_slices_cube": int(nz_total),
        "weighting": weighting,
    }

    if n_sub and n_sub > 1:
        subs, sub_errs, sub_ranges = [], [], []
        for g in range(n_groups):
            si, se = _combine(sum_wd[g], sum_w[g])
            si[poor] = np.nan
            se[poor] = np.nan
            subs.append(si)
            sub_errs.append(se)
            sub_ranges.append((float(wave_sel[edges[g]]), float(wave_sel[edges[g + 1] - 1])))
        out.update(sub_images=subs, sub_errs=sub_errs, sub_wave_ranges=sub_ranges)

    return out


def _cat_col(cat, new_name, old_name):
    """photutils 3.0 renamed a pile of SourceCatalog attributes
    (xcentroid -> x_centroid, segment_fluxerr -> segment_flux_err, ...).
    Read whichever one this install actually has, without tripping the
    deprecation warning on the newer ones."""
    for name in (new_name, old_name):
        if hasattr(cat, name):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return getattr(cat, name)
    raise AttributeError(f"SourceCatalog has neither {new_name!r} nor {old_name!r}")


def _detect_sources_compat(data, threshold, npixels, mask):
    from photutils.segmentation import detect_sources
    try:
        return detect_sources(data, threshold, n_pixels=npixels, mask=mask)
    except TypeError:
        return detect_sources(data, threshold, npixels=npixels, mask=mask)


def _deblend_sources_compat(data, segm, npixels):
    from photutils.segmentation import deblend_sources
    kw = dict(contrast=0.001, progress_bar=False)
    try:
        return deblend_sources(data, segm, n_pixels=npixels, n_levels=32, **kw)
    except TypeError:
        return deblend_sources(data, segm, npixels=npixels, nlevels=32, **kw)


def find_centroid_sourcex(image, err_image=None, x0=None, y0=None, search_radius=8.0,
                           nsigma=3.0, npixels=5, kernel_fwhm=2.5, deblend=True,
                           subtract_background=True, bkg_boxsize=None,
                           max_area_factor=25.0):
    """SExtractor-style source detection and centroiding on a single 2D image.

    Runs the same background-mesh -> filter -> threshold -> deblend ->
    measure sequence Source Extractor does, via photutils' segmentation
    machinery, rather than fitting an analytic profile in a fixed box. That
    matters here because it locates the source's own footprint instead of
    assuming the guess position is already centred, so it can recover from a
    guess several pixels off.

    The mesh background subtraction (`subtract_background`, on by default) is
    the part that makes this usable on a raw Stage-3 MRS cube. A supernova
    sits on top of its host galaxy, and a collapsed MRS cube of that field is
    mostly smooth, bright, extended galaxy emission -- detect on it directly
    and the segmentation returns a handful of enormous blobs covering half
    the FOV, whose "centroids" are meaningless. `Background2D` with a mesh a
    few PSF widths across models that smooth component and removes it,
    leaving the point source to be detected on its own. `bkg_boxsize`
    defaults to ~4x `kernel_fwhm`; make it smaller to remove more aggressive
    structure, larger to preserve more real extended flux.

    `max_area_factor` rejects detections whose area is more than this many
    times the PSF footprint -- leftover extended junk rather than a point
    source. Set to None to keep everything.

    If `x0`/`y0` are given, the detected source nearest that position (within
    `search_radius` pixels) is returned; otherwise the brightest detection
    wins. Returns a dict with x, y, x_err, y_err, snr, flux, and the full
    detection catalog; raises RuntimeError if nothing is detected."""
    from astropy.convolution import convolve
    from photutils.segmentation import SourceCatalog

    try:
        from photutils.psf import make_2dgaussian_kernel
        ksize = max(3, int(2 * round(kernel_fwhm) + 1))
        kernel = make_2dgaussian_kernel(kernel_fwhm, size=ksize)
    except ImportError:  # photutils < 1.9
        from astropy.convolution import Gaussian2DKernel
        from astropy.stats import gaussian_fwhm_to_sigma
        kernel = Gaussian2DKernel(kernel_fwhm * gaussian_fwhm_to_sigma)
        kernel.normalize()

    image = np.asarray(image, dtype=float)
    finite = np.isfinite(image)
    if not finite.any():
        raise RuntimeError("The collapsed image is entirely non-finite -- nothing to detect.")

    mean, median, std = sigma_clipped_stats(image[finite], sigma=3.0)
    bkg2d = None

    if subtract_background:
        from photutils.background import Background2D, MedianBackground
        from astropy.stats import SigmaClip
        box = int(bkg_boxsize) if bkg_boxsize else max(5, int(round(4 * kernel_fwhm)))
        box = min(box, max(5, min(image.shape) // 3))
        try:
            bkg2d = Background2D(image, box_size=box, filter_size=3,
                                  mask=~finite, sigma_clip=SigmaClip(sigma=3.0),
                                  bkg_estimator=MedianBackground(),
                                  exclude_percentile=90.0)
            back, rms = bkg2d.background, bkg2d.background_rms
        except Exception as exc:
            warnings.warn(f"Background2D failed ({exc!r}); falling back to a constant "
                           "background. Detection on an extended-emission field will be poor.")
            back, rms = np.full_like(image, median), np.full_like(image, std)
    else:
        back, rms = np.full_like(image, median), np.full_like(image, std)

    # The cube's own ERR underestimates pixel-to-pixel noise (cube_build's
    # resampling correlates neighbouring spaxels), so the empirical mesh RMS
    # is the better detection noise -- but never let it fall below ERR.
    if err_image is not None and np.isfinite(err_image).any():
        e = np.where(np.isfinite(err_image) & (err_image > 0), err_image, np.nan)
        e = np.where(np.isfinite(e), e, np.nanmedian(e))
        rms = np.maximum(rms, e)
    rms = np.where(np.isfinite(rms) & (rms > 0), rms, np.nanmedian(rms[np.isfinite(rms)]))

    data = np.where(finite, image - back, 0.0)
    convolved = convolve(data, kernel, nan_treatment="fill", fill_value=0.0)

    segm = None
    used_scale = 1.0
    for thresh_scale in (1.0, 0.7, 0.5):
        segm = _detect_sources_compat(convolved, nsigma * thresh_scale * rms, npixels, ~finite)
        if segm is not None:
            used_scale = thresh_scale
            break
    if segm is None:
        raise RuntimeError(
            f"No source detected in the collapsed image down to {0.5 * nsigma:.1f}sigma "
            f"(npixels={npixels}). Sigma-clipped image stats: median={median:.4g}, "
            f"std={std:.4g}. Try a smaller `npixels`, a lower `nsigma`, or restricting "
            f"`wave_range` to a channel where the source is actually bright.")
    if used_scale != 1.0:
        print(f"  (no detection at {nsigma:.1f}sigma; found source(s) at "
              f"{nsigma * used_scale:.1f}sigma)")

    if deblend:
        try:
            segm = _deblend_sources_compat(convolved, segm, npixels)
        except Exception as exc:
            warnings.warn(f"Deblending failed ({exc!r}); using the un-deblended segmentation.")

    cat = SourceCatalog(data, segm, convolved_data=convolved, error=rms, mask=~finite)

    xs = np.asarray(_cat_col(cat, "x_centroid", "xcentroid"), dtype=float)
    ys = np.asarray(_cat_col(cat, "y_centroid", "ycentroid"), dtype=float)
    flux = np.asarray(cat.segment_flux, dtype=float)
    fluxerr = np.asarray(_cat_col(cat, "segment_flux_err", "segment_fluxerr"), dtype=float)
    area = np.asarray(cat.area, dtype=float)
    labels = np.asarray(cat.labels)

    # Drop obviously-extended blobs; a point source is ~a PSF footprint.
    keep = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(flux)
    if max_area_factor is not None:
        psf_area = np.pi * (kernel_fwhm ** 2)
        compact = area <= max_area_factor * psf_area
        if compact.any():
            keep &= compact
    idx_pool = np.where(keep)[0]
    if idx_pool.size == 0:
        idx_pool = np.where(np.isfinite(xs) & np.isfinite(ys))[0]
    if idx_pool.size == 0:
        raise RuntimeError("Sources were detected but none had a usable centroid.")

    if x0 is not None and y0 is not None:
        dist = np.hypot(xs[idx_pool] - x0, ys[idx_pool] - y0)
        within = dist <= search_radius
        if within.any():
            idx = int(idx_pool[within][np.argmin(dist[within])])
        else:
            idx = int(idx_pool[np.nanargmax(flux[idx_pool])])
            print(f"  NOTE: no detection within {search_radius:.1f} px of the guess "
                  f"({x0:.2f}, {y0:.2f}); falling back to the brightest compact source, "
                  f"{np.nanmin(dist):.1f} px away. Check this before trusting it.")
    else:
        idx = int(idx_pool[np.nanargmax(flux[idx_pool])])

    x, y = float(xs[idx]), float(ys[idx])

    # Prefer the quadratic centroid (sub-pixel, SExtractor's WIN-style
    # refinement) but keep the isophotal one if it wanders or fails.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            quad = np.atleast_2d(np.asarray(cat.centroid_quad, dtype=float))[idx]
        xq, yq = float(quad[0]), float(quad[1])
        if np.isfinite(xq) and np.isfinite(yq) and np.hypot(xq - x, yq - y) < 2.0:
            x, y = xq, yq
    except Exception:
        pass

    snr = (float(flux[idx] / fluxerr[idx])
           if np.isfinite(fluxerr[idx]) and fluxerr[idx] > 0 else np.nan)
    # Centroiding precision for a PSF of this width at this SNR -- the
    # classic sigma_PSF / SNR result. Ignores pixelation, but it is the right
    # order of magnitude, and it is floored so it never claims to beat what a
    # pixel grid can actually deliver.
    sigma_psf = kernel_fwhm / 2.3548
    pos_err = float(max(sigma_psf / snr, 0.01)) if np.isfinite(snr) and snr > 0 else np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_det = int(segm.nlabels if hasattr(segm, "nlabels") else segm.n_labels)
        tbl = cat.to_table()

    return {"x": x, "y": y, "x_err": pos_err, "y_err": pos_err, "snr": snr,
            "flux": float(flux[idx]), "flux_err": float(fluxerr[idx]),
            "area": float(area[idx]), "n_detected": n_det, "label": int(labels[idx]),
            "catalog": tbl, "segment_map": segm, "background": back,
            "detection_image": data, "rms": rms,
            "image_median": float(median), "image_std": float(std)}


def find_source_centroid(cube_file, x0=None, y0=None, wave_range=None, clip_sigma=5.0,
                          n_sub=4, search_radius=8.0, nsigma=3.0, npixels=5,
                          kernel_fwhm=None, plot_file=None, min_frac=0.1,
                          subtract_background=True, bkg_boxsize=None,
                          max_area_factor=25.0):
    """Find a point source's centroid in an IFU cube by collapsing the whole
    cube into one inverse-variance weighted image (`make_weighted_image`) and
    running SExtractor-style detection on it (`find_centroid_sourcex`).

    This is the replacement for the bin-every-N-slices-and-fit-each-bin
    approach: one deep image, one detection, no dependence on any individual
    slice being a detection on its own and no dependence on the ERR array
    being NaN-free (it falls back to a per-slice robust RMS if ERR is
    unusable).

    `n_sub` (default 4) additionally splits the wavelength range into that
    many independent sub-band images and centroids each one. The spread of
    those positions is reported as `x_scatter`/`y_scatter` and is the number
    to actually judge the measurement by -- if the source is real and
    well-centred, the sub-bands agree to a fraction of a pixel; if they
    scatter by several pixels, the "detection" is probably noise, a blended
    neighbour, or the host galaxy pulling the centroid around with
    wavelength.

    `kernel_fwhm` defaults to the MRS PSF FWHM (in pixels) at the mean
    wavelength actually used, via `_mrs_fwhm_pixels`. Note that on a
    full-range (5-28 um) collapse the PSF varies by a factor of several
    across the stack, so restricting `wave_range` to a single channel gives
    a better-matched filter as well as a sharper PSF.

    Set `plot_file` to write a QA PNG.

    Returns a dict with x, y, x_err, y_err, x_scatter, y_scatter, snr, the
    collapsed `image`/`err`/`snr_image`, and the per-sub-band positions."""
    coll = make_weighted_image(cube_file, wave_range=wave_range, clip_sigma=clip_sigma,
                               n_sub=n_sub, min_frac=min_frac)

    if kernel_fwhm is None:
        fwhm_pix = _mrs_fwhm_pixels(cube_file, wave=coll["wave"])
        kernel_fwhm = float(np.clip(np.nanmean(fwhm_pix), 1.5, 8.0))

    print(f"Collapsed {coll['n_slices']} slices "
          f"({coll['wave_range'][0]:.2f}-{coll['wave_range'][1]:.2f} um) using "
          f"{coll['weighting']} weights; detection kernel FWHM = {kernel_fwhm:.2f} px")

    det_kw = dict(search_radius=search_radius, nsigma=nsigma, npixels=npixels,
                  kernel_fwhm=kernel_fwhm, subtract_background=subtract_background,
                  bkg_boxsize=bkg_boxsize, max_area_factor=max_area_factor)

    det = find_centroid_sourcex(coll["image"], err_image=coll["err"], x0=x0, y0=y0, **det_kw)

    sub_x, sub_y, sub_wave = [], [], []
    for img, erri, wr in zip(coll.get("sub_images", []), coll.get("sub_errs", []),
                              coll.get("sub_wave_ranges", [])):
        try:
            sd = find_centroid_sourcex(img, err_image=erri, x0=det["x"], y0=det["y"], **det_kw)
            sub_x.append(sd["x"])
            sub_y.append(sd["y"])
            sub_wave.append(0.5 * (wr[0] + wr[1]))
        except RuntimeError:
            continue

    if len(sub_x) >= 2:
        x_scatter = float(np.std(sub_x, ddof=1))
        y_scatter = float(np.std(sub_y, ddof=1))
    else:
        x_scatter = y_scatter = np.nan

    result = {
        "x": det["x"], "y": det["y"],
        "x_err": det["x_err"], "y_err": det["y_err"],
        "x_scatter": x_scatter, "y_scatter": y_scatter,
        "snr": det["snr"], "flux": det["flux"], "flux_err": det["flux_err"],
        "n_detected": det["n_detected"], "catalog": det["catalog"],
        "segment_map": det["segment_map"],
        "detection_image": det["detection_image"], "background": det["background"],
        "image": coll["image"], "err": coll["err"], "snr_image": coll["snr"],
        "coverage": coll["coverage"], "wave": coll["wave"],
        "wave_range": coll["wave_range"], "n_slices": coll["n_slices"],
        "weighting": coll["weighting"],
        "sub_x": np.array(sub_x), "sub_y": np.array(sub_y),
        "sub_wavelength": np.array(sub_wave),
        "kernel_fwhm": kernel_fwhm, "x0": x0, "y0": y0,
    }

    if plot_file:
        plot_centroid_qa(result, plot_file)
    return result


def pixel_extent(x_start, y_start, nx, ny):
    """`extent=` for `imshow` of a cutout starting at pixel (x_start, y_start).

    matplotlib's `extent` gives the outer *edges* of the displayed array, but
    every centroid in this module uses the numpy/photutils convention where
    integer coordinates fall at pixel *centres*. Passing the naive
    `[x_start, x_start + nx, ...]` therefore shifts the image half a pixel
    relative to any marker plotted on top of it -- which makes a perfectly
    good centroid look visibly off-centre. Always::

        cut = img[y0:y0+ny, x0:x0+nx]
        ax.imshow(cut, origin="lower", extent=pixel_extent(x0, y0, nx, ny))
    """
    return (x_start - 0.5, x_start + nx - 0.5,
            y_start - 0.5, y_start + ny - 0.5)


def plot_centroid_qa(result, plot_file):
    """Four-panel QA figure for a `find_source_centroid` result: the weighted
    collapsed image, the mesh background that was removed, the
    background-subtracted detection image the measurement was actually made
    on, and the segmentation map. Each is marked with the initial guess
    (cyan x) and the measured centroid (red +).

    Look at panel 3: if the source is not an obvious compact blob there, the
    measured position is not trustworthy no matter what the formal error
    says."""
    try:
        import matplotlib
        # matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from astropy.visualization import ZScaleInterval

        img = result["image"]
        det_img = result.get("detection_image")
        back = result.get("background")

        fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))

        def _show(a, arr, title, cmap="viridis"):
            finite = np.isfinite(arr)
            if finite.any():
                vmin, vmax = ZScaleInterval().get_limits(arr[finite])
            else:
                vmin, vmax = 0, 1
            a.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
            a.set_title(title, fontsize=10)

        _show(ax[0], img, f"Weighted collapse\n{result['wave_range'][0]:.1f}-"
                          f"{result['wave_range'][1]:.1f} um, {result['n_slices']} slices")
        _show(ax[1], back if back is not None else img, "Mesh background removed")
        _show(ax[2], det_img if det_img is not None else img,
              "Detection image (bkg-subtracted)")

        ax[3].imshow(result["segment_map"].data, origin="lower", cmap="tab20",
                     interpolation="nearest")
        ax[3].set_title(f"Segmentation ({result['n_detected']} source(s))", fontsize=10)

        zoom = 12
        for a in ax:
            if result.get("x0") is not None:
                a.plot(result["x0"], result["y0"], "x", color="cyan", ms=9, mew=2,
                       label="guess")
            a.plot(result["x"], result["y"], "+", color="red", ms=12, mew=2,
                   label="centroid")
        ax[2].set_xlim(result["x"] - zoom, result["x"] + zoom)
        ax[2].set_ylim(result["y"] - zoom, result["y"] + zoom)
        ax[2].set_title("Detection image (zoom)", fontsize=10)
        ax[0].legend(loc="upper right", fontsize=8)

        fig.suptitle(f"x={result['x']:.2f}+/-{result['x_err']:.2f}, "
                     f"y={result['y']:.2f}+/-{result['y_err']:.2f}  |  "
                     f"SNR={result['snr']:.1f}  |  sub-band scatter "
                     f"({result['x_scatter']:.2f}, {result['y_scatter']:.2f}) px")
        plt.tight_layout()
        outdir = os.path.dirname(os.path.abspath(plot_file))
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        plt.savefig(plot_file, dpi=120)
        plt.close(fig)
        print(f"  Centroid QA plot: {plot_file}")
    except Exception as exc:
        print(f"  (skipped centroid QA plot: {exc})")


def refine_and_reprocess(stage3_cube_file, out_dir, product_name, x0=None, y0=None,
                          bgsub_cube_file=None, centroid_cube=None,
                          bkg_mode="polynomial", aper_rad=7, ann_width=6, fwhm_scale=2.5,
                          poly_k=3, bin_size=9, cube_resolution="high", pool_size=None,
                          min_aper_rad=5.0, use_abi_err=False,
                          wave_range=None, clip_sigma=5.0, n_sub=4, search_radius=8.0,
                          nsigma=3.0, npixels=5, kernel_fwhm=None, min_frac=0.1,
                          refined_tag="refined", ifu_rfcorr=True, ifu_rscale=None,
                          ifu_covar_scale=1.0, save_plots=True):
    """Centroid -> re-fit background -> extract, the whole second pass.

      1. Collapse a cube into one inverse-variance weighted image and find
         the source with SExtractor-style detection (`find_source_centroid`).
         This replaces the old bin-and-fit-every-N-slices approach, which
         needed each bin to be an independent detection and fell over
         entirely when the ERR array came back NaN.

         By default the centroid is measured on `stage3_cube_file` itself.
         Pass `bgsub_cube_file` (or `centroid_cube`) to centroid on a
         background-subtracted cube instead, which gives a cleaner image if
         the field has structured background -- but it is no longer required,
         so a first pass is not a prerequisite for this function.

      2. Re-run AstroBkgInterp at the refined position on the *original*
         Stage-3 cube. Background masking is position-sensitive, so redoing
         it properly means going back to the un-subtracted cube rather than
         subtracting again from an already-subtracted one. Saved as
         '<product_name>_<refined_tag>_abi_bgsub_s3d.fits' so the first-pass
         result is not overwritten.

      3. Extract a flux-calibrated, background-subtracted 1D spectrum from
         that new cube at the refined position
         (`extract_spectrum_manual`, point-source mode with aperture
         correction and residual-fringe correction on), saved next to the
         cube as '<...>_refined_x1d.fits'.

    Parameters worth knowing about
    ------------------------------
    wave_range : (lo, hi) microns, or None
        Restrict the centroiding collapse. The MRS PSF is several times
        sharper at 5 um than at 28 um, so e.g. `(4.9, 7.7)` (channel 1)
        centroids more precisely, if the source is bright enough there.
    n_sub : int
        Number of independent sub-band images used to measure the position
        scatter. Judge the result by that scatter, not the formal error.

    Returns (centroid_dict, new_bgsub_cube_path, (wave, flux, err)).
    """
    # The old signature was (stage3_cube, bgsub_cube, out_dir, product_name, x0, y0)
    # with the background-subtracted cube second and positional. Catch that call
    # rather than silently treating a FITS path as an output directory.
    if isinstance(out_dir, str) and out_dir.lower().endswith((".fits", ".fits.gz")):
        raise TypeError(
            "refine_and_reprocess()'s signature changed: the background-subtracted cube is "
            "no longer a required second positional argument (the centroid is measured on "
            "the Stage-3 cube by default). Call it as refine_and_reprocess(stage3_cube, "
            "out_dir, product_name, x0=..., y0=...), and pass bgsub_cube_file=... only if "
            "you specifically want to centroid on an already-background-subtracted cube.")

    cen_cube = centroid_cube or bgsub_cube_file or stage3_cube_file
    print(f"Centroiding on: {cen_cube}")

    plot_file = (os.path.join(out_dir, f"{product_name}_{refined_tag}_centroid_qa.png")
                 if save_plots else None)

    centroid = find_source_centroid(
        cen_cube, x0=x0, y0=y0, wave_range=wave_range, clip_sigma=clip_sigma,
        n_sub=n_sub, search_radius=search_radius, nsigma=nsigma, npixels=npixels,
        kernel_fwhm=kernel_fwhm, min_frac=min_frac, plot_file=plot_file)

    x_ref, y_ref = centroid["x"], centroid["y"]
    print(f"Refined position: x={x_ref:.3f}+/-{centroid['x_err']:.3f}, "
          f"y={y_ref:.3f}+/-{centroid['y_err']:.3f}  (detection SNR={centroid['snr']:.1f}, "
          f"{centroid['n_detected']} source(s) in frame)")
    if np.isfinite(centroid["x_scatter"]):
        print(f"  Sub-band scatter over {len(centroid['sub_x'])} bands: "
              f"x={centroid['x_scatter']:.3f}, y={centroid['y_scatter']:.3f} px")
        if max(centroid["x_scatter"], centroid["y_scatter"]) > 1.0:
            print("  NOTE: sub-band positions disagree by more than a pixel -- the "
                  "detection may be noise-dominated or blended. Check the QA plot "
                  "before trusting this position.")
    if x0 is not None and y0 is not None:
        print(f"  Moved {np.hypot(x_ref - x0, y_ref - y0):.2f} px from the initial guess "
              f"({x0:.2f}, {y0:.2f})")

    refined_product_name = f"{product_name}_{refined_tag}"
    new_bgsub_path = run_astrobkginterp_background(
        stage3_cube_file, x_ref, y_ref, out_dir, refined_product_name,
        bkg_mode=bkg_mode, aper_rad=aper_rad, ann_width=ann_width, fwhm_scale=fwhm_scale,
        poly_k=poly_k, bin_size=bin_size, cube_resolution=cube_resolution, pool_size=pool_size,
        min_aper_rad=min_aper_rad, use_abi_err=use_abi_err,
    )
    print(f"Re-fit background at the refined position: {new_bgsub_path}")

    x1d_file = os.path.join(out_dir, f"{refined_product_name}_x1d.fits")
    wave, flux, err, _, meta = extract_spectrum_manual(
        new_bgsub_path, out_file=x1d_file, x=x_ref, y=y_ref, ifu_rfcorr=ifu_rfcorr,
        ifu_rscale=ifu_rscale, ifu_covar_scale=ifu_covar_scale, return_meta=True)
    print(f"Extracted 1D spectrum ({meta['n_good']}/{len(wave)} finite points, "
          f"{wave[0]:.2f}-{wave[-1]:.2f} um, flux in {meta['flux_unit']}): {x1d_file}")

    centroid["x1d_file"] = x1d_file
    centroid["bgsub_cube"] = new_bgsub_path
    centroid["spectrum_meta"] = meta

    if save_plots:
        _plot_spectrum_qa(wave, flux, err, meta,
                          os.path.join(out_dir, f"{refined_product_name}_x1d.png"),
                          title=f"{product_name} @ (x={x_ref:.2f}, y={y_ref:.2f})")

    return centroid, new_bgsub_path, (wave, flux, err)


def _plot_spectrum_qa(wave, flux, err, meta, plot_file, title=""):
    """Quick-look plot of the extracted 1D spectrum with its error band."""
    try:
        import matplotlib
        # matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        good = np.isfinite(wave) & np.isfinite(flux)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(wave[good], flux[good], lw=0.7, color="k")
        if err is not None and np.isfinite(err).any():
            ax.fill_between(wave[good], (flux - err)[good], (flux + err)[good],
                            color="C0", alpha=0.3, lw=0)
        ax.axhline(0, color="grey", lw=0.6, ls="--")
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel(f"Flux ({meta.get('flux_unit', '')})")
        ax.set_title(title)
        finite_flux = flux[good]
        if finite_flux.size:
            lo, hi = np.nanpercentile(finite_flux, [1, 99.5])
            pad = 0.15 * (hi - lo) if hi > lo else 1.0
            ax.set_ylim(lo - pad, hi + pad)
        plt.tight_layout()
        plt.savefig(plot_file, dpi=120)
        plt.close(fig)
        print(f"  Spectrum QA plot: {plot_file}")
    except Exception as exc:
        print(f"  (skipped spectrum QA plot: {exc})")


# ===========================================================================
# Blended / binary sources: PSF deblending and per-component extraction
# ===========================================================================

def _psf_image(ny, nx, xc, yc, fwhm, model="moffat", beta=2.5):
    """A unit-sum 2D PSF of the given FWHM (pixels) centred at (xc, yc).

    'moffat' is the default because a real MRS cube PSF has broad wings that
    a Gaussian badly under-predicts -- and when you are subtracting one
    component off another, the wings are exactly the part that lands on top
    of your target.

    `model` may also be a **callable** `f(ny, nx, xc, yc, fwhm) -> 2D array`,
    which is the hook for a real empirical PSF. Neither analytic form matches
    the true MRS profile: fitting one to this data leaves a symmetric +/-8%
    dipole residual at the core, which is large enough that a close pair can
    be faked or hidden by it. If you install `stpsf` (the successor to
    webbpsf, which does model MIRI MRS IFU PSFs) you can pass a callable that
    renders a proper PSF and every fit here improves without further changes.
    See `make_stamp_psf` for turning a rendered stamp into such a callable."""
    if callable(model):
        img = np.asarray(model(ny, nx, xc, yc, fwhm), dtype=float)
        if img.shape != (ny, nx):
            raise ValueError(f"Callable psf_model returned {img.shape}, expected {(ny, nx)}.")
        img = np.where(np.isfinite(img), img, 0.0)
    else:
        y, x = np.mgrid[0:ny, 0:nx]
        r2 = (x - xc) ** 2 + (y - yc) ** 2
        if model == "gaussian":
            sigma = fwhm / 2.3548
            img = np.exp(-0.5 * r2 / sigma ** 2)
        elif model == "moffat":
            alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
            img = (1.0 + r2 / alpha ** 2) ** (-beta)
        else:
            raise ValueError(
                f"Unknown psf_model {model!r} (use 'moffat', 'gaussian', or a callable).")
    total = img.sum()
    return img / total if total > 0 else img


def make_stamp_psf(stamp, stamp_center=None, oversample=1, normalize_fwhm=None):
    """Wrap a rendered PSF stamp into a callable usable as `psf_model`.

    Use this to feed a real MIRI MRS PSF (e.g. from `stpsf`) into
    `fit_blended_sources` / `psf_deblend_cube` in place of the analytic
    Moffat::

        import stpsf
        miri = stpsf.MIRI()
        miri.mode = 'IFU'
        miri.band = '1A'
        hdul = miri.calc_psf(oversample=4, fov_pixels=25, monochromatic=5.3e-6)
        psf = make_stamp_psf(hdul[0].data, oversample=4)
        blend = fit_blended_sources(cube, x0, y0, psf_model=psf)

    `stamp` is shifted to the requested (xc, yc) by bicubic interpolation and
    resampled onto the cube grid. The `fwhm` argument is ignored unless
    `normalize_fwhm` is given, in which case the stamp is treated as having
    that FWHM (in cube pixels) and is rescaled so the fitter's width
    parameter still means something."""
    from scipy.ndimage import map_coordinates

    stamp = np.asarray(stamp, dtype=float)
    stamp = np.where(np.isfinite(stamp), stamp, 0.0)
    sny, snx = stamp.shape
    if stamp_center is None:
        stamp_center = ((snx - 1) / 2.0, (sny - 1) / 2.0)
    sx0, sy0 = stamp_center

    def psf(ny, nx, xc, yc, fwhm):
        scale = float(oversample)
        if normalize_fwhm:
            scale = scale * (normalize_fwhm / fwhm) if fwhm > 0 else scale
        y, x = np.mgrid[0:ny, 0:nx]
        # Cube pixel -> oversampled stamp pixel.
        sxx = (x - xc) * scale + sx0
        syy = (y - yc) * scale + sy0
        return map_coordinates(stamp, [syy.ravel(), sxx.ravel()], order=3,
                               mode="constant", cval=0.0).reshape(ny, nx)

    return psf


def _fit_n_components(img, err, n, fwhm0, model, beta, fit_fwhm_scale, init_xy,
                       fixed_scale=None):
    """Least-squares fit of `n` PSFs plus a tilted-plane background to a
    cutout. Returns (params_dict, chi2, n_free).

    `fixed_scale` pins the PSF width instead of fitting it. This matters a
    lot for model comparison: if the width is free, a two-component fit can
    always mimic one broad PSF by shrinking both components and separating
    them slightly, so it "wins" on chi2 even when the source is single. The
    caller fits the width once on the 1-component model and then holds it."""
    from scipy.optimize import least_squares

    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    good = np.isfinite(img) & np.isfinite(err) & (err > 0)
    if good.sum() < 4 * n + 4:
        raise RuntimeError("Not enough usable pixels in the cutout to fit.")

    d, e = img[good], err[good]
    peak = np.nanmax(img[good])
    free_scale = fit_fwhm_scale and fixed_scale is None

    p0, lo, hi = [], [], []
    if free_scale:
        p0.append(1.0); lo.append(0.7); hi.append(2.0)
    for (xi, yi) in init_xy[:n]:
        p0 += [xi, yi, max(peak, 1e-12)]
        lo += [-0.5, -0.5, 0.0]
        hi += [nx - 0.5, ny - 0.5, np.inf]
    p0 += [float(np.nanmedian(img[good])), 0.0, 0.0]
    lo += [-np.inf, -np.inf, -np.inf]
    hi += [np.inf, np.inf, np.inf]

    def unpack(p):
        k = 0
        if free_scale:
            scale = p[0]; k = 1
        else:
            scale = 1.0 if fixed_scale is None else float(fixed_scale)
        comps = []
        for _ in range(n):
            comps.append((p[k], p[k + 1], p[k + 2])); k += 3
        return scale, comps, p[k:k + 3]

    def model_image(p):
        scale, comps, bg = unpack(p)
        m = bg[0] + bg[1] * (xx - nx / 2.0) + bg[2] * (yy - ny / 2.0)
        for (xi, yi, ai) in comps:
            m = m + ai * _psf_image(ny, nx, xi, yi, fwhm0 * scale, model, beta)
        return m

    res = least_squares(lambda p: (model_image(p)[good] - d) / e, p0,
                        bounds=(lo, hi), method="trf", max_nfev=4000)
    scale, comps, bg = unpack(res.x)
    chi2 = float(np.sum(((model_image(res.x)[good] - d) / e) ** 2))
    return ({"scale": scale, "components": comps, "background": bg,
             "model": model_image(res.x), "ngood": int(good.sum())},
            chi2, len(p0))


def fit_blended_sources(cube_file, x0, y0, n_sources=2, wave_range=(4.9, 7.7),
                         cutout_half=9, psf_model="moffat", beta=2.5,
                         fit_fwhm_scale=True, clip_sigma=5.0, min_frac=0.1,
                         min_separation_fwhm=0.75, plot_file=None):
    """Fit `n_sources` overlapping PSFs to a weighted collapse of the cube,
    and report whether that many components is actually justified.

    Centroiding a blend returns the flux-weighted midpoint of the pair, which
    is not where either source is. This fits them as separate components
    instead.

    `wave_range` defaults to channel 1 (4.9-7.7 um) because that is where the
    MRS PSF is sharpest (~2.4 px FWHM, versus ~9 px at 28 um) and therefore
    where a close pair is most likely to be separable at all. Positions
    measured here are the ones to hold fixed when deblending the rest of the
    cube.

    Fits are run for 1..`n_sources` components and compared by BIC, so you can
    see whether a second component is genuinely required by the data or is
    just soaking up noise. `delta_bic` in the returned dict is
    BIC(1) - BIC(n_sources): strongly positive means the extra component is
    justified (>10 is conventionally decisive), around zero or negative means
    it is not, and you should treat the source as single.

    Returns a dict with `positions` (list of (x, y) in full-frame pixels),
    `fluxes`, `separation`, `fwhm` (the fitted PSF FWHM in pixels),
    `fits` (per-n statistics) and `delta_bic`."""
    coll = make_weighted_image(cube_file, wave_range=wave_range, clip_sigma=clip_sigma,
                               n_sub=0, min_frac=min_frac)
    img_full, err_full = coll["image"], coll["err"]

    fwhm0 = float(np.nanmean(_mrs_fwhm_pixels(cube_file, wave=coll["wave"])))

    xi, yi = int(round(x0)), int(round(y0))
    h = int(cutout_half)
    ylo, yhi = max(yi - h, 0), min(yi + h + 1, img_full.shape[0])
    xlo, xhi = max(xi - h, 0), min(xi + h + 1, img_full.shape[1])
    img = np.array(img_full[ylo:yhi, xlo:xhi], dtype=float)
    err = np.array(err_full[ylo:yhi, xlo:xhi], dtype=float)

    # Floor the errors. A weighted collapse can leave a few spaxels with an
    # essentially zero uncertainty (very few contributing slices, or a cube
    # whose ERR extension is broken), and 1/err^2 weighting then hands those
    # pixels near-infinite leverage: they alone drive the fit and drag the
    # centroid off the source. Clip to a small fraction of the typical error
    # so no single pixel can dominate. Also fall back entirely if ERR is
    # unusable, which is the case for the first-pass cube here.
    good_err = err[np.isfinite(err) & (err > 0)]
    if good_err.size < 0.1 * err.size:
        err = np.full_like(img, np.nanstd(img))
    else:
        floor = 0.05 * np.median(good_err)
        err = np.where(np.isfinite(err) & (err > 0), err, np.median(good_err))
        err = np.maximum(err, floor)

    # Remove a local pedestal so the amplitude guesses are sane; the fit
    # carries its own tilted-plane background term on top of this.
    img_med = np.nanmedian(img)

    # Start the components split along the isophote's major axis, which is
    # the direction a blend is elongated in -- starting both at the same
    # place gives the fitter no reason to separate them.
    cx, cy = x0 - xlo, y0 - ylo
    try:
        from photutils.morphology import data_properties
        props = data_properties(np.nan_to_num(img - img_med))
        theta = float(np.asarray(getattr(props.orientation, "value", props.orientation)))
    except Exception:
        theta = 0.0

    def _starts(n):
        """Initial component layouts to try for an n-component fit.

        A single start is not enough: least_squares is local, and if the
        components begin much closer together than the true pair it happily
        converges to both sitting on the same peak (reporting a spurious
        'single source'), while if they begin too far apart it latches onto
        background structure. Scan a range of separations and a few position
        angles and keep whichever converges best."""
        if n == 1:
            return [[(cx, cy)]]
        out = []
        angles = [theta, theta + np.pi / 4, theta + np.pi / 2, theta + 3 * np.pi / 4]
        for s in (0.5, 1.0, 1.6, 2.5) if fwhm0 else (1.0,):
            for ang in angles:
                sep = s * fwhm0
                out.append([(cx + (i - (n - 1) / 2.0) * sep * np.cos(ang),
                             cy + (i - (n - 1) / 2.0) * sep * np.sin(ang))
                            for i in range(n)])
        return out

    fits_by_n = {}
    for n in range(1, int(n_sources) + 1):
        best_try = None
        for init_xy in _starts(n):
            try:
                # The PSF width stays free at every n. Locking it to the
                # 1-component value is tempting but wrong: on a real blend the
                # 1-component fit inflates its width to span the pair, and
                # forcing that width on the 2-component fit makes both
                # components collapse onto the centre -- a false negative.
                fit, chi2, npar = _fit_n_components(img, err, n, fwhm0, psf_model,
                                                     beta, fit_fwhm_scale, init_xy)
            except Exception as exc:
                warnings.warn(f"{n}-component fit failed: {exc!r}")
                continue
            if best_try is None or chi2 < best_try[1]:
                best_try = (fit, chi2, npar)
        if best_try is None:
            continue
        fit, chi2, npar = best_try
        fits_by_n[n] = {"fit": fit, "chi2": chi2, "npar": npar,
                         "ngood": fit["ngood"],
                         "chi2_red": chi2 / max(fit["ngood"] - npar, 1)}

    if not fits_by_n:
        raise RuntimeError("No component fit converged on this cutout.")

    # The cube's ERR badly underestimates the true scatter here: cube_build
    # correlates neighbouring spaxels, and no analytic PSF matches the real
    # MRS profile to the precision those errors imply. Reduced chi2 therefore
    # comes out in the thousands, and an un-rescaled BIC would "decisively"
    # favour extra components every single time. Rescale the errors so the
    # 1-component fit has chi2_red = 1, then compare -- that asks the only
    # meaningful question: does a second component help *more than the
    # model's own systematic mismatch already hurts*?
    s2 = fits_by_n[1]["chi2_red"] if 1 in fits_by_n else 1.0
    s2 = s2 if np.isfinite(s2) and s2 > 0 else 1.0
    for n, f in fits_by_n.items():
        f["bic"] = f["chi2"] / s2 + f["npar"] * np.log(f["ngood"])

    n_use = max(fits_by_n)
    # Accept extra components only if they are BOTH statistically justified and
    # physically resolvable. Two sources closer than ~half a FWHM cannot be
    # separated from imaging at all, so a "detection" at that separation is the
    # fitter widening one PSF by splitting it in two -- which is precisely how
    # an imperfect analytic PSF gets absorbed. Walk down until a fit passes both.
    rejected = []
    while n_use > 1:
        cand = fits_by_n[n_use]["fit"]
        cxy = [(c[0], c[1]) for c in cand["components"]]
        min_sep = min(np.hypot(a[0] - b[0], a[1] - b[1])
                      for i, a in enumerate(cxy) for b in cxy[i + 1:])
        prev = max(k for k in fits_by_n if k < n_use)
        dbic = fits_by_n[prev]["bic"] - fits_by_n[n_use]["bic"]
        sep_fwhm = min_sep / (fwhm0 * fits_by_n[n_use]["fit"]["scale"])
        if sep_fwhm < min_separation_fwhm:
            rejected.append(f"n={n_use}: closest pair only {sep_fwhm:.2f} FWHM apart "
                            f"(below the {min_separation_fwhm:.2f} FWHM resolution limit)")
        elif dbic <= 10:
            rejected.append(f"n={n_use}: delta_BIC={dbic:.1f} over n={prev}, not significant")
        else:
            break
        n_use = prev

    best = fits_by_n[n_use]["fit"]

    comps = sorted(best["components"], key=lambda c: -c[2])   # brightest first
    positions = [(float(c[0] + xlo), float(c[1] + ylo)) for c in comps]
    fluxes = [float(c[2]) for c in comps]
    fwhm_fit = float(fwhm0 * best["scale"])

    delta_bic = (fits_by_n[1]["bic"] - fits_by_n[n_use]["bic"]) if 1 in fits_by_n and n_use > 1 else np.nan

    sep = np.nan
    if len(positions) >= 2:
        sep = float(np.hypot(positions[0][0] - positions[1][0],
                              positions[0][1] - positions[1][1]))

    print(f"Fit {n_use} component(s) on {coll['wave_range'][0]:.2f}-"
          f"{coll['wave_range'][1]:.2f} um; PSF FWHM = {fwhm_fit:.2f} px "
          f"(expected {fwhm0:.2f})")
    for i, ((px, py), fl) in enumerate(zip(positions, fluxes)):
        print(f"  component {i}: x={px:.3f}, y={py:.3f}, relative flux={fl / max(fluxes):.3f}")
    if np.isfinite(sep):
        print(f"  separation = {sep:.2f} px = {sep / fwhm_fit:.2f} PSF FWHM")

    print(f"  1-component fit chi2_red = {s2:.1f} (errors rescaled by "
          f"{np.sqrt(s2):.1f}x before comparing models)")
    if s2 > 25:
        psf_name = ("a supplied empirical PSF" if callable(psf_model)
                    else f"an analytic {psf_model}")
        print("  NOTE: the 1-component fit is a poor match in an absolute sense. That is "
              f"PSF-model mismatch -- {psf_name} only approximates the real MRS "
              "profile -- and it is why the comparison below is done on rescaled "
              "errors with a resolution-limit cut rather than on raw chi2.")

    for msg in rejected:
        print(f"  REJECTED {msg}")

    if np.isfinite(delta_bic):
        verdict = ("clearly justified" if delta_bic > 10 else
                   "marginal" if delta_bic > 2 else "NOT justified")
        print(f"  delta_BIC(1 -> {n_use}) = {delta_bic:.1f} -- extra component(s) {verdict}")

    if n_use == 1:
        print("  => SINGLE SOURCE. No deblending needed; extract normally at this position.")
        if rejected:
            print("     (A multi-component fit did lower chi2, but only by splitting one PSF "
                  "into sub-resolution pieces. Check the residual panel: a real companion is "
                  "a one-sided positive blob, while PSF mismatch is a symmetric dipole at the core.)")
    else:
        print(f"  => {n_use} COMPONENTS resolved. Inspect the residual panel before trusting it.")

    result = {"positions": positions, "fluxes": fluxes, "separation": sep,
              "separation_fwhm": sep / fwhm_fit if np.isfinite(sep) else np.nan,
              "fwhm": fwhm_fit, "fwhm_expected": fwhm0, "delta_bic": delta_bic,
              "chi2_red_single": s2,
              "n_components": n_use, "fits": fits_by_n, "psf_model": psf_model,
              "beta": beta, "cutout_origin": (xlo, ylo), "cutout_image": img,
              "cutout_err": err, "wave_range": coll["wave_range"]}

    if plot_file:
        _plot_blend_fit(result, plot_file)
    return result


def _plot_blend_fit(result, plot_file):
    """Data / model / residual panels plus a radial profile for a
    `fit_blended_sources` result.

    The residual gets its own symmetric colour scale -- on the data's scale a
    residual that is a few percent of the peak is invisible, which is exactly
    the regime that decides whether a fit is good. The radial profile panel is
    the real diagnostic for a suspected blend: a true companion shows up as a
    one-sided bump at its separation, whereas a PSF-model mismatch shows up as
    a smooth symmetric over/under-shoot at all radii."""
    try:
        import matplotlib
        # matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        img = result["cutout_image"]
        fits_by_n = result["fits"]
        n_use = result["n_components"]
        model = fits_by_n[n_use]["fit"]["model"]
        resid = img - model
        xlo, ylo = result["cutout_origin"]

        fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
        vmin, vmax = np.nanpercentile(img, [2, 99.5])
        for a, arr, t in zip(ax[:2], [img, model], ["Data (weighted collapse)", "Model"]):
            im = a.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
            a.set_title(t, fontsize=10)
            plt.colorbar(im, ax=a, fraction=0.046)

        rlim = np.nanpercentile(np.abs(resid), 99.5)
        rlim = rlim if np.isfinite(rlim) and rlim > 0 else 1.0
        im = ax[2].imshow(resid, origin="lower", vmin=-rlim, vmax=rlim, cmap="RdBu_r")
        peak = np.nanmax(img)
        ax[2].set_title(f"Residual (+/-{100 * rlim / peak:.1f}% of peak)", fontsize=10)
        plt.colorbar(im, ax=ax[2], fraction=0.046)

        for i, (px, py) in enumerate(result["positions"]):
            for a in ax[:3]:
                a.plot(px - xlo, py - ylo, "+", color="red", ms=12, mew=2)
            ax[0].annotate(str(i), (px - xlo, py - ylo), color="white",
                           fontsize=11, xytext=(4, 4), textcoords="offset points")

        # Radial profile about the brightest component.
        px, py = result["positions"][0]
        ny, nx = img.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        r = np.hypot(xx - (px - xlo), yy - (py - ylo)).ravel()
        order = np.argsort(r)
        ax[3].plot(r[order], img.ravel()[order], ".", color="0.4", ms=3, label="data")
        for n in sorted(fits_by_n):
            m = fits_by_n[n]["fit"]["model"].ravel()[order]
            ax[3].plot(r[order], m, "-", lw=1.4, label=f"{n}-component model")
        ax[3].set_xlabel("radius from component 0 (px)")
        ax[3].set_ylabel("surface brightness")
        ax[3].set_xlim(0, min(nx, ny) / 2.0)
        ax[3].set_yscale("symlog", linthresh=max(peak, 1e-12) * 1e-3)
        ax[3].legend(fontsize=8)
        ax[3].set_title("Radial profile", fontsize=10)

        sep_txt = (f"sep={result['separation']:.2f} px "
                   f"({result['separation_fwhm']:.2f} FWHM), "
                   f"delta_BIC={result['delta_bic']:.1f}"
                   if n_use > 1 else "single component")
        fig.suptitle(f"{n_use} component(s), {sep_txt}, "
                     f"1-comp chi2_red={result['chi2_red_single']:.0f}")
        plt.tight_layout()
        outdir = os.path.dirname(os.path.abspath(plot_file))
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        plt.savefig(plot_file, dpi=120)
        plt.close(fig)
        print(f"  Blend-fit plot: {plot_file}")
    except Exception as exc:
        print(f"  (skipped blend-fit plot: {exc})")


def psf_deblend_cube(cube_file, positions, psf_model="moffat", beta=2.5,
                      fwhm_scale=1.0, cutout_half=9, chunk=512):
    """Fit per-wavelength amplitudes for PSFs fixed at `positions`.

    Positions and PSF widths are held fixed, so at every slice the only free
    parameters are the component amplitudes and a tilted-plane local
    background -- which makes the problem *linear*, solvable with one
    least-squares call per slice and no iteration or starting guesses. That
    is both far faster and far more stable than re-fitting positions slice by
    slice, which is hopeless anyway at long wavelengths where the PSF is
    several times wider than the separation.

    Returns a dict with `amplitudes` (n_components, nz), `amplitude_errs`,
    and the fixed inputs needed to rebuild the model cube."""
    positions = [(float(px), float(py)) for px, py in positions]
    n_comp = len(positions)

    wave = get_mrs_wavelength(cube_file)
    fwhm_all = _mrs_fwhm_pixels(cube_file, wave=wave) * float(fwhm_scale)
    nz = len(wave)

    cx = int(round(np.mean([p[0] for p in positions])))
    cy = int(round(np.mean([p[1] for p in positions])))

    with fits.open(cube_file, memmap=True) as hdul:
        ny, nx = hdul["SCI"].data.shape[1:]
    h = int(cutout_half)
    ylo, yhi = max(cy - h, 0), min(cy + h + 1, ny)
    xlo, xhi = max(cx - h, 0), min(cx + h + 1, nx)
    cny, cnx = yhi - ylo, xhi - xlo
    yy, xx = np.mgrid[0:cny, 0:cnx]

    amps = np.full((n_comp, nz), np.nan)
    amp_errs = np.full((n_comp, nz), np.nan)

    with fits.open(cube_file, memmap=True) as hdul:
        sci = hdul["SCI"].data
        has_err = "ERR" in hdul
        errext = hdul["ERR"].data if has_err else None

        for start in range(0, nz, chunk):
            stop = min(start + chunk, nz)
            dblk = np.asarray(sci[start:stop, ylo:yhi, xlo:xhi], dtype=np.float32).astype(float)
            if has_err:
                eblk = np.squeeze(np.asarray(errext[start:stop, ylo:yhi, xlo:xhi],
                                              dtype=np.float32)).astype(float)
                eblk = eblk.reshape(dblk.shape)
            else:
                eblk = np.ones_like(dblk)

            for k in range(stop - start):
                kk = start + k
                d, e = dblk[k], eblk[k]
                ok = np.isfinite(d) & np.isfinite(e) & (e > 0)
                if ok.sum() < n_comp + 4:
                    continue
                cols = [_psf_image(cny, cnx, px - xlo, py - ylo, fwhm_all[kk],
                                    psf_model, beta)[ok]
                        for px, py in positions]
                cols += [np.ones(ok.sum()),
                         (xx - cnx / 2.0)[ok], (yy - cny / 2.0)[ok]]
                A = np.vstack(cols).T / e[ok][:, None]
                b = d[ok] / e[ok]
                try:
                    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
                    cov = np.linalg.pinv(A.T @ A)
                except np.linalg.LinAlgError:
                    continue
                amps[:, kk] = coef[:n_comp]
                amp_errs[:, kk] = np.sqrt(np.maximum(np.diag(cov)[:n_comp], 0.0))

    return {"amplitudes": amps, "amplitude_errs": amp_errs, "wave": wave,
            "positions": positions, "fwhm": fwhm_all, "psf_model": psf_model,
            "beta": beta, "cutout": (xlo, xhi, ylo, yhi)}


def write_companion_subtracted_cube(cube_file, deblend, keep_index, out_path):
    """Write a copy of `cube_file` with every fitted component *except*
    `keep_index` subtracted off.

    Subtracting the contaminants and then re-extracting normally (rather than
    just reporting the fitted amplitude of the component you want) keeps the
    whole jwst flux calibration path intact -- the point-source aperture
    correction, the MJy/sr -> Jy conversion, the covariance error scaling.
    The PSF model is only ever used to remove the neighbour, never to measure
    your target's flux, so an imperfect analytic PSF costs you a small
    residual rather than a systematically wrong flux scale."""
    xlo, xhi, ylo, yhi = deblend["cutout"]
    cny, cnx = yhi - ylo, xhi - xlo
    amps, positions, fwhm_all = deblend["amplitudes"], deblend["positions"], deblend["fwhm"]
    n_comp = len(positions)

    with fits.open(cube_file) as hdul:
        sci = np.array(hdul["SCI"].data, dtype=np.float32)
        for k in range(sci.shape[0]):
            contam = np.zeros((cny, cnx))
            for i in range(n_comp):
                if i == keep_index or not np.isfinite(amps[i, k]):
                    continue
                contam += amps[i, k] * _psf_image(cny, cnx, positions[i][0] - xlo,
                                                   positions[i][1] - ylo, fwhm_all[k],
                                                   deblend["psf_model"], deblend["beta"])
            sci[k, ylo:yhi, xlo:xhi] -= contam.astype(np.float32)

        hdul["SCI"].data = sci.astype(hdul["SCI"].data.dtype)
        hdul[0].header["DEBLEND"] = (keep_index, "PSF-deblended: kept this component index")
        hdul[0].header["DEBLENDN"] = (n_comp, "Number of PSF components fitted")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        hdul.writeto(out_path, overwrite=True)
    return out_path


# ===========================================================================
# Epoch bookkeeping and spectrum products
# ===========================================================================

def cube_epoch_info(cube_file):
    """Observation metadata for a Stage-3 cube: target, programme, MJD and a
    human date label. Used both to name per-epoch products and to stamp the
    saved spectra so an epoch can never be mistaken for another later on."""
    info = {"file": cube_file, "target": None, "program": None, "mjd": None,
            "mjd_beg": None, "mjd_end": None, "date": None, "instrument": None}
    with fits.open(cube_file) as hdul:
        hdr = hdul[0].header
        info["target"] = hdr.get("TARGNAME") or hdr.get("TARGPROP")
        info["program"] = hdr.get("PROGRAM")
        info["instrument"] = hdr.get("INSTRUME")
        for key in ("MJD-AVG", "MJD-BEG", "MJD-OBS", "EXPSTART"):
            if hdr.get(key) is not None:
                info["mjd"] = float(hdr[key])
                break
        for key, dst in (("MJD-BEG", "mjd_beg"), ("EXPSTART", "mjd_beg"),
                          ("MJD-END", "mjd_end"), ("EXPEND", "mjd_end")):
            if info[dst] is None and hdr.get(key) is not None:
                info[dst] = float(hdr[key])
    if info["mjd"] is not None:
        try:
            info["date"] = mjd_to_datestr(info["mjd"])
        except Exception:
            info["date"] = f"mjd{info['mjd']:.2f}"
    return info


def find_epoch_cubes(stage3_dir, pattern="*_s3d.fits", max_epoch_gap_days=1.0):
    """Discover the Stage-3 cubes under `stage3_dir` and sort them into
    epochs by MJD.

    Stage 3 already produces one cube per (target, epoch) -- `get_epoch_groups`
    enforces that when the association is built -- so normally each cube *is*
    an epoch. This re-checks it after the fact: cubes of the same target
    whose MJDs are within `max_epoch_gap_days` are flagged as combinable
    (they should have gone into one Stage-3 association), and anything
    further apart is kept strictly separate. That is what stops a spectrum
    taken ten days later from being averaged into the first one.

    Returns a list of epoch dicts, time-ordered, each with `key`, `target`,
    `mjd`, `date`, `cubes` and `label`."""
    files = sorted(glob.glob(os.path.join(stage3_dir, pattern)))
    if not files:
        return []

    infos = [cube_epoch_info(f) for f in files]
    by_target = defaultdict(list)
    for info in infos:
        by_target[info["target"] or "UNKNOWN_TARGET"].append(info)

    epochs = []
    for target, group in by_target.items():
        dated = sorted((i for i in group if i["mjd"] is not None), key=lambda i: i["mjd"])
        undated = [i for i in group if i["mjd"] is None]

        clusters, current, prev = [], [], None
        for info in dated:
            if prev is not None and (info["mjd"] - prev) > max_epoch_gap_days:
                clusters.append(current)
                current = []
            current.append(info)
            prev = info["mjd"]
        if current:
            clusters.append(current)
        clusters.extend([[i] for i in undated])

        for cluster in clusters:
            mjds = [i["mjd"] for i in cluster if i["mjd"] is not None]
            mjd = float(np.mean(mjds)) if mjds else None
            date = mjd_to_datestr(mjd) if mjd is not None else "unknown-date"
            # Target names routinely contain spaces ("SN 2024aecx"); the label
            # becomes a directory and filename, so make it path-safe.
            safe_target = re.sub(r"[^A-Za-z0-9._+-]+", "", str(target))
            epochs.append({
                "key": (target, date),
                "target": target,
                "mjd": mjd,
                "mjd_span": (min(mjds), max(mjds)) if mjds else (None, None),
                "date": date,
                "label": f"{safe_target}_{date}",
                "cubes": [i["file"] for i in cluster],
                "combinable": len(cluster) > 1,
            })

    epochs.sort(key=lambda e: (e["mjd"] is None, e["mjd"] if e["mjd"] is not None else 0.0))
    return epochs


def save_spectrum(wave, flux, err, out_base, meta=None, source_cube=None,
                   component=None, position=None, epoch=None, extra_header=None):
    """Write one spectrum as both a FITS binary table and an ECSV.

    The FITS file is the archival product: columns WAVELENGTH / FLUX /
    FLUX_ERROR with units attached, and a header carrying everything needed
    to identify the spectrum later without opening the pipeline -- target,
    programme, MJD and date of the epoch, the component index and its pixel
    position, the cube it came from, and the extraction settings. The ECSV
    is the same data in a form that reads straight into a plotting script or
    a text editor. Returns (fits_path, ecsv_path)."""
    from astropy.table import Table

    meta = meta or {}
    flux_unit = meta.get("flux_unit", "Jy")
    os.makedirs(os.path.dirname(os.path.abspath(out_base)) or ".", exist_ok=True)

    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    err = (np.asarray(err, dtype=np.float64) if err is not None
           else np.full_like(flux, np.nan))

    cards = {
        "TARGNAME": (epoch or {}).get("target"),
        "PROGRAM": (epoch or {}).get("program"),
        "MJD": (epoch or {}).get("mjd"),
        "DATEOBS": (epoch or {}).get("date"),
        "EPOCH": (epoch or {}).get("label"),
        "COMPNUM": component,
        "SRC_X": position[0] if position is not None else None,
        "SRC_Y": position[1] if position is not None else None,
        "SRCCUBE": os.path.basename(source_cube) if source_cube else None,
        "FLUXUNIT": flux_unit,
        "WAVEUNIT": meta.get("wave_unit", "um"),
        "EXTRTYPE": meta.get("extract_type", "Extract1dStep"),
        "RFCORR": meta.get("ifu_rfcorr"),
        "RSCALE": meta.get("ifu_rscale"),
        "COVSCALE": meta.get("ifu_covar_scale"),
        "NWAVE": int(wave.size),
        "NFINITE": int(np.isfinite(flux).sum()),
    }
    comments = {
        "MJD": "Mean MJD of the epoch this spectrum belongs to",
        "EPOCH": "Epoch label; spectra of different epochs are never combined",
        "COMPNUM": "Deblended component index (0 = primary/target)",
        "SRC_X": "Extraction x position, cube pixels (0-based)",
        "SRC_Y": "Extraction y position, cube pixels (0-based)",
        "SRCCUBE": "Cube the spectrum was extracted from",
        "NFINITE": "Number of finite flux samples",
    }

    if extra_header:
        for key, value in extra_header.items():
            # Callers may pass either a bare value or a (value, comment) pair.
            if isinstance(value, tuple) and len(value) == 2:
                cards[key] = value[0]
                comments[key] = value[1]
            else:
                cards[key] = value

    tab = Table({"WAVELENGTH": wave, "FLUX": flux, "FLUX_ERROR": err})
    tab["WAVELENGTH"].unit = meta.get("wave_unit", "um")
    tab["FLUX"].unit = flux_unit
    tab["FLUX_ERROR"].unit = flux_unit
    tab.meta.update({k: v for k, v in cards.items() if v is not None})

    fits_path = f"{out_base}.fits"
    ecsv_path = f"{out_base}.ecsv"

    hdu = fits.BinTableHDU(tab, name="EXTRACT1D")
    # tab.meta is carried for the ECSV, but astropy also folds it into the
    # FITS header, where a long SRCCUBE path warns about card truncation.
    # The cards are written explicitly just below, so clear them here first.
    for key in cards:
        hdu.header.pop(key, None)
    primary = fits.PrimaryHDU()
    for hdr in (primary.header, hdu.header):
        for key, value in cards.items():
            if value is None:
                continue
            # A long filename leaves no room for a comment; writing one only
            # produces a truncation warning, so drop it in that case.
            comment = comments.get(key, "")
            if isinstance(value, str) and len(value) > 50:
                comment = ""
            hdr[key] = (value, comment)
    primary.header["ORIGIN"] = ("miri_mrs_auto_pipeline", "Produced by this pipeline")
    fits.HDUList([primary, hdu]).writeto(fits_path, overwrite=True)
    tab.write(ecsv_path, format="ascii.ecsv", overwrite=True)
    return fits_path, ecsv_path


def save_component_spectra(components, out_dir, product_name, source_cube=None,
                            epoch=None, blend=None):
    """Save every deblended component with `save_spectrum`, plus a combined
    multi-extension FITS holding all components in one file and a CSV index
    listing what was written. Returns the list of written paths, and records
    them back onto each component dict under `spec_files`."""
    from astropy.table import Table

    os.makedirs(out_dir, exist_ok=True)
    written, rows = [], []

    for c in components:
        base = os.path.join(out_dir, f"{product_name}_comp{c['index']}_spec")
        extra = {}
        if blend is not None:
            if np.isfinite(blend.get("separation", np.nan)):
                extra["SEPPIX"] = (float(blend["separation"]),
                                   "Component separation, pixels")
            extra["NCOMP"] = (int(blend.get("n_components", len(components))),
                              "Number of components fitted")
        fpath, epath = save_spectrum(
            c["wave"], c["flux"], c["err"], base, meta=c.get("meta"),
            source_cube=c.get("cube_file") or source_cube,
            component=c["index"], position=c["position"], epoch=epoch,
            extra_header=extra)
        c["spec_files"] = {"fits": fpath, "ecsv": epath}
        written += [fpath, epath]
        rows.append({
            "component": c["index"],
            "x": c["position"][0],
            "y": c["position"][1],
            "epoch": (epoch or {}).get("label", ""),
            "mjd": (epoch or {}).get("mjd", np.nan),
            "median_flux": float(np.nanmedian(c["flux"])),
            "median_err": float(np.nanmedian(c["err"])) if c.get("err") is not None else np.nan,
            "n_finite": int(np.isfinite(c["flux"]).sum()),
            "fits": os.path.basename(fpath),
            "ecsv": os.path.basename(epath),
        })

    # One file holding every component, for convenience.
    combined = os.path.join(out_dir, f"{product_name}_allcomps_spec.fits")
    hdus = [fits.PrimaryHDU()]
    hdus[0].header["NCOMP"] = (len(components), "Number of components in this file")
    if epoch:
        for key, val in (("TARGNAME", epoch.get("target")), ("MJD", epoch.get("mjd")),
                          ("EPOCH", epoch.get("label"))):
            if val is not None:
                hdus[0].header[key] = val
    for c in components:
        tab = Table({"WAVELENGTH": np.asarray(c["wave"], float),
                     "FLUX": np.asarray(c["flux"], float),
                     "FLUX_ERROR": np.asarray(c["err"], float)})
        hdu = fits.BinTableHDU(tab, name=f"COMP{c['index']}")
        hdu.header["SRC_X"] = (c["position"][0], "Extraction x, cube pixels")
        hdu.header["SRC_Y"] = (c["position"][1], "Extraction y, cube pixels")
        hdu.header["FLUXUNIT"] = c.get("meta", {}).get("flux_unit", "Jy")
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(combined, overwrite=True)
    written.append(combined)

    index_path = os.path.join(out_dir, f"{product_name}_spectra_index.csv")
    Table(rows).write(index_path, format="ascii.csv", overwrite=True)
    written.append(index_path)

    print(f"\nSaved {len(components)} component spectrum/spectra to {out_dir}:")
    for r in rows:
        print(f"  comp {r['component']} @ ({r['x']:.2f}, {r['y']:.2f})  "
              f"median flux {r['median_flux']:.4g}  -> {r['fits']} / {r['ecsv']}")
    print(f"  all components:  {os.path.basename(combined)}")
    print(f"  index:           {os.path.basename(index_path)}")
    return written


def deblend_and_extract_all(cube_file, out_dir, product_name, x0, y0, n_sources=2,
                             fit_wave_range=(4.9, 7.7), positions=None,
                             psf_model="moffat", beta=2.5, cutout_half=9,
                             ifu_rfcorr=True, ifu_rscale=None, ifu_covar_scale=1.0,
                             save_cubes=True, save_plots=True):
    """Deblend a close pair (or group) and extract a clean spectrum for
    *every* component, so you can decide which one is your target by
    comparing the spectra.

    Run this on an already-background-subtracted cube (the
    `*_abi_bgsub_s3d.fits` output of `run_astrobkginterp_background` or
    `refine_and_reprocess`).

    Steps:
      1. `fit_blended_sources` on a channel-1 collapse to get positions, and
         to tell you via `delta_bic` whether more than one component is
         actually warranted.
      2. `psf_deblend_cube` to fit per-wavelength amplitudes with those
         positions held fixed.
      3. For each component, subtract the *other* components' PSF models from
         the cube and run the normal flux-calibrated `Extract1dStep` at that
         component's position.

    Nothing here decides which component is yours -- all of them are
    extracted and returned, in descending order of brightness in the fitting
    band. Compare the spectra (and, if you have them, positions from a higher
    resolution instrument) to make that call.

    Returns (blend_fit_dict, [component_dict, ...]) where each component dict
    has `index`, `position`, `wave`, `flux`, `err`, `x1d_file`, `cube_file`."""
    os.makedirs(out_dir, exist_ok=True)

    if positions is None:
        blend = fit_blended_sources(
            cube_file, x0, y0, n_sources=n_sources, wave_range=fit_wave_range,
            cutout_half=cutout_half, psf_model=psf_model, beta=beta,
            plot_file=(os.path.join(out_dir, f"{product_name}_blendfit.png")
                       if save_plots else None))
        positions = blend["positions"]
        fwhm_scale = blend["fwhm"] / blend["fwhm_expected"]
    else:
        positions = [(float(a), float(b)) for a, b in positions]
        blend = {"positions": positions, "delta_bic": np.nan,
                 "n_components": len(positions), "separation": np.nan}
        fwhm_scale = 1.0

    if len(positions) == 1:
        # Nothing to deblend: there is no companion to subtract, so skip the
        # amplitude fit and the 1.4 GB cube copy entirely and just extract
        # from the input cube at the fitted position.
        print("\nOnly one component -- no deblending needed; extracting directly.")
        px, py = positions[0]
        x1d_file = os.path.join(out_dir, f"{product_name}_comp0_x1d.fits")
        wave, flux, err, _, meta = extract_spectrum_manual(
            cube_file, out_file=x1d_file, x=px, y=py, ifu_rfcorr=ifu_rfcorr,
            ifu_rscale=ifu_rscale, ifu_covar_scale=ifu_covar_scale,
            return_meta=True)
        blend["deblend"] = None
        comp = {"index": 0, "position": (px, py), "wave": wave, "flux": flux,
                "err": err, "x1d_file": x1d_file, "cube_file": cube_file,
                "meta": meta}
        if save_plots:
            _plot_spectrum_qa(wave, flux, err, meta,
                              os.path.join(out_dir, f"{product_name}_comp0_spec.png"),
                              title=f"{product_name} (single source)")
        save_component_spectra([comp], out_dir, product_name, source_cube=cube_file,
                               epoch=cube_epoch_info(cube_file), blend=blend)
        return blend, [comp]

    print(f"\nFitting per-wavelength amplitudes for {len(positions)} component(s)...")
    deb = psf_deblend_cube(cube_file, positions, psf_model=psf_model, beta=beta,
                            fwhm_scale=fwhm_scale, cutout_half=cutout_half)
    blend["deblend"] = deb

    components = []
    for i, (px, py) in enumerate(positions):
        tag = f"{product_name}_comp{i}"
        print(f"\n--- component {i} at (x={px:.3f}, y={py:.3f}) ---")
        clean_cube = os.path.join(out_dir, f"{tag}_deblended_s3d.fits")
        write_companion_subtracted_cube(cube_file, deb, i, clean_cube)

        x1d_file = os.path.join(out_dir, f"{tag}_x1d.fits")
        wave, flux, err, _, meta = extract_spectrum_manual(
            clean_cube, out_file=x1d_file, x=px, y=py, ifu_rfcorr=ifu_rfcorr,
            ifu_rscale=ifu_rscale, ifu_covar_scale=ifu_covar_scale, return_meta=True)

        if not save_cubes:
            os.remove(clean_cube)
            clean_cube = None

        components.append({"index": i, "position": (px, py), "wave": wave,
                            "flux": flux, "err": err, "x1d_file": x1d_file,
                            "cube_file": clean_cube, "meta": meta,
                            "amplitude": deb["amplitudes"][i]})
        print(f"  -> {x1d_file}  (median flux {np.nanmedian(flux):.4g} {meta['flux_unit']})")

    if save_plots:
        _plot_component_spectra(components, blend,
                                 os.path.join(out_dir, f"{product_name}_components.png"),
                                 title=product_name)
    save_component_spectra(components, out_dir, product_name, source_cube=cube_file,
                           epoch=cube_epoch_info(cube_file), blend=blend)
    return blend, components


def _plot_component_spectra(components, blend, plot_file, title=""):
    """Overlay every deblended component's spectrum, plus their flux ratio --
    which is usually what tells the two apart (a supernova and a stellar
    companion or host knot have very different mid-IR slopes)."""
    try:
        import matplotlib
        # matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
        for c in components:
            g = np.isfinite(c["wave"]) & np.isfinite(c["flux"])
            ax[0].plot(c["wave"][g], c["flux"][g], lw=0.7,
                       label=f"comp {c['index']} @ ({c['position'][0]:.2f}, "
                             f"{c['position'][1]:.2f})")
        ax[0].axhline(0, color="grey", ls="--", lw=0.6)
        ax[0].set_ylabel(f"Flux ({components[0]['meta'].get('flux_unit', 'Jy')})")
        ax[0].legend(fontsize=9)
        allf = np.concatenate([c["flux"][np.isfinite(c["flux"])] for c in components])
        if allf.size:
            lo, hi = np.nanpercentile(allf, [1, 99.5])
            ax[0].set_ylim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))

        if len(components) >= 2:
            a, b = components[0], components[1]
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(np.abs(b["flux"]) > 0, a["flux"] / b["flux"], np.nan)
            ax[1].plot(a["wave"], ratio, lw=0.7, color="k")
            ax[1].set_ylabel("comp0 / comp1")
            ax[1].set_ylim(np.nanpercentile(ratio, 2), np.nanpercentile(ratio, 98))
        ax[1].set_xlabel("Wavelength (um)")

        sep = blend.get("separation", np.nan)
        dbic = blend.get("delta_bic", np.nan)
        fig.suptitle(f"{title}  |  separation {sep:.2f} px, delta_BIC {dbic:.1f}")
        plt.tight_layout()
        plt.savefig(plot_file, dpi=120)
        plt.close(fig)
        print(f"\n  Component comparison plot: {plot_file}")
    except Exception as exc:
        print(f"  (skipped component spectra plot: {exc})")


# ===========================================================================
# Main
# ===========================================================================

# ===========================================================================
# Multi-epoch driver
# ===========================================================================

def process_all_epochs(stage3_dir, out_root, x0=None, y0=None,
                        max_epoch_gap_days=1.0, n_sources=2,
                        pattern="*_s3d.fits", save_plots=True,
                        overlay_plot=True, **kwargs):
    """Run the full second pass -- refine centroid, re-fit background,
    deblend, extract -- once per epoch, keeping the epochs separate.

    Observations sharing a proposal ID are *not* averaged together just
    because they are the same programme: `find_epoch_cubes` clusters them by
    MJD, and only exposures within `max_epoch_gap_days` count as the same
    epoch. A spectrum taken ten days later gets its own directory, its own
    background fit and its own saved products, so a genuine light-curve /
    spectral-evolution comparison is possible.

    Each epoch's outputs land in `<out_root>/<target>_<date>/`. `kwargs` are
    passed straight through to `refine_and_reprocess` (fwhm_scale,
    min_aper_rad, bin_size, pool_size, ...).

    Returns a list of per-epoch result dicts."""
    epochs = find_epoch_cubes(stage3_dir, pattern=pattern,
                               max_epoch_gap_days=max_epoch_gap_days)
    if not epochs:
        raise SystemExit(f"No cubes matching {pattern!r} in {stage3_dir}")

    print(f"Found {len(epochs)} epoch(s) across {sum(len(e['cubes']) for e in epochs)} cube(s), "
          f"max_epoch_gap_days={max_epoch_gap_days}:")
    for e in epochs:
        span = e["mjd_span"]
        span_txt = (f"MJD {span[0]:.3f}-{span[1]:.3f}" if span[0] is not None else "no MJD")
        print(f"  {e['label']:<40s} {span_txt}  ({len(e['cubes'])} cube(s))")
        if e["combinable"]:
            print(f"    note: {len(e['cubes'])} cubes are within {max_epoch_gap_days} d and "
                  f"are treated as one epoch; combine them at Stage 3 for a single cube.")

    results = []
    for e in epochs:
        epoch_dir = os.path.join(out_root, e["label"])
        os.makedirs(epoch_dir, exist_ok=True)
        print(f"\n{'=' * 78}\nEPOCH {e['label']}  (MJD {e['mjd']})\n{'=' * 78}")

        for cube in e["cubes"]:
            product_name = os.path.basename(cube).replace("_s3d.fits", "")
            info = cube_epoch_info(cube)
            info["label"] = e["label"]
            try:
                centroid, bgsub_cube, _spec = refine_and_reprocess(
                    cube, epoch_dir, product_name, x0=x0, y0=y0,
                    save_plots=save_plots, **kwargs)
                xr, yr = centroid["x"], centroid["y"]

                blend, components = deblend_and_extract_all(
                    bgsub_cube, epoch_dir, product_name, xr, yr,
                    n_sources=n_sources, save_plots=save_plots)
                save_component_spectra(components, epoch_dir, product_name,
                                        source_cube=bgsub_cube, epoch=info, blend=blend)
                results.append({"epoch": e, "cube": cube, "centroid": centroid,
                                "bgsub_cube": bgsub_cube,
                                "blend": blend, "components": components,
                                "out_dir": epoch_dir, "error": None})
            except Exception as exc:
                print(f"  FAILED for {os.path.basename(cube)}: {exc}")
                results.append({"epoch": e, "cube": cube, "error": str(exc),
                                "out_dir": epoch_dir})

    ok = [r for r in results if r.get("error") is None]
    if overlay_plot and len(ok) > 1:
        _plot_epoch_comparison(ok, os.path.join(out_root, "epoch_comparison.png"))

    print(f"\n{len(ok)}/{len(results)} epoch products succeeded; outputs under {out_root}")
    return results


def _plot_epoch_comparison(results, plot_file):
    """Overlay the primary component's spectrum from every epoch, so
    spectral evolution between epochs is visible at a glance -- the point of
    keeping them separate in the first place."""
    try:
        import matplotlib
        # matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        for r in sorted(results, key=lambda r: r["epoch"]["mjd"] or 0.0):
            comps = r.get("components") or []
            if not comps:
                continue
            c = comps[0]
            g = np.isfinite(c["wave"]) & np.isfinite(c["flux"])
            ax.plot(c["wave"][g], c["flux"][g], lw=0.7,
                    label=f"{r['epoch']['date']} (MJD {r['epoch']['mjd']:.1f})")
        ax.axhline(0, color="grey", ls="--", lw=0.6)
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel("Flux (Jy)")
        ax.set_title("Primary component, epoch by epoch (never co-added)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_file, dpi=130)
        plt.close(fig)
        print(f"  epoch comparison plot -> {plot_file}")
    except Exception as exc:
        print(f"  (skipped epoch comparison plot: {exc})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proposal", required=True, help="JWST proposal/program number, e.g. 1523")
    parser.add_argument("--outdir", required=True,
                    help="Output directory for this target. Instrument and epoch subdirectories are created inside it: <outdir>/<INSTRUMENT>/<target>_<date>/{uncal,stage1,stage2,stage3,reextract,final}")
    parser.add_argument("--skip-download", action="store_true",
                         help="Reuse existing files in <outdir>/uncal/ instead of querying MAST")
    parser.add_argument("--from-stage3", action="store_true",
                         help="Skip download/Detector1/Spec2/Spec3 entirely and reuse whatever "
                              "'*_s3d.fits' cubes already exist in <outdir>/stage3/ -- for "
                              "iterating on background/extraction settings (this flag, --src-x/"
                              "--src-y/--ra/--dec, --bkg-mode, etc.) without re-running the "
                              "reduction. Overrides --skip-download (moot -- download isn't run).")
    parser.add_argument("--max-cores", default="half", choices=["none", "quarter", "half", "all"],
                         help="Cores to use for the jump step (default: half)")
    parser.add_argument("--log-file", default=None,
                         help="Also write the jwst/stpipe log to this file. Pairs well with "
                              "the default quiet console: clean terminal, full detail on "
                              "disk if a run needs debugging afterwards.")
    parser.add_argument("--verbose", action="store_true",
                         help="Show full jwst/stpipe/CRDS logging output (step parameter dumps, "
                              "CRDS reference-file lookups, etc.) instead of the default "
                              "WARNING-only console. tqdm bars stay on either way, but expect "
                              "them to interleave messily with the extra log lines.")
    
    parser.add_argument("--redo", action="store_true",
                        help="Re-run Detector1 even if the rate file already exists in <outdir>/stage1/")
    parser.add_argument("--dataset", nargs="+", default=None,
                         help="Restrict to specific dataset(s)/exposure(s) instead of the whole "
                              "proposal, e.g. --dataset jw04297010001_02103_00004. Accepts one or "
                              "more IDs; a file is kept if its name contains any of them.")

    # -- AstroBkgInterp background settings --
    parser.add_argument("--src-x", type=float, default=None,
                         help="Source x pixel position for background masking (and as the "
                              "AstroBkgInterp source-masking center). If omitted, each (target, "
                              "epoch) group auto-locates its own brightest source via "
                              "DAOStarFinder. Only meaningful for a single-group run -- see notes.")
    parser.add_argument("--src-y", type=float, default=None, help="Source y pixel position; see --src-x.")
    parser.add_argument("--ra", type=float, default=None,
                         help="Source RA (deg), as an alternative to --src-x/--src-y -- converted "
                              "to pixel coordinates via the cube's WCS (all_world2pix). Give "
                              "--ra/--dec or --src-x/--src-y, not both.")
    parser.add_argument("--dec", type=float, default=None, help="Source Dec (deg); see --ra.")
    parser.add_argument("--bkg-mode", choices=["simple", "polynomial"], default="polynomial",
                         help="AstroBkgInterp background-fitting mode -- these are the only two "
                              "AstroBkgInterp itself implements, there is no 'skip' mode "
                              "(default: polynomial)")
    parser.add_argument("--aper-rad", type=float, default=7,
                         help="Source-masking aperture radius in pixels. NOTE: AstroBkgInterp "
                              "ignores this for cubes -- it recomputes the radius per slice as "
                              "ceil(--fwhm-scale * FWHM(lambda)). Use --fwhm-scale/--min-aper-rad "
                              "instead (default: 7)")
    parser.add_argument("--ann-width", type=float, default=6,
                         help="Width in pixels of the annulus used to sample the background "
                              "under the masked source aperture (default: 6)")
    parser.add_argument("--fwhm-scale", type=float, default=2.5,
                         help="Scales the wavelength-dependent MRS PSF FWHM to set the "
                              "per-slice source-mask radius. This is the real lever on how much "
                              "of a bright target is protected from the background fit; too "
                              "small and the PSF wings are fitted as background and subtracted "
                              "off the source (default: 2.5)")
    parser.add_argument("--min-aper-rad", type=float, default=5.0,
                         help="Floor (pixels) on the per-slice source-mask radius, so the mask "
                              "does not shrink to a few pixels at the blue end where the PSF is "
                              "narrowest (default: 5)")
    parser.add_argument("--use-abi-err", action="store_true",
                         help="Propagate AstroBkgInterp's own background-error term. Off by "
                              "default because it is buggy (it uses the lstsq sum-of-squared-"
                              "residuals as if it were fit coefficients), inflating the cube ERR "
                              "by ~5 orders of magnitude. Leave off unless reproducing old runs.")
    parser.add_argument("--poly-k", type=int, default=3,
                         help="Degree of the 2D polynomial fit when --bkg-mode polynomial (default: 3)")
    parser.add_argument("--bin-size", type=int, default=9,
                         help="Sliding-window size (pixels) for the polynomial background fit "
                              "when --bkg-mode polynomial (default: 9)")
    parser.add_argument("--cube-resolution", choices=["low", "medium", "high"], default="high",
                         help="Sampling density for the polynomial background fit (default: high)")
    parser.add_argument("--pool-size", type=int, default=None,
                         help="Multiprocessing pool size for AstroBkgInterp -- this is the main "
                              "lever for AstroBkgInterp's runtime, which is otherwise single-"
                              "threaded per cube. Default: (cpu count - 1), i.e. multiprocessing "
                              "is on by default now; pass 1 to force single-threaded.")

    parser.add_argument("--max-epoch-gap-days", type=float, default=1.0,
                         help="Exposures more than this many days apart (by MJD) are never combined "
                              "into the same Stage-3 cube (default: 1.0)")
    parser.add_argument("--tag", default=None,
                         help="Optional string included in the output directory name and in every "
                              "Stage-3/background-subtracted filename. Omitted by default.")
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
    tag_suffix = f"_{args.tag}" if args.tag else ""
    outdir = os.path.abspath(args.outdir)   # absolute: association files need it

    # Staging area for the raw downloads. Epochs are not known until the
    # headers have been read, so everything lands here first and is then
    # distributed into per-epoch trees.
    download_dir = os.path.join(outdir, "_downloaded_uncal")
    os.makedirs(download_dir, exist_ok=True)

    print(f"jwst version: {jwst.__version__}  |  output directory: {outdir}")

    t0 = time.perf_counter()

    # to_process entries are (product_name, cube_file, epoch_paths), where
    # epoch_paths is the layout dict for that epoch (or None when running
    # --from-stage3 over a tree we did not build).
    if args.from_stage3:
        to_process = collect_existing_cubes(outdir, args.tag)
        if not to_process:
            raise SystemExit(
                f"--from-stage3: no existing '*_s3d.fits' cubes found under {outdir}. "
                f"Expected them in <outdir>/MIRI-MRS/<target>_<date>/stage3/.")
        print(f"--from-stage3: reusing {len(to_process)} existing cube(s) "
              f"(download/Detector1/Spec2/Spec3 skipped)")
    else:
        if args.skip_download:
            uncal_files = find_staged_uncal(outdir, download_dir)
            if args.dataset:
                uncal_files = [f for f in uncal_files
                               if any(d in os.path.basename(f) for d in args.dataset)]
            print(f"--skip-download: using {len(uncal_files)} existing uncal file(s).")
        else:
            uncal_files = query_and_download(proposal_padded, proposal_num,
                                             download_dir, dataset_ids=args.dataset)
        if not uncal_files:
            raise SystemExit("No uncalibrated files to process.")

        check_mrs_coverage(uncal_files)

        # Epochs are decided here, on the uncal headers, *before* any
        # calibration runs. Doing it up front (rather than at Stage 3, as
        # this script used to) means each epoch gets its own self-contained
        # directory tree and exposures from different visits can never share
        # a Stage-3 association by accident.
        epoch_groups = get_epoch_groups(uncal_files, args.max_epoch_gap_days)
        n_targets = len(set(k[0] for k in epoch_groups))
        print(f"{len(epoch_groups)} (target, epoch) group(s) across {n_targets} target(s) "
              f"(>{args.max_epoch_gap_days} day gap = new epoch)\n")

        to_process = []
        for (targname, epoch), group_uncal in epoch_groups.items():
            tag = args.tag or targname
            paths = ensure_epoch_tree(outdir, "MIRI-MRS", tag, epoch)
            product_name = paths["stem"]
            print(f"=== {product_name}: {len(group_uncal)} exposure(s) -> "
                  f"{os.path.relpath(paths['epoch'], outdir)}")

            staged = stage_uncal(group_uncal, paths["uncal"])

            rate_files = run_detector1(staged, paths["stage1"],
                                       args.max_cores, redo=args.redo)

            # BKGDTARG classification is no longer used to drive a dedicated
            # master_background path -- AstroBkgInterp masks the source
            # directly on the science cube. Dedicated background exposures
            # (if any) are simply not processed further.
            sci_rate = filter_list(rate_files, "BKGDTARG", False)

            cal_files = run_spec2(sci_rate, paths["asn"], paths["stage2"])
            if not cal_files:
                print(f"  WARNING: no Stage-2 products for {product_name} -- skipping.")
                continue

            cube_file = run_spec3_group(cal_files, paths["asn"],
                                        paths["stage3"], product_name)
            if cube_file is not None:
                to_process.append((product_name, cube_file, paths))
                record_stage3_products(outdir, paths, tag, epoch)

    if (have_xy or have_radec) and len(to_process) > 1:
        print("  NOTE: an explicit source position was given but there is more than one "
              "(target, epoch) group in this run -- the same position will be applied to all "
              "of them, which is probably not what you want. Consider narrowing the run (e.g. "
              "--tag, or a proposal with a single target/epoch) when overriding the source "
              "position.\n")

    bgsub_outputs = {}
    for product_name, cube_file, paths in tqdm(to_process, desc="AstroBkgInterp", unit="cube"):
        bgsub_dir = paths["reextract"] if paths else os.path.join(outdir, "bgsub")
        os.makedirs(bgsub_dir, exist_ok=True)
        if have_xy or have_radec:
            src_x, src_y = resolve_source_pixel(cube_file, ra=args.ra, dec=args.dec,
                                                 x=args.src_x, y=args.src_y)
        else:
            located = locate_source_pixel(cube_file)
            if located is None:
                tqdm.write(f"  WARNING: auto-locate found no source for {product_name} -- "
                           f"skipping (re-run with --src-x/--src-y or --ra/--dec for this target).")
                continue
            src_x, src_y = located
        tqdm.write(f"  {product_name}: source at (x={src_x:.2f}, y={src_y:.2f})")

        out_path = run_astrobkginterp_background(
            cube_file, src_x, src_y, bgsub_dir, product_name,
            bkg_mode=args.bkg_mode, aper_rad=args.aper_rad, ann_width=args.ann_width,
            fwhm_scale=args.fwhm_scale, poly_k=args.poly_k, bin_size=args.bin_size,
            min_aper_rad=args.min_aper_rad, use_abi_err=args.use_abi_err,
            cube_resolution=args.cube_resolution, pool_size=args.pool_size,
        )
        bgsub_outputs[product_name] = out_path

    t_end = time.perf_counter()
    print(f"\nDone in {(t_end - t0) / 60:.1f} min. Background-subtracted cubes:")
    for key, path in bgsub_outputs.items():
        print(f"  {key}: {os.path.relpath(path, outdir)}")

    print(f"\nProduct index: {os.path.join(outdir, MANIFEST_NAME)}")
    print("\nNo 1D spectrum was extracted automatically. Either run the "
          "AstroBkgInterp re-extraction, which writes the final combined "
          "spectrum and records it in the product index:\n"
          f"    python -m jspec.pipelines.abi_reextract_pfpc --outdir {outdir}\n"
          "or extract a single cube by hand:\n"
          "    from jspec.pipelines.miri_mrs_auto_pipeline import extract_spectrum_manual\n"
          "    wave, flux, err, (x, y) = extract_spectrum_manual(\n"
          "        '<bgsub_cube>.fits', x=34.2, y=51.7)          # pixel coords, or:\n"
          "    wave, flux, err, (x, y) = extract_spectrum_manual(\n"
          "        '<bgsub_cube>.fits', ra=166.45643, dec=-0.0348)  # RA/Dec (deg)")


if __name__ == "__main__":
    main()