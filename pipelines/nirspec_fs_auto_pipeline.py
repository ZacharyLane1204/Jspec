#!/usr/bin/env python3
"""
nirspec_fs_auto_pipeline.py
===========================

The STScI NIRSpec FS notebook (nirspec_x1d_process.ipynb) as a script, run
for a whole proposal. Same steps, same pipeline settings; the only additions
are (1) automatic download, and (2) EPOCH awareness.

  1. Download every NIRSpec/SLIT _uncal file of the proposal from MAST.
  2. Stage 1  Detector1Pipeline   uncal -> rate
  3. Stage 2  Spec2Pipeline       rate  -> cal / s2d / x1d, per exposure,
                                  with nod-based background subtraction
  4. Stage 3  Spec3Pipeline       cal   -> s2d / x1d, ONE product per
                                  (target, epoch, slit, grating, filter)
  5. Final    stitch the Stage-3 x1d files of the different gratings into one
              spectrum per (target, epoch, slit).

Epochs
------
Exposures of one target are sorted by start time; a gap larger than
--max-epoch-gap-days (default 1) starts a new epoch. Epochs are NEVER mixed:
  * nod partners used as background in Stage 2 must be from the same epoch
  * every Stage-3 association contains a single epoch
  * every final spectrum is a single epoch (the date is in the file name)
Run with --plan-only first to see how your exposures are grouped.

Final spectra are the pipeline's own Stage-3 x1d spectra (6-pixel box at the
source position, Jy) joined across gratings. To move the extraction box, use
nirspec_refine_extract.py.

Usage
-----
    python nirspec_fs_auto_pipeline.py --proposal 9258 --plan-only
    python nirspec_fs_auto_pipeline.py --proposal 9258 --target 2024aecx --tag sn2024aecx
    python nirspec_fs_auto_pipeline.py --proposal 9258 --skip-download --from-stage 2
    python nirspec_fs_auto_pipeline.py --proposal 9258 --skip-download --from-stage 4   # rebuild final spectra only
    python nirspec_fs_auto_pipeline.py --proposal 9258 --skip-download --from-stage 2 --source-type POINT
    python nirspec_fs_auto_pipeline.py --proposal 9258 --list-mast          # what does MAST hold?
    python nirspec_fs_auto_pipeline.py --proposal 9258 --tag sn2024aecx --require-slit S400A1
    python nirspec_fs_auto_pipeline.py --proposal 9258 --tag sn2024aecx --slit S400A1   # only that slit

Downloads are restart-safe (complete files are skipped, truncated ones are
retried) and every uncal file MAST lists for the proposal is fetched -- no size
cut. Data still under exclusive access needs `export MAST_API_TOKEN=<token>`.
Stage 1 skips exposures whose rate file already exists (--redo to force).

Stages can be restarted with --from-stage {1,2,3,4}; 4 only rebuilds the final
spectra from existing Stage-3 products. Keep nirspec_fs_refining.py alongside.
"""

import argparse
import glob
import os
import time
import warnings
from collections import defaultdict

import numpy as np
from astropy.io import fits

warnings.simplefilter("ignore", RuntimeWarning)

# CRDS setup must happen before importing jwst
if os.getenv("CRDS_PATH") is None:
    os.environ["CRDS_PATH"] = os.path.join(os.path.expanduser("~"), "crds_cache")
if os.getenv("CRDS_SERVER_URL") is None:
    os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

import jwst                                                                # noqa: E402
from astroquery.mast import Observations                                   # noqa: E402
from jwst.pipeline import Detector1Pipeline, Spec2Pipeline, Spec3Pipeline  # noqa: E402
from jwst.associations import asn_from_list as afl                         # noqa: E402
from jwst.associations.lib.rules_level2_base import DMSLevel2bBase         # noqa: E402
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base        # noqa: E402

from jspec.layout import (                                                 # noqa: E402
    ensure_epoch_tree, find_staged_uncal, stage_files)
from jspec.manifest import MANIFEST_NAME, record_product                   # noqa: E402

from jspec.pipelines.nirspec_fs_refining import (                                          # noqa: E402
    FS_EXP_TYPE, read_meta, assign_epochs, sanitize, mjd_to_datestr,
    read_x1d, stitch, save_spectrum, plot_spectrum,
)


# ===========================================================================
# Download
# ===========================================================================

MAST_CHUNK = 25          # observations per get_product_list call
MAST_RETRIES = 3


def mast_login():
    """Optional: set MAST_API_TOKEN (https://auth.mast.stsci.edu/token) to fetch
    data still under exclusive access (common for DDT / recent programs)."""
    token = os.getenv("MAST_API_TOKEN")
    if token:
        try:
            Observations.login(token=token)
            print("  Logged in to MAST with MAST_API_TOKEN.")
        except Exception as exc:
            print(f"  WARNING: MAST login failed ({exc}); exclusive-access data will not download.")


def list_mast_uncal(proposal_padded, proposal_num, manifest_path=None):
    """Every level-1 *_uncal.fits of the proposal as a list of dicts (name, uri,
    size, rights, obs_id). NO size filtering: the old 'larger than the mean
    product size' cut could silently drop smaller uncal files (short ramps,
    last segments of a segmented exposure, or whole configurations), because
    uncal files are 16-bit and the mean is dominated by float32 cal/rateints
    products. Uncal files are already selected by name, so no cut is needed."""
    print(f"Querying MAST for NIRSpec/SLIT observations in proposal {proposal_padded} ...")
    obs_table = Observations.query_criteria(
        obs_collection="JWST", instrument_name=["NIRSPEC/SLIT"], provenance_name=["CALJWST"],
        proposal_id=[str(proposal_num)])
    print(f"  Found {len(obs_table)} observation entries (provenance CALJWST).")

    # Fallback query WITHOUT the provenance filter. An exposure whose calibrated
    # (level-2) product MAST does not carry -- e.g. because STScI's own pipeline
    # run failed on it -- can still have its uncal file in the archive but no
    # CALJWST observation entry, so the query above never sees it. Still limited
    # to NIRSPEC/SLIT so no MSA / IFU data is pulled in.
    try:
        obs_any = Observations.query_criteria(
            obs_collection="JWST", instrument_name=["NIRSPEC/SLIT"], proposal_id=[str(proposal_num)])
        have_ids = {str(x) for x in obs_table["obsid"]}
        extra_rows = [i for i, x in enumerate(obs_any["obsid"]) if str(x) not in have_ids]
        if extra_rows:
            print(f"  Fallback query (no provenance filter) found {len(extra_rows)} additional "
                  f"observation entr{'y' if len(extra_rows) == 1 else 'ies'} -- adding them.")
            from astropy.table import vstack
            obs_table = vstack([obs_table, obs_any[extra_rows]], metadata_conflicts="silent")
    except Exception as exc:
        print(f"  (fallback query skipped: {exc})")

    found = {}
    for i in range(0, len(obs_table), MAST_CHUNK):
        chunk = obs_table[i:i + MAST_CHUNK]
        try:
            products = Observations.get_product_list(chunk)
        except Exception as exc:
            print(f"  WARNING: product list for observations {i}-{i + len(chunk) - 1} failed ({exc}); "
                  f"retrying one by one.")
            products = []
            for row in chunk:
                try:
                    products.extend(list(Observations.get_product_list(row)))
                except Exception as exc2:
                    print(f"  WARNING: could not list products for {row['obs_id']}: {exc2}")
        cols = set(getattr(products, "colnames", []) or (products[0].colnames if len(products) else []))
        for p in products:
            uri = str(p["dataURI"])
            name = str(p["productFilename"]) if "productFilename" in cols else os.path.basename(uri)
            is_uncal = name.endswith("_uncal.fits") or (
                "productSubGroupDescription" in cols
                and str(p["productSubGroupDescription"]).upper() == "UNCAL" and name.endswith(".fits"))
            if not is_uncal or f"jw{proposal_padded}" not in name:
                continue
            try:
                size = int(p["size"])
            except Exception:
                size = 0
            found[name] = dict(name=name, uri=uri, size=size,
                               rights=str(p["dataRights"]) if "dataRights" in cols else "?",
                               obs_id=str(p["obs_id"]) if "obs_id" in cols else "?")
    entries = [found[k] for k in sorted(found)]

    # observation-level summary: which observations contributed how many uncal files
    per_obs = defaultdict(int)
    for e in entries:
        per_obs[e["obs_id"]] += 1
    print(f"  {len(entries)} uncal exposure file(s) in {len(per_obs)} observation(s):")
    for row in obs_table:
        oid = str(row["obs_id"])
        extra = "  ".join(f"{c}={row[c]}" for c in ("target_name", "filters") if c in obs_table.colnames)
        n_u = per_obs.get(oid, 0)
        print(f"    {oid}: {n_u} uncal   {extra}" + ("   <-- NO uncal file listed" if n_u == 0 else ""))
    n_prop = sum(1 for e in entries if e["rights"].upper() not in ("PUBLIC", "?"))
    if n_prop:
        print(f"  NOTE: {n_prop} file(s) are under exclusive access -- downloading them needs "
              f"MAST_API_TOKEN (and rights to the program).")

    if manifest_path:
        with open(manifest_path, "w") as fh:
            fh.write("# uncal files MAST lists for this proposal\n# name  size_MB  rights  obs_id\n")
            for e in entries:
                fh.write(f"{e['name']}  {e['size'] / 1e6:.1f}  {e['rights']}  {e['obs_id']}\n")
        print(f"  manifest: {manifest_path}")
    return entries


def _complete(path, expected_size):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    if expected_size and os.path.getsize(path) < 0.999 * expected_size:
        return False                                   # truncated download
    try:
        fits.getheader(path, 0)
    except Exception:
        return False
    return True


def query_and_download(proposal_padded, proposal_num, uncal_dir, manifest_path=None):
    mast_login()
    entries = list_mast_uncal(proposal_padded, proposal_num, manifest_path)
    failed = []
    for i, e in enumerate(entries, 1):
        dest = os.path.join(uncal_dir, e["name"])
        if _complete(dest, e["size"]):
            continue                                   # already have it (restart-safe)
        for attempt in range(1, MAST_RETRIES + 1):
            print(f"  [{i}/{len(entries)}] downloading {e['name']} ({e['size'] / 1e6:.0f} MB)"
                  + (f"  attempt {attempt}" if attempt > 1 else ""))
            status = None
            try:
                res = Observations.download_file(e["uri"], local_path=uncal_dir)
                status = res[0] if isinstance(res, (tuple, list)) else res
            except Exception as exc:
                status = f"exception: {exc}"
            if _complete(dest, e["size"]):
                break
            print(f"    not complete ({status})")
            if os.path.exists(dest):
                os.remove(dest)                        # do not keep a truncated file
            time.sleep(3 * attempt)
        else:
            failed.append(e)

    have = sorted(glob.glob(os.path.join(uncal_dir, "*uncal.fits")))
    print(f"  {len(entries) - len(failed)}/{len(entries)} uncal files present in {uncal_dir}")
    if failed:
        print("  !! FAILED downloads (re-run to retry; rights problems need MAST_API_TOKEN):")
        for e in failed:
            print(f"     {e['name']}  rights={e['rights']}")
    return have


# ===========================================================================
# Exposure bookkeeping (classification, epochs)
# ===========================================================================

FAILURES = []            # (stage, file name, slit, message) -- printed at the end


def print_inventory(files, title):
    """What is physically in a directory, straight from the headers, per slit.
    This is the first place to look when a slit 'never shows up': it prints
    every EXP_TYPE / slit / target combination, including the ones the later
    steps ignore, so you can see whether the S400A1 file is on disk at all."""
    print(f"\n{title}: {len(files)} file(s)")
    if not files:
        return
    tally = defaultdict(list)
    for f in files:
        try:
            m = read_meta(f)
        except Exception as exc:
            print(f"  !! cannot read header of {os.path.basename(f)}: {exc}")
            continue
        tally[(str(m["exp_type"]), str(m["slit"]), m["target"], m["is_bkg"])].append(m)
    for (et, slit, target, isb), ms in sorted(tally.items()):
        cfgs = sorted({f"{m['filter']}/{m['grating']}/{m['detector']}" for m in ms})
        print(f"  {et:15s} slit={slit:8s} target={target!r:28s} bkgd={str(isb):5s} "
              f"{len(ms):3d} file(s)  {', '.join(cfgs)}")
    print()

def exclude_edge_nods(metas):
    """S1600A1 5-point nod: drop the positions the notebook drops (same rule
    as the notebook's 'excluded_files' cell)."""
    return [m for m in metas
            if not (m["slit"] == "S1600A1" and m["pridtpts"] == 5 and m["nod"] in (1, 5))]


def collect_exposures(files, target_filter, gap_days, slits=None):
    """Read headers, keep fixed-slit exposures only, split science /
    background, assign epochs. Backgrounds (dedicated-background exposures,
    BKGDTARG=T) inherit the epoch of the nearest science exposure if it is
    within the gap; otherwise they are dropped.

    `slits` (e.g. ["S400A1"]) restricts the result to those slits. It is
    applied AFTER the epochs are assigned, so epoch numbers / dates are the
    same as in a full run.

    Returns (sci, bkg, notes); `notes` are coverage warnings (see
    check_coverage), computed on ALL slits before any --slit restriction."""
    metas = [read_meta(f) for f in files]
    fs = [m for m in metas if m["exp_type"] == FS_EXP_TYPE]
    if len(fs) < len(metas):
        odd = defaultdict(int)
        for m in metas:
            if m["exp_type"] != FS_EXP_TYPE:
                odd[(str(m["exp_type"]), str(m["slit"]))] += 1
        print(f"  Ignoring {len(metas) - len(fs)} non-fixed-slit exposure(s): "
              + ", ".join(f"{n}x {et} (slit {sl})" for (et, sl), n in sorted(odd.items())))
    sci = [m for m in fs if not m["is_bkg"]]
    bkg = [m for m in fs if m["is_bkg"]]
    if bkg:
        print(f"  {len(bkg)} exposure(s) flagged BKGDTARG=T (dedicated background), slit(s): "
              f"{sorted({str(m['slit']) for m in bkg})}")
    if target_filter:
        # Match on names with spaces/punctuation removed, in both directions, so that
        # "SN2024aecx", "SN 2024aecx" and "2024aecx" all select the same target
        # (the same SN is called 'SN 2024aecx' in one visit and '2024aecx' in another).
        t = sanitize(target_filter)

        def _match(m):
            for name in (sanitize(m["target"]), sanitize(m["targprop"])):
                if name and (t in name or (len(name) >= 5 and name in t)):
                    return True
            return False
        keep = [m for m in sci if _match(m)]
        dropped = [m for m in sci if m not in keep]
        if dropped:
            # a slit whose TARGNAME differs from the others is the classic way an
            # exposure silently disappears when --target is used
            why = defaultdict(int)
            for m in dropped:
                why[(str(m["slit"]), m["target"], m["targprop"])] += 1
            print(f"  --target {target_filter!r} DROPPED {len(dropped)} science exposure(s):")
            for (sl, tn, tp), n in sorted(why.items()):
                print(f"      slit={sl}  TARGNAME={tn!r}  TARGPROP={tp!r}  x{n}")
        sci = keep
    n_edge = len(sci) + len(bkg)
    sci, bkg = exclude_edge_nods(sci), exclude_edge_nods(bkg)
    if n_edge != len(sci) + len(bkg):
        print(f"  Dropped {n_edge - len(sci) - len(bkg)} S1600A1 5-point-nod edge exposure(s) (positions 1 and 5).")

    assign_epochs(sci, gap_days)
    kept_bkg = []
    for b in bkg:
        if b["mjd_start"] is None or not sci:
            continue
        nearest = min((s for s in sci if s["mjd_start"] is not None),
                      key=lambda s: abs(s["mjd_start"] - b["mjd_start"]), default=None)
        if nearest is not None and abs(nearest["mjd_start"] - b["mjd_start"]) <= gap_days:
            b["epoch_key"] = nearest["epoch_key"]
            kept_bkg.append(b)

    notes = check_coverage(sci)
    if slits:
        want = {s.upper() for s in slits}
        sci = [m for m in sci if str(m["slit"]).upper() in want]
        kept_bkg = [m for m in kept_bkg if str(m["slit"]).upper() in want]
    return sci, kept_bkg, notes


def check_coverage(sci):
    """Things that usually mean 'a download is missing', found from the headers:
      * an epoch that lacks a slit other epochs have (e.g. S400A1 in epoch 1 only)
      * an incomplete nod set (fewer nod positions than the header's PRIDTPTS)
        -- with a missing partner Stage 2 silently subtracts less background.
    Returns a list of message strings."""
    notes = []
    slits_by_epoch = defaultdict(set)
    for m in sci:
        slits_by_epoch[m["epoch_key"]].add(m["slit"])
    all_slits = set().union(*slits_by_epoch.values()) if slits_by_epoch else set()
    date_of = {m["epoch_key"]: m["epoch_date"] for m in sci}
    for ek in sorted(slits_by_epoch, key=str):
        for s in sorted(all_slits - slits_by_epoch[ek], key=str):
            notes.append(f"epoch {ek[1]} ({date_of[ek]}) of {ek[0]} has NO {s} exposures, "
                         f"but another epoch does")

    cfg = defaultdict(list)
    for m in sci:
        cfg[(m["epoch_key"], m["filter"], m["grating"], m["slit"], m["detector"])].append(m)
    for (ek, filt, grat, slit, det), ms in sorted(cfg.items(), key=lambda kv: str(kv[0])):
        m0 = ms[0]
        if "NOD" not in m0["patttype"].split("-") or not m0["pridtpts"] or slit == "S1600A1":
            continue
        nods = sorted({m["nod"] for m in ms})
        if len(nods) < int(m0["pridtpts"]):
            notes.append(f"epoch {ek[1]} ({m0['epoch_date']}) {filt}/{grat} {slit} {det}: "
                         f"{len(nods)} of {m0['pridtpts']} nod positions present (have {nods}) "
                         f"-> incomplete nod set, background subtraction will be degraded")
    return notes


def print_notes(notes):
    if notes:
        print("\nCOVERAGE WARNINGS:")
        for n in notes:
            print(f"  ! {n}")
        print()


def require_slits_or_exit(sci, required):
    """Exit if any epoch of the selected target(s) lacks one of `required` slits."""
    if not required:
        return
    need = {r.upper() for r in required}
    have = defaultdict(set)
    for m in sci:
        have[m["epoch_key"]].add(str(m["slit"]).upper())
    date_of = {m["epoch_key"]: m["epoch_date"] for m in sci}
    bad = [f"epoch {ek[1]} ({date_of[ek]}): missing {', '.join(sorted(need - s))}"
           for ek, s in sorted(have.items(), key=lambda kv: str(kv[0])) if need - s]
    if bad:
        raise SystemExit("Required slit(s) not found -- " + "; ".join(bad) +
                         ".\nCheck <outdir>/mast_uncal_manifest.txt: if the file is listed there but not on "
                         "disk, re-run (the download is restart-safe); if it is exclusive-access, set "
                         "MAST_API_TOKEN; if MAST does not list it at all, it is not in the archive yet.")


def format_plan(sci, gap_days):
    lines = [f"Epoch plan (a gap > {gap_days:g} d between exposures starts a new epoch)"]
    by_epoch = defaultdict(list)
    for m in sci:
        by_epoch[m["epoch_key"]].append(m)
    for (target, epoch), members in sorted(by_epoch.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        starts = [m["mjd_start"] for m in members if m["mjd_start"] is not None]
        ends = [m["mjd_end"] for m in members if m["mjd_end"] is not None]
        span = f"MJD {min(starts):.3f}-{max(ends):.3f}" if starts else "MJD unknown"
        d = members[0]["epoch_date"]
        lines.append(f"\n  {target}  epoch {epoch}  ({d[:4]}-{d[4:6]}-{d[6:]}, {span})")
        cfgs = defaultdict(list)
        for m in members:
            cfgs[(m["filter"], m["grating"], m["slit"])].append(m)
        for (filt, grat, slit), ms in sorted(cfgs.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
            dets = sorted({m["detector"] for m in ms})
            nods = sorted({m["nod"] for m in ms})
            lines.append(f"      {filt}/{grat}  {slit}: {len(ms)} file(s), nod position(s) {nods}, {', '.join(dets)}")
    return "\n".join(lines)


# ===========================================================================
# Stage 1
# ===========================================================================

def _slit_of(path):
    try:
        return str(fits.getval(path, "FXD_SLIT", 0))
    except Exception:
        return "?"


def run_detector1(uncal_files, det1_dir, max_cores, redo=False):
    os.makedirs(det1_dir, exist_ok=True)
    steps = defaultdict(dict)
    steps["linearity"]["skip"] = False
    steps["ramp_fit"]["suppress_one_group"] = False
    steps["jump"]["maximum_cores"] = max_cores
    steps["jump"]["expand_large_events"] = True
    steps["picture_frame"]["skip"] = False
    for f in uncal_files:
        rate = os.path.join(det1_dir, os.path.basename(f).replace("_uncal.fits", "_rate.fits"))
        if os.path.exists(rate) and not redo:
            try:
                fits.getheader(rate, 0)
                fits.getdata(rate, "SCI")               # a truncated file fails here
                print(f"[Stage 1] {os.path.basename(f)}: rate file exists, skipping (--redo to force)")
                continue
            except Exception:
                print(f"[Stage 1] {os.path.basename(f)}: existing rate file is unreadable -> re-running")
        print(f"[Stage 1] {os.path.basename(f)}")
        try:
            Detector1Pipeline.call(f, save_results=True, steps=steps, output_dir=det1_dir)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            FAILURES.append(("Stage 1", os.path.basename(f), _slit_of(f), str(exc)))
        if not os.path.exists(rate):
            print(f"  !! no rate file was produced for {os.path.basename(f)}")
            FAILURES.append(("Stage 1", os.path.basename(f), _slit_of(f), "no rate file produced"))


# ===========================================================================
# Stage 2
# ===========================================================================

def _same_setup(a, b):
    """Same detector / filter / grating / slit AND same target+epoch."""
    return (all(a[k] == b[k] for k in ("detector", "filter", "grating", "slit"))
            and a["epoch_key"] == b["epoch_key"])


def write_l2_asn(m, sci, bkg, asnfile):
    """Level-2 association for one science exposure, as in the notebook:
    for nodded data the other nod positions are the background; otherwise use
    dedicated background exposures. Only exposures of the same setup and the
    same epoch are eligible."""
    asn = afl.asn_from_list([m["path"]], rule=DMSLevel2bBase, product_name="Level2")
    asn.data["program"] = m["program"]
    members = asn["products"][0]["members"]

    use_sci = [s for s in sci if _same_setup(m, s)]
    if "NOD" in m["patttype"].split("-"):
        used = use_sci
        for s in use_sci:
            if s["nod"] != m["nod"]:
                members.append({"expname": s["path"], "exptype": "background"})
    else:
        use_bkg = [b for b in bkg if _same_setup(m, b)]
        used = use_sci + use_bkg
        for b in use_bkg:
            members.append({"expname": b["path"], "exptype": "background"})
    for s in used:
        members.append({"expname": s["path"], "exptype": "selfcal"})

    _, serialized = asn.dump()
    with open(asnfile, "w") as fh:
        fh.write(serialized)
    return sum(1 for mem in members if mem["exptype"] == "background")


def run_spec2(sci, bkg, asn_dir, spec2_dir, skip_pixel_bg, source_type=None):
    os.makedirs(asn_dir, exist_ok=True)
    os.makedirs(spec2_dir, exist_ok=True)
    steps = defaultdict(dict)
    steps["bkg_subtract"]["skip"] = skip_pixel_bg      # False -> nod background is subtracted
    if source_type:
        # Force the source type of the primary slit instead of trusting SRCTYAPT / the
        # step's guess (a supernova is a POINT source). Non-primary slits stay EXTENDED.
        steps["srctype"]["source_type"] = source_type
    for m in sci:
        asnfile = os.path.join(asn_dir, m["name"].replace("_rate.fits", "_l2asn.json"))
        try:
            n_bkg = write_l2_asn(m, sci, bkg, asnfile)
            print(f"[Stage 2] {m['name']}  ({m['slit']}, nod {m['nod']}, {n_bkg} background exposure(s))")
            if n_bkg == 0 and not skip_pixel_bg:
                print("    WARNING: no nod partner / background exposure of the same setup and epoch "
                      "-> NO background is subtracted for this exposure")
            Spec2Pipeline.call(asnfile, save_results=True, steps=steps, output_dir=spec2_dir)
        except Exception as exc:
            # e.g. no slit lands on NRS2 for this exposure
            if "No open slits fall on detector" in str(exc):
                # the slit's spectrum simply does not land on this detector (e.g. NRS2 for
                # S200A1/S400A1 with G235M/G395M): expected, not a failure
                print(f"  skipped {m['name']}: slit {m['slit']} does not fall on {m['detector']} (expected)")
                open(cal_path(m, spec2_dir) + ".noslit", "w").close()
                continue
            print(f"  skipped {m['name']}: {exc}")
            FAILURES.append(("Stage 2", m["name"], str(m["slit"]), str(exc)))
            continue
        if not os.path.exists(cal_path(m, spec2_dir)):
            print(f"  !! Stage 2 finished but wrote no cal file for {m['name']}")
            FAILURES.append(("Stage 2", m["name"], str(m["slit"]), "no cal file written"))


def cal_path(m, spec2_dir):
    return os.path.join(spec2_dir, m["name"].replace("_rate.fits", "_cal.fits"))


# ===========================================================================
# Stage 3
# ===========================================================================

def group_metas_by_epoch(metas, tag=None):
    """Partition exposure metadata into {(tag, date): [meta, ...]}.

    `collect_exposures` has already assigned every exposure an
    `epoch_key` of (target, epoch_number) and an `epoch_date`; this just
    turns that into the (tag, date) pair the directory layout uses. The
    tag is the sanitised target name unless overridden, so one proposal
    covering two targets still lands in two directory trees.
    """
    groups = defaultdict(list)
    for m in metas:
        target = m["epoch_key"][0]
        groups[(tag or sanitize(target) or "target", m["epoch_date"])].append(m)
    return dict(sorted(groups.items()))


def stage_epoch_uncal(metas, uncal_files, uncal_dir):
    """Stage just this epoch's uncal files into its uncal/ directory.

    `metas` were read from the uncal files themselves, so each one
    already carries the path it came from; `uncal_files` is only used as
    a fallback match by basename.
    """
    by_name = {os.path.basename(f): f for f in uncal_files}
    wanted = []
    for m in metas:
        path = m.get("path")
        if path and os.path.exists(path) and path.endswith("_uncal.fits"):
            wanted.append(path)
        elif m["name"] in by_name:
            wanted.append(by_name[m["name"]])
    return stage_files(wanted, uncal_dir)


def build_groups(sci, spec2_dir, tag):
    """One group per (target, epoch, filter, grating, slit); same S200A1 +
    S200A2 merge as the notebook. Only exposures that produced a cal file."""
    grouped = defaultdict(list)
    for m in sci:
        if os.path.exists(cal_path(m, spec2_dir)):
            grouped[(m["epoch_key"], m["filter"], m["grating"], m["slit"])].append(m)
    for (ek, filt, grat, slit) in list(grouped):
        other = (ek, filt, grat, "S200A2")
        if slit == "S200A1" and other in grouped:
            grouped[(ek, filt, grat, "S200A1_S200A2")] = grouped.pop((ek, filt, grat, slit)) + grouped.pop(other)

    groups = []
    for (ek, filt, grat, slit), members in sorted(grouped.items(), key=lambda kv: str(kv[0])):
        target, epoch = ek
        date = members[0]["epoch_date"]
        product = f"{filt}_{grat}_{slit}_{sanitize(target)}_{date}{tag}".lower()
        groups.append(dict(
            product=product, target=target, epoch=epoch, epoch_date=date, slit=slit,
            filter=filt, grating=grat, members=members,
            cal_files=[cal_path(m, spec2_dir) for m in members],
            mjd_min=min(m["mjd_start"] for m in members if m["mjd_start"] is not None) if any(m["mjd_start"] is not None for m in members) else None,
            mjd_max=max(m["mjd_end"] for m in members if m["mjd_end"] is not None) if any(m["mjd_end"] is not None for m in members) else None,
        ))
    return groups


def find_products(spec3_dir, product, suffix):
    pats = [f"{product}_{suffix}.fits", f"{product}_*_{suffix}.fits"]
    return sorted({f for p in pats for f in glob.glob(os.path.join(spec3_dir, p))})


def run_spec3(groups, asn_dir, spec3_dir):
    os.makedirs(spec3_dir, exist_ok=True)
    steps = defaultdict(dict)
    steps["master_background"]["skip"] = True
    for g in groups:
        asnfile = os.path.join(asn_dir, f"{g['product']}_l3asn.json")
        asn = afl.asn_from_list(g["cal_files"], rule=DMS_Level3_Base, product_name=g["product"])
        with open(asnfile, "w") as fh:
            fh.write(asn.dump()[1])
        print(f"[Stage 3] {g['product']}  ({len(g['cal_files'])} cal files)")
        try:
            Spec3Pipeline.call(asnfile, save_results=True, steps=steps, output_dir=spec3_dir)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            FAILURES.append(("Stage 3", g["product"], g["slit"], str(exc)))
        if not find_products(spec3_dir, g["product"], "x1d"):
            print(f"  !! no Stage-3 x1d written for {g['product']}")
            FAILURES.append(("Stage 3", g["product"], g["slit"], "no x1d written"))


# ===========================================================================
# Final spectra
# ===========================================================================

def build_final_spectra(groups, spec3_dir, final_dir, tag):
    """One final spectrum per (target, epoch, slit): the Stage-3 x1d files of
    its gratings stitched together. Slits are never mixed."""
    os.makedirs(final_dir, exist_ok=True)
    by_spec = defaultdict(list)
    for g in groups:
        by_spec[(g["target"], g["epoch"], g["slit"])].append(g)

    outputs = {}
    for (target, epoch, slit), gs in by_spec.items():
        pieces = []
        for g in gs:
            x1ds = find_products(spec3_dir, g["product"], "x1d")
            if not x1ds:
                print(f"  WARNING: no Stage-3 x1d found for {g['product']}")
            for f in x1ds:
                # only this group's slit(s): a fixed-slit x1d can also hold the other
                # (EXTENDED, MJy/sr) slits, which must never be stitched into the spectrum
                pieces += read_x1d(f, label=f"{g['grating']}/{g['filter']}", slit=str(slit).split("_"))
        date = gs[0]["epoch_date"]
        print(f"[Final] {target} {slit} epoch {epoch} ({date}): {len(pieces)} piece(s)")
        if not pieces:
            continue
        wave, flux, err, unit = stitch(pieces)
        mjds_lo = [g["mjd_min"] for g in gs if g["mjd_min"] is not None]
        mjds_hi = [g["mjd_max"] for g in gs if g["mjd_max"] is not None]
        meta = dict(TARGET=target, SLIT=slit, EPOCH=epoch, EPOCHDT=date,
                    GRATINGS=",".join(sorted({f"{g['grating']}/{g['filter']}" for g in gs})),
                    MJD_MIN=min(mjds_lo) if mjds_lo else None,
                    MJD_MAX=max(mjds_hi) if mjds_hi else None,
                    MJD_MEAN=0.5 * (min(mjds_lo) + max(mjds_hi)) if mjds_lo and mjds_hi else None,
                    EXTRACT="pipeline Stage-3 x1d")
        base = os.path.join(final_dir, f"{sanitize(target)}_{slit.lower()}_{date}{tag}_final")
        save_spectrum(base, wave, flux, err, unit, meta)
        plot_spectrum(base + ".png", [(p["label"], p["wave"], p["flux"]) for p in pieces] + [("stitched", wave, flux)],
                      unit, title=f"{target}  {slit}  {date}")
        outputs[f"{target}_{slit}_{date}"] = base + ".fits"
    return outputs


# ===========================================================================
# End-of-run report: where does each slit drop out?
# ===========================================================================

def print_funnel(uncal_counts, sci, spec2_dir, groups, finals, required=None, spec3_dir=None):
    """One row per slit: uncal on disk -> rate -> Stage-2 cal -> Stage-3 x1d ->
    final spectrum. A slit whose numbers shrink from left to right tells you
    exactly which stage lost it; the failure list below says why."""
    def parts(slit):
        return ["S200A1", "S200A2"] if str(slit) == "S200A1_S200A2" else [str(slit)]

    slits = set(str(m["slit"]) for m in sci) | set(uncal_counts or {})
    for g in groups:
        slits.update(parts(g["slit"]))
    rows, problems = [], []
    for sl in sorted(slits):
        n_unc = (uncal_counts or {}).get(sl, None)
        n_rate = sum(1 for m in sci if str(m["slit"]) == sl)
        n_cal = sum(1 for m in sci if str(m["slit"]) == sl and os.path.exists(cal_path(m, spec2_dir)))
        n_na = sum(1 for m in sci if str(m["slit"]) == sl and not os.path.exists(cal_path(m, spec2_dir))
                   and os.path.exists(cal_path(m, spec2_dir) + ".noslit"))
        gs = [g for g in groups if sl in parts(g["slit"])]
        n_x1d = sum(1 for g in gs if find_products(spec3_dir, g["product"], "x1d"))
        n_fin = sum(1 for g in gs if any(f"_{g['slit']}_" in k for k in finals))
        rows.append((sl, n_unc, n_rate, f"{n_cal}" + (f"+{n_na}" if n_na else ""), f"{n_x1d}/{len(gs)}", n_fin))
        lost = (n_unc is not None and n_unc != n_rate) or (n_cal + n_na < n_rate)
        if lost or n_x1d < len(gs) or n_fin < len(gs) or not gs:
            problems.append(sl)
    print("\nPer-slit funnel  (uncal -> rate -> cal[+n = slit not on that detector, expected] -> Stage-3 x1d/groups -> final spectra)")
    print(f"  {'slit':16s}{'uncal':>7s}{'rate':>7s}{'cal':>7s}{'stage3':>9s}{'final':>7s}")
    for sl, a, b, c, d, e in rows:
        print(f"  {sl:16s}{('-' if a is None else a):>7}{b:>7}{c:>7}{d:>9}{e:>7}")
    for sl in problems:
        print(f"  ! slit {sl} loses exposures somewhere along the chain (see the failure list)")
    for r in (required or []):
        if str(r).upper() not in {x.upper() for x in slits} or not any(str(r).upper() in [p.upper() for p in parts(g["slit"])] for g in groups):
            print(f"  !!! required slit {r} produced NO Stage-3 product in this run")
    if FAILURES:
        print("\nFAILURES:")
        for stage, name, slit, msg in FAILURES:
            print(f"  [{stage}] slit={slit}  {name}: {msg[:300]}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    import matplotlib
    matplotlib.use("Agg")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proposal", required=True, help="JWST proposal number, e.g. 9258")
    p.add_argument("--outdir", required=True,
               help="Output directory for this target. Instrument and epoch subdirectories are created inside it: <outdir>/<INSTRUMENT>/<target>_<date>/{uncal,stage1,stage2,stage3,reextract,final}")
    p.add_argument("--target", default=None, help="Only process science targets whose name contains this (case-insensitive)")
    p.add_argument("--tag", default=None, help="Appended to the output dir and to product / final-spectrum names")
    p.add_argument("--max-epoch-gap-days", type=float, default=1.0,
                   help="Exposures of a target further apart than this belong to different epochs (default 1.0)")
    p.add_argument("--plan-only", action="store_true", help="Download (unless --skip-download), print the epoch grouping and stop")
    p.add_argument("--skip-download", action="store_true", help="Use files already in <outdir>/uncal/")
    p.add_argument("--from-stage", type=int, choices=[1, 2, 3, 4], default=1,
                   help="Start at this stage (1 detector1, 2 spec2, 3 spec3, 4 final spectra only)")
    p.add_argument("--max-cores", default="half", choices=["none", "quarter", "half", "all"])
    p.add_argument("--no-pixel-background", action="store_true", help="Skip nod background subtraction in Stage 2")
    p.add_argument("--source-type", choices=["POINT", "EXTENDED"], default=None,
                   help="Force the Stage-2 source type of the primary slit (default: as set in the APT / by the pipeline). "
                        "Needs a Stage-2 re-run (--from-stage 2)")
    p.add_argument("--slit", nargs="+", default=None, metavar="SLIT",
                   help="Only process these slits, e.g. --slit S400A1 (epochs are still assigned from all slits)")
    p.add_argument("--require-slit", nargs="+", default=None, metavar="SLIT",
                   help="Stop before Stage 1 if any epoch lacks one of these slits, e.g. --require-slit S400A1")
    p.add_argument("--redo", action="store_true", help="Re-run Stage 1 even if the rate file already exists")
    p.add_argument("--list-mast", action="store_true",
                   help="Only list the uncal files MAST has for the proposal (writes mast_uncal_manifest.txt) and stop")
    args = p.parse_args()

    proposal_num = int(args.proposal)
    padded = f"{proposal_num:05d}"
    tag = f"_{args.tag}" if args.tag else ""
    outdir = os.path.abspath(args.outdir)   # absolute: association files need it
    os.makedirs(outdir, exist_ok=True)

    # Raw downloads are staged here; epochs are not known until the headers
    # have been read, after which each epoch gets its own directory tree.
    download_dir = os.path.join(outdir, "_downloaded_uncal")
    os.makedirs(download_dir, exist_ok=True)

    print(f"JWST Calibration Pipeline version: {jwst.__version__}")
    print(f"Output directory: {outdir}\n")
    t0 = time.perf_counter()

    # ---- files -----------------------------------------------------------
    manifest = os.path.join(outdir, "mast_uncal_manifest.txt")
    if args.list_mast:
        mast_login()
        list_mast_uncal(padded, proposal_num, manifest)
        return
    if args.skip_download or args.from_stage > 1:
        uncal = find_staged_uncal(outdir, download_dir)
    else:
        uncal = query_and_download(padded, proposal_num, download_dir, manifest)
    if not uncal:
        raise SystemExit("No uncal files to process.")

    # ---- plan, on the uncal headers, before anything slow runs -----------
    print_inventory(uncal, "Uncal files on disk (every EXP_TYPE / slit / target, "
                            "before any filtering)")
    sci_all, _, notes_u = collect_exposures(uncal, args.target, args.max_epoch_gap_days)
    if not sci_all:
        raise SystemExit("No fixed-slit science exposures found.")
    uncal_counts_all = defaultdict(int)
    for m in sci_all:
        uncal_counts_all[str(m["slit"])] += 1
    sci_u = [m for m in sci_all
             if not args.slit or str(m["slit"]).upper() in {x.upper() for x in args.slit}]
    plan = format_plan(sci_u, args.max_epoch_gap_days)
    print(plan)
    print_notes(notes_u)
    require_slits_or_exit(sci_all, args.require_slit)
    with open(os.path.join(outdir, "epoch_plan.txt"), "w") as fh:
        fh.write(plan + "\n")
    if args.plan_only:
        return

    # ---- one self-contained directory tree per epoch ---------------------
    # Epochs are split here, on the uncal headers, so that Stage 1 onwards
    # runs entirely inside the epoch's own tree. Nod-partner selection in
    # Stage 2 and association building in Stage 3 are then physically
    # unable to reach across epochs.
    epochs = group_metas_by_epoch(sci_u, args.tag)
    print(f"\n{len(epochs)} epoch(s) to process\n")

    all_finals = {}
    for (tag_name, date), metas in epochs.items():
        paths = ensure_epoch_tree(outdir, "NIRSpec-FS", tag_name, date)
        stem = paths["stem"]
        print(f"=== {stem}: {len(metas)} exposure(s) -> "
              f"{os.path.relpath(paths['epoch'], outdir)}")

        epoch_uncal = stage_epoch_uncal(metas, uncal, paths["uncal"])
        uncal_counts = defaultdict(int)
        for m in metas:
            uncal_counts[str(m["slit"])] += 1

        if args.from_stage == 1:
            run_detector1(epoch_uncal, paths["stage1"], args.max_cores, args.redo)
            print(f"Stage 1 done ({(time.perf_counter() - t0) / 60:.1f} min)\n")

        rate_files = sorted(glob.glob(os.path.join(paths["stage1"], "*_rate.fits")))
        if not rate_files:
            print(f"  WARNING: no rate files in {stem}/stage1 -- skipping epoch.")
            continue

        sci, bkg, notes = collect_exposures(rate_files, args.target,
                                             args.max_epoch_gap_days, args.slit)
        print_notes(notes)
        rate_counts = defaultdict(int)
        for m in sci:
            rate_counts[str(m["slit"])] += 1
        for sl, n in sorted(uncal_counts.items()):
            if args.slit and sl.upper() not in {x.upper() for x in args.slit}:
                continue
            if rate_counts.get(sl, 0) < n:
                print(f"  !! slit {sl}: {n} uncal exposure(s) but only "
                      f"{rate_counts.get(sl, 0)} rate file(s) -- Stage 1 lost "
                      f"{n - rate_counts.get(sl, 0)} (see FAILURES at the end)")
        print(f"  {len(sci)} science and {len(bkg)} dedicated-background exposure(s).")
        if not sci:
            print(f"  WARNING: no science exposures selected for {stem} -- skipping.")
            continue

        # ---- Stage 2 -----------------------------------------------------
        if args.from_stage <= 2:
            run_spec2(sci, bkg, paths["asn"], paths["stage2"],
                      args.no_pixel_background, args.source_type)
            print(f"Stage 2 done ({(time.perf_counter() - t0) / 60:.1f} min)\n")

        groups = build_groups(sci, paths["stage2"], tag)
        if not groups:
            print(f"  WARNING: no Stage-2 cal files for {stem} -- skipping.")
            continue

        # ---- Stage 3 -----------------------------------------------------
        if args.from_stage <= 3:
            run_spec3(groups, paths["asn"], paths["stage3"])
            print(f"Stage 3 done ({(time.perf_counter() - t0) / 60:.1f} min)\n")

        # ---- final -------------------------------------------------------
        finals = build_final_spectra(groups, paths["stage3"], paths["final"], tag)
        print_funnel(uncal_counts, sci, paths["stage2"], groups, finals,
                      args.require_slit, paths["stage3"])
        for key, path in finals.items():
            ecsv = path[:-5] + ".ecsv" if path.endswith(".fits") else path
            record_product(outdir, "NIRSpec-FS", tag_name, date, ecsv,
                            product="final", level="final",
                            fits_path=path, quiet=True)
        all_finals.update({f"{stem} {k}": v for k, v in finals.items()})

    print(f"\nAll done ({(time.perf_counter() - t0) / 60:.1f} min).")
    print(f"  output directory: {outdir}")
    print(f"  product index:    {os.path.join(outdir, MANIFEST_NAME)}")
    for k, v in sorted(all_finals.items()):
        print(f"    {k}: {os.path.relpath(v, outdir)}")


if __name__ == "__main__":
    main()