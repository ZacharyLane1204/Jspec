#!/usr/bin/env python3
"""
miri_lrs_fs_auto_pipeline.py
============================

MIRI LRS fixed-slit reduction for a whole proposal, structured like
nirspec_fs_auto_pipeline.py (same options, same safeguards).

  1. Download every MIRI/SLIT _uncal file of the proposal from MAST
     (restart-safe, no size cut, optional MAST_API_TOKEN).
  2. Stage 1  Detector1Pipeline   uncal -> rate
  3. Stage 2  Spec2Pipeline       rate  -> cal / s2d / x1d, per exposure,
                                  with nod-based background subtraction
  4. Stage 3  Spec3Pipeline       cal   -> s2d / x1d (or c1d), ONE product
                                  per (target, epoch, slit)
  5. Final    one spectrum per (target, epoch, slit), Jy vs micron, with
              MJD_MIN / MJD_MAX / MJD_MEAN in the FITS header and ECSV meta.

Epochs
------
Exposures of one target are sorted by start time; a gap larger than
--max-epoch-gap-days (default 1) starts a new epoch. Epochs are NEVER mixed:
  * nod partners used as background in Stage 2 come from the same epoch
    (and the same visit when the visit has a partner)
  * every Stage-3 association contains a single epoch
  * every final spectrum is a single epoch (the date is in the file name)
Run with --plan-only first to see how your exposures are grouped.

Self-contained: this script does NOT import combine_nirspec_orders.py.
(That module's reader looks for an EXTRACT1D table, but Stage-3 *_c1d files --
which is what --optimal-extraction produces -- store theirs in COMBINE1D. That
mismatch is what raised KeyError: "Extension 'EXTRACT1D' not found.")
The reader here accepts EXTRACT1D or COMBINE1D (or any table that has
WAVELENGTH and FLUX columns), FLUX_ERROR or ERROR, keeps the same point
selection as combine_nirspec_orders (DQ == 0; --dq-loose relaxes it) and prints
the file's HDU list if no spectrum table is found.
If nirspec_fs_refining.py is importable, its save_spectrum / plot_spectrum are
reused so the final files have the same format as the NIRSpec finals.

Usage
-----
    python miri_lrs_fs_auto_pipeline.py --proposal 1536 --plan-only
    python miri_lrs_fs_auto_pipeline.py --proposal 1536 --target 2024aecx --tag sn2024aecx
    python miri_lrs_fs_auto_pipeline.py --proposal 1536 --skip-download --from-stage 2
    python miri_lrs_fs_auto_pipeline.py --proposal 1536 --from-stage 4      # rebuild finals only
    python miri_lrs_fs_auto_pipeline.py --proposal 1536 --list-mast
    python miri_lrs_fs_auto_pipeline.py --proposal 1536 --optimal-extraction --tag optimal

Data still under exclusive access needs `export MAST_API_TOKEN=<token>`.
Stage 1 skips exposures whose rate file already exists (--redo to force).
--from-stage {1,2,3,4}: 4 only rebuilds the final spectra from Stage-3 products.
"""

import argparse
import glob
import os
import re
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

# Optional: reuse the NIRSpec writers so the final files share one format.
try:
    from pipelines.nirspec_fs_refining import save_spectrum as _shared_save          # noqa: E402
    from pipelines.nirspec_fs_refining import plot_spectrum as _shared_plot          # noqa: E402
except Exception:                                                          # noqa: BLE001
    _shared_save = _shared_plot = None

LRS_EXP_TYPE = "MIR_LRS-FIXEDSLIT"
FAILURES = []            # (stage, file name, slit, message) -- printed at the end


# ===========================================================================
# Small utilities
# ===========================================================================

def sanitize(s):
    """'SN 2024aecx' -> 'sn2024aecx' (for matching and file names)."""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def canon_target(s):
    """Name used to decide 'same target': 'SN 2024aecx', 'SN2024aecx' and
    '2024aecx' all become '2024aecx' (the same SN is often named differently in
    different visits)."""
    return re.sub(r"^(sn|at)(?=\d)", "", sanitize(s))


def mjd_to_datestr(mjd):
    from astropy.time import Time
    return Time(mjd, format="mjd").strftime("%Y%m%d")


def _truthy(v):
    return str(v).strip().upper() in ("T", "TRUE", "1")


def _num(v, cast=float):
    try:
        return cast(v)
    except (TypeError, ValueError):
        return None


# ===========================================================================
# Header bookkeeping
# ===========================================================================

def read_meta(path):
    """Everything the planner needs, from the primary header of an uncal / rate
    / cal file."""
    h = fits.getheader(path, 0)
    return dict(
        path=os.path.abspath(path), name=os.path.basename(path),
        exp_type=h.get("EXP_TYPE"),
        slit=str(h.get("APERNAME") or "UNKNOWN_SLIT"),
        target=str(h.get("TARGNAME") or "UNKNOWN_TARGET"),
        targprop=str(h.get("TARGPROP") or ""),
        is_bkg=_truthy(h.get("BKGDTARG", False)),
        patttype=str(h.get("PATTTYPE") or "NONE").upper(),
        nod=_num(h.get("PATT_NUM"), int),
        npts=_num(h.get("NUMDTHPT"), int),
        observtn=h.get("OBSERVTN"), visit=h.get("VISIT"),
        filter=str(h.get("FILTER") or ""),
        subarray=str(h.get("SUBARRAY") or ""),
        program=h.get("PROGRAM"),
        mjd_start=_num(h.get("EXPSTART")), mjd_end=_num(h.get("EXPEND")),
    )


def assign_epochs(metas, gap_days):
    """Sets m['epoch_key'] = (target, epoch number) and m['epoch_date'] on every
    entry. A gap > gap_days between an exposure's start and the latest end of
    the earlier exposures of the same target starts a new epoch."""
    by_t = defaultdict(list)
    for m in metas:
        by_t[canon_target(m["target"])].append(m)
    for ms in by_t.values():
        ms.sort(key=lambda m: m["mjd_start"] if m["mjd_start"] is not None else np.inf)
        disp = ms[0]["target"]
        epoch, latest_end, members = 1, None, []

        def close(members, epoch):
            starts = [x["mjd_start"] for x in members if x["mjd_start"] is not None]
            date = mjd_to_datestr(min(starts)) if starts else "unknowndate"
            for x in members:
                x["epoch_key"], x["epoch_date"] = (disp, epoch), date

        for m in ms:
            if (members and m["mjd_start"] is not None and latest_end is not None
                    and m["mjd_start"] - latest_end > gap_days):
                close(members, epoch)
                epoch, members = epoch + 1, []
            members.append(m)
            end = m["mjd_end"] if m["mjd_end"] is not None else m["mjd_start"]
            if end is not None:
                latest_end = end if latest_end is None else max(latest_end, end)
        close(members, epoch)


# ===========================================================================
# Download
# ===========================================================================

MAST_CHUNK = 25          # observations per get_product_list call
MAST_RETRIES = 3


def mast_login():
    """Optional: set MAST_API_TOKEN (https://auth.mast.stsci.edu/token) to fetch
    data still under exclusive access."""
    token = os.getenv("MAST_API_TOKEN")
    if token:
        try:
            Observations.login(token=token)
            print("  Logged in to MAST with MAST_API_TOKEN.")
        except Exception as exc:
            print(f"  WARNING: MAST login failed ({exc}); exclusive-access data will not download.")


def list_mast_uncal(proposal_padded, proposal_num, manifest_path=None):
    """Every level-1 *_uncal.fits of the proposal as a list of dicts (name, uri,
    size, rights, obs_id). No size filtering."""
    print(f"Querying MAST for MIRI/SLIT observations in proposal {proposal_padded} ...")
    obs_table = Observations.query_criteria(
        obs_collection="JWST", instrument_name=["MIRI/SLIT"], provenance_name=["CALJWST"],
        proposal_id=[str(proposal_num)])
    print(f"  Found {len(obs_table)} observation entries (provenance CALJWST).")

    # Fallback without the provenance filter: an exposure whose calibrated product
    # MAST does not carry can still have its uncal file in the archive.
    try:
        obs_any = Observations.query_criteria(
            obs_collection="JWST", instrument_name=["MIRI/SLIT"], proposal_id=[str(proposal_num)])
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
                res = Observations.download_file(e["uri"], local_path=dest)
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
# Exposure selection, epochs, coverage checks
# ===========================================================================

def print_inventory(files, title):
    """What is physically in a directory, straight from the headers. Prints every
    EXP_TYPE / slit / target, including the ones later steps ignore (e.g. target
    acquisition)."""
    print(f"\n{title}: {len(files)} file(s)")
    tally = defaultdict(list)
    for f in files:
        try:
            m = read_meta(f)
        except Exception as exc:
            print(f"  !! cannot read header of {os.path.basename(f)}: {exc}")
            continue
        tally[(str(m["exp_type"]), m["slit"], m["target"], m["is_bkg"])].append(m)
    for (et, slit, target, isb), ms in sorted(tally.items()):
        print(f"  {et:20s} slit={slit:12s} target={target!r:28s} bkgd={str(isb):5s} {len(ms):3d} file(s)")
    print()


def collect_exposures(files, target_filter, gap_days):
    """Read headers, keep LRS fixed-slit exposures only, split science /
    dedicated-background, assign epochs. Dedicated backgrounds (BKGDTARG=T)
    inherit the epoch of the nearest science exposure if it is within the gap;
    otherwise they are dropped. Returns (sci, bkg, notes)."""
    metas = [read_meta(f) for f in files]
    fs = [m for m in metas if m["exp_type"] == LRS_EXP_TYPE]
    if len(fs) < len(metas):
        odd = defaultdict(int)
        for m in metas:
            if m["exp_type"] != LRS_EXP_TYPE:
                odd[str(m["exp_type"])] += 1
        print("  Ignoring " + str(len(metas) - len(fs)) + " non-LRS-fixed-slit exposure(s): "
              + ", ".join(f"{n}x {et}" for et, n in sorted(odd.items())))
    sci = [m for m in fs if not m["is_bkg"]]
    bkg = [m for m in fs if m["is_bkg"]]
    if bkg:
        print(f"  {len(bkg)} exposure(s) flagged BKGDTARG=T (dedicated background)")

    if target_filter:
        # names with spaces/punctuation removed, matched in both directions, so
        # "SN2024aecx", "SN 2024aecx" and "2024aecx" all select the same target
        t = sanitize(target_filter)

        def _match(m):
            for name in (sanitize(m["target"]), sanitize(m["targprop"])):
                if name and (t in name or (len(name) >= 5 and name in t)):
                    return True
            return False
        keep = [m for m in sci if _match(m)]
        dropped = [m for m in sci if m not in keep]
        if dropped:
            why = defaultdict(int)
            for m in dropped:
                why[(m["slit"], m["target"], m["targprop"])] += 1
            print(f"  --target {target_filter!r} DROPPED {len(dropped)} science exposure(s):")
            for (sl, tn, tp), n in sorted(why.items()):
                print(f"      slit={sl}  TARGNAME={tn!r}  TARGPROP={tp!r}  x{n}")
        sci = keep

    assign_epochs(sci, gap_days)
    kept_bkg = []
    for b in bkg:
        if b["mjd_start"] is None or not sci:
            continue
        nearest = min((s for s in sci if s["mjd_start"] is not None),
                      key=lambda s: abs(s["mjd_start"] - b["mjd_start"]), default=None)
        if nearest is not None and abs(nearest["mjd_start"] - b["mjd_start"]) <= gap_days:
            b["epoch_key"], b["epoch_date"] = nearest["epoch_key"], nearest["epoch_date"]
            kept_bkg.append(b)
    return sci, kept_bkg, check_coverage(sci)


def check_coverage(sci):
    """Things that usually mean 'a download is missing': an incomplete nod set
    (fewer nod positions than NUMDTHPT) in a visit -- with a missing partner
    Stage 2 silently subtracts less background."""
    notes = []
    vis = defaultdict(list)
    for m in sci:
        vis[(m["epoch_key"], m["slit"], m["observtn"], m["visit"])].append(m)
    for (ek, slit, obs, visit), ms in sorted(vis.items(), key=lambda kv: str(kv[0])):
        m0 = ms[0]
        if "NOD" not in m0["patttype"] or not m0["npts"]:
            continue
        nods = sorted({m["nod"] for m in ms if m["nod"] is not None})
        if len(nods) < m0["npts"]:
            notes.append(f"epoch {ek[1]} ({m0['epoch_date']}) {slit} obs {obs} visit {visit}: "
                         f"{len(nods)} of {m0['npts']} nod positions present (have {nods}) "
                         f"-> incomplete nod set, background subtraction will be degraded")
    return notes


def print_notes(notes):
    if notes:
        print("\nCOVERAGE WARNINGS:")
        for n in notes:
            print(f"  ! {n}")
        print()


def format_plan(sci, gap_days):
    lines = [f"Epoch plan (a gap > {gap_days:g} d between exposures starts a new epoch)"]
    by_epoch = defaultdict(list)
    for m in sci:
        by_epoch[m["epoch_key"]].append(m)
    for (target, epoch), members in sorted(by_epoch.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        starts = [m["mjd_start"] for m in members if m["mjd_start"] is not None]
        ends = [m["mjd_end"] for m in members if m["mjd_end"] is not None]
        span = f"MJD {min(starts):.3f}-{max(ends):.3f}" if starts and ends else "MJD unknown"
        d = members[0]["epoch_date"]
        lines.append(f"\n  {target}  epoch {epoch}  ({d[:4]}-{d[4:6]}-{d[6:]}, {span})")
        cfgs = defaultdict(list)
        for m in members:
            cfgs[(m["slit"], m["filter"], m["subarray"])].append(m)
        for (slit, filt, sub), ms in sorted(cfgs.items()):
            nods = sorted({m["nod"] for m in ms if m["nod"] is not None})
            lines.append(f"      {slit} {filt} {sub}: {len(ms)} file(s), nod position(s) {nods}, "
                         f"pattern {ms[0]['patttype']}")
    return "\n".join(lines)


# ===========================================================================
# Stage 1
# ===========================================================================

def _rate_path(uncal_path, det1_dir):
    return os.path.join(det1_dir, os.path.basename(uncal_path).replace("_uncal.fits", "_rate.fits"))


def run_detector1(metas, det1_dir, max_cores, find_showers, rejection_threshold, redo=False):
    os.makedirs(det1_dir, exist_ok=True)
    steps = defaultdict(dict)
    steps["jump"]["find_showers"] = find_showers
    steps["jump"]["rejection_threshold"] = rejection_threshold
    steps["jump"]["maximum_cores"] = max_cores
    for m in metas:
        f = m["path"]
        rate = _rate_path(f, det1_dir)
        if os.path.exists(rate) and not redo:
            try:
                fits.getheader(rate, 0)
                fits.getdata(rate, "SCI")               # a truncated file fails here
                print(f"[Stage 1] {m['name']}: rate file exists, skipping (--redo to force)")
                continue
            except Exception:
                print(f"[Stage 1] {m['name']}: existing rate file is unreadable -> re-running")
        print(f"[Stage 1] {m['name']}")
        try:
            Detector1Pipeline.call(f, save_results=True, steps=steps, output_dir=det1_dir)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            FAILURES.append(("Stage 1", m["name"], m["slit"], str(exc)))
        if not os.path.exists(rate):
            print(f"  !! no rate file was produced for {m['name']}")
            FAILURES.append(("Stage 1", m["name"], m["slit"], "no rate file produced"))


# ===========================================================================
# Stage 2
# ===========================================================================

def cal_path(m, spec2_dir):
    return os.path.join(spec2_dir, m["name"].replace("_rate.fits", "_cal.fits"))


def _same_setup(a, b):
    """Same slit / subarray AND same target+epoch."""
    return (a["slit"] == b["slit"] and a["subarray"] == b["subarray"]
            and a["epoch_key"] == b["epoch_key"])


def write_l2_asn(m, sci, bkg, asnfile):
    """Level-2 association for one science exposure. Nodded data: the other nod
    position(s) of the same setup and epoch are the background (those from the
    same visit if there are any, otherwise from the whole epoch). Otherwise:
    dedicated background exposures of the same setup and epoch."""
    asn = afl.asn_from_list([m["path"]], rule=DMSLevel2bBase, product_name="Level2")
    asn.data["program"] = m["program"]
    members = asn["products"][0]["members"]

    use_sci = [s for s in sci if _same_setup(m, s)]
    if "NOD" in m["patttype"]:
        cands = [s for s in use_sci if s["nod"] != m["nod"]]
        same_visit = [s for s in cands if (s["observtn"], s["visit"]) == (m["observtn"], m["visit"])]
        bkg_members = same_visit or cands
    else:
        bkg_members = [b for b in bkg if _same_setup(m, b)]
    for b in bkg_members:
        members.append({"expname": b["path"], "exptype": "background"})

    _, serialized = asn.dump()
    with open(asnfile, "w") as fh:
        fh.write(serialized)
    return len(bkg_members)


def run_spec2(sci, bkg, asn_dir, spec2_dir, skip_bkg):
    os.makedirs(asn_dir, exist_ok=True)
    os.makedirs(spec2_dir, exist_ok=True)
    steps = defaultdict(dict)
    steps["bkg_subtract"]["skip"] = skip_bkg           # False -> nod background is subtracted
    for m in sci:
        asnfile = os.path.join(asn_dir, m["name"].replace("_rate.fits", "_l2asn.json"))
        try:
            n_bkg = write_l2_asn(m, sci, bkg, asnfile)
            print(f"[Stage 2] {m['name']}  ({m['slit']}, nod {m['nod']}, {n_bkg} background exposure(s))")
            if n_bkg == 0 and not skip_bkg:
                print("    WARNING: no nod partner / background exposure of the same setup and epoch "
                      "-> NO background is subtracted for this exposure")
            Spec2Pipeline.call(asnfile, save_results=True, steps=steps, output_dir=spec2_dir)
        except Exception as exc:
            print(f"  skipped {m['name']}: {exc}")
            FAILURES.append(("Stage 2", m["name"], m["slit"], str(exc)))
            continue
        if not os.path.exists(cal_path(m, spec2_dir)):
            print(f"  !! Stage 2 finished but wrote no cal file for {m['name']}")
            FAILURES.append(("Stage 2", m["name"], m["slit"], "no cal file written"))


# ===========================================================================
# Stage 3
# ===========================================================================

def build_groups(sci, spec2_dir, tag):
    """One group per (target, epoch, slit); only exposures that produced a cal file."""
    grouped = defaultdict(list)
    for m in sci:
        if os.path.exists(cal_path(m, spec2_dir)):
            grouped[(m["epoch_key"], m["slit"])].append(m)
    groups = []
    for (ek, slit), members in sorted(grouped.items(), key=lambda kv: str(kv[0])):
        target, epoch = ek
        date = members[0]["epoch_date"]
        product = f"{sanitize(target)}_{slit}_{date}{tag}".lower()
        starts = [m["mjd_start"] for m in members if m["mjd_start"] is not None]
        ends = [m["mjd_end"] for m in members if m["mjd_end"] is not None]
        groups.append(dict(
            product=product, target=target, epoch=epoch, epoch_date=date, slit=slit,
            members=members, cal_files=[cal_path(m, spec2_dir) for m in members],
            mjd_min=min(starts) if starts else None, mjd_max=max(ends) if ends else None))
    return groups


def find_products(spec3_dir, product, suffix):
    pats = [f"{product}_{suffix}.fits", f"{product}_*_{suffix}.fits"]
    return sorted({f for p in pats for f in glob.glob(os.path.join(spec3_dir, p))})


def run_spec3(groups, asn_dir, spec3_dir, skip_pixel_replace, optimal_extraction):
    os.makedirs(spec3_dir, exist_ok=True)
    steps = defaultdict(dict)
    steps["master_background"]["skip"] = True
    steps["pixel_replace"]["skip"] = skip_pixel_replace
    if optimal_extraction:
        steps["resample_spec"]["skip"] = True
        steps["pixel_replace"]["skip"] = True
        steps["extract_1d"] = {
            "extraction_type": "optimal",
            "use_source_posn": True,
            "model_nod_pair": True,
            "optimize_psf_location": True,
            "save_profile": True,
            "save_residual_image": True,
        }
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
        if not (find_products(spec3_dir, g["product"], "x1d")
                or find_products(spec3_dir, g["product"], "c1d")):
            print(f"  !! no Stage-3 x1d/c1d written for {g['product']}")
            FAILURES.append(("Stage 3", g["product"], g["slit"], "no x1d/c1d written"))


# ===========================================================================
# Reading / combining / saving spectra
# ===========================================================================

_TABLE_NAMES = ("EXTRACT1D", "COMBINE1D")
_FLUX_ERR_NAMES = ("FLUX_ERROR", "FLUX_ERR", "ERROR", "ERR")


def read_x1d(path, label=None, dq_strict=True):
    """Read every spectrum table in an x1d or c1d file. Accepts EXTRACT1D and
    COMBINE1D tables (any EXTVER) and, failing that, any table with WAVELENGTH
    and FLUX columns. The error column may be FLUX_ERROR (x1d) or ERROR (c1d).

    Point selection matches combine_nirspec_orders.read_x1d: finite wavelength,
    flux and error with error > 0, and DQ == 0 (dq_strict=True). With
    dq_strict=False only the DO_NOT_USE bit rejects a point. Output is sorted
    by wavelength (LRS tables are stored in descending order).

    Returns a list of dicts (label, wave, flux, err, unit). Raises RuntimeError
    listing the file's HDUs if no spectrum is found."""
    pieces = []
    with fits.open(path) as hdul:
        def has_cols(h):
            names = {n.upper() for n in (h.columns.names if hasattr(h, "columns") else [])}
            return "WAVELENGTH" in names and "FLUX" in names
        tables = [h for h in hdul[1:] if isinstance(h, fits.BinTableHDU)
                  and h.header.get("EXTNAME", "").upper() in _TABLE_NAMES and has_cols(h)]
        if not tables:
            tables = [h for h in hdul[1:] if isinstance(h, fits.BinTableHDU) and has_cols(h)]
        if not tables:
            listing = ", ".join(f"{h.header.get('EXTNAME', '?')}({type(h).__name__})" for h in hdul)
            raise RuntimeError(f"no spectrum table (WAVELENGTH + FLUX columns) in {os.path.basename(path)}; "
                               f"HDUs: {listing}")
        for h in tables:
            d = h.data
            cols = {n.upper(): n for n in d.columns.names}
            wave = np.asarray(d[cols["WAVELENGTH"]], float)
            flux = np.asarray(d[cols["FLUX"]], float)
            en = next((cols[n] for n in _FLUX_ERR_NAMES if n in cols), None)
            err = np.asarray(d[en], float) if en else np.full_like(flux, np.nan)
            ok = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
            if en:
                ok &= np.isfinite(err) & (err > 0)
            if "DQ" in cols:
                dq = np.asarray(d[cols["DQ"]]).astype(np.int64)
                ok &= (dq == 0) if dq_strict else ((dq & 1) == 0)
            if not ok.any():
                continue
            wave, flux, err = wave[ok], flux[ok], err[ok]
            srt = np.argsort(wave)
            unit = str(h.columns[cols["FLUX"]].unit or "Jy")
            pieces.append(dict(label=label or os.path.basename(path).replace(".fits", ""),
                               wave=wave[srt], flux=flux[srt], err=err[srt], unit=unit))
    if not pieces:
        raise RuntimeError(f"{os.path.basename(path)}: spectrum table(s) present but no valid points "
                           f"(DQ/error cuts removed everything; try --dq-loose)")
    return pieces


def combine_pieces(pieces):
    """One input: returned unchanged. Several (same epoch, same wavelength
    coverage): inverse-variance-weighted mean on the wavelength grid of the
    longest piece; equal weights where an error is missing."""
    if len(pieces) == 1:
        p = pieces[0]
        return p["wave"], p["flux"], p["err"], p["unit"]
    ref = max(pieces, key=lambda p: len(p["wave"]))
    grid = ref["wave"]
    num = np.zeros_like(grid)
    den = np.zeros_like(grid)
    for p in pieces:
        inside = (grid >= p["wave"][0]) & (grid <= p["wave"][-1])
        f = np.interp(grid, p["wave"], p["flux"])
        e = np.interp(grid, p["wave"], p["err"])
        w = np.where(np.isfinite(e) & (e > 0), 1.0 / np.where(e > 0, e, 1.0) ** 2, 1.0)
        w = np.where(inside, w, 0.0)
        num += w * f
        den += w
    good = den > 0
    flux = np.where(good, num / np.where(good, den, 1), np.nan)
    err = np.where(good, 1.0 / np.sqrt(np.where(good, den, 1)), np.nan)
    if not np.isfinite(ref["err"]).any():                # no errors anywhere -> don't invent them
        err = np.full_like(flux, np.nan)
    return grid[good], flux[good], err[good], ref["unit"]


def _local_save(base, wave, flux, err, unit, meta):
    """Same layout as combine_nirspec_orders.save_combined_spectrum: a table with
    WAVELENGTH / FLUX / FLUX_ERROR columns, metadata (incl. MJD_MIN/MAX/MEAN)
    carried into the FITS extension header and the ECSV header block."""
    from astropy.table import Table
    t = Table({"WAVELENGTH": wave, "FLUX": flux, "FLUX_ERROR": err})
    t["WAVELENGTH"].unit, t["FLUX"].unit, t["FLUX_ERROR"].unit = "um", unit, unit
    for k, v in meta.items():
        if v is not None:
            t.meta[k] = v
    t.write(base + ".fits", overwrite=True)
    t.write(base + ".ecsv", overwrite=True)
    print(f"Saved combined spectrum to {base}.fits / .ecsv")


def _local_plot(path, series, unit, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    for label, w, f in series:
        ax.plot(w, f, lw=1.2, label=label)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel(f"Flux ({unit})")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if len(series) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_spectrum(base, wave, flux, err, unit, meta):
    if _shared_save is not None:
        try:
            _shared_save(base, wave, flux, err, unit, meta)
            return
        except Exception as exc:
            print(f"  (nirspec_fs_refining.save_spectrum failed: {exc}; using the built-in writer)")
    _local_save(base, wave, flux, err, unit, meta)


def plot_spectrum(path, series, unit, title=""):
    if _shared_plot is not None:
        try:
            _shared_plot(path, series, unit, title=title)
            return
        except Exception as exc:
            print(f"  (nirspec_fs_refining.plot_spectrum failed: {exc}; using the built-in plot)")
    _local_plot(path, series, unit, title)


# ===========================================================================
# Final spectra
# ===========================================================================

def build_final_spectra(groups, spec3_dir, final_dir, tag, optimal_extraction, dq_strict=True):
    """One final spectrum per (target, epoch, slit). Box extraction -> the Stage-3
    x1d; optimal extraction -> the Stage-3 c1d (combined) product. Falls back to
    the other product type if the preferred one is missing."""
    os.makedirs(final_dir, exist_ok=True)
    order = ("c1d", "x1d") if optimal_extraction else ("x1d", "c1d")
    outputs = {}
    for g in groups:
        target, slit, date = g["target"], g["slit"], g["epoch_date"]
        files, used = [], None
        for suffix in order:
            files = find_products(spec3_dir, g["product"], suffix)
            if files:
                used = suffix
                break
        if not files:
            print(f"[Final] {target} {slit} ({date}): WARNING no Stage-3 x1d/c1d found for {g['product']}")
            continue
        try:
            pieces = []
            for f in files:
                pieces += read_x1d(f, dq_strict=dq_strict)
            wave, flux, err, unit = combine_pieces(pieces)
            print(f"[Final] {target} {slit} epoch {g['epoch']} ({date}): {len(pieces)} spectrum table(s) "
                  f"from {len(files)} {used} file(s), {len(wave)} points")
            lo, hi = g["mjd_min"], g["mjd_max"]
            meta = dict(TARGET=target, SLIT=slit, EPOCH=g["epoch"], EPOCHDT=date,
                        MJD_MIN=lo, MJD_MAX=hi,
                        MJD_MEAN=0.5 * (lo + hi) if lo is not None and hi is not None else None,
                        NEXP=len(g["members"]),
                        EXTRACT=("optimal (Stage-3 c1d)" if optimal_extraction else "box (Stage-3 x1d)")
                        if used == order[0] else f"Stage-3 {used}")
            base = os.path.join(final_dir, f"{sanitize(target)}_{slit.lower()}_{date}{tag}_final")
            save_spectrum(base, wave, flux, err, unit, meta)
            try:
                plot_spectrum(base + ".png", [("final", wave, flux)], unit, title=f"{target}  {slit}  {date}")
            except Exception as exc:
                print(f"  (quick-look plot skipped: {exc})")
            outputs[f"{target}_{slit}_{date}"] = base + ".fits"
        except Exception as exc:
            print(f"  FAILED building final spectrum for {g['product']}: {exc}")
            FAILURES.append(("Final", g["product"], slit, str(exc)))
    return outputs


# ===========================================================================
# End-of-run report
# ===========================================================================

def print_funnel(uncal_count, sci, spec2_dir, groups, finals, spec3_dir):
    """uncal -> rate -> Stage-2 cal -> Stage-3 product -> final spectrum, per slit."""
    slits = sorted({str(m["slit"]) for m in sci})
    print("\nPer-slit funnel  (uncal -> rate -> cal -> Stage-3 products/groups -> final spectra)")
    print(f"  {'slit':16s}{'uncal':>7s}{'rate':>7s}{'cal':>7s}{'stage3':>9s}{'final':>7s}")
    problems = []
    for sl in slits:
        n_unc = (uncal_count or {}).get(sl)
        n_rate = sum(1 for m in sci if m["slit"] == sl)
        n_cal = sum(1 for m in sci if m["slit"] == sl and os.path.exists(cal_path(m, spec2_dir)))
        gs = [g for g in groups if g["slit"] == sl]
        n_s3 = sum(1 for g in gs if find_products(spec3_dir, g["product"], "x1d")
                   or find_products(spec3_dir, g["product"], "c1d"))
        n_fin = sum(1 for g in gs if f"{g['target']}_{g['slit']}_{g['epoch_date']}" in finals)
        print(f"  {sl:16s}{('-' if n_unc is None else n_unc):>7}{n_rate:>7}{n_cal:>7}{f'{n_s3}/{len(gs)}':>9}{n_fin:>7}")
        if (n_unc is not None and n_unc != n_rate) or n_cal < n_rate or n_s3 < len(gs) or n_fin < len(gs) or not gs:
            problems.append(sl)
    for sl in problems:
        print(f"  ! slit {sl} loses exposures somewhere along the chain (see the failure list)")
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
    p.add_argument("--proposal", required=True, help="JWST proposal number, e.g. 1536")
    p.add_argument("--outdir", default=None, help="Base output directory (default ./lrs_data_<proposal>[_tag])")
    p.add_argument("--target", default=None, help="Only process science targets whose name contains this (case-insensitive)")
    p.add_argument("--tag", default=None, help="Appended to the output dir and to product / final-spectrum names")
    p.add_argument("--max-epoch-gap-days", type=float, default=1.0,
                   help="Exposures of a target further apart than this belong to different epochs (default 1.0)")
    p.add_argument("--plan-only", action="store_true", help="Download (unless --skip-download), print the epoch grouping and stop")
    p.add_argument("--skip-download", action="store_true", help="Use files already in <outdir>/uncal/")
    p.add_argument("--from-stage", type=int, choices=[1, 2, 3, 4], default=1,
                   help="Start at this stage (1 detector1, 2 spec2, 3 spec3, 4 final spectra only)")
    p.add_argument("--max-cores", default="half", choices=["none", "quarter", "half", "all"])
    p.add_argument("--no-shower-detection", action="store_true", help="Disable cosmic-ray shower detection in the jump step")
    p.add_argument("--jump-threshold", type=float, default=5.0, help="Jump-step rejection threshold (default 5.0)")
    p.add_argument("--no-background-subtraction", "--no-pixel-background", dest="no_bkg", action="store_true",
                   help="Skip nod background subtraction in Stage 2")
    p.add_argument("--no-pixel-replace", action="store_true", help="Disable bad-pixel interpolation (pixel_replace) in Stage 3")
    p.add_argument("--optimal-extraction", action="store_true",
                   help="Optimal extraction in Stage 3 (no resampling); final spectrum comes from the c1d product")
    p.add_argument("--dq-loose", action="store_true",
                   help="Final spectrum: reject only DO_NOT_USE points instead of every DQ != 0 point "
                        "(default matches combine_nirspec_orders: DQ == 0)")
    p.add_argument("--redo", action="store_true", help="Re-run Stage 1 even if the rate file already exists")
    p.add_argument("--list-mast", action="store_true",
                   help="Only list the uncal files MAST has for the proposal (writes mast_uncal_manifest.txt) and stop")
    args = p.parse_args()

    proposal_num = int(args.proposal)
    padded = f"{proposal_num:05d}"
    tag = f"_{args.tag}" if args.tag else ""
    outdir = os.path.abspath(args.outdir or f"./lrs_data_{padded}{tag}")   # absolute: association files need it
    d = {k: os.path.join(outdir, k) for k in ("uncal", "stage1", "asn", "stage2", "stage3", "final_spectrum")}
    for path in d.values():
        os.makedirs(path, exist_ok=True)
    print(f"JWST Calibration Pipeline version: {jwst.__version__}")
    print(f"Output directory: {outdir}")
    print(f"Spectrum writer: {'nirspec_fs_refining (same format as NIRSpec finals)' if _shared_save else 'built-in'}\n")
    t0 = time.perf_counter()

    # ---- files -----------------------------------------------------------
    manifest = os.path.join(outdir, "mast_uncal_manifest.txt")
    uncal_count = None
    if args.list_mast:
        mast_login()
        list_mast_uncal(padded, proposal_num, manifest)
        return
    if args.skip_download or args.from_stage > 1:
        uncal = sorted(glob.glob(os.path.join(d["uncal"], "*uncal.fits")))
    else:
        uncal = query_and_download(padded, proposal_num, d["uncal"], manifest)

    sci_u = bkg_u = None
    if args.plan_only or args.from_stage == 1:
        if not uncal:
            raise SystemExit("No uncal files to plan from.")
        print_inventory(uncal, "Uncal files on disk (every EXP_TYPE / slit / target, before any filtering)")
        sci_u, bkg_u, notes_u = collect_exposures(uncal, args.target, args.max_epoch_gap_days)
        uncal_count = defaultdict(int)
        for m in sci_u:
            uncal_count[m["slit"]] += 1
        print(format_plan(sci_u, args.max_epoch_gap_days))
        print_notes(notes_u)
        if args.plan_only:
            return
        if not sci_u:
            raise SystemExit("No LRS fixed-slit science exposures selected.")

    # ---- Stage 1 ---------------------------------------------------------
    if args.from_stage == 1:
        run_detector1(sci_u + bkg_u, d["stage1"], args.max_cores,
                      find_showers=not args.no_shower_detection,
                      rejection_threshold=args.jump_threshold, redo=args.redo)
        print(f"Stage 1 done ({(time.perf_counter() - t0) / 60:.1f} min)\n")

    rate_files = sorted(glob.glob(os.path.join(d["stage1"], "*_rate.fits")))
    if not rate_files:
        raise SystemExit("No rate files in stage1/.")

    sci, bkg, notes = collect_exposures(rate_files, args.target, args.max_epoch_gap_days)
    if uncal_count:
        rate_count = defaultdict(int)
        for m in sci:
            rate_count[m["slit"]] += 1
        for sl, n in sorted(uncal_count.items()):
            if rate_count.get(sl, 0) < n:
                print(f"  !! slit {sl}: {n} uncal exposure(s) but only {rate_count.get(sl, 0)} rate file(s) -- "
                      f"Stage 1 lost {n - rate_count.get(sl, 0)} (see FAILURES at the end)")
    plan = format_plan(sci, args.max_epoch_gap_days)
    print(plan)
    print_notes(notes)
    with open(os.path.join(outdir, "epoch_plan.txt"), "w") as fh:
        fh.write(plan + "\n")
    print(f"\n{len(sci)} science and {len(bkg)} dedicated-background exposure(s).\n")
    if not sci:
        raise SystemExit("No science exposures selected.")

    # ---- Stage 2 ---------------------------------------------------------
    if args.from_stage <= 2:
        run_spec2(sci, bkg, d["asn"], d["stage2"], args.no_bkg)
        print(f"Stage 2 done ({(time.perf_counter() - t0) / 60:.1f} min)\n")

    groups = build_groups(sci, d["stage2"], tag)
    if not groups:
        raise SystemExit("No Stage-2 cal files found for the selected exposures.")

    # ---- Stage 3 ---------------------------------------------------------
    if args.from_stage <= 3:
        run_spec3(groups, d["asn"], d["stage3"], args.no_pixel_replace, args.optimal_extraction)
        print(f"Stage 3 done ({(time.perf_counter() - t0) / 60:.1f} min)\n")

    # ---- final -----------------------------------------------------------
    finals = build_final_spectra(groups, d["stage3"], d["final_spectrum"], tag, args.optimal_extraction,
                                 dq_strict=not args.dq_loose)
    print_funnel(uncal_count, sci, d["stage2"], groups, finals, d["stage3"])

    print(f"\nAll done ({(time.perf_counter() - t0) / 60:.1f} min).")
    print(f"  per-exposure spectra (Stage 2): {d['stage2']}")
    print(f"  per-epoch Stage-3 products: {d['stage3']}")
    print(f"  final spectra: {d['final_spectrum']}")
    for k, v in finals.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()