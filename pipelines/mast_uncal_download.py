#!/usr/bin/env python3
"""
mast_uncal_download.py
======================

Small shared helper for the proposal-level pipelines: list and download every
level-1 ``*_uncal.fits`` of a JWST proposal from MAST, and *tell you* when
something did not arrive.

What it adds over a bare ``query_criteria`` + ``filter_products`` +
``download_file`` loop:

  * every download is checked (file exists, size matches what MAST lists,
    FITS header readable); truncated files are removed and retried
    (``retries`` times) instead of being silently kept;
  * failures are reported at the end, with the data-rights status, so an
    exclusive-access file is recognisable (set ``MAST_API_TOKEN`` to fetch
    those);
  * a manifest of everything MAST lists for the proposal is written, and the
    per-observation uncal counts are printed, so "MAST does not have it" and
    "the download missed it" can be told apart;
  * restart-safe: complete files are skipped on a re-run.

Selection is by file name (``*_uncal.fits`` containing ``jw<proposal>``, plus
any ``name_contains`` substrings such as ``mirifu``) -- there is deliberately
NO size-based cut.
"""
from __future__ import annotations

import glob
import os
import time
from collections import defaultdict

from astropy.io import fits
from astroquery.mast import Observations

MAST_CHUNK = 25
MAST_RETRIES = 3


def mast_login():
    """Optional: ``export MAST_API_TOKEN=<token>`` (https://auth.mast.stsci.edu/token)
    to download data still under exclusive access."""
    token = os.getenv("MAST_API_TOKEN")
    if token:
        try:
            Observations.login(token=token)
            print("  Logged in to MAST with MAST_API_TOKEN.")
        except Exception as exc:                                    # noqa: BLE001
            print(f"  WARNING: MAST login failed ({exc}); exclusive-access data will not download.")


def list_uncal(instrument_names, proposal_padded, proposal_num, name_contains=(), manifest_path=None):
    """All matching level-1 uncal files MAST lists, as a sorted list of dicts
    (name, uri, size, rights, obs_id)."""
    print(f"Querying MAST ({', '.join(instrument_names)}) for proposal {proposal_padded} ...")
    obs_table = Observations.query_criteria(
        obs_collection="JWST", instrument_name=list(instrument_names),
        provenance_name=["CALJWST"], proposal_id=[str(proposal_num)])
    print(f"  Found {len(obs_table)} observation entries.")

    found = {}
    for i in range(0, len(obs_table), MAST_CHUNK):
        chunk = obs_table[i:i + MAST_CHUNK]
        try:
            products = Observations.get_product_list(chunk)
        except Exception as exc:                                    # noqa: BLE001
            print(f"  WARNING: product list for observations {i}-{i + len(chunk) - 1} failed ({exc}); "
                  f"retrying one by one.")
            products = []
            for row in chunk:
                try:
                    products.extend(list(Observations.get_product_list(row)))
                except Exception as exc2:                           # noqa: BLE001
                    print(f"  WARNING: could not list products for {row['obs_id']}: {exc2}")
        cols = set(getattr(products, "colnames", []) or (products[0].colnames if len(products) else []))
        for p in products:
            uri = str(p["dataURI"])
            name = str(p["productFilename"]) if "productFilename" in cols else os.path.basename(uri)
            if not name.endswith("_uncal.fits") or f"jw{proposal_padded}" not in name:
                continue
            if any(s not in name for s in name_contains):
                continue
            try:
                size = int(p["size"])
            except Exception:                                       # noqa: BLE001
                size = 0
            found[name] = dict(name=name, uri=uri, size=size,
                               rights=str(p["dataRights"]) if "dataRights" in cols else "?",
                               obs_id=str(p["obs_id"]) if "obs_id" in cols else "?")
    entries = [found[k] for k in sorted(found)]

    per_obs = defaultdict(int)
    for e in entries:
        per_obs[e["obs_id"]] += 1
    print(f"  {len(entries)} uncal file(s) in {len(per_obs)} observation(s):")
    for row in obs_table:
        oid = str(row["obs_id"])
        extra = "  ".join(f"{c}={row[c]}" for c in ("target_name", "filters") if c in obs_table.colnames)
        print(f"    {oid}: {per_obs.get(oid, 0)} uncal   {extra}")
    n_prop = sum(1 for e in entries if e["rights"].upper() not in ("PUBLIC", "?"))
    if n_prop:
        print(f"  NOTE: {n_prop} file(s) are under exclusive access -- downloading them needs MAST_API_TOKEN "
              f"(and rights to the program).")
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
        return False                                    # truncated
    try:
        fits.getheader(path, 0)
    except Exception:                                   # noqa: BLE001
        return False
    return True


def download_uncal(instrument_names, proposal_padded, proposal_num, uncal_dir,
                   name_contains=(), dataset_ids=None, manifest_path=None, retries=MAST_RETRIES):
    """Download every matching uncal file (restart-safe, verified). Returns the
    sorted list of ``*uncal.fits`` present in ``uncal_dir``."""
    os.makedirs(uncal_dir, exist_ok=True)
    mast_login()
    if manifest_path is None:
        manifest_path = os.path.join(os.path.dirname(os.path.abspath(uncal_dir)), "mast_uncal_manifest.txt")
    entries = list_uncal(instrument_names, proposal_padded, proposal_num, name_contains, manifest_path)

    if dataset_ids:
        before = len(entries)
        entries = [e for e in entries if any(d in e["name"] for d in dataset_ids)]
        print(f"  {before} uncal file(s) in the proposal; {len(entries)} match --dataset {dataset_ids}.")
        missing = [d for d in dataset_ids if not any(d in e["name"] for e in entries)]
        if missing:
            print(f"  WARNING: no matching files for --dataset entries: {missing}")

    failed = []
    for i, e in enumerate(entries, 1):
        dest = os.path.join(uncal_dir, e["name"])
        if _complete(dest, e["size"]):
            continue
        for attempt in range(1, retries + 1):
            print(f"  [{i}/{len(entries)}] downloading {e['name']} ({e['size'] / 1e6:.0f} MB)"
                  + (f"  attempt {attempt}" if attempt > 1 else ""))
            status = None
            try:
                res = Observations.download_file(e["uri"], local_path=dest)
                status = res[0] if isinstance(res, (tuple, list)) else res
            except Exception as exc:                                # noqa: BLE001
                status = f"exception: {exc}"
            if _complete(dest, e["size"]):
                break
            print(f"    not complete ({status})")
            if os.path.exists(dest):
                os.remove(dest)
            time.sleep(3 * attempt)
        else:
            failed.append(e)

    print(f"  {len(entries) - len(failed)}/{len(entries)} uncal files present in {uncal_dir}")
    if failed:
        print("  !! FAILED downloads (re-run to retry; rights problems need MAST_API_TOKEN):")
        for e in failed:
            print(f"     {e['name']}  rights={e['rights']}")

    files = sorted(glob.glob(os.path.join(uncal_dir, "*uncal.fits")))
    if dataset_ids:
        files = [f for f in files if any(d in os.path.basename(f) for d in dataset_ids)]
    return files