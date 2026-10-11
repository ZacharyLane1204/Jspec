#!/usr/bin/env python3
"""
mast_uncal_download.py
======================

Small shared helper for the proposal-level pipelines: list and download every
level-1 ``*_uncal.fits`` of a JWST proposal from MAST, and *tell you* when
something did not arrive.

What it adds over a bare ``query_criteria`` + ``filter_products`` +
``download_file`` loop:

  * every download is checked -- not just "does the primary header read",
    which a 3%-complete file passes, but every extension header, the size
    MAST advertises, and the presence of the ASDF extension the JWST
    datamodels need (see ``fits_problem``). Truncated files are removed
    and retried (``retries`` times) instead of being silently kept, and
    the same check is re-applied to files already on disk, so an
    interrupted earlier run self-heals on the next invocation;
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
import warnings
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


def fits_problem(path, expected_size=0):
    """Why `path` is not a usable FITS file, or None if it looks complete.

    A truncated download is the failure this is really guarding against,
    and it is easy to miss: the *primary* header sits in the first few kB,
    so ``fits.getheader(path, 0)`` happily succeeds on a file that stopped
    downloading at 3% and is missing every extension after it. The JWST
    pipeline then dies much later and much less clearly, with
    ``KeyError: "Extension 'ASDF' not found."`` out of
    ``stdatamodels.read_metadata`` -- after CRDS lookups, and after you
    have already waited for pfpc_proc to get that far.

    So check four things, cheapest first:

      1. non-zero length, and at least the size MAST advertises (when
         known -- `expected_size` is 0 if MAST's product list omitted it,
         which is exactly when the other checks have to carry the load);
      2. every *header* parses (``lazy_load_hdus=False`` + ``.info()``),
         which is where astropy raises its "File may have been truncated"
         warning -- promoted here from a warning to a failure;
      3. the end of the last HDU's data, as implied by its header, is
         actually within the file. On a complete JWST product these agree
         to the byte;
      4. an ASDF extension is present, since that is what every JWST
         datamodel reads its metadata from.

    Data is never loaded, so this stays fast enough to run over a whole
    uncal directory (headers only, ~ms per file).
    """
    if not os.path.exists(path):
        return "missing"
    size = os.path.getsize(path)
    if size == 0:
        return "zero length"
    if expected_size and size < 0.999 * expected_size:
        return (f"truncated: {size / 1e6:.1f} MB on disk, "
                f"MAST lists {expected_size / 1e6:.1f} MB")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with fits.open(path, memmap=False, lazy_load_hdus=False) as hdul:
                hdul.info(output=False)
                names = [h.name for h in hdul]
                last = hdul[-1]
                end = last._data_offset + last._data_size
        for w in caught:
            if "truncat" in str(w.message).lower():
                return f"truncated: {size / 1e6:.1f} MB on disk (astropy: file is short)"
    except Exception as exc:                                # noqa: BLE001
        return f"unreadable: {type(exc).__name__}: {exc}"
    if end > size:
        return (f"truncated: headers imply {end / 1e6:.1f} MB, "
                f"file is {size / 1e6:.1f} MB")
    if "ASDF" not in names:
        return "no ASDF extension (not a complete JWST product)"
    return None


def _complete(path, expected_size=0):
    """True if `path` is a complete, readable FITS file."""
    return fits_problem(path, expected_size) is None


def download_uncal(instrument_names, proposal_padded, proposal_num, uncal_dir,
                   name_contains=(), dataset_ids=None, manifest_path=None,
                   retries=MAST_RETRIES, force=False):
    """Download every matching uncal file (restart-safe, verified). Returns the
    sorted list of ``*uncal.fits`` present in ``uncal_dir``.

    Files already on disk are re-verified with `fits_problem`, not merely
    counted, so a truncated file left over from an interrupted run is
    re-fetched rather than reused. `force=True` re-downloads everything
    regardless, for the case where you suspect the local copies but cannot
    say which.
    """
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
    n_ok = n_redo = 0
    for i, e in enumerate(entries, 1):
        dest = os.path.join(uncal_dir, e["name"])
        if force and os.path.exists(dest):
            os.remove(dest)                             # --force: do not trust it
            n_redo += 1
        else:
            why = fits_problem(dest, e["size"])
            if why is None:
                n_ok += 1
                continue
            if why != "missing":
                # Report it: a bad local file is worth seeing, because it
                # explains any earlier failure that pointed at this file.
                print(f"  re-downloading {e['name']}: {why}")
                os.remove(dest)
                n_redo += 1
        for attempt in range(1, retries + 1):
            print(f"  [{i}/{len(entries)}] downloading {e['name']} ({e['size'] / 1e6:.0f} MB)"
                  + (f"  attempt {attempt}" if attempt > 1 else ""))
            status = None
            try:
                res = Observations.download_file(e["uri"], local_path=dest)
                status = res[0] if isinstance(res, (tuple, list)) else res
            except Exception as exc:                                # noqa: BLE001
                status = f"exception: {exc}"
            why = fits_problem(dest, e["size"])
            if why is None:
                break
            print(f"    not complete ({status}): {why}")
            if os.path.exists(dest):
                os.remove(dest)
            time.sleep(3 * attempt)
        else:
            failed.append(e)

    if n_ok:
        print(f"  {n_ok} file(s) already present and verified complete -- skipped.")
    if n_redo:
        print(f"  {n_redo} file(s) re-downloaded (incomplete locally or --force).")
    print(f"  {len(entries) - len(failed)}/{len(entries)} uncal files present in {uncal_dir}")
    if failed:
        print("  !! FAILED downloads (re-run to retry; rights problems need MAST_API_TOKEN):")
        for e in failed:
            print(f"     {e['name']}  rights={e['rights']}")

    files = sorted(glob.glob(os.path.join(uncal_dir, "*uncal.fits")))
    if dataset_ids:
        files = [f for f in files if any(d in os.path.basename(f) for d in dataset_ids)]
    return files