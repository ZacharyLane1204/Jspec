#!/usr/bin/env python3
"""
prepare_mrs_pfpc_epochs.py
============================

Two things MRS-PFPC's pfpc_proc doesn't do on its own:

  1. Download uncal files only if they're not already on disk.
  2. Keep separate epochs (separate JWST observations, taken on different
     dates) out of the same Level-3 association.

pfpc_proc has no stage-level entry point to hook epoch-aware association
building into partway through -- it runs Detector1 + Spec2 + Spec3 as one
call and names its Level-3 products after whatever subdirectory you point
it at. So rather than patching pfpc_proc's internals (fragile -- breaks on
every PFPC update), this script:

  - downloads uncal files for a proposal only if they aren't already
    present (checked by filename, not re-downloaded blindly every time),
  - groups them into epochs by MJD-OBS/EXPSTART (reusing
    combine_nirspec_orders.get_mjd / group_files_by_epoch -- the same
    epoch logic your other scripts already use), and
  - symlinks (default) or copies each epoch's uncal files into its own
    subdirectory, named so pfpc_proc's own product-naming falls out
    cleanly separated per epoch.

You still run pfpc_proc / pfpc_cor yourself, once per printed subdirectory
-- this script only prepares the inputs. That's deliberate: a JWST Stage
1-3 run is expensive and worth watching per epoch, not auto-launched
silently in a loop.

Usage
-----
    # Download only what's missing, then split into epoch subdirectories:
    python prepare_mrs_pfpc_epochs.py --proposal 9233 --workdir SN2024aecx/

    # Already have the uncal files locally (skip download, just split):
    python prepare_mrs_pfpc_epochs.py --uncal-dir SN2024aecx/SN2024aecx_uncal \\
        --workdir SN2024aecx/

    # Copy instead of symlink (e.g. filesystem doesn't support symlinks):
    python prepare_mrs_pfpc_epochs.py --uncal-dir SN2024aecx/SN2024aecx_uncal \\
        --workdir SN2024aecx/ --copy
"""
import argparse
import glob
import os
import re
import shutil

from astropy.io import fits

from jspec.pipelines.combine_nirspec_orders import get_mjd, group_files_by_epoch, mjd_to_datestr


def uncal_exists(outdir, proposal_padded):
    """Existing uncal files for this proposal already in outdir, if any."""
    pattern = os.path.join(outdir, f"jw{proposal_padded}*mirifu*_uncal.fits")
    return sorted(glob.glob(pattern))


def download_if_missing(proposal, outdir, dataset_ids=None):
    """Download uncal files for `proposal` into `outdir`, unless matching
    files are already there. Reuses the same download_uncal() call
    miri_mrs_auto_pipeline.query_and_download() makes -- see
    download_target_uncal.py for the plain wrapper."""
    proposal_padded = f"{proposal:05d}"
    existing = uncal_exists(outdir, proposal_padded)
    if existing:
        print(f"{len(existing)} uncal file(s) for proposal {proposal_padded} "
              f"already in {outdir} -- skipping download.")
        return existing

    from jspec.pipelines.mast_uncal_download import download_uncal
    os.makedirs(outdir, exist_ok=True)
    files = download_uncal(["MIRI/IFU"], proposal_padded, proposal, outdir,
                            name_contains=("mirifu",), dataset_ids=dataset_ids)
    print(f"Downloaded {len(files)} uncal file(s) to {outdir}")
    return files


def get_target(path):
    """TARGNAME/TARGPROP from a FITS primary header, path-safe."""
    with fits.open(path) as h:
        hdr = h[0].header
        name = hdr.get("TARGNAME") or hdr.get("TARGPROP") or "target"
    return re.sub(r"[^A-Za-z0-9._+-]+", "", str(name))


def split_by_epoch(uncal_files, workdir, max_gap_days=1.0, copy=False):
    """Cluster `uncal_files` into epochs (MJD-based, group_files_by_epoch)
    and link/copy each cluster into its own subdirectory under `workdir`,
    named "<target>_<date>_uncal". Returns a list of dicts: {label,
    subdir, date, files}, time-ordered."""
    if not uncal_files:
        raise SystemExit("No uncal files given to split.")

    clusters = group_files_by_epoch(uncal_files, max_gap_days=max_gap_days)
    target = get_target(uncal_files[0])
    print(f"\n{len(uncal_files)} uncal file(s) -> {len(clusters)} epoch(s) "
          f"for target {target!r} (>{max_gap_days} day gap = new epoch)")

    results = []
    for cluster in clusters:
        mjds = [m for m in (get_mjd(f) for f in cluster) if m is not None]
        date = mjd_to_datestr(sum(mjds) / len(mjds)) if mjds else "unknown-date"
        label = f"{target}_{date}"
        subdir = os.path.join(workdir, f"{label}_uncal")
        os.makedirs(subdir, exist_ok=True)

        for f in cluster:
            dst = os.path.join(subdir, os.path.basename(f))
            if os.path.exists(dst) or os.path.islink(dst):
                continue
            if copy:
                shutil.copy2(f, dst)
            else:
                os.symlink(os.path.abspath(f), dst)

        print(f"  {label}: {len(cluster)} file(s) -> {subdir}")
        results.append({"label": label, "subdir": subdir, "date": date,
                         "files": cluster})

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--proposal", type=int,
                      help="Proposal ID to download (if not already present)")
    src.add_argument("--uncal-dir",
                      help="Directory of uncal files already on disk (skip download)")
    ap.add_argument("--workdir", required=True,
                     help="Parent directory to create epoch subdirectories in "
                          "(and to download into, if --proposal is used)")
    ap.add_argument("--max-gap-days", type=float, default=1.0)
    ap.add_argument("--dataset", nargs="*", default=None,
                     help="With --proposal: only these dataset IDs")
    ap.add_argument("--copy", action="store_true",
                     help="Copy files into epoch subdirectories instead of symlinking")
    args = ap.parse_args()

    if args.proposal is not None:
        dl_dir = os.path.join(args.workdir, "_downloaded_uncal")
        uncal_files = download_if_missing(args.proposal, dl_dir, dataset_ids=args.dataset)
    else:
        uncal_files = sorted(glob.glob(os.path.join(args.uncal_dir, "*mirifu*uncal.fits")))
        if not uncal_files:
            raise SystemExit(f"No *mirifu*uncal.fits files found in {args.uncal_dir}")

    results = split_by_epoch(uncal_files, args.workdir,
                              max_gap_days=args.max_gap_days, copy=args.copy)

    print("\nRun pfpc_proc separately for each epoch, e.g.:")
    print(f"  cd {args.workdir}")
    for r in results:
        print(f"  pfpc_proc {os.path.basename(r['subdir'])}")


if __name__ == "__main__":
    main()