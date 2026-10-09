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

from jspec.layout import ensure_epoch_tree
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


def split_by_epoch(uncal_files, outdir, max_gap_days=1.0, copy=False, tag=None):
    """Cluster `uncal_files` into epochs and lay out one directory each.

    The directories follow the standard jspec layout
    (<outdir>/MIRI-MRS/<tag>_<date>/), but with one concession to how
    MRS-PFPC works: the uncal files are placed *flat* in the epoch
    directory as well as in its uncal/ subdirectory.

    That is not redundancy for its own sake. pfpc_proc globs
    "<objname>/jw*mirifu*uncal.fits" relative to the current directory
    and writes every output back into that same directory, naming each
    Level-3 product after it ("<objname>_level3_ch1-short_s3d.fits").
    Pointing it at the epoch directory is therefore the only way to get
    products named <tag>_<date>_level3_*, which is what the rest of the
    layout expects. Afterwards, import_pfpc_products sorts the flat dump
    into stage1/stage2/stage3/pfpc.

    Returns a list of time-ordered dicts: {label, subdir, date, tag,
    paths, files}.
    """
    if not uncal_files:
        raise SystemExit("No uncal files given to split.")

    clusters = group_files_by_epoch(uncal_files, max_gap_days=max_gap_days)
    target = tag or get_target(uncal_files[0])
    print(f"\n{len(uncal_files)} uncal file(s) -> {len(clusters)} epoch(s) "
          f"for target {target!r} (>{max_gap_days} day gap = new epoch)")

    results = []
    for cluster in clusters:
        mjds = [m for m in (get_mjd(f) for f in cluster) if m is not None]
        date = mjd_to_datestr(sum(mjds) / len(mjds)) if mjds else "unknown-date"
        paths = ensure_epoch_tree(outdir, "MIRI-MRS", target, date)
        label = paths["stem"]
        subdir = paths["epoch"]

        for f in cluster:
            for dst_dir in (subdir, paths["uncal"]):
                dst = os.path.join(dst_dir, os.path.basename(f))
                if os.path.exists(dst) or os.path.islink(dst):
                    continue
                if copy and dst_dir is subdir:
                    shutil.copy2(f, dst)
                else:
                    os.symlink(os.path.abspath(f), dst)

        print(f"  {label}: {len(cluster)} file(s) -> "
              f"{os.path.relpath(subdir, outdir)}")
        results.append({"label": label, "subdir": subdir, "date": date,
                         "tag": target, "paths": paths, "files": cluster})

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--proposal", type=int,
                      help="Proposal ID to download (if not already present)")
    src.add_argument("--uncal-dir",
                      help="Directory of uncal files already on disk (skip download)")
    ap.add_argument("--outdir", "--workdir", dest="outdir", required=True,
                     help="Output directory for this target. Epoch directories "
                          "are created at <outdir>/MIRI-MRS/<target>_<date>/.")
    ap.add_argument("--tag", default=None,
                     help="Override the target name used in directory names")
    ap.add_argument("--max-gap-days", type=float, default=1.0)
    ap.add_argument("--dataset", nargs="*", default=None,
                     help="With --proposal: only these dataset IDs")
    ap.add_argument("--copy", action="store_true",
                     help="Copy files into epoch subdirectories instead of symlinking")
    args = ap.parse_args()

    if args.proposal is not None:
        dl_dir = os.path.join(args.outdir, "_downloaded_uncal")
        uncal_files = download_if_missing(args.proposal, dl_dir, dataset_ids=args.dataset)
    else:
        uncal_files = sorted(glob.glob(os.path.join(args.uncal_dir, "*mirifu*uncal.fits")))
        if not uncal_files:
            raise SystemExit(f"No *mirifu*uncal.fits files found in {args.uncal_dir}")

    results = split_by_epoch(uncal_files, args.outdir,
                              max_gap_days=args.max_gap_days, copy=args.copy,
                              tag=args.tag)

    mrs_dir = os.path.dirname(results[0]["subdir"])
    first = results[0]["label"]

    # pfpc_proc resolves its argument as "<objname>/" relative to the
    # current directory, so the cd is required, not cosmetic -- and the
    # argument must be a bare directory name. Passing a path puts a "/"
    # inside the Level-3 product name, and the run then fails at the save
    # step, after Detector1, Spec2 and Spec3 have all completed.
    print("\nNext: run MRS-PFPC once per epoch (in your jspec or mrs_pfpc env).")
    print(f"  cd {mrs_dir}")
    for r in results:
        print(f"  pfpc_proc {r['label']} && pfpc_cor {r['label']}")
    print("\npfpc_proc runs Detector1+Spec2+Spec3 and writes its outputs back")
    print("into the epoch directory; pfpc_cor then applies the point-source")
    print("fringe/photometric correction, giving:")
    print(f"  {first}/{first}_pfpc_ch*-*_x1d.fits")

    print("\nThen sort those outputs into the standard layout:")
    print(f"  python -m jspec.pipelines.import_pfpc_products --outdir {args.outdir}")
    print("\nand re-extract with AstroBkgInterp to get the final spectrum")
    print("(this is the recommended MIRI MRS product, and is what gets")
    print("recorded in products.csv):")
    print(f"  python -m jspec.pipelines.abi_reextract_pfpc --outdir {args.outdir}")


if __name__ == "__main__":
    main()