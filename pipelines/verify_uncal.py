#!/usr/bin/env python3
"""
verify_uncal.py
===============

Check downloaded JWST FITS files for truncation and other corruption,
*before* handing them to a pipeline run that takes hours.

Why this exists
---------------

A download that is interrupted partway through a file leaves a FITS that
looks fine to every cheap test. The primary header is the first few kB, so
``fits.getheader(path, 0)`` succeeds, the file opens, and the filename is
right. What is missing is everything after it -- including the ASDF
extension that every JWST datamodel reads its metadata from.

The JWST pipeline then fails like this, typically well into a run::

    File ".../stdatamodels/jwst/datamodels/util.py", line 511, in read_metadata
      bs = io.BytesIO(hdulist["ASDF"].data.tobytes())
    KeyError: "Extension 'ASDF' not found."

preceded, many lines earlier and easy to scroll past, by::

    AstropyUserWarning: File may have been truncated: actual file length
    (7208960) is smaller than the expected size (253647360)

That is the real error, demoted to a warning. This script promotes it back,
and tells you which file is bad and how to replace it.

What it checks
--------------

For each file: non-zero length; every extension header parses; the end of
the last HDU implied by its headers lies inside the file; and an ASDF
extension exists. Headers only -- no pixel data is read, so a few hundred
files take seconds.

Usage
-----
    # Check a whole target tree (follows symlinks into epoch directories):
    python -m jspec.pipelines.verify_uncal SN2024aecx

    # Check one directory:
    python -m jspec.pipelines.verify_uncal SN2024aecx/_downloaded_uncal

    # Delete the bad ones so the next download re-fetches them:
    python -m jspec.pipelines.verify_uncal SN2024aecx --delete

    # Check calibrated products too, not just *_uncal.fits:
    python -m jspec.pipelines.verify_uncal SN2024aecx --pattern '*.fits'

Exit status is 0 if everything is complete, 1 if anything is not, so this
can gate a longer run::

    python -m jspec.pipelines.verify_uncal SN2024aecx && pfpc_proc ...
"""
import argparse
import glob
import os

from jspec.pipelines.mast_uncal_download import fits_problem


def find_files(root, pattern="*_uncal.fits"):
    """Matching files under `root`, de-duplicated by what they point at.

    Epoch directories are full of symlinks back into ``_downloaded_uncal``
    (and the same file is linked from both the epoch directory and its
    ``uncal/`` subdirectory), so a naive walk reports the same bad download
    three or four times under different names. Resolve and de-duplicate, and
    keep the shortest path as the one to show.
    """
    if os.path.isfile(root):
        return [root]
    hits = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    by_target = {}
    for p in hits:
        real = os.path.realpath(p)
        prev = by_target.get(real)
        # Prefer the real file over any symlink to it: that is the copy you
        # actually have to delete or re-download, so it is the one worth
        # printing. Among symlinks, the shortest path is the least noisy.
        # (Test islink rather than comparing against realpath -- on macOS
        # /tmp is itself a symlink, so the comparison misfires there.)
        if prev is None:
            by_target[real] = p
        elif os.path.islink(prev) and (not os.path.islink(p) or len(p) < len(prev)):
            by_target[real] = p
    return sorted(by_target.values())


def verify(paths, verbose=False):
    """Check each path. Returns the list of (path, reason) for bad files."""
    bad = []
    for p in paths:
        why = fits_problem(p)
        if why is None:
            if verbose:
                print(f"  ok   {os.path.basename(p)}")
        else:
            bad.append((p, why))
    return bad


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="+",
                     help="File(s) or directory tree(s) to check")
    ap.add_argument("--pattern", default="*_uncal.fits",
                     help="Filename pattern to check (default: %(default)s)")
    ap.add_argument("--delete", action="store_true",
                     help="Delete incomplete files so the next download "
                          "re-fetches them. Symlinks pointing at a deleted "
                          "file are removed too.")
    ap.add_argument("-v", "--verbose", action="store_true",
                     help="List good files as well as bad ones")
    args = ap.parse_args()

    files = []
    for root in args.path:
        files.extend(find_files(root, args.pattern))
    if not files:
        raise SystemExit(f"No files matching {args.pattern!r} under {args.path}")

    print(f"Checking {len(files)} file(s) matching {args.pattern!r} ...")
    bad = verify(files, verbose=args.verbose)

    if not bad:
        print(f"\nAll {len(files)} file(s) complete.")
        return

    print(f"\n{len(bad)} of {len(files)} file(s) are incomplete:\n")
    proposals = set()
    for p, why in bad:
        print(f"  {p}")
        print(f"      {why}")
        name = os.path.basename(p)
        if name.startswith("jw") and len(name) > 7:
            proposals.add(name[2:7])

    if args.delete:
        n = 0
        for p, _ in bad:
            real = os.path.realpath(p)
            # Remove the symlinks first, then the file they share, so no
            # epoch directory is left pointing at something that is gone.
            for link in find_files_pointing_at(args.path, real):
                os.remove(link)
                n += 1
            if os.path.exists(real):
                os.remove(real)
                n += 1
        print(f"\nDeleted {n} path(s).")
        if proposals:
            print("Re-download with:")
            for prop in sorted(proposals):
                print(f"  python -m jspec.pipelines.prepare_mrs_pfpc_epochs "
                      f"--proposal {int(prop)} --outdir <outdir>")
            print("\n(Complete files already on disk are verified and kept, so "
                  "this only re-fetches what is missing.)")
    else:
        print("\nRe-run with --delete to remove them, then re-download with")
        print("prepare_mrs_pfpc_epochs --proposal <id> --outdir <outdir>, which")
        print("verifies what is already there and fetches only what is missing.")

    raise SystemExit(1)


def find_files_pointing_at(roots, real_target):
    """Symlinks under `roots` resolving to `real_target`."""
    links = []
    for root in roots:
        if os.path.isfile(root):
            candidates = [root]
        else:
            candidates = glob.glob(os.path.join(root, "**", "*.fits"), recursive=True)
        for p in candidates:
            if os.path.islink(p) and os.path.realpath(p) == real_target:
                links.append(p)
    return links


if __name__ == "__main__":
    main()
