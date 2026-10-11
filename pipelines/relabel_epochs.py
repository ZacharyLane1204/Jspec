"""
Rename existing epoch directories and products onto the current layout.

Why this exists
---------------
The NIRSpec-FS and MIRI-LRS routes used to label epochs ``YYYYMMDD`` and
derive the tag from each epoch's own ``TARGNAME``, while the MRS route
used ``YYYY-MM-DD``. One target reduced over several visits therefore
produced directories like::

    NIRSpec-FS/2024aecx_20250601/
    NIRSpec-FS/2024aecx_20251227/
    NIRSpec-FS/sn2024aecx_20250116/
    MIRI-MRS/SN2024aecx_2025-06-01/

which are four spellings of two things. Worse, ``jspec.layout``
recognises an epoch directory by a trailing ``YYYY-MM-DD``, so the
compact-date directories were invisible to ``find_epoch_dirs`` -- the
product index rebuild silently skipped every NIRSpec-FS and MIRI-LRS
epoch in the tree.

Both pipelines now write ``<tag>_<YYYY-MM-DD>``. This tool brings a tree
reduced with the older code onto that layout **without re-running the
pipeline**, which matters because Stage 1-3 is hours of CPU per epoch.

What it renames
---------------
* The epoch directory itself.
* Any file beneath it whose name contains *both* a recognisable date
  token and the old tag. The tokens are removed and the name is rebuilt
  as ``<newtag>_<newdate>_<whatever was left>``, so::

      2024aecx_s400a1_20250601_final.ecsv
          -> SN2024aecx_2025-06-01_s400a1_final.ecsv
      f290lp_g395m_s400a1_2024aecx_20250601_x1d.fits
          -> SN2024aecx_2025-06-01_f290lp_g395m_s400a1_x1d.fits

  which is byte-for-byte what the current pipelines would have written.
  That equality is the point: a later ``--from-stage`` re-run finds the
  renamed Stage-3 products instead of rebuilding them.

* Files carrying neither token -- ``jw09258...._rate.fits`` and friends
  -- are left alone. They are named by the JWST pipeline, not by us.

It also rewrites ``products.csv`` paths and rebuilds the index.

Usage
-----
Preview (the default -- nothing is written)::

    python -m jspec.pipelines.relabel_epochs --outdir SN2024aecx --tag SN2024aecx

Apply it::

    python -m jspec.pipelines.relabel_epochs --outdir SN2024aecx \\
        --tag SN2024aecx --apply

Restrict to one instrument::

    python -m jspec.pipelines.relabel_epochs --outdir SN2024aecx \\
        --tag SN2024aecx --instrument NIRSpec-FS --apply

Caveat
------
Association JSON files under ``asn/`` embed absolute paths and the old
product name. They are rewritten too, but an association is cheap to
regenerate; if one looks wrong, delete it and re-run from Stage 3.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from jspec.layout import (canon_object, canonical_tag, epoch_dirname,
                          instrument_label)

__all__ = ["plan_renames", "split_tokens", "new_basename", "main"]

#: ``20250601`` or ``2025-06-01`` as a whole underscore-delimited token.
_DATE_TOKEN = re.compile(r"^(\d{4})-?(\d{2})-?(\d{2})$")
#: Trailing date on a directory name, either spelling.
_DIR_DATE = re.compile(r"[_-]((?:\d{4}-\d{2}-\d{2})|(?:\d{8}))$")


def _norm_date(token: str) -> str | None:
    """``20250601`` or ``2025-06-01`` -> ``2025-06-01``; None if not a date."""
    m = _DATE_TOKEN.match(str(token))
    if not m:
        return None
    y, mo, d = m.groups()
    # Reject obvious non-dates (a detector ID, an exposure counter) rather
    # than silently renaming a file around a number that is not a date.
    if not (1990 <= int(y) <= 2100 and 1 <= int(mo) <= 12 and 1 <= int(d) <= 31):
        return None
    return f"{y}-{mo}-{d}"


def parse_old_dirname(name: str):
    """``<tag>_<date>`` in either date spelling -> ``(tag, 'YYYY-MM-DD')``.

    Returns ``(name, None)`` when the directory does not end in a date, so
    a caller can skip non-epoch directories without raising.
    """
    base = os.path.basename(str(name).rstrip(os.sep))
    m = _DIR_DATE.search(base)
    if not m:
        return base, None
    date = _norm_date(m.group(1))
    if date is None:
        return base, None
    return base[: m.start()], date


def split_tokens(basename: str, old_tag: str):
    """Split a filename into ``(kept_tokens, date, extension)``.

    Returns ``(None, None, None)`` unless the name contains both a date
    token and a token matching `old_tag`. Requiring *both* is what keeps
    ``jw09258010001_03101_00003_nrs1_rate.fits`` untouched: it has no tag
    token, and its ``03101`` is not a plausible date.
    """
    stem, ext = os.path.splitext(basename)
    # Compound extensions such as ".ecsv" are single, but a few products
    # use ".fits.gz"; keep whatever trails the last date/tag-free token.
    while True:
        stem2, ext2 = os.path.splitext(stem)
        if ext2 and ext2.lower() in (".fits", ".tar"):
            stem, ext = stem2, ext2 + ext
            continue
        break

    tokens = stem.split("_")
    key = canon_object(old_tag)
    date = None
    kept = []
    seen_tag = False
    for tok in tokens:
        d = _norm_date(tok)
        if d is not None and date is None:
            date = d
            continue
        if not seen_tag and canon_object(tok) == key and key:
            seen_tag = True
            continue
        kept.append(tok)
    if date is None or not seen_tag:
        return None, None, None
    return kept, date, ext


def new_basename(basename: str, old_tag: str, new_tag: str) -> str | None:
    """Rebuilt filename, or None if this file should not be renamed."""
    kept, date, ext = split_tokens(basename, old_tag)
    if kept is None:
        return None
    parts = [epoch_dirname(new_tag, date)] + [t for t in kept if t]
    return "_".join(parts) + ext


def plan_renames(outdir: str, instrument: str | None = None,
                 override_tag: str | None = None):
    """Work out every rename without performing any of them.

    Returns ``(dir_moves, file_moves, notes)``. `dir_moves` is a list of
    ``(old_path, new_path)`` for epoch directories; `file_moves` is the
    same for files, expressed in terms of the *old* directory so the
    caller can rename files first and directories afterwards.
    """
    dir_moves, file_moves, notes = [], [], []
    wanted = instrument_label(instrument) if instrument else None

    for inst_name in sorted(os.listdir(outdir)):
        inst_path = os.path.join(outdir, inst_name)
        if not os.path.isdir(inst_path) or inst_name.startswith("_"):
            continue
        if wanted is not None and inst_name != wanted:
            continue

        entries = []
        for epoch_name in sorted(os.listdir(inst_path)):
            epoch_path = os.path.join(inst_path, epoch_name)
            if not os.path.isdir(epoch_path):
                continue
            tag, date = parse_old_dirname(epoch_name)
            if date is None:
                notes.append(f"{inst_name}/{epoch_name}: no date in the name, skipped")
                continue
            entries.append((epoch_name, epoch_path, tag, date))

        if not entries:
            continue
        # Resolve one tag per instrument from every spelling present, so
        # the three NIRSpec directories collapse onto one name rather
        # than each being normalised independently.
        new_tag = canonical_tag([t for _, _, t, _ in entries],
                                override=override_tag)

        for epoch_name, epoch_path, tag, date in entries:
            new_name = epoch_dirname(new_tag, date)
            for root, _dirs, files in os.walk(epoch_path):
                for fn in sorted(files):
                    nb = new_basename(fn, tag, new_tag)
                    if nb and nb != fn:
                        file_moves.append((os.path.join(root, fn),
                                           os.path.join(root, nb)))
            if new_name != epoch_name:
                dir_moves.append((epoch_path,
                                  os.path.join(inst_path, new_name)))
    return dir_moves, file_moves, notes


def _rewrite_asn(path: str, replacements):
    """Replace old stems inside an association JSON, in place.

    Associations are plain JSON with the product name and member paths as
    strings, so a textual substitution is both sufficient and far less
    fragile than reconstructing the association from the rules.
    """
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return False
    out = text
    for old, new in replacements:
        out = out.replace(old, new)
    if out == text:
        return False
    try:
        json.loads(out)                      # refuse to write damaged JSON
    except Exception:                        # noqa: BLE001
        return False
    with open(path, "w") as fh:
        fh.write(out)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Rename legacy epoch directories and products onto the "
                    "current <tag>_<YYYY-MM-DD> layout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Runs as a preview unless --apply is given.")
    ap.add_argument("--outdir", required=True,
                    help="Target output directory, the one holding "
                         "<INSTRUMENT>/ and products.csv")
    ap.add_argument("--tag", default=None,
                    help="Tag to use for every epoch, e.g. SN2024aecx. "
                         "Without it the most specific spelling already "
                         "present is chosen, per instrument.")
    ap.add_argument("--instrument", default=None,
                    help="Restrict to one instrument directory, e.g. NIRSpec-FS")
    ap.add_argument("--apply", action="store_true",
                    help="Actually rename. Without this nothing is written.")
    args = ap.parse_args(argv)

    outdir = os.path.abspath(args.outdir)
    if not os.path.isdir(outdir):
        sys.exit(f"--outdir: no such directory {args.outdir!r}")

    dir_moves, file_moves, notes = plan_renames(
        outdir, args.instrument, args.tag)

    for n in notes:
        print(f"note: {n}")
    if not dir_moves and not file_moves:
        print("Nothing to rename: this tree already matches the current layout.")
        return 0

    print(f"\n{len(file_moves)} file(s) and {len(dir_moves)} directory(ies) to rename:\n")
    for old, new in dir_moves:
        print(f"  DIR  {os.path.relpath(old, outdir)}"
              f"\n    -> {os.path.relpath(new, outdir)}")
    shown = file_moves if len(file_moves) <= 40 else file_moves[:40]
    for old, new in shown:
        print(f"  FILE {os.path.relpath(old, outdir)}"
              f"\n    -> {os.path.basename(new)}")
    if len(file_moves) > len(shown):
        print(f"  ... and {len(file_moves) - len(shown)} more")

    # Refuse to clobber. A destination that already exists means the tree
    # has been half-migrated or holds two real epochs that normalise to
    # the same name; either way, stopping is better than merging them.
    clashes = [new for _old, new in dir_moves + file_moves if os.path.exists(new)]
    if clashes:
        print("\nABORT: these destinations already exist:")
        for c in clashes[:20]:
            print(f"  {os.path.relpath(c, outdir)}")
        print("Move or delete them first.")
        return 1

    if not args.apply:
        print("\nPreview only. Re-run with --apply to perform the renames.")
        return 0

    replacements = []
    for old, new in file_moves:
        os.rename(old, new)
        o, n = os.path.splitext(os.path.basename(old))[0], os.path.splitext(os.path.basename(new))[0]
        if (o, n) not in replacements:
            replacements.append((o, n))
    for old, new in dir_moves:
        os.rename(old, new)
        replacements.append((os.path.basename(old), os.path.basename(new)))
    print(f"\nRenamed {len(file_moves)} file(s), {len(dir_moves)} directory(ies).")

    n_asn = 0
    for root, _dirs, files in os.walk(outdir):
        for fn in files:
            if fn.endswith(".json") and os.path.basename(root) == "asn":
                n_asn += _rewrite_asn(os.path.join(root, fn), replacements)
    if n_asn:
        print(f"Updated {n_asn} association file(s).")

    try:
        from jspec.manifest import rebuild_manifest
        rebuild_manifest(outdir)
    except Exception as exc:                            # noqa: BLE001
        print(f"products.csv not rebuilt ({exc}); run 'jspec products "
              f"{args.outdir} --rebuild' yourself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
