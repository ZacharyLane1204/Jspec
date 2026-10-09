"""
jspec.layout
============

One place that decides where a file goes and what it is called.

Before this module each pipeline invented its own answer: output roots
were ``fs_data_<proposal>``, ``nrs_ifu_data_<proposal>``,
``lrs_data_<proposal>``, ``mrs_data_<proposal>`` and so on; epochs were
sometimes a subdirectory, sometimes just part of a filename, sometimes
nothing at all. Finding "the final MIRI spectrum for the June visit"
meant knowing which script wrote it.

The layout
----------
Everything for one instrument and one epoch lives in one directory::

    <outdir>/
        products.csv                     <- the index (see manifest.py)
        MIRI-MRS/
            SN2024aecx_2025-06-01/
                uncal/  stage1/  stage2/  stage3/  reextract/  final/
            SN2024aecx_2025-12-07/
                ...
        NIRSpec-FS/
            SN2024aecx_2025-06-01/
                ...

The epoch directory name is ``<tag>_<date>``, where ``tag`` defaults to
the target name from the FITS headers and ``date`` is the ``YYYY-MM-DD``
of the epoch. This is deliberately the same shape MRS-PFPC uses, so the
two routes produce directories you can read the same way.

Because an epoch directory is self-contained -- its own ``uncal``
through its own ``final`` -- exposures from different visits cannot end
up in the same Stage-3 association by accident. That failure mode is
quiet and expensive when it happens, so it is worth the duplicated
directory structure.

File naming
-----------
Product files repeat the epoch stem, so a file is still identifiable
once it has been copied somewhere else::

    SN2024aecx_2025-06-01_level3_ch1-short_s3d.fits    (Stage 3 cube)
    SN2024aecx_2025-06-01_level3_ch1-short_x1d.fits    (Stage 3 spectrum)
    SN2024aecx_2025-06-01_abi_ch1-short_x1d.fits       (re-extraction)
    SN2024aecx_2025-06-01_abi_combined.ecsv            (final spectrum)

Again this follows MRS-PFPC's ``<objname>_level3_ch<N>-<band>_<suffix>``
convention rather than inventing a parallel one.
"""
from __future__ import annotations

import os
import re

__all__ = [
    "INSTRUMENT_LABELS", "STAGE_DIRS",
    "instrument_label", "safe_name", "epoch_dirname", "epoch_dir",
    "stage_dir", "ensure_epoch_tree", "product_basename", "product_path",
    "parse_epoch_dirname", "find_epoch_dirs",
    "stage_files", "find_staged_uncal",
]


#: Canonical directory name for each instrument/mode. Keys are matched
#: case-insensitively and ignoring separators, so "miri_mrs", "MIRI/MRS"
#: and "mrs" all resolve to "MIRI-MRS".
INSTRUMENT_LABELS = {
    "mrs": "MIRI-MRS",
    "mirimrs": "MIRI-MRS",
    "miriifu": "MIRI-MRS",
    "lrs": "MIRI-LRS",
    "mirilrs": "MIRI-LRS",
    "mirifs": "MIRI-LRS",
    "miriimaging": "MIRI-Imaging",
    "miriimage": "MIRI-Imaging",
    "nirspecfs": "NIRSpec-FS",
    "nrsfs": "NIRSpec-FS",
    "fs": "NIRSpec-FS",
    "nirspecifu": "NIRSpec-IFU",
    "nrsifu": "NIRSpec-IFU",
    "nirspecmos": "NIRSpec-MOS",
    "nircamimaging": "NIRCam-Imaging",
    "nircamimage": "NIRCam-Imaging",
}

#: Subdirectories created inside every epoch directory. "reextract" and
#: "final" are only used by the spectroscopic pipelines, but they are
#: cheap to create and their absence is more confusing than their
#: emptiness.
STAGE_DIRS = ("uncal", "stage1", "asn", "stage2", "stage3",
              "reextract", "final")

_SEP_RE = re.compile(r"[^a-z0-9]+")
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._+-]+")
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})$")


def instrument_label(instrument: str) -> str:
    """Canonical directory name for an instrument/mode.

    Accepts the loose spellings scattered through the pipelines
    ("MIRI/IFU", "nirspec_fs", "mrs") and returns one of the
    ``INSTRUMENT_LABELS`` values. An unrecognised name is passed through
    with separators tidied rather than rejected, so a new mode does not
    need this table updated before it can be used.
    """
    if not instrument:
        raise ValueError("instrument must be a non-empty string")
    key = _SEP_RE.sub("", str(instrument).lower())
    if key in INSTRUMENT_LABELS:
        return INSTRUMENT_LABELS[key]
    return _UNSAFE_RE.sub("-", str(instrument)).strip("-")


def safe_name(name: str, default: str = "target") -> str:
    """Strip characters that cause trouble in paths and product names.

    A product name containing "/" is the specific failure this guards
    against: it silently becomes a subdirectory at save time, and the
    pipeline only finds out after Stage 3 has finished.
    """
    cleaned = _UNSAFE_RE.sub("", str(name or ""))
    return cleaned or default


def epoch_dirname(tag: str, date: str) -> str:
    """``<tag>_<date>`` -- the epoch directory name and product stem."""
    tag = safe_name(tag)
    date = safe_name(date, default="unknown-date")
    return f"{tag}_{date}"


def epoch_dir(outdir: str, instrument: str, tag: str, date: str) -> str:
    """``<outdir>/<INSTRUMENT>/<tag>_<date>``."""
    return os.path.join(outdir, instrument_label(instrument),
                         epoch_dirname(tag, date))


def stage_dir(outdir: str, instrument: str, tag: str, date: str,
              stage: str) -> str:
    """A single stage directory inside the epoch directory."""
    if stage not in STAGE_DIRS:
        raise ValueError(f"Unknown stage {stage!r}; expected one of "
                          f"{', '.join(STAGE_DIRS)}")
    return os.path.join(epoch_dir(outdir, instrument, tag, date), stage)


def ensure_epoch_tree(outdir: str, instrument: str, tag: str, date: str,
                      stages=STAGE_DIRS) -> dict:
    """Create the epoch directory and its stage subdirectories.

    Returns a dict of stage name -> path, plus ``"epoch"`` for the epoch
    directory itself and ``"stem"`` for the ``<tag>_<date>`` product
    stem. Pipelines are expected to use this instead of joining paths by
    hand, so the layout only has to change in one place.
    """
    root = epoch_dir(outdir, instrument, tag, date)
    paths = {"epoch": root, "stem": epoch_dirname(tag, date)}
    os.makedirs(root, exist_ok=True)
    for stage in stages:
        path = os.path.join(root, stage)
        os.makedirs(path, exist_ok=True)
        paths[stage] = path
    return paths


def product_basename(tag: str, date: str, kind: str = "",
                     channel: str | None = None,
                     suffix: str | None = None) -> str:
    """Build a product filename following the MRS-PFPC convention.

    ``<tag>_<date>[_<kind>][_<channel>][_<suffix>]``, e.g.
    ``SN2024aecx_2025-06-01_level3_ch1-short_s3d.fits``.

    `suffix` is the trailing token and may carry the extension
    ("s3d.fits", "x1d.fits", "combined.ecsv"); it is not added for you,
    because callers variously want .fits, .ecsv and no extension at all.
    """
    parts = [epoch_dirname(tag, date)]
    for token in (kind, channel, suffix):
        if token:
            parts.append(str(token).strip("_"))
    return "_".join(parts)


def product_path(outdir: str, instrument: str, tag: str, date: str,
                 stage: str, kind: str = "", channel: str | None = None,
                 suffix: str | None = None) -> str:
    """``stage_dir(...)`` joined with ``product_basename(...)``."""
    return os.path.join(
        stage_dir(outdir, instrument, tag, date, stage),
        product_basename(tag, date, kind, channel, suffix))


def parse_epoch_dirname(name: str):
    """Split ``<tag>_<date>`` back into ``(tag, date)``.

    Returns ``(name, None)`` if the directory does not end in a date, so
    callers can skip directories that are not epochs without raising.
    """
    base = os.path.basename(str(name).rstrip(os.sep))
    match = _DATE_RE.search(base)
    if not match:
        return base, None
    date = match.group(1)
    return base[: -len(date)].rstrip("_"), date


def find_epoch_dirs(outdir: str, instrument: str | None = None):
    """Walk ``outdir`` and yield every epoch directory found.

    Yields ``(instrument_label, tag, date, path)`` tuples, sorted. Used
    by the manifest rebuild, and generally for "what is actually on
    disk?" questions.
    """
    if not os.path.isdir(outdir):
        return
    wanted = instrument_label(instrument) if instrument else None
    for inst_name in sorted(os.listdir(outdir)):
        inst_path = os.path.join(outdir, inst_name)
        if not os.path.isdir(inst_path):
            continue
        if wanted is not None and inst_name != wanted:
            continue
        for epoch_name in sorted(os.listdir(inst_path)):
            epoch_path = os.path.join(inst_path, epoch_name)
            if not os.path.isdir(epoch_path):
                continue
            tag, date = parse_epoch_dirname(epoch_name)
            if date is None:
                continue
            yield inst_name, tag, date, epoch_path


# ===========================================================================
# Staging helpers
#
# Every pipeline has the same two needs: get this epoch's raw files into
# this epoch's uncal/ directory, and find raw files again on a re-run.
# They are here rather than duplicated per pipeline.
# ===========================================================================

def stage_files(files, dest_dir):
    """Link (or copy) `files` into `dest_dir`; return the staged paths.

    Symlinks keep a re-run from duplicating tens of gigabytes of raw
    data. Filesystems that refuse symlinks (some network mounts) fall
    back to a copy, so this never fails for that reason alone.
    """
    import shutil
    os.makedirs(dest_dir, exist_ok=True)
    staged = []
    for src in files:
        dst = os.path.join(dest_dir, os.path.basename(src))
        if not (os.path.exists(dst) or os.path.islink(dst)):
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                shutil.copy2(src, dst)
        staged.append(dst)
    return sorted(staged)


def find_staged_uncal(outdir, download_dir=None, pattern="*uncal.fits"):
    """Find raw files anywhere sensible under `outdir`.

    Checks the download staging area, each epoch tree's uncal/, and a
    plain <outdir>/uncal/ -- the last so that trees built before this
    layout existed still work. Duplicates are resolved by real path,
    since the epoch directories hold symlinks into the staging area.
    """
    import glob as _glob
    roots = [os.path.join(download_dir or os.path.join(outdir, "_downloaded_uncal"), pattern),
             os.path.join(outdir, "uncal", pattern),
             os.path.join(outdir, "*", "*", "uncal", pattern),
             os.path.join(outdir, pattern)]
    seen, unique = set(), []
    for root in roots:
        for path in sorted(_glob.glob(root)):
            key = os.path.realpath(path)
            if key not in seen:
                seen.add(key)
                unique.append(path)
    return unique
