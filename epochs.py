"""
jspec.epochs
============

Epoch grouping for the analysis layer.

Why this module exists
----------------------

Everything in :mod:`jspec.stitch` operates on a list of segments and asks
"how do these join together in wavelength?". That question has no opinion
about *time*. Point it at a directory holding ch1-short from June and
ch1-short from November and it will cheerfully treat them as two segments of
one spectrum and solve a scale factor between them -- silently turning real
variability into a calibration correction, and producing a single "spectrum"
that never existed.

That is exactly the failure this project's pipeline scripts already guard
against, via ``group_files_by_epoch`` in ``combine_nirspec_orders``. This
module **reuses that same function** rather than reimplementing it, so there
is exactly one definition of "epoch" in the codebase and the analysis layer
cannot drift from the reduction layer. The MJD is read from ``MJD-OBS``,
falling back to ``EXPSTART``, from FITS headers, or from ``MJD_MEAN`` /
``MJD_MIN`` table metadata for ``.ecsv`` products.

The rule, unchanged: sort by exposure-start MJD, and start a new epoch
whenever the gap to the previous file exceeds ``max_gap_days`` (default 1.0).
Files with no usable MJD keyword each get their own trailing single-file
group rather than being lumped into a real epoch.

Use
---

::

    from jspec.epochs import group_segments_by_epoch
    from jspec.stitch import stitch

    for ep in group_segments_by_epoch("stage3/*_x1d.fits", redshift=0.002665):
        report = stitch(ep.segments)
        report.spectrum.write(f"{target}_{ep.label}.ecsv")

or just use the CLI, which does this by default::

    python -m jspec stitch "stage3/*_x1d.fits" -o SN2024aecx --per-epoch
"""
from __future__ import annotations

import glob as _glob
import os
import warnings
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "EpochGroup",
    "mjd_of",
    "discover_inputs",
    "group_paths_by_epoch",
    "group_segments_by_epoch",
    "epoch_span_days",
    "check_single_epoch",
]


def _pipeline_helpers():
    """Import the canonical epoch helpers from the reduction layer.

    Imported lazily and by name so that this module still works if the
    reduction scripts are unavailable, and so that importing ``jspec`` never
    drags in the pipeline package at import time.
    """
    from .pipelines import combine_nirspec_orders as cno
    return cno


def mjd_of(path):
    """Exposure-start MJD of a file, or ``None`` if it carries no usable
    time keyword.

    Delegates to ``combine_nirspec_orders.get_mjd`` -- same keyword
    preference (``MJD-OBS`` then ``EXPSTART`` for FITS, ``MJD_MEAN`` then
    ``MJD_MIN`` for ``.ecsv``) as the rest of the project.
    """
    try:
        return _pipeline_helpers().get_mjd(path)
    except Exception as exc:                                # noqa: BLE001
        warnings.warn(f"could not read an MJD from {path}: "
                      f"{type(exc).__name__}: {exc}", stacklevel=2)
        return None


@dataclass
class EpochGroup:
    """One epoch's worth of files, and the segments read from them."""
    index: int
    paths: list
    mjd_min: float = np.nan
    mjd_max: float = np.nan
    mjd_mean: float = np.nan
    date: str = "unknown-date"
    segments: list = field(default_factory=list)

    @property
    def label(self) -> str:
        """Filename-safe epoch label, e.g. ``2025-06-01``."""
        return self.date

    @property
    def span_days(self) -> float:
        if not np.isfinite(self.mjd_min) or not np.isfinite(self.mjd_max):
            return np.nan
        return float(self.mjd_max - self.mjd_min)

    def __repr__(self):
        names = ",".join(s.name for s in self.segments) if self.segments else ""
        return (f"<Epoch {self.index} {self.date} n={len(self.paths)} "
                f"span={self.span_days:.2f}d {names}>")

    def summary(self) -> str:
        L = [f"epoch {self.index}: {self.date}  "
             f"({len(self.paths)} file(s), span {self.span_days:.3f} d, "
             f"MJD {self.mjd_min:.4f}-{self.mjd_max:.4f})"]
        for p, s in zip(self.paths,
                        self.segments or [None] * len(self.paths)):
            if s is None:
                L.append(f"    {os.path.basename(p)}")
            else:
                L.append(f"    {s.name:<16} {s.wmin:7.3f}-{s.wmax:7.3f} um  "
                         f"{os.path.basename(p)}")
        return "\n".join(L)


_SPECTRUM_EXTS = (".ecsv", ".fits")
_SKIP_DIRS = {"uncal", "stage1", "stage2", "asn", "_downloaded_uncal",
              "__pycache__"}


def discover_inputs(directory):
    """Work out which files in a pipeline output tree to stitch.

    Pointing at ``<outdir>`` should Just Work, and that means answering a
    question with a right answer rather than globbing: an epoch directory
    holds Stage-3 cubes, per-channel spectra, PFPC output *and* the final
    product, and stitching all of them together is meaningless.

    So the manifest is consulted first. ``products.csv`` already records
    which product is the recommended one per instrument and epoch, which
    is exactly the question being asked. The same product can appear
    twice -- once where it was produced, once copied into ``final/`` --
    so entries are de-duplicated on (instrument, date, product) with the
    copy under ``final/`` preferred.

    Only if there is no manifest does it fall back to a recursive search,
    skipping the intermediate stage directories.
    """
    from .manifest import read_manifest, is_recommended, _product_from_name
    from .layout import find_epoch_dirs

    def _prefer_final(chosen, key, full):
        prev = chosen.get(key)
        # Prefer the copy that lives in final/, so the path matches the
        # one the documentation tells you to analyse.
        if prev is None or (os.sep + "final" + os.sep) in full:
            chosen[key] = full

    # 1. The manifest already answers this question.
    rows = [r for r in read_manifest(directory)
            if str(r.get("recommended", "")).strip().lower() in ("yes", "true", "1")]
    chosen = {}
    for r in rows:
        full = os.path.join(directory, r.get("path") or "")
        if os.path.exists(full):
            _prefer_final(chosen,
                          (r.get("instrument"), r.get("date"), r.get("product")),
                          full)
    if chosen:
        return sorted(chosen.values())

    # 2. No manifest, but a recognisable pipeline tree: apply the same
    #    rule straight off disk. Do NOT fall through to a blind walk
    #    here -- an epoch directory holds per-channel Stage-3 spectra
    #    that carry no MJD, so they would each be treated as a separate
    #    epoch and stitched alone.
    try:
        epoch_dirs = list(find_epoch_dirs(directory))
    except Exception:                                    # noqa: BLE001
        epoch_dirs = []
    if epoch_dirs:
        for label, tag, date, epoch_path in epoch_dirs:
            for stage in ("final", "reextract"):
                stage_path = os.path.join(epoch_path, stage)
                if not os.path.isdir(stage_path):
                    continue
                for name in sorted(os.listdir(stage_path)):
                    if not name.endswith(".ecsv"):
                        continue
                    product = _product_from_name(name, tag, date) or stage
                    if is_recommended(label, product):
                        _prefer_final(chosen, (label, date, product),
                                      os.path.join(stage_path, name))
        if chosen:
            return sorted(chosen.values())
        raise FileNotFoundError(
            f"found {len(epoch_dirs)} epoch director(ies) under {directory!r} "
            "but no recommended final product in any of them. Run "
            f"'jspec products {directory} --rebuild' to see what is there.")

    # 3. A plain folder of spectra. Collapse duplicate basenames, since
    #    the same product is often written twice.
    found = {}
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for f in files:
            if not f.lower().endswith(_SPECTRUM_EXTS) or f.startswith("."):
                continue
            full = os.path.join(root, f)
            _prefer_final(found, f, full)
    return sorted(found.values())


def group_paths_by_epoch(paths_or_glob, max_gap_days=1.0):
    """Cluster files into :class:`EpochGroup` objects by exposure MJD.

    Parameters
    ----------
    paths_or_glob : str or iterable of str
        A glob pattern, a directory, or an explicit list of paths.
    max_gap_days : float
        A gap larger than this starts a new epoch. Default 1.0, matching
        every other epoch-aware script in this project.

    Returns
    -------
    list of EpochGroup, in time order.
    """
    cno = _pipeline_helpers()

    if isinstance(paths_or_glob, str):
        if os.path.isdir(paths_or_glob):
            paths = discover_inputs(paths_or_glob)
        else:
            paths = sorted(_glob.glob(paths_or_glob))
        if not paths:
            raise FileNotFoundError(f"no files match {paths_or_glob!r}")
    else:
        paths = []
        for item in paths_or_glob:
            if os.path.isdir(item):
                paths.extend(discover_inputs(item))
            else:
                paths.append(item)
        paths = sorted(dict.fromkeys(paths))
        if not paths:
            raise FileNotFoundError(f"no usable files in {paths_or_glob!r}")

    # Never hand a directory to the FITS reader.
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        raise FileNotFoundError(f"no usable files match {paths_or_glob!r}")

    clusters = cno.group_files_by_epoch(paths, max_gap_days=max_gap_days)

    groups = []
    for i, files in enumerate(clusters):
        info = cno.get_mjd_info(files)
        # Prefer a date token already present in the filenames: an exposure
        # just after 0h UTC otherwise lands on the neighbouring calendar day
        # and stops matching the names you already use.
        date = cno.date_from_filenames(files) or cno.mjd_to_datestr(
            info.get("MJD_MEAN"))
        groups.append(EpochGroup(
            index=i, paths=list(files),
            mjd_min=float(info["MJD_MIN"]) if info["MJD_MIN"] is not None else np.nan,
            mjd_max=float(info["MJD_MAX"]) if info["MJD_MAX"] is not None else np.nan,
            mjd_mean=float(info["MJD_MEAN"]) if info["MJD_MEAN"] is not None else np.nan,
            date=date))
    return groups


def group_segments_by_epoch(paths_or_glob, max_gap_days=1.0, redshift=0.0,
                            names=None, verbose=True, **read_kwargs):
    """Group files into epochs *and* read each epoch's segments.

    This is the function to use before :func:`jspec.stitch.stitch`. Each
    returned :class:`EpochGroup` has its ``segments`` populated and sorted
    blue to red, ready to stitch independently.

    Unreadable files are skipped with a warning rather than aborting the run,
    and an epoch left with no usable segments is dropped (with a warning), so
    one bad file costs you that file and not the other eleven.

    Extra keyword arguments are passed to
    :func:`jspec.spectrum.read_spectrum` -- notably ``mask_red_end``.
    """
    from .spectrum import read_segments

    groups = group_paths_by_epoch(paths_or_glob, max_gap_days=max_gap_days)
    if verbose:
        print(f"{sum(len(g.paths) for g in groups)} file(s) -> "
              f"{len(groups)} epoch(s)  "
              f"(> {max_gap_days:g} day gap starts a new epoch)")

    out = []
    for g in groups:
        try:
            g.segments = read_segments(g.paths, redshift=redshift,
                                       names=names, **read_kwargs)
        except Exception as exc:                            # noqa: BLE001
            warnings.warn(f"epoch {g.date}: could not read segments "
                          f"({type(exc).__name__}: {exc}); skipping",
                          stacklevel=2)
            continue
        if not g.segments:
            warnings.warn(f"epoch {g.date}: no usable segments; skipping",
                          stacklevel=2)
            continue
        out.append(g)
        if verbose:
            print(g.summary())
    return out


def epoch_span_days(segments):
    """Total MJD span covered by a list of segments, or NaN if unknown.

    Reads ``MJD_MEAN`` / ``MJD_MIN`` / ``MJD_MAX`` from each segment's
    spectrum metadata, which is what this project's combined products carry.
    """
    lo, hi = [], []
    for s in segments:
        meta = getattr(s.spectrum, "meta", {}) or {}
        vals = [meta.get(k) for k in ("MJD_MIN", "MJD_MEAN", "MJD_MAX")]
        vals = [float(v) for v in vals if v is not None]
        if vals:
            lo.append(min(vals))
            hi.append(max(vals))
    if not lo:
        return np.nan
    return float(max(hi) - min(lo))


def check_single_epoch(segments, max_gap_days=1.0, raise_on_fail=False):
    """Verify that a set of segments really does come from one epoch.

    Call this immediately before stitching anything you assembled by hand.
    Stitching across epochs does not fail loudly -- it produces a plausible
    spectrum with a scale factor silently absorbing the source's real
    variability -- so the check has to be explicit.

    Returns ``(ok, message)``. With ``raise_on_fail=True`` raises
    :class:`ValueError` instead of returning ``ok=False``.
    """
    span = epoch_span_days(segments)
    if not np.isfinite(span):
        msg = ("no MJD metadata on these segments, so the epoch could not be "
               "verified. If they came from different visits, stitching them "
               "will absorb real variability into the scale factors.")
        if raise_on_fail:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
        return False, msg
    if span > max_gap_days:
        msg = (f"these segments span {span:.2f} days, more than "
               f"max_gap_days={max_gap_days:g}. They are NOT one epoch. "
               f"Group them with jspec.epochs.group_segments_by_epoch and "
               f"stitch each epoch separately.")
        if raise_on_fail:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
        return False, msg
    return True, f"single epoch, span {span:.3f} d"
