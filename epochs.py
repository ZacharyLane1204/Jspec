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
            paths = sorted(_glob.glob(os.path.join(paths_or_glob, "*")))
        else:
            paths = sorted(_glob.glob(paths_or_glob))
        if not paths:
            raise FileNotFoundError(f"no files match {paths_or_glob!r}")
    else:
        paths = list(paths_or_glob)

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
