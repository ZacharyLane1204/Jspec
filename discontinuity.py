"""Find step discontinuities in a spectrum, using flux alone.

A jump between two segments is normally caught by comparing them in their
overlap, which is what :mod:`jspec.stitch` does. That only works when the
two sides are *labelled* as different segments. The failure this module
exists for is the opposite case: a discontinuity sitting in the middle of
what claims to be one continuous spectrum.

Both real examples came from MIRI MRS. The ABI re-extraction of
``SN2024aecx_2025-12-07`` steps by a factor 2.7 at 16.2 um, a third of
the way into ch3-long, because the background model changes there -- the
PFPC extraction of the same cube is flat across that wavelength. And
ch4-short jumps by a factor 2.1 across its own blue edge at 18.0 um,
inside the overlap the combiner uses to rescale it, so the bad region
sets the scale factor for the whole channel.

Neither is visible to a segment-aware check, because by then both are
inside a single combined product carrying a single segment label. The
only way to catch them is to look at the flux.

The test is deliberately the same one a light curve uses for a break:
fit the level on each side of a candidate boundary, and ask whether the
difference is significant against the scatter *within* each side. A real
spectral feature is a bump with two edges; a step has one, and the levels
either side do not come back together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

__all__ = ["Step", "find_steps", "steps_table"]


@dataclass
class Step:
    """One flux discontinuity.

    Attributes
    ----------
    wavelength : float
        Where the level changes, in microns.
    lo_level, hi_level : float
        Robust flux level immediately blueward and redward.
    ratio : float
        ``hi_level / lo_level``. The natural measure for a calibration
        offset, which is multiplicative.
    delta, delta_err : float
        Difference in level and its uncertainty from the scatter either
        side.
    snr : float
        ``delta / delta_err``. Signed.
    n_lo, n_hi : int
        Points used on each side.
    """

    wavelength: float
    lo_level: float
    hi_level: float
    ratio: float
    delta: float
    delta_err: float
    snr: float
    n_lo: int
    n_hi: int

    def __repr__(self) -> str:
        return (f"<Step {self.wavelength:.3f}um x{self.ratio:.2f} "
                f"S/N={self.snr:+.1f}>")


def _levels(y):
    """Median and its standard error, robustly."""
    y = y[np.isfinite(y)]
    n = y.size
    if n < 3:
        return np.nan, np.nan, n
    med = float(np.median(y))
    mad = 1.4826 * float(np.median(np.abs(y - med)))
    return med, mad / np.sqrt(n), n


def transition_width_um(wave, flux, centre, lo_level, delta, window_um=0.25,
                        smooth_um=0.01):
    """How many microns the flux takes to complete a change of ``delta``.

    This is the one test that separates an instrumental step from a real
    spectral feature on physics rather than statistics. A calibration or
    background-model step is imposed on the data at a single wavelength, so
    it completes within a pixel or two. A real band edge -- the blue edge of
    the 11.3 um aromatic complex, say -- is resolved, and takes many.

    Returns the 10%-to-90% transition distance in microns, or ``nan`` if it
    cannot be measured. Compare it against the local sampling: a value of a
    few times ``dw`` is a step, a value of many times it is structure.
    """
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    near = (wave >= centre - window_um) & (wave <= centre + window_um)
    if near.sum() < 5 or not np.isfinite(delta) or delta == 0:
        return float("nan")
    ws, fs = wave[near], flux[near]
    dws = np.diff(np.sort(ws))
    dw = float(np.median(dws)) if dws.size else 0.0
    if not np.isfinite(dw) or dw <= 0:
        return float("nan")
    k = max(3, int(round(smooth_um / dw)) | 1)
    fsm = median_filter(np.where(np.isfinite(fs), fs, lo_level), k, mode="nearest")
    lo_t = lo_level + 0.1 * delta
    hi_t = lo_level + 0.9 * delta
    if delta > 0:
        c10 = np.flatnonzero(fsm >= lo_t)
        c90 = np.flatnonzero(fsm >= hi_t)
    else:
        c10 = np.flatnonzero(fsm <= lo_t)
        c90 = np.flatnonzero(fsm <= hi_t)
    if not c10.size or not c90.size:
        return float("nan")
    return abs(float(ws[c90[0]] - ws[c10[0]]))


def find_steps(wave, flux, window_um=0.25, gap_um=0.02, snr_min=6.0,
               min_ratio=1.25, step=None, edge_pad=None,
               min_points=12, persist=0.5, grad_factor=3.0,
               max_rise_um=0.06) -> list[Step]:
    """Locate significant level changes.

    Parameters
    ----------
    window_um : float
        Half-width of the block averaged on each side of a candidate
        boundary. Wide enough to beat down the noise, narrow enough that
        a genuine broad feature does not fill both blocks.
    gap_um : float
        Dead zone skipped either side of the boundary, so a step that is
        smeared over a few pixels is not diluted.
    snr_min, min_ratio : float
        A step must be both statistically significant *and* large enough
        to matter. The significance test alone is not sufficient: on a
        high-S/N spectrum the continuum's own slope is significant over a
        0.25 um baseline. Requiring a ratio away from 1 as well is what
        separates a calibration step from a steep but continuous
        spectrum.
    edge_pad : float, optional
        Skip this far in from each end, where the window would be
        one-sided. Defaults to ``window_um``.
    persist : float
        Fraction of the jump that must still be present one window
        further on. **This is what separates a step from a feature.**
        The blue edge of the 11.3 um aromatic band is a factor-7 rise
        over 0.25 um and trips every significance test there is -- but
        by 11.5 um the flux is back down, because a band is a bump with
        two edges. A calibration step has one edge and never comes back.
        Set to 0 to report every edge.

    Returns
    -------
    list of Step
        Sorted by descending ``|snr|``. Adjacent candidates are collapsed
        so one discontinuity is reported once, at its strongest point.

    Notes
    -----
    This cannot tell a calibration step from an astrophysical edge, and
    it is not supposed to. It says "the level changes abruptly here",
    which is the thing worth a human's attention either way.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    good = np.isfinite(wave) & np.isfinite(flux)
    if good.sum() < 2 * min_points:
        return []
    order = np.argsort(wave[good])
    w, f = wave[good][order], flux[good][order]

    dw = float(np.median(np.diff(w)))
    if step is None:
        step = max(dw * 4.0, 0.01)
    if edge_pad is None:
        edge_pad = window_um

    probes = np.arange(w[0] + edge_pad, w[-1] - edge_pad, step)
    if probes.size == 0:
        return []

    # Typical level change between adjacent windows, which sets the scale
    # a genuine discontinuity has to beat.
    ref = 0.0
    if grad_factor > 0:
        deltas = []
        for b in probes:
            lo = (w >= b - window_um - gap_um) & (w <= b - gap_um)
            hi = (w >= b + gap_um) & (w <= b + window_um + gap_um)
            if lo.sum() < min_points or hi.sum() < min_points:
                continue
            a_m, _, _ = _levels(f[lo])
            b_m, _, _ = _levels(f[hi])
            if np.isfinite(a_m) and np.isfinite(b_m):
                deltas.append(abs(b_m - a_m))
        if deltas:
            ref = float(np.median(deltas))

    cand: list[Step] = []
    for b in probes:
        lo = (w >= b - window_um - gap_um) & (w <= b - gap_um)
        hi = (w >= b + gap_um) & (w <= b + window_um + gap_um)
        if lo.sum() < min_points or hi.sum() < min_points:
            continue
        a_med, a_err, n_lo = _levels(f[lo])
        b_med, b_err, n_hi = _levels(f[hi])
        if not np.isfinite(a_med) or not np.isfinite(b_med):
            continue
        derr = float(np.hypot(a_err, b_err))
        if not np.isfinite(derr) or derr <= 0:
            continue
        delta = b_med - a_med
        snr = delta / derr
        ratio = b_med / a_med if a_med != 0 else np.inf
        if abs(snr) < snr_min:
            continue
        if np.isfinite(ratio) and ratio > 0 and \
                (1.0 / min_ratio) < ratio < min_ratio:
            continue
        if ref > 0 and abs(delta) < grad_factor * ref:
            continue
        if max_rise_um > 0:
            rise = transition_width_um(w, f, b, a_med, delta,
                                       window_um=window_um)
            if np.isfinite(rise) and rise > max_rise_um:
                continue
        if persist > 0:
            far = (w > b + gap_um + window_um) & \
                  (w <= b + gap_um + 3.0 * window_um)
            if far.sum() >= min_points:
                f_med, _, _ = _levels(f[far])
                if np.isfinite(f_med) and \
                        abs(f_med - a_med) < persist * abs(delta):
                    continue
        cand.append(Step(wavelength=float(b), lo_level=a_med, hi_level=b_med,
                         ratio=float(ratio), delta=float(delta),
                         delta_err=derr, snr=float(snr),
                         n_lo=int(n_lo), n_hi=int(n_hi)))

    # One discontinuity trips every probe within a window of it. Keep the
    # strongest of each run so it is reported once.
    cand.sort(key=lambda s: s.wavelength)
    merged: list[Step] = []
    for s in cand:
        if merged and (s.wavelength - merged[-1].wavelength) <= window_um:
            if abs(s.snr) > abs(merged[-1].snr):
                merged[-1] = s
        else:
            merged.append(s)

    merged.sort(key=lambda s: -abs(s.snr))
    return merged


def steps_table(steps):
    """:class:`~astropy.table.Table` of :func:`find_steps` output."""
    from astropy.table import Table

    cols = ["wavelength", "lo_level", "hi_level", "ratio", "delta",
            "delta_err", "snr", "n_lo", "n_hi"]
    rows = [[getattr(s, c) for c in cols] for s in steps]
    return Table(rows=rows, names=cols) if rows else Table(names=cols)
