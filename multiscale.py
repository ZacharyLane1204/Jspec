"""Multi-scale feature detection with locally measured baselines.

Why this exists
---------------
The original detector decomposed a spectrum into exactly two scales: a
continuum smoothed at ``broad_scale_kms`` (30000 km/s by default) and a
"broad" component smoothed at ``narrow_scale_kms`` (600 km/s). Anything
whose width fell *between* those two numbers had nowhere to live. A
supernova feature spanning 6.1-6.4 um is ~14000 km/s wide: most of its
power was absorbed into the continuum, and the ripple that survived was
chopped by the 600 km/s smoother into two dozen separate "detections" of
180-680 km/s each. One real feature, reported as 23 fake ones.

The fix is to stop guessing the width. This module runs a bank of matched
filters spanning the resolution limit up to tens of thousands of km/s and
lets the data choose the scale. That gives, for free, a width estimate and
a natural way to merge fragments: a broad feature detected at a large scale
claims its whole extent and absorbs the small-scale ripples riding on top
of it.

Significance is measured the way a transient is measured in a light curve,
which is what you actually want here:

* the signal is the filtered flux at the candidate position,
* the *baseline* is the filtered flux in flanking windows either side of
  the candidate, excluding the candidate itself, and
* the *noise* is the observed scatter of that same signal-minus-baseline
  statistic in the surrounding spectrum.

Measuring the noise on the filtered statistic rather than propagating the
per-pixel error is the important part. Smoothing correlates neighbouring
samples, so a propagated estimate has to assume a correlation length and
gets the answer wrong by its square root. The empirical scatter of the
statistic already contains the correlation, whatever it happens to be, and
it also absorbs any low-level continuum error, fringing or stitching
residual that a formal error array knows nothing about.
"""

from __future__ import annotations

import numpy as np

from .spectrum import Spectrum, resolving_power

C_KMS = 299792.458

__all__ = ["significance_bank", "extract_peaks", "default_scales"]


# ---------------------------------------------------------------- grid ----

def _log_grid(wave, oversample=2.0, max_points=400_000):
    """A grid uniform in ln(wavelength), i.e. uniform in velocity.

    On this grid a filter of fixed width in km/s is a fixed number of
    samples, so one convolution serves the whole spectrum.
    """
    lw = np.log(np.asarray(wave, dtype=float))
    d = np.diff(lw)
    d = d[d > 0]
    if d.size == 0:
        raise ValueError("wavelength array is not increasing")
    # Median, not minimum. A stitched spectrum has near-duplicate
    # wavelengths at the segment joins where the minimum step is
    # essentially zero, and that would demand an unusable grid.
    dln = float(np.median(d)) / float(oversample)
    n = int(np.ceil((lw[-1] - lw[0]) / dln)) + 1
    if n > max_points:
        dln = (lw[-1] - lw[0]) / max_points
        n = max_points + 1
    return lw[0] + dln * np.arange(n), dln


def _gaussian_kernel(sigma_pix, truncate=4.0):
    half = max(int(np.ceil(truncate * sigma_pix)), 1)
    u = np.arange(-half, half + 1)
    g = np.exp(-0.5 * (u / sigma_pix) ** 2)
    return g / g.sum()


def _annulus_kernel(sigma_pix, inner=3.0, outer=6.0):
    """Flanking-window kernel: the local baseline either side of a
    candidate, excluding the candidate itself.

    ``inner`` is in units of the filter sigma. At 3 sigma a Gaussian has
    0.3% of its flux outside, so self-subtraction of a genuinely Gaussian
    feature is negligible; a feature with heavier wings will be slightly
    suppressed, which is the conservative direction to err in.
    """
    half = max(int(np.ceil(outer * sigma_pix)), 2)
    u = np.abs(np.arange(-half, half + 1))
    k = ((u >= inner * sigma_pix) & (u <= outer * sigma_pix)).astype(float)
    if k.sum() == 0:
        k[:] = 1.0
    return k / k.sum()


def _convolve(y, kernel):
    pad = len(kernel) // 2
    return np.convolve(np.pad(y, pad, mode="reflect"), kernel, mode="valid")


def _rolling_mad(y, window, max_anchors=1500):
    """Sigma from the median absolute deviation in a sliding window.

    Evaluated on a coarse set of anchors and interpolated: the window is
    wide by construction, so the result varies slowly and there is nothing
    to gain from evaluating it at every sample.
    """
    n = y.size
    window = int(np.clip(window, 16, n))
    if window >= n:
        s = y[np.isfinite(y)]
        if s.size < 8:
            return np.full(n, np.nan)
        return np.full(n, 1.4826 * np.median(np.abs(s - np.median(s))))
    step = max(window // 8, n // max_anchors, 1)
    centres = np.arange(0, n, step)
    out = np.full(centres.size, np.nan)
    for j, c in enumerate(centres):
        seg = y[max(0, c - window // 2):min(n, c + window // 2)]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 8:
            out[j] = 1.4826 * np.median(np.abs(seg - np.median(seg)))
    ok = np.isfinite(out) & (out > 0)
    if not ok.any():
        return np.full(n, np.nan)
    return np.interp(np.arange(n), centres[ok], out[ok])


def default_scales(spectrum: Spectrum, n=14, min_kms=None, max_kms=30000.0):
    """A log-spaced ladder of trial widths from the resolution to ``max_kms``.

    The low end is set by the instrument, not by taste: there is no point
    filtering below the line-spread function, and doing so just amplifies
    noise.
    """
    if min_kms is None:
        w = spectrum.wavelength
        w = w[np.isfinite(w)]
        r = resolving_power(float(np.nanmedian(w)), mode=getattr(spectrum, "mode", None))
        min_kms = C_KMS / r if np.isfinite(r) and r > 0 else 100.0
        min_kms = max(min_kms, 30.0)
    if max_kms <= min_kms * 1.5:
        max_kms = min_kms * 1.5
    return np.geomspace(min_kms, max_kms, int(n))


# -------------------------------------------------------- significance ----

def significance_bank(wave, resid, scales_kms, oversample=2.0,
                      noise_window=30.0, inner=3.0, outer=6.0):
    """Significance of a feature of each trial width at each wavelength.

    Parameters
    ----------
    wave, resid : array
        Wavelength in microns and the continuum-subtracted flux.
    scales_kms : array
        Trial FWHMs.
    noise_window : float
        Width, in units of the filter sigma, of the window used to measure
        the scatter of the detection statistic.

    Returns
    -------
    lam : array
        The log-spaced wavelength grid the maps are defined on.
    Z : array, shape (n_scales, n_grid)
        Signal minus local baseline, divided by the measured scatter of
        that quantity.
    D : array, shape (n_scales, n_grid)
        The signal-minus-baseline amplitude itself, in flux units.
    """
    lnw, dln = _log_grid(wave, oversample)
    good = np.isfinite(resid)
    if good.sum() < 16:
        raise ValueError("too few finite points to detect features in")
    y = np.interp(lnw, np.log(np.asarray(wave)[good]), np.asarray(resid)[good])

    scales_kms = np.atleast_1d(np.asarray(scales_kms, dtype=float))
    Z = np.full((scales_kms.size, lnw.size), np.nan)
    D = np.full_like(Z, np.nan)
    for i, s in enumerate(scales_kms):
        sigma_pix = max((s / C_KMS) / dln / 2.3548, 0.7)
        if 2 * int(np.ceil(outer * sigma_pix)) + 1 >= lnw.size:
            continue  # filter wider than the spectrum
        signal = _convolve(y, _gaussian_kernel(sigma_pix))
        base = _convolve(y, _annulus_kernel(sigma_pix, inner, outer))
        d = signal - base
        n = _rolling_mad(d, noise_window * sigma_pix)
        D[i] = d
        Z[i] = d / np.where(n > 0, n, np.inf)
    return np.exp(lnw), Z, D


# ------------------------------------------------------------- extract ----

def _extent(lam, d, j, frac_full=0.15):
    """Half-maximum and full extent of the filtered profile around ``j``.

    ``lo/hi`` walk out to ``frac_full`` of the peak so that the reported
    extent covers the visible feature (which is what gets shaded on a plot
    and handed to a fitter), while the FWHM is taken from the half-maximum
    crossings so it remains a width.
    """
    pk = d[j]
    n = d.size
    if not np.isfinite(pk) or pk == 0:
        return lam[j], lam[j], lam[j], lam[j]

    def walk(level, step):
        i = j
        while 0 <= i + step < n:
            v = d[i + step]
            if not np.isfinite(v) or v / pk <= 0 or abs(v) < abs(level):
                break
            i += step
        return i

    a_h, b_h = walk(0.5 * pk, -1), walk(0.5 * pk, +1)
    a_f, b_f = walk(frac_full * pk, -1), walk(frac_full * pk, +1)
    return lam[a_f], lam[b_f], lam[a_h], lam[b_h]


def extract_peaks(lam, Z, D, scales_kms, zmin=5.0, kill_width=1.5,
                  edge_trim=3.0, valid=None, max_features=500,
                  outer=6.0, sidelobe_pad=2.5, signed=True):
    """Collapse the (scale, wavelength) significance map into a feature list.

    A real feature lights up at several adjacent scales and at several
    adjacent wavelengths. Taking every local maximum would report all of
    those. Instead: take the most significant point anywhere in the map,
    measure its extent, blank that wavelength range at *every* scale, and
    repeat. A broad feature therefore claims the small-scale ripples
    sitting on top of it rather than being reported alongside them.

    Emission and absorption are extracted together, in one descending pass
    over ``|Z|``, and this matters. The baseline annulus sits a few sigma
    either side of each candidate, so a strong emission line inflates the
    baseline of its own neighbours and produces negative sidelobes a few
    sigma out -- which a separate absorption pass would faithfully report
    as absorption features flanking every bright line. Extracting jointly
    lets the line claim its own sidelobes. The blanked region is therefore
    at least as wide as the annulus that created them.

    Parameters
    ----------
    signed : bool
        Search both signs at once (default). Set False to search only
        positive excursions of the supplied map.
    outer : float
        Outer radius of the baseline annulus, in filter sigmas. Must match
        the value used in :func:`significance_bank`, since it sets how far
        the sidelobes reach.
    sidelobe_pad : float
        Extra blanking beyond the annulus, in filter sigmas, to cover the
        sidelobe's own width.
    """
    Zw = np.array(Z, dtype=float)
    if valid is not None:
        Zw[:, ~np.asarray(valid, dtype=bool)] = np.nan

    # Blank each scale's own edges, where the filter and its baseline
    # annulus hang off the end of the data and manufacture signal.
    for i, s in enumerate(scales_kms):
        pad = edge_trim * (s / C_KMS) * lam
        Zw[i, (lam < lam[0] + pad) | (lam > lam[-1] - pad)] = np.nan

    rank = np.abs(Zw) if signed else Zw
    out = []
    while len(out) < max_features and np.any(np.isfinite(rank)):
        k = int(np.nanargmax(rank))
        i, j = np.unravel_index(k, rank.shape)
        z = Zw[i, j]
        if not np.isfinite(z) or abs(z) < zmin:
            break
        sgn = 1.0 if z > 0 else -1.0
        lo, hi, hlo, hhi = _extent(lam, sgn * D[i], j)
        fwhm_um = max(hhi - hlo, (scales_kms[i] / C_KMS) * lam[j] * 0.5)
        out.append(dict(center=float(lam[j]), z=float(abs(z)),
                        scale_kms=float(scales_kms[i]),
                        lo=float(lo), hi=float(hi),
                        fwhm_um=float(fwhm_um),
                        sign="emission" if sgn > 0 else "absorption",
                        amplitude=float(D[i, j])))
        # Blank the feature *and* the annulus that generated its sidelobes.
        # The negative sidelobe of a (Gaussian minus annulus) filter peaks
        # where the feature falls in the annulus, so it reaches the outer
        # annulus radius plus roughly the filter's own width -- blanking
        # only to `outer` leaves the sidelobe sitting exactly on the
        # boundary, which is where it is strongest.
        sigma_um = (scales_kms[i] / C_KMS) * lam[j] / 2.3548
        half = max(0.5 * kill_width * max(hi - lo, fwhm_um),
                   (outer + sidelobe_pad) * sigma_um)
        dead = (lam >= lam[j] - half) & (lam <= lam[j] + half)
        Zw[:, dead] = np.nan
        rank[:, dead] = np.nan
    return out
