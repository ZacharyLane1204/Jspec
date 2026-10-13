"""
jspec.continuum
===============

Continuum and pseudo-continuum estimation, and the scale-separation that makes
"broad component" and "narrow line" well-defined terms.

The scale problem
-----------------

"Broad feature" and "continuum" are not physically distinct categories -- they
are the same thing viewed at different wavelength scales. A 3000 km/s ejecta
line at 10 um is 0.1 um wide; the silicate feature is 4 um wide; the thermal
continuum varies over the whole 5-28 um range. Any detection algorithm that
does not state its scale separation is implicitly choosing one anyway, and
will silently reclassify a feature as continuum (or vice versa) when you hand
it a different wavelength range.

This module therefore makes the scales explicit and constant in *velocity*,
not in microns. :func:`decompose` splits a spectrum into three additive
components using two scale parameters:

    flux = continuum  +  broad  +  narrow  +  noise

* ``continuum``: everything smoother than ``broad_scale_kms`` (default
  30000 km/s, i.e. 10% in wavelength) -- the thermal/power-law shape and the
  very broad solid-state bands.
* ``broad``: structure between ``narrow_scale_kms`` and ``broad_scale_kms``
  -- ejecta lines, molecular band complexes, PAH features.
* ``narrow``: structure finer than ``narrow_scale_kms`` (default 600 km/s)
  -- host and CSM lines, unresolved features, and also cosmic-ray residuals
  and bad pixels, which is why detection applies a significance cut.

Working in velocity means a single pair of numbers behaves consistently from
0.97 to 28 um, which a pair of numbers in microns cannot.

Method
------

The smoother is a Whittaker-Henderson penalized least squares with
asymmetric reweighting (arPLS, Baek et al. 2015). Compared with a median
filter or a spline through hand-picked anchors:

* it is asymmetric, so emission lines pull the baseline up far less than they
  would in a symmetric fit -- essential for an emission-line-dominated
  nebular spectrum where a symmetric continuum sits well above the true one;
* it needs no anchor points, so the answer does not depend on where someone
  clicked;
* its single smoothness parameter has a closed-form relationship to a cutoff
  scale (see :func:`_lambda_for_scale`), so "smoother than 30000 km/s" is a
  statement about the filter, not a hope.

All smoothing is done on a uniform log-wavelength grid and interpolated back.
This matters: a stitched JWST spectrum has wildly non-uniform sampling (MRS
ch1 samples ~0.0008 um, ch4 ~0.006 um), and a difference-penalty smoother
applied to a non-uniform grid imposes a smoothness that varies with the local
pixel size -- which would make the continuum stiffer in ch1 than in ch4 for no
physical reason, and put a spurious inflection at every band join.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.ndimage import uniform_filter1d
from scipy.sparse.linalg import spsolve

from .spectrum import C_KMS, Spectrum

__all__ = [
    "Decomposition",
    "arpls",
    "whittaker",
    "estimate_continuum",
    "anchored_continuum",
    "band_mask_ranges",
    "decompose",
    "normalise",
    "local_continuum",
]


# ---------------------------------------------------------------------------
# log-grid helpers
# ---------------------------------------------------------------------------
def _log_grid(wave, oversample=1.0):
    """Uniform grid in ln(lambda) spanning ``wave``.

    Uniform in ln(lambda) means uniform in velocity, which is what makes a
    fixed smoothing length correspond to a fixed km/s everywhere.
    """
    lw = np.log(wave)
    n = max(int(oversample * wave.size), 16)
    return np.linspace(lw[0], lw[-1], n)


def _lambda_for_scale(scale_kms, dlnlam):
    """Whittaker smoothness parameter giving a half-power cutoff at
    ``scale_kms``.

    For a second-difference penalty the transfer function is
    ``1 / (1 + lam * (2 - 2 cos w)^2)``, which is at half power when
    ``2 - 2 cos w = lam^-1/2``; for small ``w`` that is ``w = lam^-1/4``, i.e.
    a cutoff period of ``2 pi lam^(1/4)`` samples. Inverting, a cutoff of
    ``N`` samples needs ``lam = (N / 2 pi)^4``.

    With a grid uniform in ln(lambda), ``N = (scale_kms / c) / dlnlam``.
    """
    n_samples = (scale_kms / C_KMS) / dlnlam
    n_samples = max(n_samples, 2.0)
    return (n_samples / (2.0 * np.pi)) ** 4


def _d2_matrix(n):
    """Sparse second-difference operator."""
    return sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")


# ---------------------------------------------------------------------------
# smoothers
# ---------------------------------------------------------------------------
def whittaker(y, lam, w=None, order=2):
    """Symmetric Whittaker-Henderson smoother.

    Minimises ``sum w_i (y_i - z_i)^2 + lam * ||D^order z||^2``. Returns ``z``.
    Used as the inner solve of :func:`arpls` and on its own where symmetric
    smoothing is genuinely wanted (smoothing a *residual*, where positive and
    negative excursions are equally meaningful).
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n <= order + 1:
        return y.copy()
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc") \
        if order == 2 else sparse.diags([-1.0, 1.0], [0, 1], shape=(n - 1, n), format="csc")
    W = sparse.diags(np.ones(n) if w is None else np.asarray(w, dtype=float), format="csc")
    A = (W + lam * (D.T @ D)).tocsc()
    b = W @ y
    return spsolve(A, b)


def arpls(y, lam, ratio=1e-3, max_iter=50, w0=None, w_min=1e-3,
          local_window=None, valid=None):
    """Asymmetrically reweighted penalized least squares baseline.

    (Baek, Park, Ahn & Choo 2015, *Analyst* 140, 250, with two modifications
    described below that are necessary for JWST spectra.)

    Iteratively fits a smooth baseline, down-weighting points that lie *above*
    it via a logistic function of the residual measured against the spread of
    the points lying *below* it. Points below the baseline keep full weight.
    The result follows the lower envelope of an emission-line spectrum without
    being dragged by the lines.

    For an *absorption*-dominated spectrum, pass ``-y`` and negate the result,
    or use :func:`estimate_continuum` with ``direction='absorption'``, which
    does exactly that.

    Parameters
    ----------
    lam : float
        Smoothness; use :func:`_lambda_for_scale` to set it from a velocity.
    ratio : float
        Convergence threshold on the fractional change in the weight vector.
    w_min : float
        Floor on the point weights. Published arPLS lets a weight go to zero,
        which means a region where every point sits above the baseline becomes
        *completely unconstrained* -- the penalty term alone then extrapolates
        it, and with a stiff penalty that extrapolation runs away. The floor
        guarantees every region keeps some pull from its own data.
    local_window : int, optional
        Number of samples over which the residual statistics ``(mean, sigma)``
        driving the reweighting are computed. **This is the important
        modification.** The published algorithm uses a single global
        ``(mean, sigma)`` of the sub-baseline residuals. On a stitched JWST
        spectrum the noise varies by more than an order of magnitude between
        NIRSpec and MRS ch4, so a global sigma is far too large for the quiet
        bands and far too small for the noisy ones; in the noisy ones *every*
        point ends up classified as "above the baseline", its weight collapses,
        and the baseline falls away from the data entirely -- which on real
        data produced a continuum that went negative beyond 20 um while the
        flux was rising. Local statistics make the asymmetry test mean the
        same thing in every band. Defaults to four times the smoother's own
        cutoff length, with a floor of 101 samples.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 8:
        return y.copy()
    D = _d2_matrix(n)
    DTD = (D.T @ D) * lam
    w = np.ones(n) if w0 is None else np.asarray(w0, dtype=float).copy()
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape == w.shape:
            w = np.where(valid, w, 0.0)
        else:
            valid = None

    if local_window is None:
        cutoff_samples = 2.0 * np.pi * lam ** 0.25
        local_window = int(max(101, min(n, 4.0 * cutoff_samples)))
    local_window = int(max(11, min(local_window, n)))

    z = y.copy()
    for _ in range(max_iter):
        W = sparse.diags(w, format="csc")
        z = spsolve((W + DTD).tocsc(), w * y)
        d = y - z
        m, s = _local_negative_stats(d, local_window)
        if not np.any(np.isfinite(s)):
            break
        # Logistic reweighting: ~1 well below the baseline, ~0 well above it.
        with np.errstate(over="ignore", invalid="ignore"):
            arg = np.clip(2.0 * (d - (2.0 * s - m)) / np.where(s > 0, s, np.inf),
                          -500, 500)
            wt = 1.0 / (1.0 + np.exp(arg))
        wt = np.where(np.isfinite(wt), wt, 1.0)
        wt = np.maximum(wt, w_min)
        if valid is not None:
            wt = np.where(valid, wt, 0.0)
        if np.linalg.norm(w - wt) / max(np.linalg.norm(w), 1e-30) < ratio:
            w = wt
            break
        w = wt
    return z


def _local_negative_stats(d, window):
    """Rolling mean and standard deviation of the *negative* part of ``d``.

    Computed with running sums over the indicator ``d < 0`` so the cost is
    O(n) rather than O(n * window) -- this runs inside every arPLS iteration,
    so a naive implementation dominates the runtime on a 14000-point spectrum.

    Where a window contains too few sub-baseline points to characterise (fewer
    than three), the global statistics are substituted rather than NaN, so the
    reweighting stays defined instead of silently disabling itself.
    """
    d = np.asarray(d, dtype=float)
    n = d.size
    neg = (d < 0) & np.isfinite(d)
    k = float(window)
    cnt = uniform_filter1d(neg.astype(float), size=window, mode="nearest") * k
    s1 = uniform_filter1d(np.where(neg, d, 0.0), size=window, mode="nearest") * k
    s2 = uniform_filter1d(np.where(neg, d * d, 0.0), size=window, mode="nearest") * k

    ok = cnt >= 3
    mean = np.zeros(n)
    sigma = np.zeros(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean[ok] = s1[ok] / cnt[ok]
        var = s2[ok] / cnt[ok] - mean[ok] ** 2
        sigma[ok] = np.sqrt(np.clip(var, 0.0, None))

    dn = d[neg]
    gm = float(dn.mean()) if dn.size >= 2 else 0.0
    gs = float(dn.std()) if dn.size >= 2 else 0.0
    mean[~ok] = gm
    sigma[~ok] = gs
    # A zero local sigma (a perfectly flat run) would make the logistic a step
    # function and lock those weights at the floor forever.
    bad = ~np.isfinite(sigma) | (sigma <= 0)
    sigma[bad] = gs if gs > 0 else 1e-12
    mean[~np.isfinite(mean)] = gm
    return mean, sigma


# ---------------------------------------------------------------------------
# continuum
# ---------------------------------------------------------------------------
def estimate_continuum(spectrum: Spectrum, scale_kms=30000.0, method="arpls",
                       direction="emission", mask_ranges=None, rest=False,
                       oversample=0.5, max_iter=50, transform="asinh") -> np.ndarray:
    """Estimate the continuum, returned on the spectrum's own wavelength grid.

    Parameters
    ----------
    scale_kms : float
        Structure broader than this is continuum; narrower is a feature. The
        default 30000 km/s (10% in wavelength) keeps the broad solid-state
        bands *in* the continuum. Lower it to ~8000 km/s if you want the
        silicate feature treated as a feature rather than as continuum -- that
        is a scientific choice, not a technical one, and it is the single
        parameter that most changes what this package reports.
    method : {'arpls', 'whittaker', 'percentile'}
        ``arpls`` follows the lower envelope (right for emission spectra);
        ``whittaker`` is symmetric (right when you genuinely want a
        mid-line through features, e.g. for a spectrum with comparable
        emission and absorption); ``percentile`` is a robust rolling
        percentile, the most conservative and the least smooth.
    direction : {'emission', 'absorption', 'symmetric'}
        Which way features depart from the continuum. ``absorption`` flips the
        asymmetry so the baseline follows the *upper* envelope.
    mask_ranges : list of (lo, hi), optional
        Wavelength ranges excluded from the fit (not from the output) --
        e.g. a saturated line, or a known detector artefact. The continuum is
        still evaluated there, by interpolation through the gap.
    rest : bool
        Interpret ``mask_ranges`` in the rest frame.
    transform : {'asinh', 'log', 'linear'}
        Flux-space in which the smoothing is done.

        **This defaults to 'asinh' for a reason.** A difference-penalty
        smoother penalizes *absolute* curvature, so on a spectrum spanning two
        decades in flux -- which every stitched NIRSpec-to-MIRI spectrum of a
        dusty source does -- the penalty is dominated entirely by the bright
        red end, and the fit overshoots into negative continuum in exactly the
        20-28 um region where the thermal dust continuum rises steepest. A
        negative continuum then silently inverts every normalized feature
        computed from it.

        ``asinh`` smooths ``asinh(f / s)`` with ``s`` set from the typical
        uncertainty, which behaves like ``f/s`` near zero and like
        ``ln(2f/s)`` far above it. That compresses the dynamic range (so the
        penalty means the same thing at 2 um and 27 um), keeps the transform
        monotonic and invertible, and -- unlike a plain log -- copes with the
        genuinely negative fluxes that over-subtracted backgrounds produce.
        Use ``'log'`` only for a strictly positive spectrum, and ``'linear'``
        to reproduce naive behaviour.

    Returns
    -------
    ndarray
        Continuum sampled at ``spectrum.wavelength``. NaN where the spectrum
        has no usable data to constrain it.
    """
    g = spectrum.good
    if mask_ranges:
        w = spectrum.rest_wavelength if rest else spectrum.wavelength
        for lo, hi in mask_ranges:
            g = g & ~((w >= lo) & (w <= hi))
    if g.sum() < 16:
        return np.full(spectrum.wavelength.shape, np.nan)

    w = spectrum.wavelength[g]
    f = spectrum.flux[g]
    fwd, inv = _make_transform(transform, f, spectrum.flux_err[g])

    lg = _log_grid(w, oversample=oversample)
    dln = float(lg[1] - lg[0])
    y = np.interp(lg, np.log(w), fwd(f))

    # np.interp fills masked gaps with a straight chord, and those filler
    # samples are indistinguishable from data to the smoother. Mark the
    # ones that have no retained sample within a grid step so they can be
    # held at zero weight instead of being fitted.
    lw = np.log(w)
    j = np.clip(np.searchsorted(lw, lg), 1, lw.size - 1)
    near = np.minimum(np.abs(lg - lw[j - 1]), np.abs(lg - lw[j]))
    valid = near <= dln

    if method == "percentile":
        z = _rolling_percentile(y, scale_kms, dln,
                                pct=10.0 if direction == "emission"
                                else (90.0 if direction == "absorption" else 50.0))
    else:
        lam = _lambda_for_scale(scale_kms, dln)
        if method == "whittaker" or direction == "symmetric":
            z = whittaker(y, lam, w=valid.astype(float))
        elif direction == "absorption":
            z = -arpls(-y, lam, max_iter=max_iter, valid=valid)
        else:
            z = arpls(y, lam, max_iter=max_iter, valid=valid)

    out = np.full(spectrum.wavelength.shape, np.nan)
    inside = ((spectrum.wavelength >= w.min()) & (spectrum.wavelength <= w.max()))
    out[inside] = inv(np.interp(np.log(spectrum.wavelength[inside]), lg, z))

    if mask_ranges:
        out = _bridge_masked_gaps(spectrum.wavelength, out, g, inside)
    return out


def _make_transform(kind, flux, flux_err=None):
    """Build a monotonic (forward, inverse) flux transform pair.

    See the ``transform`` argument of :func:`estimate_continuum` for why this
    exists. The asinh softening scale is the median 1-sigma uncertainty (or,
    failing that, a small fraction of the robust flux scale): below that scale
    the transform is linear, so noise around zero is not blown up, and above
    it the transform is logarithmic, so the dynamic range is compressed.
    """
    kind = (kind or "linear").lower()
    if kind == "linear":
        return (lambda x: np.asarray(x, dtype=float)), (lambda x: np.asarray(x, dtype=float))

    if kind == "log":
        pos = np.asarray(flux, dtype=float)
        pos = pos[np.isfinite(pos) & (pos > 0)]
        floor = float(np.percentile(pos, 1)) if pos.size else 1e-30
        floor = max(floor, 1e-30)
        return (lambda x: np.log(np.clip(np.asarray(x, dtype=float), floor, None)),
                lambda x: np.exp(np.asarray(x, dtype=float)))

    if kind != "asinh":
        raise ValueError(f"unknown transform {kind!r}")

    s = np.nan
    if flux_err is not None:
        e = np.asarray(flux_err, dtype=float)
        e = e[np.isfinite(e) & (e > 0)]
        if e.size:
            s = float(np.median(e))
    if not np.isfinite(s) or s <= 0:
        f = np.asarray(flux, dtype=float)
        f = f[np.isfinite(f)]
        s = float(0.01 * np.percentile(np.abs(f), 90)) if f.size else 1.0
    s = max(s, 1e-30)
    return (lambda x, s=s: np.arcsinh(np.asarray(x, dtype=float) / s),
            lambda x, s=s: s * np.sinh(np.clip(np.asarray(x, dtype=float), -700, 700)))


def _rolling_percentile(y, scale_kms, dlnlam, pct=10.0):
    """Rolling percentile with a window of ``scale_kms``, then smoothed.

    The raw rolling percentile is a step function (it changes only when a
    point enters or leaves the window), so it is passed through a light
    Whittaker smoother afterwards -- an unsmoothed percentile baseline puts
    small discontinuities into every normalized spectrum built from it.
    """
    n = y.size
    win = int(max(5, round((scale_kms / C_KMS) / dlnlam)))
    win += (win + 1) % 2
    half = win // 2
    pad = np.pad(y, half, mode="edge")
    strided = np.lib.stride_tricks.sliding_window_view(pad, win)
    z = np.percentile(strided, pct, axis=-1)[:n]
    return whittaker(z, _lambda_for_scale(0.3 * scale_kms, dlnlam))


def local_continuum(spectrum: Spectrum, center_um, half_width_um,
                    exclude_half_width_um, order=1, rest=False):
    """Fit a low-order polynomial continuum in a window around one feature,
    excluding the feature itself.

    This is what :mod:`jspec.fitting` uses to flatten the spectrum around a
    line before fitting its profile. A *local* continuum is used rather than
    the global one because a profile fit is sensitive to the continuum at the
    few-percent level, and a global continuum optimised for the whole 5-28 um
    range is not that accurate anywhere in particular.

    Returns
    -------
    (continuum_values, window_mask, coefficients)
        ``continuum_values`` is evaluated over the whole window (including the
        excluded feature region, by extrapolation of the fit).

    Notes
    -----
    ``order`` should stay at 1 or 2. A higher-order local continuum will start
    absorbing the wings of the very feature you are trying to measure, which
    biases the fitted flux low and the fitted width narrow -- the classic way
    to "measure" a line into non-existence.
    """
    w = spectrum.rest_wavelength if rest else spectrum.wavelength
    win = (w >= center_um - half_width_um) & (w <= center_um + half_width_um)
    feat = (w >= center_um - exclude_half_width_um) & (w <= center_um + exclude_half_width_um)
    fit_mask = win & ~feat & spectrum.good
    if fit_mask.sum() < order + 2:
        raise ValueError(
            f"only {int(fit_mask.sum())} continuum points around {center_um:.4f} um; "
            f"widen half_width_um or narrow exclude_half_width_um")
    x = w[fit_mask] - center_um
    y = spectrum.flux[fit_mask]
    e = spectrum.flux_err[fit_mask]
    cf = np.polyfit(x, y, order, w=1.0 / np.clip(e, 1e-30, None))
    return np.polyval(cf, w[win] - center_um), win, cf


# ---------------------------------------------------------------------------
# scale decomposition
# ---------------------------------------------------------------------------
@dataclass
class Decomposition:
    """Additive scale decomposition of a spectrum.

    ``flux == continuum + broad + narrow + residual`` to machine precision at
    every good point. Keeping it exactly additive means a feature's flux can
    be integrated from whichever component it belongs to without worrying
    about double counting.
    """
    spectrum: Spectrum
    continuum: np.ndarray
    broad: np.ndarray
    narrow: np.ndarray
    residual: np.ndarray
    noise: np.ndarray
    broad_scale_kms: float
    narrow_scale_kms: float

    @property
    def broad_spectrum(self) -> Spectrum:
        """The broad component as a :class:`Spectrum`, continuum removed."""
        s = self.spectrum.copy()
        s.flux = self.broad
        return s

    @property
    def narrow_spectrum(self) -> Spectrum:
        """The narrow component as a :class:`Spectrum`, continuum and broad
        component removed."""
        s = self.spectrum.copy()
        s.flux = self.narrow
        return s

    @property
    def normalised(self) -> Spectrum:
        """``flux / continuum`` -- the flattened spectrum, with errors scaled
        consistently. Masked where the continuum is non-positive, because a
        ratio to a negative continuum is meaningless and silently produces
        inverted 'features'."""
        s = self.spectrum.copy()
        bad = ~np.isfinite(self.continuum) | (self.continuum <= 0)
        c = np.where(bad, np.nan, self.continuum)
        s.flux = self.spectrum.flux / c
        s.flux_err = self.spectrum.flux_err / c
        s.mask = s.mask | bad
        s.flux_unit = "normalised"
        return s

    def __repr__(self):
        return (f"<Decomposition broad<{self.broad_scale_kms:.0f} km/s, "
                f"narrow<{self.narrow_scale_kms:.0f} km/s>")


def decompose(spectrum: Spectrum, broad_scale_kms=30000.0,
              narrow_scale_kms=600.0, direction="emission",
              mask_ranges=None, rest=False, noise_window_kms=None,
              method="arpls", transform="asinh",
              continuum=None) -> Decomposition:
    """Split a spectrum into continuum, broad, narrow and residual components.

    The split is a cascade of two smoothers:

    1. ``continuum`` = asymmetric baseline at ``broad_scale_kms``.
    2. ``smooth_resid`` = *symmetric* smoothing of ``flux - continuum`` at
       ``narrow_scale_kms``. Symmetric here, not asymmetric, because the
       residual contains both emission and absorption excursions and an
       asymmetric filter would systematically flatten the dips -- which are
       exactly the absorption features we want to detect.
    3. ``broad`` = ``smooth_resid``; ``narrow`` = what is left after
       additionally smoothing at the resolution scale; ``residual`` = the
       final high-frequency remainder, which is the noise estimate.

    Parameters
    ----------
    noise_window_kms : float, optional
        Window over which the local noise is estimated from the residual by
        MAD. Defaults to 20x ``narrow_scale_kms``, which is wide enough for a
        stable MAD and narrow enough to track the strong wavelength
        dependence of JWST noise across a stitched spectrum. The *measured*
        noise is used in preference to the pipeline error array because
        pipeline errors in these products are frequently optimistic by tens of
        percent, and detection thresholds built on them produce correspondingly
        inflated detection counts.
    continuum : array, optional
        A continuum already computed for this spectrum -- normally from
        :func:`anchored_continuum`. Supplying it skips the internal
        :func:`estimate_continuum` call, so the decomposition and whatever
        produced the continuum cannot disagree about where the baseline
        lies. Must be the same length as ``spectrum``.
    """
    if continuum is None:
        cont = estimate_continuum(spectrum, scale_kms=broad_scale_kms,
                                  method=method, direction=direction,
                                  mask_ranges=mask_ranges, rest=rest,
                                  transform=transform)
    else:
        cont = np.asarray(continuum, dtype=float)
        if cont.size != len(spectrum):
            raise ValueError(
                f"continuum has {cont.size} points but the spectrum has "
                f"{len(spectrum)}")

    g = spectrum.good & np.isfinite(cont)
    n = len(spectrum)
    broad = np.full(n, np.nan)
    narrow = np.full(n, np.nan)
    resid = np.full(n, np.nan)
    noise = np.full(n, np.nan)
    if g.sum() < 16:
        return Decomposition(spectrum, cont, broad, narrow, resid, noise,
                             broad_scale_kms, narrow_scale_kms)

    w = spectrum.wavelength[g]
    d = spectrum.flux[g] - cont[g]

    lg = _log_grid(w, oversample=1.0)
    dln = float(lg[1] - lg[0])
    y = np.interp(lg, np.log(w), d)

    b = whittaker(y, _lambda_for_scale(narrow_scale_kms, dln))
    # Everything finer than the narrow scale, then split again at the
    # resolution scale to separate real narrow lines from pixel-to-pixel noise.
    fine = y - b
    nres = whittaker(fine, _lambda_for_scale(max(narrow_scale_kms / 6.0, 30.0), dln))

    lnw = np.log(w)
    broad[g] = np.interp(lnw, lg, b)
    narrow[g] = np.interp(lnw, lg, nres)
    resid[g] = spectrum.flux[g] - cont[g] - broad[g] - narrow[g]

    nw = noise_window_kms or (20.0 * narrow_scale_kms)
    noise[g] = _local_mad(spectrum.flux[g] - cont[g] - broad[g], w, nw)
    return Decomposition(spectrum, cont, broad, narrow, resid, noise,
                         broad_scale_kms, narrow_scale_kms)


def _local_mad(y, wave, window_kms):
    """Rolling MAD-sigma of ``y`` in a velocity-width window.

    A single global noise estimate is useless on a stitched 1-28 um spectrum
    where the noise varies by more than an order of magnitude between bands
    and jumps at every join; a rolling estimate is the minimum needed for a
    detection threshold that means the same thing everywhere.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 8:
        return np.full(n, np.nan)
    lnw = np.log(wave)
    half = 0.5 * window_kms / C_KMS
    lo = np.searchsorted(lnw, lnw - half, side="left")
    hi = np.searchsorted(lnw, lnw + half, side="right")
    out = np.empty(n)
    for i in range(n):
        a, b = lo[i], max(hi[i], lo[i] + 3)
        seg = y[a:b]
        seg = seg[np.isfinite(seg)]
        if seg.size < 3:
            out[i] = np.nan
        else:
            out[i] = 1.4826 * np.median(np.abs(seg - np.median(seg)))
    # A zero MAD (a flat run of identical values, e.g. a padded region)
    # would make every neighbouring point infinitely significant.
    bad = ~np.isfinite(out) | (out <= 0)
    if bad.all():
        return np.full(n, np.nan)
    out[bad] = np.nanmedian(out[~bad])
    return out


def normalise(spectrum: Spectrum, continuum=None, **kwargs) -> Spectrum:
    """Convenience wrapper: divide by the continuum, estimating it if needed."""
    if continuum is None:
        continuum = estimate_continuum(spectrum, **kwargs)
    s = spectrum.copy()
    bad = ~np.isfinite(continuum) | (continuum <= 0)
    c = np.where(bad, np.nan, continuum)
    s.flux = spectrum.flux / c
    s.flux_err = spectrum.flux_err / c
    s.mask = s.mask | bad
    s.flux_unit = "normalised"
    return s



def _bridge_masked_gaps(wave, cont, fitted, inside):
    """Replace the continuum inside masked gaps with a monotone bridge.

    A difference-penalty smoother does not interpolate across a long gap,
    it *extrapolates* into it from both sides, and the slope at the gap
    edge carries straight through. Across the 2.5 um aromatic+silicate
    mask that sends the baseline from 1.1e-4 at 7.2 um to negative by
    8.6 um -- below zero, and far below the observed flux, which then
    reports the whole band as one flat elevated plateau that the matched
    filter's local baseline cancels out. Masking the band therefore
    failed for the opposite reason to not masking it.

    PCHIP through the anchor values fixes this because it is
    shape-preserving: the bridge cannot overshoot the anchors that
    bracket it, so it stays between the continuum levels either side and
    stays positive if they are. Interpolating in log-log makes the
    default bridge a power law, which is the right prior for a dust
    continuum.
    """
    try:
        from scipy.interpolate import PchipInterpolator
    except Exception:
        return cont

    gap = inside & ~fitted & np.isfinite(wave)
    if not gap.any():
        return cont

    src = fitted & inside & np.isfinite(cont) & (wave > 0)
    if src.sum() < 4:
        return cont

    lw = np.log(wave[src])
    cv = cont[src]
    order = np.argsort(lw)
    lw, cv = lw[order], cv[order]
    lw, uniq = np.unique(lw, return_index=True)
    cv = cv[uniq]
    if lw.size < 4:
        return cont

    # Log-space where the anchors allow it, linear where they do not --
    # a continuum that legitimately crosses zero must not be forced
    # positive just to keep the interpolation in log space.
    positive = np.all(cv > 0)
    yv = np.log(cv) if positive else cv
    bridge = PchipInterpolator(lw, yv, extrapolate=False)(np.log(wave[gap]))
    if positive:
        bridge = np.exp(bridge)

    out = cont.copy()
    idx = np.flatnonzero(gap)
    ok = np.isfinite(bridge)
    out[idx[ok]] = bridge[ok]
    return out


# ------------------------------------------------- anchored continuum ----

def band_mask_ranges(bands=None, kinds=None, pad_frac=0.05,
                     max_width_frac=None, scale_kms=30000.0, redshift=0.0,
                     max_cover_frac=0.55):
    """Observed-frame wavelength ranges of known dust and molecular bands.

    These are the regions a continuum fit must *not* be allowed to follow.
    A generic smoothness prior cannot tell a 1.5 um wide aromatic complex
    from curvature of the underlying dust continuum -- both are smooth on
    the same scale -- so no choice of stiffness separates them. The only
    way out is to say in advance where the bands are, fit the continuum
    through the anchor regions in between, and interpolate across.

    Parameters
    ----------
    max_width_frac : float, optional
        Bands wider than this fraction of their own peak wavelength are
        **not** masked. Defaults to ``scale_kms / c``, which is the honest
        answer: a band broader than the continuum's own stiffness is not
        distinguishable from continuum curvature by this method, and
        masking it would leave no anchors on either side -- the fit there
        would be pure extrapolation, which is worse than the
        feature-absorbing fit it replaced.

        The practical consequence is that the amorphous silicate bands at
        9.7 and 18 um (each roughly half their own peak wavelength wide)
        are treated as continuum at the default stiffness. That is a
        scientific choice and it is reversible: raise ``scale_kms`` to
        make the continuum stiffer and they will be masked and measured as
        features instead.
    max_cover_frac : float
        Upper bound on the fraction of the covered wavelength span that
        may be masked. If the bands exceed it, every block is shrunk about
        its own centre by one common factor. Uniform shrinking is the
        point: it keeps each mask where its band actually is, whereas
        shrinking an over-wide merged block in isolation slides the mask
        off the band entirely.
    scale_kms : float
        The continuum stiffness the mask is being built for.
    redshift : float
        Shifts the (rest-frame) band definitions to the observed frame.
    """
    from .linedb import load_bands
    if bands is None:
        bands = load_bands()
    if kinds is not None:
        kinds = {str(k).lower() for k in kinds}
        bands = bands[bands["kind"].astype(str).str.lower().isin(kinds)]
    if max_width_frac is None:
        # 1.5x the continuum's own stiffness length. The factor matters:
        # at exactly scale_kms / c the cap is 0.10 at the default
        # stiffness, which rejects PAH 7.7 (0.130 of its peak) by a hair
        # and leaves the strongest aromatic band in the mid-infrared
        # unmasked -- the baseline then arches over it and absorbs it,
        # which is precisely the failure this function exists to prevent.
        # At 1.5x the cap is 0.15, which admits PAH 7.7, PAH 8.6, CS and
        # the silica band while still rejecting the plateaux (0.25-0.50)
        # and the featureless carbon continua (1.6-1.8).
        max_width_frac = float(np.clip(1.5 * scale_kms / C_KMS, 0.05, 0.45))

    out = []
    for _, row in bands.iterrows():
        lo, hi = row.get("lam_min_um"), row.get("lam_max_um")
        peak = row.get("lam_peak_um")
        try:
            lo, hi, peak = float(lo), float(hi), float(peak)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            continue
        if not (np.isfinite(peak) and peak > 0):
            peak = 0.5 * (lo + hi)
        if (hi - lo) / peak > max_width_frac:
            # Deliberately skipped, not shrunk. Entries this wide are
            # plateaux and continua rather than bands -- 'PAH plateau 6-9',
            # 'SiO fundamental' (7.5-9.5), 'Amorphous carbon' (2-30) -- and
            # masking even their cores chains every aromatic and silicate
            # entry into one block spanning 5.5-13 um, which the continuum
            # can only bridge by extrapolation. Skipping them is what keeps
            # the individual band masks separated by usable anchors.
            continue
        pad = pad_frac * (hi - lo)
        out.append(((lo - pad) * (1.0 + redshift), (hi + pad) * (1.0 + redshift)))

    # Adjacent bands chain into one block that swallows the spectrum -- the
    # aromatics between 6 and 9 um do exactly this, merging into a single
    # 5.5-12.5 um span. Two things must not happen when that is cut back:
    # the mask must not *move* (shrinking a merged block about its centre
    # relocates it to 9.8-11.3 um, leaving the 7.7 um complex completely
    # unmasked and the baseline free to arch over it), and no single block
    # may be wide enough that bridging it is pure extrapolation.
    #
    # So shrink every block about its own centre by one common factor,
    # chosen as the largest that brings total coverage within
    # max_cover_frac. A uniform factor preserves where the masks are and
    # their relative sizes, and always leaves anchors between them.
    merged = _merge_ranges(out)
    if not merged:
        return merged
    span_lo = min(a for a, _ in merged)
    span_hi = max(b for _, b in merged)
    total = span_hi - span_lo
    covered = sum(b - a for a, b in merged)
    if total > 0 and covered / total > max_cover_frac:
        f = (max_cover_frac * total) / covered
        merged = [(0.5 * (a + b) - 0.5 * f * (b - a),
                   0.5 * (a + b) + 0.5 * f * (b - a)) for a, b in merged]
    return _merge_ranges(merged)


def _merge_ranges(ranges):
    """Union of possibly overlapping (lo, hi) intervals."""
    rs = sorted((float(a), float(b)) for a, b in ranges if b > a)
    if not rs:
        return []
    out = [list(rs[0])]
    for a, b in rs[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(r) for r in out]


def anchored_continuum(spectrum: Spectrum, scale_kms=30000.0, bands=True,
                       band_kinds=None, n_iter=2, zmin=4.0, grow=1.0,
                       max_mask_frac=0.5, scales_kms=None, extra_masks=None,
                       **kwargs) -> np.ndarray:
    """Continuum fitted through anchor regions, with features masked out.

    This is the continuum to use when you care about broad features. It
    differs from :func:`estimate_continuum` in two ways:

    1. Known dust and molecular band windows are excluded from the fit
       (see :func:`band_mask_ranges`).
    2. It iterates. A first pass gives a provisional continuum; features
       detected against it are masked; the continuum is refitted through
       what is left. Two passes are plenty -- the masks stop moving.

    The result is a continuum that passes *under* a broad band instead of
    arching over it, so the band's flux ends up in the feature where it
    belongs rather than being quietly absorbed into the baseline.

    Parameters
    ----------
    grow : float
        Masks are widened by this fraction of each feature's width on each
        side, so the fit is not pulled up by the feature's wings.
    max_mask_frac : float
        Safety valve. If masking would remove more than this fraction of
        the spectrum there are too few anchors left to constrain anything,
        so the feature masks are dropped and only the band masks are kept.
    """
    from .multiscale import significance_bank, extract_peaks, default_scales

    masks = list(extra_masks or [])
    if bands:
        masks += band_mask_ranges(kinds=band_kinds, scale_kms=scale_kms,
                                  redshift=float(getattr(spectrum, "redshift", 0.0) or 0.0))
    masks = _merge_ranges(masks)

    cont = estimate_continuum(spectrum, scale_kms=scale_kms,
                              mask_ranges=masks or None, **kwargs)
    if n_iter <= 0:
        return cont

    if scales_kms is None:
        scales_kms = default_scales(spectrum, max_kms=min(scale_kms * 0.6, 30000.0))

    w = spectrum.wavelength
    total = float(np.nanmax(w) - np.nanmin(w))
    base = list(masks)
    for _ in range(int(n_iter)):
        resid = spectrum.flux - cont
        g = spectrum.good & np.isfinite(resid)
        if g.sum() < 32:
            break
        try:
            lam, Z, D = significance_bank(w[g], resid[g], scales_kms)
        except ValueError:
            break
        found = extract_peaks(lam, Z, D, scales_kms, zmin=zmin, signed=True)
        if not found:
            break
        feat = []
        for f in found:
            half = grow * max(f["hi"] - f["lo"], f["fwhm_um"])
            feat.append((f["lo"] - half, f["hi"] + half))
        trial = _merge_ranges(base + feat)
        covered = sum(b - a for a, b in trial)
        if total > 0 and covered / total > max_mask_frac:
            break
        masks = trial
        cont = estimate_continuum(spectrum, scale_kms=scale_kms,
                                  mask_ranges=masks or None, **kwargs)
    return cont
