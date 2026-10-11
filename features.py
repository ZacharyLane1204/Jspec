"""
jspec.features
==============

Detection and characterisation of broad components, narrow lines, absorption
dips, and high-signal-to-noise regions.

What counts as a detection
--------------------------

A feature is reported when an excursion in one of the scale-separated
components of :func:`jspec.continuum.decompose` exceeds a significance
threshold, where significance is measured against the *empirically estimated*
local noise rather than the pipeline error array. The pipeline errors on these
products are routinely optimistic -- they do not capture residual background
structure, fringe residuals, or the correlated noise introduced by cube
building and extraction -- and a detection list built on them is inflated in
exactly the regime where you most want it to be honest.

Two consequences follow, and both are deliberate:

* **Significance is computed per component, at that component's own effective
  resolution.** The broad component has been smoothed over many pixels, so its
  noise per point is lower than the per-pixel noise by roughly the square root
  of the smoothing length. Comparing a smoothed component against the
  *per-pixel* noise would systematically understate broad-feature significance
  by a factor of several, which is how genuine broad components get missed.
* **Features are tested against segment joins.** A residual calibration step
  at a band join looks exactly like a broad feature to any detection
  algorithm. Every reported feature carries ``on_join``, set when its extent
  straddles a boundary between instrument segments, and ``join_fraction``, the
  fraction of its extent inside a blended overlap region. Treat any
  high-``join_fraction`` "feature" as a stitching artefact until proven
  otherwise -- this is the single most common false positive in multi-band
  JWST spectroscopy, and it is the reason :mod:`jspec.stitch` labels every
  point with its segment.

Widths and velocities
---------------------

Every width is reported three ways: in microns, as an observed velocity FWHM,
and as an *intrinsic* velocity FWHM with the instrumental line-spread function
removed in quadrature. A feature whose observed width is at or below the local
resolution is labelled ``unresolved`` and its intrinsic width is reported as an
upper limit, not as a number -- deconvolving a marginally-resolved width gives
a precise-looking value that is dominated by how well you happen to know R.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from .continuum import Decomposition, decompose
from .spectrum import C_KMS, Spectrum, instrumental_sigma_kms, resolving_power

__all__ = [
    "Feature",
    "Region",
    "find_features",
    "find_dips",
    "high_snr_regions",
    "identify",
    "features_table",
    "features_from_table",
]

_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass
class Feature:
    """One detected spectral feature.

    Attributes
    ----------
    center, rest_center : float
        Flux-weighted centroid in microns, observed and rest frame. The
        centroid is used rather than the peak pixel because for an asymmetric
        or noisy profile the peak pixel is a poor and quantised estimator.
    center_err : float
        Statistical centroid uncertainty from the noise.
    lo, hi : float
        Extent of the feature in observed microns.
    component : {'broad', 'narrow'}
        Which scale component it was detected in.
    sign : {'emission', 'absorption'}
        ``absorption`` is a dip below the continuum.
    peak_snr : float
        Significance at the extremum, against the component's effective noise.
    integrated_snr : float
        Significance of the integrated flux. For a well-sampled feature this
        is the number that matters; ``peak_snr`` can be high for a single hot
        pixel that carries no integrated flux.
    flux : float
        Integrated flux of the feature above (or below) the continuum, in
        ``flux_unit * um``.
    eqw_um : float
        Equivalent width in microns. Positive for emission.
    fwhm_um, fwhm_kms : float
        Observed width.
    intrinsic_fwhm_kms : float
        Width with the instrumental LSF removed in quadrature, or NaN when the
        feature is unresolved.
    width_class : {'unresolved', 'narrow', 'broad'}
    resolution_fwhm_kms : float
        Local instrumental FWHM, for reference.
    on_join : bool
    join_fraction : float
        Fraction of the feature's extent lying in a stitched overlap region.
    segments : tuple of str
        Which instrument segments the feature spans.
    matches : list
        Candidate identifications, filled in by :func:`identify`.
    bands : list of str
        Dust/molecular bands whose wavelength bracket contains this feature.
    """
    center: float
    rest_center: float
    center_err: float
    lo: float
    hi: float
    component: str
    sign: str
    peak_snr: float
    integrated_snr: float
    amplitude: float
    flux: float
    eqw_um: float
    fwhm_um: float
    fwhm_kms: float
    intrinsic_fwhm_kms: float
    width_class: str
    resolution_fwhm_kms: float
    continuum_level: float
    on_join: bool = False
    join_fraction: float = 0.0
    segments: tuple = ()
    matches: list = field(default_factory=list)
    bands: list = field(default_factory=list)

    @property
    def best_match(self):
        return self.matches[0] if self.matches else None

    def __repr__(self):
        bm = self.best_match
        tag = f" ~ {bm.label} ({bm.velocity_kms:+.0f} km/s)" if bm else ""
        warn = " [ON JOIN]" if self.on_join else ""
        return (f"<{self.sign[:3]} {self.component} {self.rest_center:.4f}um "
                f"S/N={self.integrated_snr:.1f} FWHM={self.fwhm_kms:.0f}km/s "
                f"({self.width_class}){tag}{warn}>")


@dataclass
class Region:
    """A contiguous stretch of spectrum exceeding a S/N threshold."""
    lo: float
    hi: float
    rest_lo: float
    rest_hi: float
    peak_snr: float
    median_snr: float
    peak_wavelength: float
    n_points: int
    segments: tuple = ()
    features: list = field(default_factory=list)
    bands: list = field(default_factory=list)

    @property
    def width_um(self) -> float:
        return self.hi - self.lo

    def __repr__(self):
        return (f"<Region {self.lo:.3f}-{self.hi:.3f}um peak S/N={self.peak_snr:.1f} "
                f"@{self.peak_wavelength:.4f} ({len(self.features)} features)>")


# ---------------------------------------------------------------------------
# effective noise of a smoothed component
# ---------------------------------------------------------------------------
def _effective_noise(dec: Decomposition, component: str) -> np.ndarray:
    """Noise appropriate to a scale-filtered component.

    ``dec.noise`` is a per-pixel noise estimate. A component that has been
    smoothed over roughly ``N`` pixels has had its noise reduced by about
    ``sqrt(N)``, so testing the smoothed component against the per-pixel noise
    would understate its significance by that factor -- for the broad
    component on MRS data, typically five to fifteen.

    The reduction factor is derived from the smoothing length in *velocity*
    converted to pixels using the local sampling, so it adapts across a
    stitched spectrum where the pixel size changes by an order of magnitude
    between bands.

    This is an approximation: a Whittaker smoother is not a boxcar, and the
    real noise reduction depends on its transfer function and on how
    correlated the input noise already was. It is accurate to tens of percent,
    which is adequate for a detection threshold, and it errs toward
    *over*-estimating the noise (hence under-reporting significance) because
    the input pixel noise is itself correlated. Treat the reported S/N of a
    broad component as good to a factor ~1.5, not as a precise number.
    """
    w = dec.spectrum.wavelength
    n = len(w)
    noise = dec.noise.copy()
    if n < 3:
        return noise

    # local fractional pixel size d(lambda)/lambda
    dln = np.gradient(np.log(np.clip(w, 1e-12, None)))
    dln = np.abs(dln)
    dln[~np.isfinite(dln) | (dln <= 0)] = np.nanmedian(dln[dln > 0]) if np.any(dln > 0) else 1e-4

    scale = (dec.broad_scale_kms if component == "broad" else dec.narrow_scale_kms)
    if component == "broad":
        # the broad component retains structure between narrow_scale and
        # broad_scale; its effective smoothing length is the narrow scale
        scale = dec.narrow_scale_kms
    else:
        scale = max(dec.narrow_scale_kms / 6.0, 30.0)

    npix = np.clip((scale / C_KMS) / dln, 1.0, None)
    return noise / np.sqrt(npix)


def _join_info(spectrum: Spectrum, lo_idx: int, hi_idx: int):
    """Which segments a span covers, and how much of it sits inside a blended
    overlap (a label containing '+')."""
    if spectrum.segment is None:
        return False, 0.0, ()
    seg = spectrum.segment[lo_idx:hi_idx + 1]
    if seg.size == 0:
        return False, 0.0, ()
    labs = [str(s) for s in seg]
    blended = np.array(["+" in s for s in labs])
    distinct = set()
    for s in labs:
        distinct.update(s.split("+"))
    frac = float(blended.mean())
    on_join = bool(frac > 0.0 or len(distinct) > 1)
    return on_join, frac, tuple(sorted(distinct))


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def _measure(spectrum, dec, comp, eff_noise, i_peak, sign, snr_min,
             component_name, min_points, mode):
    """Characterise one peak in ``comp``; returns a Feature or None."""
    w = spectrum.wavelength
    n = w.size
    sgn = 1.0 if sign == "emission" else -1.0
    y = sgn * comp
    sig = y / np.where(eff_noise > 0, eff_noise, np.inf)

    # Extent: walk out until the significance drops below half the threshold
    # or the signal changes sign. Half-threshold (not zero-crossing) keeps a
    # feature from absorbing its neighbours across a noisy baseline.
    stop = max(0.5 * snr_min, 1.0)
    i_lo = i_peak
    while i_lo > 0 and sig[i_lo - 1] > stop and y[i_lo - 1] > 0:
        i_lo -= 1
    i_hi = i_peak
    while i_hi < n - 1 and sig[i_hi + 1] > stop and y[i_hi + 1] > 0:
        i_hi += 1
    if (i_hi - i_lo + 1) < min_points:
        return None

    sl = slice(i_lo, i_hi + 1)
    ww, yy = w[sl], y[sl]
    nn = eff_noise[sl]
    cont = dec.continuum[sl]
    if ww.size < 2 or not np.isfinite(yy).any():
        return None

    wt = np.clip(yy, 0.0, None)
    swt = wt.sum()
    if swt <= 0:
        return None
    centre = float(np.sum(ww * wt) / swt)
    # Centroid uncertainty by propagating the noise through the weighted mean.
    with np.errstate(invalid="ignore"):
        dc = np.sqrt(np.sum(((ww - centre) * nn) ** 2)) / swt
    centre_err = float(dc) if np.isfinite(dc) else np.nan

    flux = float(np.trapezoid(yy, ww)) if hasattr(np, "trapezoid") else float(np.trapz(yy, ww))
    # Integrated noise. Neighbouring points in a smoothed component are
    # strongly correlated, so adding their variances in quadrature as if
    # independent would badly overstate the integrated significance. The
    # correlation length is taken as the component's own smoothing length.
    dw = np.gradient(ww)
    var_terms = (nn * dw) ** 2
    corr_len = max(1.0, ww.size / max(_independent_samples(ww, dec, component_name), 1.0))
    flux_err = float(np.sqrt(np.sum(var_terms) * corr_len))
    integrated_snr = abs(flux) / flux_err if flux_err > 0 else np.inf

    amp = float(np.max(yy))
    peak_snr = float(np.max(sig[sl]))

    # FWHM from half-maximum crossings of the component, interpolated.
    half = 0.5 * amp
    above = yy >= half
    if above.sum() >= 2:
        idx = np.flatnonzero(above)
        a, b = idx[0], idx[-1]
        wl = _cross(ww, yy, a, half, -1)
        wr = _cross(ww, yy, b, half, +1)
        fwhm_um = float(wr - wl)
    else:
        fwhm_um = float(ww[-1] - ww[0])
    if not np.isfinite(fwhm_um) or fwhm_um <= 0:
        fwhm_um = float(np.gradient(ww).mean())

    fwhm_kms = C_KMS * fwhm_um / centre
    r = float(resolving_power(centre, mode=mode))
    res_fwhm = C_KMS / r if np.isfinite(r) and r > 0 else np.nan

    if np.isfinite(res_fwhm) and fwhm_kms <= 1.2 * res_fwhm:
        width_class = "unresolved"
        intrinsic = np.nan
    else:
        intrinsic = (np.sqrt(max(fwhm_kms ** 2 - res_fwhm ** 2, 0.0))
                     if np.isfinite(res_fwhm) else fwhm_kms)
        width_class = "broad" if intrinsic >= 1000.0 else "narrow"

    cmed = float(np.nanmedian(cont))
    eqw = float(flux / cmed) if np.isfinite(cmed) and cmed != 0 else np.nan

    on_join, jfrac, segs = _join_info(spectrum, i_lo, i_hi)
    z = spectrum.redshift
    return Feature(
        center=centre, rest_center=centre / (1.0 + z), center_err=centre_err,
        lo=float(ww[0]), hi=float(ww[-1]), component=component_name, sign=sign,
        peak_snr=peak_snr, integrated_snr=float(integrated_snr),
        amplitude=sgn * amp, flux=sgn * flux, eqw_um=sgn * eqw,
        fwhm_um=fwhm_um, fwhm_kms=float(fwhm_kms),
        intrinsic_fwhm_kms=float(intrinsic) if np.isfinite(intrinsic) else np.nan,
        width_class=width_class,
        resolution_fwhm_kms=float(res_fwhm) if np.isfinite(res_fwhm) else np.nan,
        continuum_level=cmed, on_join=on_join, join_fraction=jfrac, segments=segs)


def _independent_samples(ww, dec, component_name):
    """How many independent resolution elements a span contains."""
    if ww.size < 2:
        return 1.0
    span_kms = C_KMS * (ww[-1] - ww[0]) / np.mean(ww)
    scale = (dec.narrow_scale_kms if component_name == "broad"
             else max(dec.narrow_scale_kms / 6.0, 30.0))
    return max(span_kms / scale, 1.0)


def _cross(w, y, i, level, direction):
    """Linearly interpolated wavelength where ``y`` crosses ``level`` just
    outside index ``i`` in ``direction``."""
    j = i + direction
    if j < 0 or j >= w.size:
        return w[i]
    y0, y1 = y[i], y[j]
    if y1 == y0:
        return w[i]
    t = (level - y0) / (y1 - y0)
    t = np.clip(t, 0.0, 1.0)
    return w[i] + t * (w[j] - w[i])


def find_features(spectrum: Spectrum, decomposition: Decomposition | None = None,
                  components=("broad", "narrow"), signs=("emission", "absorption"),
                  snr_min=5.0, min_points=4, min_separation_kms=None,
                  mode=None, exclude_joins=False, max_join_fraction=1.0,
                  **decompose_kwargs) -> list[Feature]:
    """Detect features in a spectrum.

    Parameters
    ----------
    decomposition : Decomposition, optional
        Reuse a decomposition instead of recomputing it. Pass one when you
        are calling this repeatedly with different thresholds -- the
        decomposition is the expensive part.
    components : tuple
        Which scale components to search. Searching ``'broad'`` finds ejecta
        lines and band complexes; ``'narrow'`` finds host/CSM and unresolved
        lines.
    signs : tuple
        ``'emission'``, ``'absorption'``, or both. Absorption detections are
        the dips.
    snr_min : float
        Peak significance threshold, against the component's effective noise.
    min_separation_kms : float, optional
        Minimum separation between peaks in the same component. Defaults to
        the component's own smoothing scale, which is the smallest separation
        at which two peaks are actually distinguishable after filtering --
        allowing closer peaks just reports the same feature twice.
    exclude_joins : bool
        Drop features flagged ``on_join``. Off by default: a real line can
        legitimately fall at a band join, and silently deleting those creates
        wavelength ranges where nothing is ever detected. Prefer inspecting
        ``join_fraction``.
    max_join_fraction : float
        Drop features with more than this fraction of their extent inside a
        blended overlap. Setting this to ~0.5 is a good compromise: it removes
        features that exist *only* in the blend (stitching artefacts) while
        keeping real lines that merely overlap a join.

    Returns
    -------
    list of Feature, sorted by rest wavelength.
    """
    dec = decomposition or decompose(spectrum, **decompose_kwargs)
    out = []
    for component_name in components:
        comp = dec.broad if component_name == "broad" else dec.narrow
        if not np.isfinite(comp).any():
            continue
        eff = _effective_noise(dec, component_name)
        scale = (dec.narrow_scale_kms if component_name == "broad"
                 else max(dec.narrow_scale_kms / 6.0, 30.0))
        sep_kms = min_separation_kms if min_separation_kms is not None else scale

        for sign in signs:
            sgn = 1.0 if sign == "emission" else -1.0
            y = sgn * comp
            sig = np.where(eff > 0, y / eff, 0.0)
            sig = np.where(np.isfinite(sig), sig, 0.0)
            # distance in samples corresponding to sep_kms, using median sampling
            dln = np.median(np.abs(np.gradient(np.log(np.clip(
                spectrum.wavelength, 1e-12, None)))))
            dist = max(int(round((sep_kms / C_KMS) / dln)), 1) if dln > 0 else 1
            peaks, _ = find_peaks(sig, height=snr_min, distance=dist)
            for i in peaks:
                f = _measure(spectrum, dec, comp, eff, int(i), sign, snr_min,
                             component_name, min_points, mode)
                if f is None:
                    continue
                if exclude_joins and f.on_join:
                    continue
                if f.join_fraction > max_join_fraction:
                    continue
                out.append(f)

    out = _dedupe(out)
    out.sort(key=lambda f: f.rest_center)
    return out


def _dedupe(features, overlap_frac=0.7):
    """Remove duplicate detections of the same feature.

    The same physical line can be picked up in both the broad and narrow
    components (a moderately broad line leaks into both), and as both an
    emission peak and the shoulder of a neighbouring dip. Where two detections
    overlap by more than ``overlap_frac`` of the narrower one's extent and
    share a sign, the more significant one is kept.
    """
    if not features:
        return []
    feats = sorted(features, key=lambda f: -f.integrated_snr)
    kept = []
    for f in feats:
        dup = False
        for k in kept:
            if k.sign != f.sign:
                continue
            lo, hi = max(f.lo, k.lo), min(f.hi, k.hi)
            if hi <= lo:
                continue
            ov = (hi - lo) / max(min(f.hi - f.lo, k.hi - k.lo), 1e-12)
            if ov >= overlap_frac:
                dup = True
                break
        if not dup:
            kept.append(f)
    return kept


def find_dips(spectrum, **kwargs) -> list[Feature]:
    """Detect absorption features only. Thin wrapper on
    :func:`find_features` with ``signs=('absorption',)``."""
    kwargs["signs"] = ("absorption",)
    return find_features(spectrum, **kwargs)


# ---------------------------------------------------------------------------
# high signal-to-noise regions
# ---------------------------------------------------------------------------
def high_snr_regions(spectrum: Spectrum, snr_min=10.0, min_width_kms=500.0,
                     use="feature", decomposition=None, features=None,
                     bands=None, merge_gap_kms=None, **decompose_kwargs) -> list[Region]:
    """Find contiguous stretches of high signal-to-noise.

    Parameters
    ----------
    use : {'feature', 'flux'}
        ``'feature'`` (default) thresholds on the significance of departures
        from the continuum -- i.e. it finds regions where there is *something
        to measure*. ``'flux'`` thresholds on the raw flux signal-to-noise,
        which on a bright thermal continuum is high everywhere and therefore
        says nothing about where the interesting structure is. Use
        ``'flux'`` only when you specifically want to know where the continuum
        itself is well measured, e.g. before fitting a dust model.
    min_width_kms : float
        Regions narrower than this are discarded, which removes single-pixel
        spikes without needing a separate cosmic-ray filter.
    merge_gap_kms : float, optional
        Regions separated by less than this are merged. Defaults to
        ``min_width_kms``; prevents a single feature with a noisy core from
        being reported as two regions.
    features, bands : optional
        If given, each returned region is annotated with the features and
        dust bands it contains.
    """
    dec = decomposition or decompose(spectrum, **decompose_kwargs)
    w = spectrum.wavelength

    if use == "flux":
        snr = spectrum.snr
    else:
        eff = _effective_noise(dec, "broad")
        dev = np.abs(spectrum.flux - dec.continuum)
        snr = np.where(eff > 0, dev / eff, np.nan)
    snr = np.where(np.isfinite(snr), snr, 0.0)
    snr = np.where(spectrum.good, snr, 0.0)

    above = snr >= snr_min
    if not above.any():
        return []

    merge_gap_kms = merge_gap_kms if merge_gap_kms is not None else min_width_kms
    edges = np.flatnonzero(np.diff(above.astype(int)))
    starts = list(edges[::2] + 1) if not above[0] else [0] + list(edges[1::2] + 1)
    # Simpler and less error-prone: derive runs directly.
    runs = []
    i = 0
    n = above.size
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1

    merged = []
    for a, b in runs:
        if merged:
            pa, pb = merged[-1]
            gap_kms = C_KMS * (w[a] - w[pb]) / w[pb]
            if gap_kms <= merge_gap_kms:
                merged[-1] = (pa, b)
                continue
        merged.append((a, b))

    out = []
    for a, b in merged:
        width_kms = C_KMS * (w[b] - w[a]) / (0.5 * (w[a] + w[b]))
        if width_kms < min_width_kms:
            continue
        k = a + int(np.argmax(snr[a:b + 1]))
        _, _, segs = _join_info(spectrum, a, b)
        z = spectrum.redshift
        reg = Region(lo=float(w[a]), hi=float(w[b]),
                     rest_lo=float(w[a] / (1 + z)), rest_hi=float(w[b] / (1 + z)),
                     peak_snr=float(snr[k]),
                     median_snr=float(np.median(snr[a:b + 1])),
                     peak_wavelength=float(w[k]), n_points=int(b - a + 1),
                     segments=segs)
        if features:
            reg.features = [f for f in features if f.center >= reg.lo and f.center <= reg.hi]
        if bands is not None:
            sel = ((bands["lam_min_um"] <= reg.rest_hi)
                   & (bands["lam_max_um"] >= reg.rest_lo))
            reg.bands = list(bands.loc[sel, "name"])
        out.append(reg)
    out.sort(key=lambda r: r.lo)
    return out


# ---------------------------------------------------------------------------
# identification
# ---------------------------------------------------------------------------
def identify(features, lines=None, bands=None, tol_kms=None,
             tol_res_elements=2.0, use_width_class=True, mode=None,
             categories=None, exclude_categories=None, max_candidates=8):
    """Attach candidate line identifications and band memberships to features.

    Modifies the features in place (filling ``matches`` and ``bands``) and
    returns them, so it can be chained.

    ``use_width_class`` restricts broad features to lines that can be broad
    and narrow features to lines that can be narrow -- see
    :mod:`jspec.linedb`. Switch it off when searching for something unusual,
    e.g. genuinely broad H2, but be aware the candidate lists get much longer
    and correspondingly less informative.

    For a *broad* feature the default tolerance is widened to the feature's own
    half-width rather than the instrument resolution: the centroid of a
    5000 km/s ejecta line is not known to two MRS resolution elements, it is
    known to a fraction of its own width, and using the instrumental tolerance
    would reject the correct identification of every asymmetric line.
    """
    from .linedb import bands_covering, load_bands, load_lines, match_line

    if lines is None:
        lines = load_lines()
    if bands is None:
        bands = load_bands()

    for f in features:
        tol = tol_kms
        if tol is None and f.width_class == "broad" and np.isfinite(f.fwhm_kms):
            tol = max(0.5 * f.fwhm_kms, tol_res_elements * (f.resolution_fwhm_kms
                                                            if np.isfinite(f.resolution_fwhm_kms)
                                                            else 300.0))
        f.matches = match_line(
            f.rest_center, lines=lines, observed_wavelength_um=f.center,
            tol_kms=tol, tol_res_elements=tol_res_elements,
            width=f.width_class if use_width_class else None,
            categories=categories, exclude_categories=exclude_categories,
            mode=mode, max_candidates=max_candidates)
        f.bands = list(bands_covering(f.rest_center, bands=bands)["name"])
    return features


def features_table(features):
    """Build an astropy Table of detected features, suitable for writing to
    ecsv/csv. One row per feature; the best identification and the number of
    alternatives are included so the table stays honest about ambiguity."""
    from astropy.table import Table

    rows = []
    for f in features:
        bm = f.best_match
        rows.append(dict(
            rest_center_um=f.rest_center, center_um=f.center,
            center_err_um=f.center_err, lo_um=f.lo, hi_um=f.hi,
            component=f.component, sign=f.sign,
            peak_snr=f.peak_snr, integrated_snr=f.integrated_snr,
            flux=f.flux, eqw_um=f.eqw_um,
            fwhm_um=f.fwhm_um, fwhm_kms=f.fwhm_kms,
            intrinsic_fwhm_kms=f.intrinsic_fwhm_kms,
            width_class=f.width_class, resolution_fwhm_kms=f.resolution_fwhm_kms,
            on_join=f.on_join, join_fraction=f.join_fraction,
            segments="+".join(f.segments),
            best_id=bm.label if bm else "",
            best_id_species=bm.species if bm else "",
            best_id_velocity_kms=bm.velocity_kms if bm else np.nan,
            n_candidates=len(f.matches),
            bands=";".join(f.bands),
        ))
    if not rows:
        return Table()
    return Table(rows)


def features_from_table(table):
    """Rebuild :class:`Feature` objects from a table written by
    :func:`features_table`, or from the path to one.

    ``jspec detect`` saves its results, and the obvious next thing to do
    is plot them -- but the plotting functions take Feature objects, not
    a table, so without this you would have to re-run the detection just
    to draw it. Two fields (``amplitude`` and ``continuum_level``) are
    not written to the table and come back as NaN; nothing in plotting
    or identification uses them.

    The best identification is restored as a single-element ``matches``
    list, which is what ``best_match`` and the plot labels read. The
    rejected alternatives are not in the table, so ``n_candidates`` is
    preserved in ``_n_candidates`` rather than being faked.
    """
    from astropy.table import Table

    from .linedb import LineMatch

    if isinstance(table, (str, os.PathLike)):
        table = Table.read(table)
    if not len(table):
        return []

    def col(row, name, default=np.nan):
        try:
            v = row[name]
        except KeyError:
            return default
        return default if v is None else v

    out = []
    for row in table:
        label = str(col(row, "best_id", "") or "")
        matches = []
        if label:
            rest_um = float(col(row, "rest_center_um"))
            v = float(col(row, "best_id_velocity_kms"))
            # The table stores the line's velocity offset, not its
            # laboratory wavelength, so recover the latter from the two.
            lab = rest_um / (1.0 + v / C_KMS) if np.isfinite(v) else rest_um
            matches.append(LineMatch(
                label=label, species=str(col(row, "best_id_species", "") or ""),
                rest_um=lab, observed_rest_um=rest_um, velocity_kms=v,
                delta_um=rest_um - lab, category="", kind="",
                expected_width=str(col(row, "width_class", "") or "")))

        segs = str(col(row, "segments", "") or "")
        bands = str(col(row, "bands", "") or "")
        f = Feature(
            center=float(col(row, "center_um")),
            rest_center=float(col(row, "rest_center_um")),
            center_err=float(col(row, "center_err_um")),
            lo=float(col(row, "lo_um")), hi=float(col(row, "hi_um")),
            component=str(col(row, "component", "") or ""),
            sign=str(col(row, "sign", "") or ""),
            peak_snr=float(col(row, "peak_snr")),
            integrated_snr=float(col(row, "integrated_snr")),
            amplitude=np.nan,
            flux=float(col(row, "flux")),
            eqw_um=float(col(row, "eqw_um")),
            fwhm_um=float(col(row, "fwhm_um")),
            fwhm_kms=float(col(row, "fwhm_kms")),
            intrinsic_fwhm_kms=float(col(row, "intrinsic_fwhm_kms")),
            width_class=str(col(row, "width_class", "") or ""),
            resolution_fwhm_kms=float(col(row, "resolution_fwhm_kms")),
            continuum_level=np.nan,
            on_join=bool(col(row, "on_join", False)),
            join_fraction=float(col(row, "join_fraction", 0.0)),
            segments=tuple(x for x in segs.split("+") if x),
            matches=matches,
            bands=[x for x in bands.split(";") if x])
        f._n_candidates = int(col(row, "n_candidates", len(matches)) or 0)
        out.append(f)
    return out
