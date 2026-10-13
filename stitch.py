"""
jspec.stitch
============

Segment matching and stitching, rebuilt to remove the two things that produce
visible jumps in the existing sequential merger.

Why the old approach leaves steps
---------------------------------

``combine_epoch_spectra.merge_pair_adaptive`` walks bluest-first, rescaling
only the newly added segment onto the frozen composite. That one-directional
rule was the right fix for its own failure mode (a noisy red overlap
retroactively rescaling everything blue of it), but it has two consequences
that show up as steps:

1. **Error accumulates monotonically down the chain.** Each join's scale is
   measured against a composite that already carries every earlier join's
   error, and nothing downstream can ever revise an earlier decision. By
   ch4 on a twelve-band MRS spectrum the composite's blue end and red end
   have been tied together through eleven independent, never-revisited
   ratio measurements. When one of those ratios is rejected by the guard
   rails, the segments on either side of it are simply never tied together
   at all -- and *that* is the jump.

2. **Hard inverse-variance blending in the overlap is not continuous.**
   Inside the overlap the composite is a variance-weighted mean of two
   segments; immediately outside it, it is one segment alone. If the two
   segments disagree by even a few percent after scaling -- which they will,
   because a single scalar cannot absorb a wavelength-dependent mismatch --
   the composite steps discontinuously at *both* edges of every overlap,
   not at the middle. Those are the small sawtooth steps at band joins.

What this module does instead
-----------------------------

**Global scale solve.** Every overlapping pair contributes a log-ratio
constraint ``s_i - s_j = -ln R_ij`` with its own weight; all of them are
solved simultaneously by weighted least squares for the per-segment log
scales ``s``. No ordering, no freezing, no accumulation: a well-measured
ch1/ch2 overlap and a well-measured ch3/ch4 overlap both constrain the same
global solution, and a single bad overlap is outvoted instead of poisoning
everything red of it. The anchor segment (default: the highest-S/N one) is
held at unit scale so the solution is unique.

**Gap bridging.** Adjacent segments that do *not* overlap -- the case that
makes NIRSpec-to-MIRI joins occasionally misbehave -- still get a constraint,
by linearly extrapolating each side into the gap and comparing at the gap
midpoint, with an uncertainty inflated by the extrapolation distance. A weak,
honest constraint is much better than no constraint, which is what the
sequential merger falls back to.

**Edge trimming before measuring.** The outermost few percent of any MRS
sub-band or NIRSpec grating is where throughput rolls off, the noise inflates,
and the flux calibration is least reliable. Those pixels dominate the
pointwise ratio in a narrow overlap precisely because a narrow overlap *is*
mostly band edge. They are excluded from the ratio measurement by default.

**Tapered blending.** In the overlap, segment weights are the inverse variance
multiplied by a raised-cosine ramp that is 1 at that segment's interior edge
and 0 at its outer edge. The composite therefore equals the blue segment
exactly at the blue edge of the overlap and the red segment exactly at the
red edge, with a smooth transition between -- continuous by construction, so
residual disagreement becomes a gentle slope across the overlap instead of
two steps at its edges.

**Diagnostics that name the problem.** :func:`stitch` returns a
:class:`StitchReport` whose ``joins`` list records, per join, the measured
ratio, whether it was used, why not if not, and -- after stitching -- the
residual step left at the join. :func:`residual_step` is the same local
linear + curvature test used for the NIRSpec/MIRI boundary, applied to every
join, so "did stitching work" has a number attached to it instead of being an
eyeball judgement.
"""
from __future__ import annotations

import textwrap
import warnings
from dataclasses import dataclass, field

import numpy as np

from .spectrum import C_KMS, Segment, Spectrum

__all__ = [
    "OverlapMeasurement",
    "SegmentQuality",
    "JoinReport",
    "StitchReport",
    "measure_overlap",
    "assess_segment",
    "solve_scales",
    "stitch",
    "residual_step",
]


# ---------------------------------------------------------------------------
# robust helpers
# ---------------------------------------------------------------------------
def _short(name, width):
    """Shorten a long segment name from the left, keeping the tail.

    Segment names are built from tag + date + instrument, so the part
    that distinguishes two of them is almost always at the end.
    """
    name = str(name)
    return name if len(name) <= width else "..." + name[-(width - 3):]


def _mad_sigma(x):
    """Median absolute deviation rescaled to a Gaussian sigma. Used instead of
    the standard deviation everywhere a few bad pixels could otherwise set the
    scale of the problem."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _trimmed_range(wave, trim_frac, trim_min_um=0.0):
    """Interior wavelength range of a segment after removing ``trim_frac`` of
    its span from each end (at least ``trim_min_um``)."""
    lo, hi = float(np.min(wave)), float(np.max(wave))
    span = hi - lo
    cut = max(trim_frac * span, trim_min_um)
    if 2 * cut >= span:         # degenerate: segment is shorter than the trim
        cut = 0.25 * span
    return lo + cut, hi - cut


def _local_line(wave, flux, err, lo, hi, centre, min_points=8):
    """Weighted straight-line fit of flux vs wavelength over ``[lo, hi]``,
    evaluated at ``centre``.

    Returns ``(value_at_centre, sigma, slope, curvature_significance)`` or
    None. ``curvature_significance`` is |c2| / sigma_c2 from a quadratic fit:
    a large value means the spectrum is genuinely curved here (a line wing, a
    PAH shoulder, the silicate feature), so a *step* measured across this
    region is partly real spectral shape and should not be fully absorbed
    into a calibration scale.
    """
    m = (wave >= lo) & (wave <= hi) & np.isfinite(flux)
    if m.sum() < min_points:
        return None
    x = wave[m] - centre
    y = flux[m]
    w = np.ones_like(y)
    if err is not None:
        e = err[m]
        ok = np.isfinite(e) & (e > 0)
        if ok.sum() >= min_points:
            x, y, w = x[ok], y[ok], 1.0 / e[ok] ** 2
        else:
            return None

    A = np.vander(x, 2)
    W = np.diag(w)
    try:
        cov = np.linalg.inv(A.T @ W @ A)
    except np.linalg.LinAlgError:
        return None
    p = cov @ (A.T @ W @ y)
    resid = y - A @ p
    # Rescale the formal covariance by the actual scatter: the pipeline error
    # arrays are frequently optimistic, and an overconfident sigma here turns
    # directly into an overconfident "this is a real step" verdict.
    dof = max(len(y) - 2, 1)
    chi2_red = float(np.sum(w * resid ** 2) / dof)
    cov = cov * max(chi2_red, 1.0)
    value, sigma, slope = float(p[1]), float(np.sqrt(cov[1, 1])), float(p[0])

    curv_sig = np.nan
    if len(y) >= 12:
        A2 = np.vander(x, 3)
        try:
            cov2 = np.linalg.inv(A2.T @ W @ A2)
            p2 = cov2 @ (A2.T @ W @ y)
            r2 = y - A2 @ p2
            chi2r2 = float(np.sum(w * r2 ** 2) / max(len(y) - 3, 1))
            cov2 = cov2 * max(chi2r2, 1.0)
            curv_sig = float(abs(p2[0]) / np.sqrt(cov2[0, 0]))
        except np.linalg.LinAlgError:
            pass
    return value, sigma, slope, curv_sig


# ---------------------------------------------------------------------------
# per-segment quality: is there a source in this segment at all?
# ---------------------------------------------------------------------------
@dataclass
class SegmentQuality:
    """What one segment looks like on its own, before any stitching.

    The stitcher is good at matching two segments that both contain the
    source. It is equally good at matching a segment that contains the
    source to one that does not -- it will happily find a scale factor
    that lifts pure background noise onto a real continuum, and the
    result looks entirely plausible. This is the guard against that.
    """
    name: str
    n_points: int = 0
    median_snr: float = np.nan       # median of flux/err
    mean_snr: float = np.nan         # significance of the weighted mean flux
    frac_positive: float = np.nan    # fraction of points with flux > 0
    chi2_const: float = np.nan       # chi2/dof of a constant model, binned
    median_flux: float = np.nan
    flat: bool = False               # consistent with a constant
    blank: bool = False              # consistent with no source
    negative: bool = False           # flux sits below zero over most of the span
    #: Size of a negative pedestal as a fraction of the neighbouring
    #: segment's flux. Distinguishes "faint segment with a small offset"
    #: from "extraction failed"; NaN when there is no neighbour to compare to.
    pedestal_frac: float = np.nan
    reason: str = ""

    @property
    def suspect(self) -> bool:
        return self.blank or self.negative

    def __repr__(self):
        tag = ("BLANK" if self.blank else "NEGATIVE" if self.negative
               else "flat" if self.flat else "ok")
        return (f"<SegmentQuality {self.name}: {tag}, "
                f"median S/N={self.median_snr:.2f}, chi2/dof={self.chi2_const:.2f}>")


def assess_segment(seg: Segment, *, n_bins=20, min_points=30,
                   snr_floor=3.0, flat_chi2=3.0, mean_sigma_floor=5.0,
                   neg_snr=-3.0, neg_frac=0.25) -> SegmentQuality:
    """Decide whether `seg` actually contains a detected source.

    Two independent questions are asked, because either one alone has a
    false-positive mode:

    *Is it flat?* The segment is split into `n_bins` equal-count bins and
    a constant is fitted to the bin means. A real astrophysical spectrum
    -- even a featureless one -- has a continuum slope and lines, so it
    rejects a constant with a large chi2. Pure background noise does
    not. Binning first is what makes this work: it averages down the
    per-pixel noise so that *structure* is being tested, not scatter.
    Flatness alone is not damning, because a genuinely well-detected
    source observed over a short, featureless stretch can look flat.

    *Is it blank?* The median per-pixel S/N, the significance of the
    weighted mean flux, and the fraction of positive points. An
    off-target spectrum sits at zero (often slightly negative, from
    background over-subtraction) with half its points below zero.
    Brightness alone is not damning either, because a segment can be
    bright and still be the wrong thing.

    A segment is called ``blank`` only when it is *both* flat and
    consistent with no source. That conjunction is what makes the flag
    worth acting on rather than worth muting.

    *Is it negative?* Separately, and regardless of flatness, a segment
    whose flux sits significantly below zero over most of its span is
    reported. A source cannot have a negative continuum, so this needs no
    corroboration: it means the background or nod subtraction removed
    more than was there. It is a distinct failure from a blank segment
    and the flatness test cannot see it, because an over-subtracted
    spectrum keeps the structure of whatever was subtracted and so
    rejects a constant easily.

    The thresholds are deliberately permissive. A missed marginal case
    costs nothing extra -- the residual-step warning still fires -- while
    a false alarm on real data trains you to ignore the output.
    """
    sp = seg.spectrum.compressed()
    w = np.asarray(sp.wavelength, float)
    f = np.asarray(sp.flux, float)
    e = np.asarray(sp.flux_err, float)
    ok = np.isfinite(w) & np.isfinite(f)
    q = SegmentQuality(name=seg.name, n_points=int(ok.sum()))
    if q.n_points < min_points:
        q.reason = f"only {q.n_points} usable points; not assessed"
        return q

    w, f, e = w[ok], f[ok], e[ok]
    have_err = np.isfinite(e) & (e > 0)
    # Fall back to the point-to-point scatter when the file carries no
    # usable errors. Without this the whole check silently no-ops on the
    # products that most need it.
    if have_err.sum() < 0.5 * f.size:
        e = np.full_like(f, _mad_sigma(np.diff(f)) / np.sqrt(2.0))
        have_err = np.isfinite(e) & (e > 0)
        q.reason = "no usable errors; scatter used instead"
    f, e, w = f[have_err], e[have_err], w[have_err]
    if f.size < min_points:
        q.reason = f"only {f.size} points with usable errors; not assessed"
        return q

    q.median_flux = float(np.median(f))
    q.median_snr = float(np.median(f / e))
    q.frac_positive = float(np.mean(f > 0))

    wgt = 1.0 / e ** 2
    mean = float(np.sum(wgt * f) / np.sum(wgt))
    mean_err = float(1.0 / np.sqrt(np.sum(wgt)))
    q.mean_snr = mean / mean_err if mean_err > 0 else np.nan

    # Binned constant-model chi2. Equal-count bins keep every bin's error
    # comparable, so one noisy stretch cannot dominate the statistic.
    nb = int(max(4, min(n_bins, f.size // 10)))
    edges = np.linspace(0, f.size, nb + 1).astype(int)
    bm, be = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 3:
            continue
        ww = wgt[a:b]
        bm.append(np.sum(ww * f[a:b]) / np.sum(ww))
        be.append(1.0 / np.sqrt(np.sum(ww)))
    bm, be = np.asarray(bm), np.asarray(be)
    if bm.size >= 4:
        bw = 1.0 / be ** 2
        const = np.sum(bw * bm) / np.sum(bw)
        q.chi2_const = float(np.sum(((bm - const) / be) ** 2) / (bm.size - 1))

    q.flat = bool(np.isfinite(q.chi2_const) and q.chi2_const < flat_chi2)
    no_source = (
        (not np.isfinite(q.median_snr) or q.median_snr < snr_floor)
        # One-sided on purpose. A mean that is significantly *negative*
        # is not evidence of a source; it is the signature of background
        # over-subtraction on an empty aperture, which is exactly the
        # case being caught here.
        and (not np.isfinite(q.mean_snr) or q.mean_snr < mean_sigma_floor)
        and q.frac_positive < 0.75
    )
    q.blank = bool(q.flat and no_source)
    q.negative = bool(np.isfinite(q.median_snr) and q.median_snr <= neg_snr
                      and q.frac_positive <= neg_frac)
    if q.negative and not q.blank:
        q.reason = (
            f"median S/N {q.median_snr:+.2f} with only "
            f"{100 * q.frac_positive:.0f}% of points above zero")
    if q.blank and not q.reason:
        q.reason = (
            f"median S/N {q.median_snr:+.2f}, mean flux "
            f"{q.mean_snr:+.1f} sigma, {100 * q.frac_positive:.0f}% of points "
            f"positive, and consistent with a constant "
            f"(chi2/dof={q.chi2_const:.2f})")
    return q


# ---------------------------------------------------------------------------
# overlap measurement
# ---------------------------------------------------------------------------
@dataclass
class OverlapMeasurement:
    """Result of comparing two segments where they meet.

    ``ratio`` is defined as *blue over red*: multiply the red segment by
    ``ratio`` to bring it onto the blue segment's flux level. ``ln_ratio`` and
    ``ln_sigma`` are what the global solve actually consumes -- working in the
    log makes the problem linear and makes a factor of 1.05 and a factor of
    1/1.05 symmetric, which they are not in linear ratio space.
    """
    blue: str
    red: str
    kind: str                  # 'overlap' or 'gap'
    wlo: float = np.nan
    whi: float = np.nan
    ratio: float = np.nan
    ratio_sigma: float = np.nan
    ln_ratio: float = np.nan
    ln_sigma: float = np.nan
    n_points: int = 0
    snr_blue: float = np.nan
    snr_red: float = np.nan
    tilt_per_um: float = np.nan      # d ln(ratio) / d lambda across the overlap
    tilt_sigma: float = np.nan
    curvature_sig: float = np.nan
    usable: bool = False
    reason: str | None = None

    @property
    def step_pct(self) -> float:
        """Percentage by which the red segment sits above the blue one."""
        return 100.0 * (1.0 / self.ratio - 1.0) if np.isfinite(self.ratio) and self.ratio else np.nan

    def __repr__(self):
        tag = "ok " if self.usable else "SKIP"
        r = f"{self.ratio:.4f}+/-{self.ratio_sigma:.4f}" if np.isfinite(self.ratio) else "n/a"
        return (f"<{tag} {self.blue}|{self.red} {self.kind} "
                f"{self.wlo:.3f}-{self.whi:.3f}um ratio={r}"
                + (f" ({self.reason})" if self.reason else "") + ">")


def measure_overlap(blue: Segment, red: Segment, *, edge_trim=0.02,
                    edge_trim_min_um=0.0, min_points=10, min_snr=3.0,
                    max_scale_factor=2.0, max_rel_uncertainty=0.30,
                    clip_sigma=3.0, max_gap_um=0.15,
                    curvature_reject=None) -> OverlapMeasurement:
    """Measure the flux ratio between two segments where they meet.

    Two regimes, chosen automatically:

    *Overlap* -- the segments share wavelength coverage. Both are trimmed by
    ``edge_trim`` (a fraction of each segment's own span) so the roll-off
    pixels at band edges do not drive the answer, interpolated onto the union
    of their grids inside what remains of the overlap, and compared pointwise.
    The ratio is the sigma-clipped median of ``f_blue / f_red``, and its
    uncertainty is the clipped scatter divided by sqrt(N_eff) added in
    quadrature to the propagated photometric error -- so a wide, clean overlap
    is correctly treated as far more informative than a two-pixel one.

    *Gap* -- the segments do not overlap but are separated by less than
    ``max_gap_um``. Each side is extrapolated linearly to the gap midpoint and
    the two extrapolations are compared. The uncertainty is inflated by the
    extrapolation distance in units of the fitting baseline, because
    extrapolating a straight line across a gap as wide as the data you fit it
    to is a guess, and should be weighted like one. This is the regime that
    matters for the NIRSpec-to-MIRI boundary when the two instruments'
    coverage does not actually meet.

    ``curvature_reject``, if set, is a significance threshold above which a
    locally curved overlap is refused: genuine curvature (a line wing, the
    silicate shoulder) means part of the apparent step is real spectral shape,
    and absorbing it into a scalar calibration factor would flatten a real
    feature. Left as None by default because in an *overlap* both segments see
    the same curvature and it cancels in the ratio; it is applied
    automatically in the gap regime, where it does not cancel.
    """
    sb, sr = blue.spectrum, red.spectrum
    gb, gr = sb.good, sr.good
    out = OverlapMeasurement(blue=blue.name, red=red.name, kind="overlap")

    if gb.sum() < min_points or gr.sum() < min_points:
        out.reason = "too few good points in one segment"
        return out

    wb, fb, eb = sb.wavelength[gb], sb.flux[gb], sb.flux_err[gb]
    wr, fr, er = sr.wavelength[gr], sr.flux[gr], sr.flux_err[gr]

    b_lo, b_hi = _trimmed_range(wb, edge_trim, edge_trim_min_um)
    r_lo, r_hi = _trimmed_range(wr, edge_trim, edge_trim_min_um)

    # The *raw* overlap decides which regime we are in. Deciding on the
    # trimmed overlap instead would push a genuinely overlapping pair into the
    # gap branch whenever the overlap happens to be narrower than the trim --
    # which is exactly the case for adjacent MRS sub-bands, whose overlaps are
    # only a few percent of a band wide. The trim then only controls *where
    # inside* a real overlap we measure.
    raw_lo, raw_hi = max(wb.min(), wr.min()), min(wb.max(), wr.max())
    ov_lo, ov_hi = max(b_lo, r_lo), min(b_hi, r_hi)

    # ---------------- gap regime ----------------
    if raw_hi <= raw_lo:
        raw_gap = float(wr.min() - wb.max())
        out.kind = "gap"
        out.wlo, out.whi = float(wb.max()), float(wr.min())
        if raw_gap > max_gap_um:
            out.reason = f"no overlap and gap {raw_gap:.3f} um > max_gap_um={max_gap_um:g}"
            return out
        centre = 0.5 * (wb.max() + wr.min())
        # Fit over a baseline comparable to the gap, but widened until each
        # side actually contains enough pixels to constrain a slope -- a fixed
        # baseline in microns means something completely different for an MRS
        # ch1 band and a NIRSpec prism spectrum.
        base = max(3.0 * max(raw_gap, 1e-3), 0.05)
        for _ in range(8):
            nb = int((wb >= wb.max() - base).sum())
            nr = int((wr <= wr.min() + base).sum())
            if nb >= min_points and nr >= min_points:
                break
            base *= 1.8
        lb = _local_line(wb, fb, eb, wb.max() - base, wb.max(), centre, min_points)
        lr = _local_line(wr, fr, er, wr.min(), wr.min() + base, centre, min_points)
        if lb is None or lr is None:
            out.reason = "too few points to extrapolate across the gap"
            return out
        vb, svb, _, cb = lb
        vr, svr, _, cr = lr
        out.curvature_sig = float(np.nanmax([cb, cr]))
        out.snr_blue = abs(vb) / svb if svb > 0 else np.inf
        out.snr_red = abs(vr) / svr if svr > 0 else np.inf
        if min(out.snr_blue, out.snr_red) < min_snr:
            out.reason = (f"extrapolated level S/N too low "
                          f"(blue={out.snr_blue:.1f}, red={out.snr_red:.1f}, need >={min_snr:g})")
            return out
        if vb <= 0 or vr <= 0:
            out.reason = "extrapolated flux is non-positive on one side; a ratio is meaningless"
            return out
        # Inflate by how far we extrapolated relative to the fitting baseline.
        inflate = 1.0 + (raw_gap / base)
        rel_b, rel_r = svb / vb * inflate, svr / vr * inflate
        out.ratio = float(vb / vr)
        out.ln_ratio = float(np.log(out.ratio))
        out.ln_sigma = float(np.hypot(rel_b, rel_r))
        out.ratio_sigma = float(out.ratio * out.ln_sigma)
        out.n_points = int(((wb >= wb.max() - base).sum() + (wr <= wr.min() + base).sum()))
        thr = 2.5 if curvature_reject is None else curvature_reject
        if np.isfinite(out.curvature_sig) and out.curvature_sig > thr:
            out.reason = (f"gap flanks are curved ({out.curvature_sig:.1f} sigma); "
                          f"a linear extrapolation across the gap is not trustworthy")
            return out
    # ---------------- overlap regime ----------------
    else:
        if ov_hi <= ov_lo:
            # Edge trimming consumed the whole overlap. Fall back to the raw
            # overlap rather than giving up: a narrow overlap measured from
            # its edge pixels is still far better than no constraint, which
            # would leave these two segments uncalibrated against each other.
            ov_lo, ov_hi = raw_lo, raw_hi
            out.reason = "overlap narrower than the edge trim: measured untrimmed"
        out.wlo, out.whi = ov_lo, ov_hi
        mb = (wb >= ov_lo) & (wb <= ov_hi)
        mr = (wr >= ov_lo) & (wr <= ov_hi)
        if mb.sum() < min_points or mr.sum() < min_points:
            # Retry on the untrimmed overlap for the same reason as above.
            mb = (wb >= raw_lo) & (wb <= raw_hi)
            mr = (wr >= raw_lo) & (wr <= raw_hi)
            if mb.sum() < min_points or mr.sum() < min_points:
                out.reason = (f"overlap has too few points "
                              f"(blue={int(mb.sum())}, red={int(mr.sum())}, need >={min_points})")
                return out
            out.wlo, out.whi = raw_lo, raw_hi
            out.reason = "narrow overlap: edge trimming skipped"

        grid = np.unique(np.concatenate([wb[mb], wr[mr]]))
        fbi = np.interp(grid, wb, fb)
        fri = np.interp(grid, wr, fr)
        ebi = np.interp(grid, wb, eb)
        eri = np.interp(grid, wr, er)

        out.snr_blue = float(np.nanmedian(np.abs(fbi) / ebi))
        out.snr_red = float(np.nanmedian(np.abs(fri) / eri))
        if not np.isfinite(out.snr_blue) or not np.isfinite(out.snr_red) \
                or min(out.snr_blue, out.snr_red) < min_snr:
            out.reason = (f"overlap S/N too low (blue={out.snr_blue:.2f}, "
                          f"red={out.snr_red:.2f}, need >={min_snr:g})")
            return out

        # Work in the log so the estimator is symmetric under swapping the
        # two segments, and so a sign flip in a noise-dominated pixel is
        # excluded rather than producing a wild negative ratio.
        pos = (fbi > 0) & (fri > 0) & np.isfinite(fbi) & np.isfinite(fri)
        if pos.sum() < min_points:
            out.reason = ("fewer than min_points pixels are positive in both "
                          "segments; the overlap is noise-dominated and a flux "
                          "ratio is not defined there")
            return out
        lr_pt = np.log(fbi[pos] / fri[pos])
        # Per-point uncertainty on the log-ratio, propagated from both errors.
        lr_err = np.hypot(ebi[pos] / fbi[pos], eri[pos] / fri[pos])

        keep = np.ones(lr_pt.size, dtype=bool)
        for _ in range(5):
            med = np.median(lr_pt[keep])
            scat = _mad_sigma(lr_pt[keep])
            if not np.isfinite(scat) or scat == 0:
                break
            new = np.abs(lr_pt - med) <= clip_sigma * scat
            if new.sum() < min_points or (new == keep).all():
                break
            keep = new

        n = int(keep.sum())
        out.n_points = n
        med = float(np.median(lr_pt[keep]))
        scat = _mad_sigma(lr_pt[keep])

        # Neighbouring pixels in a JWST x1d are correlated (the extraction
        # and the drizzle both smear flux between them), so the number of
        # *independent* samples in an overlap is smaller than the number of
        # pixels. Treating N pixels as N independent measurements would make
        # a wide overlap look ~sqrt(3) more certain than it is, which matters
        # because that certainty decides whether a correction is applied.
        n_eff = max(n / 3.0, 1.0)
        sem = (scat / np.sqrt(n_eff)) if np.isfinite(scat) else np.inf
        phot = float(np.median(lr_err[keep])) / np.sqrt(n_eff)
        out.ln_ratio = med
        out.ln_sigma = float(np.hypot(sem, phot))
        out.ratio = float(np.exp(med))
        out.ratio_sigma = float(out.ratio * out.ln_sigma)

        # Is a single scalar enough, or is the mismatch wavelength dependent?
        # Reported always, acted on only by the caller -- a significant tilt
        # means a scalar will leave a residual slope across this join.
        if n >= 20:
            xg = grid[pos][keep]
            xc = 0.5 * (out.wlo + out.whi)
            try:
                wgt = 1.0 / np.clip(lr_err[keep], 1e-12, None) ** 2
                A = np.vander(xg - xc, 2)
                W = np.diag(wgt)
                cov = np.linalg.inv(A.T @ W @ A)
                p = cov @ (A.T @ W @ lr_pt[keep])
                res = lr_pt[keep] - A @ p
                cov = cov * max(float(np.sum(wgt * res ** 2) / max(n - 2, 1)), 1.0)
                out.tilt_per_um = float(p[0])
                out.tilt_sigma = float(np.sqrt(cov[0, 0]))
            except np.linalg.LinAlgError:
                pass

    # ---------------- shared guard rails ----------------
    rel_unc = out.ln_sigma if np.isfinite(out.ln_sigma) else np.inf
    reasons = []
    if not np.isfinite(out.ratio) or out.ratio <= 0:
        reasons.append("ratio is not a positive finite number")
    elif not (1.0 / max_scale_factor <= out.ratio <= max_scale_factor):
        reasons.append(f"ratio {out.ratio:.3f} outside "
                       f"[{1/max_scale_factor:.3f}, {max_scale_factor:.3f}]")
    if rel_unc > max_rel_uncertainty:
        reasons.append(f"relative uncertainty {rel_unc:.2f} > {max_rel_uncertainty:g}")

    if reasons:
        out.usable = False
        out.reason = "; ".join(reasons) + (f" [{out.reason}]" if out.reason else "")
    else:
        out.usable = True
    return out


# ---------------------------------------------------------------------------
# global scale solve
# ---------------------------------------------------------------------------
def solve_scales(names, measurements, anchor=None, anchor_weight=1e6,
                 ridge=1e-6, robust_iters=2, robust_clip=3.0):
    """Solve for one multiplicative scale per segment from all pairwise
    constraints simultaneously.

    Each usable measurement contributes ``s_blue - s_red = -ln R`` with weight
    ``1 / sigma_ln^2``, where ``s = ln(scale)``. This is a weighted
    least-squares problem on a graph: segments are nodes, overlaps are edges.
    Solving it in one shot rather than sequentially is the whole point --
    every constraint sees every other one, so no single marginal overlap can
    dictate the relative calibration of two halves of the spectrum, and the
    solution does not depend on the order segments happen to be listed in.

    Parameters
    ----------
    names : sequence of str
        Segment names, defining the solution order.
    measurements : sequence of OverlapMeasurement
        Only those with ``usable=True`` are used.
    anchor : str, optional
        Segment held at scale exactly 1. Everything else moves onto it, so
        this should be the segment whose absolute flux calibration you trust
        most -- by default, the one with the most/strongest constraints
        attached, which is the one the solution is most tightly tied to
        anyway. **The anchor choice sets the absolute flux scale of the whole
        result**; it does not change the *shape*.
    ridge : float
        Tiny Tikhonov term pulling all scales toward 1. Present only to keep
        the normal equations invertible for degenerate inputs; at 1e-6 it is
        numerically irrelevant to a well-posed solve.
    robust_iters : int
        Rounds of residual-based down-weighting. After each solve, any
        constraint whose residual exceeds ``robust_clip`` times its own sigma
        is down-weighted, so one discrepant overlap is outvoted by the rest
        instead of dragging the solution. With a sequential merger this is
        structurally impossible.

    Returns
    -------
    dict
        ``{'scales': {name: factor}, 'log_scales': {...},
        'scale_sigma': {name: sigma_of_factor}, 'components': [[names], ...],
        'anchor': str, 'residuals': [(blue, red, resid_sigma), ...]}``

    Notes
    -----
    If the overlap graph is disconnected -- say NIRSpec and MIRI share no
    usable constraint -- each connected component is solved and anchored
    *separately*, and the components are reported. Components are NOT forced
    together by assumption: inventing a cross-component scale would be
    fabricating a calibration tie that the data does not support. Stitching
    continues, and the join between components is flagged in the report.
    """
    names = list(names)
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    used = [m for m in measurements if m.usable and np.isfinite(m.ln_ratio)
            and np.isfinite(m.ln_sigma) and m.ln_sigma > 0]

    # connected components over usable constraints
    adj = {i: set() for i in range(n)}
    for m in used:
        i, j = idx[m.blue], idx[m.red]
        adj[i].add(j)
        adj[j].add(i)
    seen, components = set(), []
    for i in range(n):
        if i in seen:
            continue
        stack, comp = [i], []
        seen.add(i)
        while stack:
            k = stack.pop()
            comp.append(k)
            for q in adj[k]:
                if q not in seen:
                    seen.add(q)
                    stack.append(q)
        components.append(sorted(comp))

    log_scales = np.zeros(n)
    sigmas = np.full(n, np.nan)

    # constraint strength per node, for the default anchor choice
    strength = np.zeros(n)
    for m in used:
        w = 1.0 / m.ln_sigma ** 2
        strength[idx[m.blue]] += w
        strength[idx[m.red]] += w

    anchors = []
    for comp in components:
        comp_meas = [m for m in used if idx[m.blue] in comp and idx[m.red] in comp]
        if not comp_meas:
            anchors.append(names[comp[0]])
            sigmas[comp[0]] = 0.0
            continue
        if anchor is not None and idx.get(anchor) in comp:
            a_local = idx[anchor]
        else:
            a_local = comp[int(np.argmax(strength[comp]))]
        anchors.append(names[a_local])

        local = {g: k for k, g in enumerate(comp)}
        nc = len(comp)
        weights = np.array([1.0 / m.ln_sigma ** 2 for m in comp_meas])

        for it in range(max(robust_iters, 0) + 1):
            A = np.zeros((len(comp_meas) + 1, nc))
            b = np.zeros(len(comp_meas) + 1)
            w = np.zeros(len(comp_meas) + 1)
            for k, m in enumerate(comp_meas):
                A[k, local[idx[m.blue]]] = 1.0
                A[k, local[idx[m.red]]] = -1.0
                b[k] = -m.ln_ratio
                w[k] = weights[k]
            A[-1, local[a_local]] = 1.0
            b[-1] = 0.0
            w[-1] = anchor_weight

            W = np.diag(w)
            N = A.T @ W @ A + ridge * np.eye(nc)
            try:
                cov = np.linalg.inv(N)
            except np.linalg.LinAlgError:
                cov = np.linalg.pinv(N)
            s = cov @ (A.T @ W @ b)

            if it == max(robust_iters, 0):
                break
            resid = (A @ s - b)[:-1]
            sig = np.array([m.ln_sigma for m in comp_meas])
            z = np.abs(resid) / np.where(sig > 0, sig, np.inf)
            # Soft down-weighting (Huber-like): a 5-sigma outlier keeps 60% of
            # its weight at clip=3 rather than being deleted, so a genuinely
            # large-but-real offset is still allowed to pull the solution.
            weights = (1.0 / sig ** 2) * np.where(
                z > robust_clip, robust_clip / np.maximum(z, 1e-9), 1.0)

        for g in comp:
            log_scales[g] = s[local[g]]
            sigmas[g] = float(np.sqrt(max(cov[local[g], local[g]], 0.0)))

    residuals = []
    for m in used:
        pred = log_scales[idx[m.blue]] - log_scales[idx[m.red]]
        residuals.append((m.blue, m.red,
                          float((pred + m.ln_ratio) / m.ln_sigma)))

    scales = {nm: float(np.exp(log_scales[i])) for i, nm in enumerate(names)}
    return {
        "scales": scales,
        "log_scales": {nm: float(log_scales[i]) for i, nm in enumerate(names)},
        "scale_sigma": {nm: float(scales[nm] * sigmas[i]) for i, nm in enumerate(names)},
        "components": [[names[i] for i in comp] for comp in components],
        "anchor": anchors[0] if len(anchors) == 1 else anchors,
        "residuals": residuals,
    }


# ---------------------------------------------------------------------------
# blending
# ---------------------------------------------------------------------------
def _taper(x, lo, hi):
    """Raised-cosine ramp from 0 at ``lo`` to 1 at ``hi``.

    Chosen over a linear ramp because its derivative vanishes at both ends,
    so the composite's *slope* is continuous at the overlap edges too, not
    just its value. A linear ramp removes the step but leaves a visible kink.
    """
    if hi <= lo:
        return np.where(x >= hi, 1.0, 0.0)
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * t))


def _blend_pair(w1, f1, e1, w2, f2, e2, taper=True, edge_trim_output=0.0):
    """Merge two already-scaled spectra, tapering through their overlap.

    Outside the overlap each segment is passed through untouched. Inside it,
    the weights are ``taper * inverse-variance``, so the result is continuous
    with segment 1 at the blue edge and segment 2 at the red edge while still
    favouring whichever segment is less noisy in between.
    """
    if w1.min() > w2.min():
        w1, f1, e1, w2, f2, e2 = w2, f2, e2, w1, f1, e1

    ov_lo, ov_hi = w2.min(), w1.max()
    if ov_hi <= ov_lo:
        w = np.concatenate([w1, w2])
        f = np.concatenate([f1, f2])
        e = np.concatenate([e1, e2])
        o = np.argsort(w)
        return w[o], f[o], e[o], (np.nan, np.nan)

    if edge_trim_output > 0:
        span = ov_hi - ov_lo
        pad = edge_trim_output * span
        ov_lo_b, ov_hi_b = ov_lo + pad, ov_hi - pad
    else:
        ov_lo_b, ov_hi_b = ov_lo, ov_hi

    m1_out = w1 < ov_lo
    m2_out = w2 > ov_hi
    grid = np.unique(np.concatenate([w1[(w1 >= ov_lo) & (w1 <= ov_hi)],
                                     w2[(w2 >= ov_lo) & (w2 <= ov_hi)]]))
    if grid.size == 0:
        w = np.concatenate([w1, w2])
        f = np.concatenate([f1, f2])
        e = np.concatenate([e1, e2])
        o = np.argsort(w)
        return w[o], f[o], e[o], (ov_lo, ov_hi)

    f1i, e1i = np.interp(grid, w1, f1), np.interp(grid, w1, e1)
    f2i, e2i = np.interp(grid, w2, f2), np.interp(grid, w2, e2)

    iv1 = 1.0 / np.clip(e1i, 1e-30, None) ** 2
    iv2 = 1.0 / np.clip(e2i, 1e-30, None) ** 2
    if taper:
        t = _taper(grid, ov_lo_b, ov_hi_b)
        a, b = (1.0 - t), t
    else:
        a = b = np.ones_like(grid)

    wa, wb = a * iv1, b * iv2
    tot = wa + wb
    bad = tot <= 0
    tot = np.where(bad, 1.0, tot)
    f_ov = (f1i * wa + f2i * wb) / tot
    # Variance of the weighted combination with *fixed* weights (the taper is
    # not a free parameter), not 1/sum(iv): using the latter would claim the
    # full inverse-variance improvement while actually having thrown part of
    # it away via the taper, i.e. it would understate the error.
    v_ov = (wa ** 2 / iv1 + wb ** 2 / iv2) / tot ** 2
    e_ov = np.sqrt(np.clip(v_ov, 0.0, None))
    f_ov = np.where(bad, np.nan, f_ov)
    e_ov = np.where(bad, np.nan, e_ov)

    w = np.concatenate([w1[m1_out], grid, w2[m2_out]])
    f = np.concatenate([f1[m1_out], f_ov, f2[m2_out]])
    e = np.concatenate([e1[m1_out], e_ov, e2[m2_out]])
    o = np.argsort(w)
    return w[o], f[o], e[o], (ov_lo, ov_hi)


# ---------------------------------------------------------------------------
# post-stitch verification
# ---------------------------------------------------------------------------
def residual_step(spectrum: Spectrum, centre: float, half_width=0.25, gap=0.01,
                  min_points=8, max_rise_um=0.06):
    """Measure the flux discontinuity at ``centre`` in an already-stitched
    spectrum.

    Fits a line to each side, excluding a small ``gap`` around the join itself
    (where the blend lives), extrapolates both to ``centre``, and reports the
    difference. This is the honest after-the-fact check: the stitcher can
    always *claim* it matched two segments, but a straight line from each side
    either meets in the middle or it does not.

    Returns a dict with ``step_pct``, ``sigma``, ``curvature_sig`` and a
    ``verdict`` string, or None if either side has too few points.

    A large ``curvature_sig`` means the spectrum is genuinely bending here, so
    a nominally significant step is expected and does not indicate a
    calibration failure -- that distinction is why the curvature term is
    computed at all.

    The curvature term alone is not sufficient, though. Fitting a straight
    line to each side and extrapolating both into the middle is very easily
    fooled: any steep or structured continuum makes the two extrapolations
    miss each other by a lot, and the result is a confident "residual
    calibration step" where the data is in fact perfectly continuous. On
    SN2024aecx_2025-12-28 this fired at five of eleven joins while an
    independent search for sharp discontinuities found none of them.

    So the measured difference is also required to be *unresolved*: a genuine
    calibration step is imposed at one wavelength and completes within a pixel
    or two, whereas real structure takes many (see
    :func:`jspec.discontinuity.transition_width_um`). ``max_rise_um=0``
    disables that check and restores the old linear-extrapolation-only
    behaviour.
    """
    g = spectrum.good
    w, f, e = spectrum.wavelength[g], spectrum.flux[g], spectrum.flux_err[g]
    blue = _local_line(w, f, e, centre - half_width, centre - gap, centre, min_points)
    red = _local_line(w, f, e, centre + gap, centre + half_width, centre, min_points)
    if blue is None or red is None:
        return None
    vb, sb, _, cb = blue
    vr, sr, _, cr = red
    diff = vr - vb
    sig_diff = float(np.hypot(sb, sr))
    curv = float(np.nanmax([cb, cr]))
    denom = 0.5 * (abs(vb) + abs(vr))
    step_pct = 100.0 * diff / denom if denom > 0 else np.nan
    nsig = abs(diff) / sig_diff if sig_diff > 0 else np.inf

    # Sharpness has to be measured against robust *medians* of the real data
    # on each side, not against the line extrapolations above: vb and vr are
    # values the spectrum may never actually take near the join, so the 10%
    # and 90% thresholds would both be crossed at the first sample and every
    # join would look infinitely sharp.
    rise = float("nan")
    if max_rise_um > 0:
        from .discontinuity import _levels, transition_width_um
        mb = (w >= centre - half_width) & (w <= centre - gap)
        mr = (w >= centre + gap) & (w <= centre + half_width)
        lo_med, _, nb = _levels(f[mb])
        hi_med, _, nr = _levels(f[mr])
        if nb >= 3 and nr >= 3 and np.isfinite(lo_med) and np.isfinite(hi_med) \
                and hi_med != lo_med:
            rise = transition_width_um(w, f, centre, lo_med, hi_med - lo_med,
                                       window_um=half_width)

    if nsig < 2:
        verdict = "continuous"
    elif np.isfinite(curv) and curv > 3:
        verdict = "step significant but spectrum is locally curved here (likely real structure)"
    elif np.isfinite(rise) and rise > max_rise_um:
        verdict = (f"level differs but the change is resolved "
                   f"({rise:.3f} um to complete, > {max_rise_um:g}); "
                   f"real structure, not a join artefact")
    else:
        verdict = "residual calibration step"
    return dict(centre=float(centre), blue_level=vb, red_level=vr,
                step_pct=float(step_pct), sigma=float(nsig),
                curvature_sig=curv, rise_um=rise, verdict=verdict)


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
@dataclass
class JoinReport:
    blue: str
    red: str
    measurement: OverlapMeasurement
    centre: float = np.nan
    residual: dict | None = None

    def __repr__(self):
        r = self.residual
        tail = (f" -> residual {r['step_pct']:+.1f}% ({r['sigma']:.1f} sigma, {r['verdict']})"
                if r else "")
        return f"<Join {self.blue}|{self.red}{tail}>"


@dataclass
class StitchReport:
    """Everything the stitcher decided and everything that is left over."""
    spectrum: Spectrum
    scales: dict = field(default_factory=dict)
    scale_sigma: dict = field(default_factory=dict)
    joins: list = field(default_factory=list)
    components: list = field(default_factory=list)
    anchor: object = None
    warnings: list = field(default_factory=list)
    quality: dict = field(default_factory=dict)   # name -> SegmentQuality
    dropped: list = field(default_factory=list)   # segments left out of the output

    def summary(self) -> str:
        """A human-readable block suitable for printing or pasting into a log.
        This is the thing to look at to answer 'did this epoch stitch
        cleanly'."""
        lines = ["Stitch report", "=" * 60,
                 f"anchor: {self.anchor}", ""]
        if len(self.components) > 1:
            lines.append(f"WARNING: {len(self.components)} disconnected calibration "
                         f"groups -- their relative flux scales are NOT tied by data:")
            for c in self.components:
                lines.append(f"    {c}")
            lines.append("")
        if self.dropped:
            lines.append("DROPPED from the output (no source detected): "
                         + ", ".join(self.dropped))
            lines.append("")
        lines.append(f"{'segment':<36}{'scale':>10}{'+/-':>10}")
        for k in self.scales:
            lines.append(f"{_short(k, 35):<36}{self.scales[k]:>10.4f}"
                         f"{self.scale_sigma.get(k, float('nan')):>10.4f}")
        lines.append("")
        if self.quality:
            lines.append(f"{'segment':<36}{'med S/N':>9}{'chi2/dof':>10}"
                         f"{'+ve':>7}   verdict")
            for k, q in self.quality.items():
                verdict = ("NO SOURCE DETECTED" if q.blank
                           else "NEGATIVE FLUX" if q.negative
                           else "flat" if q.flat else "ok")
                lines.append(f"{_short(k, 35):<36}{q.median_snr:>9.2f}"
                             f"{q.chi2_const:>10.1f}"
                             f"{100 * q.frac_positive:>6.0f}%   {verdict}")
            lines.append("")
        lines.append(f"{'join':<46}{'kind':<9}{'ratio':>9}{'used':>6}   residual")
        for j in self.joins:
            m = j.measurement
            ratio = f"{m.ratio:.4f}" if np.isfinite(m.ratio) else "   n/a"
            res = (f"{j.residual['step_pct']:+6.1f}% ({j.residual['sigma']:.1f}s) "
                   f"{j.residual['verdict']}" if j.residual else "n/a")
            pair = _short(j.blue, 21) + " | " + _short(j.red, 21)
            lines.append(f"{pair:<46}{m.kind:<9}{ratio:>9}"
                         f"{'yes' if m.usable else 'NO':>6}   {res}")
            if not m.usable and m.reason:
                lines.append(f"{'':<4}reason: {m.reason}")
            if np.isfinite(m.tilt_sigma) and m.tilt_sigma > 0 \
                    and abs(m.tilt_per_um) > 3 * m.tilt_sigma:
                lines.append(f"{'':<4}note: ratio is wavelength dependent "
                             f"({m.tilt_per_um:+.3f}+/-{m.tilt_sigma:.3f} dex/um); "
                             f"a single scale factor cannot fully match this join")
        for w in self.warnings:
            # These can be long explanations, so wrap rather than emit one
            # 500-character line that the terminal mangles.
            wrapped = textwrap.wrap(str(w), width=72) or [""]
            lines.append(f"! {wrapped[0]}")
            lines.extend(f"  {ln}" for ln in wrapped[1:])
        return "\n".join(lines)

    def __repr__(self):
        bad = sum(1 for j in self.joins
                  if j.residual and j.residual["verdict"] == "residual calibration step")
        return (f"<StitchReport {len(self.joins)} joins, "
                f"{len(self.scales)} segments, {bad} residual steps>")


def _single_segment_report(seg, *, label_segments=True, quality=None,
                           dropped=None, warns=None) -> "StitchReport":
    """Package one segment as a finished stitch.

    Factored out because there are two ways to arrive here -- one input,
    or several of which only one survived the quality check -- and the
    output metadata has to look the same either way. It previously did
    not, so a spectrum that lost a segment carried no record of the
    segments it was built from.
    """
    sp = seg.spectrum.copy()
    if label_segments:
        sp.segment = np.array([seg.name] * len(sp), dtype=object)
    sp.meta["SEGMENTS"] = seg.name
    sp.meta["SCALES"] = f"{seg.name}=1.00000"
    if dropped:
        sp.meta["DROPPED_SEGMENTS"] = ",".join(dropped)
    quality = quality or {}
    neg = [n for n, q in quality.items() if q.negative and not q.blank]
    if neg:
        sp.meta["NEGATIVE_SEGMENTS"] = ",".join(neg)
    blank = [n for n, q in quality.items() if q.blank]
    if blank:
        sp.meta["BLANK_SEGMENTS"] = ",".join(blank)
    return StitchReport(spectrum=sp, scales={seg.name: 1.0},
                        scale_sigma={seg.name: 0.0}, anchor=seg.name,
                        components=[[seg.name]], quality=quality,
                        dropped=list(dropped or []), warnings=list(warns or []))


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def stitch(segments, *, anchor=None, edge_trim=0.02, edge_trim_min_um=0.0,
           min_points=10, min_snr=3.0, max_scale_factor=2.0,
           max_rel_uncertainty=0.30, clip_sigma=3.0, max_gap_um=0.15,
           apply_scales=True, taper=True, robust_iters=2,
           verify=True, verify_half_width=0.25, redshift=None,
           label_segments=True, check_epoch=True,
           max_gap_days=1.0, assess_quality=True,
           drop_blank=True) -> StitchReport:
    """Stitch instrument segments into one continuous spectrum.

    Parameters
    ----------
    segments : sequence of Segment
        From :func:`jspec.spectrum.read_segments`. Order does not matter;
        they are sorted blue to red internally, and the scale solve is
        order-independent by construction.
    anchor : str, optional
        Segment whose flux calibration everything else is brought onto. The
        default picks the best-constrained segment. Set this explicitly when
        one segment has an absolute calibration you trust (e.g. the one with a
        contemporaneous photometric tie).
    edge_trim : float
        Fraction of each segment's own wavelength span excluded from *ratio
        measurement* at each end. The default 2% removes the band-edge
        roll-off that otherwise dominates narrow overlaps. It does not remove
        data from the output.
    apply_scales : bool
        If False, measure and report everything but stitch at native flux
        levels. Useful for diagnosing before committing.
    taper : bool
        Raised-cosine blending through overlaps (see module docstring). Turn
        off only to reproduce the old hard inverse-variance behaviour.
    verify : bool
        Re-measure the residual step at every join after stitching.
    redshift : float, optional
        Stored on the output spectrum; defaults to the first segment's.
    check_epoch : bool
        Warn if the segments carry MJD metadata showing they span more than
        ``max_gap_days``. Stitching across epochs does not fail loudly: it
        produces a perfectly plausible spectrum in which a scale factor has
        quietly absorbed the source's real variability. The warning is the
        only thing standing between you and that, so it is on by default.
        Use :func:`jspec.epochs.group_segments_by_epoch` to split first.
    max_gap_days : float
        Epoch tolerance for ``check_epoch``. Matches the project-wide default.
    assess_quality : bool
        Check each segment on its own for a detected source before
        stitching (see :func:`assess_segment`) and warn about any that
        contain none. A blank segment is the one failure the matching
        machinery cannot catch, because it matches a flat noise segment
        to a real continuum perfectly well. Results land in
        ``StitchReport.quality``.
    drop_blank : bool
        Leave segments that contain no detected source out of the output
        entirely, rather than only warning about them. On by default: a
        warning does not stop the file being written, and a stitched
        spectrum whose blue half is scaled background noise looks
        entirely convincing. Never drops the last detected segment.

    Returns
    -------
    StitchReport
    """
    segs = sorted([s for s in segments if len(s.spectrum)], key=lambda s: s.wmin)
    if not segs:
        raise ValueError("no segments to stitch")
    # The same file supplied twice -- easy to do with a glob over a tree
    # that keeps a copy of each product. Matching a spectrum to itself
    # yields a scale of exactly 1 and a meaningless join, so drop the
    # repeat rather than renaming it.
    if len({s.name for s in segs}) != len(segs):
        uniq, seen_sig, repeats = [], set(), []
        for sg in segs:
            sig = (sg.name, len(sg.spectrum), round(float(sg.wmin), 6),
                   round(float(sg.wmax), 6))
            if sig in seen_sig:
                repeats.append(sg.name)
                continue
            seen_sig.add(sig)
            uniq.append(sg)
        if repeats:
            segs = uniq
            warnings.warn(
                f"ignored {len(repeats)} duplicate segment(s) "
                f"({', '.join(sorted(set(repeats)))}): the same spectrum was "
                "supplied more than once", stacklevel=2)

    if len(segs) == 1:
        q = {segs[0].name: assess_segment(segs[0])} if assess_quality else {}
        return _single_segment_report(segs[0], label_segments=label_segments,
                                      quality=q)

    names = [s.name for s in segs]
    if check_epoch and len(segs) > 1:
        from .epochs import check_single_epoch
        check_single_epoch(segs, max_gap_days=max_gap_days)
    if len(set(names)) != len(names):
        # Distinct segments that merely share a name: rename, because
        # collapsing them in the scale dictionary would silently apply
        # one segment's correction to another.
        seen = {}
        for i, nm in enumerate(names):
            seen[nm] = seen.get(nm, 0) + 1
            if seen[nm] > 1:
                names[i] = f"{nm}#{seen[nm]}"
                segs[i] = Segment(names[i], segs[i].spectrum, segs[i].path)

    warns = []

    # --- is every segment actually a detection? --------------------------
    # Done before anything is measured, because a blank segment poisons
    # everything downstream: the scale solve will try to match it, the
    # blend will mix it in, and the result is a plausible-looking
    # spectrum whose blue half is background noise multiplied up to the
    # level of the half that did contain the source.
    quality = {}
    dropped = []
    if assess_quality:
        for s in segs:
            try:
                quality[s.name] = assess_segment(s)
            except Exception as exc:                     # noqa: BLE001
                warns.append(f"could not assess segment {s.name}: {exc}")
        # Scale a negative pedestal against the brightest neighbour, so
        # the warning can say whether it is a rounding error on a faint
        # segment or the whole signal.
        levels = {}
        for sgi in segs:
            spc = sgi.spectrum.compressed()
            fl = np.asarray(spc.flux, float)
            fl = fl[np.isfinite(fl)]
            if fl.size:
                levels[sgi.name] = float(np.median(fl))
        for nm, qq in quality.items():
            if not qq.negative:
                continue
            others = [abs(v) for k, v in levels.items()
                      if k != nm and np.isfinite(v) and v > 0]
            if others and np.isfinite(qq.median_flux):
                qq.pedestal_frac = qq.median_flux / max(others)

        blank = [q for q in quality.values() if q.blank]
        detected = [q for q in quality.values()
                    if not q.suspect and not q.flat]
        for q in blank:
            msg = (f"{q.name} contains no detected source: {q.reason}. ")
            if detected:
                msg += (
                    "Other segments in this stitch are clearly detected, so "
                    "the most likely explanation is that this exposure missed "
                    "the target (a mis-centred fixed slit, or a re-extraction "
                    "at the wrong position) rather than that the source is "
                    "faint here. ")
                if drop_blank:
                    msg += ("It has been DROPPED and does not appear in the "
                            "output; pass drop_blank=False (--keep-flagged) to "
                            "stitch it anyway. Re-extract it at the correct "
                            "position if you want this wavelength range.")
                else:
                    msg += ("It is being stitched anyway because "
                            "drop_blank=False, so the output blueward of the "
                            "join is background noise scaled onto a real "
                            "continuum. Do not use it.")
            else:
                msg += ("No segment in this stitch is clearly detected; check "
                        "that the extraction position is right at all. It has "
                        "been kept, since dropping it would leave nothing.")
            warns.append(msg)

        # Actually remove them. A warning alone is not enough: the output
        # file still gets written, still looks like a spectrum, and is
        # the thing that ends up in a plot three weeks later.
        if drop_blank and detected:
            drop = {q.name for q in blank}
            keep = [s for s in segs if s.name not in drop]
            if keep:
                dropped[:] = sorted(drop)
                segs = keep
                names = [s.name for s in segs]
                if len(segs) == 1:
                    return _single_segment_report(
                        segs[0], label_segments=label_segments,
                        quality=quality, dropped=dropped, warns=warns)
        for q in quality.values():
            if q.negative and not q.blank:
                msg = (f"{q.name} sits below zero over most of its span: "
                       f"{q.reason}. ")
                if np.isfinite(q.pedestal_frac):
                    msg += (f"The offset is close to constant and amounts to "
                            f"{100 * abs(q.pedestal_frac):.0f}% of the "
                            f"neighbouring segment's flux, so this is a small "
                            f"over-subtraction pedestal on a genuinely faint "
                            f"segment rather than a failed extraction. ")
                else:
                    msg += ("A source cannot have a negative continuum, so "
                            "some flux has been over-subtracted. ")
                msg += ("It is kept and stitched, since the shape may still be "
                        "usable, but the scale factor is fitted in the overlap "
                        "where the flux is positive -- so the join can look "
                        "clean while the rest of the segment stays below zero. "
                        "Treat absolute fluxes and equivalent widths here as "
                        "unreliable, and prefer a background region that "
                        "excludes the source if you re-reduce.")
                warns.append(msg)

    # --- measure every adjacent pair, plus any non-adjacent true overlaps ---
    measurements = []
    pairs = [(i, i + 1) for i in range(len(segs) - 1)]
    for i in range(len(segs)):
        for j in range(i + 2, len(segs)):
            # A non-adjacent pair that genuinely overlaps carries real
            # information (it ties two segments together without going
            # through the one between them) and is free to include.
            if segs[j].wmin < segs[i].wmax:
                pairs.append((i, j))

    for i, j in pairs:
        m = measure_overlap(
            segs[i], segs[j], edge_trim=edge_trim,
            edge_trim_min_um=edge_trim_min_um, min_points=min_points,
            min_snr=min_snr, max_scale_factor=max_scale_factor,
            max_rel_uncertainty=max_rel_uncertainty, clip_sigma=clip_sigma,
            max_gap_um=max_gap_um)
        measurements.append(m)

    sol = solve_scales(names, measurements, anchor=anchor,
                       robust_iters=robust_iters)
    scales = sol["scales"] if apply_scales else {n: 1.0 for n in names}
    if len(sol["components"]) > 1:
        warns.append(
            "the usable-overlap graph is disconnected: "
            + " | ".join(",".join(c) for c in sol["components"])
            + ". Segments in different groups are NOT cross-calibrated against "
              "each other; the flux step between groups is whatever the data "
              "happened to have. Fix this upstream (recover the overlap, or "
              "supply an external photometric tie) rather than by forcing a "
              "scale here.")

    # --- apply and blend ---
    scaled = []
    for s in segs:
        sp = s.spectrum.compressed()
        k = scales.get(s.name, 1.0)
        scaled.append((s.name, sp.wavelength, sp.flux * k, sp.flux_err * abs(k)))

    name0, w_acc, f_acc, e_acc = scaled[0]
    seg_acc = np.array([name0] * w_acc.size, dtype=object)
    join_centres = []

    for name, w2, f2, e2 in scaled[1:]:
        w_new, f_new, e_new, (olo, ohi) = _blend_pair(
            w_acc, f_acc, e_acc, w2, f2, e2, taper=taper)
        # Re-derive the per-point segment label on the new grid. Points inside
        # the overlap get a combined label so a later analysis can tell that a
        # feature sits on a join.
        prev_lab = np.full(w_new.size, None, dtype=object)
        in_old = w_new <= w_acc.max()
        in_new = w_new >= w2.min()
        both = in_old & in_new
        idx_old = np.searchsorted(w_acc, w_new)
        idx_old = np.clip(idx_old, 0, seg_acc.size - 1)
        prev_lab[in_old & ~both] = seg_acc[idx_old[in_old & ~both]]
        prev_lab[in_new & ~both] = name
        if both.any():
            olab = seg_acc[idx_old[both]]
            prev_lab[both] = np.array([f"{o}+{name}" for o in olab], dtype=object)
        prev_lab[prev_lab == None] = name  # noqa: E711
        seg_acc = prev_lab
        w_acc, f_acc, e_acc = w_new, f_new, e_new
        if np.isfinite(olo) and np.isfinite(ohi):
            join_centres.append(0.5 * (olo + ohi))
        else:
            join_centres.append(np.nan)

    z = redshift if redshift is not None else segs[0].spectrum.redshift
    meta = dict(segs[0].spectrum.meta)
    meta["STITCH"] = ("jspec.stitch.stitch: global weighted least-squares "
                      f"scales, anchor={sol['anchor']}, "
                      f"taper={'on' if taper else 'off'}, edge_trim={edge_trim}")
    meta["SEGMENTS"] = ",".join(names)
    if dropped:
        meta["DROPPED_SEGMENTS"] = ",".join(dropped)
    # Record what was actually applied. Without this the output file is a
    # dead end: you can see that it was stitched but not by how much, so a
    # 3% cross-instrument offset cannot be told from a 30% one after the
    # fact.
    meta["SCALES"] = ",".join(f"{n}={scales.get(n, 1.0):.5f}" for n in names)
    if quality:
        blank_names = [n for n, q in quality.items() if q.blank]
        if blank_names:
            meta["BLANK_SEGMENTS"] = ",".join(blank_names)
        neg_names = [n for n, q in quality.items() if q.negative and not q.blank]
        if neg_names:
            meta["NEGATIVE_SEGMENTS"] = ",".join(neg_names)
    out = Spectrum(w_acc, f_acc, e_acc, redshift=z,
                   flux_unit=segs[0].spectrum.flux_unit,
                   segment=seg_acc if label_segments else None, meta=meta)

    # --- verify ---
    adj = {(m.blue, m.red): m for m in measurements}
    joins = []
    for k, (a, b) in enumerate(zip(names[:-1], names[1:])):
        m = adj.get((a, b))
        if m is None:
            continue
        centre = join_centres[k] if k < len(join_centres) else np.nan
        if not np.isfinite(centre):
            centre = 0.5 * (segs[k].wmax + segs[k + 1].wmin)
        res = None
        if verify:
            try:
                res = residual_step(out, centre, half_width=verify_half_width)
            except Exception as exc:                      # noqa: BLE001
                warns.append(f"residual check failed at {centre:.3f} um: {exc}")
        joins.append(JoinReport(blue=a, red=b, measurement=m,
                                centre=float(centre), residual=res))

    for j in joins:
        if j.residual and j.residual["verdict"] == "residual calibration step":
            warns.append(
                f"{j.blue}|{j.red} still steps by {j.residual['step_pct']:+.1f}% "
                f"({j.residual['sigma']:.1f} sigma) after stitching"
                + ("; the measured ratio was rejected, so no correction was applied "
                   f"({j.measurement.reason})" if not j.measurement.usable else
                   "; the scale was applied but a single factor did not fully match "
                   "the join -- check the tilt note above, this usually means the "
                   "mismatch is wavelength dependent (aperture-correction or "
                   "residual-background shape) rather than a flat offset"))

    return StitchReport(spectrum=out, scales=scales,
                        scale_sigma=sol["scale_sigma"], joins=joins,
                        components=sol["components"], anchor=sol["anchor"],
                        warnings=warns, quality=quality, dropped=dropped)
