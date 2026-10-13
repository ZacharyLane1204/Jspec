#!/usr/bin/env python3
"""
combine_epoch_spectra.py
=========================

Build one final, epoch-separated, multi-channel spectrum per (target, epoch)
from a directory of per-channel/band JWST x1d files -- e.g. the
"<name>_ch{1..4}-{short,medium,long}_x1d.fits" products of a Stage-3 run,
or the "*_comp0_x1d.fits" products of miri_mrs_auto_pipeline.py /
miri_mrs_refine_extract.py.

This is a thin driver on top of combine_nirspec_orders.py -- it reuses
read_x1d / get_mjd / get_mjd_info / group_files_by_epoch / mjd_to_datestr /
save_combined_spectrum directly rather than duplicating them.

  1. group_files_by_epoch() (MJD-OBS/EXPSTART) clusters the input files so
     that files taken >max_gap_days apart are NEVER combined into the same
     "final" spectrum, even if they're the same target/proposal. NOTE: this
     only works if the input files still carry per-exposure MJD keywords --
     if they were already merged across epochs inside a single Stage-3
     association, that information is gone and no post-hoc script can
     recover it. Fix that upstream (per-epoch associations at Spec3), not
     here.

  2. Within each epoch cluster, spectra are stitched pairwise, bluest-first,
     onto a growing composite -- but only the new segment being added can
     ever be rescaled, and only onto the already-established composite,
     never the reverse. Once a segment is locked into the composite its
     calibration never changes again, so one noisy overlap late in the
     sequence (e.g. a faint source's reddest channel) cannot retroactively
     rescale everything combined before it -- a real failure mode this was
     built to fix, not a hypothetical. A segment is only rescaled if its
     overlap passes three checks: enough S/N to trust a ratio at all
     (min_scale_snr), the resulting factor lands within a physically
     plausible range (max_scale_factor -- real MRS channel-to-channel
     offsets are modest, not order-of-magnitude), and the factor itself is
     well constrained (max_rel_uncertainty). Failing any of these, the
     segment blends in at its native flux level instead.

Usage
-----
    python combine_epoch_spectra.py --indir stage3/ --pattern "*_x1d.fits" \\
        --outdir final_spectra/ --max-gap-days 1.0

or from a notebook:
    from combine_epoch_spectra import combine_epoch_spectra
    results = combine_epoch_spectra("stage3/", pattern="*_x1d.fits",
                                     outdir="final_spectra/")
"""
import argparse
import glob
import os
import re

import numpy as np
from astropy.stats import sigma_clip

from jspec.pipelines.combine_nirspec_orders import (
    read_x1d, get_mjd, get_mjd_info, group_files_by_epoch, mjd_to_datestr,
    save_combined_spectrum,
)


def _sigma_clipped_ratio(f_hi, f_lo, sigma=3.0, maxiters=5):
    """Robust multiplicative factor to bring f_lo onto f_hi's level, from
    the pointwise ratio f_hi / f_lo, sigma-clipped so a few outlier pixels
    (a cosmic-ray residual, a channel-edge artifact) can't set the scale.
    Returns (factor, scatter_of_kept_ratios)."""
    ratio = f_hi / f_lo
    good = np.isfinite(ratio)
    if not good.any():
        return 1.0, np.nan
    clipped = sigma_clip(ratio[good], sigma=sigma, maxiters=maxiters, masked=True)
    kept = clipped.compressed()
    if kept.size == 0:
        return 1.0, np.nan
    return float(np.median(kept)), float(np.std(kept))


def _snr_in_overlap(wave, flux, err, lo, hi):
    m = (wave >= lo) & (wave <= hi) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    if not m.any():
        return np.nan
    return float(np.nanmedian(np.abs(flux[m]) / err[m]))


def merge_pair_adaptive(wave1, flux1, err1, wave2, flux2, err2,
                         auto_scale=True, clip_sigma=3.0, min_scale_snr=3.0,
                         max_scale_factor=2.0, max_rel_uncertainty=0.3):
    """Merge two spectra in wavelength, blending their overlap with an
    inverse-variance weighted average -- same core algorithm as
    combine_nirspec_orders._merge_pair, but with adaptive, one-directional
    rescaling: only spectrum 2 (the new segment being added) can ever be
    rescaled, and only onto spectrum 1 (the already-established composite
    built from every earlier merge). Spectrum 1 is NEVER rescaled.

    This matters because this function is called repeatedly, bluest-first,
    with spectrum 1 being the growing composite of everything merged so
    far. Earlier versions chose whichever side had the lower S/N and
    rescaled *that* one -- which meant a single noisy overlap late in the
    sequence (e.g. a faint source's reddest, lowest-S/N channel) could
    retroactively rescale the ENTIRE already-combined spectrum built from
    every earlier, well-behaved overlap. One-directional rescaling makes
    that structurally impossible: once a segment is blended into the
    composite, its calibration is locked and later merges cannot touch it.
    The scale factor for the new segment, if trusted at all, is a single
    number applied to that whole segment (a channel/band calibration
    offset is expected to be uniform across its own segment) -- but it
    only ever affects that one new segment, never anything upstream of it.

    No rescaling is applied when any of the following hold -- each is a
    real failure mode, not a hypothetical:
      - either side's overlap S/N is below `min_scale_snr` (too noisy to
        trust a ratio from at all);
      - the factor falls outside [1/max_scale_factor, max_scale_factor].
        Real MRS channel-to-channel calibration mismatches are modest (a
        few percent up to perhaps 30-50%); a much larger "correction" at a
        faint, noisy overlap is a sign the ratio itself is noise-dominated,
        not a real offset, however clean the nominal S/N looked;
      - the factor's own sigma-clipped uncertainty exceeds
        `max_rel_uncertainty` of the factor itself (factor_std / |factor|)
        -- a poorly-constrained ratio should not be trusted even when it
        happens to land in a plausible range. This is exactly what caught
        the bad ch4-short/ch4-medium case: 16.67 +/- 7.74 has ~46%
        relative uncertainty, which a bare S/N or range check alone
        would have missed.
    There's also no rescaling when there's no wavelength overlap at all.
    In every rejected case, spectrum 2 blends in at its native flux level
    -- the inverse-variance weighting in the overlap already downweights
    whichever side is noisier, without needing an explicit rescale.

    Returns (wave, flux, err, info), where info = {"scaled": bool,
    "factor": float, "factor_std": float, "reason": str or None}.
    `reason` explains why no rescaling was applied when "scaled" is False
    and there WAS an overlap (as opposed to none existing)."""
    if wave1.min() > wave2.min():
        wave1, flux1, err1, wave2, flux2, err2 = wave2, flux2, err2, wave1, flux1, err1

    ov_lo, ov_hi = wave2.min(), wave1.max()
    info = {"scaled": False, "factor": 1.0, "factor_std": np.nan, "reason": None}

    if ov_hi <= ov_lo:
        wave = np.concatenate([wave1, wave2])
        flux = np.concatenate([flux1, flux2])
        err = np.concatenate([err1, err2])
        order = np.argsort(wave)
        return wave[order], flux[order], err[order], info

    blue_mask = wave1 < ov_lo
    red_mask = wave2 > ov_hi
    ov1_mask = (wave1 >= ov_lo) & (wave1 <= ov_hi)
    ov2_mask = (wave2 >= ov_lo) & (wave2 <= ov_hi)

    if auto_scale:
        snr1 = _snr_in_overlap(wave1, flux1, err1, ov_lo, ov_hi)
        snr2 = _snr_in_overlap(wave2, flux2, err2, ov_lo, ov_hi)
        if not (np.isfinite(snr1) and np.isfinite(snr2) and min(snr1, snr2) >= min_scale_snr):
            info["reason"] = (f"overlap S/N too low (established={snr1:.2f}, "
                               f"new={snr2:.2f}, need >={min_scale_snr})")
        else:
            ov_grid = np.unique(np.concatenate([wave1[ov1_mask], wave2[ov2_mask]]))
            f1_i = np.interp(ov_grid, wave1[ov1_mask], flux1[ov1_mask])
            f2_i = np.interp(ov_grid, wave2[ov2_mask], flux2[ov2_mask])
            # Always bring the new segment (2) onto the established composite (1).
            factor, fstd = _sigma_clipped_ratio(f1_i, f2_i, sigma=clip_sigma)

            rel_unc = (fstd / abs(factor)) if (np.isfinite(fstd) and factor != 0) else np.inf
            out_of_range = not (1.0 / max_scale_factor <= factor <= max_scale_factor)
            poorly_constrained = rel_unc > max_rel_uncertainty

            if out_of_range or poorly_constrained:
                why = []
                if out_of_range:
                    why.append(f"factor {factor:.3f} outside "
                               f"[{1/max_scale_factor:.3f}, {max_scale_factor:.3f}]")
                if poorly_constrained:
                    why.append(f"relative uncertainty {rel_unc:.2f} > {max_rel_uncertainty}")
                info["reason"] = "rejected: " + "; ".join(why)
            else:
                flux2 = flux2 * factor
                err2 = err2 * abs(factor)
                info.update(scaled=True, factor=factor, factor_std=fstd)

    ov_grid = np.unique(np.concatenate([wave1[ov1_mask], wave2[ov2_mask]]))
    f1_i = np.interp(ov_grid, wave1[ov1_mask], flux1[ov1_mask])
    e1_i = np.interp(ov_grid, wave1[ov1_mask], err1[ov1_mask])
    f2_i = np.interp(ov_grid, wave2[ov2_mask], flux2[ov2_mask])
    e2_i = np.interp(ov_grid, wave2[ov2_mask], err2[ov2_mask])

    w1, w2 = 1.0 / e1_i**2, 1.0 / e2_i**2
    flux_ov = (f1_i * w1 + f2_i * w2) / (w1 + w2)
    err_ov = np.sqrt(1.0 / (w1 + w2))

    wave_out = np.concatenate([wave1[blue_mask], ov_grid, wave2[red_mask]])
    flux_out = np.concatenate([flux1[blue_mask], flux_ov, flux2[red_mask]])
    err_out = np.concatenate([err1[blue_mask], err_ov, err2[red_mask]])
    order = np.argsort(wave_out)
    return wave_out[order], flux_out[order], err_out[order], info


def combine_epoch_spectrum(x1d_files, dq_reject=True, auto_scale=True,
                            clip_sigma=3.0, min_scale_snr=3.0,
                            max_scale_factor=2.0, max_rel_uncertainty=0.3,
                            exclude_ranges=None, carry_forward=False,
                            trim_frac=0.02, min_overlap_um=0.05,
                            solver="global"):
    """... existing docstring ...
    exclude_ranges : list of (lo, hi) tuples, optional
        Wavelength ranges (um) to drop from every input spectrum before
        merging -- e.g. a known bad plane/cosmic-ray-contaminated bin in
        one band. Applied per-file, before the bluest-first merge.
    trim_frac : float
        Fraction of each input's wavelength span to discard from each
        end before merging. MRS band extremes have poor and rapidly
        varying throughput, and the background estimate there is the
        least constrained, so the edge is both the worst-calibrated part
        of a band *and* the part the next band is scaled against.

        On ``SN2024aecx_2025-12-07`` the ch4-short spectrum jumps by a
        factor 2.1 across its own blue edge at 18.0 um. Its only overlap
        with ch3-long is 17.70-17.98 um, which lies entirely inside the
        bad region, so the scale factor for the whole band was derived
        from it and the error was carried across the full 17.7-21 um.
        Trimming the edge first means the factor is measured where both
        bands are trustworthy.

        Kept small (2% by default) because MRS overlaps are only
        0.14-0.28 um wide; see `min_overlap_um`.
    min_overlap_um : float
        Trimming is backed off for a pair that would be left with less
        overlap than this. There is no point trimming a bad edge if it
        leaves nothing to scale on -- better to merge on marginal data
        and have the scale log say so.
    solver : {"global", "sequential"}
        How the per-band scale factors are determined.

        ``"global"`` (default) hands the bands to :func:`jspec.stitch.stitch`,
        which measures every overlap and then solves all 12 scale factors
        **simultaneously** as one weighted least-squares problem on the
        overlap graph, with robust down-weighting of discrepant joins.

        ``"sequential"`` is the original bluest-first chain: merge band 1
        into band 2, scale band 3 onto that composite, and so on. It is kept
        for reproducing old outputs, but it has two structural faults that
        showed up on real MRS data. First, it is **order-dependent and
        cumulative**: an error in one seam is inherited by everything redward
        of it. Second, when a ratio fails its quality cuts the band is simply
        left at its native flux level -- which does not avoid a step, it
        guarantees one. On ``SN2024aecx_2025-12-07`` eight of eleven seams
        were rejected this way, so most of the spectrum was never calibrated
        onto anything.

        The global solve has neither problem: a marginal overlap contributes
        with a low weight instead of being discarded, and no seam can
        dictate the relative calibration of two halves of the spectrum. It
        also reports when the overlap graph is **disconnected**, which the
        sequential merger cannot even represent -- on the same dataset ch2-long
        has no usable overlap on either side, so ch1+ch2 and ch3+ch4 are two
        independently anchored groups whose relative flux level is simply
        whatever the data happened to have."""
    specs = [read_x1d(f, dq_reject=dq_reject) for f in x1d_files]

    if exclude_ranges:
        cleaned = []
        for wave, flux, err in specs:
            keep = np.ones_like(wave, dtype=bool)
            for lo, hi in exclude_ranges:
                keep &= ~((wave >= lo) & (wave <= hi))
            cleaned.append((wave[keep], flux[keep], err[keep]))
        specs = cleaned

    if solver == "global":
        # Deliberately *before* the trim block. The trim below deletes points,
        # which shrinks the overlaps the scale factors are measured in; on
        # SN2024aecx_2025-12-07 that broke the overlap graph into four
        # disconnected calibration groups instead of three. stitch() achieves
        # the same thing non-destructively -- its own `edge_trim` excludes the
        # band edges from the *ratio measurement* while keeping the data in
        # the output -- so hand `trim_frac` to it and leave the arrays alone.
        return _combine_global(specs, x1d_files, edge_trim=trim_frac,
                               clip_sigma=clip_sigma,
                               max_scale_factor=max_scale_factor,
                               max_rel_uncertainty=max_rel_uncertainty,
                               min_scale_snr=min_scale_snr,
                               auto_scale=auto_scale)

    if trim_frac and trim_frac > 0:
        # How much each band would like to lose from each end.
        want = []
        for wave_i, _f, _e in specs:
            if wave_i.size < 50:
                want.append(0.0)
                continue
            span = float(wave_i.max() - wave_i.min())
            # Never trim away more than a third of a band, whatever the
            # caller asked for.
            want.append(min(trim_frac * span, span / 6.0))

        # Budget the trim seam by seam. A single tight pair must only
        # disable trimming at *that* seam -- an all-or-nothing back-off
        # lets one narrow overlap leave every other band's bad edge in
        # place, which is the opposite of what this is for.
        cut_blue = [0.0] * len(specs)
        cut_red = [0.0] * len(specs)
        byblue = list(np.argsort([sp[0].min() for sp in specs]))
        for n, idx in enumerate(byblue):
            if n == 0:                      # bluest end faces nothing
                cut_blue[idx] = want[idx]
            if n == len(byblue) - 1:        # reddest end faces nothing
                cut_red[idx] = want[idx]
        starved = []
        for a, b in zip(byblue[:-1], byblue[1:]):
            overlap = float(specs[a][0].max() - specs[b][0].min())
            budget = max(0.0, overlap - min_overlap_um) / 2.0
            cut_red[a] = min(want[a], budget)
            cut_blue[b] = min(want[b], budget)
            if overlap > 0 and budget <= 0:
                starved.append(f"{specs[a][0].max():.2f}/"
                               f"{specs[b][0].min():.2f}")

        trimmed, n_trimmed = [], 0
        for idx, (wave_i, flux_i, err_i) in enumerate(specs):
            lo_cut, hi_cut = cut_blue[idx], cut_red[idx]
            if lo_cut <= 0 and hi_cut <= 0:
                trimmed.append((wave_i, flux_i, err_i))
                continue
            keep = (wave_i >= wave_i.min() + lo_cut) & \
                   (wave_i <= wave_i.max() - hi_cut)
            if keep.sum() < 50:
                trimmed.append((wave_i, flux_i, err_i))
                continue
            trimmed.append((wave_i[keep], flux_i[keep], err_i[keep]))
            n_trimmed += 1
        specs = trimmed
        if n_trimmed:
            print(f"  edge trim ({trim_frac:.0%}): trimmed {n_trimmed} of "
                  f"{len(specs)} bands")
        if starved:
            print(f"  edge trim held back at {len(starved)} seam(s) with "
                  f"< {min_overlap_um} um to spare: {', '.join(starved)}")

    order = np.argsort([s[0].min() for s in specs])
    specs = [specs[i] for i in order]
    files_sorted = [x1d_files[i] for i in order]

    wave, flux, err = specs[0]
    scale_log = []
    cum = 1.0   # cumulative scale already applied to the composite's red end
    for (w2, f2, e2), fname in zip(specs[1:], files_sorted[1:]):
        if carry_forward and cum != 1.0:
            f2, e2 = f2 * cum, e2 * abs(cum)      # start from the last trusted calibration
        wave, flux, err, info = merge_pair_adaptive(
            wave, flux, err, w2, f2, e2, auto_scale=auto_scale,
            clip_sigma=clip_sigma, min_scale_snr=min_scale_snr,
            max_scale_factor=max_scale_factor, max_rel_uncertainty=max_rel_uncertainty)
        if info["scaled"]:
            cum *= info["factor"]
        info["cumulative_factor"] = cum
        info["file"] = os.path.basename(fname)
        scale_log.append(info)

    keep = np.argsort(wave)
    wave, flux, err = wave[keep], flux[keep], err[keep]
    mjd_info = get_mjd_info(x1d_files)
    return wave, flux, err, mjd_info, scale_log



def _band_name(path):
    """ch2-long out of ..._ch2-long_abi_x1d.fits, else the bare stem."""
    base = os.path.basename(path)
    m = re.search(r"(ch\d-(?:short|medium|long))", base)
    if m:
        return m.group(1)
    return re.sub(r"_x1d\.fits$", "", base)


def _combine_global(specs, x1d_files, edge_trim=0.02, clip_sigma=3.0,
                    max_scale_factor=2.0, max_rel_uncertainty=0.3,
                    min_scale_snr=3.0, auto_scale=True):
    """Scale and merge the bands with the global (graph) solver.

    Returns the same ``(wave, flux, err, mjd_info, scale_log)`` tuple as the
    sequential path, so callers do not need to know which solver ran. The
    scale_log entries keep the ``scaled``/``factor``/``reason`` keys the
    printing code expects, and add ``joins`` and ``report`` on the first
    entry for callers that want the full diagnosis.
    """
    from jspec.spectrum import Segment, Spectrum
    from jspec.stitch import stitch

    segments = []
    for (w, f, e), path in zip(specs, x1d_files):
        if w.size < 10:
            continue
        segments.append(Segment(name=_band_name(path),
                                spectrum=Spectrum(wavelength=w, flux=f, flux_err=e),
                                path=path))
    if len(segments) < 2:
        w, f, e = specs[0]
        keep = np.argsort(w)
        return w[keep], f[keep], e[keep], get_mjd_info(x1d_files), []

    report = stitch(segments, edge_trim=edge_trim, clip_sigma=clip_sigma,
                    max_scale_factor=max_scale_factor,
                    max_rel_uncertainty=max_rel_uncertainty,
                    min_snr=min_scale_snr, apply_scales=auto_scale,
                    check_epoch=False, assess_quality=True,
                    drop_blank=False, verify=True)

    sp = report.spectrum
    scales = getattr(report, "scales", {}) or {}
    by_name = {s.name: s for s in segments}

    scale_log = []
    for name, seg in by_name.items():
        factor = float(scales.get(name, 1.0))
        sigma = float((getattr(report, "scale_sigma", {}) or {}).get(name, np.nan))
        scaled = abs(factor - 1.0) > 1e-9
        scale_log.append({
            "scaled": scaled,
            "factor": factor,
            "factor_std": 0.0 if not np.isfinite(sigma) else sigma,
            "cumulative_factor": factor,
            "reason": "" if scaled else "solved scale is 1 (anchor, or no "
                                        "usable overlap constraint)",
            "file": os.path.basename(seg.path or name),
        })
    if scale_log:
        scale_log[0]["report"] = report

    return (np.asarray(sp.wavelength), np.asarray(sp.flux),
            np.asarray(sp.flux_err), get_mjd_info(x1d_files), scale_log)


def combine_epoch_spectra(indir, pattern="*_x1d.fits", outdir="final_spectra",
                           max_gap_days=1.0, auto_scale=True, clip_sigma=3.0,
                           min_scale_snr=3.0, max_scale_factor=2.0,
                           max_rel_uncertainty=0.3, dq_reject=True,
                           exclude_ranges=None, carry_forward=False):
    """Discover x1d files under `indir`, cluster them into epochs
    (group_files_by_epoch, MJD-based, never combining across
    `max_gap_days`), and write one combined spectrum per epoch into
    `outdir`. Returns a list of dicts: {epoch_date, files, out_base,
    n_files, scale_log}."""
    os.makedirs(outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(indir, pattern)))
    if not files:
        raise SystemExit(f"No files matching {pattern!r} found in {indir}")

    clusters = group_files_by_epoch(files, max_gap_days=max_gap_days)
    print(f"{len(files)} file(s) -> {len(clusters)} epoch cluster(s) "
          f"(>{max_gap_days} day gap = new epoch)")

    results = []
    for cluster in clusters:
        mjds = [m for m in (get_mjd(f) for f in cluster) if m is not None]
        date = mjd_to_datestr(np.mean(mjds)) if mjds else "unknown-date"
        wave, flux, err, mjd_info, scale_log = combine_epoch_spectrum(
            cluster, dq_reject=dq_reject, auto_scale=auto_scale,
            clip_sigma=clip_sigma, min_scale_snr=min_scale_snr,
            max_scale_factor=max_scale_factor, max_rel_uncertainty=max_rel_uncertainty,
            exclude_ranges=exclude_ranges, carry_forward=carry_forward)
        out_base = os.path.join(outdir, f"final_combined_{date}")
        save_combined_spectrum(wave, flux, err, out_base, mjd_info=mjd_info)
        print(f"\nEpoch {date}: {len(cluster)} file(s)")
        for s in scale_log:
            if s["scaled"]:
                print(f"  scaled new segment by {s['factor']:.4f} "
                      f"+/- {s['factor_std']:.4f} to match established composite "
                      f"(file: {s['file']})")
            else:
                print(f"  no rescale applied (file: {s['file']}) -- {s['reason']}")
        results.append({"epoch_date": date,
                         "files": [os.path.basename(f) for f in cluster],
                         "out_base": out_base, "n_files": len(cluster),
                         "scale_log": scale_log})
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", required=True, help="Directory containing x1d files")
    ap.add_argument("--pattern", default="*_x1d.fits",
                     help="Glob pattern within --indir (default: *_x1d.fits)")
    ap.add_argument("--outdir", default="final_spectra")
    ap.add_argument("--max-gap-days", type=float, default=1.0,
                     help="Files more than this many days apart (MJD) are "
                          "never combined into the same final spectrum")
    ap.add_argument("--no-auto-scale", action="store_true",
                     help="Disable adaptive overlap rescaling; blend as-is")
    ap.add_argument("--clip-sigma", type=float, default=3.0,
                     help="Sigma-clip threshold for the overlap ratio")
    ap.add_argument("--min-scale-snr", type=float, default=3.0,
                     help="Skip rescaling if either side's overlap S/N is below this")
    ap.add_argument("--max-scale-factor", type=float, default=2.0,
                     help="Skip rescaling if the factor falls outside "
                          "[1/this, this] -- real channel-to-channel offsets "
                          "are modest; a larger 'correction' means the ratio "
                          "is noise-dominated, not real")
    ap.add_argument("--max-rel-uncertainty", type=float, default=0.3,
                     help="Skip rescaling if factor_std / |factor| exceeds this "
                          "-- a poorly-constrained ratio, even if plausible-looking")
    args = ap.parse_args()

    results = combine_epoch_spectra(
        args.indir, pattern=args.pattern, outdir=args.outdir,
        max_gap_days=args.max_gap_days, auto_scale=not args.no_auto_scale,
        clip_sigma=args.clip_sigma, min_scale_snr=args.min_scale_snr,
        max_scale_factor=args.max_scale_factor,
        max_rel_uncertainty=args.max_rel_uncertainty)

    print(f"\n{len(results)} epoch(s) combined:")
    for r in results:
        print(f"  {r['epoch_date']}: {r['n_files']} file(s) -> "
              f"{r['out_base']}.fits / .ecsv")


if __name__ == "__main__":
    main()