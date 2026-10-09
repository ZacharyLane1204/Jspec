#!/usr/bin/env python3
"""
nirspec_miri_calibration_utils.py
==================================

Two helpers for reconciling `nirspec_fs_auto_pipeline.py` /
`nirspec_refine_extract.py` output with `miri_mrs_refine_extract.py` output:

1. `standardize_miri_ecsv()` -- rewrite a MIRI "_full"/"_clean" ecsv in the
   same column/meta convention as the NIRSpec "_final" ecsv (lowercase
   WAVELENGTH/FLUX/FLUX_ERROR, same meta keys), so downstream code (plotting,
   PySOOT, combine_nirspec_orders-style stitching) can treat both the same
   way.

2. `check_instrument_overlap()` -- the NIRSpec/MIRI equivalent of
   `combine_nirspec_orders.check_overlap_offset()`, but instrument-aware:
   fits a local line to each side of the join and flags whether a step is
   a flat calibration offset or real spectral curvature (PAH/silicate/line
   wing), the same test `miri_mrs_refine_extract.stitch_steps()` already
   uses for the internal MRS sub-band joins -- just applied to the
   NIRSpec/MIRI boundary, which that function does not check.

Usage
-----
    from nirspec_miri_calibration_utils import standardize_miri_ecsv, check_instrument_overlap

    standardize_miri_ecsv(
        "SN2024aecx_01062025_full.ecsv", "sn2024aecx_miri_20250601_final.ecsv",
        err_column="FLUX_ERR_TOTAL", clean_only=True,
        target="2024aecx", channel="ch1-4", epoch=2, epoch_date="20250601",
    )

    check_instrument_overlap(
        "2024aecx_s400a1_20250601_final.ecsv",   # NIRSpec, lowercase schema
        "sn2024aecx_miri_20250601_final.ecsv",   # MIRI, now same schema
    )
"""
from __future__ import annotations

import numpy as np
from astropy.table import Table


# ---------------------------------------------------------------------------
# 1. schema harmonization
# ---------------------------------------------------------------------------
def standardize_miri_ecsv(miri_ecsv_path, out_path, err_column="FLUX_ERR_TOTAL",
                           clean_only=True, target=None, channel=None,
                           epoch=None, epoch_date=None, extract_note=None):
    """Rewrite a miri_mrs_refine_extract.py "_full"/"_clean" ecsv with the
    same column names and meta layout as nirspec_fs_auto_pipeline.py's
    "_final" ecsv, so both can be read/plotted/combined identically.

    Parameters
    ----------
    miri_ecsv_path : str
        The MIRI "*_full.ecsv" (or "*_clean.ecsv") from miri_mrs_refine_extract.py.
    out_path : str
        Where to write the standardized ecsv.
    err_column : str
        Which MIRI error column becomes the single `FLUX_ERROR`. Default
        "FLUX_ERR_TOTAL" (= sqrt(FLUX_ERROR**2 + FLUX_SYS**2)), the closest
        analogue of NIRSpec's single formal `FLUX_ERROR` -- it folds in both
        the random-noise inflation (R_white) and the hole-fill/wing-bias
        systematic (FLUX_SYS). Use "FLUX_ERROR" instead only if you
        specifically want the random-only term (e.g. for an inverse-
        variance fit where you are handling the correlated systematic
        separately -- FLUX_SYS is correlated across neighbouring points in
        a way FLUX_ERROR is not, so quadrature-summing it into a per-point
        weight for a least-squares fit will understate how much two
        adjacent points really agree).
    clean_only : bool
        If True (default), drop rows flagged EDGE/SPIKE/HOSTLINE/BKG_FAIL/
        RED_END before writing -- matching what `_clean.ecsv` already does,
        so this is a no-op if you pass a "_clean.ecsv" in the first place.
        Set False to keep every row from a "_full.ecsv" (e.g. to inspect
        flagged points yourself).
    target, channel, epoch, epoch_date, extract_note : optional
        Meta values to carry into the output header, mirrored from the
        NIRSpec convention (TARGET, SLIT, EPOCH, EPOCHDT, GRATINGS, EXTRACT).
        `channel` fills the NIRSpec "SLIT" slot's role and also a new
        "CHANNEL" key (MIRI has no slit). Anything you don't pass through
        is carried over from the input file's own meta where a matching
        key exists (TARGNAME -> TARGET, MJD -> MJD_MEAN).
    """
    t = Table.read(miri_ecsv_path, format="ascii.ecsv")

    if clean_only and "FLAG" in t.colnames:
        FLAG = dict(EDGE=1, SPIKE=2, HOSTLINE=4, BKG_CAPPED=8, BKG_FAIL=16,
                    RED_END=32, BLENDED=64, SYS_EXTRAP=128)
        exclude = FLAG["EDGE"] | FLAG["SPIKE"] | FLAG["HOSTLINE"] | FLAG["BKG_FAIL"] | FLAG["RED_END"]
        t = t[(t["FLAG"].data.astype(int) & exclude) == 0]

    if err_column not in t.colnames:
        raise KeyError(f"{err_column!r} not in {miri_ecsv_path}; available: {t.colnames}")

    from jspec.columns import spectrum_table
    out = spectrum_table(
        t["WAVELENGTH"].data.astype(float),
        t["FLUX"].data.astype(float),
        t[err_column].data.astype(float),
        unit=str(t["FLUX"].unit or "Jy"),
    )

    meta_in = t.meta
    out.meta["TARGET"] = target or meta_in.get("TARGNAME")
    out.meta["CHANNEL"] = channel or "MRS"
    out.meta["EPOCH"] = epoch
    out.meta["EPOCHDT"] = epoch_date
    mjd = meta_in.get("MJD")
    out.meta["MJD_MIN"] = mjd
    out.meta["MJD_MAX"] = mjd
    out.meta["MJD_MEAN"] = mjd
    out.meta["EXTRACT"] = extract_note or (
        f"miri_mrs_refine_extract.py, err={err_column}, "
        f"{'clean' if clean_only else 'full'}, RSCALE={meta_in.get('RSCALE')}"
    )
    out.meta = {k: v for k, v in out.meta.items() if v is not None}

    out.write(out_path, format="ascii.ecsv", overwrite=True)
    print(f"wrote {out_path}  ({len(out)} rows, FLUX_ERROR <- {err_column}, "
          f"{'clean' if clean_only else 'full'})")
    return out


# ---------------------------------------------------------------------------
# 2. NIRSpec/MIRI overlap step check
# ---------------------------------------------------------------------------
def _local_linear_fit(w, f, centre, half_width, gap):
    """Fit a line to points within [centre-half_width, centre-gap] (side<0)
    or [centre+gap, centre+half_width] (side>0); same construction as
    miri_mrs_refine_extract.stitch_steps(). Returns (intercept_at_centre,
    sigma, curvature_flag) or None if too few points."""
    results = []
    for side in (-1, 1):
        lo, hi = (centre - half_width, centre - gap) if side < 0 else (centre + gap, centre + half_width)
        m = (w >= lo) & (w <= hi) & np.isfinite(f)
        if m.sum() < 10:
            return None
        x, y = w[m] - centre, f[m]
        p = np.polyfit(x, y, 1)
        res = y - np.polyval(p, x)
        s_res = 1.4826 * np.median(np.abs(res - np.median(res)))
        A = np.vander(x, 2)
        cov = s_res ** 2 * np.linalg.inv(A.T @ A)

        p2 = np.polyfit(x, y, 2)
        r2 = y - np.polyval(p2, x)
        s2 = 1.4826 * np.median(np.abs(r2 - np.median(r2)))
        A2 = np.vander(x, 3)
        c2 = s2 ** 2 * np.linalg.inv(A2.T @ A2)
        clean = bool(abs(p2[0]) < 3.0 * np.sqrt(c2[0, 0]))
        results.append((p[1], float(np.sqrt(cov[1, 1])), clean))
    return results


def check_instrument_overlap(nirspec_ecsv_path, miri_ecsv_path, half_width=0.3, gap=0.02, snr_min=3.0):
    """Report the flux step between NIRSpec (blue side) and MIRI (red side)
    right at their wavelength overlap, using the same local-linear-fit +
    curvature test as the internal MRS sub-band joins in
    miri_mrs_refine_extract.stitch_steps() -- but this join (NIRSpec/MIRI)
    is never checked by either pipeline on its own, so run it explicitly
    per epoch.

    Column names are normalised on read, so both the canonical
    WAVELENGTH/FLUX/FLUX_ERROR schema and the older lower-case one work
    (run `standardize_miri_ecsv` on the MIRI file first).

    A step with `clean=True` on both sides (no significant local curvature)
    and `step_pct` outside roughly +/- 2*err_pct is evidence of an
    instrumental flux offset (under/over-subtraction, aperture-fraction
    mismatch) rather than real spectral shape -- a step under 1 sigma, or
    one where either side is flagged not-clean, should not be corrected by
    rescaling.
    """
    ns = Table.read(nirspec_ecsv_path, format="ascii.ecsv")
    mi = Table.read(miri_ecsv_path, format="ascii.ecsv")
    from jspec.columns import normalize_columns
    normalize_columns(ns); normalize_columns(mi)
    w_ns, f_ns = ns["WAVELENGTH"].data.astype(float), ns["FLUX"].data.astype(float)
    w_mi, f_mi = mi["WAVELENGTH"].data.astype(float), mi["FLUX"].data.astype(float)

    ov_lo, ov_hi = w_mi.min(), w_ns.max()
    if ov_hi <= ov_lo:
        print(f"No wavelength overlap: NIRSpec ends at {w_ns.max():.3f} um, "
              f"MIRI starts at {w_mi.min():.3f} um.")
        return None
    centre = 0.5 * (ov_lo + ov_hi)
    print(f"Overlap band: {ov_lo:.3f}-{ov_hi:.3f} um (centre {centre:.3f} um)")

    blue = _local_linear_fit(w_ns, f_ns, centre, half_width, gap)
    red = _local_linear_fit(w_mi, f_mi, centre, half_width, gap)
    if blue is None or red is None:
        print("Too few points within +/-{:.2f} um of the join on one side "
              "(widen half_width or check the overlap band is real).".format(half_width))
        return None

    l, el, clean_l = blue[1]   # NIRSpec fit, side>0 (its red/near-join side)
    u, eu, clean_u = red[0]    # MIRI fit, side<0 (its blue/near-join side)

    snr_l, snr_u = abs(l) / el if el > 0 else np.inf, abs(u) / eu if eu > 0 else np.inf
    if snr_l < snr_min or snr_u < snr_min:
        weak = "NIRSpec" if snr_l < snr_min else "MIRI"
        print(f"NIRSpec @ join: {l:.4g} +/- {el:.2g}  (S/N {snr_l:.1f})")
        print(f"MIRI    @ join: {u:.4g} +/- {eu:.2g}  (S/N {snr_u:.1f})")
        print(f"{weak} is consistent with a non-detection at the join (S/N < {snr_min:g}) -- "
              f"a percentage step is not meaningful here. Absolute difference: "
              f"{u - l:.3g} +/- {np.hypot(el, eu):.2g} "
              f"({abs(u - l) / np.hypot(el, eu):.1f} sigma from zero difference).")
        return dict(overlap=(ov_lo, ov_hi), nirspec_val=(l, el), miri_val=(u, eu),
                    step_pct=np.nan, err_pct=np.nan, clean=clean_l and clean_u,
                    sigma=np.nan, non_detection_side=weak)

    step_pct = 100 * (u / l - 1)
    err_pct = 100 * np.hypot(el, eu) / abs(l)
    clean = clean_l and clean_u
    sig = abs(step_pct) / err_pct if err_pct > 0 else np.nan

    print(f"NIRSpec @ join: {l:.4g} +/- {el:.2g}  (locally {'featureless' if clean_l else 'curved'})")
    print(f"MIRI    @ join: {u:.4g} +/- {eu:.2g}  (locally {'featureless' if clean_u else 'curved'})")
    print(f"step: {step_pct:+.1f}% +/- {err_pct:.1f}%  ({sig:.1f} sigma)  "
          f"-> {'likely calibration offset' if clean and sig > 2 else 'not a clear calibration offset'}")
    if clean and sig > 2:
        print(f"  suggested multiplicative scale to bring MIRI onto NIRSpec at the join: {l / u:.4f}")
    return dict(overlap=(ov_lo, ov_hi), nirspec_val=(l, el), miri_val=(u, eu),
                step_pct=step_pct, err_pct=err_pct, clean=clean, sigma=sig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--miri-full", required=True, help="MIRI *_full.ecsv or *_clean.ecsv from miri_mrs_refine_extract.py")
    ap.add_argument("--nirspec-final", required=True, help="NIRSpec *_final.ecsv from nirspec_fs_auto_pipeline.py")
    ap.add_argument("--out", default=None, help="Where to write the standardized MIRI ecsv (default: alongside --miri-full)")
    ap.add_argument("--err-column", default="FLUX_ERR_TOTAL",
                    choices=["FLUX_ERROR", "FLUX_SYS", "FLUX_ERR_TOTAL", "FLUX_ERR_FORMAL"])
    ap.add_argument("--keep-flagged", action="store_true")
    a = ap.parse_args()
    out = a.out or a.miri_full.replace("_full.ecsv", "_std.ecsv").replace("_clean.ecsv", "_std.ecsv")
    standardize_miri_ecsv(a.miri_full, out, err_column=a.err_column, clean_only=not a.keep_flagged)
    check_instrument_overlap(a.nirspec_final, out)