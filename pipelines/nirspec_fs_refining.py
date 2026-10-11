#!/usr/bin/env python3
"""
nirspec_fs_refining.py
======================

Small shared toolbox for the NIRSpec fixed-slit scripts. Nothing here is
clever -- it is the handful of things the STScI notebook
(nirspec_x1d_process.ipynb) does by hand, written once:

  * header helpers and EPOCH grouping (so different visits are never mixed)
  * reading the pipeline's x1d / s2d products
  * a cheap, never-failing check of where the source sits (`estimate_position`)
  * stitching gratings into one spectrum (inverse-variance mean in overlaps)
  * saving a spectrum (ECSV + FITS) and making a QA plot
  * re-running the pipeline's own Extract1dStep on the Stage-3 s2d with the
    extraction box moved to a row you choose (notebook section 9)

What is deliberately NOT here any more: matched-filter centroiding (a faint
source made it fail), polynomial background refits, hole-filling, deblending,
error-budget decomposition, and the "manual extraction + calibration transfer"
trick. The pipeline already knows
where the source is (the nod/slit position), and Extract1dStep on the s2d
gives flux-calibrated Jy directly.

Depends only on numpy / astropy / matplotlib. jwst is imported lazily, and only
by `pipeline_extraction_box`, `reextract_at_row` and `_wave_from_wcs`.
"""
from __future__ import annotations

import glob
import json
import os
import re
import warnings
from collections import defaultdict

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

FS_EXP_TYPE = "NRS_FIXEDSLIT"


# ===========================================================================
# Headers, epochs
# ===========================================================================

def sanitize(name):
    """'SN 2024aecx' -> 'sn2024aecx' (safe for filenames / product names)."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _as_bool(v):
    if isinstance(v, str):
        return v.strip().upper() in ("T", "TRUE", "1")
    return bool(v) if v is not None else False


def get_mjd_range(header):
    """(start, end) MJD from EXPSTART/EXPEND, falling back to DATE-OBS +
    TIME-OBS. (None, None) if the header has neither."""
    start, end = header.get("EXPSTART"), header.get("EXPEND")
    if start is None:
        d, t = header.get("DATE-OBS"), header.get("TIME-OBS")
        if d and t:
            try:
                start = Time(f"{d}T{t}", format="isot", scale="utc").mjd
            except Exception:
                start = None
    if start is None:
        return None, None
    return float(start), float(end if end is not None else start)


def mjd_to_datestr(mjd):
    """MJD -> 'YYYY-MM-DD' (UTC).

    Delegates to :func:`jspec.layout.epoch_datestr` so that the FS route
    labels epochs exactly the way the MRS route does. This used to
    return 'YYYYMMDD', which is why one target could produce both
    ``2024aecx_20250601`` and ``SN2024aecx_2025-06-01`` directories.
    """
    from jspec.layout import epoch_datestr
    return epoch_datestr(mjd)


def read_meta(path):
    """Everything the pipeline scripts need to know about one exposure, read
    once from the primary header (works on uncal / rate / cal / s2d / x1d)."""
    h = fits.getheader(path, 0)
    start, end = get_mjd_range(h)
    patt_num, subpxpts = h.get("PATT_NUM"), h.get("SUBPXPTS") or 1
    return dict(
        path=os.path.abspath(path),
        name=os.path.basename(path),
        program=h.get("PROGRAM"),
        target=str(h.get("TARGNAME") or h.get("TARGPROP") or "unknown").strip(),
        targprop=str(h.get("TARGPROP") or "").strip(),
        exp_type=h.get("EXP_TYPE"),
        detector=h.get("DETECTOR"),
        filter=h.get("FILTER"),
        grating=h.get("GRATING"),
        slit=h.get("FXD_SLIT"),
        patttype=str(h.get("PATTTYPE") or ""),
        pridtpts=h.get("PRIDTPTS"),
        nod=((int(patt_num) - 1) // int(subpxpts)) if patt_num is not None else 0,
        is_bkg=_as_bool(h.get("BKGDTARG")),
        mjd_start=start,
        mjd_end=end,
    )


def assign_epochs(metas, gap_days=1.0):
    """Add 'epoch' (1, 2, ...), 'epoch_date' ('YYYYMMDD') and 'epoch_key'
    to every exposure dict, in place.

    Exposures are grouped by target and sorted by start time; a gap of more
    than `gap_days` between consecutive exposures starts a new epoch. This is
    what keeps separate visits from ever being background-subtracted against,
    or combined with, each other."""
    by_target = defaultdict(list)
    for m in metas:
        by_target[m["target"]].append(m)

    for target, lst in by_target.items():
        lst.sort(key=lambda m: m["mjd_start"] if m["mjd_start"] is not None else 0.0)
        epoch, prev = 0, None
        for m in lst:
            t = m["mjd_start"]
            if prev is None or (t is not None and t - prev > gap_days):
                epoch += 1
            if t is not None:
                prev = t
            m["epoch"] = epoch
        for e in {m["epoch"] for m in lst}:
            members = [m for m in lst if m["epoch"] == e]
            starts = [m["mjd_start"] for m in members if m["mjd_start"] is not None]
            date = mjd_to_datestr(min(starts)) if starts else "nodate"
            for m in members:
                m["epoch_date"] = date
                m["epoch_key"] = (target, e)
    return metas


def check_same_epoch(paths, max_gap_days=1.0):
    """Raise if the files' start times span more than `max_gap_days`.
    Returns (mjd_min, mjd_max) (None, None if headers have no times)."""
    starts, ends = [], []
    for p in paths:
        s, e = get_mjd_range(fits.getheader(p, 0))
        if s is not None:
            starts.append(s)
            ends.append(e)
    if not starts:
        return None, None
    if max(starts) - min(starts) > max_gap_days:
        raise ValueError(
            f"These files span {max(starts) - min(starts):.2f} days (> {max_gap_days}); "
            f"refusing to merge different epochs into one spectrum.")
    return min(starts), max(ends)


# ===========================================================================
# Reading products
# ===========================================================================

def _slit_set(slit):
    """None -> None; 'S200A1' -> {'S200A1'}; iterable -> upper-cased set."""
    if slit is None:
        return None
    if isinstance(slit, str):
        slit = [slit]
    return {str(s).strip().upper() for s in slit}


def read_x1d(path, label=None, slit=None, strict=True):
    """Return a list of spectrum 'pieces' (one per EXTRACT1D table) with
    wave [um], flux, err, unit. Point sources -> FLUX/FLUX_ERROR (Jy);
    extended -> SURF_BRIGHT/SB_ERROR (MJy/sr). Non-finite / zero-error rows
    are dropped.

    slit: keep only the EXTRACT1D tables of this slit (name or iterable of
    names, matched on the SLTNAME header keyword). A fixed-slit x1d can hold
    several slits; the non-primary ones are EXTENDED (MJy/sr) by default and
    are not your target, so they must not be mixed into the spectrum. If no
    table carries SLTNAME, nothing is filtered (with a warning). If SLTNAME is
    present but no table matches, the file belongs to another slit: with
    strict=True (default) it is skipped and [] is returned; with strict=False
    every table is used (with a warning).

    Each piece also carries 'slit' and 'srctype' (from the header)."""
    want = _slit_set(slit)
    tables = []
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.name == "EXTRACT1D" and hdu.data is not None:
                tables.append((str(hdu.header.get("SLTNAME", "") or "").strip().upper(),
                               str(hdu.header.get("SRCTYPE", "") or "").strip().upper(),
                               hdu.header.copy(), hdu.data))
    fname = os.path.basename(path)
    if want is not None and tables:
        names = {n for n, _, _, _ in tables}
        keep = [t_ for t_ in tables if t_[0] in want]
        if not any(names):
            print(f"    WARNING: {fname}: no SLTNAME in the EXTRACT1D headers; cannot select slit "
                  f"{sorted(want)} -- using every table.")
        elif not keep:
            found = sorted(n or "?" for n in names)
            if strict:
                print(f"    skipping {fname}: it holds slit {found}, not {sorted(want)}")
                return []
            print(f"    WARNING: {fname}: no EXTRACT1D table for slit {sorted(want)} "
                  f"(found {found}) -- using every table.")
        else:
            dropped = sorted(n or "?" for n, _, _, _ in tables if n not in want)
            if dropped:
                print(f"    {fname}: kept slit {sorted(want)}, ignored other slit table(s) {dropped}")
            tables = keep

    pieces = []
    for slitname, srctype, header, data in tables:
        cols = [c.upper() for c in data.columns.names]
        extended = srctype == "EXTENDED"
        sets = [("FLUX", "FLUX_ERROR", "Jy"), ("SURF_BRIGHT", "SB_ERROR", "MJy/sr")]
        if extended:
            sets.reverse()
        chosen = None
        for i, (fcol, ecol, unit) in enumerate(sets):
            if fcol in cols and ecol in cols:
                f = np.asarray(data[fcol], float)
                if np.isfinite(f).any() and np.nanmax(np.abs(f)) > 0:
                    if i > 0:
                        print(f"    WARNING: {fname} [{slitname or '?'}]: preferred column {sets[0][0]} is empty; "
                              f"fell back to {fcol} ({unit}).")
                    chosen = (f, np.asarray(data[ecol], float), unit)
                    break
        if chosen is None:
            continue
        f, e, unit = chosen
        w = np.asarray(data["WAVELENGTH"], float)
        good = np.isfinite(w) & np.isfinite(f) & np.isfinite(e) & (e > 0)
        if not good.any():
            continue
        order = np.argsort(w[good])
        lab = label or fname
        if slitname and want is not None and len(want) > 1:
            lab = f"{lab} [{slitname}]"
        pieces.append(dict(wave=w[good][order], flux=f[good][order], err=e[good][order],
                           unit=unit, source=path, label=lab, slit=slitname, srctype=srctype))
    return pieces


def _wave_from_wcs(path):
    """Fallback only: per-column wavelength from the s2d's WCS."""
    from stdatamodels.jwst import datamodels
    with datamodels.open(path) as m:
        ny, nx = m.data.shape
        world = m.meta.wcs(np.arange(nx, dtype=float), np.full(nx, ny / 2.0 - 0.5))
    return np.asarray(world[2] if len(world) == 3 else world[-1], float)


def read_s2d(path):
    """SCI image (rows = slit position, cols = wavelength), a per-column
    wavelength vector (or None) and the primary header. Pixels with zero
    drizzle weight are set to NaN."""
    with fits.open(path) as hdul:
        sci = np.array(hdul["SCI"].data, float)
        if "WHT" in hdul and hdul["WHT"].data is not None and hdul["WHT"].data.shape == sci.shape:
            sci = np.where(np.asarray(hdul["WHT"].data, float) > 0, sci, np.nan)
        wave = None
        if "WAVELENGTH" in hdul and hdul["WAVELENGTH"].data is not None:
            wa = np.array(hdul["WAVELENGTH"].data, float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wave = np.nanmedian(wa, axis=0) if wa.ndim == 2 else wa
        header = hdul[0].header.copy()
    if wave is None:
        try:
            wave = _wave_from_wcs(path)
        except Exception:
            wave = None
    if wave is not None and wave.shape[0] != sci.shape[1]:
        wave = None
    return dict(sci=sci, wave=wave, header=header)


def slit_name(s2d_path):
    for ext in ("SCI", 0):
        try:
            v = fits.getval(s2d_path, "SLTNAME" if ext == "SCI" else "FXD_SLIT", ext)
            if v and str(v).upper() not in ("N/A", "NONE"):
                return str(v)
        except Exception:
            pass
    raise KeyError(f"Cannot find the slit name in {s2d_path}")


def cross_profile(sci, wave=None, wave_range=None):
    """Median over wavelength of the s2d -> one value per slit row."""
    cols = np.ones(sci.shape[1], bool)
    if wave is not None and wave_range is not None:
        sel = np.isfinite(wave) & (wave >= wave_range[0]) & (wave <= wave_range[1])
        if sel.any():
            cols = sel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmedian(sci[:, cols], axis=1)


def _peak(profile, lo_row, hi_row):
    """Sub-pixel row of the brightest 3-pixel-smoothed point of `profile`
    inside rows lo_row..hi_row (parabola through the peak). NaN if empty."""
    seg = np.asarray(profile, float)[lo_row:hi_row + 1]
    if seg.size < 3 or not np.isfinite(seg).any():
        return np.nan
    seg = np.where(np.isfinite(seg), seg, np.nanmedian(seg))
    sm = np.convolve(np.pad(seg, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")
    k = int(np.argmax(sm))
    off = 0.0
    if 0 < k < sm.size - 1:
        curv = sm[k - 1] - 2 * sm[k] + sm[k + 1]
        if curv < 0:
            off = float(np.clip(0.5 * (sm[k - 1] - sm[k + 1]) / curv, -0.5, 0.5))
    return lo_row + k + off


def estimate_position(sci, wave=None, wave_range=None, y0=None, radius=6.0, n_sub=4):
    """Where does the source sit? A hint to compare with the pipeline's box --
    never applied automatically, and it cannot fail: a faint or absent source
    just gives inconsistent sub-band rows, which is reported, not raised.

    The brightest point of the median cross-slit profile is found within
    +-`radius` rows of `y0` (whole slit if y0 is None). The same is done in
    `n_sub` separate wavelength chunks; if those agree the position is
    trustworthy, if they scatter the source is too faint to locate this way
    in this band (try --wave-range around bright signal, or set --y yourself).

    Returns dict(y, bands, scatter, consistent)."""
    ny, nx = sci.shape
    lo_row, hi_row = 0, ny - 1
    if y0 is not None:
        lo_row, hi_row = max(0, int(round(y0 - radius))), min(ny - 1, int(round(y0 + radius)))

    if wave is not None and np.isfinite(wave).any():
        lo, hi = wave_range if wave_range else (np.nanmin(wave), np.nanmax(wave))
        edges = np.linspace(lo, hi, n_sub + 1)
        chunks = [(edges[i], edges[i + 1]) for i in range(n_sub)]
        y_all = _peak(cross_profile(sci, wave, (lo, hi)), lo_row, hi_row)
        bands = [_peak(cross_profile(sci, wave, c), lo_row, hi_row) for c in chunks]
    else:                                              # no wavelength info: split by column
        cuts = np.linspace(0, nx, n_sub + 1).astype(int)
        y_all = _peak(cross_profile(sci), lo_row, hi_row)
        bands = [_peak(cross_profile(sci[:, a:b]), lo_row, hi_row) for a, b in zip(cuts[:-1], cuts[1:])]

    good = np.array([b for b in bands if np.isfinite(b)])
    scatter = float(np.std(good)) if good.size >= 2 else np.nan
    return dict(y=float(y_all), bands=[float(b) for b in bands], scatter=scatter,
                consistent=bool(np.isfinite(y_all) and good.size >= 2 and scatter <= 1.0))


def pipeline_extraction_box(x1d_path):
    """(ystart, ystop) rows (0-based, inclusive) the pipeline actually
    extracted, read from the x1d exactly as the notebook does. None if it
    cannot be read."""
    try:
        from stdatamodels.jwst import datamodels
        with datamodels.open(x1d_path) as m:
            for spec in m.spec:
                ys, ye = spec.extraction_ystart, spec.extraction_ystop
                if ys is not None and ye is not None and np.isfinite(ys) and np.isfinite(ye) and ys > 0:
                    return float(ys) - 1.0, float(ye) - 1.0
    except Exception as exc:
        print(f"  (could not read the pipeline extraction box: {exc})")
    return None


# ===========================================================================
# Stitching, saving, plotting
# ===========================================================================

def stitch(pieces, verbose=True):
    """Join spectrum pieces (e.g. G140M + G235M + G395M, or NRS1 + NRS2) into
    one spectrum, in order of wavelength. Where two pieces overlap, the
    overlap is the inverse-variance-weighted mean of the two, evaluated on the
    redder piece's wavelength grid (the earlier piece is interpolated onto it,
    but never across a gap in its own coverage). Outside overlaps the data are
    untouched. The median flux ratio in each overlap is printed as a check on
    the relative flux scale of the two pieces.

    Returns wave, flux, err, unit."""
    if not pieces:
        raise ValueError("Nothing to stitch.")
    units = {p["unit"] for p in pieces}
    if len(units) > 1:
        detail = "; ".join(f"{p['label']} ({os.path.basename(str(p.get('source', '?')))}, "
                           f"slit={p.get('slit') or '?'}, SRCTYPE={p.get('srctype') or '?'}) = {p['unit']}"
                           for p in pieces)
        raise ValueError(f"Mixed flux units {sorted(units)}; cannot stitch. Pieces: {detail}. "
                         f"Jy = point-source extraction, MJy/sr = extended. A point-source (SN) spectrum should be "
                         f"Jy throughout: check the slit selection and the SRCTYPE of each Stage-3 x1d.")
    pieces = sorted(pieces, key=lambda p: p["wave"][0])
    w, f, e = pieces[0]["wave"], pieces[0]["flux"], pieces[0]["err"]
    for p in pieces[1:]:
        pw, pf, pe = p["wave"], p["flux"], p["err"]
        if pw[-1] <= w[-1]:
            if verbose:
                print(f"    {p['label']}: fully inside existing coverage -> skipped")
            continue
        lo, hi = pw[0], w[-1]
        if lo >= hi:                                   # no overlap: just append
            w, f, e = np.concatenate([w, pw]), np.concatenate([f, pf]), np.concatenate([e, pe])
            continue
        m = (pw >= lo) & (pw <= hi)                    # new points inside the overlap
        fi, ei = np.interp(pw[m], w, f), np.interp(pw[m], w, e)
        idx = np.clip(np.searchsorted(w, pw[m]), 1, w.size - 1)
        near = np.minimum(np.abs(w[idx] - pw[m]), np.abs(w[idx - 1] - pw[m]))
        has_old = near <= 2.0 * np.median(np.diff(w))  # is there really old data here?
        w_old = np.where(has_old, 1.0 / ei ** 2, 0.0)
        w_new = 1.0 / pe[m] ** 2
        fc = (w_old * fi + w_new * pf[m]) / (w_old + w_new)
        ec = 1.0 / np.sqrt(w_old + w_new)
        if verbose:
            ok = has_old & (np.abs(fi) > 0)
            if ok.any():
                print(f"    overlap {lo:.3f}-{hi:.3f} um: median flux ratio "
                      f"({p['label']} / previous) = {np.median(pf[m][ok] / fi[ok]):.3f}")
        keep_old, after = w < lo, pw > hi
        w = np.concatenate([w[keep_old], pw[m], pw[after]])
        f = np.concatenate([f[keep_old], fc, pf[after]])
        e = np.concatenate([e[keep_old], ec, pe[after]])
    return w, f, e, pieces[0]["unit"]


def save_spectrum(out_base, wave, flux, err, unit, meta=None):
    """Write <out_base>.ecsv and <out_base>.fits (WAVELENGTH, FLUX, FLUX_ERROR)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_base)), exist_ok=True)
    # Canonical upper-case columns (jspec.columns) -- this used to write
    # lower-case "wavelength"/"flux"/"flux_err", which disagreed with
    # every other pipeline and with the x1d tables themselves.
    from jspec.columns import spectrum_table
    meta = {k: v for k, v in (meta or {}).items() if v is not None}
    t = spectrum_table(wave, flux, err, unit=unit, meta=meta)
    t.write(out_base + ".ecsv", format="ascii.ecsv", overwrite=True)
    t.meta = {k: v for k, v in meta.items() if len(k) <= 8 and len(str(v)) < 60}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t.write(out_base + ".fits", overwrite=True)
    return out_base + ".ecsv", out_base + ".fits"


def _break_gaps(w, f, factor=5.0):
    """Insert a NaN wherever there is a wavelength gap, so plots don't draw a
    line across the NRS1/NRS2 gap."""
    if w.size < 3:
        return w, f
    dw = np.diff(w)
    idx = np.where(dw > factor * np.median(dw))[0]
    return np.insert(w, idx + 1, np.nan), np.insert(f, idx + 1, np.nan)


def _plot_spectra(ax, spectra, unit=""):
    allf, colors = [], {}
    for label, w, f in spectra:
        ww, ff = _break_gaps(w, f)
        line, = ax.plot(ww, ff, drawstyle="steps-mid", lw=0.8, color=colors.get(label),
                        label=None if label in colors else label)   # NRS1/NRS2 pieces share colour + legend entry
        colors.setdefault(label, line.get_color())
        allf.append(f[np.isfinite(f)])
    allf = np.concatenate(allf) if allf else np.array([])
    if allf.size:
        lo, hi = np.percentile(allf, [1, 99.5])
        pad = 0.1 * (hi - lo if hi > lo else 1.0)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel(f"Flux ({unit})")
    ax.legend(fontsize=8)


def plot_spectrum(out_png, spectra, unit, title=""):
    """spectra = [(label, wave, flux), ...]"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.5))
    _plot_spectra(ax, spectra, unit)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_qa(out_png, s2d, profile, box_pipe, box_new, spectra, unit, title="", est_row=None):
    """Three panels: the s2d image with the extraction box(es), the
    cross-slit profile, and the extracted spectrum(s). This is what you
    look at to decide whether the box is on the source."""
    import matplotlib.pyplot as plt
    sci, wave = s2d["sci"], s2d["wave"]
    ny, nx = sci.shape
    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1.0, 1.5], hspace=0.4)

    ax = fig.add_subplot(gs[0])
    fin = sci[np.isfinite(sci)]
    if fin.size:
        lo, hi = np.percentile(fin, [2, 99.5])
        hi = hi if hi > lo else lo + 1.0
        if wave is not None and np.isfinite(wave).any():
            extent = [np.nanmin(wave), np.nanmax(wave), -0.5, ny - 0.5]
            ax.set_xlabel("Wavelength (um)")
        else:
            extent = [-0.5, nx - 0.5, -0.5, ny - 0.5]
            ax.set_xlabel("Column")
        ax.imshow(sci, origin="lower", aspect="auto", extent=extent, cmap="viridis", vmin=lo, vmax=hi)
    ax.set_ylabel("Slit row")
    ax.set_title(title)

    axp = fig.add_subplot(gs[1])
    rows = np.arange(ny)
    axp.plot(rows, profile, "k-", lw=1)
    axp.axhline(0, color="0.6", lw=0.5)
    axp.set_xlabel("Slit row")
    axp.set_ylabel("Median flux")

    for box, color, ls, lab in ((box_pipe, "red", "-", "pipeline box"),
                                (box_new, "orange", "--", "re-extraction box")):
        if box is None:
            continue
        ax.axhline(box[0] - 0.5, color=color, ls=ls, lw=1.2)
        ax.axhline(box[1] + 0.5, color=color, ls=ls, lw=1.2)
        axp.axvspan(box[0] - 0.5, box[1] + 0.5, color=color, alpha=0.2, label=f"{lab}: rows {box[0]:.1f}-{box[1]:.1f}")
    if est_row is not None and np.isfinite(est_row):
        axp.axvline(est_row, color="k", ls=":", lw=1.2, label=f"profile peak: row {est_row:.1f}")
    if axp.get_legend_handles_labels()[0]:
        axp.legend(fontsize=8)

    ax1 = fig.add_subplot(gs[2])
    if spectra:
        _plot_spectra(ax1, spectra, unit)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Re-extraction with the pipeline's own Extract1dStep (notebook section 9)
# ===========================================================================

def shift_aperture(params, slit, center):
    """Move the `slit` aperture of an EXTRACT1D reference dict so its box is
    centred on row `center` (0-based, s2d pixels). extract_width is NOT
    changed -- there is no aperture correction for NIRSpec, so the width must
    stay at its reference value (6 px; 5 for S1600A1) to keep the flux
    calibration valid. The aperture is matched on its 'id' (or 'name').

    Returns (ystart, ystop, width, (old_ystart, old_ystop))."""
    aps = params.get("apertures")
    if not aps:
        raise KeyError("Reference file has no 'apertures' list.")
    want = str(slit).upper()
    matches = [a for a in aps if want in (str(a.get("id", "")).upper(), str(a.get("name", "")).upper())]
    if len(matches) != 1:
        ids = [a.get("id", a.get("name", "?")) for a in aps]
        raise KeyError(f"Could not uniquely match slit '{slit}' among the reference apertures {ids}; "
                       f"pass --slit explicitly.")
    ap = matches[0]
    width = ap.get("extract_width")
    if width is None:
        width = ap["ystop"] - ap["ystart"] + 1
    old = (ap["ystart"], ap["ystop"])
    ap["ystart"] = float(center) - (width - 1) / 2.0
    ap["ystop"] = float(center) + (width - 1) / 2.0
    return ap["ystart"], ap["ystop"], width, old


def _setup_crds():
    os.environ.setdefault("CRDS_PATH", os.path.join(os.path.expanduser("~"), "crds_cache"))
    os.environ.setdefault("CRDS_SERVER_URL", "https://jwst-crds.stsci.edu")


def reextract_at_row(s2d_path, center, out_dir, ref_path=None, slit=None):
    """Re-run Extract1dStep on the Stage-3 s2d with the extraction box
    centred on row `center`. Same call as the notebook: override_extract1d
    with an edited copy of the reference JSON, use_source_posn=False.

    Returns (x1d_mod_path, (ystart, ystop))."""
    _setup_crds()
    from jwst.extract_1d import Extract1dStep
    from stdatamodels.jwst import datamodels

    os.makedirs(out_dir, exist_ok=True)
    slit = slit or slit_name(s2d_path)

    if ref_path is None:
        try:
            with datamodels.open(s2d_path) as m:
                ref_path = Extract1dStep().get_reference_file(m, "extract1d")
        except Exception as exc:
            raise RuntimeError(
                f"Could not look up the EXTRACT1D reference file ({exc}). Pass it explicitly "
                f"(--extract1d-ref), e.g. $CRDS_PATH/references/jwst/nirspec/jwst_nirspec_extract1d_*.json")
    if not ref_path or str(ref_path).upper() == "N/A" or not os.path.exists(ref_path):
        raise RuntimeError(f"EXTRACT1D reference file not usable: {ref_path!r}")

    with open(ref_path) as fh:
        params = json.load(fh)
    ystart, ystop, width, old = shift_aperture(params, slit, center)
    mod_ref = os.path.join(out_dir, f"extract1d_{slit.lower()}_row{center:.2f}.json")
    with open(mod_ref, "w") as fh:
        json.dump(params, fh, indent=4)
    print(f"  reference file: {os.path.basename(ref_path)}")
    print(f"  aperture {slit}: ystart/ystop {old[0]}/{old[1]} (nominal) -> {ystart:.1f}/{ystop:.1f}; "
          f"extract_width stays {width}")

    stem = os.path.basename(s2d_path).rsplit("_s2d", 1)[0]
    Extract1dStep.call(s2d_path, save_results=True, output_dir=out_dir, output_use_model=True,
                       suffix="x1d_mod", use_source_posn=False, override_extract1d=mod_ref)

    out = os.path.join(out_dir, f"{stem}_x1d_mod.fits")
    if not os.path.exists(out):
        cands = sorted(glob.glob(os.path.join(out_dir, "*x1d_mod.fits")), key=os.path.getmtime)
        if not cands:
            raise RuntimeError(f"Extract1dStep wrote no *_x1d_mod.fits in {out_dir}")
        out = cands[-1]
    return out, (ystart, ystop)