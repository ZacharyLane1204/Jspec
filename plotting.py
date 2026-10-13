"""
jspec.plotting
==============

Publication-oriented diagnostic plots.

Everything here takes already-computed objects (a :class:`~jspec.spectrum.Spectrum`,
a list of :class:`~jspec.features.Feature`, a
:class:`~jspec.stitch.StitchReport`) and returns the matplotlib ``Axes`` it drew
on, so plots compose: draw the dust bands, then overplot the features, then add
the high-S/N regions, all on one axis.

Conventions
-----------

* **The x axis is rest-frame by default** whenever the spectrum carries a
  non-zero redshift, because every reference wavelength in this package is
  rest-frame and plotting observed wavelengths against rest-frame band
  markers is the easiest way to produce a confidently wrong figure. The axis
  label always states which frame is shown.
* **Masked points are not drawn.** They are also not silently skipped: a gap
  appears where they were, rather than a straight line joining across them,
  so a region with no usable data looks like a region with no usable data.
* **Nothing is renormalised behind your back.** If a plot shows normalised
  flux it is because you passed a normalised spectrum.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .spectrum import Spectrum

__all__ = [
    "plot_spectrum",
    "plot_dust_bands",
    "plot_features",
    "plot_high_snr",
    "plot_decomposition",
    "plot_stitch_diagnostics",
    "annotate_lines",
    "quicklook",
]

_KIND_DEFAULT_COLORS = {
    "molecular": "#d62728",
    "pah": "#ff7f0e",
    "silicate": "#8c564b",
    "crystalline": "#2ca02c",
    "ice": "#1f77b4",
    "carbon": "#7f7f7f",
    "other": "#e377c2",
}


def _as_spectrum(obj, redshift=0.0, **kw):
    """Accept a Spectrum, a Table, or a path to one.

    Every plotting entry point runs its first argument through this, so
    ``plot_spectrum("SN2024aecx_2025-06-01.ecsv")`` works as well as
    ``plot_spectrum(sp)``. Reading a file is the overwhelmingly common
    case at a prompt, and making the caller remember the reader import
    buys nothing.
    """
    import os

    from .spectrum import Spectrum as _S

    if isinstance(obj, _S):
        return obj
    if isinstance(obj, (str, os.PathLike)):
        from .spectrum import read_spectrum
        return read_spectrum(str(obj), redshift=redshift, **kw)

    from astropy.table import Table
    if isinstance(obj, Table):
        from .spectrum import _table_to_spectrum
        return _table_to_spectrum(obj, redshift)
    raise TypeError(
        f"expected a Spectrum, an astropy Table, or a path to a spectrum "
        f"file, got {type(obj).__name__}. If you meant to pass an Axes, note "
        f"that the spectrum comes first and the axes are the keyword `ax=`.")


def _as_features(obj):
    """Accept a list of Features, a Table, or a path to a saved one."""
    import os

    if obj is None:
        return []
    if isinstance(obj, (str, os.PathLike)):
        from .features import features_from_table
        return features_from_table(str(obj))

    from astropy.table import Table
    if isinstance(obj, Table):
        from .features import features_from_table
        return features_from_table(obj)
    return list(obj)


def _axis_wave(spectrum: Spectrum, rest):
    if rest is None:
        rest = bool(spectrum.redshift)
    w = spectrum.rest_wavelength if rest else spectrum.wavelength
    label = (r"Rest wavelength [$\mu$m]" if rest
             else r"Observed wavelength [$\mu$m]")
    return w, label, rest


def _broken(w, f, mask):
    """Insert NaNs at masked points so matplotlib breaks the line there
    instead of drawing a chord across a data gap."""
    f = np.where(mask, np.nan, f)
    return w, f


def plot_spectrum(spectrum: Spectrum, ax=None, rest=None, show_err=True,
                  rebin=1, color="k", lw=0.8, label=None, log_y=False,
                  err_alpha=0.25, **kwargs):
    """Draw a spectrum with a shaded 1-sigma band.

    Parameters
    ----------
    rebin : int
        Inverse-variance bin before plotting. Use this rather than smoothing:
        binning keeps the error bars honest, smoothing makes a noisy spectrum
        look better than it is.
    log_y : bool
        Log flux axis. Automatically falls back to a symmetric-log scale if
        the spectrum contains non-positive flux, which a plain log axis would
        silently drop.
    """
    import matplotlib.pyplot as plt
    spectrum = _as_spectrum(spectrum)

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.2))
    sp = spectrum.rebin(rebin) if rebin and rebin > 1 else spectrum
    w, xlabel, rest = _axis_wave(sp, rest)
    bad = ~sp.good
    ww, ff = _broken(w, sp.flux, bad)

    if show_err:
        lo = np.where(bad, np.nan, sp.flux - sp.flux_err)
        hi = np.where(bad, np.nan, sp.flux + sp.flux_err)
        ax.fill_between(w, lo, hi, color=color, alpha=err_alpha, lw=0, zorder=1)
    ax.plot(ww, ff, color=color, lw=lw, label=label, zorder=2, **kwargs)

    if log_y:
        if np.nanmin(sp.flux[sp.good]) <= 0:
            scale = np.nanpercentile(np.abs(sp.flux[sp.good]), 5)
            ax.set_yscale("symlog", linthresh=max(scale, 1e-30))
        else:
            ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Flux [{sp.flux_unit}]")
    return ax


def plot_dust_bands(spectrum: Spectrum = None, bands=None, ax=None, rest=None,
                    kinds=None, names=None, alpha=0.16, label_bands=True,
                    label_rotation=90, label_size=7, label_y=0.97,
                    stagger=True, min_width_frac=0.0, legend=True,
                    draw_peak=True, features=None, include_continuum=False,
                    max_width_frac=0.35, max_bands=18,
                    min_label_sep=0.035, **spectrum_kwargs):
    """Shade known dust and molecular bands over a spectrum.

    This is the "where are CO, SiO, AlO, PAH and silicate" figure. Each entry
    in the band table (see :func:`jspec.linedb.load_bands`) is drawn as an
    ``axvspan`` from ``lam_min_um`` to ``lam_max_um``, optionally with a
    dotted line at the band peak and a rotated label.

    Parameters
    ----------
    spectrum : Spectrum, optional
        Drawn underneath if given. If omitted, only the bands are drawn, so
        this can be overlaid on an axis you have already populated.
    bands : DataFrame, optional
        From :func:`jspec.linedb.load_bands`; loaded on demand.
    kinds : sequence of str, optional
        Restrict to band families, e.g. ``['pah', 'silicate']``. Drawing all
        families at once over the full MIRI range produces an unreadable
        figure -- there are overlapping PAH, crystalline silicate and ice
        bands at nearly every wavelength -- so filtering is usually what you
        want.
    names : sequence of str, optional
        Restrict to specific band names, e.g. ``['CO fundamental',
        'CO first overtone', 'Silicate 9.7']``.
    stagger : bool
        Alternate label heights so neighbouring band labels do not overprint.
    min_width_frac : float
        Skip bands narrower than this fraction of the plotted range. Useful
        when zoomed out, where a 0.05 um ice band is a line not a band.
    draw_peak : bool
        Dotted vertical line at ``lam_peak_um``.
    features : sequence of Feature, optional
        Draw only bands that actually bracket a detected feature. This is
        usually the figure you want: it answers "which bands do I have
        evidence for" instead of "which bands exist", and it is the single
        most effective way to stop the overlay swamping the data.
    include_continuum : bool
        Include the featureless entries -- amorphous carbon, graphite, HAC
        -- which span the whole mid-infrared. Excluded by default because
        shading them covers every other band on the axis. They are
        recognised by having no peak wavelength.
    max_width_frac : float
        Skip bands wider than this fraction of the plotted range, for the
        same reason.
    max_bands : int or None
        Keep at most this many bands, preferring the narrowest (most
        specific) ones. Set to ``None`` for no cap. The band table has ~80
        entries and the mid-infrared has overlapping PAH, silicate and ice
        bands nearly everywhere, so drawing them all is unreadable.
    min_label_sep : float
        Minimum separation between two labels on the same stagger row, as a
        fraction of the x-range. Labels that would overprint are dropped
        rather than written on top of each other; the band is still shaded.

    Notes
    -----
    The shading is drawn in the **rest frame** when the spectrum has a
    redshift, because the band table is rest-frame. If you pass
    ``rest=False`` with a non-zero redshift the bands are redshifted to the
    observed frame for you -- they are never drawn at the wrong wavelength
    silently.
    """
    import matplotlib.pyplot as plt
    spectrum = _as_spectrum(spectrum) if spectrum is not None else None

    from .linedb import load_bands

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4.6))
    if bands is None:
        bands = load_bands()

    z = spectrum.redshift if spectrum is not None else 0.0
    if spectrum is not None:
        plot_spectrum(spectrum, ax=ax, rest=rest, **spectrum_kwargs)
        _, _, rest = _axis_wave(spectrum, rest)
    elif rest is None:
        rest = True

    df = bands
    if kinds:
        df = df[df["kind"].isin(list(kinds))]
    if names:
        df = df[df["name"].isin(list(names))]
    if not include_continuum and not names:
        # Featureless entries have no peak; they bracket the entire MIR and
        # would shade every other band out of visibility.
        df = df[pd.to_numeric(df["lam_peak_um"], errors="coerce").notna()]

    x0, x1 = ax.get_xlim()
    span = x1 - x0
    # Convert band wavelengths into whichever frame the axis is in.
    shift = 1.0 if rest else (1.0 + z)

    if features is not None:
        # The band table is rest-frame, so compare rest-frame centres
        # regardless of which frame the axis happens to be drawn in.
        ref = [c for c in (getattr(f, "rest_center", None) for f in features)
               if c is not None and np.isfinite(c)]
        keep = df.apply(
            lambda r: any(r["lam_min_um"] <= c <= r["lam_max_um"] for c in ref),
            axis=1)
        df = df[keep] if len(df) else df

    width = (pd.to_numeric(df["lam_max_um"], errors="coerce")
             - pd.to_numeric(df["lam_min_um"], errors="coerce")) * shift
    if max_width_frac is not None and span > 0 and len(df):
        df = df[(width / span) <= max_width_frac]
        width = width[df.index]
    if max_bands is not None and len(df) > max_bands:
        # Narrowest first: a specific band identifies something, a wide
        # complex mostly just colours the background.
        df = df.reindex(width.sort_values().index[:max_bands]).sort_values(
            "lam_min_um")

    seen_kinds = {}
    label_rows = {}
    k = 0
    for _, row in df.iterrows():
        lo, hi = row["lam_min_um"] * shift, row["lam_max_um"] * shift
        if hi < x0 or lo > x1:
            continue
        if span > 0 and (hi - lo) / span < min_width_frac:
            continue
        kind = str(row.get("kind", "other"))
        color = row.get("color")
        if not isinstance(color, str) or not color.startswith("#"):
            color = _KIND_DEFAULT_COLORS.get(kind, "#999999")
        ax.axvspan(max(lo, x0), min(hi, x1), color=color, alpha=alpha, lw=0, zorder=0)
        if draw_peak and np.isfinite(row.get("lam_peak_um", np.nan)):
            p = row["lam_peak_um"] * shift
            if x0 <= p <= x1:
                ax.axvline(p, color=color, ls=":", lw=0.8, alpha=0.75, zorder=0)
        if label_bands:
            xc = 0.5 * (max(lo, x0) + min(hi, x1))
            nrow = 3 if stagger else 1
            placed = False
            for attempt in range(nrow):
                r = (k + attempt) % nrow
                last = label_rows.get(r)
                if last is None or span <= 0 or abs(xc - last) / span >= min_label_sep:
                    label_rows[r] = xc
                    yy = label_y - 0.06 * r
                    ax.text(xc, yy, str(row["name"]),
                            rotation=label_rotation, ha="center", va="top",
                            fontsize=label_size, color=color, alpha=0.95,
                            transform=ax.get_xaxis_transform(), clip_on=True)
                    placed = True
                    break
            if not placed:
                # Every row is occupied nearby; shade without a label rather
                # than overprint.
                pass
        seen_kinds.setdefault(kind, color)
        k += 1

    if legend and seen_kinds:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=c, alpha=0.45, label=kd)
                   for kd, c in sorted(seen_kinds.items())]
        ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=2,
                  framealpha=0.85)
    ax.set_xlim(x0, x1)
    return ax


def _add_headroom(ax, top=0.0, bottom=0.0):
    """Grow the y-range by a fraction of itself, without shrinking it.

    Called more than once (bands, then features) this must not compound,
    so it tracks what it has already added.
    """
    lo, hi = ax.get_ylim()
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return
    have = getattr(ax, "_jspec_headroom", (0.0, 0.0))
    base = span / (1.0 + have[0] + have[1])
    want = (max(top, have[0]), max(bottom, have[1]))
    if want == have:
        return
    ax.set_ylim(lo - base * (want[1] - have[1]),
                hi + base * (want[0] - have[0]))
    ax._jspec_headroom = want


def plot_features(spectrum: Spectrum, features, ax=None, rest=None,
                  label=True, label_size=7, min_snr=0.0, components=None,
                  signs=None, mark_joins=True, max_labels=60,
                  label_field="best_id", **spectrum_kwargs):
    """Mark and label detected features on a spectrum.

    Emission features are marked above the spectrum, absorption below, so the
    two never collide. Features flagged ``on_join`` are drawn in red with a
    dashed marker when ``mark_joins`` is set -- those are the ones most likely
    to be stitching artefacts rather than astrophysics, and the figure should
    say so rather than leaving it to the caller to cross-reference a table.

    ``max_labels`` caps how many labels are drawn (highest integrated S/N
    first); markers are still drawn for all of them. A 400-feature spectrum
    cannot be usefully labelled in full, and silently drawing 400 overlapping
    labels is worse than drawing the 60 that matter.
    """
    import matplotlib.pyplot as plt
    spectrum = _as_spectrum(spectrum)
    features = _as_features(features)

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4.6))
        plot_spectrum(spectrum, ax=ax, rest=rest, **spectrum_kwargs)
    _, _, rest = _axis_wave(spectrum, rest)

    sel = [f for f in features if f.integrated_snr >= min_snr]
    if components:
        sel = [f for f in sel if f.component in components]
    if signs:
        sel = [f for f in sel if f.sign in signs]

    labelled = sorted(sel, key=lambda f: -f.integrated_snr)[:max_labels]
    label_set = {id(f) for f in labelled}

    # Rotated labels sit outside the marker span and so outside the axes
    # box. Clipping them to it truncates every one of them, which is how
    # you end up with a row of two-character stubs; let them overflow and
    # save with bbox_inches="tight" instead. The extra headroom keeps the
    # markers off the spectrum itself.
    if sel and label:
        has_abs = any(f.sign != "emission" for f in labelled)
        # Emission labels overflow the top of the axes, where there is
        # nothing but whitespace; absorption labels cannot do the same at
        # the bottom without landing on the tick labels and the axis
        # title, so they are drawn upward into reserved space inside the
        # axes instead.
        _add_headroom(ax, top=0.30, bottom=0.42 if has_abs else 0.12)

    for f in sel:
        x = f.rest_center if rest else f.center
        artefact = mark_joins and f.on_join
        color = "crimson" if artefact else ("tab:blue" if f.sign == "emission" else "tab:purple")
        ls = "--" if artefact else "-"
        y0, y1 = (0.88, 0.97) if f.sign == "emission" else (0.03, 0.12)
        ax.plot([x, x], [y0, y1], color=color, lw=1.0, ls=ls, alpha=0.9,
                transform=ax.get_xaxis_transform(), clip_on=True, zorder=4)
        if label and id(f) in label_set:
            if label_field == "best_id" and f.best_match is not None:
                txt = f.best_match.label
            elif label_field == "wavelength":
                txt = f"{f.rest_center:.3f}"
            else:
                txt = getattr(f, label_field, "") or f"{f.rest_center:.3f}"
            if artefact:
                txt += " (join)"
            va, yy = (("bottom", y1 + 0.005) if f.sign == "emission"
                      else ("bottom", y1 + 0.005))
            ax.text(x, yy, txt, rotation=90, ha="center", va=va,
                    fontsize=label_size, color=color,
                    transform=ax.get_xaxis_transform(),
                    clip_on=f.sign != "emission", zorder=5)
    return ax


def plot_high_snr(spectrum: Spectrum, regions, ax=None, rest=None,
                  color="tab:green", alpha=0.18, label=True, label_size=7,
                  features=None, **spectrum_kwargs):
    """Shade and label high-S/N regions.

    Each region is annotated with its peak S/N and, where features were
    attached to it, the best identification of its strongest feature -- so the
    figure answers "which parts of this spectrum are worth fitting, and what
    is in them" in one pass.
    """
    import matplotlib.pyplot as plt
    spectrum = _as_spectrum(spectrum)
    features = _as_features(features) if features is not None else None

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4.6))
        plot_spectrum(spectrum, ax=ax, rest=rest, **spectrum_kwargs)
    _, _, rest = _axis_wave(spectrum, rest)

    for r in regions:
        lo = r.rest_lo if rest else r.lo
        hi = r.rest_hi if rest else r.hi
        ax.axvspan(lo, hi, color=color, alpha=alpha, lw=0, zorder=0)
        if label:
            txt = f"S/N {r.peak_snr:.0f}"
            feats = r.features or ([f for f in features
                                    if lo <= (f.rest_center if rest else f.center) <= hi]
                                   if features else [])
            if feats:
                best = max(feats, key=lambda f: f.integrated_snr)
                if best.best_match is not None:
                    txt = f"{best.best_match.label}\nS/N {r.peak_snr:.0f}"
            ax.text(0.5 * (lo + hi), 0.5, txt, rotation=90, ha="center",
                    va="center", fontsize=label_size, color="darkgreen",
                    transform=ax.get_xaxis_transform(), clip_on=True, zorder=3)
    return ax


def plot_decomposition(dec, axes=None, rest=None, rebin=1):
    """Four-panel view of a :class:`jspec.continuum.Decomposition`:
    data with continuum, broad component, narrow component, residual.

    This is the figure to look at when a detection list looks wrong. Almost
    every surprising detection traces back to the continuum being in the wrong
    place, and that is immediately obvious here and almost impossible to see
    in the detection table.
    """
    import matplotlib.pyplot as plt

    sp = dec.spectrum
    w, xlabel, rest = _axis_wave(sp, rest)
    if axes is None:
        _, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    bad = ~sp.good

    axes[0].plot(*_broken(w, sp.flux, bad), color="k", lw=0.7, label="data")
    axes[0].plot(w, dec.continuum, color="tab:orange", lw=1.4, label="continuum")
    axes[0].set_ylabel(f"Flux [{sp.flux_unit}]")
    axes[0].legend(fontsize=8)

    eff = dec.noise
    axes[1].axhline(0, color="0.6", lw=0.6)
    axes[1].plot(*_broken(w, dec.broad, bad), color="tab:red", lw=0.8)
    axes[1].set_ylabel(f"broad\n(<{dec.broad_scale_kms:.0f} km/s)")

    axes[2].axhline(0, color="0.6", lw=0.6)
    axes[2].plot(*_broken(w, dec.narrow, bad), color="tab:blue", lw=0.7)
    axes[2].set_ylabel(f"narrow\n(<{dec.narrow_scale_kms:.0f} km/s)")

    axes[3].axhline(0, color="0.6", lw=0.6)
    axes[3].plot(*_broken(w, dec.residual, bad), color="0.4", lw=0.5)
    axes[3].plot(w, eff, color="tab:green", lw=0.9, label="noise estimate")
    axes[3].plot(w, -eff, color="tab:green", lw=0.9)
    axes[3].set_ylabel("residual")
    axes[3].set_xlabel(xlabel)
    axes[3].legend(fontsize=8)
    return axes


def plot_stitch_diagnostics(report, axes=None, rest=False, pad_um=0.4):
    """Two-panel stitching diagnostic: the full stitched spectrum with every
    segment join marked, and a zoom strip around each join.

    The top panel shows where the joins are; the zoom strips show whether each
    one is actually continuous. A join that looks fine at full scale can still
    carry a several-percent step, which is only visible zoomed in -- and a
    several-percent step at a join is exactly the artefact that turns into a
    spurious "broad feature" later.
    """
    import matplotlib.pyplot as plt

    sp = report.spectrum
    joins = [j for j in report.joins if np.isfinite(j.centre)]
    n = len(joins)
    ncols = max(min(n, 6), 1)
    if axes is None:
        fig = plt.figure(figsize=(13, 7))
        gs = fig.add_gridspec(2, ncols, height_ratios=[2, 1.4], hspace=0.35)
        ax_top = fig.add_subplot(gs[0, :])
        ax_zoom = [fig.add_subplot(gs[1, i]) for i in range(ncols)]
    else:
        ax_top, ax_zoom = axes

    plot_spectrum(sp, ax=ax_top, rest=rest, log_y=True)
    w, _, rest_used = _axis_wave(sp, rest)
    for j in joins:
        c = j.centre / (1 + sp.redshift) if rest_used else j.centre
        bad = j.residual and j.residual["verdict"] == "residual calibration step"
        ax_top.axvline(c, color="crimson" if bad else "tab:green",
                       ls="--" if bad else ":", lw=1.0, alpha=0.85)
        ax_top.text(c, 1.005, f"{j.blue}|{j.red}", rotation=90, fontsize=6,
                    ha="center", va="bottom",
                    transform=ax_top.get_xaxis_transform(),
                    color="crimson" if bad else "tab:green")
    ax_top.set_title(f"stitched: anchor={report.anchor}", fontsize=9)

    for i, ax in enumerate(ax_zoom):
        if i >= n:
            ax.axis("off")
            continue
        j = joins[i]
        c = j.centre
        sub = sp.select(c - pad_um, c + pad_um)
        if len(sub) < 3:
            ax.axis("off")
            continue
        plot_spectrum(sub, ax=ax, rest=False, show_err=True, lw=0.7)
        ax.axvline(c, color="crimson", ls="--", lw=1.0)
        res = j.residual
        ttl = f"{j.blue}|{j.red}"
        if res:
            ttl += f"\n{res['step_pct']:+.1f}% ({res['sigma']:.1f}$\\sigma$)"
        ax.set_title(ttl, fontsize=7)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=6)
    return ax_top, ax_zoom


def annotate_lines(ax, lines, rest=True, redshift=0.0, y=0.5, color="0.3",
                   fontsize=7, species=None, categories=None, lw=0.7):
    """Draw and label reference lines from the line list on an existing axis.

    Unlike :func:`plot_features`, this draws lines at their *catalogue*
    positions regardless of whether anything was detected there -- which is
    the right way to show that an expected line is *absent*.
    """
    df = lines
    if species:
        df = df[df["species"].isin(list(species))]
    if categories:
        df = df[df["category"].isin(list(categories))]
    x0, x1 = ax.get_xlim()
    shift = 1.0 if rest else (1.0 + redshift)
    for _, row in df.iterrows():
        x = float(row["wavelength_um"]) * shift
        if not (x0 <= x <= x1):
            continue
        ax.axvline(x, color=color, lw=lw, alpha=0.6, zorder=1)
        ax.text(x, y, str(row.get("label", row["species"])), rotation=90,
                ha="right", va="center", fontsize=fontsize, color=color,
                transform=ax.get_xaxis_transform(), clip_on=True, zorder=3)
    return ax


def quicklook(spectrum, redshift=None, features=None, regions=None,
              bands=True, kinds=None, names=None, snr=5.0, region_snr=15.0,
              label_snr=0.0, max_labels=40, rest=None, log_y=False, rebin=1,
              ax=None, figsize=(13, 5.2), title=None, output=None, dpi=160,
              **kwargs):
    """Draw a labelled spectrum in one call -- the batteries-included plot.

    Everything is optional except the spectrum, and the spectrum may be a
    path:

    >>> quicklook("SN2024aecx/SN2024aecx_2025-06-01.ecsv")   # doctest: +SKIP

    That shades the known dust and molecular bands, detects features,
    identifies them, marks the high-S/N regions, and returns ``(fig, ax)``
    so you can keep working on the figure.

    The individual ``plot_*`` functions are still there when you want to
    build a figure up piece by piece, but they need a Spectrum, a list of
    Features and a list of Regions, and producing those means knowing
    about :mod:`jspec.continuum` and :mod:`jspec.features`. This does
    that for you.

    Parameters
    ----------
    spectrum : Spectrum, Table or path
    redshift : float, optional
        Overrides whatever the file stores. A stitched spectrum already
        carries its redshift, so this is usually unnecessary.
    features, regions : optional
        Pre-computed, or a path to a table saved by ``jspec detect``. Pass
        ``False`` to skip that layer, or leave as ``None`` to compute it.
    bands : bool
        Shade known dust/molecular bands.
    snr, region_snr : float
        Detection thresholds, used only when features/regions are computed.
    label_snr, max_labels : float, int
        Which features are marked and labelled. By default only the
        ``max_labels`` strongest are drawn: a real spectrum yields
        several hundred detections, and marking all of them produces a
        picket fence that hides the spectrum underneath. Set
        ``label_snr`` explicitly to mark everything above a threshold,
        or ``max_labels=None`` for no cap.
    output : path, optional
        Save the figure here as well as returning it.
    """
    import matplotlib.pyplot as plt

    if isinstance(spectrum, (str, os.PathLike)) and title is None:
        title = os.path.basename(str(spectrum))
    sp = _as_spectrum(spectrum, redshift=redshift or 0.0)
    if redshift is not None and redshift != sp.redshift:
        sp = sp.with_redshift(redshift)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if bands:
        plot_dust_bands(sp, ax=ax, kinds=kinds, names=names, rest=rest,
                        log_y=log_y, rebin=rebin, **kwargs)
    else:
        plot_spectrum(sp, ax=ax, rest=rest, log_y=log_y, rebin=rebin, **kwargs)

    # Only decompose if something actually needs it -- it is the expensive
    # step, and a bands-only plot does not.
    need = (features is not False or regions is not False)
    feats = _as_features(features) if features not in (None, False) else None
    regs = regions if regions not in (None, False) else None
    if isinstance(regs, (str, os.PathLike)):
        raise TypeError("regions cannot be read from a file; leave it as "
                        "None to compute them, or pass a list of Regions")

    if need and (feats is None or regs is None):
        from .continuum import decompose
        from .features import find_features, high_snr_regions, identify
        dec = decompose(sp)
        if feats is None and features is not False:
            feats = identify(find_features(sp, decomposition=dec, snr_min=snr))
        if regs is None and regions is not False:
            regs = high_snr_regions(sp, decomposition=dec,
                                    snr_min=region_snr, features=feats or None)

    if regs:
        plot_high_snr(sp, regs, ax=ax, rest=rest, label=False)
    if feats:
        # Markers and labels should agree. plot_features caps labels but
        # marks everything above min_snr, so derive the threshold from
        # max_labels and the N-th strongest feature -- otherwise a few
        # dozen labels sit on top of several hundred tick marks.
        thresh = label_snr
        if not label_snr and max_labels and len(feats) > max_labels:
            ranked = sorted((f.integrated_snr for f in feats), reverse=True)
            thresh = ranked[max_labels - 1]
            print(f"marking the {max_labels} strongest of {len(feats)} "
                  f"features (S/N > {thresh:.0f}); raise max_labels or set "
                  f"label_snr to change that")
        plot_features(sp, feats, ax=ax, rest=rest, min_snr=thresh,
                      max_labels=max_labels or len(feats))

    if title:
        # Feature labels overflow the top of the axes, so a title at the
        # default pad lands on top of them.
        used = getattr(ax, "_jspec_headroom", (0.0, 0.0))[0]
        pad = None
        if used:
            ax_pts = ax.get_position().height * fig.get_size_inches()[1] * 72
            pad = 6.0 + used * ax_pts
        ax.set_title(title, pad=pad)
    if output:
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"wrote {output}")
    return fig, ax
