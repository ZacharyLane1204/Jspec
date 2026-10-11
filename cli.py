"""
jspec.cli
=========

Command-line front end for the analysis layer::

    python -m jspec stitch   seg1.ecsv seg2.ecsv ... -o combined.ecsv
    python -m jspec detect   combined.ecsv -z 0.0077 -o features.ecsv
    python -m jspec plot     combined.ecsv -z 0.0077 -o figure.pdf
    python -m jspec fit      combined.ecsv -z 0.0077 --line 12.81355
    python -m jspec lines    --range 12.5 13.0
    python -m jspec info     combined.ecsv

Every subcommand that produces numbers also prints the provenance needed to
reproduce it, and ``stitch`` always prints its diagnostic report -- the point
of the report is that it is seen, not that it is available.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _p_common(p):
    p.add_argument("-z", "--redshift", type=float, default=0.0,
                   help="source redshift (required for any rest-frame work)")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--keep-red-end", action="store_true",
                   help="do not mask the MIRI MRS ch4-long red end "
                        "(masked beyond 27.5 um by default: negative fluxes "
                        "and S/N collapse set in there)")
    p.add_argument("--red-end", type=float, default=None, metavar="UM",
                   help="observed wavelength beyond which to mask MRS data "
                        "(default 27.5)")
    return p


def _red_end(args):
    """Translate the red-end CLI flags into read_spectrum's argument."""
    if getattr(args, "keep_red_end", False):
        return False
    return getattr(args, "red_end", None) or True


# ---------------------------------------------------------------------------
def cmd_stitch(args):
    import fnmatch
    from .epochs import group_segments_by_epoch
    from .spectrum import read_segments
    from .stitch import stitch

    def _drop_excluded(segs, label):
        """Remove segments the user asked to leave out.

        The usual reason is that the quality check flagged one as
        containing no detected source. Matching is by glob against the
        segment name, so a partial name is enough and the same pattern
        can cover every epoch.
        """
        if not args.exclude:
            return segs
        keep, dropped = [], []
        for s in segs:
            if any(fnmatch.fnmatch(s.name, pat) or pat in s.name
                   for pat in args.exclude):
                dropped.append(s.name)
            else:
                keep.append(s)
        if dropped and not args.quiet:
            where = f" in epoch {label}" if label else ""
            print(f"excluding {len(dropped)} segment(s){where}: "
                  + ", ".join(dropped))
        return keep

    if args.no_epochs:
        groups = [None]
        if not args.quiet:
            print("WARNING: --no-epochs given; every input will be stitched "
                  "into ONE spectrum regardless of observation date. If these "
                  "files span multiple visits, real variability will be "
                  "absorbed into the scale factors.")
    else:
        groups = group_segments_by_epoch(
            args.inputs if len(args.inputs) > 1 else args.inputs[0],
            max_gap_days=args.max_gap_days, redshift=args.redshift,
            mask_red_end=_red_end(args), verbose=not args.quiet)
        if not groups:
            print("no usable epochs found")
            return 1

    failures = 0
    for g in groups:
        if g is None:
            segs = read_segments(args.inputs, redshift=args.redshift,
                                 mask_red_end=_red_end(args))
            label = None
        else:
            segs = g.segments
            label = g.label
            print(f"\n{'=' * 70}\nepoch {label}  ({len(segs)} segments)\n"
                  f"{'=' * 70}")
        segs = _drop_excluded(segs, label)
        if len(segs) < 1:
            if not args.quiet:
                print("nothing left to stitch after exclusions")
            continue

        report = stitch(segs, anchor=args.anchor, edge_trim=args.trim,
                        taper=not args.no_blend, clip_sigma=args.robust_clip,
                        max_scale_factor=args.max_scale,
                        max_gap_days=args.max_gap_days,
                        check_epoch=not args.no_epochs,
                        drop_blank=not args.keep_flagged)
        print()
        print(report.summary())

        if args.output:
            out = _epoch_path(args.output, label)
            report.spectrum.write(out, overwrite=True)
            if not args.quiet:
                print(f"\nwrote {out}  ({len(report.spectrum)} points, "
                      f"{report.spectrum.wavelength.min():.3f}-"
                      f"{report.spectrum.wavelength.max():.3f} um)")
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from .plotting import plot_stitch_diagnostics
            plot_stitch_diagnostics(report)
            pp = _epoch_path(args.plot, label)
            plt.savefig(pp, dpi=130, bbox_inches="tight")
            plt.close("all")
            print(f"wrote {pp}")

        failures += sum(
            1 for j in report.joins
            if j.residual and j.residual["verdict"] == "residual calibration step")
        # A segment with no source in it is a harder failure than a
        # residual step: the output is not merely mis-scaled, part of it
        # is not the target at all.
        failures += sum(1 for q in report.quality.values() if q.suspect)

    return 1 if (failures and args.strict) else 0


def _epoch_path(template, label):
    """Insert an epoch label into an output path.

    ``out.ecsv`` + ``2025-06-01`` -> ``out_2025-06-01.ecsv``. A path with no
    extension is treated as a prefix. One file per epoch is the whole point:
    writing every epoch to the same name would leave only the last one.
    """
    if label is None:
        return template
    root, ext = os.path.splitext(template)
    return f"{root}_{label}{ext or '.ecsv'}"


# ---------------------------------------------------------------------------
def cmd_detect(args):
    from .continuum import decompose
    from .features import features_table, find_features, high_snr_regions
    from .spectrum import read_spectrum

    sp = read_spectrum(args.input, redshift=args.redshift,
                       mask_red_end=_red_end(args))
    if args.range:
        sp = sp.select(args.range[0], args.range[1], rest=bool(args.redshift))
    dec = decompose(sp, broad_scale_kms=args.broad_scale,
                    narrow_scale_kms=args.narrow_scale)
    feats = find_features(sp, decomposition=dec, snr_min=args.snr,
                          max_join_fraction=args.max_join_fraction)
    if args.identify:
        from .features import identify
        identify(feats, tol_res_elements=args.tol)

    tab = features_table(feats)
    print(f"{len(feats)} features at S/N > {args.snr:g} "
          f"({sum(f.sign == 'emission' for f in feats)} emission, "
          f"{sum(f.sign == 'absorption' for f in feats)} absorption; "
          f"{sum(f.on_join for f in feats)} overlap a segment join)")
    if not args.quiet:
        cols = [c for c in ("rest_center_um", "sign", "component",
                            "fwhm_kms", "width_class", "integrated_snr",
                            "on_join", "best_id", "best_dv_kms")
                if c in tab.colnames]
        sub = tab[cols]
        sub.sort("integrated_snr", reverse=True)
        sub[: args.top].pprint(max_width=-1)

    regs = high_snr_regions(sp, decomposition=dec, snr_min=args.region_snr,
                            features=feats)
    print(f"\n{len(regs)} high-S/N regions (peak S/N > {args.region_snr:g}):")
    for r in regs[: args.top]:
        print(f"  {r}")

    if args.output:
        tab.write(args.output, overwrite=True)
        print(f"\nwrote {args.output}")
    return 0


# ---------------------------------------------------------------------------
def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .continuum import decompose
    from .features import find_features, high_snr_regions, identify
    from .plotting import (plot_dust_bands, plot_features, plot_high_snr,
                           plot_spectrum)
    from .spectrum import read_spectrum

    sp = read_spectrum(args.input, redshift=args.redshift,
                       mask_red_end=_red_end(args))
    if args.range:
        sp = sp.select(args.range[0], args.range[1], rest=bool(args.redshift))

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    if args.bands:
        plot_dust_bands(sp, ax=ax, kinds=args.kinds or None,
                        names=args.names or None, log_y=args.log,
                        rebin=args.rebin, label_bands=not args.no_band_labels)
    else:
        plot_spectrum(sp, ax=ax, log_y=args.log, rebin=args.rebin)

    if args.features or args.regions:
        dec = decompose(sp)
        feats = identify(find_features(sp, decomposition=dec, snr_min=args.snr))
        if args.features:
            plot_features(sp, feats, ax=ax, min_snr=args.label_snr,
                          max_labels=args.max_labels)
        if args.regions:
            regs = high_snr_regions(sp, decomposition=dec,
                                    snr_min=args.region_snr, features=feats)
            plot_high_snr(sp, regs, ax=ax, label=False)
    if args.title:
        ax.set_title(args.title)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.output}")
    return 0


# ---------------------------------------------------------------------------
def cmd_fit(args):
    from .fitting import fit_and_select, fit_feature
    from .spectrum import read_spectrum

    sp = read_spectrum(args.input, redshift=args.redshift,
                       mask_red_end=_red_end(args))
    kw = dict(window_kms=args.window, continuum_order=args.continuum_order,
              allow_absorption=args.absorption, nlive=args.nlive,
              sigma_max_kms=args.sigma_max)
    if args.select:
        print(f"model grid at {args.line:.5f} um (rest):")
        best, table, _ = fit_and_select(
            sp, args.line, max_components=args.max_components,
            profiles=tuple(args.profiles), delta_lnz=args.delta_lnz,
            verbose=True, **kw)
        print()
        print(f"preferred: {best.profile} x{best.n_components} "
              f"(requires delta_lnZ > {args.delta_lnz:g} to add complexity)")
        print()
        print(best.summary())
        r = best
    else:
        r = fit_feature(sp, args.line, n_components=args.n_components,
                        profile_name=args.profile,
                        print_progress=not args.quiet, **kw)
        print(r.summary())

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        r.plot()
        plt.savefig(args.plot, dpi=130, bbox_inches="tight")
        print(f"\nwrote {args.plot}")
    if args.output:
        np.savez(args.output, samples=r.samples,
                 param_names=np.array(r.param_names), logz=r.logz,
                 logz_err=r.logz_err, velocity=r.velocity, flux=r.flux,
                 flux_err=r.flux_err, model=r.model)
        print(f"wrote {args.output}")
    return 0


# ---------------------------------------------------------------------------
def cmd_lines(args):
    from .linedb import load_bands, load_lines

    if args.bands:
        df = load_bands()
        if args.range:
            df = df[(df["lam_max_um"] >= args.range[0])
                    & (df["lam_min_um"] <= args.range[1])]
        if args.kind:
            df = df[df["kind"].isin(args.kind)]
        cols = ["name", "species", "lam_peak_um", "lam_min_um", "lam_max_um", "kind"]
    else:
        df = load_lines()
        if args.range:
            df = df[(df["wavelength_um"] >= args.range[0])
                    & (df["wavelength_um"] <= args.range[1])]
        if args.species:
            df = df[df["species"].isin(args.species)]
        if args.kind:
            df = df[df["category"].isin(args.kind)]
        cols = ["species", "wavelength_um", "label", "kind", "category",
                "expected_width"]
    cols = [c for c in cols if c in df.columns]
    if len(df) == 0:
        print("no entries match")
        return 0
    import pandas as pd
    with pd.option_context("display.max_rows", 500, "display.width", 200):
        print(df[cols].to_string(index=False))
    print(f"\n{len(df)} entries")
    return 0


# ---------------------------------------------------------------------------
def cmd_epochs(args):
    """Dry run: report the epoch grouping and stop. Always check this before
    a long reduction -- it is far cheaper to find a mis-grouped visit here
    than after Stage 3."""
    from .epochs import group_paths_by_epoch

    groups = group_paths_by_epoch(
        args.inputs if len(args.inputs) > 1 else args.inputs[0],
        max_gap_days=args.max_gap_days)
    n = sum(len(g.paths) for g in groups)
    print(f"{n} file(s) -> {len(groups)} epoch(s) "
          f"(> {args.max_gap_days:g} day gap starts a new epoch)\n")
    for g in groups:
        print(g.summary())
        print()
    if len(groups) > 1:
        print("These will be processed SEPARATELY. Nothing is combined "
              "across epochs unless you pass --no-epochs.")
    return 0


def cmd_info(args):
    from .spectrum import read_spectrum

    sp = read_spectrum(args.input, redshift=args.redshift,
                       mask_red_end=_red_end(args))
    g = sp.good
    print(f"{args.input}")
    print(f"  points      : {len(sp)}  ({int(g.sum())} good, "
          f"{int((~g).sum())} masked)")
    print(f"  wavelength  : {sp.wavelength.min():.4f} - "
          f"{sp.wavelength.max():.4f} um (observed, vacuum)")
    if sp.redshift:
        print(f"  rest-frame  : {sp.rest_wavelength.min():.4f} - "
              f"{sp.rest_wavelength.max():.4f} um  (z = {sp.redshift})")
    print(f"  flux unit   : {sp.flux_unit}")
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = sp.flux[g] / sp.flux_err[g]
    print(f"  median S/N  : {np.nanmedian(snr):.1f}")
    print(f"  negative    : {int(np.sum(sp.flux[g] < 0))} points")
    if sp.segment is not None:
        import collections
        c = collections.Counter(sp.segment[g])
        print("  segments    :")
        for k, n in sorted(c.items()):
            m = sp.segment == k
            print(f"    {k:<22} {n:6d} pts  "
                  f"{sp.wavelength[m].min():7.3f} - {sp.wavelength[m].max():7.3f} um")
    for k, v in (sp.meta or {}).items():
        if not isinstance(v, (dict, list, np.ndarray)):
            print(f"  {k:<12}: {v}")
    return 0


# ---------------------------------------------------------------------------
def cmd_verify(args):
    """Check downloaded FITS files for truncation before a long run.

    An interrupted download leaves a file whose primary header still reads
    cleanly, so it passes every casual check and then fails hours later
    inside the JWST pipeline as a missing ASDF extension. This finds it in
    seconds instead.
    """
    from jspec.pipelines.verify_uncal import find_files, verify

    files = []
    for root in args.path:
        files.extend(find_files(root, args.pattern))
    if not files:
        print(f"No files matching {args.pattern!r} under {args.path}")
        return 1

    print(f"Checking {len(files)} file(s) matching {args.pattern!r} ...")
    bad = verify(files, verbose=args.verbose)
    if not bad:
        print(f"All {len(files)} file(s) complete.")
        return 0

    print(f"\n{len(bad)} of {len(files)} file(s) are incomplete:\n")
    for path, why in bad:
        print(f"  {path}\n      {why}")
    print("\nDelete and re-download them with:")
    print("  python -m jspec.pipelines.verify_uncal <dir> --delete")
    return 1


# ---------------------------------------------------------------------------
def cmd_relabel(args):
    """Rename a pre-existing output tree onto the current naming scheme.

    Early runs of the NIRSpec and MIRI-LRS pipelines derived the epoch
    directory name per-visit, so one target could end up spread across
    ``2024aecx_20250601`` and ``sn2024aecx_20250116``. Those names are
    invisible to the product index, which expects a trailing
    ``YYYY-MM-DD``. This renames them in place rather than making you
    re-run Stage 1-3, which is hours of work for a cosmetic problem.

    Nothing is written without ``--apply``.
    """
    from .pipelines.relabel_epochs import main as relabel_main
    argv = ["--outdir", args.outdir]
    if args.tag:
        argv += ["--tag", args.tag]
    if args.instrument:
        argv += ["--instrument", args.instrument]
    if args.apply:
        argv += ["--apply"]
    return relabel_main(argv)


# ---------------------------------------------------------------------------
def cmd_products(args):
    """List (or rebuild) the final-product index for an output directory.

    This is the quickest answer to "which file do I actually analyse?":
    the pipelines record every final-level spectrum they write, and flag
    the one that is the recommended starting point per instrument.
    """
    from .manifest import MANIFEST_NAME, read_manifest, rebuild_manifest

    rows = (rebuild_manifest(args.outdir) if args.rebuild
            else read_manifest(args.outdir))
    if args.instrument:
        from .layout import instrument_label
        want = instrument_label(args.instrument)
        rows = [r for r in rows if r.get("instrument") == want]
    if args.recommended:
        rows = [r for r in rows if r.get("recommended") == "yes"]

    if not rows:
        print(f"No products recorded in {os.path.join(args.outdir, MANIFEST_NAME)}."
              "\nRun with --rebuild to scan the tree, or run a pipeline first.")
        return 1

    width = max(len(r.get("instrument", "")) for r in rows)
    epoch_width = max(len(r.get("epoch", "")) for r in rows)
    for row in rows:
        star = "*" if row.get("recommended") == "yes" else " "
        span = (f"{row['wave_min_um']}-{row['wave_max_um']} um"
                if row.get("wave_min_um") else "")
        print(f"{star} {row.get('instrument',''):<{width}}  "
              f"{row.get('epoch',''):<{epoch_width}}  "
              f"{row.get('product',''):<14}  {span:<20}  {row.get('path','')}")
    if any(r.get("recommended") == "yes" for r in rows):
        print("\n* = recommended product to analyse for this instrument/epoch")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(
        prog="jspec", description="JWST spectral analysis toolkit")
    ap.add_argument("--version", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    # stitch -------------------------------------------------------------
    p = _p_common(sub.add_parser(
        "stitch", help="match and stitch segments into one spectrum"))
    p.add_argument("inputs", nargs="+",
                   help="segment files, or a directory of them")
    p.add_argument("-o", "--output")
    p.add_argument("--anchor", default=None,
                   help="segment name held at scale 1 (default: highest-S/N)")
    p.add_argument("--trim", type=float, default=0.02,
                   help="fraction of each segment's span trimmed before "
                        "measuring ratios (default 0.02)")
    p.add_argument("--no-blend", action="store_true",
                   help="concatenate instead of tapered-blending overlaps")
    p.add_argument("--robust-clip", type=float, default=3.0)
    p.add_argument("--max-scale", type=float, default=3.0,
                   help="reject ratios implying a scale beyond this factor")
    p.add_argument("--plot", help="write a stitching diagnostic figure here")
    p.add_argument("--max-gap-days", type=float, default=1.0,
                   help="a gap larger than this starts a new epoch "
                        "(default 1.0, matching the reduction scripts)")
    p.add_argument("--no-epochs", action="store_true",
                   help="DANGEROUS: stitch all inputs into one spectrum "
                        "regardless of observation date")
    p.add_argument("--exclude", action="append", metavar="NAME",
                   help="Leave this segment out. Matched as a glob against "
                        "the segment name, or as a substring. Repeatable. Use "
                        "this to drop a segment the quality check reports as "
                        "containing no detected source")
    p.add_argument("--keep-flagged", action="store_true",
                   help="Stitch segments that contain no detected source "
                        "instead of dropping them. They are dropped by "
                        "default, because a warning alone still leaves you "
                        "with a written file that looks like a spectrum")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero if any join is flagged as a residual "
                        "calibration step")
    p.set_defaults(func=cmd_stitch)

    # detect -------------------------------------------------------------
    p = _p_common(sub.add_parser(
        "detect", help="find and identify broad/narrow features and dips"))
    p.add_argument("input")
    p.add_argument("-o", "--output", help="write the feature table here")
    p.add_argument("--snr", type=float, default=5.0)
    p.add_argument("--region-snr", type=float, default=15.0)
    p.add_argument("--broad-scale", type=float, default=30000.0,
                   help="velocity scale separating continuum from broad (km/s)")
    p.add_argument("--narrow-scale", type=float, default=600.0)
    p.add_argument("--max-join-fraction", type=float, default=1.0,
                   help="drop features whose extent lies more than this "
                        "fraction inside a blended segment overlap")
    p.add_argument("--tol", type=float, default=2.0,
                   help="identification tolerance, in resolution elements")
    p.add_argument("--no-identify", dest="identify", action="store_false")
    p.add_argument("--range", nargs=2, type=float, metavar=("LO", "HI"))
    p.add_argument("--top", type=int, default=40)
    p.set_defaults(func=cmd_detect)

    # plot ---------------------------------------------------------------
    p = _p_common(sub.add_parser("plot", help="make an annotated figure"))
    p.add_argument("input")
    p.add_argument("-o", "--output", default="spectrum.pdf")
    p.add_argument("--bands", action="store_true", default=True,
                   help="shade known dust/molecular bands (default on)")
    p.add_argument("--no-bands", dest="bands", action="store_false")
    p.add_argument("--kinds", nargs="*",
                   help="band families to show, e.g. molecular pah silicate")
    p.add_argument("--names", nargs="*", help="specific band names to show")
    p.add_argument("--no-band-labels", action="store_true")
    p.add_argument("--features", action="store_true",
                   help="detect and label features")
    p.add_argument("--regions", action="store_true",
                   help="shade high-S/N regions")
    p.add_argument("--snr", type=float, default=5.0)
    p.add_argument("--label-snr", type=float, default=15.0)
    p.add_argument("--region-snr", type=float, default=15.0)
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--rebin", type=int, default=1)
    p.add_argument("--log", action="store_true")
    p.add_argument("--range", nargs=2, type=float, metavar=("LO", "HI"))
    p.add_argument("--title")
    p.add_argument("--width", type=float, default=14.0)
    p.add_argument("--height", type=float, default=5.0)
    p.add_argument("--dpi", type=int, default=160)
    p.set_defaults(func=cmd_plot)

    # fit ----------------------------------------------------------------
    p = _p_common(sub.add_parser(
        "fit", help="nested-sampling profile fit of one feature"))
    p.add_argument("input")
    p.add_argument("--line", type=float, required=True,
                   help="rest-frame wavelength of the feature, in um")
    p.add_argument("--profile", default="gaussian")
    p.add_argument("-n", "--n-components", type=int, default=1)
    p.add_argument("--select", action="store_true",
                   help="fit a grid of profiles/components and choose by "
                        "Bayesian evidence")
    p.add_argument("--profiles", nargs="*",
                   default=["gaussian", "lorentzian", "exp_gaussian"],
                   help="profile grid for --select")
    p.add_argument("--max-components", type=int, default=3)
    p.add_argument("--delta-lnz", type=float, default=5.0)
    p.add_argument("--window", type=float, default=None,
                   help="half-width of the fit window in km/s")
    p.add_argument("--sigma-max", type=float, default=20000.0)
    p.add_argument("--continuum-order", type=int, default=1, choices=(0, 1, 2))
    p.add_argument("--absorption", action="store_true",
                   help="allow negative amplitudes (fit a dip)")
    p.add_argument("--nlive", type=int, default=500)
    p.add_argument("--plot")
    p.add_argument("-o", "--output", help="write posterior samples to .npz")
    p.set_defaults(func=cmd_fit)

    # lines --------------------------------------------------------------
    p = sub.add_parser("lines", help="query the line / dust-band reference data")
    p.add_argument("--bands", action="store_true",
                   help="list dust bands instead of atomic/molecular lines")
    p.add_argument("--range", nargs=2, type=float, metavar=("LO", "HI"))
    p.add_argument("--species", nargs="*")
    p.add_argument("--kind", nargs="*")
    p.set_defaults(func=cmd_lines)

    # epochs -------------------------------------------------------------
    p = _p_common(sub.add_parser(
        "epochs", help="show how files group into epochs, without processing"))
    p.add_argument("inputs", nargs="+")
    p.add_argument("--max-gap-days", type=float, default=1.0)
    p.set_defaults(func=cmd_epochs)

    # products -----------------------------------------------------------
    p = sub.add_parser(
        "products",
        help="list or rebuild the final-product index of an output directory")
    p.add_argument("outdir", help="Pipeline output directory")
    p.add_argument("--rebuild", action="store_true",
                   help="Rescan the tree and rewrite products.csv from what "
                        "is actually on disk, instead of just reading it")
    p.add_argument("--instrument", help="Only show this instrument")
    p.add_argument("--recommended", action="store_true",
                   help="Only show the product to analyse for each instrument "
                        "and epoch")
    p.set_defaults(func=cmd_products)

    # verify -------------------------------------------------------------
    p = sub.add_parser(
        "verify",
        help="check downloaded FITS files for truncation/corruption")
    p.add_argument("path", nargs="+", help="File(s) or directory tree(s) to check")
    p.add_argument("--pattern", default="*_uncal.fits",
                   help="Filename pattern to check (default: %(default)s)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="List good files as well as bad ones")
    p.set_defaults(func=cmd_verify)

    # relabel ------------------------------------------------------------
    p = sub.add_parser(
        "relabel",
        help="rename an older output tree onto the <TAG>_<YYYY-MM-DD> scheme")
    p.add_argument("outdir", help="Pipeline output directory")
    p.add_argument("--tag", help="Display tag to use, e.g. SN2024aecx. "
                                 "Default: inferred from the existing names")
    p.add_argument("--instrument", help="Only touch this instrument folder")
    p.add_argument("--apply", action="store_true",
                   help="Actually rename. Without this, only a preview is printed")
    p.set_defaults(func=cmd_relabel)

    # info ---------------------------------------------------------------
    p = _p_common(sub.add_parser("info", help="summarise a spectrum file"))
    p.add_argument("input")
    p.set_defaults(func=cmd_info)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if getattr(args, "version", False):
        from .jspec import __version__
        print(f"jspec {__version__}")
        return 0
    if not getattr(args, "cmd", None):
        ap.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":                                 # pragma: no cover
    sys.exit(main())
