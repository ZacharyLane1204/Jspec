"""
Build a LaTeX table of JWST observations for a given proposal ID, using
Stage 3 (calib_level=3) extracted 1D spectral products (X1D) queried via
astroquery.mast.

For each observation this pulls:
    - Date / UT start / MJD          -> from the FITS primary header
    - Instrument/setup               -> INSTRUME (+ GRATING/FILTER, or
                                         "MRS" for MIRI IFU) from the header
    - Aperture                       -> APERNAME (or similar) in the header
    - Wavelength range & # of points -> read directly from the x1d
                                         spectrum table(s) in the product(s)

Handles MIRI MRS observations, where a single "observation" is split across
several channel/band X1D files: these are grouped together and their
wavelength coverage is combined into one row.

Usage:
    python make_jwst_table.py --proposal_id 9233 \
        --caption "\\textit{JWST} observations of SN\\,2024aecx used in this analysis." \
        --label tab:jwst-observations \
        --outfile jwst_table.tex

If it still finds nothing, run:
    python make_jwst_table.py --proposal_id 9233 --diagnose
to print the actual productType / productSubGroupDescription / calib_level
values MAST has for this proposal, and adjust --subgroups accordingly.
"""

import argparse
import os
import re
import tempfile
from collections import defaultdict

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astroquery.mast import Observations


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def diagnose_products(proposal_id):
    """Print the actual productType / subgroup / calib_level values MAST
    has for this proposal, safely handling masked/missing entries."""
    obs = Observations.query_criteria(proposal_id=str(proposal_id), obs_collection="JWST")
    products = Observations.get_product_list(obs)

    def safe_unique(colname):
        col = products[colname]
        vals = set()
        for v in col:
            if v is np.ma.masked or v is None:
                continue
            vals.add(str(v))
        return vals

    print(f"Observations found: {len(obs)}")
    print(f"Products found: {len(products)}")
    print("productType values:", safe_unique("productType"))
    print("productSubGroupDescription values:", safe_unique("productSubGroupDescription"))
    print("calib_level values:", safe_unique("calib_level"))
    print("Example filenames:")
    for fn in list(products["productFilename"])[:15]:
        print("  ", fn)


# --------------------------------------------------------------------------
# Metadata extraction helpers
# --------------------------------------------------------------------------

def get_instrument_setup(hdr):
    """
    Build a compact instrument/setup string, e.g. 'NIRSPEC/G395M/F290LP',
    'MIRI/P750L' (LRS), or 'MIRI/MRS' (medium resolution IFU spectrometer).
    """
    instrument = (hdr.get("INSTRUME") or "").upper()
    exp_type = (hdr.get("EXP_TYPE") or "").upper()
    grating = hdr.get("GRATING")
    filt = hdr.get("FILTER")

    if instrument == "MIRI" and "MRS" in exp_type:
        return "MIRI/MRS"

    parts = [instrument]
    for val in (grating, filt):
        val = (val or "").upper()
        if val and val not in ("N/A", "NONE", "CLEAR", ""):
            parts.append(val)
    return "/".join(parts)


def get_aperture(hdr):
    """Pull the aperture name from a FITS header, trying common keywords."""
    for key in ("APERNAME", "PPS_APER", "APERTURE"):
        if key in hdr and hdr[key]:
            return hdr[key]
    return "UNKNOWN"


def get_mjd_start(hdr):
    """Try the common keywords JWST pipelines use for exposure start MJD."""
    for key in ("MJD-BEG", "EXPSTART", "MJD_BEG"):
        if key in hdr and hdr[key] not in (None, ""):
            return float(hdr[key])
    raise KeyError("No MJD start keyword found in header (checked MJD-BEG, EXPSTART).")


def read_spectrum_wavelength(filepath):
    """
    Open a Stage 3 x1d (extracted 1D spectrum) FITS product and return the
    wavelength array (microns) from its EXTRACT1D/COMBINE1D table.
    A single x1d file can contain multiple EXTRACT1D extensions (e.g. one
    per slit/order); all are concatenated.
    """
    waves = []
    with fits.open(filepath) as hdul:
        for hdu in hdul:
            cols = getattr(hdu, "columns", None)
            if cols is not None and "WAVELENGTH" in cols.names:
                w = np.asarray(hdu.data["WAVELENGTH"], dtype=float)
                waves.append(w[np.isfinite(w)])
    if not waves:
        raise ValueError(f"No WAVELENGTH table found in {filepath}")
    return np.concatenate(waves)


# --------------------------------------------------------------------------
# MAST query + download
# --------------------------------------------------------------------------

def query_products(proposal_id, calib_level=3, product_type="SCIENCE",
                    subgroups=("X1D",)):
    """
    Query MAST for Stage-3 extracted-spectrum products belonging to a
    proposal. Returns the filtered product Table (rows are files, possibly
    several per observation, e.g. MIRI MRS channels/bands).
    """
    obs = Observations.query_criteria(
        proposal_id=str(proposal_id),
        obs_collection="JWST",
    )
    if len(obs) == 0:
        raise RuntimeError(f"No JWST observations found for proposal {proposal_id}")

    products = Observations.get_product_list(obs)
    filtered = Observations.filter_products(
        products,
        productType=product_type,
        calib_level=[calib_level],
        productSubGroupDescription=list(subgroups),
    )
    if len(filtered) == 0:
        diagnose_products(proposal_id)
        raise RuntimeError(
            f"No calib_level={calib_level} productType={product_type!r} products "
            f"with subgroup(s) {subgroups} found for proposal {proposal_id}. "
            "See the diagnostic listing above and adjust --subgroups / --product_type."
        )
    return filtered


def group_key_for_row(row):
    """
    Group product rows belonging to the same observation together, so that
    e.g. MIRI MRS channel/band files (multiple rows, one observation) end
    up as a single table row. Prefers the numeric parent observation id
    ('obsID') if present; otherwise falls back to stripping the
    channel/band/detector suffix off the filename.
    """
    for key in ("obsID", "parent_obsid", "obs_id"):
        if key in row.colnames and row[key] is not np.ma.masked:
            return str(row[key])
    # Fallback: strip things like "_ch1-shortmediumlong_x1d.fits" or
    # "_nrs1_x1d.fits" down to the "jw<prog>-o<obs>_t<targ>_<instr>" stem.
    fn = str(row["productFilename"])
    stem = re.sub(r"(_[a-z0-9-]+)?_x1d\.fits$", "", fn, flags=re.IGNORECASE)
    stem = re.sub(r"_(nrs1|nrs2|mirifushort|mirifulong)$", "", stem, flags=re.IGNORECASE)
    return stem


def download_and_build_rows(products, download_dir):
    """
    Download all matching products, group them by observation, and build
    one table row per group (combining wavelength coverage across files
    within a group, e.g. MIRI MRS channels/bands).
    """
    manifest = Observations.download_products(
        products, download_dir=download_dir, flat=True
    )
    path_by_filename = {
        os.path.basename(p): p for p in manifest["Local Path"] if os.path.exists(p)
    }

    groups = defaultdict(list)
    for row in products:
        fn = str(row["productFilename"])
        local_path = path_by_filename.get(fn)
        if local_path is None:
            continue
        groups[group_key_for_row(row)].append(local_path)

    rows = []
    for key, paths in groups.items():
        try:
            all_waves = []
            hdr = None
            apertures = set()
            for p in paths:
                with fits.open(p) as hdul:
                    this_hdr = hdul[0].header
                    if hdr is None:
                        hdr = this_hdr  # use first file's header for date/instrument
                    apertures.add(get_aperture(this_hdr))
                all_waves.append(read_spectrum_wavelength(p))
            wave = np.concatenate(all_waves)

            mjd_start = get_mjd_start(hdr)
            t = Time(mjd_start, format="mjd")

            rows.append({
                "date": t.datetime.strftime("%Y-%m-%d"),
                "ut_start": t.datetime.strftime("%H:%M:%S"),
                "mjd": mjd_start,
                "instrument_setup": get_instrument_setup(hdr),
                "aperture": "; ".join(sorted(apertures)),
                "wave_min": float(wave.min()),
                "wave_max": float(wave.max()),
                "npoints": int(wave.size),
            })
        except Exception as e:
            print(f"  [skip group {key}] {e}")

    rows.sort(key=lambda r: r["mjd"])
    return rows


# --------------------------------------------------------------------------
# LaTeX table construction
# --------------------------------------------------------------------------

def escape_latex(s):
    """Escape underscores for LaTeX (aperture/instrument names contain them)."""
    return str(s).replace("_", r"\_")


def build_latex_table(rows, caption, label):
    lines = []
    lines.append(r"\begin{table*}")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{llrllrr}")
    lines.append(r"\hline")
    lines.append(
        r"Date & UT start & MJD & Instrument/setup & Aperture & "
        r"Wavelength range ($\mu$m) & Points \\"
    )
    lines.append(r"\hline")
    for r in rows:
        lines.append(
            f"{r['date']} & {r['ut_start']} & {r['mjd']:.3f} & "
            f"{escape_latex(r['instrument_setup'])} & {escape_latex(r['aperture'])} & "
            f"{r['wave_min']:.3f}--{r['wave_max']:.3f} & {r['npoints']} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--proposal_id", required=True, help="JWST proposal ID, e.g. 9233")
    parser.add_argument(
        "--caption",
        default=r"\textit{JWST} observations used in this analysis.",
    )
    parser.add_argument("--label", default="tab:jwst-observations")
    parser.add_argument("--outfile", default="jwst_table.tex")
    parser.add_argument("--download_dir", default=None)
    parser.add_argument("--product_type", default="SCIENCE")
    parser.add_argument(
        "--subgroups", nargs="+", default=["X1D"],
        help="productSubGroupDescription values to keep (e.g. X1D C1D)",
    )
    parser.add_argument(
        "--diagnose", action="store_true",
        help="Just print the available product types/subgroups for this proposal and exit.",
    )
    args = parser.parse_args()

    if args.diagnose:
        diagnose_products(args.proposal_id)
        return

    download_dir = args.download_dir or tempfile.mkdtemp(prefix="jwst_products_")
    print(f"Downloading products to: {download_dir}")

    products = query_products(
        args.proposal_id, product_type=args.product_type, subgroups=tuple(args.subgroups)
    )
    print(f"Found {len(products)} matching product files.")

    rows = download_and_build_rows(products, download_dir)
    if not rows:
        raise RuntimeError("No rows were successfully built — check product/header assumptions.")

    latex = build_latex_table(rows, args.caption, args.label)

    with open(args.outfile, "w") as f:
        f.write(latex + "\n")

    print(f"Wrote {len(rows)} rows to {args.outfile}")


if __name__ == "__main__":
    main()