"""
jspec.linedb
============

Reference data: atomic/ionic lines, molecular and dust bands, and the matching
machinery that turns a detected feature into a candidate identification.

The two tables ship with the package as CSVs under ``jspec/data/`` so they can
be edited, extended, or replaced wholesale without touching code --
:func:`load_lines` and :func:`load_bands` accept a path, and every consumer
takes the loaded table as an argument. Add your own lines by copying the CSV,
editing it, and passing the new path.

Matching philosophy
-------------------

Identification is reported as a *ranked list of candidates with a velocity
offset*, never as a single answer, because at MIRI MRS resolution a great many
astrophysically plausible transitions sit within a few hundred km/s of each
other -- [Ne V] 14.32 and [Cl II] 14.37 are 1000 km/s apart, [O IV] 25.89 and
[Fe II] 25.99 are 1100 km/s apart. Anything that prints one species and hides
the alternatives is lying about how well the data constrains the answer.

Three things narrow the list honestly:

* **Tolerance scaled to the instrument.** The search window defaults to a
  multiple of the *local resolution element*, not a fixed number of microns
  or a fixed velocity. A 300 km/s window is generous at 5 um where MRS
  resolves 85 km/s and meaningless at 1.2 um where the NIRSpec prism resolves
  9000 km/s.
* **Width regime.** A feature detected as broad is not matched against lines
  flagged ``narrow`` in the list (pure-rotational H2, host [O III]), and vice
  versa. See the ``expected_width`` column.
* **Family corroboration.** :func:`corroborate` checks whether the *other*
  lines of a species that was proposed are also present where they should be.
  A lone 1.644 um detection identified as [Fe II] is a guess; the same
  detection with 1.257 um also present at the same velocity is an
  identification.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .spectrum import C_KMS, resolving_power

__all__ = [
    "DATA_DIR", "LINE_LIST_PATH", "DUST_BANDS_PATH",
    "load_lines", "load_bands", "match_line", "match_lines",
    "bands_covering", "corroborate", "LineMatch", "refine_redshift",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LINE_LIST_PATH = os.path.join(DATA_DIR, "line_list.csv")
DUST_BANDS_PATH = os.path.join(DATA_DIR, "dust_bands.csv")

_LINE_CACHE: dict[str, pd.DataFrame] = {}
_BAND_CACHE: dict[str, pd.DataFrame] = {}


def load_lines(path: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Load the atomic/molecular line list.

    Expected columns: ``group, species, wavelength_um, label, kind, category,
    expected_width, notes``. Only ``species``, ``wavelength_um`` and ``label``
    are strictly required; the rest are filled with sensible defaults so a
    user-supplied two-column CSV still works.

    ``wavelength_um`` must be rest-frame vacuum microns.
    """
    path = path or LINE_LIST_PATH
    if use_cache and path in _LINE_CACHE:
        return _LINE_CACHE[path].copy()
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    if "wavelength_um" not in df.columns:
        for alt in ("wavelength", "lam_um", "lambda_um", "wave_um"):
            if alt in df.columns:
                df = df.rename(columns={alt: "wavelength_um"})
                break
    required = {"species", "wavelength_um"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path}: missing required column(s) {sorted(missing)}")
    df["wavelength_um"] = pd.to_numeric(df["wavelength_um"], errors="coerce")
    df = df[np.isfinite(df["wavelength_um"])].copy()
    for col, default in (("label", None), ("group", "other"), ("kind", "unknown"),
                         ("category", "other"), ("expected_width", "either"),
                         ("notes", "")):
        if col not in df.columns:
            df[col] = default
    df["label"] = df["label"].fillna(
        df["species"].astype(str) + " " + df["wavelength_um"].round(4).astype(str))
    df["expected_width"] = df["expected_width"].fillna("either").str.strip().str.lower()
    df = df.sort_values("wavelength_um").reset_index(drop=True)
    _LINE_CACHE[path] = df
    return df.copy()


def load_bands(path: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Load the dust / molecular band table.

    Expected columns: ``name, species, lam_peak_um, lam_min_um, lam_max_um,
    kind, transition, color, notes``. ``name``, ``lam_min_um`` and
    ``lam_max_um`` are required; ``lam_peak_um`` defaults to the bracket
    midpoint.
    """
    path = path or DUST_BANDS_PATH
    if use_cache and path in _BAND_CACHE:
        return _BAND_CACHE[path].copy()
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"name", "lam_min_um", "lam_max_um"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path}: missing required column(s) {sorted(missing)}")
    for c in ("lam_min_um", "lam_max_um", "lam_peak_um"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "lam_peak_um" not in df.columns:
        df["lam_peak_um"] = np.nan
    df["lam_peak_um"] = df["lam_peak_um"].fillna(
        0.5 * (df["lam_min_um"] + df["lam_max_um"]))
    for col, default in (("species", ""), ("kind", "other"),
                         ("transition", ""), ("color", None), ("notes", "")):
        if col not in df.columns:
            df[col] = default
    df = df[np.isfinite(df["lam_min_um"]) & np.isfinite(df["lam_max_um"])]
    df = df.sort_values("lam_min_um").reset_index(drop=True)
    _BAND_CACHE[path] = df
    return df.copy()


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------
@dataclass
class LineMatch:
    """One candidate identification for an observed feature."""
    label: str
    species: str
    rest_um: float
    observed_rest_um: float
    velocity_kms: float
    delta_um: float
    category: str
    kind: str
    expected_width: str
    notes: str = ""
    resolution_elements: float = np.nan   # |offset| in units of the local FWHM

    def __repr__(self):
        return (f"<{self.label} ({self.species}) dv={self.velocity_kms:+.0f} km/s"
                f", {self.resolution_elements:.2f} res.elem>")


def _width_compatible(expected: str, observed: str | None) -> bool:
    """Is a line that is normally ``expected`` width an acceptable ID for a
    feature observed as ``observed`` width?"""
    if observed is None or expected in (None, "", "either"):
        return True
    observed = observed.strip().lower()
    if observed not in ("broad", "narrow"):
        return True
    return expected == "either" or expected == observed


def match_line(rest_wavelength_um, *, lines=None, observed_wavelength_um=None,
               tol_kms=None, tol_res_elements=2.0, width=None,
               categories=None, exclude_categories=None, mode=None,
               max_candidates=8) -> list[LineMatch]:
    """Find candidate identifications for one feature.

    Parameters
    ----------
    rest_wavelength_um : float
        The feature's measured centre, already de-redshifted.
    lines : DataFrame, optional
        From :func:`load_lines`; loaded on demand if omitted.
    observed_wavelength_um : float, optional
        The *observed*-frame centre. Needed to compute the local instrument
        resolution, which sets the default tolerance. If omitted, the rest
        wavelength is used, which is correct only at z=0 and will make the
        tolerance slightly too tight at higher redshift.
    tol_kms : float, optional
        Fixed search window in km/s. Overrides ``tol_res_elements``. Use this
        when you know the feature's own velocity uncertainty dominates, e.g.
        for a broad ejecta line whose centroid is uncertain by thousands of
        km/s regardless of how well the instrument resolves.
    tol_res_elements : float
        Default window, in multiples of the local resolution FWHM. Two
        resolution elements is deliberately generous: wavelength solutions,
        centroiding of asymmetric profiles, and real bulk velocity offsets all
        contribute, and a tight window produces confident wrong answers.
    width : {'broad', 'narrow', None}
        Filter candidates by the regime they are normally seen in.
    categories, exclude_categories : sequence of str, optional
        e.g. ``categories=['H', 'He']``, or ``exclude_categories=['coronal']``
        when the object has no hard ionising source.
    mode : str, optional
        Instrument mode passed to :func:`jspec.spectrum.resolving_power`.

    Returns
    -------
    list of LineMatch, sorted by |velocity offset|.
    """
    if lines is None:
        lines = load_lines()
    df = lines
    if categories:
        df = df[df["category"].isin(list(categories))
                | df["group"].isin(list(categories))]
    if exclude_categories:
        df = df[~df["category"].isin(list(exclude_categories))
                & ~df["group"].isin(list(exclude_categories))]
    if df.empty:
        return []

    obs = observed_wavelength_um if observed_wavelength_um is not None else rest_wavelength_um
    r = resolving_power(obs, mode=mode)
    fwhm_kms = C_KMS / r if np.isfinite(r) and r > 0 else 1000.0

    tol = tol_kms if tol_kms is not None else tol_res_elements * fwhm_kms

    lam = df["wavelength_um"].to_numpy(dtype=float)
    dv = C_KMS * (rest_wavelength_um - lam) / lam
    sel = np.abs(dv) <= tol

    out = []
    for i in np.flatnonzero(sel):
        row = df.iloc[i]
        if not _width_compatible(str(row.get("expected_width", "either")), width):
            continue
        out.append(LineMatch(
            label=str(row.get("label")),
            species=str(row.get("species")),
            rest_um=float(row["wavelength_um"]),
            observed_rest_um=float(rest_wavelength_um),
            velocity_kms=float(dv[i]),
            delta_um=float(rest_wavelength_um - lam[i]),
            category=str(row.get("category", "other")),
            kind=str(row.get("kind", "unknown")),
            expected_width=str(row.get("expected_width", "either")),
            notes=str(row.get("notes", "") or ""),
            resolution_elements=float(abs(dv[i]) / fwhm_kms) if fwhm_kms > 0 else np.nan,
        ))
    out.sort(key=lambda m: abs(m.velocity_kms))
    return out[:max_candidates]


def match_lines(features, **kwargs) -> dict:
    """Match many features at once.

    ``features`` is an iterable of objects with ``rest_center`` and
    ``center`` attributes (i.e. :class:`jspec.features.Feature`) or of
    ``(rest_um, observed_um)`` tuples. Returns ``{feature_index: [LineMatch,
    ...]}``. The line list is loaded once and reused, which matters when
    matching a few hundred detections.
    """
    kwargs.setdefault("lines", load_lines())
    out = {}
    for i, f in enumerate(features):
        if hasattr(f, "rest_center"):
            rest, obs = f.rest_center, f.center
            kwargs_i = dict(kwargs)
            kwargs_i.setdefault("width", getattr(f, "width_class", None))
        else:
            rest, obs = (f if isinstance(f, (tuple, list)) else (f, f))
            kwargs_i = kwargs
        out[i] = match_line(rest, observed_wavelength_um=obs, **kwargs_i)
    return out


def bands_covering(rest_wavelength_um, bands=None, kinds=None) -> pd.DataFrame:
    """Which dust/molecular bands bracket this rest wavelength.

    Unlike :func:`match_line` this is a containment test, not a proximity
    test: a solid-state band is an extended wavelength interval, so "is my
    feature inside the 9.7 um silicate feature" is the meaningful question,
    and a velocity offset from a band centre is not.
    """
    if bands is None:
        bands = load_bands()
    df = bands
    if kinds:
        df = df[df["kind"].isin(list(kinds))]
    sel = ((df["lam_min_um"] <= rest_wavelength_um)
           & (df["lam_max_um"] >= rest_wavelength_um))
    return df[sel].copy()


def corroborate(matches, detected_rest_um, lines=None, tol_kms=None,
                tol_res_elements=2.0, mode=None) -> list[dict]:
    """Check each candidate species against the rest of its own transitions.

    For every distinct species proposed in ``matches``, look up all of its
    lines in the list, work out which fall inside the wavelength range spanned
    by ``detected_rest_um``, and count how many of those actually have a
    detection near them.

    This is the single most useful discriminator available without fitting
    anything: species produce *families* of lines with fixed wavelength
    ratios, and a species whose other strong lines are missing where they
    should be detectable is almost certainly the wrong identification, no
    matter how well its one line matches.

    Returns, per species, a dict with ``species``, ``n_expected_in_range``,
    ``n_detected``, ``fraction``, ``detected_lines``, ``missing_lines``, and a
    mean ``velocity_kms`` over the matched members -- a *consistent* velocity
    across family members is itself strong corroboration, while a species
    whose members scatter over thousands of km/s is a coincidence pile-up.

    Caveat: ``n_expected_in_range`` counts transitions, not *detectable*
    transitions. A species whose other lines are intrinsically weak, or fall
    in a low-S/N part of the spectrum, will score poorly for entirely
    uninteresting reasons. Use the fraction to rank candidates, not to reject
    them outright.
    """
    if lines is None:
        lines = load_lines()
    detected = np.asarray(sorted(detected_rest_um), dtype=float)
    if detected.size == 0:
        return []
    wmin, wmax = detected.min(), detected.max()

    species_list = sorted({m.species for m in matches})
    results = []
    for sp in species_list:
        fam = lines[(lines["species"] == sp)
                    & (lines["wavelength_um"] >= wmin)
                    & (lines["wavelength_um"] <= wmax)]
        if fam.empty:
            continue
        hits, misses, vels = [], [], []
        for _, row in fam.iterrows():
            lam = float(row["wavelength_um"])
            r = resolving_power(lam, mode=mode)
            fwhm_kms = C_KMS / r if np.isfinite(r) and r > 0 else 1000.0
            tol = tol_kms if tol_kms is not None else tol_res_elements * fwhm_kms
            dv = C_KMS * (detected - lam) / lam
            k = int(np.argmin(np.abs(dv)))
            if abs(dv[k]) <= tol:
                hits.append(str(row.get("label", sp)))
                vels.append(float(dv[k]))
            else:
                misses.append(str(row.get("label", sp)))
        n_exp = len(fam)
        results.append(dict(
            species=sp,
            n_expected_in_range=n_exp,
            n_detected=len(hits),
            fraction=len(hits) / n_exp if n_exp else np.nan,
            detected_lines=hits,
            missing_lines=misses,
            velocity_kms=float(np.mean(vels)) if vels else np.nan,
            velocity_scatter_kms=float(np.std(vels)) if len(vels) > 1 else np.nan,
        ))
    results.sort(key=lambda d: (-d["n_detected"], -d["fraction"]))
    return results


def refine_redshift(features, lines=None, search_kms=6000.0, step_kms=20.0,
                    min_snr=10.0, max_fwhm_kms=3000.0, categories=None,
                    use_emission=True, weight_by_snr=True, current_z=0.0):
    """Estimate the systematic velocity offset of a detected feature list
    against the reference line list.

    Why this exists: every rest-frame result in this package -- every
    identification, every fitted velocity, every abundance -- is downstream of
    the assumed redshift. If that redshift is wrong, the identifications are
    still *self-consistently* wrong, so nothing visibly fails; the lines just
    all come out offset by the same amount, and the offset gets quietly
    reported as a physical velocity. This function makes that failure mode
    visible by asking a different question: is there a single shift that makes
    the whole detected line list line up better?

    The method is a simple cross-correlation in velocity: for each trial
    shift, each selected feature contributes a Gaussian-weighted score for its
    nearest catalogue line, and the scores are summed. The peak of that curve
    is the systematic offset.

    Parameters
    ----------
    features : list of Feature
        From :func:`jspec.features.find_features`. Only reasonably narrow,
        high-S/N features are used -- a 6000 km/s SN line has no centroid
        precise enough to contribute, and including broad features both
        dilutes the peak and biases it, because genuinely blueshifted ejecta
        lines are broad.
    search_kms : float
        Half-range of the velocity search.
    min_snr, max_fwhm_kms : float
        Selection cuts on the features used.
    categories : sequence of str, optional
        Restrict the catalogue to e.g. ``['fine-structure']``. Narrow
        fine-structure lines from the host or from shocked CSM are the best
        anchors; recombination lines in SN ejecta are usually not.

    Returns
    -------
    dict with keys ``dv_kms`` (the offset; positive means the features sit
    redward of the catalogue, i.e. the assumed redshift is too *low*),
    ``z_suggested``, ``n_used``, ``significance`` (peak height over the
    scatter of the correlation curve away from the peak), ``velocities``
    and ``score`` for plotting the curve yourself.

    Caveats
    -------
    A peak here is suggestive, not authoritative. With a dense catalogue and
    few features it is entirely possible to find a spurious alignment --
    always check ``n_used`` and ``significance``, and sanity-check the
    suggested redshift against an independent host measurement before
    adopting it. This will also happily report a real physical bulk velocity
    (an expanding shell, an outflow) as a redshift error; it cannot tell the
    two apart, and you have to.
    """
    if lines is None:
        lines = load_lines()
    df = lines
    if categories:
        df = df[df["category"].isin(list(categories))]
    cat = np.sort(df["wavelength_um"].to_numpy(dtype=float))
    if cat.size == 0:
        raise ValueError("no catalogue lines selected")

    sel = [f for f in features
           if f.integrated_snr >= min_snr
           and np.isfinite(f.fwhm_kms) and f.fwhm_kms <= max_fwhm_kms
           and (f.sign == "emission" or not use_emission)
           and not f.on_join]
    if len(sel) < 3:
        raise ValueError(
            f"only {len(sel)} features pass the cuts (need >= 3); relax "
            f"min_snr / max_fwhm_kms, or there may genuinely be too few "
            f"narrow lines to do this")

    obs = np.array([f.rest_center for f in sel])
    wts = np.array([f.integrated_snr for f in sel]) if weight_by_snr \
        else np.ones(len(sel))
    # Each feature's own width sets how sharply it votes: a narrow line
    # localises the shift well, a marginally-resolved one should not pretend
    # to.
    sig = np.array([max(f.fwhm_kms / 2.355, 30.0) for f in sel])

    shifts = np.arange(-search_kms, search_kms + step_kms, step_kms)
    score = np.zeros_like(shifts)
    for i, dv in enumerate(shifts):
        w_shift = obs / (1.0 + dv / C_KMS)
        idx = np.clip(np.searchsorted(cat, w_shift), 1, cat.size - 1)
        near = np.where(np.abs(cat[idx] - w_shift) < np.abs(cat[idx - 1] - w_shift),
                        cat[idx], cat[idx - 1])
        resid = C_KMS * (w_shift - near) / near
        score[i] = float(np.sum(wts * np.exp(-0.5 * (resid / sig) ** 2)))

    k = int(np.argmax(score))
    dv_best = float(shifts[k])
    off_peak = score[np.abs(shifts - dv_best) > 3.0 * np.median(sig)]
    sig_level = ((score[k] - np.median(off_peak)) / np.std(off_peak)
                 if off_peak.size > 5 and np.std(off_peak) > 0 else np.nan)

    return {
        "dv_kms": dv_best,
        "z_suggested": (1.0 + current_z) * (1.0 + dv_best / C_KMS) - 1.0,
        "n_used": len(sel),
        "significance": float(sig_level),
        "velocities": shifts,
        "score": score,
    }
