"""
jspec.spectrum
==============

One container type for a 1-D spectrum, plus readers that accept every file
format this project already produces, so that nothing downstream has to care
whether a spectrum arrived as an ecsv with ``WAVELENGTH/FLUX/FLUX_ERROR``
columns (``combine_epoch_spectra``), an ecsv with lowercase
``wavelength/flux/flux_err`` (``nirspec_fs_auto_pipeline``), a MIRI
``_full.ecsv`` with ``FLUX_ERR_TOTAL`` and a bitmask ``FLAG`` column
(``miri_mrs_refine_extract``), a bare csv, or a pipeline ``x1d`` FITS file.

Design notes
------------

**Wavelength is always microns, vacuum, observed frame, and strictly
increasing.** Readers normalize to that. ``Spectrum.rest_wavelength`` applies
the redshift, but the stored ``wavelength`` stays observed -- so a spectrum
can be re-used with a corrected redshift without reloading, and so the
instrument resolving power (which is a property of the observed frame) stays
computable at any time.

**Flux is whatever unit it arrived in, carried as a string.** Converting Jy to
F_lambda is lossy-by-convention (it needs the wavelength), so it is an explicit
method, never implicit. The one thing that *is* enforced is that ``flux_err``
is a 1-sigma uncertainty in the same unit as ``flux``.

**Bad data is carried as a mask, never deleted.** Dropping flagged rows at
read time is what makes a stitched spectrum silently change length between
runs, and it destroys the information needed to tell a real gap (no coverage)
from a masked artefact (coverage, rejected). ``Spectrum.good`` is the boolean
"use this point" view; ``Spectrum.compressed()`` materializes it only when a
consumer genuinely needs a dense array.
"""
from __future__ import annotations

import glob as _glob
import os
import re
import warnings
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

import numpy as np
from astropy.io import fits
from astropy.table import Table

__all__ = [
    "Spectrum",
    "Segment",
    "read_spectrum",
    "read_segments",
    "read_x1d",
    "resolving_power",
    "band_of",
    "air_to_vac",
    "vac_to_air",
    "C_KMS",
    "MRS_RED_END_UM",
]

#: Speed of light in km/s -- used everywhere velocities appear.
C_KMS = 299792.458

#: Observed wavelength beyond which MIRI MRS channel-4-long data are masked by
#: default (see :meth:`Spectrum.mask_red_end`).
#:
#: Chosen empirically rather than from the nominal band edge of 28.78 um.
#: Measured on this project's own combined products, the degradation is not
#: gradual -- it switches on:
#:
#: =========  ========  ===========  ==============
#: lambda     med S/N   % negative   scatter/error
#: =========  ========  ===========  ==============
#: 26.5-27.0      12.7          0.0            2.84
#: 27.0-27.5      11.4          0.0            2.84
#: 27.5-28.0       9.7          4.8            2.14
#: 28.0-28.5       4.4         14.5            2.59
#: 28.5-28.6       3.9         23.8            3.03
#: =========  ========  ===========  ==============
#:
#: Negative fluxes appear for the first time at 27.5 um and then triple and
#: triple again; the median S/N falls below 10 at the same point. Below 27.5
#: there are no negatives at all. That is the boundary.
#:
#: Note the last column: the point-to-point scatter exceeds the quoted errors
#: by a factor of ~2.5 throughout channel 4, not only past the cutoff. The
#: pipeline uncertainties are underestimated across the whole of ch4-long.
#: That is not corrected here -- :func:`jspec.continuum.decompose` estimates
#: the noise empirically for exactly this reason -- but it is worth knowing
#: before quoting a formal error on anything redward of ~24 um.
MRS_RED_END_UM = 27.5

# ---------------------------------------------------------------------------
# column-name synonyms seen across this project's own output files
# ---------------------------------------------------------------------------
_WAVE_KEYS = ("wavelength", "wave", "lam", "lambda", "wl", "WAVELENGTH", "WAVE")
_FLUX_KEYS = ("flux", "FLUX", "fnu", "f_nu", "flux_jy")
_ERR_KEYS = (
    # ordered by preference: a total error that already folds in the
    # systematic term is a better default than the random-only term, because
    # every consumer in this package treats flux_err as "how much do I
    # believe this point".
    "flux_err", "FLUX_ERR_TOTAL", "FLUX_ERROR", "flux_error", "error", "ERROR",
    "err", "ERR", "sigma", "SIGMA", "FLUX_ERR_FORMAL", "FLUX_ERR",
)

#: Bit meanings used by ``miri_mrs_refine_extract.py``'s ``FLAG`` column.
MRS_FLAG = dict(
    EDGE=1, SPIKE=2, HOSTLINE=4, BKG_CAPPED=8, BKG_FAIL=16,
    RED_END=32, BLENDED=64, SYS_EXTRAP=128,
)

#: Which of those bits mean "do not use this point by default". ``BKG_CAPPED``,
#: ``BLENDED`` and ``SYS_EXTRAP`` are deliberately *not* in here: they mark
#: points whose uncertainty is inflated or whose interpretation needs care,
#: not points that are wrong, and masking them by default would punch holes in
#: otherwise fine spectra.
MRS_FLAG_BAD = (
    MRS_FLAG["EDGE"] | MRS_FLAG["SPIKE"] | MRS_FLAG["HOSTLINE"]
    | MRS_FLAG["BKG_FAIL"] | MRS_FLAG["RED_END"]
)


# ---------------------------------------------------------------------------
# JWST instrument bands and resolving power
# ---------------------------------------------------------------------------
#: MIRI MRS sub-bands: name -> (lambda_min, lambda_max, R_at_min, R_at_max).
#: Wavelength coverage and resolving power follow the in-flight
#: characterisation (Labiano et al. 2021; Argyriou et al. 2023); R is
#: interpolated linearly in wavelength within a band, which reproduces the
#: published curves to well within the few-percent level that matters for
#: deciding whether a line is resolved.
MRS_BANDS = {
    "ch1-short":  (4.90, 5.74, 3320, 3710),
    "ch1-medium": (5.66, 6.63, 3190, 3750),
    "ch1-long":   (6.53, 7.65, 3100, 3610),
    "ch2-short":  (7.51, 8.77, 2990, 3110),
    "ch2-medium": (8.67, 10.13, 2750, 3170),
    "ch2-long":   (10.02, 11.70, 2860, 3300),
    "ch3-short":  (11.55, 13.47, 2530, 2880),
    "ch3-medium": (13.34, 15.57, 2470, 2640),
    "ch3-long":   (15.41, 17.98, 1980, 2790),
    "ch4-short":  (17.70, 20.95, 1460, 1930),
    "ch4-medium": (20.69, 24.48, 1680, 1760),
    "ch4-long":   (24.19, 28.78, 1330, 1630),
}

#: NIRSpec dispersers: name -> (lambda_min, lambda_max, R_at_min, R_at_max).
#: The medium and high gratings are close to constant in R; the prism is
#: strongly wavelength dependent and its entry is a coarse bracket only --
#: ``resolving_power`` interpolates the published prism curve separately.
NIRSPEC_BANDS = {
    "g140m": (0.90, 1.89, 700, 1300),
    "g235m": (1.66, 3.17, 700, 1300),
    "g395m": (2.87, 5.27, 700, 1300),
    "g140h": (0.95, 1.89, 1900, 3600),
    "g235h": (1.66, 3.17, 1900, 3600),
    "g395h": (2.87, 5.27, 1900, 3600),
    "prism": (0.60, 5.30, 30, 300),
}

#: NIRSpec PRISM resolving power, tabulated (wavelength um, R). The prism R
#: dips to ~30 near 1.2 um and climbs steeply to the red; a linear bracket
#: would be wrong by a factor of several, which is why it gets its own table.
_PRISM_R = np.array([
    [0.60, 38], [0.70, 48], [0.80, 56], [0.90, 59], [1.00, 51],
    [1.10, 41], [1.20, 33], [1.30, 31], [1.40, 34], [1.60, 43],
    [1.80, 53], [2.00, 63], [2.50, 89], [3.00, 117], [3.50, 148],
    [4.00, 181], [4.50, 219], [5.00, 262], [5.30, 288],
])

#: MIRI LRS: slitless/slit low resolution spectroscopy, R ~ 40-160 over
#: 5-12 um, rising with wavelength.
_LRS_R = np.array([[5.0, 40], [7.5, 75], [10.0, 110], [12.0, 160]])


def band_of(wavelength_um: float, instrument: str | None = None) -> str | None:
    """Return the name of the JWST band covering ``wavelength_um``.

    With ``instrument=None`` the MRS bands are searched first and the NIRSpec
    gratings second, which is unambiguous above 5.3 um and below 4.9 um but
    overlaps in between -- pass ``instrument='miri'`` or ``'nirspec'`` when the
    answer matters in the 4.9-5.3 um region where NIRSpec G395 and MRS ch1
    both have coverage.
    """
    wavelength_um = float(wavelength_um)
    pools = []
    if instrument is None or instrument.lower().startswith("miri"):
        pools.append(MRS_BANDS)
    if instrument is None or instrument.lower().startswith("nirspec"):
        pools.append(NIRSPEC_BANDS)
    for pool in pools:
        for name, (lo, hi, _, _) in pool.items():
            if lo <= wavelength_um <= hi:
                return name
    return None


def resolving_power(wavelength_um, mode: str | None = None):
    """Resolving power R = lambda / dlambda at the given observed wavelength.

    Parameters
    ----------
    wavelength_um : float or array
        Observed-frame wavelength in microns. Rest-frame wavelengths give the
        wrong answer: R is a property of the spectrograph, which sees photons
        in the observed frame.
    mode : str, optional
        One of the keys of :data:`MRS_BANDS` or :data:`NIRSPEC_BANDS`, or
        ``'prism'``, ``'lrs'``, ``'mrs'``, ``'nirspec-m'``, ``'nirspec-h'``.
        If omitted, the band is inferred per-wavelength from
        :func:`band_of`, which is the right default for a *stitched* spectrum
        that spans several bands.

    Returns
    -------
    float or ndarray
        R, or NaN where no band covers the wavelength and none was forced.

    Notes
    -----
    This is what converts a fitted line width in the fitter into a *physical*
    velocity: an observed sigma must have the instrumental sigma
    (``c / (R * 2.3548)`` for a Gaussian LSF quoted as FWHM) removed in
    quadrature before it means anything about the source. A fit that skips
    that step will report a floor of a few tens of km/s for MRS and a couple
    of hundred for the NIRSpec medium gratings no matter how narrow the line
    really is.
    """
    w = np.atleast_1d(np.asarray(wavelength_um, dtype=float))
    out = np.full(w.shape, np.nan)

    if mode is not None:
        key = mode.strip().lower()
        if key == "prism":
            out = np.interp(w, _PRISM_R[:, 0], _PRISM_R[:, 1],
                            left=_PRISM_R[0, 1], right=_PRISM_R[-1, 1])
            return float(out[0]) if np.isscalar(wavelength_um) else out
        if key in ("lrs", "miri-lrs", "lrs-slit", "lrs-slitless"):
            out = np.interp(w, _LRS_R[:, 0], _LRS_R[:, 1],
                            left=_LRS_R[0, 1], right=_LRS_R[-1, 1])
            return float(out[0]) if np.isscalar(wavelength_um) else out
        if key in ("nirspec-m", "medium", "m"):
            out = np.interp(w, [0.9, 5.3], [700, 1300])
            return float(out[0]) if np.isscalar(wavelength_um) else out
        if key in ("nirspec-h", "high", "h"):
            out = np.interp(w, [0.9, 5.3], [1900, 3600])
            return float(out[0]) if np.isscalar(wavelength_um) else out
        if key == "mrs":
            mode = None  # fall through to per-wavelength MRS band lookup
        else:
            pool = {**MRS_BANDS, **NIRSPEC_BANDS}
            if key in pool:
                lo, hi, r_lo, r_hi = pool[key]
                frac = np.clip((w - lo) / (hi - lo), 0.0, 1.0)
                out = r_lo + frac * (r_hi - r_lo)
                return float(out[0]) if np.isscalar(wavelength_um) else out
            raise KeyError(f"unknown mode {mode!r}; known: {sorted(pool) + ['prism', 'lrs', 'mrs']}")

    for i, wi in enumerate(w):
        name = band_of(wi)
        if name is None:
            continue
        pool = MRS_BANDS if name in MRS_BANDS else NIRSPEC_BANDS
        if name == "prism":
            out[i] = np.interp(wi, _PRISM_R[:, 0], _PRISM_R[:, 1])
            continue
        lo, hi, r_lo, r_hi = pool[name]
        out[i] = r_lo + (wi - lo) / (hi - lo) * (r_hi - r_lo)
    return float(out[0]) if np.isscalar(wavelength_um) else out


def instrumental_sigma_kms(wavelength_um, mode: str | None = None):
    """Instrumental line-spread 1-sigma width in km/s.

    Assumes a Gaussian LSF whose FWHM is ``c / R``, so
    ``sigma = c / (R * 2*sqrt(2*ln2))``. This is the quantity to subtract in
    quadrature from a fitted Gaussian sigma, and the quantity to convolve a
    model with when fitting a line that is close to unresolved.
    """
    r = resolving_power(wavelength_um, mode=mode)
    return C_KMS / (np.asarray(r, dtype=float) * 2.0 * np.sqrt(2.0 * np.log(2.0)))


# ---------------------------------------------------------------------------
# air / vacuum conversions
# ---------------------------------------------------------------------------
def _n_air(wave_um):
    """Refractive index of standard air (Ciddor 1996), wavelength in um."""
    s2 = (1.0 / np.asarray(wave_um, dtype=float)) ** 2
    return 1.0 + 0.05792105 / (238.0185 - s2) + 0.00167917 / (57.362 - s2)


def air_to_vac(wave_um):
    """Convert air wavelengths (um) to vacuum. Below 0.2 um the formula is
    extrapolated and the result should not be trusted, but nothing in the
    JWST range is anywhere near that."""
    w = np.asarray(wave_um, dtype=float)
    return w * _n_air(w)


def vac_to_air(wave_um):
    """Convert vacuum wavelengths (um) to air -- the inverse of
    :func:`air_to_vac`, iterated once, which is already converged to far
    better than any JWST wavelength solution."""
    w = np.asarray(wave_um, dtype=float)
    return w / _n_air(w / _n_air(w))


# ---------------------------------------------------------------------------
# the container
# ---------------------------------------------------------------------------
@dataclass
class Spectrum:
    """A 1-D spectrum on a strictly increasing, vacuum, observed-frame,
    micron wavelength grid.

    Attributes
    ----------
    wavelength : ndarray
        Observed-frame vacuum wavelength in microns, strictly increasing.
    flux : ndarray
    flux_err : ndarray
        1-sigma uncertainty, same unit as ``flux``.
    mask : ndarray of bool
        True means *masked out* (do not use), matching numpy masked-array
        convention. Points are masked, never deleted -- see module docstring.
    redshift : float
        Used by :attr:`rest_wavelength` and by everything that compares
        against a rest-frame line list. Zero means "already rest frame or
        don't care".
    flux_unit : str
    segment : ndarray of str or None
        Per-point label of which instrument segment the point came from,
        populated by :func:`jspec.stitch.stitch`. This is what makes it
        possible, after the fact, to ask "is this feature real or is it
        sitting exactly on the ch2-long/ch3-short join?".
    meta : dict
    """

    wavelength: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    mask: np.ndarray | None = None
    redshift: float = 0.0
    flux_unit: str = "Jy"
    segment: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    # -- construction -------------------------------------------------------
    def __post_init__(self):
        self.wavelength = np.asarray(self.wavelength, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        if self.flux_err is None:
            self.flux_err = np.full(self.flux.shape, np.nan)
        self.flux_err = np.asarray(self.flux_err, dtype=float)

        n = self.wavelength.size
        if not (self.flux.size == n and self.flux_err.size == n):
            raise ValueError(
                f"length mismatch: wavelength={n}, flux={self.flux.size}, "
                f"flux_err={self.flux_err.size}")

        if self.mask is None:
            self.mask = np.zeros(n, dtype=bool)
        else:
            self.mask = np.asarray(self.mask, dtype=bool)
        if self.segment is not None:
            self.segment = np.asarray(self.segment, dtype=object)

        # Anything non-finite in wavelength can never be used and would break
        # every searchsorted/interp downstream, so those rows really are
        # dropped rather than masked.
        finite_w = np.isfinite(self.wavelength)
        if not finite_w.all():
            self._take(finite_w)

        # A non-finite flux is masked, not dropped: the wavelength is still
        # evidence of coverage.
        self.mask |= ~np.isfinite(self.flux)

        if self.wavelength.size and not np.all(np.diff(self.wavelength) > 0):
            self._sort_and_dedupe()

    def _take(self, idx):
        self.wavelength = self.wavelength[idx]
        self.flux = self.flux[idx]
        self.flux_err = self.flux_err[idx]
        self.mask = self.mask[idx]
        if self.segment is not None:
            self.segment = self.segment[idx]

    def _sort_and_dedupe(self):
        """Sort by wavelength and collapse exactly-duplicated wavelengths.

        Duplicates are real in this project: concatenating two bands that
        share an overlap, or an x1d that repeats a pixel, both produce them,
        and a non-monotonic grid silently corrupts ``np.interp`` (which
        assumes sorted input and returns nonsense otherwise rather than
        raising). Duplicates are combined by inverse-variance average so no
        signal is thrown away.
        """
        order = np.argsort(self.wavelength, kind="mergesort")
        self._take(order)
        w = self.wavelength
        if w.size < 2:
            return
        dup = np.isclose(np.diff(w), 0.0, rtol=0.0, atol=0.0)
        if not dup.any():
            return
        # group boundaries: a new group starts where the wavelength changes
        new_group = np.concatenate([[True], ~dup])
        gid = np.cumsum(new_group) - 1
        ngroups = gid[-1] + 1

        wv = np.zeros(ngroups)
        fl = np.zeros(ngroups)
        er = np.zeros(ngroups)
        mk = np.zeros(ngroups, dtype=bool)
        seg = np.empty(ngroups, dtype=object) if self.segment is not None else None
        for g in range(ngroups):
            sel = gid == g
            wv[g] = self.wavelength[sel][0]
            good = sel & ~self.mask & np.isfinite(self.flux_err) & (self.flux_err > 0)
            if good.any():
                wgt = 1.0 / self.flux_err[good] ** 2
                fl[g] = np.sum(self.flux[good] * wgt) / np.sum(wgt)
                er[g] = np.sqrt(1.0 / np.sum(wgt))
                mk[g] = False
            else:
                fl[g] = np.nanmean(self.flux[sel])
                er[g] = np.nanmean(self.flux_err[sel])
                mk[g] = True
            if seg is not None:
                seg[g] = self.segment[sel][0]
        self.wavelength, self.flux, self.flux_err, self.mask = wv, fl, er, mk
        self.segment = seg

    # -- basic views --------------------------------------------------------
    def __len__(self):
        return int(self.wavelength.size)

    def __repr__(self):
        if len(self) == 0:
            return "<Spectrum empty>"
        return (f"<Spectrum {len(self)} pts "
                f"{self.wavelength[0]:.3f}-{self.wavelength[-1]:.3f} um, "
                f"{(~self.mask).sum()} good, z={self.redshift:g}, "
                f"unit={self.flux_unit}>")

    @property
    def good(self) -> np.ndarray:
        """Boolean "use this point": unmasked and finite in both flux and
        error, with a strictly positive error (a zero error is a division by
        zero waiting to happen and always means the error was never filled
        in, not that the point is infinitely precise)."""
        return (~self.mask
                & np.isfinite(self.flux)
                & np.isfinite(self.flux_err)
                & (self.flux_err > 0))

    @property
    def rest_wavelength(self) -> np.ndarray:
        """Rest-frame wavelength in microns, ``lambda_obs / (1 + z)``."""
        return self.wavelength / (1.0 + self.redshift)

    @property
    def snr(self) -> np.ndarray:
        """Per-point |flux| / error, NaN where the error is unusable."""
        out = np.full(self.flux.shape, np.nan)
        ok = np.isfinite(self.flux) & np.isfinite(self.flux_err) & (self.flux_err > 0)
        out[ok] = np.abs(self.flux[ok]) / self.flux_err[ok]
        return out

    def copy(self) -> "Spectrum":
        return Spectrum(
            self.wavelength.copy(), self.flux.copy(), self.flux_err.copy(),
            mask=self.mask.copy(), redshift=self.redshift,
            flux_unit=self.flux_unit,
            segment=None if self.segment is None else self.segment.copy(),
            meta=dict(self.meta),
        )

    def with_redshift(self, z: float) -> "Spectrum":
        """Return a copy at a different redshift. Cheap -- nothing is
        recomputed, because ``wavelength`` stays observed-frame."""
        out = self.copy()
        out.redshift = float(z)
        return out

    # -- selection ----------------------------------------------------------
    def select(self, lo: float, hi: float, rest: bool = False) -> "Spectrum":
        """Return the sub-spectrum within ``[lo, hi]`` microns.

        ``rest=True`` interprets the bounds in the rest frame, which is what
        you want when slicing around a line from a rest-frame line list.
        """
        w = self.rest_wavelength if rest else self.wavelength
        sel = (w >= lo) & (w <= hi)
        out = self.copy()
        out._take(sel)
        return out

    def masked_where(self, condition) -> "Spectrum":
        """Return a copy with ``condition`` (a boolean array over points)
        added to the mask."""
        out = self.copy()
        out.mask = out.mask | np.asarray(condition, dtype=bool)
        return out

    def mask_ranges(self, ranges: Iterable[Sequence[float]], rest: bool = False) -> "Spectrum":
        """Mask every point inside any of the given ``(lo, hi)`` wavelength
        ranges. Used to exclude known-bad planes or strong host lines without
        changing the array length."""
        out = self.copy()
        w = out.rest_wavelength if rest else out.wavelength
        for lo, hi in ranges:
            out.mask |= (w >= lo) & (w <= hi)
        return out

    def mask_red_end(self, lam_min: float = None) -> "Spectrum":
        """Mask the MIRI MRS channel-4-long red end.

        Past roughly 27.5 um the MRS throughput calibration breaks down: the
        flux rises steeply in a way that is not astrophysical, a growing
        fraction of points go negative, and the median S/N collapses. See
        :data:`MRS_RED_END_UM` for the measured numbers behind the default
        cutoff.

        The cut is applied in the **observed** frame, not the rest frame,
        because this is a property of the instrument and has nothing to do
        with the source. Applying it in the rest frame would move the cutoff
        around with redshift, which would be wrong in a way that is very easy
        not to notice.

        The points are masked, not deleted, so the array length and any index
        you are holding stay valid, and you can always inspect what was
        removed via ``~spectrum.good``.

        Parameters
        ----------
        lam_min : float, optional
            Observed wavelength in um beyond which to mask. Defaults to
            :data:`MRS_RED_END_UM`. Pass ``np.inf`` to disable.
        """
        lam = MRS_RED_END_UM if lam_min is None else float(lam_min)
        out = self.copy()
        hit = out.wavelength > lam
        if hit.any():
            out.mask |= hit
            out.meta = dict(out.meta)
            out.meta["RED_END_MASK_UM"] = lam
            out.meta["RED_END_MASKED_N"] = int(hit.sum())
        return out

    def compressed(self) -> "Spectrum":
        """Return a copy containing only the :attr:`good` points. Use
        sparingly -- see module docstring on why masking beats deleting."""
        out = self.copy()
        out._take(out.good)
        return out

    # -- unit handling ------------------------------------------------------
    def to_flam(self) -> "Spectrum":
        """Convert Jy -> erg s^-1 cm^-2 Angstrom^-1.

        ``F_lambda = F_nu * c / lambda^2``; with F_nu in Jy (1e-23 erg/s/cm2/Hz)
        and lambda in microns this is ``F_lambda = 2.99792458e-5 * F_nu /
        lambda_um^2`` in erg/s/cm2/Angstrom.
        """
        if self.flux_unit.lower().startswith("erg"):
            return self.copy()
        if self.flux_unit.lower() not in ("jy", "jansky"):
            raise ValueError(f"cannot convert from {self.flux_unit!r}; expected Jy")
        k = 2.99792458e-5 / self.wavelength ** 2
        out = self.copy()
        out.flux = self.flux * k
        out.flux_err = self.flux_err * k
        out.flux_unit = "erg / (s cm2 Angstrom)"
        return out

    def to_jy(self) -> "Spectrum":
        """Inverse of :meth:`to_flam`."""
        if self.flux_unit.lower() in ("jy", "jansky"):
            return self.copy()
        k = self.wavelength ** 2 / 2.99792458e-5
        out = self.copy()
        out.flux = self.flux * k
        out.flux_err = self.flux_err * k
        out.flux_unit = "Jy"
        return out

    def velocity(self, rest_wavelength_um: float) -> np.ndarray:
        """Relativistic-free (optical convention) velocity axis in km/s
        relative to a rest-frame line centre, computed in the rest frame.

        The optical convention ``v = c (lambda - lambda_0) / lambda_0`` is
        used because it is what line-profile work in this wavelength range
        conventionally quotes; over the few x 10^4 km/s relevant to SN ejecta
        the difference from the relativistic form is a percent-level effect
        that matters only if you are comparing to a relativistically-defined
        measurement.
        """
        return C_KMS * (self.rest_wavelength - rest_wavelength_um) / rest_wavelength_um

    def resolving_power(self, mode: str | None = None) -> np.ndarray:
        """Per-point instrument resolving power -- see
        :func:`resolving_power`. ``mode`` defaults to per-point band
        inference, correct for a stitched multi-band spectrum."""
        return resolving_power(self.wavelength, mode=mode)

    # -- resampling ---------------------------------------------------------
    def interp_to(self, new_wave, fill=np.nan) -> "Spectrum":
        """Linearly interpolate onto a new wavelength grid.

        Errors are interpolated too, which *understates* the error on a grid
        finer than the original (neighbouring interpolated points become
        correlated). That is acceptable for plotting and for ratio-taking in
        an overlap, and is why :func:`jspec.stitch.stitch` never resamples the
        science spectrum onto a finer grid than it started on.
        """
        new_wave = np.asarray(new_wave, dtype=float)
        g = self.good
        if g.sum() < 2:
            raise ValueError("need at least 2 good points to interpolate")
        f = np.interp(new_wave, self.wavelength[g], self.flux[g], left=fill, right=fill)
        e = np.interp(new_wave, self.wavelength[g], self.flux_err[g], left=fill, right=fill)
        # Anything outside the original coverage is masked, not silently
        # extrapolated at the edge value.
        outside = (new_wave < self.wavelength[g][0]) | (new_wave > self.wavelength[g][-1])
        return Spectrum(new_wave, f, e, mask=outside, redshift=self.redshift,
                        flux_unit=self.flux_unit, meta=dict(self.meta))

    def rebin(self, factor: int) -> "Spectrum":
        """Inverse-variance bin by an integer ``factor``.

        Unlike interpolation this genuinely reduces the noise, and the output
        errors are honest, because each output point is built from disjoint
        input points. Use it to make a low-S/N continuum visible without
        pretending the original sampling was coarser than it was.
        """
        factor = int(factor)
        if factor < 2:
            return self.copy()
        n = (len(self) // factor) * factor
        w = self.wavelength[:n].reshape(-1, factor)
        f = self.flux[:n].reshape(-1, factor)
        e = self.flux_err[:n].reshape(-1, factor)
        g = self.good[:n].reshape(-1, factor)

        wgt = np.where(g, 1.0 / np.where(e > 0, e, np.inf) ** 2, 0.0)
        swgt = wgt.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            fo = np.where(swgt > 0, (f * wgt).sum(axis=1) / swgt, np.nan)
            eo = np.where(swgt > 0, np.sqrt(1.0 / swgt), np.nan)
        wo = w.mean(axis=1)
        return Spectrum(wo, fo, eo, mask=~(swgt > 0), redshift=self.redshift,
                        flux_unit=self.flux_unit, meta=dict(self.meta))

    # -- output -------------------------------------------------------------
    def to_table(self, lowercase: bool = False, include_rest: bool = True) -> Table:
        """Build an astropy Table. Column case follows this project's two
        conventions: uppercase ``WAVELENGTH/FLUX/FLUX_ERROR`` by default (what
        ``combine_epoch_spectra`` writes), lowercase on request (what
        ``nirspec_fs_auto_pipeline`` writes)."""
        names = (("wavelength", "flux", "flux_err")
                 if lowercase else ("WAVELENGTH", "FLUX", "FLUX_ERROR"))
        t = Table({names[0]: self.wavelength, names[1]: self.flux,
                   names[2]: self.flux_err})
        t[names[0]].unit = "um"
        t[names[1]].unit = self.flux_unit
        t[names[2]].unit = self.flux_unit
        if include_rest and self.redshift:
            t["REST_WAVELENGTH" if not lowercase else "rest_wavelength"] = self.rest_wavelength
        t["MASK" if not lowercase else "mask"] = self.mask.astype(np.uint8)
        if self.segment is not None:
            t["SEGMENT" if not lowercase else "segment"] = [str(s) for s in self.segment]
        t.meta.update(self.meta)
        if self.redshift:
            t.meta["REDSHIFT"] = self.redshift
        return t

    def write(self, path: str, lowercase: bool = False, overwrite: bool = True):
        """Write to ``.ecsv`` (preserves meta and units), ``.csv``, ``.txt``
        or ``.fits`` based on the extension."""
        t = self.to_table(lowercase=lowercase)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".ecsv":
            t.write(path, format="ascii.ecsv", overwrite=overwrite)
        elif ext == ".csv":
            t.write(path, format="csv", overwrite=overwrite)
        elif ext in (".txt", ".dat"):
            t.write(path, format="ascii.commented_header", overwrite=overwrite)
        elif ext in (".fits", ".fit"):
            t.write(path, format="fits", overwrite=overwrite)
        else:
            raise ValueError(f"unsupported output extension {ext!r}")
        return path


@dataclass
class Segment:
    """One instrument segment (an MRS sub-band, a NIRSpec grating setting, a
    single x1d) awaiting stitching.

    ``name`` is what shows up in stitching diagnostics and in
    :attr:`Spectrum.segment`, so it should be something you can act on --
    ``'ch2-long'``, ``'g395m'`` -- not a bare file index.
    """
    name: str
    spectrum: Spectrum
    path: str | None = None

    @property
    def wmin(self) -> float:
        g = self.spectrum.good
        return float(self.spectrum.wavelength[g].min()) if g.any() else np.nan

    @property
    def wmax(self) -> float:
        g = self.spectrum.good
        return float(self.spectrum.wavelength[g].max()) if g.any() else np.nan

    def __repr__(self):
        return f"<Segment {self.name}: {self.wmin:.3f}-{self.wmax:.3f} um, {len(self.spectrum)} pts>"


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------
def _pick(colnames, candidates):
    lower = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand in colnames:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _table_to_spectrum(t: Table, redshift=0.0, err_column=None,
                       apply_flags=True, path=None, mask_red_end=True) -> Spectrum:
    wcol = _pick(t.colnames, _WAVE_KEYS)
    fcol = _pick(t.colnames, _FLUX_KEYS)
    if wcol is None or fcol is None:
        raise KeyError(f"could not find wavelength/flux columns in {t.colnames}")
    ecol = err_column or _pick(t.colnames, _ERR_KEYS)

    wave = np.asarray(t[wcol], dtype=float)
    flux = np.asarray(t[fcol], dtype=float)
    err = (np.asarray(t[ecol], dtype=float) if ecol is not None
           else np.full(flux.shape, np.nan))

    # A wavelength column in Angstrom or nm would silently put everything in
    # the wrong band; normalize on the declared unit if there is one, and
    # otherwise on magnitude, since no JWST spectrum spans 0.6-30 *anything
    # else*.
    unit = str(getattr(t[wcol], "unit", "") or "").lower()
    if unit in ("angstrom", "a", "aa"):
        wave = wave * 1e-4
    elif unit in ("nm", "nanometer", "nanometre"):
        wave = wave * 1e-3
    elif unit in ("m", "meter", "metre"):
        wave = wave * 1e6
    elif not unit and np.nanmedian(wave) > 100:
        warnings.warn(
            f"wavelength column {wcol!r} has no unit and a median of "
            f"{np.nanmedian(wave):.4g}; assuming Angstrom and converting to um",
            stacklevel=2)
        wave = wave * 1e-4

    mask = np.zeros(flux.shape, dtype=bool)
    if apply_flags and "FLAG" in t.colnames:
        mask |= (np.asarray(t["FLAG"], dtype=int) & MRS_FLAG_BAD) != 0
    if "MASK" in t.colnames:
        mask |= np.asarray(t["MASK"], dtype=bool)
    elif "mask" in t.colnames:
        mask |= np.asarray(t["mask"], dtype=bool)
    if "DQ" in t.colnames:
        mask |= np.asarray(t["DQ"], dtype=int) != 0

    seg = None
    for c in ("SEGMENT", "segment"):
        if c in t.colnames:
            seg = np.asarray([str(s) for s in t[c]], dtype=object)
            break

    meta = dict(t.meta)
    if path:
        meta.setdefault("SOURCE_FILE", os.path.basename(path))
    z = redshift if redshift else float(meta.get("REDSHIFT", 0.0) or 0.0)

    funit = str(getattr(t[fcol], "unit", "") or "Jy")
    sp = Spectrum(wave, flux, err, mask=mask, redshift=z,
                  flux_unit=funit, segment=seg, meta=meta)
    if mask_red_end:
        sp = sp.mask_red_end(
            None if mask_red_end is True else mask_red_end)
    return sp


def read_x1d(path, extname=None, dq_reject=True, redshift=0.0,
             order_by_snr=True, mask_red_end=True) -> Spectrum:
    """Read a JWST ``x1d`` FITS product.

    Parameters
    ----------
    extname : str or int, optional
        Which EXTRACT1D extension to read. By default, when a file has
        several (multi-slit, or multiple integrations), the one with the
        highest median S/N is taken if ``order_by_snr`` else the first --
        taking the first blindly is how an empty background slit ends up
        masquerading as the science spectrum.
    dq_reject : bool
        Mask points with non-zero DQ rather than dropping them.
    """
    with fits.open(path, memmap=False) as hdul:
        cands = []
        for i, hdu in enumerate(hdul):
            if getattr(hdu, "data", None) is None or not hasattr(hdu, "columns"):
                continue
            names = [c.upper() for c in hdu.columns.names]
            if "WAVELENGTH" in names and "FLUX" in names:
                cands.append(i)
        if not cands:
            raise ValueError(f"no EXTRACT1D-like table extension in {path}")

        if extname is not None:
            idx = (extname if isinstance(extname, int)
                   else hdul.index_of(extname))
        elif len(cands) == 1 or not order_by_snr:
            idx = cands[0]
        else:
            best, best_snr = cands[0], -np.inf
            for i in cands:
                d = hdul[i].data
                f = np.asarray(d["FLUX"], dtype=float)
                e = (np.asarray(d["FLUX_ERROR"], dtype=float)
                     if "FLUX_ERROR" in [c.upper() for c in hdul[i].columns.names]
                     else np.full(f.shape, np.nan))
                ok = np.isfinite(f) & np.isfinite(e) & (e > 0)
                s = np.nanmedian(np.abs(f[ok]) / e[ok]) if ok.any() else -np.inf
                if np.isfinite(s) and s > best_snr:
                    best, best_snr = i, s
            idx = best

        d = hdul[idx].data
        names = {c.upper(): c for c in hdul[idx].columns.names}
        wave = np.asarray(d[names["WAVELENGTH"]], dtype=float)
        flux = np.asarray(d[names["FLUX"]], dtype=float)
        ecol = next((names[k] for k in ("FLUX_ERROR", "ERROR", "FLUX_ERR")
                     if k in names), None)
        err = (np.asarray(d[ecol], dtype=float) if ecol
               else np.full(flux.shape, np.nan))
        mask = np.zeros(flux.shape, dtype=bool)
        if dq_reject and "DQ" in names:
            mask |= np.asarray(d[names["DQ"]], dtype=int) != 0

        hdr0 = hdul[0].header
        meta = {k: hdr0[k] for k in
                ("TARGPROP", "TARGNAME", "INSTRUME", "DETECTOR", "FILTER",
                 "GRATING", "CHANNEL", "BAND", "MJD-OBS", "EXPSTART", "DATE-OBS")
                if k in hdr0}
        meta["SOURCE_FILE"] = os.path.basename(path)

    sp = Spectrum(wave, flux, err, mask=mask, redshift=redshift,
                  flux_unit="Jy", meta=meta)
    if mask_red_end:
        sp = sp.mask_red_end(None if mask_red_end is True else mask_red_end)
    return sp


def read_spectrum(path, redshift=0.0, err_column=None, apply_flags=True,
                  mask_red_end=True, **kwargs) -> Spectrum:
    """Read a spectrum from any format this project produces.

    Dispatches on extension: ``.fits`` goes to :func:`read_x1d` if it looks
    like an x1d and to the generic table reader otherwise; ``.ecsv``,
    ``.csv``, ``.txt``, ``.dat`` go to the generic reader.

    Parameters
    ----------
    redshift : float
        Stored on the result; overrides a ``REDSHIFT`` key in the file meta.
    err_column : str, optional
        Force a particular error column. Mainly for MIRI ``_full.ecsv``
        files, where ``FLUX_ERR_TOTAL`` (default) folds in the systematic
        term and ``FLUX_ERROR`` is random-only. Use ``FLUX_ERROR`` when you
        are about to inverse-variance weight neighbouring points against each
        other, because the systematic term is correlated between neighbours
        and including it in a per-point weight overstates how independent
        adjacent points are.
    apply_flags : bool
        Honour a MIRI ``FLAG`` bitmask column (see :data:`MRS_FLAG_BAD`).
    mask_red_end : bool or float
        Mask the MIRI MRS channel-4-long red end, where the throughput
        calibration breaks down (see :meth:`Spectrum.mask_red_end` and
        :data:`MRS_RED_END_UM`). ``True`` (the default) uses the standard
        27.5 um cutoff; pass a float to set your own, or ``False`` to keep
        the data. The points are masked, not removed, so you can always see
        what went via ``~spectrum.good``.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fits", ".fit"):
        try:
            return read_x1d(path, redshift=redshift,
                            mask_red_end=mask_red_end, **kwargs)
        except (ValueError, KeyError):
            t = Table.read(path)
            return _table_to_spectrum(t, redshift, err_column, apply_flags,
                                      path, mask_red_end)
    if ext == ".ecsv":
        t = Table.read(path, format="ascii.ecsv")
    elif ext == ".csv":
        t = Table.read(path, format="csv")
    else:
        t = Table.read(path, format="ascii")
    return _table_to_spectrum(t, redshift, err_column, apply_flags, path,
                              mask_red_end)


_SEGMENT_PATTERNS = (
    # MRS: ...ch2-long... or ...ch2_long... or ...ch2short...
    (re.compile(r"ch([1-4])[-_ ]?(short|medium|long|a|b|c)", re.I),
     lambda m: f"ch{m.group(1)}-" + {"a": "short", "b": "medium", "c": "long"}.get(
         m.group(2).lower(), m.group(2).lower())),
    # NIRSpec gratings
    (re.compile(r"(g[12][34]0[mh]|g395[mh]|prism)", re.I),
     lambda m: m.group(1).lower()),
)


def segment_name_from_path(path, fallback=None) -> str:
    """Guess a segment label from a filename. Recognises MRS ``chN-band`` in
    any of the spellings this project's products use, and NIRSpec grating
    names. Falls back to the basename without extension, which at least stays
    unique."""
    base = os.path.basename(path)
    for pat, fmt in _SEGMENT_PATTERNS:
        m = pat.search(base)
        if m:
            return fmt(m)
    return fallback or os.path.splitext(base)[0]


def read_segments(paths_or_glob, redshift=0.0, names=None, **kwargs) -> list[Segment]:
    """Read many files into :class:`Segment` objects, sorted blue to red.

    ``paths_or_glob`` may be a glob string or an iterable of paths. Segment
    names are taken from ``names`` if given, else guessed with
    :func:`segment_name_from_path`. Files that fail to read are reported and
    skipped rather than aborting the run -- with a dozen MRS sub-bands, one
    unreadable file should not cost you the other eleven.
    """
    if isinstance(paths_or_glob, str):
        paths = sorted(_glob.glob(paths_or_glob))
        if not paths:
            raise FileNotFoundError(f"no files match {paths_or_glob!r}")
    else:
        paths = list(paths_or_glob)

    segs = []
    for i, p in enumerate(paths):
        try:
            sp = read_spectrum(p, redshift=redshift, **kwargs)
        except Exception as exc:       # noqa: BLE001 - deliberate: keep going
            warnings.warn(f"skipping {p}: {type(exc).__name__}: {exc}", stacklevel=2)
            continue
        if not sp.good.any():
            warnings.warn(f"skipping {p}: no usable points", stacklevel=2)
            continue
        name = names[i] if names is not None else segment_name_from_path(p)
        segs.append(Segment(name=name, spectrum=sp, path=p))

    segs.sort(key=lambda s: s.wmin)
    return segs
