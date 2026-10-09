#!/usr/bin/env python3
"""
miri_mrs_refine_extract.py
==========================

One-shot replacement for refining.ipynb.  Input: the Stage-3 cube from
miri_mrs_auto_pipeline.py (and optionally the first-pass *_abi_bgsub_s3d.fits,
used ONLY to find the position).  Output: a cleaned, flagged 1D spectrum with
calibrated errors, plus QA figures and a JSON report of every decision.

Every design choice below traces to one of the diagnostic plots:

 plot(s)            what it showed                         what this script does
 -----------------  -------------------------------------  ----------------------------------------
 01/01b/01c/01d     mesh ON/OFF give the same position;    keep mesh ON for detection (SNR 116 vs
                    mesh background at the source ~ -0.1   67); position is not mesh-sensitive
 02/18              y sub-band scatter (0.25 px) is ONE    robust position error: drop the outlier
                    outlier band (6.5 um); x drifts 0.06   band, report drift; adopt full-collapse fit
 03b                COM / PSF fit / centroid agree         used as the cross-check only
 06                 ΔBIC=93 but sep=0.66 FWHM, and an      companion test masks the neighbour knot,
                    unmodelled knot (~53,47) sits inside   repeats the 2-PSF fit per sub-band and asks
                    the fit cutout and shows in residuals  whether separation is constant in arcsec
                                                           (physical) or scales with FWHM (PSF artefact)
 15                 flux still rising at fwhm_scale=3.5    pick mask radius from a bias/variance
                    (0.935 -> 0.962 -> 1.0) while moat and criterion on a hole-fill test, not from a
                    sky levels are FLAT vs fwhm_scale      fixed number; quote the residual wing bias
 11/15/22           small negative moat/sky offset that    robust (sigma-clipped) polynomial: bright
                    does not depend on mask size => the    host knots no longer drag the LSQ surface
                    LSQ polynomial is pulled up by knots   up; static knot mask + per-slice clipping
 16                 other host knots in the field          masked natively (no ERR-inflation hack)
 17/22              polynomial ~ local annulus median      hole-fill test replaces "dumb estimate":
                    except in ch4                          it measures bias AND interpolation error
 10/05              bg reaches 1500 MJy/sr in ch4 and the  mask radius is capped so >=45% of valid
                    mask (~3.5 FWHM = 30 px) eats the      pixels always remain; slices where the cap
                    whole footprint                        bites are FLAGGED, not trusted
 19 / 13            spikes at ~12.86 and ~17.08 um         flagged as HOSTLINE (they sit at [NeII] and
                                                           H2 S(1) for a host at z~0.003), not deleted
 09 / 14            "dropouts" at 5-6.5 um: the source     no aperture change is made on that basis;
                    continuum is ~5e-5 Jy = noise level    rscale is chosen from an S/N-vs-radius test
 10b/20/21          empirical/formal error ratio is not    ratio measured with hole-filled fake
                    flat (0.6 ... 4.3), rises in 2C, 3C    apertures, smoothed, floored at 1.0
 (rev2)             chosen (k,f) sat at the grid floor in  tune_background() now requires
                    every channel; flux was biased low     wing_frac <= wing_tol as a hard pre-filter
                    because the old combined metric let    (bias doesn't average down like scatter
                    scatter (which DOES average down)      does, so it can't be traded off against it),
                    outvote wing bias, and the smallest-   then picks the lowest-scatter (k,f) among
                    (k,f) tie-break made it worse; the     survivors. The resulting wing_frac is then
                    measured bias was never applied to     divided back out of the flux in finalize(),
                    the flux, only to the error budget.    not just folded into e_sys.

Usage
-----
    python miri_mrs_refine_extract.py \\
        --cube  .../stage3/2024aecx_..._s3d.fits \\
        --centroid-cube .../bgsub/2024aecx_..._abi_bgsub_s3d.fits \\
        --out-dir refined/ep1 --name SN2024aecx_ep1 --x0 44.8 --y0 50 --z 0.0027

    # or give a sky position instead of a pixel guess (converted via the
    # centroid/stage-3 cube's WCS, then used exactly like --x0/--y0):
    python miri_mrs_refine_extract.py \\
        --cube  .../stage3/2024aecx_..._s3d.fits \\
        --centroid-cube .../bgsub/2024aecx_..._abi_bgsub_s3d.fits \\
        --out-dir refined/ep1 --name SN2024aecx_ep1 --ra 10.6847 --dec 41.2687 --z 0.0027

or from a notebook:  from miri_mrs_refine_extract import Config, run; run(Config(...))

Stages are cached in <out-dir>/state.json; --force bg,tune,errors,... redoes them.
Run from a real script / notebook (jwst Extract1dStep is used for the calibrated flux).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict, field
import re

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage as ndi
from numpy.polynomial import legendre as npleg

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
# MRS sub-band coverage (um), JDox; edges are approximate to ~0.02 um.
MRS_SUBBANDS = {
    "1A": (4.90, 5.74), "1B": (5.66, 6.63), "1C": (6.53, 7.65),
    "2A": (7.51, 8.77), "2B": (8.67, 10.13), "2C": (10.02, 11.70),
    "3A": (11.55, 13.47), "3B": (13.34, 15.57), "3C": (15.41, 17.98),
    "4A": (17.70, 20.95), "4B": (20.69, 24.48), "4C": (24.40, 28.70),
}
CHANNEL_SPLITS = (7.58, 11.62, 17.84)          # mid-overlap 1|2, 2|3, 3|4
_names = list(MRS_SUBBANDS)
BAND_BOUNDS = [0.5 * (MRS_SUBBANDS[b][0] + MRS_SUBBANDS[a][1]) for a, b in zip(_names[:-1], _names[1:])]

# rest wavelengths (um) of lines that show up in host emission
HOST_LINES_UM = {
    "H2 S(7)": 5.511, "H2 S(6)": 6.109, "H2 S(5)": 6.910, "[ArII]": 6.985,
    "[ArV]": 7.902, "H2 S(4)": 8.026, "[ArIII]": 8.991, "H2 S(3)": 9.665,
    "[SIV]": 10.511, "H2 S(2)": 12.279, "[NeII]": 12.814, "[NeV]": 14.322,
    "[NeIII]": 15.555, "H2 S(1)": 17.035, "[SIII]": 18.713, "[FeII]": 25.988,
    "[OIV]": 25.890, "[NeV]24": 24.318,
}
FLAG = dict(EDGE=1, SPIKE=2, HOSTLINE=4, BKG_CAPPED=8, BKG_FAIL=16,
            RED_END=32, BLENDED=64, SYS_EXTRAP=128)
CLEAN_EXCLUDE = FLAG["EDGE"] | FLAG["SPIKE"] | FLAG["HOSTLINE"] | FLAG["BKG_FAIL"] | FLAG["RED_END"]


def channel_of(wave):
    return np.digitize(wave, CHANNEL_SPLITS) + 1


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
@dataclass
class Config:
    stage3_cube: str = ""
    out_dir: str = "refined"
    product_name: str = "target"
    centroid_cube: str | None = None      # first-pass bgsub cube (position only)
    x0: float | None = None
    y0: float | None = None
    xy_fixed: tuple | None = None         # skip centroiding, use this position
    pos_wave_range: tuple = (5.0, 7.6)
    n_sub: int = 4
    # companion / blend
    components: str = "auto"              # auto | 1 | 2
    blend_wave_range: tuple = (5.0, 7.6)
    blend_cutout_half: int = 9
    min_sep_fwhm: float = 0.5             # resolvability limit for deblending
    dbic_min: float = 10.0
    psf_model: str = "moffat"
    beta: float = 2.5
    # background
    tune: bool = True
    k_grid: tuple = (2, 3, 4)
    f_grid: tuple = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0)
    k_default: int = 3
    f_default: float = 3.5
    min_mask_px: float = 5.0
    max_masked_frac: float = 0.55         # never mask more than this share of valid px
    wing_tol: float = 0.02                 # max acceptable wing (bias) fraction; bias-first selection
    max_masked_frac_ch123: float = 0.75    # looser cap for ch1-3, where footprint usually allows it
    knot_pad_fwhm: float = 1.0
    clip_lo: float = 3.0
    clip_hi: float = 2.5
    n_iter: int = 6
    tune_samples: int = 20                # sample slices per channel
    n_fake: int = 12
    tune_rscale: float = 2.0
    save_bkg_model: bool = False
    f_grid_ch4: tuple = (1.5, 2.0, 2.5, 3.0)   # ch4 PSF is ~9 px: smaller masks so the hole test can run
    tune_rscale_ch4: float = 1.25
    min_fake: int = 5
    min_fake_ch4: int = 4
    fast_channels: tuple = ()             # channels taken from the built-in fit instead of AstroBkgInterp, e.g. (4,)
    abi_reuse_path: str | None = None     # existing AstroBkgInterp output -> skip the slow run
    bkg_smooth: str = "auto"              # auto | on | off : smooth the bkg MODEL along wavelength per sub-band
    bkg_smooth_win: int = 21
    bkg_smooth_thresh: float = 0.25       # auto: smooth if bkg-model noise^2 > this x extracted-flux noise^2
    bkg_backend: str = "abi"              # "abi" = AstroBkgInterp (run ONCE), "fast" = built-in robust fit
    abi_fwhm_scale: float | None = None   # None -> taken from the tuning (max f over tunable ch1-3)
    abi_poly_k: int | None = None         # None -> most common tuned k over ch1-3
    abi_bin_size: int = 9
    abi_err_inflate: float = 1e3          # ERR multiplier on knot pixels (AstroBkgInterp is error-weighted)
    abi_fallback_capped: bool = True      # where the ABI mask would swallow the footprint, use the capped fast fit
    pos_tol: float = 0.3                  # re-centroid until the position moves less than this (px)
    pos_max_iter: int = 3
    position_only: bool = False           # stop after the position stage (cheap sanity check)
    # extraction
    rscale: float | None = None
    rscale_grid: tuple = (1.25, 1.5, 1.75, 2.0, 2.5)
    snr_tol: float = 0.95                 # take the largest r within this fraction of best S/N
    ifu_rfcorr: bool = True
    # errors
    err_scan_n: int = 400
    err_floor_ratio: float = 1.0
    err_smooth: int = 9
    noise_lag: int = 4                    # lag (samples) for the spectrum-based white-noise estimate
    noise_win: int = 301                  # running-median window (samples)
    # cleaning
    edge_trim_frac: float = 0.03
    spike_sigma: float = 6.0
    spike_window: int = 51
    lam_max_clean: float = 27.0
    redshift: float = 0.0
    line_tol_um: float = 0.05
    bin_from_um: float | None = None
    bin_factor: int = 8
    corr_len: float = 2.5
    # misc
    force: tuple = ()
    seed: int = 0
    plots: bool = True


# --------------------------------------------------------------------------
# lazy access to the existing pipeline module (jwst-heavy)
# --------------------------------------------------------------------------
_MP = None


def _mp():
    global _MP
    if _MP is None:
        import jspec.pipelines.miri_mrs_auto_pipeline as mp
        _MP = mp
    return _MP


# --------------------------------------------------------------------------
# cube container
# --------------------------------------------------------------------------
class Cube:
    def __init__(self, sci, err, wave, fwhm, path=None):
        self.sci = np.asarray(sci, dtype=np.float32)
        self.err = None if err is None else np.asarray(err, dtype=np.float32)
        self.wave = np.asarray(wave, dtype=float)
        self.fwhm = np.asarray(fwhm, dtype=float)
        self.path = path
        self.nz, self.ny, self.nx = self.sci.shape
        finite = np.isfinite(self.sci)
        if self.err is not None and np.isfinite(self.err).mean() > 0.01:
            self.valid = finite & np.isfinite(self.err) & (self.err > 0)
        else:
            self.err = None
            self.valid = finite & (self.sci != 0)
        self.valid_flat = self.valid.reshape(self.nz, -1)
        self.chan = channel_of(self.wave)

    @classmethod
    def open(cls, path):
        mp = _mp()
        with fits.open(path, memmap=False) as h:
            sci = np.asarray(h["SCI"].data, dtype=np.float32)
            err = np.asarray(h["ERR"].data, dtype=np.float32) if "ERR" in h else None
            if err is not None and err.shape != sci.shape:
                err = np.squeeze(err)
                err = err if err.shape == sci.shape else None
        wave = mp.get_mrs_wavelength(path)
        return cls(sci, err, wave, mp._mrs_fwhm_pixels(path, wave=wave), path)


def radec_to_xy(path, ra, dec):
    """Convert a sky position (deg) to (x, y) pixel coordinates in the
    spatial frame of the cube at `path`, using its celestial WCS (the
    SCI header's spectral/IFU axis is dropped)."""
    with fits.open(path, memmap=False) as h:
        header = h["SCI"].header
    w = celestial_wcs(header)
    x, y = w.all_world2pix(ra, dec, 0)
    return float(x), float(y)


# --------------------------------------------------------------------------
# robust polynomial background (batched, no per-slice python solve)
# --------------------------------------------------------------------------
class BkgFitter:
    """Legendre polynomial surface, total degree <= k, on the footprint pixels.
    fit_model() solves B independent robust fits at once (rows of D, W0)."""

    def __init__(self, footprint2d, kmax=4):
        self.ny, self.nx = footprint2d.shape
        self.pix = np.flatnonzero(footprint2d.ravel())
        yy, xx = np.divmod(self.pix, self.nx)
        self.x, self.y = xx.astype(float), yy.astype(float)
        self.N = self.pix.size
        xm, xh = 0.5 * (self.x.max() + self.x.min()), max(0.5 * np.ptp(self.x), 1.0)
        ym, yh = 0.5 * (self.y.max() + self.y.min()), max(0.5 * np.ptp(self.y), 1.0)
        Lu = npleg.legvander((self.x - xm) / xh, kmax)
        Lv = npleg.legvander((self.y - ym) / yh, kmax)
        terms = [(i, d - i) for d in range(kmax + 1) for i in range(d + 1)]
        self.A = np.stack([Lu[:, i] * Lv[:, j] for i, j in terms], axis=1)
        self.nterm = {k: (k + 1) * (k + 2) // 2 for k in range(kmax + 1)}
        self._aa = {}

    def _AA(self, n):
        if n not in self._aa:
            A = self.A[:, :n]
            self._aa[n] = (A[:, :, None] * A[:, None, :]).reshape(self.N, n * n)
        return self._aa[n]

    def fit_model(self, D, W0, k, clip=(3.0, 2.5), n_iter=6, min_per_term=8):
        """D (B,N) data (0 where invalid); W0 (B,N) bool usable pixels.
        Asymmetric iterative clipping: bright outliers (host knots, PSF wings)
        are rejected harder than dark ones.  Returns (model (B,N), info)."""
        n = self.nterm[k]
        A, AA = self.A[:, :n], self._AA(n)
        B = D.shape[0]
        W0f = W0.astype(np.float64)
        W = W0f.copy()
        lo, hi = clip
        for it in range(n_iter + 1):
            ATA = (W @ AA).reshape(B, n, n)
            ATb = (W * D) @ A
            tr = np.trace(ATA, axis1=1, axis2=2) / n
            ATA = ATA + (1e-10 * tr + 1e-30)[:, None, None] * np.eye(n)[None]
            try:
                coef = np.linalg.solve(ATA, ATb[..., None])[..., 0]
            except np.linalg.LinAlgError:
                coef = np.einsum("bij,bj->bi", np.linalg.pinv(ATA), ATb)
            if it == n_iter:
                break
            R = D - coef @ A.T
            Rm = np.where(W > 0, R, np.nan)
            med = np.nanmedian(Rm, axis=1)
            med = np.where(np.isfinite(med), med, 0.0)
            sig = 1.4826 * np.nanmedian(np.abs(Rm - med[:, None]), axis=1)
            sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.inf)
            keep = (R < (med + hi * sig)[:, None]) & (R > (med - lo * sig)[:, None])
            W = W0f * keep
        npix = W.sum(1)
        model = coef @ A.T
        fail = (npix < min_per_term * n) | ~np.isfinite(model).all(1)
        model[fail] = np.nan
        return model, dict(npix=npix, fail=fail, sigma=sig, coef=coef)


def masked_frac_arr(cfg, chan):
    """Per-slice cap on masked fraction: looser in ch1-3 (footprint usually
    allows a bigger exclusion mask before it matters), tighter in ch4 where
    the footprint itself is small."""
    chan = np.asarray(chan)
    return np.where(chan == 4, cfg.max_masked_frac, cfg.max_masked_frac_ch123)


def cap_radii(V, dist, r_nom, max_frac):
    """Largest mask radius <= r_nom that leaves >= (1-max_frac) of the valid pixels."""
    d = np.where(V, dist[None, :], np.inf)
    ds = np.sort(d, axis=1)
    nv = V.sum(1)
    idx = np.clip((max_frac * nv).astype(int) - 1, 0, ds.shape[1] - 1)
    rcap = ds[np.arange(V.shape[0]), idx]
    rcap = np.where(np.isfinite(rcap), rcap, r_nom)
    return np.minimum(r_nom, rcap), r_nom > rcap * 1.0001


class Field:
    """Cube + source position + knot mask + fitter: everything the fits share."""

    def __init__(self, cube, src_xy, knots2d, cfg):
        self.cube, self.cfg = cube, cfg
        self.sx, self.sy = float(src_xy[0]), float(src_xy[1])
        foot = cube.valid.any(axis=0)
        self.fit = BkgFitter(foot, kmax=max(cfg.k_grid + (cfg.k_default,)))
        self.N = self.fit.N
        self.dsrc = np.hypot(self.fit.x - self.sx, self.fit.y - self.sy)
        self.has_knots = knots2d is not None and knots2d.any()
        if self.has_knots:
            kd = ndi.distance_transform_edt(~knots2d)
            self.knot_d2d = kd
            self.knot_d = kd.ravel()[self.fit.pix]
        else:
            self.knot_d2d = np.full(foot.shape, np.inf)
            self.knot_d = np.full(self.N, np.inf)

    def D(self, idx):
        d = self.cube.sci.reshape(self.cube.nz, -1)[idx][..., self.fit.pix].astype(np.float64)
        return np.nan_to_num(d)

    def V(self, idx):
        return self.cube.valid_flat[idx][..., self.fit.pix]

    def E(self, idx):
        e = self.cube.err.reshape(self.cube.nz, -1)[idx][..., self.fit.pix].astype(np.float64)
        return np.nan_to_num(e)

    def knot_excl(self, fwhm):
        """(B,N) bool: pixels near static knots, padded by knot_pad_fwhm*FWHM."""
        fwhm = np.atleast_1d(fwhm)
        if not self.has_knots:
            return np.zeros((fwhm.size, self.N), bool)
        return self.knot_d[None, :] <= (self.cfg.knot_pad_fwhm * fwhm)[:, None]


# --------------------------------------------------------------------------
# hole-fill test: the direct measure of bias / interpolation error
# --------------------------------------------------------------------------
def pick_fake_centers(field, v2d, r_ap, r_excl, n, rng):
    """Non-overlapping aperture centres on empty sky, whole disc valid, clear of
    the target hole and of known knots."""
    ri = int(np.ceil(r_ap))
    yy, xx = np.mgrid[-ri:ri + 1, -ri:ri + 1]
    disc = (xx ** 2 + yy ** 2) <= r_ap ** 2
    good = ndi.binary_erosion(v2d, structure=disc, border_value=0)
    ys, xs = np.nonzero(good)
    if xs.size == 0:
        return []
    m = np.hypot(xs - field.sx, ys - field.sy) >= r_excl
    m &= field.knot_d2d[ys, xs] > r_ap + 1.0
    xs, ys = xs[m], ys[m]
    chosen = []
    for i in rng.permutation(xs.size):
        if all(np.hypot(xs[i] - cx, ys[i] - cy) >= 2.0 * r_ap for cx, cy in chosen):
            chosen.append((float(xs[i]), float(ys[i])))
            if len(chosen) >= n:
                break
    return chosen


def holefill(field, s, k, r_masks, r_aps, centers):
    """For slice s: fit the background with the target hole (and, one at a time,
    a fake hole at each centre); return aperture sums of (data - model).
    out[m, j, a]: mask radius m, j=0 target / j>=1 fake centre, aperture a."""
    fw = field.cube.fwhm[s]
    D = field.D(s)[None, :]
    V = field.V(s)[None, :]
    kn = field.knot_excl(fw)[0]
    dists = [field.dsrc] + [np.hypot(field.fit.x - cx, field.fit.y - cy) for cx, cy in centers]
    J1 = len(dists)
    rows = []
    for r in r_masks:
        for j in range(J1):
            ex = kn | (field.dsrc <= r) | (dists[j] <= r)
            rows.append(V[0] & ~ex)
    W0 = np.array(rows)
    cfg = field.cfg
    model, info = field.fit.fit_model(np.repeat(D, len(rows), 0), W0, k,
                                      clip=(cfg.clip_lo, cfg.clip_hi), n_iter=cfg.n_iter)
    resid = D - model
    out = np.full((len(r_masks), J1, len(r_aps)), np.nan)
    for m in range(len(r_masks)):
        for j in range(J1):
            row = resid[m * J1 + j]
            for a, ra in enumerate(r_aps):
                ap = dists[j] <= ra
                if ap.sum() == 0 or (V[0] & ap).sum() < 0.98 * ap.sum():
                    continue
                out[m, j, a] = np.nansum(np.where(V[0] & ap, row, 0.0))
    return out


def _robust_scatter(x):
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def sample_slices(cube, field, ch, n, r_ap_scale):
    idx = np.flatnonzero(cube.chan == ch)
    idx = idx[np.argsort(cube.wave[idx])]
    ok = []
    for s in idx:
        ap = field.dsrc <= r_ap_scale * cube.fwhm[s]
        if ap.sum() and field.V(s)[ap].mean() > 0.98:
            ok.append(s)
    ok = np.array(ok, int)
    if ok.size <= n:
        return ok
    return ok[np.linspace(0, ok.size - 1, n).astype(int)]


def tune_background(field, cfg):
    """Per channel choose (k, mask scale f) minimising
         median_s[ (bias^2 + scatter^2 + wing^2) / scatter_min^2 ]
    bias/scatter: hole-fill aperture error on empty sky (native cube units);
    wing: source aperture flux at the largest eligible f minus flux at f
    (light that the smaller mask let the polynomial absorb -> over-subtraction)."""
    cube = field.cube
    rng = np.random.default_rng(cfg.seed)
    kg = list(cfg.k_grid)
    res = {}
    for ch in (1, 2, 3, 4):
        fg = np.array(cfg.f_grid_ch4 if ch == 4 else cfg.f_grid, float)
        rs = cfg.tune_rscale_ch4 if ch == 4 else cfg.tune_rscale
        min_f = cfg.min_fake_ch4 if ch == 4 else cfg.min_fake
        samp = sample_slices(cube, field, ch, cfg.tune_samples, rs)
        rec = []
        for s in samp:
            fw = cube.fwhm[s]
            r_ap = rs * fw
            r_nom = np.maximum(fg * fw, cfg.min_mask_px)
            centers, nf = [], 0
            for i in range(len(fg) - 1, -1, -1):
                c = pick_fake_centers(field, cube.valid[s], r_ap, r_nom[i] + r_ap,
                                      cfg.n_fake, rng)
                if len(c) >= min_f:
                    centers, nf = c, i + 1
                    break
            if nf < 2:
                continue
            tab = np.full((len(kg), len(fg), 3), np.nan)   # src, bias, scatter
            for ik, k in enumerate(kg):
                o = holefill(field, s, k, r_nom[:nf], [r_ap], centers)[:, :, 0]
                tab[ik, :nf, 0] = o[:, 0]
                tab[ik, :nf, 1] = np.nanmedian(o[:, 1:], axis=1)
                tab[ik, :nf, 2] = [_robust_scatter(o[m, 1:]) for m in range(nf)]
            rec.append((s, nf, tab))
        if len(rec) < 6:
            res[ch] = dict(tunable=False, k=cfg.k_default, f=cfg.f_default, n=len(rec), f_grid=fg.tolist(),
                           wing_frac=np.nan, wing_ok=False, bias_over_sigma=np.nan, sigma_native=np.nan)
            log(f"  ch{ch}: too few usable slices/fake positions ({len(rec)}); "
                f"using defaults k={cfg.k_default}, f={cfg.f_default}")
            continue
        nf_top = int(np.percentile([r[1] for r in rec], 40))      # >=60% of slices eligible
        nf_top = max(nf_top, 2)
        rec = [r for r in rec if r[1] >= nf_top]
        per = []
        for s, nf, tab in rec:
            scmin = np.nanmin(tab[:, :nf_top, 2])
            if not np.isfinite(scmin) or scmin <= 0:
                continue
            wing = tab[:, nf_top - 1:nf_top, 0] - tab[:, :nf_top, 0]
            per.append((tab[:, :nf_top, 1] ** 2 + tab[:, :nf_top, 2] ** 2 + wing ** 2) / scmin ** 2)
        M = np.nanmedian(np.array(per), axis=0)

        # Per-(k, f) wing (bias) fraction, not just at the point M eventually picks:
        # wing is a *systematic* -- it doesn't average down the way scatter/bias do
        # under binning -- so it is used as a hard pre-filter, not blended into M.
        # Blending it into one score (the old behaviour) let a flat scatter surface
        # decide, which -- combined with a "smallest (k,f) wins ties" tie-break --
        # systematically favoured the smaller, more-biased mask.
        src_arr = np.array([r[2][:, :nf_top, 0] for r in rec])            # (n_rec, len(kg), nf_top)
        src_top_arr = src_arr[:, :, nf_top - 1:nf_top]
        wing_num = (src_top_arr - src_arr).sum(axis=0)
        wing_den = np.abs(src_arr).sum(axis=0)
        wing_grid = np.where(wing_den > 0, wing_num / np.where(wing_den > 0, wing_den, 1.0), np.nan)
        wing_grid = np.maximum(wing_grid, 0.0)                            # (len(kg), nf_top)

        ok = wing_grid <= cfg.wing_tol
        if ok.any():
            M_ok = np.where(ok, M, np.inf)
            ik, jf = np.unravel_index(np.nanargmin(M_ok), M_ok.shape)
            ties = np.argwhere(M_ok <= 1.03 * M_ok[ik, jf])
            ik, jf = sorted(map(tuple, ties))[0]                          # among bias-safe points: smallest k, f
        else:
            # nothing meets the tolerance even at the largest tested mask: take the
            # least-biased grid point available (largest f), tie-break on scatter.
            ik, jf = np.unravel_index(np.nanargmin(wing_grid), wing_grid.shape)
            log(f"  ch{ch}: WARNING no (k,f) on the grid reaches wing_tol={cfg.wing_tol:.2%}; "
                f"picking the least-biased point tested (wing={wing_grid[ik, jf]:.1%}) -- "
                f"consider extending f_grid further")
        src_top = np.array([r[2][ik, nf_top - 1, 0] for r in rec])
        src_sel = np.array([r[2][ik, jf, 0] for r in rec])
        bias = np.array([r[2][ik, jf, 1] for r in rec])
        sig = np.array([r[2][ik, jf, 2] for r in rec])
        denom = np.abs(src_sel).sum()
        res[ch] = dict(tunable=True, k=int(kg[ik]), f=float(fg[jf]), f_top=float(fg[nf_top - 1]), f_grid=fg.tolist(),
                       n=len(rec), metric=M.tolist(), wing_grid=wing_grid.tolist(), wing_ok=bool(ok.any()),
                       wing_frac=float(max(0.0, (src_top - src_sel).sum() / denom)) if denom > 0 else np.nan,
                       bias_over_sigma=float(np.nanmedian(bias) / np.nanmedian(sig)),
                       sigma_native=float(np.nanmedian(sig)))
        r = res[ch]
        log(f"  ch{ch}: k={r['k']} f={r['f']} (f_top={r['f_top']}, n={r['n']}) "
            f"wing<= {100 * r['wing_frac']:.1f}% ({'meets' if r['wing_ok'] else 'BELOW'} wing_tol={100*cfg.wing_tol:.1f}%)"
            f"  hole-fill bias/sigma={r['bias_over_sigma']:+.2f}")
    return res


# --------------------------------------------------------------------------
# full-cube background fit
# --------------------------------------------------------------------------
def fit_full_cube(field, cfg, tune, block=48):
    cube, fit = field.cube, field.fit
    out = cube.sci.copy()
    out_flat = out.reshape(cube.nz, -1)
    info = {k: np.zeros(cube.nz) for k in ("r_used", "capped", "fail", "k", "f", "npix", "sigma")}
    t0 = time.time()
    for ch in np.unique(cube.chan):
        t = tune.get(int(ch), tune.get(str(int(ch)), {}))
        k, f = int(t.get("k", cfg.k_default)), float(t.get("f", cfg.f_default))
        idx_ch = np.flatnonzero(cube.chan == ch)
        for b0 in range(0, idx_ch.size, block):
            idx = idx_ch[b0:b0 + block]
            D, V = field.D(idx), field.V(idx)
            fw = cube.fwhm[idx]
            r_nom = np.maximum(f * fw, cfg.min_mask_px)
            r_used, capped = cap_radii(V, field.dsrc, r_nom, masked_frac_arr(cfg, cube.chan[idx]))
            excl = (field.dsrc[None, :] <= r_used[:, None]) | field.knot_excl(fw)
            model, inf = fit.fit_model(D, V & ~excl, k, clip=(cfg.clip_lo, cfg.clip_hi),
                                       n_iter=cfg.n_iter)
            new = np.where(V, D - model, D)
            new[inf["fail"]] = np.where(V[inf["fail"]], np.nan, D[inf["fail"]])
            out_flat[idx[:, None], fit.pix[None, :]] = new.astype(np.float32)
            info["r_used"][idx], info["capped"][idx] = r_used, capped
            info["fail"][idx], info["k"][idx], info["f"][idx] = inf["fail"], k, f
            info["npix"][idx], info["sigma"][idx] = inf["npix"], np.where(np.isfinite(inf["sigma"]), inf["sigma"], np.nan)
        log(f"  ch{ch}: {idx_ch.size} slices done ({time.time() - t0:.0f}s)")
    return out, info


def write_bgsub_cube(src_path, out_path, bgsub, info, wave, cfg):
    with fits.open(src_path) as h:
        new = h.copy()
        new["SCI"].data = bgsub.astype(h["SCI"].data.dtype)
        tab = fits.BinTableHDU.from_columns([
            fits.Column("WAVELENGTH", "D", array=wave),
            fits.Column("R_MASK", "E", array=info["r_used"]),
            fits.Column("POLY_K", "I", array=info["k"].astype(int)),
            fits.Column("NPIX", "E", array=info["npix"]),
            fits.Column("CAPPED", "L", array=info["capped"].astype(bool)),
            fits.Column("FAILED", "L", array=info["fail"].astype(bool))], name="BKGINFO")
        new.append(tab)
        new[0].header["BKGMETH"] = ("robust-legendre-poly", "refine_extract background")
        new.writeto(out_path, overwrite=True)


def ring_levels(cube_bg, field, slices, cfg):
    """Median of the background-subtracted cube in FWHM-scaled rings (moat check)."""
    bands = [(0, 1, "core"), (1, 2, "1-2"), (2, 3, "2-3"), (3, 5, "3-5"), (5, 8, "5-8 (sky)")]
    out = {}
    for lo, hi, name in bands:
        vals = []
        for s in slices:
            fw = field.cube.fwhm[s]
            V = field.V(s)
            row = np.nan_to_num(cube_bg.reshape(field.cube.nz, -1)[s][field.fit.pix].astype(float))
            m = V & (field.dsrc >= lo * fw) & (field.dsrc < hi * fw) & ~field.knot_excl(fw)[0]
            if m.sum() > 5:
                vals.append(np.median(row[m]))
        out[name] = float(np.median(vals)) if vals else np.nan
    return out


# --------------------------------------------------------------------------
# aperture choice and empirical errors (both use the hole-fill machinery)
# --------------------------------------------------------------------------
def choose_rscale(field, cfg, tune):
    """S/N of the source aperture vs radius, noise from hole-filled fake apertures."""
    cube = field.cube
    rng = np.random.default_rng(cfg.seed + 1)
    grid = np.array(cfg.rscale_grid, float)
    table = {}
    for ch in (1, 2, 3):                                  # ch4: aperture would not fit anyway
        t = tune[ch]
        samp = sample_slices(cube, field, ch, cfg.tune_samples, grid.max())
        snr = []
        for s in samp:
            fw = cube.fwhm[s]
            r_m = max(t["f"] * fw, cfg.min_mask_px)
            r_aps = grid * fw
            c = pick_fake_centers(field, cube.valid[s], r_aps.max(), r_m + r_aps.max(), cfg.n_fake, rng)
            if len(c) < 5:
                continue
            o = holefill(field, s, t["k"], [r_m], list(r_aps), c)[0]
            sig = np.array([_robust_scatter(o[1:, a]) for a in range(grid.size)])
            snr.append((o[0] / sig, o[0]))
        if len(snr) < 5:
            continue
        S = np.array([a for a, _ in snr])
        F = np.array([b for _, b in snr])
        top = F[:, np.argmax(grid)] > np.percentile(F[:, np.argmax(grid)], 50)   # brighter half
        table[ch] = np.nanmedian(S[top], axis=0)
    if not table:
        return 2.0, {}
    avg = np.nanmean([v / np.nanmax(v) for v in table.values()], axis=0)
    ok = np.flatnonzero(avg >= cfg.snr_tol * np.nanmax(avg))
    return float(grid[ok.max()]), {int(c): v.tolist() for c, v in table.items()}


def error_scan(field, cfg, tune, rscale):
    """Empirical / formal aperture-noise ratio R(lambda), from hole-filled fake apertures."""
    cube = field.cube
    rng = np.random.default_rng(cfg.seed + 2)
    sel = np.linspace(0, cube.nz - 1, min(cfg.err_scan_n, cube.nz)).astype(int)
    w_s, R_s, b_s = [], [], []
    for s in sel:
        ch = int(cube.chan[s])
        t = tune.get(ch, tune.get(str(ch), {}))
        fw = cube.fwhm[s]
        r_ap = rscale * fw
        r_m = max(t.get("f", cfg.f_default) * fw, cfg.min_mask_px)
        c = pick_fake_centers(field, cube.valid[s], r_ap, r_m + r_ap, cfg.n_fake, rng)
        if len(c) < 5:
            continue
        o = holefill(field, s, int(t.get("k", cfg.k_default)), [r_m], [r_ap], c)[0, :, 0]
        sig = _robust_scatter(o[1:])
        ap = field.dsrc <= r_ap
        E = field.E(s)
        formal = np.sqrt((E[ap] ** 2).sum())
        if np.isfinite(sig) and formal > 0:
            w_s.append(cube.wave[s]); R_s.append(sig / formal); b_s.append(np.nanmedian(o[1:]) / sig)
    if not w_s:
        return None
    w_s, R_s, b_s = map(np.array, (w_s, R_s, b_s))
    o = np.argsort(w_s)
    return dict(wave=w_s[o], R=R_s[o], bias_over_sigma=b_s[o])


def smooth_ratio(wave, R, cfg):
    """Running median (in sub-band-aware windows) of R, floored, interpolable."""
    Rs = ndi.median_filter(R, size=cfg.err_smooth, mode="nearest")
    Rs = ndi.uniform_filter1d(Rs, 3, mode="nearest")
    return np.maximum(Rs, cfg.err_floor_ratio)


# --------------------------------------------------------------------------
# position and companion
# --------------------------------------------------------------------------
def robust_position(cen):
    x, y = cen["x"], cen["y"]
    out = dict(x=x, y=y, x_err=cen["x_err"], y_err=cen["y_err"], snr=cen["snr"],
               x_scatter=cen["x_scatter"], y_scatter=cen["y_scatter"], outlier_bands=[])
    sx, sy, sw = cen["sub_x"], cen["sub_y"], cen["sub_wavelength"]
    if len(sy) >= 4:
        for name, v in (("x", sx), ("y", sy)):
            for i in range(len(v)):
                rest = np.delete(v, i)
                if abs(v[i] - rest.mean()) > 2.5 * max(rest.std(ddof=1), 0.05):
                    out["outlier_bands"].append((name, float(sw[i]), float(v[i])))
        keep_y = np.ones(len(sy), bool)
        for n_, w_, _ in out["outlier_bands"]:
            if n_ == "y":
                keep_y &= ~np.isclose(sw, w_)
        out["y_scatter_clean"] = float(np.std(sy[keep_y], ddof=1)) if keep_y.sum() > 1 else np.nan
        o = np.argsort(sw)
        out["x_drift_px_per_um"] = float(np.polyfit(sw[o], sx[o], 1)[0])
    out["pos_err"] = float(max(cen["x_err"], cen["y_err"], out.get("y_scatter_clean", 0) or 0,
                               0.5 * abs(out.get("x_drift_px_per_um", 0)) * np.ptp(sw) if len(sw) > 1 else 0))
    return out


def knots_from_segmap(cen, xy):
    seg = np.asarray(cen["segment_map"].data)
    yy, xx = np.mgrid[:seg.shape[0], :seg.shape[1]]
    near = np.hypot(xx - xy[0], yy - xy[1]) <= 3.0
    tgt = set(np.unique(seg[near])) - {0}
    knots = (seg > 0) & ~np.isin(seg, list(tgt))
    return ndi.binary_dilation(knots, iterations=1)


def assess_companion(cube_file, xy, knots2d, cfg, cube_fwhm_fn):
    """Two-PSF fit on the full band and on each sub-band, with other knots masked.
    Physical companion => separation ~constant in pixels/arcsec, PA constant.
    PSF-shape artefact  => separation tracks the FWHM (grows ~ lambda)."""
    mp = _mp()
    coll = mp.make_weighted_image(cube_file, wave_range=cfg.blend_wave_range, n_sub=cfg.n_sub, clip_sigma=5.0)
    bands = [(coll["image"], coll["err"], coll["wave_range"])] + \
        list(zip(coll.get("sub_images", []), coll.get("sub_errs", []), coll.get("sub_wave_ranges", [])))
    h = cfg.blend_cutout_half
    xi, yi = int(round(xy[0])), int(round(xy[1]))
    ylo, xlo = max(yi - h, 0), max(xi - h, 0)
    rows, best_full = [], None
    for ib, (img_f, err_f, wr) in enumerate(bands):
        yhi, xhi = min(yi + h + 1, img_f.shape[0]), min(xi + h + 1, img_f.shape[1])
        img = np.array(img_f[ylo:yhi, xlo:xhi], float)
        err = np.array(err_f[ylo:yhi, xlo:xhi], float)
        good_err = err[np.isfinite(err) & (err > 0)]
        if good_err.size < 0.1 * err.size:
            err = np.full_like(img, np.nanstd(img))
        else:
            err = np.where(np.isfinite(err) & (err > 0), err, np.median(good_err))
            err = np.maximum(err, 0.05 * np.median(good_err))
        kn = knots2d[ylo:yhi, xlo:xhi]
        err[kn] = np.nan                                     # mask other knots
        fw0 = float(cube_fwhm_fn(0.5 * (wr[0] + wr[1])))
        cx, cy = xy[0] - xlo, xy[1] - ylo
        f1, chi1, np1 = mp._fit_n_components(img, err, 1, fw0, cfg.psf_model, cfg.beta, True, [(cx, cy)])
        s2 = max(chi1 / max(f1["ngood"] - np1, 1), 1e-12)
        starts = []
        c1 = f1["components"][0]
        starts.append([(c1[0], c1[1]), (c1[0] + 0.35 * fw0, c1[1])])
        if best_full is not None:
            starts.append([(c[0] - (xlo - best_full[1]), c[1] - (ylo - best_full[2])) for c in best_full[0]])
        for sep in (0.5, 1.0, 1.6):
            for ang in np.arange(0, np.pi, np.pi / 4):
                d = 0.5 * sep * fw0
                starts.append([(cx + d * np.cos(ang), cy + d * np.sin(ang)),
                               (cx - d * np.cos(ang), cy - d * np.sin(ang))])
        best = None
        for st in starts:
            try:
                f2, chi2, np2 = mp._fit_n_components(img, err, 2, fw0, cfg.psf_model, cfg.beta, True, st)
            except Exception:
                continue
            if best is None or chi2 < best[1]:
                best = (f2, chi2, np2)
        if best is None:
            continue
        f2, chi2, np2 = best
        n = f1["ngood"]
        dbic = (chi1 - chi2) / s2 - (np2 - np1) * np.log(n)
        comp = sorted(f2["components"], key=lambda c: -c[2])
        sep = float(np.hypot(comp[0][0] - comp[1][0], comp[0][1] - comp[1][1]))
        pa = float(np.degrees(np.arctan2(comp[1][1] - comp[0][1], comp[1][0] - comp[0][0])) % 180.0)
        fwhm_fit = fw0 * f2["scale"]
        rows.append(dict(band=("full" if ib == 0 else f"sub{ib}"), wave=0.5 * (wr[0] + wr[1]),
                         dbic=float(dbic), sep_px=sep, sep_fwhm=sep / fwhm_fit, pa=pa,
                         flux_ratio=float(comp[1][2] / comp[0][2]), fwhm=fwhm_fit,
                         positions=[(float(c[0] + xlo), float(c[1] + ylo)) for c in comp],
                         scale=float(f2["scale"])))
        if ib == 0:
            best_full = (f2["components"], xlo, ylo)
    if not rows:
        return dict(verdict="unresolved", reasons=["no 2-component fit converged"], rows=[])
    full, subs = rows[0], rows[1:]
    reasons, ok = [], True
    if full["dbic"] <= cfg.dbic_min:
        ok = False; reasons.append(f"full-band dBIC={full['dbic']:.1f} <= {cfg.dbic_min}")
    if full["sep_fwhm"] < cfg.min_sep_fwhm:
        ok = False; reasons.append(f"separation {full['sep_fwhm']:.2f} FWHM < {cfg.min_sep_fwhm}")
    if len(subs) >= 3:
        n_sig = sum(r["dbic"] > cfg.dbic_min for r in subs)
        seps = np.array([r["sep_px"] for r in subs])
        pas = np.radians([2 * r["pa"] for r in subs])
        pa_R = np.hypot(np.cos(pas).mean(), np.sin(pas).mean())      # circular concentration (axial)
        fw = np.array([r["fwhm"] for r in subs])
        slope = float(np.polyfit(np.log(fw), np.log(seps), 1)[0]) if np.ptp(np.log(fw)) > 0.05 else np.nan
        if n_sig < max(3, len(subs) - 1):
            ok = False; reasons.append(f"only {n_sig}/{len(subs)} sub-bands prefer 2 components")
        if seps.std() / seps.mean() > 0.25:
            ok = False; reasons.append(f"separation varies {100 * seps.std() / seps.mean():.0f}% between sub-bands")
        if pa_R < 0.9:
            ok = False; reasons.append(f"position angle not consistent between sub-bands (R={pa_R:.2f})")
        if np.isfinite(slope) and slope > 0.6:
            ok = False; reasons.append(f"separation scales with FWHM (slope {slope:.2f}): PSF-shape signature")
    else:
        ok = False; reasons.append("too few sub-band fits to test consistency")
    return dict(verdict="resolved" if ok else "unresolved", reasons=reasons, rows=rows,
                positions=full["positions"], sep_px=full["sep_px"], scale=full["scale"])



def _fill_nan(x):
    x = np.array(x, float)
    g = np.isfinite(x)
    if g.sum() == 0:
        return np.ones_like(x)
    if (~g).any():
        x[~g] = np.interp(np.flatnonzero(~g), np.flatnonzero(g), x[g])
    return x


def white_noise_ratio(w, f, ef, flag, cfg):
    """Random-noise inflation R_white(lambda) = true white noise / formal error, measured on the
    extracted spectrum itself with lag-`noise_lag` differences (immune to the 1-2 sample
    correlation of the cube), as a running median inside each sub-band segment."""
    o = np.argsort(w)
    ws, fs, es, fl = w[o], f[o], ef[o], flag[o]
    ok = np.isfinite(fs) & np.isfinite(es) & (es > 0) & ((fl & CLEAN_EXCLUDE) == 0)
    lag = max(int(cfg.noise_lag), 1)
    r = np.full(ws.size, np.nan)
    m = ok[lag:] & ok[:-lag]
    emid = 0.5 * (es[lag:] + es[:-lag])
    r[lag:] = np.where(m, np.abs(fs[lag:] - fs[:-lag]) / emid, np.nan)
    Rw = np.ones(ws.size)
    seg = np.digitize(ws, BAND_BOUNDS)
    for sid in np.unique(seg):
        sel = np.flatnonzero(seg == sid)
        if np.isfinite(r[sel]).sum() < 30:
            continue
        rr = _fill_nan(r[sel])
        size = int(min(cfg.noise_win, sel.size // 2 * 2 - 1))
        size = max(size, 5)
        Rw[sel] = ndi.median_filter(rr, size=size, mode="nearest") / (0.6745 * np.sqrt(2.0))
    Rw = np.maximum(Rw, cfg.err_floor_ratio)
    out = np.empty_like(Rw)
    out[o] = Rw
    return out


def hole_fill_sys_ratio(scan, w_out, Rw_out, cfg):
    """Correlated (spectrally smooth) systematic, in units of the formal error:
    R_sys = sqrt(max(R_holefill^2 - R_white^2, 0)).  Beyond the last wavelength where fake
    apertures fit, hold the median of the last few good points and flag SYS_EXTRAP."""
    if scan is None:
        return np.zeros_like(w_out), np.zeros(w_out.size, bool)
    Rh = ndi.median_filter(scan["R"], size=cfg.err_smooth, mode="nearest")
    Rh = ndi.uniform_filter1d(Rh, 3, mode="nearest")
    Rw_at = np.interp(scan["wave"], w_out[np.argsort(w_out)], Rw_out[np.argsort(w_out)])
    Rs = np.sqrt(np.maximum(Rh ** 2 - Rw_at ** 2, 0.0))
    tail = float(np.median(Rs[-6:]))
    head = float(np.median(Rs[:6]))
    out = np.interp(w_out, scan["wave"], Rs, left=head, right=tail)
    extrap = (w_out > scan["wave"].max() + 0.05) | (w_out < scan["wave"].min() - 0.05)
    return out, extrap


def stitch_steps(w, f, flag, half=0.5, gap=0.02):
    """Flux step at each sub-band join from local linear fits either side.  A step is only
    called 'clean' if both fits are featureless (residuals ~ point-to-point noise), because
    PAH/silicate structure at a join looks exactly like a calibration offset."""
    ok = np.isfinite(f) & ((flag & CLEAN_EXCLUDE) == 0)
    out = []
    for (a, bnd), c in zip(zip(_names[:-1], _names[1:]), BAND_BOUNDS):
        hw = 0.5 * (MRS_SUBBANDS[a][1] - MRS_SUBBANDS[bnd][0])
        vals, clean = [], True
        for side in (-1, 1):
            lo, hi = (c - hw - half, c - hw - gap) if side < 0 else (c + hw + gap, c + hw + half)
            m = ok & (w >= lo) & (w <= hi)
            if m.sum() < 40:
                vals = None
                break
            x, y = w[m] - c, f[m]
            p = np.polyfit(x, y, 1)
            res = y - np.polyval(p, x)
            s_res = 1.4826 * np.median(np.abs(res - np.median(res)))
            A = np.vander(x, 2)
            cov = s_res ** 2 * np.linalg.inv(A.T @ A)
            vals.append((p[1], float(np.sqrt(cov[1, 1]))))
            # curvature test: a significant quadratic term means real spectral structure
            # (PAH / silicate / line wings) at the join, not a flat calibration offset
            p2 = np.polyfit(x, y, 2)
            r2 = y - np.polyval(p2, x)
            s2 = 1.4826 * np.median(np.abs(r2 - np.median(r2)))
            A2 = np.vander(x, 3)
            c2 = s2 ** 2 * np.linalg.inv(A2.T @ A2)
            clean &= bool(abs(p2[0]) < 3.0 * np.sqrt(c2[0, 0]))
        if not vals:
            out.append(dict(join=f"{a}|{bnd}", centre=float(c), step_pct=np.nan, err_pct=np.nan, clean=False))
            continue
        (l, el), (u, eu) = vals
        out.append(dict(join=f"{a}|{bnd}", centre=float(c), step_pct=float(100 * (u / l - 1)),
                        err_pct=float(100 * np.hypot(el, eu) / abs(l)), clean=clean))
    return out


def spectral_smooth_bkg(cube, field, final, cfg):
    """Smooth the background MODEL (raw - subtracted) along wavelength, per sub-band segment,
    where its slice-to-slice noise is a meaningful share of the extracted-flux noise.
    Never crosses a sub-band join.  Modifies `final` in place; returns a report."""
    rep_ = {}
    if cfg.bkg_smooth == "off":
        return rep_
    order = np.argsort(cube.wave)
    seg = np.digitize(cube.wave[order], BAND_BOUNDS)
    win = int(cfg.bkg_smooth_win) | 1
    raw_flat = cube.sci.reshape(cube.nz, -1)
    fin_flat = final.reshape(cube.nz, -1)
    for sid in np.unique(seg):
        idx = order[seg == sid]
        name = _names[sid]
        if idx.size < 3 * win:
            continue
        b_ap = np.zeros(idx.size); s_ap = np.zeros(idx.size)
        for j, s in enumerate(idx):
            ap = field.dsrc <= 1.5 * cube.fwhm[s]
            v = cube.valid_flat[s][field.fit.pix] & ap
            r_ = np.nan_to_num(raw_flat[s][field.fit.pix].astype(float))
            f_ = np.nan_to_num(fin_flat[s][field.fit.pix].astype(float))
            b_ap[j] = (r_ - f_)[v].sum(); s_ap[j] = f_[v].sum()
        hp = lambda x: x - ndi.median_filter(x, size=win, mode="nearest")
        nb = 1.4826 * np.median(np.abs(hp(b_ap))); ns = 1.4826 * np.median(np.abs(hp(s_ap)))
        frac = float((nb / ns) ** 2) if ns > 0 else np.nan
        do = cfg.bkg_smooth == "on" or (np.isfinite(frac) and frac > cfg.bkg_smooth_thresh)
        rep_[name] = dict(bkg_noise_share=frac, smoothed=bool(do))
        if not do:
            continue
        valid = cube.valid[idx]
        bk = np.where(valid, cube.sci[idx] - final[idx], 0.0).astype(np.float32)
        bk = np.nan_to_num(bk)
        sm = ndi.median_filter(bk, size=(win, 1, 1), mode="nearest")
        sm = ndi.uniform_filter1d(sm, 5, axis=0, mode="nearest")
        new = np.where(valid & np.isfinite(final[idx]), cube.sci[idx] - sm, final[idx])
        final[idx] = new.astype(final.dtype)
    return rep_


# --------------------------------------------------------------------------
# spectrum cleaning
# --------------------------------------------------------------------------
def trusted_interior(w, trim_frac):
    ok = np.zeros(w.size, bool)
    for lo, hi in MRS_SUBBANDS.values():
        t = trim_frac * (hi - lo)
        ok |= (w >= lo + t) & (w <= hi - t)
    return ok


def flag_spectrum(w, f, cfg):
    flag = np.zeros(w.size, np.uint16)
    flag[~trusted_interior(w, cfg.edge_trim_frac)] |= FLAG["EDGE"]
    flag[w > cfg.lam_max_clean] |= FLAG["RED_END"]
    good = np.isfinite(f)
    if good.sum() > cfg.spike_window * 2:
        ff = np.interp(w, w[good], f[good])
        med = ndi.median_filter(ff, size=cfg.spike_window, mode="nearest")
        r = ff - med
        scale = 1.4826 * ndi.median_filter(np.abs(r), size=2 * cfg.spike_window + 1, mode="nearest")
        spike = np.abs(r) > cfg.spike_sigma * np.maximum(scale, 1e-12)
        spike = ndi.binary_dilation(spike, iterations=1) & (np.abs(r) > 0.5 * cfg.spike_sigma * np.maximum(scale, 1e-12))
        flag[spike] |= FLAG["SPIKE"]
        for name, lam in HOST_LINES_UM.items():
            near = np.abs(w - lam * (1 + cfg.redshift)) <= cfg.line_tol_um
            flag[near & spike] |= FLAG["HOSTLINE"]
    flag[~good] |= FLAG["SPIKE"]
    return flag


def bin_red(w, f, e, flag, cfg, e_sys=None):
    if cfg.bin_from_um is None:
        return None
    m = (w >= cfg.bin_from_um) & ((flag & CLEAN_EXCLUDE) == 0) & np.isfinite(f) & np.isfinite(e)
    wi, fi, ei = w[m], f[m], e[m]
    n = cfg.bin_factor
    nb = wi.size // n
    if nb < 2:
        return None
    sl = slice(0, nb * n)
    W = wi[sl].reshape(nb, n); F = fi[sl].reshape(nb, n); E = ei[sl].reshape(nb, n)
    wt = 1.0 / E ** 2
    fb = (F * wt).sum(1) / wt.sum(1)
    eb = np.sqrt((E ** 2).mean(1) / (n / cfg.corr_len))          # correlated adjacent samples
    if e_sys is not None:                                        # systematic does not average down
        S = e_sys[m][sl].reshape(nb, n)
        eb = np.sqrt(eb ** 2 + (S ** 2).mean(1))
    return W.mean(1), fb, eb


def pp_scatter_ratio(w, f, e, flag):
    """Robust point-to-point scatter / quoted error, per channel (want ~1)."""
    out = {}
    ok = ((flag & CLEAN_EXCLUDE) == 0) & np.isfinite(f) & np.isfinite(e)
    ch = channel_of(w)
    for c in (1, 2, 3, 4):
        m = ok & (ch == c)
        if m.sum() < 50:
            continue
        d = np.diff(f[m])
        pp = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
        out[c] = float(pp / np.median(e[m]))
    return out


# --------------------------------------------------------------------------
# products
# --------------------------------------------------------------------------
def save_products(cfg, w, f, e, e_sys, e_formal, flag, unit, meta, tag=""):
    from astropy.table import Table
    base = os.path.join(cfg.out_dir, f"{cfg.product_name}{tag}")
    tab = Table({"WAVELENGTH": w, "FLUX": f, "FLUX_ERROR": e, "FLUX_SYS": e_sys,
                 "FLUX_ERR_TOTAL": np.sqrt(e ** 2 + e_sys ** 2), "FLUX_ERR_FORMAL": e_formal,
                 "FLAG": flag})
    for c, u in (("WAVELENGTH", "um"), ("FLUX", unit), ("FLUX_ERROR", unit), ("FLUX_SYS", unit),
                 ("FLUX_ERR_TOTAL", unit), ("FLUX_ERR_FORMAL", unit)):
        tab[c].unit = u
    tab.meta.update({k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool))})
    tab.write(base + "_full.ecsv", format="ascii.ecsv", overwrite=True)
    tab.write(base + "_full.fits", format="fits", overwrite=True)
    keep = (flag & CLEAN_EXCLUDE) == 0
    clean = tab[keep]
    clean.write(base + "_clean.ecsv", format="ascii.ecsv", overwrite=True)
    clean.write(base + "_clean.fits", format="fits", overwrite=True)
    return base


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


# --------------------------------------------------------------------------
# QA figures
# --------------------------------------------------------------------------
def make_plots(cfg, R, cube, field, rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    qa = os.path.join(cfg.out_dir, "qa")
    os.makedirs(qa, exist_ok=True)

    # background
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    for a, ch in zip(ax, (1, 2, 3, 4)):
        t = rep["tune"].get(ch) or rep["tune"].get(str(ch)) or {}
        M = np.atleast_2d(np.array(t.get("metric", [[np.nan]]), float))
        tag = "" if t.get("tunable", False) else " (defaults, not tunable)"
        a.set_title(f"ch{ch}: k={t.get('k')} f={t.get('f')}{tag}", fontsize=8)
        if t.get("tunable", False) and M.shape[1] == len(M.ravel()) // M.shape[0]:
            a.imshow(np.log10(M), origin="lower", aspect="auto")
            fgp = t.get("f_grid", list(cfg.f_grid))
            a.set_xticks(range(M.shape[1])); a.set_xticklabels([str(f) for f in fgp[:M.shape[1]]], fontsize=7)
            a.set_yticks(range(M.shape[0])); a.set_yticklabels([str(k) for k in cfg.k_grid[:M.shape[0]]], fontsize=7)
            a.set_xlabel("mask scale f (FWHM)"); a.set_ylabel("poly order k")
    fig.suptitle("Hole-fill tuning metric (log10, lower is better)")
    fig.tight_layout(); fig.savefig(os.path.join(qa, "qa_background_tuning.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(cube.wave, R["bkg_info"]["r_used"] / cube.fwhm, lw=0.8)
    ax[0].set_ylabel("mask radius (FWHM)")
    ax[0].plot(cube.wave[R["bkg_info"]["capped"] > 0], (R["bkg_info"]["r_used"] / cube.fwhm)[R["bkg_info"]["capped"] > 0], "r.", ms=2, label="capped")
    ax[0].legend()
    ax[1].plot(cube.wave, R["bkg_info"]["npix"], lw=0.8); ax[1].set_ylabel("pixels used in fit")
    ax[1].set_xlabel("wavelength (um)")
    fig.tight_layout(); fig.savefig(os.path.join(qa, "qa_background_mask.png"), dpi=150); plt.close(fig)

    # errors
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    if R.get("err_scan") is not None:
        ax[0].plot(R["err_scan"]["wave"], R["err_scan"]["R"], ".", ms=3, color="0.65",
                   label="hole-fill scatter / formal (structure + noise)")
    o = np.argsort(R["wave"])
    ax[0].plot(R["wave"][o], R["R_white"][o], lw=1.2, label="R_white: random noise / formal (applied)")
    ax[0].plot(R["wave"][o], R["R_sys"][o], lw=1.0, label="R_sys: correlated systematic / formal")
    for lo, hi in MRS_SUBBANDS.values():
        ax[0].axvline(lo, color="k", lw=0.3, alpha=0.4)
    ax[0].set_yscale("log"); ax[0].set_ylabel("ratio to formal error"); ax[0].legend(fontsize=7)
    ax[1].semilogy(R["wave"][o], R["err_formal"][o], lw=0.7, label="formal")
    ax[1].semilogy(R["wave"][o], R["err"][o], lw=0.7, label="random (applied)")
    ax[1].semilogy(R["wave"][o], np.maximum(R["err_sys"][o], 1e-12), lw=0.7, label="systematic")
    ax[1].semilogy(R["wave"][o], np.abs(R["flux"][o]), lw=0.4, color="0.7", label="|flux|")
    ax[1].legend(fontsize=7, ncol=4); ax[1].set_ylabel(f"({R['unit']})"); ax[1].set_xlabel("wavelength (um)")
    fig.tight_layout(); fig.savefig(os.path.join(qa, "qa_errors.png"), dpi=150); plt.close(fig)

    # spectrum
    w, f, e, fl = R["wave"], R["flux"], R["err"], R["flag"]
    fig, ax = plt.subplots(2, 2, figsize=(14, 7))
    for a, (nm, lo, hi) in zip(ax.ravel(), [("ch1", 4.9, 7.7), ("ch2", 7.5, 11.8), ("ch3", 11.5, 18.1), ("ch4", 17.6, 28.7)]):
        m = (w >= lo) & (w <= hi)
        k = m & ((fl & CLEAN_EXCLUDE) == 0)
        a.fill_between(w[k], (f - e)[k], (f + e)[k], alpha=0.3, lw=0)
        a.plot(w[k], f[k], lw=0.6)
        bad = m & ((fl & CLEAN_EXCLUDE) != 0)
        if bad.any():
            a.plot(w[bad], f[bad], "x", color="C3", ms=3, label="flagged")
        cap = m & ((fl & FLAG["BKG_CAPPED"]) != 0)
        a.set_title(f"{nm}" + (" (bkg mask capped in part)" if cap.any() else ""), fontsize=9)
        if k.any():
            lim = np.nanpercentile(np.abs(f[k]), 99.5) * 1.3
            a.set_ylim(-0.1 * lim, lim)
        a.set_xlabel("wavelength (um)"); a.set_ylabel(f"flux ({R['unit']})")
    fig.tight_layout(); fig.savefig(os.path.join(qa, "qa_spectrum.png"), dpi=150); plt.close(fig)

    # companion
    comp = rep.get("companion")
    if comp and comp.get("rows"):
        rows = comp["rows"]
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
        subs = [r for r in rows if r["band"] != "full"]
        if subs:
            ax[0].plot([r["wave"] for r in subs], [r["sep_px"] for r in subs], "o-")
            ax[0].axhline(rows[0]["sep_px"], color="r", ls="--")
            ax[0].set_ylabel("separation (px)")
            ax[1].plot([r["wave"] for r in subs], [r["pa"] for r in subs], "o-"); ax[1].set_ylabel("PA (deg, axial)")
            ax[2].plot([r["wave"] for r in subs], [r["dbic"] for r in subs], "o-"); ax[2].axhline(cfg.dbic_min, color="r", ls="--")
            ax[2].set_ylabel("dBIC (1->2)")
        for a in ax: a.set_xlabel("sub-band wavelength (um)")
        fig.suptitle(f"Companion test: {comp['verdict']}")
        fig.tight_layout(); fig.savefig(os.path.join(qa, "qa_companion.png"), dpi=150); plt.close(fig)


def plot_final_spectrum(cfg, R, binned=None):
    """Publication-style plot of the CLEAN spectrum: full range, error band,
    sub-band edges, S/N panel.  Saved as PNG + PDF next to the products."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w, f, e, fl = R["wave"], R["flux"], R["err"], R["flag"]
    k = ((fl & CLEAN_EXCLUDE) == 0) & np.isfinite(f) & np.isfinite(e)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))
    et = R.get("err_tot", e)
    ax[0].fill_between(w[k], (f - et)[k], (f + et)[k], color="C0", alpha=0.12, lw=0, label="random + systematic")
    ax[0].fill_between(w[k], (f - e)[k], (f + e)[k], color="C0", alpha=0.3, lw=0, label="random")
    ax[0].plot(w[k], f[k], color="C0", lw=0.6, label="clean spectrum")
    if binned is not None:
        ax[0].errorbar(binned[0], binned[1], binned[2], fmt="o", ms=3, color="C3", lw=0.8,
                       label=f"binned x{cfg.bin_factor}")
    ax[0].axhline(0, color="k", lw=0.5)
    for c in CHANNEL_SPLITS:
        for a in ax:
            a.axvline(c, color="0.6", lw=0.5, ls=":")
    if k.any():
        hi = np.nanpercentile(f[k] + e[k], 99.5)
        lo = np.nanpercentile(f[k] - e[k], 0.5)
        lo, hi = min(lo, 0.0), hi
        pad = 0.08 * (hi - lo)
        ax[0].set_ylim(lo - pad, hi + pad)
    ax[0].set_ylabel(f"flux ({R['unit']})")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title(f"{cfg.product_name}: final extracted spectrum (observed frame)")
    with np.errstate(divide="ignore", invalid="ignore"):
        ax[1].plot(w[k], (f / e)[k], color="C2", lw=0.5)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel("S/N"); ax[1].set_xlabel("wavelength (um)")
    fig.tight_layout()
    base = os.path.join(cfg.out_dir, f"{cfg.product_name}_final_spectrum")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    plt.close(fig)
    return base + ".png"


# --------------------------------------------------------------------------
# state / caching
# --------------------------------------------------------------------------
class State:
    def __init__(self, out_dir):
        self.path = os.path.join(out_dir, "state.json")
        self.d = json.load(open(self.path)) if os.path.exists(self.path) else {}

    @staticmethod
    def h(params):
        return hashlib.sha1(json.dumps(_jsonable(params), sort_keys=True, default=str).encode()).hexdigest()[:12]

    def get(self, key, params, force):
        e = self.d.get(key)
        if e and e["hash"] == self.h(params) and key not in force:
            return e["result"]
        return None

    def put(self, key, params, result):
        self.d[key] = dict(hash=self.h(params), result=_jsonable(result))
        json.dump(self.d, open(self.path, "w"), indent=1)


def _file_id(p):
    return (p, os.path.getsize(p), int(os.path.getmtime(p))) if p and os.path.exists(p) else (p,)


def refine_position(mp, cfg, cen_cube):
    """Centroid, then re-centroid from the refined position until it stops moving.
    Guards the expensive background stage against a poor initial guess."""
    x0, y0 = cfg.x0, cfg.y0
    hist = []
    for it in range(max(cfg.pos_max_iter, 1)):
        cen = mp.find_source_centroid(cen_cube, x0=x0, y0=y0, wave_range=cfg.pos_wave_range, n_sub=cfg.n_sub)
        pos = robust_position(cen)
        step = np.hypot(pos["x"] - x0, pos["y"] - y0) if x0 is not None else np.inf
        hist.append((x0, y0, pos["x"], pos["y"], float(step)))
        log(f"  iter {it}: guess=({x0}, {y0}) -> ({pos['x']:.3f}, {pos['y']:.3f})  moved {step:.2f}px")
        x0, y0 = pos["x"], pos["y"]
        if step < cfg.pos_tol:
            break
    else:
        log(f"  WARNING: position still moving ({hist[-1][4]:.2f}px) after {cfg.pos_max_iter} iterations; "
            f"inspect the position QA before trusting the background")
    return cen, pos, hist


def abi_choose_params(cfg, tune):
    ks, fs = [], []
    for ch in (1, 2, 3):
        t = tune.get(ch, {})
        if t.get("tunable"):
            ks.append(int(t["k"])); fs.append(float(t["f"]))
    k = cfg.abi_poly_k if cfg.abi_poly_k is not None else (max(set(ks), key=ks.count) if ks else cfg.k_default)
    f = cfg.abi_fwhm_scale if cfg.abi_fwhm_scale is not None else (max(fs) if fs else cfg.f_default)
    return int(k), float(f)


def run_abi_backend(cfg, cube, field, knots2d, xy, fast_bgsub, fast_info, abi_k, abi_f):
    """Run AstroBkgInterp exactly once at the refined position, with the host knots
    excluded (ERR inflated on a temporary copy), or reuse an earlier run (cfg.abi_reuse_path).
    Returns the combined background-subtracted array (ABI + capped/fast-channel substitutions)."""
    mp = _mp()
    tmp = None
    src = cfg.stage3_cube
    if cfg.abi_reuse_path:
        abi_path = cfg.abi_reuse_path
        log(f"  reusing AstroBkgInterp output: {abi_path}")
        with fits.open(abi_path) as h:
            chk = np.asarray(h["SCI"].data, dtype=np.float32)
        if chk.shape != cube.sci.shape:
            raise SystemExit("abi_reuse_path cube shape does not match the stage-3 cube")
        del chk
    elif knots2d is not None and knots2d.any():
        kd = ndi.distance_transform_edt(~knots2d)
        with fits.open(cfg.stage3_cube) as h:
            new = h.copy()
            err = np.asarray(h["ERR"].data, dtype=np.float32).copy()
            for s in range(cube.nz):
                err[s][kd <= cfg.knot_pad_fwhm * cube.fwhm[s]] *= cfg.abi_err_inflate
            new["ERR"].data = err.astype(h["ERR"].data.dtype)
            tmp = os.path.join(cfg.out_dir, "_tmp_abi_input_s3d.fits")
            new.writeto(tmp, overwrite=True)
        src = tmp
    if not cfg.abi_reuse_path:
        abi_dir = os.path.join(cfg.out_dir, "abi")
        log(f"  AstroBkgInterp: fwhm_scale={abi_f}, poly_k={abi_k}, bin_size={cfg.abi_bin_size} (single run)")
        abi_path = mp.run_astrobkginterp_background(
            src, xy[0], xy[1], abi_dir, cfg.product_name, bkg_mode="polynomial", fwhm_scale=abi_f,
            poly_k=abi_k, bin_size=cfg.abi_bin_size, min_aper_rad=cfg.min_mask_px)
    if tmp and os.path.exists(tmp):
        os.remove(tmp)
    with fits.open(abi_path) as h:
        abi_sci = np.asarray(h["SCI"].data, dtype=np.float32)

    # where would the ABI mask swallow the footprint?  (ABI has no cap)
    fwhm_eff = np.maximum(cube.fwhm, cfg.min_mask_px / max(abi_f, 1e-6))
    r_abi = np.ceil(abi_f * fwhm_eff)
    capped = np.zeros(cube.nz, bool)
    for b0 in range(0, cube.nz, 64):
        idx = np.arange(b0, min(b0 + 64, cube.nz))
        _, c = cap_radii(field.V(idx), field.dsrc, r_abi[idx], masked_frac_arr(cfg, cube.chan[idx]))
        capped[idx] = c
    final = abi_sci.copy()
    n_fb = 0
    fast_sel = np.isin(cube.chan, list(cfg.fast_channels)) if cfg.fast_channels else np.zeros(cube.nz, bool)
    if fast_sel.any():
        final[fast_sel] = fast_bgsub[fast_sel]
        log(f"  channels {list(cfg.fast_channels)} taken from the built-in robust fit ({int(fast_sel.sum())} slices)")
    if cfg.abi_fallback_capped and capped.any():
        final[capped] = fast_bgsub[capped]
        n_fb = int(capped.sum())
        log(f"  ABI mask would exceed the per-channel masked-fraction cap in {n_fb} slices "
            f"(lambda >= {cube.wave[capped].min():.1f} um): using the capped robust fit there")
    info = dict(fast_info)
    use_fast = (capped & cfg.abi_fallback_capped) | fast_sel
    info["r_used"] = np.where(use_fast, fast_info["r_used"], r_abi)
    info["capped"] = np.where(use_fast, fast_info["capped"], capped.astype(float))
    info["k"] = np.where(use_fast, fast_info["k"], abi_k).astype(float)

    # agreement with the built-in fit, in source-aperture sums (native units), per channel
    cmp_ = {}
    for ch in (1, 2, 3, 4):
        idx = np.flatnonzero((cube.chan == ch) & ~capped)
        if idx.size < 5:
            continue
        idx = idx[np.linspace(0, idx.size - 1, min(60, idx.size)).astype(int)]
        d_ap = []
        f_ap = []
        for s in idx:
            ap = field.dsrc <= 2.0 * cube.fwhm[s]
            row_a = np.nan_to_num(abi_sci.reshape(cube.nz, -1)[s][field.fit.pix].astype(float))
            row_f = np.nan_to_num(fast_bgsub.reshape(cube.nz, -1)[s][field.fit.pix].astype(float))
            v = field.V(s) & ap
            d_ap.append((row_a[v] - row_f[v]).sum()); f_ap.append(row_f[v].sum())
        f_ap = np.array(f_ap)
        cmp_[ch] = dict(mean_diff_over_flux=float(np.sum(d_ap) / np.sum(np.abs(f_ap))) if np.sum(np.abs(f_ap)) > 0 else np.nan)
    return final, info, dict(abi_k=abi_k, abi_f=abi_f, n_fallback=n_fb, vs_fast=cmp_)


# --------------------------------------------------------------------------
# main driver
# --------------------------------------------------------------------------
def run(cfg: Config):
    mp = _mp()
    os.makedirs(cfg.out_dir, exist_ok=True)
    st = State(cfg.out_dir)
    force = set(cfg.force)
    rep = {"config": asdict(cfg)}
    t0 = time.time()

    if "abi_bgsub" in os.path.basename(cfg.stage3_cube).lower():
        raise SystemExit("--cube must be the raw Stage-3 cube, not an *_abi_bgsub_s3d.fits product.")

    # ---- 1. position -------------------------------------------------------
    log("[1/7] position")
    cen_cube = cfg.centroid_cube or cfg.stage3_cube
    cen, pos, pos_hist = refine_position(mp, cfg, cen_cube)
    xy = tuple(cfg.xy_fixed) if cfg.xy_fixed else (pos["x"], pos["y"])
    knots2d = knots_from_segmap(cen, xy)
    rep["position"] = dict(pos, adopted=xy, n_knot_px=int(knots2d.sum()), history=pos_hist,
                           guess=(cfg.x0, cfg.y0))
    log(f"  x={xy[0]:.3f} y={xy[1]:.3f}  err~{pos['pos_err']:.2f}px  outlier bands: {pos['outlier_bands']}")

    if cfg.position_only:
        json.dump(_jsonable(rep), open(os.path.join(cfg.out_dir, f"{cfg.product_name}_position.json"), "w"), indent=1)
        log("position-only run: stopping here.")
        return rep

    # ---- 2. cube + tuning ----------------------------------------------------
    log("[2/7] load cube, tune background (hole-fill test)")
    cube = Cube.open(cfg.stage3_cube)
    field = Field(cube, xy, knots2d, cfg)
    p_tune = dict(f=_file_id(cfg.stage3_cube), xy=xy, cfg={k: getattr(cfg, k) for k in (
        "k_grid", "f_grid", "min_mask_px", "n_fake", "tune_samples", "tune_rscale", "clip_lo", "clip_hi", "seed",
        "wing_tol")})
    tune = st.get("tune", p_tune, force) if cfg.tune else None
    if tune is None:
        if cfg.tune:
            tune = tune_background(field, cfg)
        else:
            tune = {c: dict(tunable=False, k=cfg.k_default, f=cfg.f_default, wing_frac=np.nan,
                            bias_over_sigma=np.nan) for c in (1, 2, 3, 4)}
        st.put("tune", p_tune, tune)
    tune = {int(k): v for k, v in tune.items()}
    rep["tune"] = tune

    # ---- 3. full background fit ---------------------------------------------
    log(f"[3/7] background subtraction (backend: {cfg.bkg_backend})")
    bg_path = os.path.join(cfg.out_dir, f"{cfg.product_name}_bgsub_s3d.fits")
    abi_k, abi_f = abi_choose_params(cfg, tune)
    p_bg = dict(p_tune, tune=tune, max_masked_frac=(cfg.max_masked_frac, cfg.max_masked_frac_ch123),
                knot=cfg.knot_pad_fwhm, n_iter=cfg.n_iter,
                backend=cfg.bkg_backend, abi=(abi_k, abi_f, cfg.abi_bin_size, cfg.abi_err_inflate,
                                              cfg.abi_fallback_capped, tuple(cfg.fast_channels),
                                              cfg.abi_reuse_path, cfg.bkg_smooth, cfg.bkg_smooth_win,
                                              cfg.bkg_smooth_thresh))
    info_path = bg_path + ".info.npz"
    if st.get("bg", p_bg, force) is not None and os.path.exists(bg_path) and os.path.exists(info_path):
        log("  cached (AstroBkgInterp will NOT be re-run)")
        with fits.open(bg_path) as h:
            bgsub = np.asarray(h["SCI"].data, dtype=np.float32)
        info = dict(np.load(info_path))
        rep["abi"] = (st.d.get("bg", {}).get("result") or {}).get("abi")
        rep["bkg_smooth"] = (st.d.get("bg", {}).get("result") or {}).get("bkg_smooth")
    else:
        fast_bgsub, fast_info = fit_full_cube(field, cfg, tune)
        if cfg.bkg_backend == "abi":
            bgsub, info, abi_rep = run_abi_backend(cfg, cube, field, knots2d, xy, fast_bgsub, fast_info,
                                                   abi_k, abi_f)
            rep["abi"] = abi_rep
            log("  ABI vs built-in fit, source-aperture difference / flux:",
                {c: round(v["mean_diff_over_flux"], 4) for c, v in abi_rep["vs_fast"].items()})
        else:
            bgsub, info = fast_bgsub, fast_info
        smooth_rep = spectral_smooth_bkg(cube, field, bgsub, cfg)
        rep["bkg_smooth"] = smooth_rep
        if smooth_rep:
            log("  background-model noise share / smoothed, by sub-band:",
                {k: (round(v["bkg_noise_share"], 2), v["smoothed"]) for k, v in smooth_rep.items()})
        write_bgsub_cube(cfg.stage3_cube, bg_path, bgsub, info, cube.wave, cfg)   # true stage-3 ERR
        np.savez(info_path, **info)
        st.put("bg", p_bg, dict(path=bg_path, abi=rep.get("abi"), bkg_smooth=rep.get("bkg_smooth")))
    n_cap = int((info["capped"] > 0).sum())
    log(f"  mask capped in {n_cap}/{cube.nz} slices; failed fits: {int(info['fail'].sum())}")

    # moat check with the median (what the notebook used) on a slice subsample
    samp = np.linspace(0, cube.nz - 1, 120).astype(int)
    rep["rings"] = ring_levels(bgsub, field, samp, cfg)
    log("  ring medians (bgsub):", {k: round(v, 4) for k, v in rep["rings"].items()})

    # ---- 4. companion --------------------------------------------------------
    log("[4/7] companion assessment")
    fwhm_at = lambda wl: mp._mrs_fwhm_pixels(bg_path, wave=np.array([wl]))[0]
    if cfg.components == "1":
        comp = dict(verdict="unresolved", reasons=["forced single component"], rows=[])
    else:
        comp = assess_companion(bg_path, xy, knots2d, cfg, fwhm_at)
        if cfg.components == "2":
            comp["verdict"] = "resolved"; comp["reasons"].append("forced by user")
    rep["companion"] = comp
    log(f"  verdict: {comp['verdict']}  {comp.get('reasons')}")

    # ---- 5. aperture -----------------------------------------------------------
    log("[5/7] extraction aperture")
    if cfg.rscale is not None:
        rscale, snr_tab = cfg.rscale, {}
    else:
        rscale, snr_tab = choose_rscale(field, cfg, tune)
    rep["rscale"] = dict(chosen=rscale, snr_table=snr_tab, grid=list(cfg.rscale_grid))
    log(f"  ifu_rscale = {rscale}")

    # ---- 6. extract --------------------------------------------------------------
    log("[6/7] Extract1dStep")
    wave, flux, err, _, meta = mp.extract_spectrum_manual(
        bg_path, out_file=os.path.join(cfg.out_dir, f"{cfg.product_name}_x1d.fits"), x=xy[0], y=xy[1],
        ifu_rfcorr=cfg.ifu_rfcorr, ifu_rscale=rscale, ifu_covar_scale=1.0, return_meta=True)
    unit = meta.get("flux_unit", "Jy")
    comps_out = []
    lam_res = None
    if comp["verdict"] == "resolved" and len(comp.get("positions", [])) == 2:
        # deblend only where the pair is resolvable at this wavelength
        sep = comp["sep_px"]
        okw = sep / cube.fwhm >= cfg.min_sep_fwhm
        lam_res = float(cube.wave[okw].max()) if okw.any() else None
        rep["companion"]["lam_resolved_max_um"] = lam_res
        log(f"  pair separation {sep:.2f}px is >= {cfg.min_sep_fwhm} FWHM only for lambda <= {lam_res} um")
        if lam_res:
            deb = mp.psf_deblend_cube(bg_path, comp["positions"], psf_model=cfg.psf_model, beta=cfg.beta,
                                      fwhm_scale=comp["scale"])
            deb["amplitudes"][:, deb["wave"] > lam_res] = np.nan       # beyond: leave blended
            for i, (px, py) in enumerate(comp["positions"]):
                cpath = os.path.join(cfg.out_dir, f"{cfg.product_name}_comp{i}_deblended_s3d.fits")
                mp.write_companion_subtracted_cube(bg_path, deb, i, cpath)
                w_i, f_i, e_i, _, m_i = mp.extract_spectrum_manual(
                    cpath, x=px, y=py, ifu_rfcorr=cfg.ifu_rfcorr, ifu_rscale=rscale,
                    ifu_covar_scale=1.0, return_meta=True)
                comps_out.append(dict(index=i, position=(px, py), wave=w_i, flux=f_i, err=e_i))
                os.remove(cpath)
    rep["n_finite_points"] = int(np.isfinite(flux).sum())

    # ---- 7. errors, flags, products ------------------------------------------------
    log("[7/7] empirical errors, flags, products")
    p_err = dict(p_bg, rscale=rscale, n=cfg.err_scan_n, smooth=cfg.err_smooth, floor=cfg.err_floor_ratio)
    scan_path = os.path.join(cfg.out_dir, "err_scan.npz")
    if st.get("errors", p_err, force) is not None and os.path.exists(scan_path):
        z = np.load(scan_path); scan = dict(wave=z["wave"], R=z["R"], bias_over_sigma=z["b"])
    else:
        # errors are measured on the ORIGINAL cube (holes are re-made inside holefill)
        scan = error_scan(field, cfg, tune, rscale)
        if scan is not None:
            np.savez(scan_path, wave=scan["wave"], R=scan["R"], b=scan["bias_over_sigma"])
        st.put("errors", p_err, dict(ok=scan is not None))
    # white (random) noise from the spectrum itself; correlated systematic from the hole-fill scan
    flag0 = flag_spectrum(wave, flux, cfg)
    R_white = white_noise_ratio(wave, flux, err, flag0, cfg)
    R_sys, sys_extrap = hole_fill_sys_ratio(scan, wave, R_white, cfg)
    if scan is None:
        log("  WARNING: no usable fake apertures; no hole-fill systematic available")
    rep["error_model"] = dict(
        R_white_median=float(np.median(R_white)), R_white_range=[float(R_white.min()), float(R_white.max())],
        R_sys_median=float(np.median(R_sys)), R_hole_raw_max=float(np.nanmax(scan["R"])) if scan else None,
        last_scan_wave=float(scan["wave"].max()) if scan else None)
    log(f"  random-noise inflation R_white: median {np.median(R_white):.2f} "
        f"(range {R_white.min():.2f}-{R_white.max():.2f});  systematic R_sys median {np.median(R_sys):.1f}")

    def finalize(w, f, e_formal):
        Rw = np.interp(w, wave[np.argsort(wave)], R_white[np.argsort(wave)])
        Rs = np.interp(w, wave[np.argsort(wave)], R_sys[np.argsort(wave)])
        wing = np.array([tune[int(c)].get("wing_frac", 0.0) or 0.0 for c in channel_of(w)])
        wing = np.where(np.isfinite(wing), np.clip(wing, 0.0, 0.9), 0.0)
        # Correct the flux for the measured mask-bias, rather than only inflating
        # the error bar for it: `wing` is the fraction of the aperture flux the
        # background fit is estimated to be removing at the chosen mask size, so
        # dividing it back out is the direct de-biasing step (see tune_background).
        # It's still folded into e_sys too, since the correction itself is uncertain
        # (bounded by whatever f_grid could test) -- not a reason to leave it in flux.
        f_corr = np.nan_to_num(f) / (1.0 - wing)
        e_rand = e_formal * Rw
        e_sys = np.sqrt((e_formal * Rs) ** 2 + (wing * np.abs(f_corr)) ** 2)
        fl = flag_spectrum(w, f_corr, cfg)
        o = np.argsort(cube.wave)
        j = np.clip(np.searchsorted(cube.wave[o], w), 0, cube.nz - 1)
        fl[info["capped"][o][j] > 0] |= FLAG["BKG_CAPPED"]
        fl[info["fail"][o][j] > 0] |= FLAG["BKG_FAIL"]
        if scan is not None:
            fl[(w > scan["wave"].max() + 0.05) | (w < scan["wave"].min() - 0.05)] |= FLAG["SYS_EXTRAP"]
        return f_corr, e_rand, e_sys, fl

    flux, e_rand, e_sys, flag = finalize(wave, flux, err)
    log("  wing-bias flux correction applied by channel (1/(1-wing)): " +
        ", ".join(f"ch{c}: {1.0 / (1.0 - (tune[c].get('wing_frac', 0.0) or 0.0)):.3f}x" for c in (1, 2, 3, 4) if c in tune))
    e_tot = np.sqrt(e_rand ** 2 + e_sys ** 2)
    if lam_res is not None:
        flag[wave > lam_res] |= FLAG["BLENDED"]
    R = dict(wave=wave, flux=flux, err=e_rand, err_sys=e_sys, err_tot=e_tot, flag=flag, unit=unit,
             bkg_info=info, R_white=R_white, R_sys=R_sys, err_scan=scan, err_formal=err)
    rep["stitch"] = stitch_steps(wave, flux, flag)
    for r_ in rep["stitch"]:
        if np.isfinite(r_["step_pct"]) and abs(r_["step_pct"]) > max(3 * r_["err_pct"], 5.0):
            log(f"  join {r_['join']} @ {r_['centre']:.2f} um: step {r_['step_pct']:+.1f}% +/- {r_['err_pct']:.1f}% "
                f"({'featureless' if r_['clean'] else 'feature-confounded, do not rescale on this alone'})")
    tmeta = dict(RSCALE=float(rscale), SRC_X=float(xy[0]), SRC_Y=float(xy[1]),
                 COMPVERD=comp["verdict"], COVSCALE="lambda-dep")
    try:
        ep = mp.cube_epoch_info(cfg.stage3_cube)
        tmeta.update(TARGNAME=str(ep.get("target")), MJD=ep.get("mjd") or 0.0)
    except Exception:
        pass
    base = save_products(cfg, wave, flux, e_rand, e_sys, err, flag, unit, tmeta)
    for c in comps_out:
        flux_c, er_c, es_c, fl_c = finalize(c["wave"], c["flux"], c["err"])
        if lam_res is not None:
            fl_c = fl_c.copy(); fl_c[c["wave"] > lam_res] |= FLAG["BLENDED"]
        save_products(cfg, c["wave"], flux_c, er_c, es_c, c["err"], fl_c, unit, dict(tmeta, COMPNUM=c["index"]),
                      tag=f"_comp{c['index']}")
    b = bin_red(wave, flux, e_rand, flag, cfg, e_sys)
    if b is not None:
        from astropy.table import Table
        Table({"WAVELENGTH": b[0], "FLUX": b[1], "FLUX_ERROR": b[2]}).write(
            base + "_binned_red.ecsv", format="ascii.ecsv", overwrite=True)
    rep["flags"] = {k: int(((flag & v) != 0).sum()) for k, v in FLAG.items()}
    rep["pp_scatter_over_err"] = pp_scatter_ratio(wave, flux, e_rand, flag)
    log("  flagged:", rep["flags"])
    log("  lag-1 point-to-point scatter / random error by channel (expect ~0.7-1; correlated samples lower it):",
        {c: round(v, 2) for c, v in rep["pp_scatter_over_err"].items()})
    for c, v in rep["pp_scatter_over_err"].items():
        if v > 1.3:
            log(f"  WARNING ch{c}: random errors look too small by ~{v:.1f}x vs point-to-point scatter")
    rep["elapsed_s"] = time.time() - t0
    json.dump(_jsonable(rep), open(os.path.join(cfg.out_dir, f"{cfg.product_name}_report.json"), "w"), indent=1)
    if cfg.plots:
        make_plots(cfg, R, cube, field, rep)
        log("  final spectrum plot:", plot_final_spectrum(cfg, R, b))
    log(f"done in {rep['elapsed_s']:.0f}s -> {base}_clean.ecsv")
    return rep

def celestial_wcs(header):
    """2-D (RA, Dec) WCS from a cube header, ignoring the WAVE-TAB spectral axis."""
    h = header.copy()
    for k in list(h.keys()):
        if re.match(r"^(CTYPE|CUNIT|CRVAL|CRPIX|CDELT|CRDER|CSYER|CNAME)3$|^P[SV]3_\d+$", k):
            del h[k]
        m = re.match(r"^(PC|CD)(\d)_(\d)$", k)
        if m and "3" in (m.group(2), m.group(3)):
            del h[k]
    h["WCSAXES"] = 2
    return WCS(h, naxis=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", required=True)
    ap.add_argument("--centroid-cube", default=None)
    ap.add_argument("--out-dir", default="refined")
    ap.add_argument("--name", default="target")
    ap.add_argument("--x0", type=float); ap.add_argument("--y0", type=float)
    ap.add_argument("--ra", type=float, default=None,
                     help="Target RA in decimal degrees, used instead of --x0/--y0. "
                          "Converted to a pixel position via the centroid/stage-3 cube's "
                          "WCS, then used as the initial centroid guess exactly like --x0/--y0. "
                          "Must be given together with --dec.")
    ap.add_argument("--dec", type=float, default=None,
                     help="Target Dec in decimal degrees, counterpart to --ra.")
    ap.add_argument("--xy-fixed", type=float, nargs=2, default=None)
    ap.add_argument("--z", type=float, default=0.0, help="host redshift for the line list")
    ap.add_argument("--rscale", type=float, default=None)
    ap.add_argument("--components", choices=["auto", "1", "2"], default="auto")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--lam-max-clean", type=float, default=27.0)
    ap.add_argument("--bin-from", type=float, default=None, help="also write a binned product above this um")
    ap.add_argument("--bin-factor", type=int, default=8)
    ap.add_argument("--bkg-backend", choices=["abi", "fast"], default="abi",
                    help="abi = AstroBkgInterp run once with tuned settings (default); fast = built-in robust fit")
    ap.add_argument("--abi-fwhm-scale", type=float, default=None)
    ap.add_argument("--abi-poly-k", type=int, default=None)
    ap.add_argument("--abi-bin-size", type=int, default=9)
    ap.add_argument("--position-only", action="store_true", help="stop after the position stage")
    ap.add_argument("--reuse-abi", default=None, help="path to an existing AstroBkgInterp bgsub cube: skip the slow run")
    ap.add_argument("--fast-channels", default="", help="comma list of channels taken from the built-in fit, e.g. 4")
    ap.add_argument("--bkg-smooth", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--force", default="", help="comma list: tune,bg,errors")
    a = ap.parse_args()
    if (a.ra is None) != (a.dec is None):
        ap.error("--ra and --dec must be given together.")
    x0, y0 = a.x0, a.y0
    if a.ra is not None:
        if a.x0 is not None or a.y0 is not None:
            ap.error("give either --ra/--dec or --x0/--y0, not both.")
        wcs_cube = a.centroid_cube or a.cube
        x0, y0 = radec_to_xy(wcs_cube, a.ra, a.dec)
        log(f"--ra/--dec {a.ra:.6f}, {a.dec:.6f} -> x0={x0:.3f}, y0={y0:.3f} "
            f"(via WCS of {os.path.basename(wcs_cube)})")
    cfg = Config(stage3_cube=a.cube, centroid_cube=a.centroid_cube, out_dir=a.out_dir, product_name=a.name,
                 x0=x0, y0=y0, xy_fixed=tuple(a.xy_fixed) if a.xy_fixed else None, redshift=a.z,
                 rscale=a.rscale, components=a.components, tune=not a.no_tune, lam_max_clean=a.lam_max_clean,
                 bin_from_um=a.bin_from, bin_factor=a.bin_factor, bkg_backend=a.bkg_backend,
                 abi_fwhm_scale=a.abi_fwhm_scale, abi_poly_k=a.abi_poly_k, abi_bin_size=a.abi_bin_size,
                 position_only=a.position_only, abi_reuse_path=a.reuse_abi, bkg_smooth=a.bkg_smooth,
                 fast_channels=tuple(int(c) for c in a.fast_channels.split(",") if c),
                 force=tuple(s for s in a.force.split(",") if s))
    run(cfg)


if __name__ == "__main__":
    main()