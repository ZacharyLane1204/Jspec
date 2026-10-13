"""
jspec.fitting
=============

Spectral feature fitting by nested sampling, in velocity space.

Scope
-----

Fit one feature at a time: isolate a window around a rest-frame wavelength,
model the local continuum *jointly* with up to three line components, convolve
the model with the instrumental line-spread function, and sample the posterior
with `dynesty <https://dynesty.readthedocs.io>`_. The result carries parameter
posteriors, derived velocities and widths, the Bayesian evidence, and enough
provenance to reproduce it.

Design decisions, and why
-------------------------

**Everything is parameterized in velocity, not wavelength.** A line's physical
content is its velocity centroid and velocity width; expressing the fit in
microns means the same physical line has different parameters at 2 um and
25 um, the priors have to be rewritten for each line, and an LSF correction
becomes wavelength-dependent bookkeeping. In velocity space the priors are
physical ("the ejecta are moving at between 100 and 20000 km/s") and transfer
unchanged across the whole spectrum.

**The continuum is fitted simultaneously, not divided out first.** Pre-dividing
by a continuum treats that continuum as exactly known. It is not, and the
line flux and width are both strongly covariant with it -- the dominant
systematic in any equivalent-width measurement. Fitting a low-order polynomial
jointly propagates that uncertainty into every derived quantity, which is why
the reported flux errors here are larger, and more honest, than a
fit-to-normalised-spectrum would give.

**The LSF is applied by convolution, every time.** A fitted width is not a
physical width until the instrument is removed. For a pure Gaussian this is a
quadrature subtraction, but for a Lorentzian, Moffat or asymmetric profile
there is no such shortcut -- and those are precisely the profiles used for
lines where the wings matter. The model is therefore built on a fine uniform
velocity grid, convolved numerically with the local Gaussian LSF, and
interpolated to the data. The reported widths are intrinsic by construction;
no post-hoc deconvolution is applied or needed.

**An error-inflation parameter is fitted by default.** Pipeline error arrays
for these products are optimistic -- they do not include residual background
structure, fringe residuals, or the correlated noise from cube building. A fit
that takes them at face value returns parameter uncertainties that are too
small and an evidence that over-rewards extra components. The nuisance
parameter ``lnf`` inflates the errors as ``sigma_eff^2 = sigma^2 + (f * model)^2``
and is marginalised over. Its posterior is also a diagnostic: a large fitted
``f`` means either the errors really are underestimated, or the model is
wrong.

**Model selection is by evidence, with an explicit threshold.** More
components always fit better. :func:`fit_and_select` adds a component only
when it improves ``ln Z`` by more than ``delta_lnz`` (default 5, i.e. odds of
about 150:1). Reporting the best-chi2 model without that penalty is how
two-component fits get claimed for one-component data.

**Label switching is broken by an ordering prior.** Multi-component fits are
invariant under permuting the components, which splits the posterior into
identical modes and wrecks both the sampling efficiency and the
interpretability of the marginals. Components are forced to be ordered by
increasing width, so component 0 is always the narrowest. This is a
reparameterisation, not an extra assumption.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import numpy as np

from .spectrum import C_KMS, Spectrum, instrumental_sigma_kms, resolving_power

__all__ = [
    "PROFILES",
    "FitResult",
    "profile",
    "fit_feature",
    "fit_and_select",
    "compare_models",
]

_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


# ---------------------------------------------------------------------------
# profile shapes, all defined on a velocity axis
# ---------------------------------------------------------------------------
def _gaussian(v, v0, sigma):
    z = (v - v0) / sigma
    return np.exp(-0.5 * z * z)


def _lorentzian(v, v0, gamma):
    """Normalised to unit peak. ``gamma`` is the HWHM, so FWHM = 2*gamma."""
    z = (v - v0) / gamma
    return 1.0 / (1.0 + z * z)


def _moffat(v, v0, gamma, beta):
    """Unit-peak Moffat. Interpolates between Gaussian (beta -> inf) and
    Lorentzian-like heavy wings (beta -> 1), which is exactly the freedom
    wanted when a line's wings are neither purely thermal nor purely
    radiative. FWHM = 2 * gamma * sqrt(2^(1/beta) - 1)."""
    z = (v - v0) / gamma
    return (1.0 + z * z) ** (-beta)


def _voigt(v, v0, sigma, gamma):
    """Voigt profile, unit peak: a Gaussian core with Lorentzian wings.

    The physically-motivated 'combination' profile -- thermal/turbulent
    broadening convolved with a natural or scattering wing. Implemented via
    ``scipy.special.wofz`` rather than a pseudo-Voigt sum, because the
    pseudo-Voigt approximation misrepresents the wings at the few-percent
    level, and the wings are the entire reason to choose a Voigt.
    """
    from scipy.special import wofz
    s = max(sigma, 1e-12)
    z = ((v - v0) + 1j * gamma) / (s * np.sqrt(2.0))
    w = np.real(wofz(z))
    z0 = (1j * gamma) / (s * np.sqrt(2.0))
    w0 = np.real(wofz(z0))
    return w / max(w0, 1e-300)


def _skew_gaussian(v, v0, sigma, alpha):
    """Skew-normal, unit peak. ``alpha`` is the shape parameter: 0 is
    Gaussian, positive skews the tail to the red, negative to the blue.

    Note the mode is *not* at ``v0`` for non-zero ``alpha`` -- ``v0`` is the
    location parameter. The reported velocity in :class:`FitResult` is the
    profile's actual centroid, computed numerically, not the raw ``v0``;
    quoting the location parameter as 'the velocity' of a skewed line is a
    common and significant error.
    """
    from scipy.special import erf
    z = (v - v0) / sigma
    g = np.exp(-0.5 * z * z) * (1.0 + erf(alpha * z / np.sqrt(2.0)))
    m = np.max(g)
    return g / m if m > 0 else g


def _asym_gaussian(v, v0, sigma, asym):
    """Bi-Gaussian: different widths blue and red of the peak.

    ``asym`` is the ratio of red-side to blue-side sigma. Unlike the
    skew-normal this keeps the *peak* at ``v0`` exactly, which makes it the
    better choice when the line centroid is the quantity of interest and the
    asymmetry is a nuisance (e.g. mild self-absorption on one wing).
    """
    s_blue = sigma * 2.0 / (1.0 + asym)
    s_red = sigma * 2.0 * asym / (1.0 + asym)
    s = np.where(v < v0, s_blue, s_red)
    z = (v - v0) / np.clip(s, 1e-12, None)
    return np.exp(-0.5 * z * z)


def _asym_lorentzian(v, v0, gamma, asym):
    """Bi-Lorentzian: heavy wings with different widths on each side."""
    g_blue = gamma * 2.0 / (1.0 + asym)
    g_red = gamma * 2.0 * asym / (1.0 + asym)
    g = np.where(v < v0, g_blue, g_red)
    z = (v - v0) / np.clip(g, 1e-12, None)
    return 1.0 / (1.0 + z * z)


def _exp_gaussian(v, v0, sigma, tau):
    """Exponentially modified Gaussian: a Gaussian core with a one-sided
    exponential tail of e-folding velocity ``tau``.

    This is the natural shape for an optically-thin line seen through an
    outflow or an expanding scattering medium, and is usually a better
    description of a red- or blue-tailed SN line than a skew-normal, because
    its tail is exponential rather than Gaussian-times-erf. Positive ``tau``
    gives a red tail.
    """
    from scipy.special import erfc
    t = max(abs(tau), 1e-9)
    sgn = 1.0 if tau >= 0 else -1.0
    x = sgn * (v - v0)
    arg = (sigma * sigma / (2.0 * t * t)) - (x / t)
    arg = np.clip(arg, -700.0, 700.0)
    y = np.exp(arg) * erfc((sigma / t - x / sigma) / np.sqrt(2.0))
    m = np.max(y)
    return y / m if m > 0 else y


#: Available profiles: name -> (function, extra parameter names, prior bounds
#: for those extras). Every profile takes ``(v, v0, width, *extras)`` and
#: returns a unit-peak shape.
PROFILES = {
    "gaussian": (_gaussian, (), {}),
    "lorentzian": (_lorentzian, (), {}),
    "moffat": (_moffat, ("beta",), {"beta": (0.6, 8.0)}),
    "voigt": (_voigt, ("gamma_kms",), {"gamma_kms": (1.0, 5000.0)}),
    "skew_gaussian": (_skew_gaussian, ("alpha",), {"alpha": (-8.0, 8.0)}),
    "asym_gaussian": (_asym_gaussian, ("asym",), {"asym": (0.25, 4.0)}),
    "asym_lorentzian": (_asym_lorentzian, ("asym",), {"asym": (0.25, 4.0)}),
    "exp_gaussian": (_exp_gaussian, ("tau_kms",), {"tau_kms": (-20000.0, 20000.0)}),
}


def profile(name, v, v0, width, *extras):
    """Evaluate a named profile on a velocity axis. Unit peak amplitude."""
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; available: {sorted(PROFILES)}")
    fn, extra_names, _ = PROFILES[name]
    if len(extras) != len(extra_names):
        raise TypeError(f"{name} needs extras {extra_names}, got {len(extras)}")
    return fn(v, v0, width, *extras)


def profile_list(profile_name, n_components=None):
    """Expand a profile specification into one name per component.

    A plain name (``"moffat"``) means every component has that shape. A
    ``+``-joined specification (``"gaussian+moffat"``) assigns a shape to each
    component in turn, which is how you fit a mixed model: a narrow Gaussian
    core on a Moffat's broad wings, say, where neither shape alone describes
    the feature. The component count is then fixed by the specification.
    """
    if isinstance(profile_name, (list, tuple)):
        parts = list(profile_name)
    else:
        parts = [q.strip() for q in str(profile_name).split("+") if q.strip()]
    for q in parts:
        if q not in PROFILES:
            raise KeyError(f"unknown profile {q!r}; available: {sorted(PROFILES)}")
    if len(parts) == 1 and n_components:
        parts = parts * n_components
    if n_components is not None and len(parts) != n_components:
        raise ValueError(
            f"profile {profile_name!r} specifies {len(parts)} component(s) "
            f"but n_components={n_components}; drop n_components or match it")
    return parts


def _profile_fwhm_kms(name, width, extras):
    """Analytic FWHM of a unit profile, where one exists."""
    if name in ("gaussian", "skew_gaussian"):
        return _FWHM * width
    if name == "asym_gaussian":
        return _FWHM * width
    if name in ("lorentzian", "asym_lorentzian"):
        return 2.0 * width
    if name == "moffat":
        beta = extras[0]
        return 2.0 * width * np.sqrt(max(2.0 ** (1.0 / beta) - 1.0, 1e-12))
    if name == "voigt":
        fg, fl = _FWHM * width, 2.0 * extras[0]
        # Olivero & Longbothe (1977) approximation, accurate to ~0.02%
        return 0.5346 * fl + np.sqrt(0.2166 * fl * fl + fg * fg)
    return np.nan


# ---------------------------------------------------------------------------
# model evaluation
# ---------------------------------------------------------------------------
def _model_on_grid(vgrid, components, profile_names):
    """Sum of unit-peak components scaled by their amplitudes.

    ``profile_names`` is one shape per component, so a model may mix shapes.
    """
    if isinstance(profile_names, str):
        profile_names = [profile_names] * len(components)
    out = np.zeros_like(vgrid)
    for nm, (amp, v0, width, extras) in zip(profile_names, components):
        fn = PROFILES[nm][0]
        out = out + amp * fn(vgrid, v0, width, *extras)
    return out


def _convolve_lsf(vgrid, y, lsf_sigma_kms):
    """Convolve a model sampled on a uniform velocity grid with a Gaussian
    LSF. Returns the convolved model on the same grid.

    The kernel is built on the same grid and normalised to unit area, and the
    convolution is done in 'same' mode with edge padding, so a feature near
    the window edge is not artificially suppressed -- a bare ``np.convolve``
    in 'same' mode tapers the first and last half-kernel of the window toward
    zero, which systematically narrows any line fitted close to the edge.
    """
    if not np.isfinite(lsf_sigma_kms) or lsf_sigma_kms <= 0:
        return y
    dv = vgrid[1] - vgrid[0]
    n_half = int(np.ceil(4.0 * lsf_sigma_kms / dv))
    if n_half < 1:
        return y
    kx = np.arange(-n_half, n_half + 1) * dv
    k = np.exp(-0.5 * (kx / lsf_sigma_kms) ** 2)
    k /= k.sum()
    pad = np.pad(y, n_half, mode="edge")
    return np.convolve(pad, k, mode="valid")


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------
def _fmt_derived(key, value):
    """One line of the derived-quantity block.

    Values are either a (median, lo, hi) interval or a bare scalar, and after
    a save/load round trip the interval arrives as a list rather than a tuple,
    so both sequence types have to be accepted. Non-numeric entries (the
    per-component profile name) are printed as text.
    """
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return (f"      {key:<26}{value[0]:>12.4g}  "
                f"[{value[1]:.4g}, {value[2]:.4g}]")
    if isinstance(value, str):
        return f"      {key:<26}{value:>12}"
    try:
        return f"      {key:<26}{value:>12.4g}"
    except (TypeError, ValueError):
        return f"      {key:<26}{value!s:>12}"


@dataclass
class FitResult:
    """Posterior summary of one feature fit."""
    rest_wavelength_um: float
    profile: str
    n_components: int
    logz: float
    logz_err: float
    samples: np.ndarray                 # equally-weighted posterior samples
    param_names: list
    best: dict                          # median parameter values
    lo: dict                            # 16th percentile
    hi: dict                            # 84th percentile
    derived: dict                       # per-component physical quantities
    velocity: np.ndarray                # data velocity axis
    flux: np.ndarray
    flux_err: np.ndarray
    model: np.ndarray                   # median model, including continuum
    continuum: np.ndarray
    components: list                    # per-component median model curves
    chi2: float
    n_data: int
    lsf_sigma_kms: float
    meta: dict = field(default_factory=dict)

    @property
    def reduced_chi2(self) -> float:
        dof = max(self.n_data - len(self.param_names), 1)
        return self.chi2 / dof

    def summary(self) -> str:
        """Human-readable parameter and derived-quantity report."""
        L = [f"Fit: {self.profile}, {self.n_components} component(s) "
             f"at {self.rest_wavelength_um:.5f} um (rest)",
             f"  ln Z = {self.logz:.2f} +/- {self.logz_err:.2f}   "
             f"chi2/dof = {self.reduced_chi2:.2f}   N = {self.n_data}",
             f"  instrumental LSF sigma = {self.lsf_sigma_kms:.1f} km/s "
             f"(FWHM {self.lsf_sigma_kms * _FWHM:.1f} km/s)",
             ""]
        L.append(f"  {'parameter':<22}{'median':>12}{'-1sig':>11}{'+1sig':>11}")
        for p in self.param_names:
            L.append(f"  {p:<22}{self.best[p]:>12.4g}"
                     f"{self.best[p] - self.lo[p]:>11.3g}"
                     f"{self.hi[p] - self.best[p]:>11.3g}")
        L.append("")
        L.append("  derived (intrinsic; instrumental LSF already removed by "
                 "forward-convolution):")
        for i, d in enumerate(self.derived.get("components", [])):
            shape = d.get("profile")
            L.append(f"    component {i}:" + (f"  [{shape}]" if shape else ""))
            for k, v in d.items():
                if k != "profile":
                    L.append(_fmt_derived(k, v))
        tot = self.derived.get("total", {})
        if tot:
            L.append("    total:")
            for k, v in tot.items():
                L.append(_fmt_derived(k, v))
        f_med = self.best.get("err_inflation")
        if f_med is not None and f_med > 0.25:
            L.append("")
            L.append(f"  ! error inflation f = {f_med:.2f}: the fit needed to inflate the "
                     f"uncertainties by {100 * f_med:.0f}% of the model to be consistent. "
                     f"Either the pipeline errors are underestimated here, or this "
                     f"profile/component count is the wrong model.")
        return "\n".join(L)

    def __repr__(self):
        return (f"<FitResult {self.profile} x{self.n_components} "
                f"@{self.rest_wavelength_um:.4f}um lnZ={self.logz:.1f} "
                f"chi2r={self.reduced_chi2:.2f}>")

    def plot(self, ax=None, show_components=True, residual=True):
        """Plot data, model, components and residuals in velocity space."""
        import matplotlib.pyplot as plt

        if ax is None:
            if residual:
                fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True,
                                         gridspec_kw=dict(height_ratios=[3, 1],
                                                          hspace=0.05))
                ax, axr = axes
            else:
                fig, ax = plt.subplots(figsize=(8, 4))
                axr = None
        else:
            axr = None

        ax.errorbar(self.velocity, self.flux, yerr=self.flux_err, fmt="o",
                    ms=2.2, lw=0.7, color="0.35", alpha=0.75, zorder=1)
        ax.plot(self.velocity, self.model, color="crimson", lw=1.6,
                zorder=4, label="model")
        ax.plot(self.velocity, self.continuum, color="tab:orange", lw=1.0,
                ls="--", zorder=3, label="continuum")
        if show_components:
            for i, c in enumerate(self.components):
                ax.plot(self.velocity, self.continuum + c, lw=1.0, ls="-",
                        alpha=0.85, zorder=2, label=f"comp {i}")
        ax.axvline(0.0, color="0.7", lw=0.7, zorder=0)
        ax.set_ylabel("Flux")
        ax.legend(fontsize=7)
        ax.set_title(f"{self.profile} x{self.n_components} @ "
                     f"{self.rest_wavelength_um:.4f} um  "
                     f"(lnZ={self.logz:.1f})", fontsize=9)
        if axr is not None:
            r = (self.flux - self.model) / np.where(self.flux_err > 0,
                                                    self.flux_err, np.nan)
            axr.axhline(0, color="0.6", lw=0.7)
            axr.plot(self.velocity, r, "o", ms=2.0, color="0.35")
            axr.set_ylabel(r"$\Delta/\sigma$")
            axr.set_xlabel("Velocity [km/s]")
        else:
            ax.set_xlabel("Velocity [km/s]")
        return ax


# ---------------------------------------------------------------------------
# the fitter
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def empirical_error_scale(flux, flux_err):
    """Factor by which the quoted uncertainties understate the actual scatter.

    Pipeline uncertainties are propagated from the detector model and do not
    include the systematics that dominate a real extracted spectrum: residual
    fringing, imperfect background subtraction, flat-field residuals and the
    scale uncertainty introduced when bands are matched. On MRS they are
    routinely optimistic by a factor of two or more, which makes chi2 useless
    and makes every model look rejected.

    The scatter is estimated from second differences, which cancel any smooth
    continuum, and with a median-absolute-deviation estimator, which is
    insensitive to the handful of pixels occupied by the line itself -- so the
    feature being fitted does not inflate its own error bar.
    """
    y = np.asarray(flux, float)
    e = np.asarray(flux_err, float)
    good = np.isfinite(y) & np.isfinite(e) & (e > 0)
    if good.sum() < 12:
        return 1.0
    y, e = y[good], e[good]
    d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
    mad = 1.4826 * float(np.median(np.abs(d2 - np.median(d2))))
    sigma = mad / np.sqrt(6.0)
    med = float(np.median(e))
    if not np.isfinite(sigma) or sigma <= 0 or med <= 0:
        return 1.0
    return float(max(sigma / med, 1.0))


def save_fit(result: FitResult, path):
    """Write a complete :class:`FitResult` to a ``.npz`` file.

    Everything needed to redraw the fit is stored, not just the posterior:
    the continuum, the individual component curves and the derived physical
    quantities are all kept, so :func:`load_fit` can hand back a result you
    can re-plot (or plot your own way) without re-running the sampler.
    """
    comps = np.asarray(result.components, float) if len(result.components) \
        else np.zeros((0, result.velocity.size))
    meta = dict(rest_wavelength_um=result.rest_wavelength_um,
                profile=result.profile, n_components=result.n_components,
                logz=result.logz, logz_err=result.logz_err,
                best=result.best, lo=result.lo, hi=result.hi,
                derived=result.derived, chi2=result.chi2,
                n_data=result.n_data, lsf_sigma_kms=result.lsf_sigma_kms,
                meta=result.meta)
    np.savez(path, samples=result.samples,
             param_names=np.array(result.param_names, dtype=object),
             velocity=result.velocity, flux=result.flux,
             flux_err=result.flux_err, model=result.model,
             continuum=result.continuum, components=comps,
             meta_json=json.dumps(meta, default=float))


def load_fit(path) -> FitResult:
    """Read back a fit written by :func:`save_fit`.

    The returned object is a real :class:`FitResult`, so ``.plot()``,
    ``.summary()`` and ``.reduced_chi2`` all work as they did in the session
    that produced it.
    """
    with np.load(path, allow_pickle=True) as z:
        m = json.loads(str(z["meta_json"]))
        return FitResult(
            rest_wavelength_um=m["rest_wavelength_um"], profile=m["profile"],
            n_components=int(m["n_components"]), logz=m["logz"],
            logz_err=m["logz_err"], samples=z["samples"],
            param_names=[str(x) for x in z["param_names"]],
            best=m["best"], lo=m["lo"], hi=m["hi"], derived=m["derived"],
            velocity=z["velocity"], flux=z["flux"], flux_err=z["flux_err"],
            model=z["model"], continuum=z["continuum"],
            components=[np.asarray(c) for c in z["components"]],
            chi2=m["chi2"], n_data=int(m["n_data"]),
            lsf_sigma_kms=m["lsf_sigma_kms"], meta=m.get("meta", {}))


def fit_feature(spectrum: Spectrum, rest_wavelength_um, *, n_components=1,
                profile_name="gaussian", window_kms=None, window_um=None,
                continuum_order=1, mask_ranges=None,
                v_max=None, sigma_min_kms=None, sigma_max_kms=20000.0,
                amp_scale=5.0, allow_absorption=False, fit_error_inflation=True,
                mode=None, nlive=500, dlogz=0.01, sample="auto", bound="multi",
                rstate=None, maxiter=None, oversample=6.0, print_progress=False,
                lsf_sigma_kms=None, bootstrap=0, pixel_integrate=True,
                err_scale=None) -> FitResult:
    """Fit a single feature with nested sampling.

    Parameters
    ----------
    spectrum : Spectrum
        Must carry the correct redshift; the window and the velocity axis are
        both built in the rest frame.
    rest_wavelength_um : float
        Rest-frame wavelength defining v = 0. All fitted velocities are
        relative to this.
    n_components : int
        1 to 3. See :func:`fit_and_select` to choose this by evidence rather
        than by hand.
    profile_name : str
        A key of :data:`PROFILES`.
    window_kms : float, optional
        Half-width of the fitting window in velocity. Defaults to
        ``4 * sigma_max_kms``, wide enough that the continuum is constrained
        by genuine continuum rather than by line wings. **Too narrow a window
        is the most common way to get a confidently wrong width**: the
        polynomial continuum simply absorbs the wings.
    window_um : float, optional
        Half-width in observed microns; overrides ``window_kms``.
    continuum_order : int
        0, 1 or 2. Fitted jointly. Order 2 in a wide window starts to compete
        with a broad line for the same flux; prefer order 1 unless the local
        continuum is genuinely curved.
    mask_ranges : list of (lo, hi), optional
        Rest-frame micron ranges inside the window to exclude -- a
        contaminating neighbouring line, a bad pixel cluster.
    v_max : float, optional
        Prior half-range on each component's velocity centroid. Defaults to
        half the window.
    sigma_min_kms : float, optional
        Lower prior bound on the width. Defaults to one third of the
        instrumental sigma, which lets the sampler explore genuinely
        unresolved solutions without wandering into numerically degenerate
        zero-width territory. The width prior is log-uniform, appropriate for
        a scale parameter.
    allow_absorption : bool
        Allow negative amplitudes, for fitting dips. When False the amplitude
        prior is strictly positive, which is both faster and a real prior
        statement -- use it when you know the feature is in emission.
    fit_error_inflation : bool
        Include the ``err_inflation`` nuisance parameter (see module
        docstring). Strongly recommended.
    mode : str, optional
        Instrument mode for the LSF; defaults to per-wavelength inference.
    lsf_sigma_kms : float, optional
        Override the instrumental LSF sigma entirely. Use when fitting a
        spectrum whose effective resolution is not the nominal one, e.g.
        after smoothing or rebinning.
    nlive, dlogz, sample, bound, maxiter : dynesty controls
        ``nlive=500`` is a reasonable default for up to ~12 parameters. Raise
        it for three-component fits if the evidence error exceeds ~0.3.

    Returns
    -------
    FitResult
    """
    try:
        import dynesty
    except ImportError as exc:                           # pragma: no cover
        raise ImportError(
            "fit_feature needs dynesty: pip install dynesty") from exc

    # profile_list validates each name and accepts a '+' specification
    if len(profile_list(profile_name)) > 1:
        n_components = len(profile_list(profile_name))
    if not 1 <= n_components <= 3:
        raise ValueError("n_components must be 1, 2 or 3")
    if continuum_order not in (0, 1, 2):
        raise ValueError("continuum_order must be 0, 1 or 2")

    lam0_obs = rest_wavelength_um * (1.0 + spectrum.redshift)
    lsf_sig = (lsf_sigma_kms if lsf_sigma_kms is not None
               else float(instrumental_sigma_kms(lam0_obs, mode=mode)))
    if not np.isfinite(lsf_sig) or lsf_sig <= 0:
        warnings.warn(f"no resolving power known at {lam0_obs:.3f} um; "
                      f"fitting without LSF convolution -- the reported widths "
                      f"will NOT be intrinsic", stacklevel=2)
        lsf_sig = 0.0

    if window_um is not None:
        half_um = float(window_um)
        window_kms = C_KMS * half_um / lam0_obs
    else:
        if window_kms is None:
            window_kms = 4.0 * sigma_max_kms
        half_um = lam0_obs * window_kms / C_KMS

    sub = spectrum.select(lam0_obs - half_um, lam0_obs + half_um)
    if mask_ranges:
        sub = sub.mask_ranges(mask_ranges, rest=True)
    g = sub.good
    if g.sum() < 4 * (continuum_order + 1) + 3 * n_components:
        raise ValueError(
            f"only {int(g.sum())} usable points within +/-{window_kms:.0f} km/s "
            f"of {rest_wavelength_um:.5f} um; widen the window or lower "
            f"n_components")

    v = C_KMS * (sub.rest_wavelength[g] - rest_wavelength_um) / rest_wavelength_um
    f = sub.flux[g]
    e = sub.flux_err[g]
    order = np.argsort(v)
    v, f, e = v[order], f[order], e[order]

    # Rescale the uncertainties to the scatter actually present, if asked.
    # Without this a chi2 of 100+ per degree of freedom simply reports that
    # the pipeline errors are optimistic, and every candidate model is
    # rejected for the same reason -- which tells you nothing about which
    # model is right.
    err_scale_used = 1.0
    if err_scale is not None:
        err_scale_used = (empirical_error_scale(f, e)
                          if str(err_scale).lower() == "auto"
                          else float(err_scale))
        e = e * err_scale_used

    # Fine uniform velocity grid for model building + LSF convolution.
    dv_data = np.median(np.diff(v)) if v.size > 1 else 1.0
    dv = min(dv_data, max(lsf_sig, 1e-3) / oversample) if lsf_sig > 0 else dv_data
    dv = max(dv, (v[-1] - v[0]) / 200000.0)     # keep the grid finite
    pad = 5.0 * max(lsf_sig, 1.0)
    vgrid = np.arange(v[0] - pad, v[-1] + pad + dv, dv)

    # Pixel edges, for integrating the model across each detector pixel
    # instead of sampling it at the pixel centre. This matters whenever the
    # feature is close to the sampling scale -- which is the normal case for
    # an unresolved fine-structure line in MRS, where a resolution element is
    # only about two pixels wide. Point-sampling then biases both the
    # amplitude and the centroid, because the detector reports the *average*
    # over the pixel and the average of a sharply peaked line is not its
    # value at the centre.
    if pixel_integrate and v.size >= 2:
        v_edges = np.empty(v.size + 1)
        v_edges[1:-1] = 0.5 * (v[1:] + v[:-1])
        v_edges[0] = v[0] - 0.5 * (v[1] - v[0])
        v_edges[-1] = v[-1] + 0.5 * (v[-1] - v[-2])
        v_width = np.diff(v_edges)
        _gstep = np.diff(vgrid)

        def _to_data(y):
            cum = np.concatenate(([0.0],
                                  np.cumsum(0.5 * (y[1:] + y[:-1]) * _gstep)))
            return np.diff(np.interp(v_edges, vgrid, cum)) / v_width
    else:
        def _to_data(y):
            return np.interp(v, vgrid, y)

    # ---- priors --------------------------------------------------------
    if v_max is None:
        v_max = 0.5 * window_kms
    if sigma_min_kms is None:
        sigma_min_kms = max(lsf_sig / 3.0, 1.0)
    sigma_max_kms = min(sigma_max_kms, 0.8 * window_kms)
    if sigma_max_kms <= sigma_min_kms:
        sigma_max_kms = sigma_min_kms * 10.0

    # Continuum guess from the window edges, used only to scale the amplitude
    # prior -- not as a fixed continuum.
    edge = (np.abs(v) > 0.6 * window_kms)
    c_guess = float(np.median(f[edge])) if edge.sum() >= 3 else float(np.median(f))
    span = float(np.max(f) - np.min(f))
    a_max = amp_scale * max(span, float(np.std(f)), 1e-30)
    c_scale = max(abs(c_guess), np.std(f), 1e-30)

    # Components may each have their own shape, so the parameter block is no
    # longer a fixed stride: offs[i] is where component i starts.
    # A '+' specification carries its own component count, so honour it rather
    # than making the caller repeat the number.
    _spec = profile_list(profile_name)
    if len(_spec) > 1:
        n_components = len(_spec)
    pnames = profile_list(profile_name, n_components)

    enames = [PROFILES[nm][1] for nm in pnames]
    ebounds = [PROFILES[nm][2] for nm in pnames]
    n_per_i = [3 + len(en) for en in enames]
    offs = np.concatenate(([0], np.cumsum(n_per_i))).astype(int)
    mixed = len(set(pnames)) > 1

    names = []
    for i in range(n_components):
        names += [f"amp{i}", f"v{i}_kms", f"sigma{i}_kms"]
        names += [f"{nm}{i}" for nm in enames[i]]
    names += [f"cont{k}" for k in range(continuum_order + 1)]
    if fit_error_inflation:
        names.append("err_inflation")
    ndim = len(names)

    ln_smin, ln_smax = np.log(sigma_min_kms), np.log(sigma_max_kms)

    def prior_transform(u):
        x = np.empty(ndim)
        # Ordered widths (narrowest first) break the label-switching
        # degeneracy. Implemented by sampling ordered uniforms in log-sigma:
        # u -> sorted increasing, which is a valid reparameterisation of the
        # joint prior restricted to the ordered region.
        # With mixed shapes the components are already distinguishable, so
        # ordering them by width is unnecessary and would wrongly forbid, say,
        # a narrow Moffat sitting inside a broad Gaussian.
        uw = np.array([u[offs[i] + 2] for i in range(n_components)])
        us = np.sort(uw) if (n_components > 1 and not mixed) else uw
        for i in range(n_components):
            b = int(offs[i])
            if allow_absorption:
                x[b] = (2.0 * u[b] - 1.0) * a_max
            else:
                x[b] = u[b] * a_max
            x[b + 1] = (2.0 * u[b + 1] - 1.0) * v_max
            x[b + 2] = np.exp(ln_smin + us[i] * (ln_smax - ln_smin))
            for k, nm in enumerate(enames[i]):
                lo, hi = ebounds[i][nm]
                x[b + 3 + k] = lo + u[b + 3 + k] * (hi - lo)
        cb = int(offs[n_components])
        for k in range(continuum_order + 1):
            # Wide, zero-centred priors on the polynomial coefficients. The
            # higher-order coefficients get progressively tighter scales,
            # because a steeply-curved local continuum is physically
            # implausible and an unconstrained quadratic will eat the line.
            s = c_scale * (3.0 if k == 0 else 3.0 / (v_max ** k))
            x[cb + k] = (2.0 * u[cb + k] - 1.0) * s
        if fit_error_inflation:
            # log-uniform from 1e-3 to 1: a scale parameter spanning decades
            x[-1] = 10.0 ** (-3.0 + 3.0 * u[-1])
        return x

    vg_pow = [vgrid ** k for k in range(continuum_order + 1)]
    v_pow = [v ** k for k in range(continuum_order + 1)]

    def unpack(x):
        comps = []
        for i in range(n_components):
            b = int(offs[i])
            comps.append((x[b], x[b + 1], x[b + 2],
                          tuple(x[b + 3: b + 3 + len(enames[i])])))
        cb = int(offs[n_components])
        cc = x[cb: cb + continuum_order + 1]
        fi = x[-1] if fit_error_inflation else 0.0
        return comps, cc, fi

    def model_at_data(x, return_parts=False):
        comps, cc, _ = unpack(x)
        y = _model_on_grid(vgrid, comps, pnames)
        y = _convolve_lsf(vgrid, y, lsf_sig)
        line = _to_data(y)
        cont = sum(c * p for c, p in zip(cc, v_pow))
        if not return_parts:
            return line + cont
        parts = []
        for c in comps:
            yc = _convolve_lsf(vgrid,
                               _model_on_grid(vgrid, [c], [pnames[len(parts)]]),
                               lsf_sig)
            parts.append(_to_data(yc))
        return line + cont, cont, parts

    def loglike(x):
        m = model_at_data(x)
        if not np.all(np.isfinite(m)):
            return -1e100
        _, _, fi = unpack(x)
        var = e ** 2 + (fi * m) ** 2 if fit_error_inflation else e ** 2
        var = np.where(var > 0, var, np.inf)
        return float(-0.5 * np.sum((f - m) ** 2 / var + np.log(2.0 * np.pi * var)))

    # bootstrap=0 by default: with a sharply peaked, nearly unresolved line
    # the posterior is tight enough that dynesty's bootstrapped ellipsoid
    # enlargement becomes unstable and warns about a very large expansion
    # factor, which makes uniform sampling extremely inefficient.
    sampler = dynesty.NestedSampler(
        loglike, prior_transform, ndim, nlive=nlive, bound=bound,
        sample=sample, rstate=rstate, bootstrap=bootstrap)
    sampler.run_nested(dlogz=dlogz, maxiter=maxiter, print_progress=print_progress)
    res = sampler.results

    from dynesty.utils import resample_equal
    logwt = res.logwt - res.logz[-1]
    samples = resample_equal(res.samples, np.exp(logwt))

    med = np.median(samples, axis=0)
    p16 = np.percentile(samples, 16, axis=0)
    p84 = np.percentile(samples, 84, axis=0)
    best = dict(zip(names, med))
    lo = dict(zip(names, p16))
    hi = dict(zip(names, p84))

    model, cont, parts = model_at_data(med, return_parts=True)
    var = e ** 2 + (best.get("err_inflation", 0.0) * model) ** 2
    chi2 = float(np.sum((f - model) ** 2 / np.where(e > 0, e ** 2, np.inf)))

    derived = _derive(samples, names, n_components, offs, enames,
                      pnames, vgrid, lsf_sig, rest_wavelength_um,
                      continuum_order, fit_error_inflation)

    return FitResult(
        rest_wavelength_um=float(rest_wavelength_um), profile="+".join(pnames),
        n_components=n_components, logz=float(res.logz[-1]),
        logz_err=float(res.logzerr[-1]), samples=samples, param_names=names,
        best=best, lo=lo, hi=hi, derived=derived, velocity=v, flux=f,
        flux_err=e, model=model, continuum=cont, components=parts,
        chi2=chi2, n_data=int(v.size), lsf_sigma_kms=float(lsf_sig),
        meta=dict(window_kms=float(window_kms), continuum_order=continuum_order,
                  err_scale=float(err_scale_used),
                  allow_absorption=allow_absorption, nlive=nlive,
                  redshift=spectrum.redshift, mode=mode))


def _derive(samples, names, n_components, offs, enames, pnames,
            vgrid, lsf_sig, lam0, continuum_order, fit_error_inflation):
    """Posterior summaries of physical quantities, computed per sample.

    Computing these from the samples (rather than from the median parameters)
    is what makes the quoted intervals correct for quantities that are
    non-linear in the parameters -- integrated flux and FWHM both are, and
    evaluating them at the median parameter vector would give a point estimate
    with no honest interval at all.
    """
    ns = samples.shape[0]
    take = np.linspace(0, ns - 1, min(ns, 2000)).astype(int)
    S = samples[take]

    out = {"components": [], "total": {}}
    fluxes_total = np.zeros(S.shape[0])
    for i in range(n_components):
        b = int(offs[i])
        extra_names = enames[i]
        profile_name = pnames[i]
        amp = S[:, b]
        v0 = S[:, b + 1]
        wid = S[:, b + 2]
        extras = [S[:, b + 3 + k] for k in range(len(extra_names))]

        fw = np.array([_profile_fwhm_kms(profile_name, wid[j],
                                         [ex[j] for ex in extras])
                       for j in range(S.shape[0])])
        # Integrated flux in velocity units: integral of the (unconvolved)
        # profile. The LSF conserves area, so convolution does not change it.
        area = np.empty(S.shape[0])
        centroid = np.empty(S.shape[0])
        for j in range(S.shape[0]):
            y = profile(profile_name, vgrid, v0[j], wid[j],
                        *[ex[j] for ex in extras]) * amp[j]
            a = np.trapezoid(y, vgrid) if hasattr(np, "trapezoid") else np.trapz(y, vgrid)
            area[j] = a
            sw = np.sum(np.abs(y))
            centroid[j] = np.sum(vgrid * np.abs(y)) / sw if sw > 0 else v0[j]
        fluxes_total += area

        def q(a):
            return (float(np.median(a)), float(np.percentile(a, 16)),
                    float(np.percentile(a, 84)))

        d = {
            "profile": profile_name,
            "amplitude": q(amp),
            "velocity_kms (peak/location)": q(v0),
            "velocity_kms (flux centroid)": q(centroid),
            "sigma_kms": q(wid),
            "fwhm_kms (intrinsic)": q(fw),
            "flux (flux_unit x km/s)": q(area),
            "resolved": float(np.mean(fw > _FWHM * lsf_sig)) if lsf_sig > 0 else 1.0,
        }
        for k, nm in enumerate(extra_names):
            d[nm] = q(extras[k])
        out["components"].append(d)

    out["total"]["flux (flux_unit x km/s)"] = (
        float(np.median(fluxes_total)), float(np.percentile(fluxes_total, 16)),
        float(np.percentile(fluxes_total, 84)))
    if fit_error_inflation:
        out["total"]["err_inflation"] = (
            float(np.median(S[:, -1])), float(np.percentile(S[:, -1], 16)),
            float(np.percentile(S[:, -1], 84)))
    return out


# ---------------------------------------------------------------------------
# model selection
# ---------------------------------------------------------------------------
def compare_models(results, delta_lnz=5.0):
    """Rank fits by evidence and pick a preferred model.

    ``delta_lnz=5`` corresponds to a Bayes factor of about 150, conventionally
    "strong" evidence. A simpler model is preferred unless a more complex one
    beats it by more than this -- complexity is only adopted when it is paid
    for.

    Returns ``(best_result, table)`` where ``table`` is a list of dicts sorted
    by evidence, each carrying ``delta_lnz`` relative to the best.
    """
    if not results:
        raise ValueError("no results to compare")
    rs = sorted(results, key=lambda r: -r.logz)
    top = rs[0]
    # Among models statistically indistinguishable from the top one, take the
    # one with the fewest parameters.
    tied = [r for r in rs if (top.logz - r.logz) <= delta_lnz]
    best = min(tied, key=lambda r: (len(r.param_names), -r.logz))
    table = [dict(profile=r.profile, n_components=r.n_components,
                  logz=r.logz, logz_err=r.logz_err,
                  delta_lnz=r.logz - best.logz,
                  n_params=len(r.param_names),
                  reduced_chi2=r.reduced_chi2,
                  preferred=(r is best)) for r in rs]
    return best, table


def fit_and_select(spectrum, rest_wavelength_um, *, max_components=3,
                   profiles=("gaussian",), delta_lnz=5.0, verbose=False,
                   **kwargs):
    """Fit a grid of component counts and profile shapes, and select by
    evidence.

    Parameters
    ----------
    max_components : int
        Try 1..max_components components for each profile.
    profiles : sequence of str
        Profile shapes to try. Trying all eight of :data:`PROFILES` at three
        component counts is 24 nested-sampling runs -- expensive, and usually
        unnecessary. A good default grid for SN work is
        ``('gaussian', 'lorentzian', 'exp_gaussian')``: symmetric thin, heavy
        wings, and one-sided tail cover most of the physically distinct cases,
        and the asymmetric variants are worth adding only once you can see
        asymmetry in the residuals of the symmetric fit.

        Entries may be mixed specifications such as ``"gaussian+lorentzian"``
        -- a narrow Gaussian core on broad Lorentzian wings. These are the
        right model whenever one shape fits the core and a different one fits
        the wings, which no single profile at any component count can do. A
        mixed entry states its own component count, so it is fitted once
        rather than swept over ``max_components``.
    delta_lnz : float
        Evidence margin required to prefer a more complex model.

    Returns
    -------
    (best, table, all_results)
    """
    results = []
    for pname in profiles:
        # A mixed specification such as "gaussian+moffat" already states how
        # many components it has, so it is fitted once rather than swept.
        combo = len(profile_list(pname)) > 1
        counts = [len(profile_list(pname))] if combo \
            else range(1, max_components + 1)
        for n in counts:
            try:
                r = fit_feature(spectrum, rest_wavelength_um,
                                n_components=n, profile_name=pname, **kwargs)
            except Exception as exc:                     # noqa: BLE001
                if verbose:
                    print(f"  {pname} x{n}: failed ({type(exc).__name__}: {exc})")
                continue
            results.append(r)
            if verbose:
                print(f"  {pname} x{n}: lnZ = {r.logz:.2f} +/- {r.logz_err:.2f}, "
                      f"chi2/dof = {r.reduced_chi2:.2f}")
    if not results:
        raise RuntimeError(f"no fit succeeded at {rest_wavelength_um} um")
    best, table = compare_models(results, delta_lnz=delta_lnz)
    return best, table, results
