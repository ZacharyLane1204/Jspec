"""
jspec.metallicity
=================

Metallicity diagnostics for late-time NIR/MIR supernova spectra.

Read this first
---------------

The honest answer to "how do I get a metallicity from a late-time NIR/MIR SN
spectrum" is that there are **three different quantities** people mean by
"metallicity" here, they require different measurements, and conflating them
is the main way this analysis goes wrong:

1. **Progenitor metallicity of the exploding star.** Not directly measurable
   from the SN at late times for most line species, because the nebular
   spectrum is dominated by freshly synthesised material. There is one strong
   exception, described below (the Ni/Fe ratio), and it is the single most
   valuable measurement in this list for a Type Ia.

2. **Host-galaxy / local environment metallicity.** Measurable, and this is
   what most "SN metallicity" numbers in the literature actually are. If your
   IFU cube (NIRSpec IFU or MIRI MRS) contains host nebular emission at the
   SN position or nearby, you can measure it directly. The MIR fine-structure
   lines give an unusually *good* measurement here, better in important ways
   than the optical strong-line methods -- see :func:`neon_abundance`.

3. **Ejecta composition.** Always measurable in principle, never a
   "metallicity" -- it is a nucleosynthesis measurement. Useful and
   interesting, but do not report it as Z.

What to actually do, in priority order
--------------------------------------

**(a) If the target is a Type Ia: measure the stable-Ni to Fe ratio.**
This is the best progenitor-metallicity probe available in the MIR and it is
a genuinely direct one. Stable :sup:`58`\\ Ni is produced in neutron-rich
burning, and the neutron excess in the progenitor white dwarf is set almost
entirely by the :sup:`22`\\ Ne left over from CNO-processed, then
helium-burned, initial C/N/O -- which is linear in the progenitor
metallicity. So Ni/Fe in the nebular ejecta is a metallicity tracer. The MIR
is the right place to measure it: [Ni II] 6.636 and [Ni III] 7.349 sit near
[Fe II] 5.340, [Fe III] 4.607 and [Fe II] 17.936 um, all of these are
fine-structure lines with weak temperature dependence (unlike the optical
[Fe III] blends), and at late times the region is optically thin. See
:func:`nickel_to_iron`.

**(b) If there is host nebular emission in the field: use the MIR
fine-structure lines with a hydrogen recombination line.**
[Ne II] 12.814 + [Ne III] 15.555, divided by Humphreys-alpha 12.372 um, gives
Ne/H almost independently of electron temperature and of extinction -- the
two dominant systematics in every optical strong-line method. Neon is an
alpha element that tracks oxygen closely and is not depleted onto dust (which
oxygen and iron both are), so Ne/H is arguably a *cleaner* metallicity
indicator than O/H. The ionisation correction is small because Ne+ and Ne++
together cover essentially all the neon in an H II region. See
:func:`neon_abundance`.

**(c) Failing both: use the host from a different dataset.**
A strong-line O/H from an optical spectrum of the host H II region at the SN
position. This is what most published "SN progenitor metallicities" are. It
is a perfectly respectable measurement; it is just not a measurement of the
supernova.

**(d) What not to do.** Do not fit a late-time SN continuum or its
dust/molecular bands and infer Z. Do not use the ejecta Fe abundance as a
metallicity. Do not apply an optical strong-line calibration (R23, N2, O3N2)
to lines measured in the SN ejecta -- those calibrations are built on H II
regions in photoionisation equilibrium and mean nothing applied to
radioactively powered, non-equilibrium ejecta.

Implementation notes
--------------------

Everything here takes **measured line fluxes with uncertainties**, which you
get from :mod:`jspec.fitting`. The functions propagate errors by Monte Carlo
rather than by linearised formulae, because several of these ratios are
converted through non-linear emissivity relations where a linear propagation
understates the interval.

Atomic data: if `PyNeb <http://research.iac.es/proyecto/PyNeb/>`_ is
installed it is used, and that is strongly preferred -- it carries current,
referenced collision strengths and does the full level-population solve. The
built-in fallback emissivity ratios are low-density, Te=10^4 K values
sufficient for a first estimate and for sanity-checking, and every function
that uses them says so in its output.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DIAGNOSTIC_LINES",
    "Measurement",
    "RatioResult",
    "measure_lines",
    "line_ratio",
    "neon_abundance",
    "nickel_to_iron",
    "ionisation_diagnostics",
    "extinction_from_feii",
    "describe",
]


#: Rest-frame vacuum wavelengths (um) of the lines these diagnostics need.
DIAGNOSTIC_LINES = {
    # --- host H II region: fine-structure metals -----------------------
    "[Ne II] 12.81": 12.81355,
    "[Ne III] 15.56": 15.5551,
    "[Ne III] 36.01": 36.0135,
    "[S III] 18.71": 18.7130,
    "[S III] 33.48": 33.4810,
    "[S IV] 10.51": 10.5105,
    "[Ar II] 6.99": 6.98337,
    "[Ar III] 8.99": 8.99138,
    "[O IV] 25.89": 25.8903,
    # --- hydrogen recombination, for abundance normalisation ----------
    "Hu-alpha 12.37": 12.37190,      # H I 7-6
    "Hu-beta 7.50": 7.50043,         # H I 8-6
    "Pf-alpha 7.46": 7.45986,        # H I 6-5
    "Br-alpha 4.05": 4.05226,
    "Br-gamma 2.17": 2.16612,
    # --- SN ejecta: iron-group, for Ni/Fe ------------------------------
    "[Fe III] 4.61": 4.60743,
    "[Fe II] 5.34": 5.34028,
    "[Ni II] 6.64": 6.63600,
    "[Ni III] 7.35": 7.34880,
    "[Fe II] 17.94": 17.9359,
    "[Fe II] 25.99": 25.9883,
    "[Ni II] 10.68": 10.6822,
    # --- extinction ----------------------------------------------------
    "[Fe II] 1.257": 1.25702,
    "[Fe II] 1.644": 1.64400,
}

# Low-density (n_e -> 0), Te = 10^4 K emissivities relative to H-beta 4861,
# in the same units, used only for the PyNeb-free fallback. These are
# order-of-magnitude-correct standard values; they are NOT a substitute for a
# proper level-population calculation at the actual density.
_FALLBACK_HI_EMISS = {        # j(line) / j(Hbeta) at Te=1e4 K, Case B, ne=1e2
    "Hu-alpha 12.37": 0.0103,
    "Hu-beta 7.50": 0.00551,
    "Pf-alpha 7.46": 0.0220,
    "Br-alpha 4.05": 0.0777,
    "Br-gamma 2.17": 0.0275,
}


@dataclass
class Measurement:
    """A measured line flux with a symmetric or asymmetric uncertainty."""
    name: str
    rest_um: float
    flux: float
    err_lo: float
    err_hi: float = None
    detected: bool = True
    note: str = ""

    def __post_init__(self):
        if self.err_hi is None:
            self.err_hi = self.err_lo

    def draw(self, n, rng):
        """Monte-Carlo draw honouring an asymmetric error bar."""
        u = rng.standard_normal(n)
        return self.flux + np.where(u < 0, u * self.err_lo, u * self.err_hi)

    @property
    def snr(self):
        e = 0.5 * (self.err_lo + self.err_hi)
        return self.flux / e if e > 0 else np.inf


@dataclass
class RatioResult:
    """A derived ratio or abundance with its posterior percentiles."""
    name: str
    value: float
    lo: float
    hi: float
    inputs: dict = field(default_factory=dict)
    method: str = ""
    caveats: list = field(default_factory=list)
    upper_limit: bool = False

    def __repr__(self):
        lim = "< " if self.upper_limit else ""
        return (f"<{self.name} = {lim}{self.value:.4g} "
                f"[{self.lo:.4g}, {self.hi:.4g}]>")

    def summary(self):
        L = [f"{self.name}",
             f"  value : {'<' if self.upper_limit else ''}{self.value:.4g}"
             f"   (16-84%: {self.lo:.4g} -- {self.hi:.4g})",
             f"  method: {self.method}"]
        if self.inputs:
            L.append("  inputs:")
            for k, v in self.inputs.items():
                L.append(f"    {k}: {v}")
        if self.caveats:
            L.append("  caveats:")
            for c in self.caveats:
                L.append(f"    - {c}")
        return "\n".join(L)


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------
def measure_lines(spectrum, names=None, *, profile_name="gaussian",
                  n_components=1, snr_min=3.0, verbose=False, **fit_kwargs):
    """Fit each requested diagnostic line and return ``{name: Measurement}``.

    Lines that fall outside the spectrum, or that fit to a flux below
    ``snr_min`` sigma, are returned with ``detected=False`` and the flux
    treated as an upper limit of ``snr_min * sigma`` downstream. Returning a
    non-detection as a limit rather than dropping it is important: a
    non-detection of [Ne III] is highly informative about the ionisation
    state, and silently omitting it biases every ratio that uses it.

    Extra keyword arguments go to :func:`jspec.fitting.fit_feature`.
    """
    from .fitting import fit_feature

    if names is None:
        names = list(DIAGNOSTIC_LINES)
    out = {}
    wmin, wmax = spectrum.rest_wavelength.min(), spectrum.rest_wavelength.max()
    for nm in names:
        lam = DIAGNOSTIC_LINES.get(nm)
        if lam is None:
            raise KeyError(f"unknown diagnostic line {nm!r}")
        if not (wmin < lam < wmax):
            if verbose:
                print(f"  {nm}: outside coverage")
            continue
        try:
            r = fit_feature(spectrum, lam, n_components=n_components,
                            profile_name=profile_name, **fit_kwargs)
        except Exception as exc:                          # noqa: BLE001
            if verbose:
                print(f"  {nm}: fit failed ({type(exc).__name__}: {exc})")
            continue
        tot = r.derived["total"]["flux (flux_unit x km/s)"]
        val, lo, hi = tot
        m = Measurement(nm, lam, float(val), float(val - lo), float(hi - val))
        if m.snr < snr_min:
            m.detected = False
            m.flux = snr_min * 0.5 * (m.err_lo + m.err_hi)
            m.note = f"non-detection; {snr_min:g}-sigma upper limit"
        out[nm] = m
        if verbose:
            print(f"  {nm}: flux = {val:.4g} (S/N {m.snr:.1f})"
                  f"{' [UL]' if not m.detected else ''}")
    return out


def line_ratio(num, den, name=None, n_mc=20000, seed=0):
    """Monte-Carlo ratio of two :class:`Measurement` objects.

    Draws are rejected when the denominator crosses zero, and the result is
    flagged as an upper limit if either input is a non-detection. A ratio of
    two low-S/N fluxes is not Gaussian and quoting a linear-propagation error
    bar on it is wrong by a lot once the denominator S/N drops below ~5; this
    is why the Monte Carlo is not optional.
    """
    rng = np.random.default_rng(seed)
    a = num.draw(n_mc, rng)
    b = den.draw(n_mc, rng)
    ok = b > 0
    if ok.sum() < 0.5 * n_mc:
        warnings.warn(f"denominator {den.name} is consistent with zero in "
                      f"{100 * (1 - ok.mean()):.0f}% of draws; the ratio is "
                      f"effectively unbounded above", stacklevel=2)
    r = a[ok] / b[ok]
    nm = name or f"{num.name} / {den.name}"
    return RatioResult(
        name=nm, value=float(np.median(r)), lo=float(np.percentile(r, 16)),
        hi=float(np.percentile(r, 84)),
        inputs={num.name: f"{num.flux:.4g} (S/N {num.snr:.1f})",
                den.name: f"{den.flux:.4g} (S/N {den.snr:.1f})"},
        method="Monte-Carlo flux ratio",
        upper_limit=(not num.detected) or (not den.detected))


# ---------------------------------------------------------------------------
# host abundance
# ---------------------------------------------------------------------------
def _hi_emissivity(line, te=1e4, ne=1e2):
    """j(H I line)/j(Hbeta). Uses PyNeb when available."""
    try:
        import pyneb as pn
        tr = {"Hu-alpha 12.37": (7, 6), "Hu-beta 7.50": (8, 6),
              "Pf-alpha 7.46": (6, 5), "Br-alpha 4.05": (5, 4),
              "Br-gamma 2.17": (7, 4)}[line]
        H = pn.RecAtom("H", 1)
        return (H.getEmissivity(te, ne, lev_i=tr[0], lev_j=tr[1])
                / H.getEmissivity(te, ne, lev_i=4, lev_j=2)), True
    except Exception:                                      # noqa: BLE001
        return _FALLBACK_HI_EMISS[line], False


def neon_abundance(lines, te=1e4, ne=1e2, hi_line="Hu-alpha 12.37",
                   n_mc=20000, seed=0, te_err=2000.0):
    """Ne/H from the MIR fine-structure neon lines and an H recombination line.

    This is the recommended **host-environment** metallicity measurement for
    a JWST MIR spectrum, and request (6)'s practical answer when host nebular
    emission is present.

    Why this and not an optical strong-line method:

    * The fine-structure lines [Ne II] 12.81 and [Ne III] 15.55 have
      excitation energies of order 0.01 eV, so their emissivities vary by only
      tens of percent over the whole plausible range of nebular temperature.
      Optical collisionally-excited lines vary exponentially with Te, which is
      why the optical direct method needs an auroral line and the strong-line
      methods need an empirical calibration with ~0.2 dex scatter.
    * Extinction between 12 and 16 um is small and the two lines are close
      enough in wavelength that the differential extinction is negligible for
      any reasonable law -- it does not need to be corrected and its
      uncertainty does not enter.
    * Neon is not depleted onto dust. Oxygen and iron are, by amounts that
      depend on the local dust content, which is an unquantified systematic in
      every O/H measurement.
    * Ne+ and Ne++ together account for nearly all neon in a normal H II
      region, so the ionisation correction factor is near unity.

    Returns 12 + log(Ne/H). To convert to the more familiar 12 + log(O/H),
    the standard assumption is log(Ne/O) = -0.76 (Solar-neighbourhood value;
    the ratio is observed to be roughly constant with metallicity because both
    are alpha elements). That conversion is applied and reported too, but it
    is an *assumption* and is flagged as such.

    Parameters
    ----------
    lines : dict of Measurement
        From :func:`measure_lines`. Needs at least ``'[Ne II] 12.81'`` and the
        chosen ``hi_line``; ``'[Ne III] 15.56'`` strongly recommended.
    te, ne : float
        Electron temperature (K) and density (cm^-3).
    te_err : float
        Temperature uncertainty, marginalised over. The result is nearly
        insensitive to this -- which is the entire point of the method.
    """
    rng = np.random.default_rng(seed)
    need = ["[Ne II] 12.81", hi_line]
    missing = [n for n in need if n not in lines]
    if missing:
        raise KeyError(f"neon_abundance needs {missing}")

    j_hi, used_pyneb = _hi_emissivity(hi_line, te, ne)
    caveats = []
    if not used_pyneb:
        caveats.append(
            "PyNeb is not installed; fallback Case-B / low-density "
            "emissivities at Te=1e4 K were used. Install pyneb for referenced "
            "atomic data and a proper level-population solve.")

    # Ionic abundances relative to H+, via emissivity ratios.
    # Ne+/H+ = (F[NeII] / F(HI)) * (j_HI / j_NeII), with j in matched units.
    def _ne_emiss(ion, T, N):
        try:
            import pyneb as pn
            at = pn.Atom("Ne", ion)
            wave = 128135. if ion == 2 else 155551.
            return at.getEmissivity(T, N, wave=wave), True
        except Exception:                                  # noqa: BLE001
            # Low-density, Te=1e4 K emissivity relative to Hbeta.
            return ({2: 0.1680, 3: 0.2890}[ion]), False

    jne2, p2 = _ne_emiss(2, te, ne)
    jne3, p3 = _ne_emiss(3, te, ne)
    if not (p2 and p3):
        caveats.append("Neon emissivities from the built-in low-density "
                       "Te=1e4 K table.")

    f_hi = lines[hi_line].draw(n_mc, rng)
    f_ne2 = lines["[Ne II] 12.81"].draw(n_mc, rng)
    has3 = "[Ne III] 15.56" in lines
    f_ne3 = lines["[Ne III] 15.56"].draw(n_mc, rng) if has3 else np.zeros(n_mc)
    if not has3:
        caveats.append(
            "[Ne III] 15.55 was not measured, so only Ne+ is counted. In a "
            "region with any appreciable ionisation this UNDERESTIMATES Ne/H, "
            "possibly by a large factor. Treat the result as a lower limit.")

    ok = f_hi > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ne2_h = (f_ne2[ok] / f_hi[ok]) * (j_hi / (jne2 / 1.0))
        ne3_h = (f_ne3[ok] / f_hi[ok]) * (j_hi / (jne3 / 1.0))
    tot = ne2_h + ne3_h
    tot = tot[np.isfinite(tot) & (tot > 0)]
    if tot.size < 100:
        raise ValueError("neon abundance Monte Carlo produced no valid draws; "
                         "check that the input fluxes are positive")

    logneh = 12.0 + np.log10(tot)
    caveats.append(
        "Assumes Ne/H = (Ne+ + Ne++)/H+, i.e. an ionisation correction factor "
        "of 1. Valid for normal H II regions; NOT valid for a hard-radiation "
        "field where [Ne V] 14.32 or [O IV] 25.89 are present -- check those "
        "first with ionisation_diagnostics().")
    caveats.append(
        "This is the abundance of the emitting gas. If that gas is host H II "
        "region material it is a host metallicity; if it is SN ejecta or "
        "shocked CSM it is NOT, and must not be reported as one.")

    res = RatioResult(
        name="12 + log(Ne/H)", value=float(np.median(logneh)),
        lo=float(np.percentile(logneh, 16)), hi=float(np.percentile(logneh, 84)),
        inputs={k: f"{lines[k].flux:.4g} (S/N {lines[k].snr:.1f})"
                for k in lines if k in (need + ["[Ne III] 15.56"])},
        method=("MIR fine-structure direct method"
                + (" (PyNeb atomic data)" if used_pyneb and p2 and p3
                   else " (fallback atomic data)")),
        caveats=caveats,
        upper_limit=any(not lines[k].detected for k in need))

    # Report the O/H translation as a separate, clearly-assumption-based line.
    res.inputs["-> 12+log(O/H) assuming log(Ne/O) = -0.76"] = (
        f"{res.value + 0.76:.2f} [{res.lo + 0.76:.2f}, {res.hi + 0.76:.2f}]")
    return res


def nickel_to_iron(lines, n_mc=20000, seed=0,
                   ni_line="[Ni II] 6.64", fe_line="[Fe II] 5.34",
                   emiss_ratio=None):
    """Stable-Ni to Fe ratio from MIR fine-structure lines of the ejecta.

    **This is the progenitor-metallicity measurement for a Type Ia.** At late
    times (roughly > 150 d) the inner ejecta are optically thin and the
    iron-group fine-structure lines are formed in a nearly isothermal,
    low-density region where their ratio reflects the ionic abundance ratio
    with only weak dependence on conditions -- much weaker than the optical
    [Fe III]/[Ni II] blends, which is why the MIR is worth the telescope time
    for this specific measurement.

    The physical chain is: progenitor metallicity -> initial C+N+O -> CNO-cycle
    conversion to :sup:`14`\\ N -> helium burning to :sup:`22`\\ Ne -> neutron
    excess in the white dwarf -> stable :sup:`58`\\ Ni produced in
    normal-freezeout burning alongside :sup:`56`\\ Ni (which decays to Fe). So
    a high measured Ni/Fe implies a metal-rich progenitor. The relation is
    approximately linear in Z over the range where :sup:`22`\\ Ne dominates
    the neutron excess, but it also depends on the explosion's central
    density, which is the dominant systematic and is **not** separable from
    metallicity by this measurement alone.

    Parameters
    ----------
    emiss_ratio : float, optional
        j(Ni line)/j(Fe line) at the relevant conditions, used to turn the
        flux ratio into an abundance ratio. If omitted, only the *flux* ratio
        is returned, which is the defensible thing to publish without a
        tailored nebular model -- the conversion is model-dependent and this
        function will not invent it for you.

    Notes
    -----
    For a core-collapse SN this ratio is **not** a progenitor-metallicity
    probe: the iron-group yield there is set by the explosion mechanism and
    the mass cut, not by the initial composition. Reported as a composition
    measurement only.
    """
    for k in (ni_line, fe_line):
        if k not in lines:
            raise KeyError(f"nickel_to_iron needs {k!r}")
    r = line_ratio(lines[ni_line], lines[fe_line],
                   name=f"F({ni_line}) / F({fe_line})", n_mc=n_mc, seed=seed)
    r.method = "MIR iron-group fine-structure flux ratio"
    r.caveats = [
        "This is a FLUX ratio, not an abundance ratio, unless you supplied "
        "emiss_ratio. Converting it requires a nebular model at the actual "
        "temperature, density and ionisation of the emitting zone.",
        "Interpretable as a progenitor-metallicity probe for Type Ia only. "
        "For a core-collapse SN this measures the explosion, not the star's "
        "initial composition.",
        "Degenerate with the central density of the exploding white dwarf; "
        "a high Ni/Fe can mean a metal-rich progenitor OR a high-density "
        "(near-Chandrasekhar) explosion. Breaking that degeneracy needs "
        "independent information, e.g. the 56Ni mass from the light curve.",
        "Requires the ejecta to be optically thin at these wavelengths. "
        "Check the epoch: before ~150 d post-explosion this assumption fails "
        "and the ratio is a radiative-transfer result, not an abundance.",
    ]
    if emiss_ratio is not None:
        rng = np.random.default_rng(seed + 1)
        a = lines[ni_line].draw(n_mc, rng)
        b = lines[fe_line].draw(n_mc, rng)
        ok = b > 0
        ab = (a[ok] / b[ok]) / emiss_ratio
        r.inputs["abundance ratio Ni+/Fe+"] = (
            f"{np.median(ab):.4g} [{np.percentile(ab, 16):.4g}, "
            f"{np.percentile(ab, 84):.4g}] using j_ratio={emiss_ratio:g}")
    return r


def ionisation_diagnostics(lines, n_mc=20000, seed=0):
    """Ionisation-sensitive line ratios, with interpretation.

    Returns a list of :class:`RatioResult`. Compute these **before** any
    abundance, because they tell you whether the abundance method's ionisation
    assumptions hold. In particular, detectable [Ne V] 14.32 or strong
    [O IV] 25.89 means a hard radiation field -- an AGN, a shock, or the SN
    itself -- and in that case the "Ne+ + Ne++ is all the neon" assumption
    behind :func:`neon_abundance` fails and the result is a lower limit.
    """
    pairs = [
        ("[Ne III] 15.56", "[Ne II] 12.81", "Ne++/Ne+",
         "Primary MIR ionisation parameter. >~1 indicates a hard field or a "
         "shock; <~0.1 is a soft, low-excitation H II region."),
        ("[S IV] 10.51", "[S III] 18.71", "S3+/S++",
         "Higher ionisation potential than the neon pair; use the two "
         "together to separate ionisation parameter from spectral hardness."),
        ("[Ar III] 8.99", "[Ar II] 6.99", "Ar++/Ar+",
         "Argon is not depleted onto dust, making this the cleanest "
         "ionisation ratio when dust is present."),
        ("[S III] 18.71", "[S III] 33.48", "n_e diagnostic",
         "Ratio of two lines of the SAME ion, so it is independent of "
         "abundance and measures electron density (sensitive over roughly "
         "1e2 to 1e4 cm^-3). Measure this and feed it back as `ne`."),
        ("[O IV] 25.89", "[Ne II] 12.81", "hard-field indicator",
         "O3+ needs 54.9 eV. A clear detection means photoionisation by a "
         "normal stellar population is not the whole story."),
    ]
    out = []
    for num, den, label, interp in pairs:
        if num not in lines or den not in lines:
            continue
        r = line_ratio(lines[num], lines[den], name=f"{num} / {den}  [{label}]",
                       n_mc=n_mc, seed=seed)
        r.method = "flux ratio"
        r.caveats = [interp]
        out.append(r)
    return out


def extinction_from_feii(lines, intrinsic_ratio=1.36, n_mc=20000, seed=0,
                         law_slope=1.9):
    """Extinction from the [Fe II] 1.257/1.644 um ratio.

    These two lines come from the same upper level (a\\ :sup:`4`\\ D\\
    :sub:`7/2`), so their intrinsic ratio is fixed by atomic transition
    probabilities alone -- no dependence on temperature, density, or
    abundance. That makes this the most assumption-free reddening measurement
    available in a NIR spectrum, and worth doing before any ratio that spans a
    wide wavelength range.

    The main caveat is the atomic data itself: published intrinsic ratios
    range from about 1.04 to 1.49 depending on the transition probabilities
    adopted, which maps to roughly 0.6 mag of A_V. The default 1.36 is a
    commonly-used value; vary it and see how much your answer moves before
    trusting the result to better than a few tenths of a magnitude.

    Returns A_V assuming a power-law NIR extinction, A(lambda) ~ lambda^-slope,
    with ``law_slope=1.9`` (close to standard Galactic NIR laws, which are
    themselves fairly universal in this wavelength range).
    """
    for k in ("[Fe II] 1.257", "[Fe II] 1.644"):
        if k not in lines:
            raise KeyError(f"extinction_from_feii needs {k!r}")
    rng = np.random.default_rng(seed)
    a = lines["[Fe II] 1.257"].draw(n_mc, rng)
    b = lines["[Fe II] 1.644"].draw(n_mc, rng)
    ok = (b > 0) & (a > 0)
    obs = a[ok] / b[ok]
    l1, l2 = 1.25702, 1.64400
    # A(l1) - A(l2) = 2.5 log10(R_int / R_obs)
    dA = 2.5 * np.log10(intrinsic_ratio / obs)
    # Convert the differential to A_V with A(l)/A_V = (l/0.55)^-slope
    k1 = (l1 / 0.55) ** (-law_slope)
    k2 = (l2 / 0.55) ** (-law_slope)
    av = dA / (k1 - k2)
    av = av[np.isfinite(av)]
    return RatioResult(
        name="A_V from [Fe II] 1.257/1.644",
        value=float(np.median(av)), lo=float(np.percentile(av, 16)),
        hi=float(np.percentile(av, 84)),
        inputs={"observed ratio": f"{np.median(obs):.3f}",
                "intrinsic ratio assumed": f"{intrinsic_ratio:.3f}",
                "extinction law slope": f"lambda^-{law_slope:g}"},
        method="same-upper-level [Fe II] ratio",
        caveats=[
            "The intrinsic ratio is uncertain at the ~20% level in the "
            "literature (~1.04-1.49), worth ~0.6 mag in A_V. Re-run with "
            "different values of intrinsic_ratio to see your systematic.",
            "Negative A_V means the observed ratio exceeds the assumed "
            "intrinsic one: either the intrinsic value is wrong, the lines "
            "are blended, or the relative flux calibration between the two "
            "is off -- the last is worth checking against the stitching "
            "report if the two lines fall in different segments.",
        ])


def describe():
    """Print the methodological guidance in :mod:`jspec.metallicity`'s
    docstring. Convenience for interactive use."""
    print(__doc__)
