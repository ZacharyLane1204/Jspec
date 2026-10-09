# jspec

**A JWST spectral pipeline: from a proposal ID to identified, fitted spectral
features.**

`jspec` takes JWST NIRSpec and MIRI observations all the way through:
download the raw exposures, reduce them, join the separately calibrated
pieces into one continuous spectrum, find the emission and absorption
features in it, identify them against a reference line list, and fit their
profiles to get velocities.

It is built around one observation: **a JWST spectrum arrives as a pile of
separately calibrated pieces, and most problems downstream come from how
those pieces were joined.** NIRSpec gratings, MIRI MRS sub-bands and
channels each carry their own flux calibration. Join them naively and you
get steps at the seams — and a step at a seam looks exactly like a broad
spectral feature. Much of what follows is about not doing that.

If you are new, read this file top to bottom and run the commands as you go.
Every section is a step in one continuous workflow.

---

## Contents

1. [Install](#1-install)
2. [What the workflow looks like](#2-what-the-workflow-looks-like)
3. [Download and reduce the data](#3-download-and-reduce-the-data)
4. [Keep your epochs apart](#4-keep-your-epochs-apart)
5. [Stitch the segments into one spectrum](#5-stitch-the-segments-into-one-spectrum)
6. [Check your redshift before trusting anything](#6-check-your-redshift-before-trusting-anything)
7. [Find the features](#7-find-the-features)
8. [Identify them](#8-identify-them)
9. [Plot](#9-plot)
10. [Fit profiles and measure velocities](#10-fit-profiles-and-measure-velocities)
11. [Metallicity at late times](#11-metallicity-at-late-times)
12. [Reference: the modules](#12-reference-the-modules)
13. [Things you should know](#13-things-you-should-know)

---

## 1. Install

This repository is laid out *flat*: the repository root **is** the `jspec`
package. Install it in editable mode so `import jspec` works from anywhere,
regardless of what you named the directory you cloned into:

```bash
git clone <this repo> jspec
cd jspec
conda activate jwst_pipeline        # or whatever your environment is called
pip install -e .
```

That gives you the `jspec` command and the `jspec` importable package.
Check it:

```bash
jspec --help
python -c "import jspec; print(jspec.__version__)"
```

### Dependencies

The analysis layer needs only `numpy`, `scipy`, `astropy`, `matplotlib` and
`pandas`, which `pip install -e .` will pull in. Three optional extras cover
the heavier parts:

```bash
pip install -e ".[fit]"          # dynesty, for `jspec fit`
pip install -e ".[metallicity]"  # pyneb, better emissivities
pip install -e ".[pipelines]"    # jwst, crds, astroquery, AstroBkgInterp
pip install -e ".[all]"          # everything
```

If you already work in a JWST environment you almost certainly have the
`pipelines` extras. The two usually missing are:

```bash
pip install dynesty      # required for `jspec fit` — it will not work without this
pip install pyneb        # optional; jspec.metallicity falls back without it
```

Missing `dynesty` raises a clear error the first time you fit. Missing
`pyneb` is not an error — `jspec.metallicity` uses built-in low-density
Tₑ=10⁴ K emissivities and says so in every result's `caveats` list.

### Do you even need the reduction half?

If somebody has already given you a 1-D spectrum (`.ecsv`, `.csv`, `.txt` or
an `x1d.fits`), skip to [step 5](#5-stitch-the-segments-into-one-spectrum).
Everything from there on works without `jwst` or `crds` installed.

---

## 2. What the workflow looks like

```
  MAST                                    (proposal ID, target name)
    |
    |  download + Stage 1/2/3             jspec/pipelines/*
    v
  uncal -> rate -> cal -> s3d -> x1d      one set per epoch, per band
    |
    |  group by observation date          jspec.epochs      / jspec epochs
    v
  one group of segments per epoch
    |
    |  solve scales, blend overlaps       jspec.stitch      / jspec stitch
    v
  ONE continuous spectrum per epoch       1.7 - 28 um
    |
    |  separate continuum / broad / narrow jspec.continuum
    |  detect peaks and dips               jspec.features   / jspec detect
    |  match against reference lines       jspec.linedb     / jspec lines
    v
  a feature table with identifications
    |
    |  nested-sampling profile fits        jspec.fitting    / jspec fit
    |  annotated figures                   jspec.plotting   / jspec plot
    v
  velocities, widths, fluxes, plots
```

Each arrow is one command. The whole thing, for one target, is roughly:

```bash
python -m jspec.pipelines.miri_mrs_auto_pipeline --proposal 9233 --outdir SN2024aecx/ --max-cores half
python -m jspec.pipelines.abi_reextract_pfpc     --level3-dir SN2024aecx/stage3 --outdir SN2024aecx/reex/2025-06-01 --product-name SN2024AECX_2025-06-01
jspec epochs "SN2024aecx/*/reex/*_x1d.fits"
jspec stitch "SN2024aecx/*/reex/*_x1d.fits" -z 0.002665 -o SN2024aecx.ecsv --plot stitch.png
jspec detect SN2024aecx_2025-06-01.ecsv -z 0.002665 -o features.ecsv
jspec plot   SN2024aecx_2025-06-01.ecsv -z 0.002665 --bands --features -o spectrum.png
jspec fit    SN2024aecx_2025-06-01.ecsv -z 0.002665 --line 12.814 --select
```

The rest of this file explains each of those.

---

## 3. Download and reduce the data

> Skip this section if you already have 1-D spectra.

This stage needs `jwst` and `crds`. Before the first run, point CRDS at the
JWST reference file server — the pipeline cannot calibrate anything without
it:

```bash
export CRDS_PATH=$HOME/crds_cache
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
```

The first pipeline run downloads several GB of reference files into
`CRDS_PATH`. That is normal and happens once.

### 3.0 Find your source position

**You do not need this to start.** If you give no position, the pipeline
locates the brightest point source in each cube itself (see
[3.1.1](#311-what-if-i-do-not-have-a-refined-ra-and-dec-yet)), which is the
right choice for a multi-epoch proposal. Use this step only to *refine* a
position later, or when auto-location picks the wrong object. If you already
have a Level-3 cube:

```bash
python -m jspec.pipelines.get_source_radec --cube some_s3d.fits
python -m jspec.pipelines.get_source_radec --cube some_s3d.fits --x 24 --y 27
```

Pass the RA/Dec it prints back in with `--from-stage3`, which reuses the
existing cubes and so takes minutes rather than hours.

### 3.1 MIRI MRS

**You do not run this once per epoch.** `miri_mrs_auto_pipeline` is already
epoch-aware: after Stage 2 it calls `get_epoch_groups()`, which clusters the
calibrated exposures by `(TARGNAME, epoch)` and builds a *separate Stage-3
association per group*. A proposal with one target observed four times gives
you four Level-3 cubes, named `<TARGNAME>_<epoch-date>`:

```
4 (target, epoch) group(s) across 1 target(s) (>1.0 day gap = new epoch)
  SN2024AECX_2025-01-17  SN2024AECX_2025-06-01
  SN2024AECX_2025-12-07  SN2024AECX_2025-12-28
```

So the normal MRS run is one command for the whole proposal:

```bash
python -m jspec.pipelines.miri_mrs_auto_pipeline \
    --proposal 9233 --outdir SN2024aecx/ \
    --max-epoch-gap-days 1.0 --max-cores half
```

Note what is *not* in that command: no `--ra/--dec`, and no aperture or
background options. The defaults are the recommended values, and the source
position is found automatically — see
[3.1.1](#311-what-if-i-do-not-have-a-refined-ra-and-dec-yet) and
[3.1.2](#312-choosing-the-background-and-aperture-options) below.

Then re-extract and combine, **per epoch**:

```bash
# Re-extract one epoch's cube (this is the recommended extraction).
python -m jspec.pipelines.abi_reextract_pfpc \
    --level3-dir SN2024aecx/stage3 \
    --outdir     SN2024aecx/reextract/2025-06-01 \
    --product-name SN2024AECX_2025-06-01 \
    --fwhm-scale 2.5

# Combine the twelve MRS sub-bands within that epoch.
python -m jspec.pipelines.combine_epoch_spectra \
    --indir SN2024aecx/reextract/2025-06-01 \
    --pattern "*_x1d.fits" --outdir combined_spectra --max-gap-days 1.0
```

`--product-name` is what selects the epoch: pass the Level-3 product name
the previous step printed.

To narrow a run rather than process everything, use `--dataset` (specific
observations) or `--tag` (appended to product names, so parallel runs with
different settings do not overwrite each other). Restart options
(`--skip-download`, `--from-stage3`, `--redo`) are in
[3.4](#34-restarting-without-redoing-everything).

> **The other route.** `prepare_mrs_pfpc_epochs` is for the *separate*
> MRS-PFPC `pfpc_proc` workflow, not for the pipeline above. `pfpc_proc`
> runs Stage 1–3 as a single call and names its products after whatever
> directory you point it at, so it has no hook for epoch-aware association
> building. That script therefore downloads the uncal files and symlinks
> them into one subdirectory per epoch, so the separation falls out of the
> product naming. You then run `pfpc_proc` yourself once per printed
> subdirectory. Use this only if you are using PFPC; otherwise
> `miri_mrs_auto_pipeline` already does the epoch handling for you.

#### 3.1.1 What if I do not have a refined RA and Dec yet?

**Then do not pass one.** If you omit `--ra/--dec` and `--src-x/--src-y`,
the pipeline calls `locate_source_pixel()` on each cube, which runs
DAOStarFinder on the median-collapsed cube and takes the brightest point
source. This is AstroBkgInterp's own recommended approach, and it gives
AstroBkgInterp what it actually wants — pixel coordinates — rather than
RA/Dec that then have to be run through the WCS.

It reports what it found, per cube, so you can check it:

```
SN2024AECX_2025-06-01: source at (x=24.31, y=27.08)
```

If it finds nothing it warns and skips that cube rather than guessing.

**For a multi-epoch proposal, omitting the position is actively the better
choice.** An explicit `--ra/--dec` is a single global value applied to every
(target, epoch) group in the run, which is wrong as soon as there is more
than one group — different visits have different pointings and dithers. The
pipeline warns about exactly this:

```
NOTE: an explicit source position was given but there is more than one
(target, epoch) group in this run -- the same position will be applied to
all of them, which is probably not what you want.
```

Auto-location, by contrast, is done per cube.

Give an explicit position only when auto-location fails or picks the wrong
object — a crowded field, a faint target next to a bright neighbour, or a
source too faint to be detected in the collapsed cube. In that case narrow
the run to one epoch first (`--dataset` or `--tag`), then pass the position
for that epoch.

The refinement path, once you have a first cube:

```bash
# Get RA/Dec (or pixel coords) from an existing cube.
python -m jspec.pipelines.get_source_radec --cube SN2024aecx/stage3/..._s3d.fits

# Re-run only the post-Stage-3 part with the refined position.
python -m jspec.pipelines.miri_mrs_auto_pipeline --proposal 9233 \
    --outdir SN2024aecx/ --from-stage3 --dataset <one-obs> \
    --ra 123.456 --dec -12.345
```

`--from-stage3` reuses the existing cubes, so this is minutes rather than
hours. For sub-pixel work, `find_source_centroid()` and
`refine_and_reprocess()` do a weighted cube collapse followed by
SExtractor-style detection, which is more accurate than either the header
position or eyeballing an image.

#### 3.1.2 Choosing the background and aperture options

Short version: **the defaults are the recommended values.** The long option
string in older notes is unnecessary, and one of those options does nothing
for cubes.

| Option | Default | What it actually controls |
|---|---|---|
| `--fwhm-scale` | 2.5 | **The real aperture knob.** Source-mask radius per slice = `ceil(fwhm_scale × FWHM(λ))` |
| `--min-aper-rad` | 5 px | Floor on that radius, for the blue end where the PSF is narrow |
| `--aper-rad` | 7 px | **Ignored for cubes.** 2-D input only |
| `--ann-width` | 6 px | Width of the annulus sampled for background under the masked source |
| `--bkg-mode` | `polynomial` | `polynomial` or `simple`. These are the only two AstroBkgInterp implements — there is no "skip" |
| `--poly-k` | 3 | Degree of the 2-D polynomial (polynomial mode) |
| `--bin-size` | 9 px | Sliding-window size for that fit |
| `--cube-resolution` | `high` | Sampling density of the fit |
| `--pool-size` | cores−1 | Main lever on runtime; AstroBkgInterp is otherwise single-threaded per cube |

**`--aper-rad` is a trap.** AstroBkgInterp overrides it in cube mode,
recomputing the radius per wavelength slice from the PSF FWHM. Setting
`--aper-rad 7` on an MRS run changes nothing. Use `--fwhm-scale`.

**Why `fwhm-scale` is the one to care about.** The MRS PSF has broad wings.
If the source mask is too small, those wings fall outside it, get classified
as background, and are fitted and subtracted — eating into the flux you are
trying to measure. AstroBkgInterp's old default of 1.5 FWHM does this
noticeably on bright targets, which is why the default here is 2.5.

How to tell it is wrong, and what to do:

- *Extracted spectrum looks suppressed, or the continuum sits low against
  photometry* → mask too small. Raise `--fwhm-scale` to 3.0–3.5. The cost is
  a larger interpolated hole, so the background under the source is less
  well constrained.
- *Background is over-subtracted near the source, or you see a ring* → mask
  too large relative to the field, leaving too little clean background.
  Lower it toward 2.0, or widen `--ann-width`.
- *Blue channels look over-subtracted* → the mask has collapsed to a few
  pixels where the PSF is narrowest. Raise `--min-aper-rad`.

The run prints the mask radii it used, which is the first thing to check:

```
Source mask radius: 5-14 px (fwhm_scale=2.5, min_aper_rad=5);
PSF FWHM 1.8-5.6 px
```

**`--bkg-mode`.** Keep `polynomial` for MRS. `simple` takes an annulus
median — fine for a flat background, but MRS cubes have real structure
across the field, and a flat level then leaves gradients that turn into
spurious broad features after stitching. Drop to `simple` only if the
polynomial fit is unstable, which shows up as a background that oscillates
between adjacent wavelength slices; a less aggressive `--poly-k 2` is worth
trying first.

**`--bin-size`** trades resolution against stability: smaller tracks finer
background structure but is noisier. 9 px is a reasonable middle for MRS.

> **One known bug, already handled.** AstroBkgInterp's own background-error
> term is wrong — it uses the `lstsq` sum-of-squared-residuals as if it were
> fit coefficients, inflating the cube's ERR by ~5 orders of magnitude and
> making every downstream error bar meaningless. This pipeline estimates the
> background-model uncertainty from the actual residual scatter instead.
> `--use-abi-err` restores the broken behaviour and exists only for
> reproducing old runs. Do not use it.

> **Multiprocessing caveat.** Run `miri_mrs_auto_pipeline` and
> `nirspec_ifu_auto_pipeline` from a real script guarded by
> `if __name__ == "__main__":`, or from a notebook. Otherwise Python cannot
> re-import `__main__` in the forked worker processes, and Stage 1 silently
> falls back to single-core. It will still finish; it will just take many
> hours longer than it should.

### 3.2 NIRSpec fixed slit

```bash
# Plan first. This prints the epoch grouping and exits without processing.
python -m jspec.pipelines.nirspec_fs_auto_pipeline \
    --proposal 9233 --target SN2024aecx --max-epoch-gap-days 1.0 --plan-only

# Happy with the grouping? Run it.
python -m jspec.pipelines.nirspec_fs_auto_pipeline \
    --proposal 9233 --target SN2024aecx --max-epoch-gap-days 1.0 \
    --source-type POINT --max-cores half

# Optional refinement of the extraction.
python -m jspec.pipelines.nirspec_fs_refining --help
```

### 3.3 NIRSpec IFU

```bash
python -m jspec.pipelines.nirspec_ifu_auto_pipeline \
    --proposal 9233 --outdir SN2024aecx/nrs_ifu \
    --max-epoch-gap-days 1.0 --output-type band --max-cores half

python -m jspec.pipelines.nirspec_refine_extract --help
```

`--output-type band` produces one product per grating/filter band, which is
what you want: `jspec stitch` works best with the finest segmentation you
can give it, because every extra overlap is another constraint in the scale
solve.

### 3.4 Restarting without redoing everything

All the auto-pipelines can resume:

```bash
--skip-download     # the uncal files are already on disk
--from-stage3       # rate/cal products exist, re-run Stage 3 only
--from-stage 2      # (NIRSpec FS) start at a given stage number
--redo              # ignore existing products and recompute
```

---

## 4. Keep your epochs apart

This is worth its own section because the failure is silent.

If you observe a target in June and again in December and let anything
average them together, you do not get an error. You get a perfectly
plausible-looking spectrum in which a scale factor has quietly absorbed your
target's real variability. Nothing about the output looks wrong.

So **every stage that combines data checks observation dates first**, and
refuses to cross a gap larger than one day by default:

| Stage | Control | Default |
|---|---|---|
| `miri_mrs_auto_pipeline` | `--max-epoch-gap-days` | 1.0 d |
| `prepare_mrs_pfpc_epochs` | `--max-gap-days` | 1.0 d |
| `combine_epoch_spectra` | `--max-gap-days` | 1.0 d |
| `nirspec_fs_auto_pipeline` | `--max-epoch-gap-days` | 1.0 d |
| `nirspec_ifu_auto_pipeline` | `--max-epoch-gap-days` | 1.0 d |
| `jspec stitch` / `jspec.epochs` | `--max-gap-days` | 1.0 d |

All six read the same MJD (`MJD-OBS`, else `EXPSTART`; `MJD_MEAN` then
`MJD_MIN` for `.ecsv` tables) through the same `group_files_by_epoch`, so
there is exactly one definition of "epoch" in the codebase and the analysis
layer cannot drift from the reduction layer.

**Always dry-run the grouping before a long run.** It is much cheaper to
catch a mis-grouped visit here than after Stage 3:

```bash
jspec epochs "SN2024aecx/**/reextract/*_x1d.fits"
```

```
4 file(s) -> 4 epoch(s) (> 1 day gap starts a new epoch)

epoch 0: 2025-01-17  (1 file(s), span 0.000 d, MJD 60691.9755-60691.9755)
    sn2024aecx_2025-01-17.ecsv
epoch 1: 2025-06-01  (1 file(s), span 0.000 d, MJD 60827.2369-60827.2369)
    sn2024aecx_2025-06-01.ecsv
...
These will be processed SEPARATELY. Nothing is combined across epochs
unless you pass --no-epochs.
```

Files with no usable MJD keyword each get their own single-file group rather
than being lumped into a real epoch. Epoch labels prefer a date already
present in the filename, because an exposure just after 0h UTC otherwise
lands on the neighbouring calendar day and stops matching the names you use.

In Python:

```python
from jspec.epochs import group_segments_by_epoch
from jspec.stitch import stitch

for ep in group_segments_by_epoch("reextract/*_x1d.fits", redshift=0.002665):
    report = stitch(ep.segments)
    report.spectrum.write(f"SN2024aecx_{ep.label}.ecsv")
```

`stitch()` itself carries a backstop: hand it segments whose metadata spans
more than `max_gap_days` and it warns. Override deliberately with
`--no-epochs` (CLI) or `check_epoch=False` (Python) — and only if you really
mean it.

---

## 5. Stitch the segments into one spectrum

```bash
jspec stitch "SN2024aecx/**/reextract/*_x1d.fits" \
    -z 0.002665 -o SN2024aecx.ecsv --plot stitch.png
```

This writes **one file per epoch** — `SN2024aecx_2025-06-01.ecsv`,
`SN2024aecx_2025-12-07.ecsv`, … — each independently scale-solved, with the
epoch label inserted into the filename. Point it at NIRSpec and MIRI
products together and each epoch's full 1.7–28 µm spectrum is solved in one
global fit, which is what removes the NIRSpec↔MRS jump.

In Python:

```python
from jspec.spectrum import read_segments
from jspec.stitch import stitch

segs = read_segments("reextract/*_x1d.fits", redshift=0.002665)
report = stitch(segs)
print(report.summary())
report.spectrum.write("SN2024aecx_2025-06-01.ecsv")
```

### Read the report, every time

```
anchor: ch1-medium
scales:  ch1-short 1.0183 +/- 0.0041   ch1-medium 1.0000 (anchor)
         ch1-long  0.9942 +/- 0.0038   ch2-short  0.9981 +/- 0.0052
joins:
  ch1-short | ch1-medium  overlap 0.071 um  n=118  ratio 0.9820 +/- 0.0040  -> continuous
  ch1-long  | ch2-short   GAP     0.004 um         ratio 1.0043 +/- 0.0180  -> continuous
  ch3-long  | ch4-short   overlap 0.093 um  n= 94  ratio 1.1870 +/- 0.0260  -> residual calibration step
```

Three verdicts, and they mean different things:

- **`continuous`** — after scaling, the two sides agree within the noise.
- **`step significant but spectrum is locally curved here`** — there is a
  measurable offset, but there is also real structure at the join, so the
  offset is probably a genuine feature rather than a calibration error.
  Inspect it; do not automatically "fix" it.
- **`residual calibration step`** — a real step with no spectral
  justification. Look at `--plot`, consider a different `--anchor`, and be
  suspicious of any feature you later detect at that wavelength.

Use `--strict` in a script to exit non-zero if any join is flagged.

### How the scale solve works, and why

The obvious approach — walk from the bluest segment, scale each one to its
neighbour — is what produces the jumps. Two reasons:

1. **It can never revise an earlier join.** If one overlap is noisy and its
   ratio gets rejected, that pair is left untied, and that untied pair *is*
   the jump. Later evidence cannot repair it.
2. **Hard inverse-variance blending is discontinuous at both overlap
   edges.** Even with a perfect scale factor, the composite flux jumps the
   moment the second segment's weight switches on.

Instead, `jspec.stitch` does this:

- **One global weighted least-squares solve.** Every overlap contributes a
  constraint `s_blue − s_red = −ln R`, weighted by `1/σ_ln²`, and all
  segments are solved simultaneously. Every overlap informs every scale, and
  the result does not depend on which end you start from.
- **Disconnected groups are solved separately and flagged**, never forced
  together. If two groups share no overlap there is genuinely no information
  tying their calibrations; inventing a tie would fabricate a number.
- **Robust reweighting** (Huber-like, 2 iterations, 3σ clip) so one bad
  overlap degrades gracefully instead of dragging the solution.
- **Raised-cosine tapered blending** through each overlap: the composite
  equals the blue segment exactly at the blue edge and the red segment
  exactly at the red edge, so it is continuous *by construction*. A raised
  cosine rather than a linear ramp because its derivative also vanishes at
  both ends — a linear ramp leaves a visible kink.
- **Edge trimming** (2% by default) excludes band-edge roll-off from ratio
  *measurement* only. It never removes data from the output.
- **`n_eff = n/3`** when computing ratio uncertainties, because x1d pixels
  are correlated by extraction and drizzling. Treating them as independent
  understates the error by about √3 and makes marginal joins look solid.

Where two segments do not overlap at all, the ratio is measured by linear
extrapolation to the gap midpoint, with the baseline widened up to 8× until
it is measurable, the uncertainty inflated by `1 + gap/baseline`, and
curvature rejected at 2.5σ. Extrapolating across a gap is a weak measurement
and is reported as one.

### Useful options

```bash
--anchor ch1-medium   # hold this segment at scale 1 (default: highest S/N)
--trim 0.02           # edge fraction excluded from ratio measurement
--no-blend            # concatenate instead of tapered-blending (diagnostic)
--max-scale 2.0       # reject ratios implying a larger correction than this
--plot stitch.png     # per-join diagnostic figure — look at this
```

---

## 6. Check your redshift before trusting anything

Everything downstream — identification, velocities, line ratios — is in the
rest frame. A wrong redshift does not produce an error; it produces
confident identifications of the wrong lines.

So check it against your own data before going further:

```python
from jspec.spectrum import read_spectrum
from jspec.linedb import refine_redshift

sp = read_spectrum("SN2024aecx_2025-06-01.ecsv", redshift=0.002665)
print(refine_redshift(sp))
```

```
dv = +0 km/s   (11.5 sigma, 29 features)   z_suggested = 0.002665
```

This velocity cross-correlates detected narrow, high-S/N features against
the reference catalogue. Each feature votes with a width-dependent Gaussian;
features sitting on a segment join are excluded, since a residual
calibration step is not evidence about redshift.

**A large `dv` means something is wrong — but not necessarily the
redshift.** The method cannot distinguish a wrong redshift from a real bulk
velocity, so a coherent −1500 km/s could be a bad `z` *or* a genuinely
blueshifted outflow. If `dv` is large and the narrow *host* lines move with
it, your redshift is wrong. If only the broad ejecta lines move, that is
physics, and you should leave `z` alone.

---

## 7. Find the features

```bash
jspec detect SN2024aecx_2025-06-01.ecsv -z 0.002665 -o features.ecsv
```

```
19 features at S/N > 5 (7 emission, 12 absorption; 0 overlap a segment join)
  rest_center_um   sign        component  fwhm_kms  width_class  integrated_snr  on_join  best_id
  12.814004        emission    broad        529.8   narrow        443.9          False    [Ne II] 12.81
  15.554900        emission    broad        463.7   narrow        224.5          False    [Ne III] 15.56
   6.636078        emission    broad       1354.9   broad         224.1          False    [Ni II] 6.64
  17.035075        emission    broad        380.4   narrow        161.1          False    H2 0-0 S(1)

4 high-S/N regions (peak S/N > 15):
  <Region  6.626- 6.682um peak S/N=230.0 @ 6.6540 (1 features)>
  <Region 12.824-12.872um peak S/N=773.5 @12.8480 (3 features)>
  <Region 15.574-15.618um peak S/N=439.2 @15.5960 (3 features)>
  <Region 17.066-17.096um peak S/N=397.9 @17.0800 (3 features)>
```

Both emission peaks and absorption dips are detected, and both are
identified. `--snr` defaults to 5 and `--region-snr` to 15.

### How a "feature" is defined

Detection runs on a **decomposition**, not on the raw flux:

```
flux  =  continuum  +  broad  +  narrow  +  residual
```

exactly additive. The three scales are separated in **velocity**
(`broad_scale_kms=30000`, `narrow_scale_kms=600`), not in wavelength,
because JWST resolving power varies by an order of magnitude across the
range. A fixed wavelength smoothing length means something different at 2 µm
than at 25 µm; a fixed *velocity* length means the same thing everywhere.

All smoothing happens on a uniform **ln λ** grid for the same reason —
applying a difference-penalty smoother directly to the non-uniform stitched
grid imposes band-dependent stiffness and plants spurious inflections at
every join.

Each detected `Feature` carries:

- `wavelength`, `rest_wavelength`, `amplitude`, `snr`
- `fwhm_kms`, and `width_class` ∈ `unresolved` / `narrow` / `broad`
- `is_dip` — absorption features are detected the same way, and are included
- `on_join`, `join_fraction` — **does this feature sit on a segment join?**

That last pair matters more than it looks. A residual calibration step looks
exactly like a broad feature. Every feature records how much of its extent
lies inside a blended overlap, and `--max-join-fraction` drops the ones that
are mostly seam. Before you believe any broad feature, check `on_join`.

`unresolved` means the feature is no wider than ~1.2× the local instrumental
resolution. Its intrinsic width is then reported as `NaN`, not as a number —
because a deconvolved width you cannot measure is not a small width, it is
an unknown one.

### Useful options

```bash
--snr 5              # detection threshold (default 5)
--region-snr 15      # threshold for high-S/N region finding (default 15)
--range 5 15         # restrict to a wavelength range (um, rest frame)
--top 20             # show only the N strongest
--no-identify        # skip line matching, just detect
--max-join-fraction 0.5
```

In Python:

```python
from jspec.continuum import decompose
from jspec.features import find_features, high_snr_regions, features_table

dec = decompose(sp)
feats = find_features(sp, dec, snr_min=6.0)
regions = high_snr_regions(sp, dec, snr_min=10.0)
tbl = features_table(feats)
```

---

## 8. Identify them

`jspec detect` identifies as it goes. To browse the reference data itself:

```bash
jspec lines --range 5 20                  # atomic/molecular lines in a range
jspec lines --species "Ne II" "Fe II"     # specific species
jspec lines --bands                       # dust and molecular bands instead
jspec lines --bands --kind pah silicate
```

The reference data lives in two CSV files you can edit:

- **`data/line_list.csv`** — 107 rest-frame vacuum lines, with columns
  `group, species, wavelength_um, label, kind, category, expected_width,
  notes`. Covers H recombination, forbidden fine-structure lines, iron-group
  and iron-peak species, H₂ rotational lines, and coronal lines.
- **`data/dust_bands.csv`** — broad dust and molecular features as *ranges*
  (`lam_min_um`, `lam_max_um`), not single wavelengths: CO fundamental and
  first overtone, SiO, AlO, PAH, amorphous and crystalline silicate.

Add your own rows to either file and everything picks them up — no code
change needed.

### Matching returns candidates, not an answer

`match_line()` always returns a **ranked list** of candidates with their
velocity offsets, never a single identification:

```python
from jspec.linedb import match_line
for m in match_line(12.814, redshift=0.002665, spectrum=sp):
    print(f"{m.label:20s} {m.velocity_kms:+8.1f} km/s  {m.category}")
```

```
[Ne II] 12.814        +12.3 km/s  forbidden
```

The tolerance is `tol_res_elements` (default 2.0) multiplied by the **local**
resolution element, so it automatically tightens where the instrument is
sharp and loosens where it is not — rather than applying one fixed Δλ that is
far too generous at 2 µm and far too strict at 25 µm.

Reading the output: at the correct redshift the identifications should split
cleanly into two populations. In the SN 2024aecx example, broad ejecta lines
(FWHM 3000–6000 km/s: [Ne II] 12.81, [Ni II] 6.64, [Co II] 10.52, Br-δ) and
narrow host lines (FWHM 300–630 km/s: [S III] 18.71, H₂ 0-0 S(1) 17.03,
[Ne III] 15.55, PAH 11.3). If broad and narrow features are *not*
separating like that, suspect the redshift or the stitching before you
suspect the physics.

> The CSVs were assembled from standard references (NIST ASD, van Hoof 2018,
> Labiano+2021, Argyriou+2023) but have **not** been machine-verified
> line-by-line. Spot-check anything you intend to publish.

---

## 9. Plot

```bash
jspec plot SN2024aecx_2025-06-01.ecsv -z 0.002665 \
    --bands --features --regions -o spectrum.png
```

- `--bands` shades known dust and molecular bands as labelled `axvspan`
  regions — CO fundamental and overtone, SiO, AlO, PAH, silicate. Filter
  with `--kinds molecular pah silicate` or `--names "CO fundamental"`.
- `--features` detects and labels features inline.
- `--regions` shades the high-S/N regions.
- `--rebin 3`, `--log`, `--range 5 20`, `--width`, `--height`, `--dpi` for
  presentation.

In Python the pieces are separate, so you can build your own figure:

```python
import matplotlib.pyplot as plt
from jspec.plotting import (plot_spectrum, plot_dust_bands, plot_features,
                            plot_high_snr, plot_decomposition,
                            plot_stitch_diagnostics)

ax = plot_spectrum(sp)
plot_dust_bands(ax, redshift=0.002665, kinds=["pah", "silicate"])
plot_features(ax, feats, max_labels=25)
plot_high_snr(ax, regions)
```

`plot_decomposition()` shows continuum / broad / narrow stacked — the first
thing to look at when a detection surprises you. `plot_stitch_diagnostics()`
shows each join close up, which is the first thing to look at when the
*spectrum* surprises you.

---

## 10. Fit profiles and measure velocities

Needs `dynesty` (`pip install dynesty`).

```bash
# Fit one feature with one Gaussian.
jspec fit SN2024aecx_2025-06-01.ecsv -z 0.002665 --line 12.814

# Let the evidence choose the profile and the number of components.
jspec fit SN2024aecx_2025-06-01.ecsv -z 0.002665 --line 12.814 --select \
    --profiles gaussian lorentzian voigt --max-components 3 \
    --plot fit.png -o samples.npz
```

```
[Ne II] 12.814 um   gaussian x 2   ln Z = 1043.2
  component 1:  v = -443.7 +/- 4.2 km/s   FWHM = 2114 +/- 11 km/s   flux = 4.03e-17
  component 2:  v = +1320. +/- 31  km/s   FWHM =  612 +/- 24 km/s   flux = 6.1e-18
  continuum: order 1     err_inflation = 0.004     chi2/dof = 0.97
```

### What makes these numbers trustworthy

- **Everything is parameterised in velocity, not wavelength.** A prior of
  "±5000 km/s" means the same thing at 2 µm and 25 µm; "±0.05 µm" does not.
- **The continuum is fitted jointly** (`--continuum-order 0/1/2`), not
  divided out first. Pre-dividing treats the continuum as exactly known,
  when in fact it is strongly covariant with both line flux and line width —
  and that covariance is exactly what sets your error bar.
- **The instrumental LSF is applied by forward convolution**, always, on a
  fine uniform velocity grid with edge padding. So reported widths are
  *intrinsic* by construction, with no deconvolution step. (The padding
  matters: a bare `np.convolve(..., 'same')` tapers the edges and
  systematically narrows lines near the window edge.)
- **An error-inflation nuisance parameter is fitted by default**:
  `σ_eff² = σ² + (f·model)²`. JWST pipeline uncertainties are not always
  right, and a fit that assumes they are will report an error bar that is
  too small. On synthetic data with correct errors this parameter correctly
  goes to zero.
- **Width ordering priors** break the label-switching degeneracy in
  multi-component fits, so component 1 is always the broad one.
- **Model selection by Bayesian evidence**, `--delta-lnz 5.0` (≈150:1). Among
  statistically tied models it picks the one with fewest parameters.

Eight profiles are available: `gaussian`, `lorentzian`, `moffat`, `voigt`
(true Voigt via `wofz`, not a pseudo-Voigt approximation), `skew_gaussian`,
`asym_gaussian` (bi-Gaussian; keeps the peak at `v0`), `asym_lorentzian`,
and `exp_gaussian` (exponentially modified Gaussian — usually the right
choice for an outflow tail). Up to 3 components. `--absorption` allows
negative amplitudes so you can fit a dip.

Derived quantities are computed **per posterior sample**, because they are
non-linear in the parameters and computing them from the mean parameters
gives the wrong answer. For skewed profiles both the **peak velocity** and
the **flux centroid velocity** are reported — they are different numbers and
the distinction matters when you quote a velocity.

> **Expect ~90 seconds per single-component fit** (nlive=500, ~900 points).
> `--select` over 3 profiles × 3 components is 9 fits, so budget ~15 minutes
> per feature. dynesty also emits "enlargement factor … very large"
> bootstrap warnings; they are cosmetic.

```python
from jspec.fitting import fit_feature, fit_and_select, compare_models

res = fit_feature(sp, rest_um=12.814, redshift=0.002665,
                  profile="gaussian", n_components=2)
print(res.summary())

best, all_results = fit_and_select(sp, rest_um=12.814, redshift=0.002665,
                                   profiles=["gaussian", "voigt"],
                                   max_components=3)
```

---

## 11. Metallicity at late times

**"Metallicity" means three different things here, and conflating them is
the main failure mode.** Decide which one you want first.

### Type Ia progenitor metallicity → stable Ni/Fe

This is the one genuinely direct progenitor-metallicity probe available at
late times, and the MIR is the right place to do it. Stable ⁵⁸Ni is made in
neutron-rich burning, and the neutron excess in the white dwarf comes almost
entirely from the ²²Ne left over from CNO-processed, then helium-burned,
initial C/N/O — which is linear in progenitor metallicity.

Use [Ni II] 6.636 and [Ni III] 7.349 against [Fe II] 5.340 / [Fe III] 4.607 /
[Fe II] 17.936. These are fine-structure lines with weak temperature
dependence, unlike the optical [Fe III] blends — which is what makes the MIR
worth the telescope time. → `nickel_to_iron()`

*The catch:* this is degenerate with the white dwarf's central density. High
Ni/Fe can mean a metal-rich progenitor **or** a high-density
near-Chandrasekhar explosion, and breaking that needs independent
information such as the ⁵⁶Ni mass from the light curve. It also needs the
ejecta to be optically thin — roughly >150 d post-explosion.

For a core-collapse SN this ratio is **not** a metallicity probe at all: the
iron-group yield is set by the explosion mechanism and the mass cut.

### Host / environment metallicity → MIR neon + hydrogen

If there is host nebular emission in the field,
`neon_abundance()` uses ([Ne II] 12.814 + [Ne III] 15.555) / Humphreys-α
12.372. This beats optical strong-line methods in four specific ways:

- Fine-structure lines have excitation energies ~0.01 eV, so emissivities
  vary by tens of percent across the plausible Tₑ range. Optical
  collisionally-excited lines vary *exponentially* with Tₑ — which is why
  the optical direct method needs an auroral line and strong-line methods
  carry ~0.2 dex calibration scatter.
- The lines are close enough together that differential extinction is
  negligible, so its uncertainty never enters.
- Neon is not depleted onto dust. Oxygen and iron both are, by an amount
  that depends on local dust content — an unquantified systematic in every
  O/H measurement.
- Ne⁺ and Ne²⁺ account for essentially all neon in a normal H II region, so
  the ionisation correction is near unity.

**Run `ionisation_diagnostics()` first.** Detectable [Ne V] 14.32 or strong
[O IV] 25.89 means a hard radiation field, the ICF assumption fails, and
your result becomes a lower limit. Use [S III] 18.71/33.48 — two lines of
the *same* ion, hence abundance-independent — to measure electron density
and feed it back.

Converting Ne/H to O/H assumes log(Ne/O) = −0.76. That is an assumption, and
it is flagged as one in the output.

### Ejecta composition

Always measurable, and never a metallicity. It is a nucleosynthesis
measurement. Report it as one.

### What not to do

Do not fit the late-time continuum or its dust bands and infer Z. Do not use
the ejecta Fe abundance as a metallicity. Do not apply an optical
strong-line calibration (R23, N2, O3N2) to lines measured in SN ejecta —
those calibrations are built on H II regions in photoionisation equilibrium
and mean nothing in radioactively powered, non-equilibrium gas.

### Also provided

`extinction_from_feii()` uses [Fe II] 1.257/1.644. Both lines share an upper
level, so their intrinsic ratio is fixed by transition probabilities alone —
no temperature, density or abundance dependence. That makes it the most
assumption-free reddening measurement in the NIR. Its one weakness is the
atomic data: published intrinsic ratios span ~1.04–1.49, about 0.6 mag in
A_V, so vary it and see how far your answer moves.

Errors are propagated by Monte Carlo throughout, not by linearised formulae,
because a ratio of two fluxes stops being Gaussian once the denominator S/N
drops below ~5.

---

## 12. Reference: the modules

| Module | What it does |
|---|---|
| `jspec.spectrum` | `Spectrum` container, readers, rest-frame conversion, resolving power |
| `jspec.epochs` | Groups files into observing epochs; cross-epoch guard |
| `jspec.stitch` | Global scale solve, tapered blending, join verification |
| `jspec.continuum` | Continuum / broad / narrow decomposition |
| `jspec.linedb` | Reference lines and dust bands, matching, redshift refinement |
| `jspec.features` | Peak and dip detection, high-S/N regions, identification |
| `jspec.fitting` | Nested-sampling profile fits in velocity space |
| `jspec.plotting` | Band overlays, feature labels, diagnostics |
| `jspec.metallicity` | Late-time SN abundance diagnostics |
| `jspec.cli` | The `jspec` command |
| `jspec.pipelines` | Stage 1/2/3 reduction drivers |

### `jspec.spectrum` — the container everything speaks

One class, with invariants that the rest of the package relies on:

- wavelengths in **microns**, **vacuum**, **observed frame**, strictly
  increasing;
- **bad data is masked, never deleted**, so a detection at index *i* still
  means the same thing after any cleaning step;
- exact duplicate wavelengths are inverse-variance combined on read, because
  `np.interp` returns nonsense on a non-monotonic grid and the failure is
  silent.

```python
from jspec.spectrum import read_spectrum, read_segments, resolving_power

sp = read_spectrum("spectrum.ecsv", redshift=0.002665)
sp.rest_wavelength          # observed / (1+z)
sp.good                     # boolean mask of usable points

resolving_power(12.8, "mrs")        # 2758 - generic MRS curve
resolving_power(12.8, "ch3-long")   # 1980 - that specific band
resolving_power(3.5,  "g395h")      # 2346
resolving_power(3.5,  "prism")      #  148
```

Modes are JWST band names (`ch1-short` … `ch4-long`, `g140m` … `g395h`),
or the shorthands `mrs`, `prism` and `lrs`.

Reads `.ecsv`, `.csv`, `.txt` and `x1d.fits`. For x1d files with several
`EXTRACT1D` extensions it picks the one with the highest median S/N unless
you say otherwise. NIRSpec PRISM and MIRI LRS have separately tabulated
resolution curves, because their R varies far too sharply for an analytic
approximation (the prism dips to R≈31 at 1.3 µm).

---

## 13. Things you should know

### The MRS red end is masked by default

Past about 27.5 µm the MRS channel-4-long throughput calibration breaks
down. The degradation is not gradual — it switches on. The cutoff was chosen
from the data, not from the nominal 28.78 µm band edge:

| λ (µm) | median S/N | % negative | scatter / quoted error |
|---|---|---|---|
| 26.5–27.0 | 12.7 | 0.0 | 2.84 |
| 27.0–27.5 | 11.4 | 0.0 | 2.84 |
| 27.5–28.0 | 9.7 | **4.8** | 2.14 |
| 28.0–28.5 | 4.4 | **14.5** | 2.59 |
| 28.5–28.6 | 3.9 | **23.8** | 3.03 |

Negative fluxes appear for the first time at 27.5 µm and then roughly triple
and triple again; median S/N crosses below 10 at the same point. Below 27.5
there are no negatives at all. Hence `MRS_RED_END_UM = 27.5`.

On a real 14021-point spectrum this masks 187 points and removes **all 21**
negative-flux points. The continuum behaves much better as a result:
flux/continuum at 25–27.5 µm drops from **1.65** to **1.13**.

```python
read_spectrum(path, redshift=0.002665)                      # masked at 27.5 (default)
read_spectrum(path, redshift=0.002665, mask_red_end=False)  # keep everything
read_spectrum(path, redshift=0.002665, mask_red_end=26.0)   # your own cutoff
```

```bash
jspec detect combined.ecsv -z 0.002665 --keep-red-end
jspec detect combined.ecsv -z 0.002665 --red-end 26.0
```

Two deliberate choices:

- **The cut is applied in the observed frame**, not the rest frame. It is a
  property of the instrument and has nothing to do with the source. Applying
  it in the rest frame would slide the cutoff around with redshift — wrong
  in a way that is very easy not to notice.
- **Points are masked, not deleted.** Array lengths and any index you hold
  stay valid, and you can see exactly what went via `~sp.good`,
  `sp.meta["RED_END_MASK_UM"]` and `sp.meta["RED_END_MASKED_N"]`.

### MRS channel 4 uncertainties are underestimated

In the last column of that table, point-to-point scatter exceeds the quoted
errors by ~2.5× throughout **all** of channel 4, not just past the cutoff.
This is left uncorrected deliberately — `jspec.continuum` estimates noise
empirically for exactly this reason, and `jspec.fitting` marginalises over
an error-inflation parameter — but know it before quoting a formal error on
anything redward of ~24 µm.

### Other open items

- **The `data/*.csv` reference files** have not been machine-verified
  against NIST ASD. Spot-check any line you intend to publish.
- **`pyneb` improves `jspec.metallicity`.** Without it the fallback
  emissivities are low-density Tₑ=10⁴ K values, and every result says so in
  its `caveats`.
- **`jspec.fitting` is the slow part.** Budget ~90 s per single-component
  fit, ~15 min per `--select` run.

---

## Troubleshooting

**`No module named jspec`** — you did not `pip install -e .`, or you are in a
different environment. The repository root *is* the package, so a bare
`import jspec` only works after installing.

**`attempted relative import beyond top-level package`** — a `.py` file in
the repository root contains a `from ..something import` . Because the root
*is* the `jspec` package, a single dot is already the top level and `..`
goes out of bounds. Use `from .something import`. Editors sometimes insert
the extra dot automatically when files are moved; this is what it looks like.

**`No module named 'pipelines'`** — a script in `pipelines/` is importing a
sibling as `pipelines.x` instead of `jspec.pipelines.x`. The bare form only
resolves when your current directory happens to be the repository root. All
cross-imports between pipeline scripts should be absolute
(`from jspec.pipelines.combine_nirspec_orders import get_mjd`), which works
whether the script is run with `-m`, run directly, or imported.

**`fit_feature needs dynesty: pip install dynesty`** — exactly what it says.

**CRDS errors during reduction** — set `CRDS_PATH` and `CRDS_SERVER_URL`
(see [step 3](#3-download-and-reduce-the-data)). The first run downloads
several GB of reference files.

**Stage 1 is crawling on one core** — you ran a pipeline script without an
`if __name__ == "__main__":` guard. See the caveat in
[step 3.1](#31-miri-mrs).

**A join is flagged `residual calibration step`** — look at
`jspec stitch --plot`, try a different `--anchor`, and treat any feature you
detect at that wavelength as suspect until proven otherwise.

**Every line is offset by the same velocity** — your redshift is probably
wrong. See [step 6](#6-check-your-redshift-before-trusting-anything).

**Broad and narrow features are not separating cleanly** — suspect, in
order: the redshift, the stitching, then the physics.
