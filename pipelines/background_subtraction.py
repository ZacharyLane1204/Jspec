"""
Background estimation for wide-field imaging (~25 arcmin FOV, 0.72"/px).

This module estimates and returns the sky background of an astronomical
image as a full-resolution 2D array, suitable for subtracting from the
data to leave a clean, background-free frame. It is built to cope with
three distinct situations that a simple tiled background estimator
(e.g. plain SExtractor-style background estimation) handles poorly on
its own:

1. Ordinary fields with stars and empty sky.
   A tiled background estimator (a clipped statistic computed in each of
   many small boxes across the image, then smoothed box-to-box) works
   well here: it is fast, robust to outliers (stars), and still captures
   spatially varying structure such as vignetting or optical gradients
   that a single global sky value would miss.

2. Fields containing nebulosity or a moderately-sized galaxy.
   Here, large patches of the image are not empty sky at all but real,
   diffuse astrophysical signal. A plain tiled statistic is biased high
   in those tiles, because its outlier-rejection is designed to reject
   sparse bright pixels (stars, cosmic rays), not a broad diffuse excess
   that fills most of the tile. This module handles that by:
     a) building masks that flag both point sources (stars) and diffuse
        nebulosity,
     b) running the same tiled engine on the masked image, so tiles that
        are mostly clean sky still get the fast, well-tested tiled
        statistic,
     c) for tiles that are mostly masked (dominated by nebulosity, or by
        crowding), re-estimating that tile's sky level with a more
        robust method (a KDE-mode estimate over whatever clean pixels
        remain), and
     d) blending the two per-tile estimates into a single smooth
        background surface.

3. Objects larger than the smoothing scales used to detect nebulosity
   (for example, a galaxy spanning 1000+ pixels).
   A large object's own interior can look flat relative to itself, so a
   same-scale contrast test can under-detect its middle while still
   correctly flagging its edges. This is handled with a second,
   scale-independent test: a much larger-box background estimate serves
   as a coarse "sky floor" reference, and any pixel reading significantly
   above that floor -- regardless of local contrast -- is flagged too.

On top of source/nebula detection, this module also grows each detected
nebula/galaxy region outward until its own measured flux genuinely
reaches the noise floor, rather than stopping at a fixed or hand-tuned
buffer size. This matters because a galaxy's light fades gradually over
a long radial range; masking exactly at the detection threshold leaves a
faint but real signal in what the background estimator then treats as
"clean sky," which biases the local background high and produces a
faint over-subtracted ring just outside the object once the background
is removed. Growing the mask until the excess flux is actually
consistent with noise avoids that.

Typical usage
-------------
    from background_subtraction import determine_background

    background, background_rms = determine_background(image_data)
    reduced_image = image_data - background

`background_residual_flatness` is provided as a post-hoc diagnostic: run
it on the subtracted image to get a single number describing how flat
the sky is afterwards, useful for flagging frames where the detection/
masking parameters may need adjusting for an unusual field.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from scipy.ndimage import zoom

from astropy.stats import sigma_clip, mad_std, sigma_clipped_stats

from skimage.morphology import remove_small_objects, binary_opening, disk, binary_dilation
from skimage.filters import sobel

try:
    import sep
except ImportError as e:
    raise ImportError("The `sep` package (pip install sep) is required for "
                      "background_subtraction.py. It provides the tiled SExtractor-style "
                      "background engine used as the spatially varying core estimator.") from e


def estimate_background_method(data, box_size=50, filter_size=5, verbose=False, testing_plot=False):
    """
    Quick, coarse pass over an image to flag roughly how much of it is
    covered by nebulosity, without doing a full background subtraction.

    This is a lightweight diagnostic, not the background estimate you'd
    actually subtract -- use `determine_background` for that. It works by
    running a single tiled background pass, scaling the result to a 0-1
    range, and treating pixels above a fixed fraction of that range as
    "nebulosity."

    Parameters
    ----------
    data : 2D ndarray
        Image data.
    box_size : int
        Tile size (pixels) for the underlying tiled background pass.
    filter_size : int
        Box-to-box smoothing applied by the tiled background pass.
    verbose : bool
        If True, print the estimated nebulosity fraction.
    testing_plot : bool
        If True, save a PNG of the resulting nebula mask to the current
        working directory (`nebula_mask_og.png`) for a quick visual check.

    Returns
    -------
    neb_fraction : float
        Estimated fraction of the image flagged as nebulosity, in [0, 1].
    neb_mask : 2D bool ndarray
        The corresponding mask.
    """
    finite_mask = ~np.isfinite(data)

    bkg = sep.Background(np.ascontiguousarray(data.astype(np.float32)), 
                         mask=np.ascontiguousarray(finite_mask),
                         bw=box_size, bh=box_size, fw=filter_size, fh=filter_size)
    bkg_map = bkg.back()

    finite = np.isfinite(bkg_map)
    bmin = np.nanmin(bkg_map[finite])
    bmax = np.nanmax(bkg_map[finite])
    if bmax > bmin:
        scaled_map = (bkg_map - bmin) / (bmax - bmin)
    else:
        scaled_map = np.zeros_like(bkg_map)

    neb_mask = scaled_map > 0.15
    neb_fraction = neb_mask.sum() / max(finite.sum(), 1)

    if testing_plot:
        plt.figure()
        plt.imshow(neb_mask)
        plt.title('Nebula Mask')
        plt.savefig('nebula_mask_og.png', dpi=300, bbox_inches='tight')
        plt.close()

    if verbose:
        print(f'Nebulosity area approximate: {neb_fraction}')

    return neb_fraction, neb_mask


def star_gradient_mask(data, sigma_thresh=6, dilation=3):
    """
    Flag pixels belonging to compact sources (stars) using a local
    gradient (edge) detector.

    Stars and other compact sources produce sharp, localized edges in
    the image, unlike smooth sky background. This computes a Sobel
    gradient magnitude map, flags pixels with unusually large gradient
    (relative to a robust estimate of the typical gradient level), and
    grows the result slightly so the source's immediate surroundings
    are included too.

    Parameters
    ----------
    data : 2D ndarray
        Image data.
    sigma_thresh : float
        Detection threshold, in multiples of the gradient map's own
        robust standard deviation (`astropy.stats.mad_std`).
    dilation : int
        Radius (pixels) to grow the detected mask by, so a source's
        immediate wings/surroundings are included, not just the sharp
        edge itself.

    Returns
    -------
    mask : 2D bool ndarray
        True where a compact source (or its immediate surroundings) is
        detected.
    """
    grad = sobel(data)
    finite_grad = grad[np.isfinite(grad)]
    if finite_grad.size == 0:
        return np.zeros_like(data, dtype=bool)

    noise = mad_std(finite_grad)
    mask = np.isfinite(grad) & (grad > sigma_thresh * noise)
    mask = binary_dilation(mask, disk(dilation))
    return mask


def gaussian_filter_nan(data, sigma):
    """
    Gaussian-smooth a 2D array while correctly ignoring NaNs, instead of
    letting them contaminate every pixel they touch.

    A plain `scipy.ndimage.gaussian_filter` treats NaN as if it were a
    real (very large/invalid) number, spreading NaNs into all of their
    neighbours. This instead smooths a NaN-filled-with-zero copy of the
    data and a matching "how much valid data went into this pixel" mask
    in parallel, then divides the two -- equivalent to only ever
    averaging over the valid (non-NaN) pixels that contributed to each
    output pixel, and leaving genuinely NaN-starved output pixels as NaN.

    Parameters
    ----------
    data : 2D ndarray
        Values to smooth; may contain NaNs (e.g. masked-out pixels).
    sigma : float
        Gaussian smoothing scale, in pixels.

    Returns
    -------
    result : 2D ndarray
        Smoothed array, same shape as `data`, NaN where no valid data
        was available nearby.
    """
    mask = np.isfinite(data).astype(float)
    data_filled = np.nan_to_num(data, nan=0.0)

    smooth_data = gaussian_filter(data_filled, sigma=sigma)
    smooth_mask = gaussian_filter(mask, sigma=sigma)

    with np.errstate(invalid='ignore', divide='ignore'):
        result = smooth_data / smooth_mask

    result[smooth_mask == 0] = np.nan
    return result


def nebula_scale_residual_mask(data, medium_sigma=50, large_sigma=200,
                                sigma_thresh=5, min_size=500, star_mask=None):
    """
    Detect extended diffuse emission (nebulosity, galaxy light) by
    comparing two different smoothing scales of the same image.

    The idea: smooth the image at a "medium" scale and a "large" scale.
    Over featureless sky, the two should be nearly identical. Where
    there's real diffuse structure somewhat smaller than the large scale
    (e.g. a compact H II region, or the edge of a moderately-sized
    galaxy), the medium-scale smoothing will still show it while the
    large-scale smoothing washes it out -- so their difference picks
    it out.

    This test is inherently scale-limited: an object substantially
    larger than `large_sigma` looks flat relative to itself even at the
    medium scale, so its interior can be under-detected here even though
    it is clearly not empty sky. See `large_scale_excess_mask` for a
    complementary test aimed at exactly that case (e.g. galaxies
    spanning 1000+ pixels).

    Parameters
    ----------
    data : 2D ndarray
        Image data.
    medium_sigma, large_sigma : float
        The two Gaussian smoothing scales (pixels) to compare.
    sigma_thresh : float
        Detection threshold, in multiples of the residual map's own
        robust standard deviation.
    min_size : int
        Minimum connected pixel count for a detection to be kept
        (removes small spurious detections from noise).
    star_mask : 2D bool ndarray or None
        Pixels to exclude before smoothing (typically already-detected
        stars), so bright point sources don't get mistaken for
        nebulosity or bias the smoothing near them.

    Returns
    -------
    mask : 2D bool ndarray
        True where extended diffuse emission is detected.
    """
    if star_mask is not None:
        work = data.copy()
        work[star_mask] = np.nan
    else:
        work = data

    medium = gaussian_filter_nan(work, medium_sigma)
    large = gaussian_filter_nan(work, large_sigma)

    resid = medium - large
    finite_resid = resid[np.isfinite(resid)]
    if finite_resid.size < 10:
        return np.zeros_like(data, dtype=bool)

    sigma = mad_std(finite_resid)
    if sigma == 0 or not np.isfinite(sigma):
        return np.zeros_like(data, dtype=bool)

    mask = np.isfinite(resid) & (resid > sigma_thresh * sigma)

    mask = binary_opening(mask, disk(3))
    mask = remove_small_objects(mask, min_size)
    mask = binary_dilation(mask, disk(5))

    return mask


def _coarse_floor_background(data, star_mask, box_size, coarse_multiplier=4):
    """
    Estimate a coarse, large-box sky-level and noise reference, used as
    a "floor" against which extended emission is measured.

    This runs the same tiled background engine used for the final
    background estimate, but with a much larger box size (a multiple of
    the usual `box_size`). A larger box means any single extended object
    inside it (even one spanning 1000+ pixels) still only occupies a
    fraction of that much bigger box, so the box's own robust statistic
    stays representative of the true surrounding sky, rather than being
    pulled upward by the object itself.

    Only `star_mask` (plus non-finite pixels) is excluded here -- not
    yet any detected nebulosity, since typically no nebula mask exists
    at the point this is first called; see `build_source_and_nebula_masks`
    for how this is refined iteratively once a nebula mask is available.

    Parameters
    ----------
    data : 2D ndarray
        Image data.
    star_mask : 2D bool ndarray or None
        Pixels to exclude (typically detected stars, and/or previously
        detected nebulosity on a refinement pass).
    box_size : int
        The pipeline's normal tile size; the actual box used here is
        `box_size * coarse_multiplier`.
    coarse_multiplier : float
        How much larger than the normal tile size this coarse box should
        be.

    Returns
    -------
    floor, floor_rms : 2D ndarrays (float64), same shape as `data`.
        Per-pixel sky-level and noise reference maps.
    """
    finite_mask = ~np.isfinite(data)
    mask = (star_mask | finite_mask) if star_mask is not None else finite_mask

    sep_data = np.ascontiguousarray(np.nan_to_num(data, nan=0.0).astype(np.float32))
    sep_mask = np.ascontiguousarray(mask)

    coarse_box = max(int(box_size * coarse_multiplier), box_size + 1)
    bkg = sep.Background(sep_data, mask=sep_mask, bw=coarse_box, bh=coarse_box,
                         fw=1, fh=1)

    return bkg.back().astype(np.float64), bkg.rms().astype(np.float64)


def large_scale_excess_mask(data, floor, floor_rms, star_mask=None,
                             sigma_thresh=4, denoise_sigma=15, min_size=1000):
    """
    Detect extended emission by comparing the (lightly denoised) image
    directly against a coarse sky floor -- catching structure that is
    too large to show up as local contrast in `nebula_scale_residual_mask`.

    Complementary to `nebula_scale_residual_mask`: that test looks for
    local contrast between two smoothing scales and so is blind to an
    object whose interior is flat relative to itself (anything much
    larger than its own `large_sigma`). This test instead asks a simpler
    question at every pixel -- "is this reading significantly above the
    true sky floor" -- which works regardless of the object's size,
    using `floor`/`floor_rms` from `_coarse_floor_background` as the
    sky reference.

    Parameters
    ----------
    data : 2D ndarray
        Image data.
    floor, floor_rms : 2D ndarrays
        Sky-level and noise reference maps, from `_coarse_floor_background`.
    star_mask : 2D bool ndarray or None
        Pixels to exclude (typically detected stars).
    sigma_thresh : float
        Detection threshold, in multiples of `floor_rms`.
    denoise_sigma : float
        Light Gaussian smoothing applied before comparison, purely to
        tame per-pixel read noise -- much smaller than the smoothing
        scales in `nebula_scale_residual_mask`, so it isn't meant to
        erase real structure, just noise.
    min_size : int
        Minimum connected pixel count for a detection to be kept. Set
        larger than `nebula_scale_residual_mask`'s default since this
        test targets large-scale objects specifically.

    Returns
    -------
    mask : 2D bool ndarray
        True where extended emission is detected above the sky floor.
    """
    work = data.copy()
    if star_mask is not None:
        work[star_mask] = np.nan

    smoothed = gaussian_filter_nan(work, denoise_sigma)

    finite = np.isfinite(smoothed) & np.isfinite(floor) & np.isfinite(floor_rms)
    if finite.sum() < 10:
        return np.zeros_like(data, dtype=bool)

    excess = np.zeros_like(data, dtype=np.float64)
    excess[finite] = smoothed[finite] - floor[finite]

    # Guard against a zero/degenerate rms in some region of the coarse
    # rms map (e.g. a fully-masked patch) by falling back to the median
    # of the positive rms values elsewhere, rather than dividing by zero.
    positive_rms = floor_rms[np.isfinite(floor_rms) & (floor_rms > 0)]
    fallback_rms = float(np.nanmedian(positive_rms)) if positive_rms.size else 1.0
    rms_map = np.where((floor_rms > 0) & np.isfinite(floor_rms), floor_rms, fallback_rms)

    mask = finite & (excess > sigma_thresh * rms_map)

    mask = binary_opening(mask, disk(3))
    mask = remove_small_objects(mask, min_size)
    mask = binary_dilation(mask, disk(5))

    return mask


def _adaptive_nebula_buffer(data, nebula_mask, star_mask=None, floor=None, floor_rms=None,
                             box_size=100, step_px=None, sigma_stop=1.5, consecutive_flat_rings=3,
                             max_growth_px=None, denoise_sigma=None, effective_n_cap=None):
    """
    Grow each detected nebula/galaxy region outward until its own flux
    genuinely fades into the noise, rather than stopping right at the
    detection threshold.

    Why this is needed
    -------------------
    A galaxy's disk (or any extended emission) fades gradually with
    radius, not sharply -- a sigma-threshold detector has to stop
    somewhere, but real, very faint flux typically continues beyond that
    point. If the mask stops exactly at the detection edge, the pixels
    just outside it still contain a bit of real signal but get treated
    as ordinary "clean sky" by the background estimator, biasing the
    local background estimate high right at that boundary. Subtracting
    that background then leaves a faint over-subtracted ring just
    outside the object. Growing the mask until the region's own flux is
    actually consistent with noise (rather than by a fixed or
    hand-tuned amount) avoids this without needing to be re-tuned for
    every object's size or the frame's depth.

    Method
    ------
    Each connected region in `nebula_mask` is grown independently,
    walking outward in `step_px`-wide annuli (computed efficiently via a
    Euclidean distance transform, so cost doesn't depend on how far a
    region needs to grow). At each annulus, the median EXCESS above the
    local sky floor is compared to the local noise; growth continues
    while that excess is still significant, and stops once
    `consecutive_flat_rings` annuli in a row are not (see `sigma_stop`
    and `effective_n_cap` for exactly how "significant" is defined).

    Parameters
    ----------
    data : 2D ndarray
        Image data.
    nebula_mask : 2D bool ndarray
        Initial (threshold-based) nebula/galaxy detection to grow
        outward from.
    star_mask : 2D bool ndarray or None
        Excluded from every annulus's statistics, so a foreground star
        just outside the object doesn't bias that annulus's median.
    floor : 2D ndarray or None
        Per-pixel sky-level reference (e.g. from `_coarse_floor_background`).
        Each annulus's EXCESS above this floor is what's tested against
        the noise -- not the raw pixel level, which sits at the sky
        level (typically ~100+ counts) and would never look "flat".
        Falls back to a single robust global sky estimate if not given.
    floor_rms : 2D ndarray or None
        Per-pixel noise reference (e.g. from `_coarse_floor_background`).
        Falls back to a single global `mad_std` estimate if not given.
    box_size : int
        The same tile size used elsewhere in this module (`sep`'s
        `bw`/`bh`, the KDE-rescue tile grid, `background_residual_flatness`).
        `step_px`, `denoise_sigma`, and `effective_n_cap` all derive from
        this by default (see below), so changing `box_size` for a
        different telescope or pixel scale keeps this function's
        internal scales consistent with the rest of the pipeline
        automatically.
    step_px : float or None
        Annulus width to test at each growth step. Defaults to
        `max(5, box_size // 6)` if not given.
    sigma_stop : float
        Stop growing a region once its annulus's median excess above
        the sky floor falls, and stays, below this many multiples of the
        annulus's effective noise (see `effective_n_cap`) for
        `consecutive_flat_rings` annuli in a row.
    effective_n_cap : int or None
        An annulus can contain anywhere from a few dozen to tens of
        thousands of pixels, and the statistical precision of its
        median improves with the square root of how many pixels went
        into it. Comparing directly against the raw per-pixel noise
        (as if only one pixel were used) is too lax for a large annulus
        -- a real but small systematic offset can still look "flat" and
        stop growth too early. Comparing against the annulus's full
        pixel count is too strict -- with tens of thousands of pixels,
        the resulting tolerance becomes so tight that irrelevant,
        sub-percent-level flat-fielding structure elsewhere in the frame
        would never look "flat," and growth would never stop. Capping
        the effective pixel count used for this comparison ties the
        stopping criterion to "is this region's residual bias smaller
        than the noise a single background tile already has" -- a
        practically meaningful bar. Defaults to `box_size ** 2` if not
        given (one tile's worth of pixels).
    consecutive_flat_rings : int
        Require this many flat annuli in a row before stopping, so one
        anomalously quiet annulus (or a gap between spiral arms) doesn't
        cut the growth short.
    max_growth_px : float or None
        Safety limit only -- not the primary stopping control (that's
        the flat-annulus test above). Prevents runaway growth from
        consuming most of the frame in a pathological case (e.g. an
        uncorrected large-scale gradient that never reads as "flat").
        Defaults to 40% of the image's shorter dimension if not given.
    denoise_sigma : float or None
        Light Gaussian smoothing applied before computing each annulus's
        median, purely to tame per-pixel read noise -- same role as in
        `large_scale_excess_mask`. Defaults to `max(5, box_size // 6)`
        if not given.

    Returns
    -------
    mask : 2D bool ndarray
        `nebula_mask` grown outward as described above.
    """
    from scipy import ndimage as ndi

    if not nebula_mask.any():
        return nebula_mask

    # Tile-scale-dependent defaults are derived from box_size here rather
    # than hardcoded, so they stay consistent with whatever tile size the
    # rest of the pipeline is using.
    if step_px is None:
        step_px = max(5, box_size // 6)
    if denoise_sigma is None:
        denoise_sigma = max(5, box_size // 6)
    if effective_n_cap is None:
        effective_n_cap = box_size ** 2

    ny, nx = data.shape
    if max_growth_px is None:
        max_growth_px = 0.4 * min(ny, nx)

    work = data.copy()
    if star_mask is not None:
        work[star_mask] = np.nan
    smoothed = gaussian_filter_nan(work, denoise_sigma)

    if floor is None:
        # No coarse-floor map was supplied -- fall back to a single
        # robust global sky level, so excess is still measured relative
        # to sky rather than to zero (which would never be "flat").
        _, global_floor, _ = sigma_clipped_stats(smoothed, sigma=3, mask=~np.isfinite(smoothed))
        floor_map = np.full(data.shape, global_floor, dtype=np.float64)
    else:
        floor_map = floor

    if floor_rms is None:
        finite_vals = smoothed[np.isfinite(smoothed)]
        global_rms = mad_std(finite_vals) if finite_vals.size else 1.0
        rms_map = np.full(data.shape, global_rms, dtype=np.float64)
    else:
        finite_rms = floor_rms[np.isfinite(floor_rms) & (floor_rms > 0)]
        global_rms = float(np.nanmedian(finite_rms)) if finite_rms.size else 1.0
        rms_map = floor_rms

    labeled, n_labels = ndi.label(nebula_mask)
    out = np.zeros_like(nebula_mask)

    for lbl in range(1, n_labels + 1):
        region = labeled == lbl

        # Distance of every pixel in the frame from this region -- lets
        # each successive annulus be selected with a simple range test
        # below, in time roughly independent of how far the region ends
        # up growing.
        dist = ndi.distance_transform_edt(~region)

        grown = region.copy()
        flat_count = 0
        radius = 0.0

        while radius < max_growth_px:
            ring = (dist > radius) & (dist <= radius + step_px)
            if star_mask is not None:
                ring = ring & ~star_mask
            ring_valid = ring & np.isfinite(smoothed) & np.isfinite(floor_map)

            if ring_valid.sum() < 20:
                # Not enough usable pixels to judge this annulus (e.g.
                # the frame edge was reached) -- stop growing this
                # region rather than guessing further.
                break

            ring_excess = np.nanmedian(smoothed[ring_valid] - floor_map[ring_valid])
            ring_rms = np.nanmedian(rms_map[ring_valid])
            if not np.isfinite(ring_rms) or ring_rms <= 0:
                ring_rms = global_rms

            # Effective noise on this annulus's median, with the pixel
            # count capped at effective_n_cap -- see that parameter's
            # docstring above for the reasoning.
            n_eff = min(ring_valid.sum(), effective_n_cap)
            eff_sigma = ring_rms / np.sqrt(n_eff)

            if abs(ring_excess) < sigma_stop * eff_sigma:
                flat_count += 1
            else:
                flat_count = 0
                grown |= ring

            radius += step_px

            if flat_count >= consecutive_flat_rings:
                break

        out |= grown

    return out


def build_source_and_nebula_masks(data, sigma_thresh=6, star_dilation=3,
                                   nebula_sigma_thresh=5, nebula_min_size=500,
                                   box_size=100, large_scale_rescue=True,
                                   large_scale_sigma_thresh=4,
                                   large_scale_coarse_multiplier=4,
                                   large_scale_min_size=1000,
                                   large_scale_iterations=2,
                                   nebula_buffer_step_px=None,
                                   nebula_buffer_sigma_stop=1.5,
                                   nebula_buffer_consecutive_flat=3,
                                   nebula_buffer_max_growth_px=None,
                                   nebula_buffer_effective_n_cap=None,
                                   verbose=False):
    """
    Build the full set of masks used by `determine_background`: a star
    mask, a nebula/galaxy mask, and their union.

    This combines several detection stages:

    1. `star_gradient_mask` -- compact sources (stars).
    2. `nebula_scale_residual_mask` -- extended emission, detected via
       local contrast between two smoothing scales.
    3. (if `large_scale_rescue`) `large_scale_excess_mask` -- extended
       emission too large to register as local contrast, detected via
       absolute excess above a coarse sky floor (`_coarse_floor_background`).
       This is refined over `large_scale_iterations` passes: each pass's
       detections are excluded before re-estimating the sky floor for the
       next pass, so the floor isn't itself biased by the object it's
       trying to measure around.
    4. `_adaptive_nebula_buffer` -- grows every detected nebula/galaxy
       region outward until its own flux reaches the noise floor (see
       that function's docstring for why this matters).

    Parameters
    ----------
    data : 2D ndarray
        Image data.
    sigma_thresh, star_dilation : float, int
        Passed to `star_gradient_mask`.
    nebula_sigma_thresh, nebula_min_size : float, int
        Passed to `nebula_scale_residual_mask` (as `sigma_thresh`,
        `min_size`).
    box_size : int
        The pipeline's normal tile size; used both for the coarse-floor
        box size (`box_size * large_scale_coarse_multiplier`) and,
        by default, to derive the nebula-buffer growth parameters.
    large_scale_rescue : bool
        Whether to also run the large-scale excess test (step 3 above)
        in addition to the residual-contrast test.
    large_scale_sigma_thresh, large_scale_min_size : float, int
        Passed to `large_scale_excess_mask` (as `sigma_thresh`, `min_size`).
    large_scale_coarse_multiplier : float
        Passed to `_coarse_floor_background` as `coarse_multiplier`.
    large_scale_iterations : int
        Number of refinement passes for the coarse floor/large-scale
        mask (see step 3 above). 1 disables refinement (a single pass);
        2-3 is usually enough to converge for a single dominant
        galaxy/nebula in the frame.
    nebula_buffer_step_px, nebula_buffer_sigma_stop,
    nebula_buffer_consecutive_flat, nebula_buffer_max_growth_px,
    nebula_buffer_effective_n_cap :
        Passed to `_adaptive_nebula_buffer` (as `step_px`, `sigma_stop`,
        `consecutive_flat_rings`, `max_growth_px`, `effective_n_cap`
        respectively). The step/effective-N parameters default to None,
        which tells `_adaptive_nebula_buffer` to derive them from
        `box_size` automatically -- see that function's docstring.
    verbose : bool
        If True, print the fraction of the image flagged at each stage.

    Returns
    -------
    star_mask : 2D bool ndarray
    nebula_mask : 2D bool ndarray
    source_mask : 2D bool ndarray
        `star_mask | nebula_mask`, the mask actually passed to the
        background estimator.
    """
    star_mask = star_gradient_mask(data, sigma_thresh=sigma_thresh, dilation=star_dilation)

    nebula_mask_residual = nebula_scale_residual_mask(data, sigma_thresh=nebula_sigma_thresh,
                                                      min_size=nebula_min_size, star_mask=star_mask)

    # A coarse sky-level/RMS reference is needed by the buffer-growth
    # step below regardless of whether the large-scale excess TEST is
    # enabled, so it's always computed here at least once; large_scale_rescue
    # only controls whether it's ALSO used as an additional detection test
    # (with iterative refinement) below. It's important that this excludes
    # the residual-test detections too, not just stars: for something
    # like a large galaxy, the residual test is typically where almost
    # all of the detected footprint actually comes from, and leaving it
    # unmasked here would let the object's own light bias the very
    # floor/RMS reference the buffer-growth step relies on.
    floor, floor_rms = _coarse_floor_background(
        data, star_mask | nebula_mask_residual, box_size,
        coarse_multiplier=large_scale_coarse_multiplier)

    nebula_mask = nebula_mask_residual

    if large_scale_rescue:
        combined_mask_for_floor = star_mask | nebula_mask_residual
        large_scale_mask = np.zeros_like(nebula_mask)

        for iteration in range(max(large_scale_iterations, 1)):
            floor, floor_rms = _coarse_floor_background(
                data, combined_mask_for_floor, box_size,
                coarse_multiplier=large_scale_coarse_multiplier)
            large_scale_mask = large_scale_excess_mask(
                data, floor, floor_rms, star_mask=star_mask,
                sigma_thresh=large_scale_sigma_thresh, min_size=large_scale_min_size)

            # Feed this pass's detections into the mask used for the
            # NEXT pass's floor, so the floor stops sampling the object
            # itself.
            combined_mask_for_floor = star_mask | nebula_mask_residual | large_scale_mask

        nebula_mask = nebula_mask_residual | large_scale_mask

        if verbose:
            print(f'Large-scale (coarse-floor) nebula mask fraction: {large_scale_mask.mean():.4f}')

    if nebula_mask.any():
        # floor/floor_rms are the same coarse-box sky-level/RMS maps used
        # for detection above -- reused here so each growth step is
        # tested against its own local sky level and noise, keeping
        # detection and buffer growth self-consistent.
        nebula_mask = _adaptive_nebula_buffer(
            data, nebula_mask, star_mask=star_mask, floor=floor, floor_rms=floor_rms,
            box_size=box_size, step_px=nebula_buffer_step_px, sigma_stop=nebula_buffer_sigma_stop,
            consecutive_flat_rings=nebula_buffer_consecutive_flat,
            max_growth_px=nebula_buffer_max_growth_px,
            effective_n_cap=nebula_buffer_effective_n_cap)

        if verbose:
            print(f'Nebula mask fraction after adaptive buffer: {nebula_mask.mean():.4f}')

    nebula_mask = nebula_mask & ~star_mask

    source_mask = star_mask | nebula_mask

    if verbose:
        print(f'Star mask fraction: {star_mask.mean():.4f}')
        print(f'Nebula mask fraction: {nebula_mask.mean():.4f}')

    return star_mask, nebula_mask, source_mask


def _kde_mode(pixels, max_samples=2000, grid_size=512, rng=None):
    """
    Estimate the mode (most common value) of a 1D distribution of pixel
    values using kernel density estimation (KDE).

    Used as a more robust alternative to a clipped mean/median for tiles
    where the pixel distribution is skewed -- e.g. a tile that's mostly
    clean sky but still has some contamination from a nearby source or
    nebula edge. The mode of such a distribution (the peak of the sky
    pixels' distribution) is a better estimate of the true sky level than
    the mean, which gets pulled toward the contaminated tail.

    Parameters
    ----------
    pixels : 1D ndarray
        Pixel values to estimate the mode of. May contain NaNs, which
        are dropped before estimation.
    max_samples : int
        If more than this many valid pixels are supplied, a random
        subsample of this size is used instead, for speed -- the mode
        estimate is stable well below the full pixel count for a smooth,
        roughly unimodal distribution.
    grid_size : int
        Number of points used to evaluate the KDE when locating its peak.
    rng : numpy.random.Generator or None
        Random generator used for subsampling (for reproducibility);
        a fresh default generator is created if not given.

    Returns
    -------
    mode : float
        Estimated mode of the input distribution, or NaN if fewer than
        20 valid pixels were supplied.
    """
    pixels = pixels[np.isfinite(pixels)]
    if pixels.size < 20:
        return np.nan

    if pixels.size > max_samples:
        if rng is None:
            rng = np.random.default_rng()
        pixels = rng.choice(pixels, size=max_samples, replace=False)

    try:
        kde = gaussian_kde(pixels)
    except np.linalg.LinAlgError:
        return float(np.nanmedian(pixels)) # degenerate (near-zero variance) tile -- just use the median

    x = np.linspace(pixels.min(), pixels.max(), grid_size)
    pdf = kde(x)
    return float(x[np.argmax(pdf)])


def tiled_kde_background(data, mask, box_size=100, min_clean_frac=0.2,
                          rng=None):
    """
    Estimate a per-tile sky level using `_kde_mode`, for tiles where too
    large a fraction of pixels is masked for the standard tiled
    background estimator's clipped statistic to be trustworthy.

    Only tiles with a masked fraction below `min_clean_frac`'s complement
    (i.e. `frac < min_clean_frac`, where `frac` is the CLEAN, unmasked
    fraction) are actually estimated here -- tiles at or above that
    threshold are left as NaN, since the standard tiled estimator already
    handles them fine and there's no need to spend time re-estimating
    them. Any tile below that threshold still gets an actual KDE-mode
    estimate computed from whatever clean pixels remain in it (subject to
    `_kde_mode`'s own minimum-pixel-count requirement); only tiles with
    truly too few clean pixels for even that end up NaN here, to be
    filled in from neighbouring tiles by the caller
    (`_fill_and_smooth_grid`).

    Parameters
    ----------
    data : 2D ndarray
        Image data.
    mask : 2D bool ndarray
        Combined source/nebula mask (True = excluded from the estimate).
    box_size : int
        Tile size, in pixels.
    min_clean_frac : float
        Minimum CLEAN (unmasked) pixel fraction a tile needs for the
        standard tiled estimator to be considered reliable. Tiles below
        this threshold get a KDE-mode estimate computed here instead.
    rng : numpy.random.Generator or None
        Passed through to `_kde_mode` for reproducible subsampling.

    Returns
    -------
    grid : 2D ndarray, shape (ny_tiles, nx_tiles)
        Per-tile KDE-mode sky-level estimate, NaN for tiles that either
        didn't need one or had too few clean pixels even for this method.
    valid_frac : 2D ndarray, shape (ny_tiles, nx_tiles)
        Per-tile clean (unmasked) pixel fraction.
    """
    ny, nx = data.shape
    ny_tiles = int(np.ceil(ny / box_size))
    nx_tiles = int(np.ceil(nx / box_size))

    grid = np.full((ny_tiles, nx_tiles), np.nan, dtype=np.float64)
    valid_frac = np.zeros((ny_tiles, nx_tiles), dtype=np.float64)

    for j in range(ny_tiles):
        y0, y1 = j * box_size, min((j + 1) * box_size, ny)
        for i in range(nx_tiles):
            x0, x1 = i * box_size, min((i + 1) * box_size, nx)

            tile = data[y0:y1, x0:x1]
            tile_mask = mask[y0:y1, x0:x1]

            clean = tile[(~tile_mask) & np.isfinite(tile)]
            frac = clean.size / tile.size if tile.size else 0.0
            valid_frac[j, i] = frac

            # Tiles with enough clean pixels are already handled fine by
            # the standard tiled estimator -- only tiles below the
            # trust threshold get a KDE-mode estimate computed here.
            # _kde_mode's own internal minimum-pixel-count check handles
            # the case where even this tile has too few clean pixels to
            # estimate anything meaningful (left NaN, picked up by the
            # nearest-neighbour infill in _fill_and_smooth_grid).
            if frac >= min_clean_frac:
                continue

            grid[j, i] = _kde_mode(clean, rng=rng)

    return grid, valid_frac


def _fill_and_smooth_grid(grid, smooth_sigma=1.0):
    """
    Fill any remaining NaNs in a coarse per-tile grid via nearest-
    neighbour interpolation, then lightly smooth it across tiles.

    Mirrors the box-to-box smoothing a tiled background estimator
    normally applies internally, so the KDE-rescued grid produced by
    `tiled_kde_background` behaves consistently with the rest of the
    background surface once resampled to full resolution.

    Parameters
    ----------
    grid : 2D ndarray
        Coarse per-tile values, possibly containing NaNs.
    smooth_sigma : float
        Gaussian smoothing scale (in tile units) applied after filling.

    Returns
    -------
    grid : 2D ndarray
        Filled and smoothed grid, same shape as the input.
    """
    if np.all(~np.isfinite(grid)):
        return grid

    nan_mask = ~np.isfinite(grid)
    if nan_mask.any():
        yy, xx = np.indices(grid.shape)
        valid_pts = np.column_stack([yy[~nan_mask], xx[~nan_mask]])
        valid_vals = grid[~nan_mask]
        from scipy.interpolate import griddata
        filled = griddata(valid_pts, valid_vals, (yy, xx), method='nearest')
        grid = np.where(nan_mask, filled, grid)

    if smooth_sigma > 0:
        grid = gaussian_filter(grid, sigma=smooth_sigma)

    return grid


def determine_background(data, box_size=100, filter_size=3,
                          min_clean_frac=0.2, nebula_rescue=True,
                          large_scale_rescue=True,
                          large_scale_sigma_thresh=4,
                          large_scale_coarse_multiplier=4,
                          large_scale_min_size=1000,
                          large_scale_iterations=2,
                          nebula_buffer_step_px=None,
                          nebula_buffer_sigma_stop=1.5,
                          nebula_buffer_consecutive_flat=3,
                          nebula_buffer_max_growth_px=None,
                          nebula_buffer_effective_n_cap=None,
                          blend_feather_sigma=1.5,
                          plot=False, testing_plot=False, verbose=False,
                          seed=None):
    """
    Estimate the sky background of an image, returning a full-resolution
    2D array suitable for subtracting from the data.

    This is the main entry point of the module. Pipeline:

    1. Build star and nebula/galaxy masks (`build_source_and_nebula_masks`).
    2. Run a standard tiled background estimator (`sep.Background`) on the
       masked image, giving a fast, spatially varying surface (e.g.
       capturing vignetting or optical gradients) wherever the mask
       leaves enough clean sky per tile.
    3. Identify tiles where too much of the tile is masked for that
       estimate to be trustworthy (heavily nebulous or crowded tiles).
    4. Re-estimate just those tiles using a KDE-mode estimate over their
       surviving clean pixels (`tiled_kde_background`), fill in any tile
       that still has too few clean pixels from its neighbours, and
       lightly smooth.
    5. Blend the standard and KDE-derived surfaces together using a
       smoothly varying weight (not a hard tile-by-tile switch), so a
       large contiguous rescued region (e.g. a big galaxy) doesn't leave
       a visible discontinuity in the background surface at its boundary.

    Parameters
    ----------
    data : 2D ndarray
        Image data.
    box_size : int
        Tile size (pixels) for the standard tiled background estimator,
        and the reference tile size other parameters throughout this
        module derive their own scales from.
    filter_size : int
        Box-to-box smoothing applied by the standard tiled estimator.
    min_clean_frac : float
        Minimum clean (unmasked) pixel fraction a tile needs before its
        standard estimate is trusted; passed to `tiled_kde_background`.
    nebula_rescue : bool
        If False, skip the KDE rescue step entirely and return the
        standard tiled estimate as-is, even where masking is heavy.
    large_scale_rescue, large_scale_sigma_thresh, large_scale_coarse_multiplier,
    large_scale_min_size, large_scale_iterations :
        Passed to `build_source_and_nebula_masks` (see there for details).
    nebula_buffer_step_px, nebula_buffer_sigma_stop,
    nebula_buffer_consecutive_flat, nebula_buffer_max_growth_px,
    nebula_buffer_effective_n_cap :
        Passed to `build_source_and_nebula_masks`, which forwards them to
        `_adaptive_nebula_buffer` (see there for details). The step/
        effective-N parameters default to None, which derives them from
        `box_size` automatically.
    blend_feather_sigma : float
        Smoothing scale (in tile units) applied to the sep/KDE blend
        weight before resampling to full resolution, controlling how
        gradual the transition between the two estimates is.
    plot, testing_plot : bool
        If either is True, save diagnostic PNGs of the masks and final
        background surface to the current working directory (see
        `_diagnostic_plots`).
    verbose : bool
        If True, print progress/diagnostic information at each stage.
    seed : int or None
        Random seed for the KDE-mode subsampling, for reproducibility.

    Returns
    -------
    background : 2D ndarray (float64)
        Full-resolution background surface, same shape as `data`.
        Subtract this from `data` to get a background-free image.
    background_rms : 2D ndarray (float64)
        Full-resolution per-pixel background RMS (noise) map, same
        shape as `data`, from the standard tiled estimator.
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=np.float64)
    finite_mask = ~np.isfinite(data)

    star_mask, nebula_mask, source_mask = build_source_and_nebula_masks(
        data, box_size=box_size, large_scale_rescue=large_scale_rescue,
        large_scale_sigma_thresh=large_scale_sigma_thresh,
        large_scale_coarse_multiplier=large_scale_coarse_multiplier,
        large_scale_min_size=large_scale_min_size,
        large_scale_iterations=large_scale_iterations,
        nebula_buffer_step_px=nebula_buffer_step_px,
        nebula_buffer_sigma_stop=nebula_buffer_sigma_stop,
        nebula_buffer_consecutive_flat=nebula_buffer_consecutive_flat,
        nebula_buffer_max_growth_px=nebula_buffer_max_growth_px,
        nebula_buffer_effective_n_cap=nebula_buffer_effective_n_cap,
        verbose=verbose)
    full_mask = source_mask | finite_mask


    sep_data = np.ascontiguousarray(np.nan_to_num(data, nan=0.0).astype(np.float32))
    sep_mask = np.ascontiguousarray(full_mask)

    bkg = sep.Background(sep_data, mask=sep_mask, bw=box_size, 
                         bh=box_size, fw=filter_size, fh=filter_size)
    background = bkg.back().astype(np.float64)
    background_rms = bkg.rms().astype(np.float64)

    neb_fraction = nebula_mask.sum() / max(np.isfinite(data).sum(), 1)

    if nebula_rescue and neb_fraction > 0.0:
        kde_grid, valid_frac = tiled_kde_background(data, full_mask, box_size=box_size,
                                                    min_clean_frac=min_clean_frac, rng=rng)

        rescue_tiles = valid_frac < min_clean_frac

        if rescue_tiles.any():
            kde_grid_filled = _fill_and_smooth_grid(kde_grid, smooth_sigma=1.0)

            ny_tiles, nx_tiles = kde_grid_filled.shape
            kde_full = zoom(kde_grid_filled, (data.shape[0] / ny_tiles, data.shape[1] / nx_tiles), order=1)
            kde_full = _match_shape(kde_full, data.shape)

            # Blend weight: 0 means trust the standard tiled estimate
            # fully, 1 means trust the KDE estimate fully. This is a
            # continuous value (how far below min_clean_frac a tile's
            # clean fraction is), smoothed across tiles before
            # upsampling, rather than a hard per-tile switch -- so a
            # large contiguous rescued region (e.g. a big galaxy
            # spanning many tiles) transitions gradually into the
            # standard estimate at its boundary instead of showing a
            # visible step in the background surface there.
            weight_tiles = np.clip((min_clean_frac - valid_frac) / min_clean_frac, 0.0, 1.0)
            weight_tiles = gaussian_filter(weight_tiles, sigma=blend_feather_sigma)

            weight_full = zoom(weight_tiles, (data.shape[0] / ny_tiles, data.shape[1] / nx_tiles), order=1)
            weight_full = _match_shape(weight_full, data.shape)
            weight_full = np.clip(weight_full, 0.0, 1.0)

            background = background * (1.0 - weight_full) + kde_full * weight_full

            if verbose:
                print(f'Nebulosity area approximate: {neb_fraction:.4f}; '
                      f'{rescue_tiles.sum()} / {rescue_tiles.size} tiles rescued via KDE-mode '
                      f'(feathered blend, sigma={blend_feather_sigma})')
    elif verbose:
        print(f'Nebulosity area approximate: {neb_fraction:.4f}; no rescue needed')

    if testing_plot or plot:
        _diagnostic_plots(data, star_mask, nebula_mask, source_mask, background)

    return background, background_rms


def _match_shape(arr, target_shape):
    """
    Pad or crop a 2D array so it exactly matches `target_shape`.

    Used after `scipy.ndimage.zoom`, whose output shape from a ratio-based
    zoom factor can be off by a pixel or two from the intended target
    shape due to floating-point rounding.

    Parameters
    ----------
    arr : 2D ndarray
        Array to pad/crop.
    target_shape : tuple of int
        Desired output shape.

    Returns
    -------
    out : 2D ndarray
        `arr` padded or cropped to exactly `target_shape`. Padding
        (if any) repeats the nearest edge row/column.
    """
    out = np.zeros(target_shape, dtype=arr.dtype)
    ys = min(arr.shape[0], target_shape[0])
    xs = min(arr.shape[1], target_shape[1])
    out[:ys, :xs] = arr[:ys, :xs]
    if ys < target_shape[0]:
        out[ys:, :xs] = arr[-1, :xs]
    if xs < target_shape[1]:
        out[:, xs:] = out[:, xs - 1:xs]
    return out


def _diagnostic_plots(data, star_mask, nebula_mask, source_mask, background):
    """
    Save PNG diagnostic plots of the input data, each mask, and the final
    background surface to the current working directory. Called from
    `determine_background` when `plot`/`testing_plot` is True.
    """
    plt.figure()
    plt.imshow(data, vmin=np.nanpercentile(data, 5), vmax=np.nanpercentile(data, 95))
    plt.title('Data')
    plt.savefig('data_with_nebulosity.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.imshow(star_mask)
    plt.title('Star Mask')
    plt.savefig('star_mask.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.imshow(nebula_mask)
    plt.title('Nebula Mask')
    plt.savefig('nebula_mask.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.imshow(source_mask)
    plt.title('Source Mask')
    plt.savefig('source_mask.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.imshow(background)
    plt.title('Final Background Surface')
    plt.savefig('final_background_surface.png', dpi=300, bbox_inches='tight')
    plt.close()


def background_residual_flatness(data, background, source_mask, box_size=100):
    """
    Measure how flat the sky is after background subtraction -- a
    single-number diagnostic for spotting frames where the detection/
    masking parameters may need adjusting.

    Computes the median residual (data minus background) in each tile,
    restricted to non-source pixels, and returns the standard deviation
    of those per-tile medians. A well-subtracted, genuinely flat sky
    should give a small value; a large value indicates spatially
    structured residual sky background -- e.g. an over- or
    under-subtracted region around a bright/extended object, or
    leftover large-scale gradient the estimator didn't fully capture.

    Recommended usage: compute this for every reduced frame and log it,
    so problems on an unusual field show up as an outlier value rather
    than only being noticed by eye later.

    Parameters
    ----------
    data : 2D ndarray
        Original (pre-subtraction) image data.
    background : 2D ndarray
        Background surface that was (or would be) subtracted from `data`.
    source_mask : 2D bool ndarray
        Pixels to exclude from the flatness measurement (stars, nebula).
    box_size : int
        Tile size (pixels) used for the per-tile median.

    Returns
    -------
    flatness : float
        Standard deviation of per-tile median residual sky level. Smaller
        is better (closer to a perfectly flat, background-subtracted sky).
    tile_medians : 2D ndarray
        Per-tile median residual, for optional plotting.
    """
    residual = data - background
    ny, nx = residual.shape
    ny_tiles = int(np.ceil(ny / box_size))
    nx_tiles = int(np.ceil(nx / box_size))

    tile_medians = np.full((ny_tiles, nx_tiles), np.nan)

    for j in range(ny_tiles):
        y0, y1 = j * box_size, min((j + 1) * box_size, ny)
        for i in range(nx_tiles):
            x0, x1 = i * box_size, min((i + 1) * box_size, nx)
            tile = residual[y0:y1, x0:x1]
            tile_mask = source_mask[y0:y1, x0:x1]
            clean = tile[(~tile_mask) & np.isfinite(tile)]
            if clean.size > 20:
                tile_medians[j, i] = np.median(clean)

    flatness = np.nanstd(tile_medians)
    return float(flatness), tile_medians