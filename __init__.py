"""
jspec
=====

A spectral reduction and analysis toolkit for JWST NIRSpec and MIRI data.

Two halves:

``jspec.pipelines``
    The existing Stage-1/2/3 reduction and re-extraction drivers (MIRI MRS,
    MIRI LRS, MIRI/NIRCam imaging, NIRSpec FS and IFU), moved here unchanged
    from the repository root. Thin shim modules at the root keep legacy
    ``import miri_mrs_auto_pipeline`` style imports working.

the analysis layer (this package's own modules)
    Everything downstream of a 1-D extraction:

    ``jspec.spectrum``     unified :class:`~jspec.spectrum.Spectrum` container
                           and readers for ecsv / csv / txt / x1d FITS, plus
                           rest-frame conversion and JWST resolving power.
    ``jspec.stitch``       segment matching and stitching -- globally solved
                           scale factors, edge trimming, tapered blending.
    ``jspec.continuum``    robust continuum / pseudo-continuum estimation.
    ``jspec.linedb``       line and dust-band reference data and matching.
    ``jspec.features``     broad and narrow feature (and dip) detection,
                           high-S/N region finding, and line identification.
    ``jspec.fitting``      nested-sampling profile fitting with velocities.
    ``jspec.plotting``     dust-band overlay and annotated feature plots.
    ``jspec.metallicity``  late-time SN NIR/MIR abundance diagnostics.

The intended end-to-end flow is::

    reduce (jspec.pipelines)  ->  stitch (jspec.stitch)
      ->  detect (jspec.features)  ->  identify (jspec.linedb)
      ->  fit (jspec.fitting)      ->  plot (jspec.plotting)

A command line front end for the analysis layer is available as
``python -m jspec <subcommand>`` -- see :mod:`jspec.cli`.
"""
from __future__ import annotations

__version__ = "0.1.0"

from ..spectrum import (
    Spectrum,
    Segment,
    read_spectrum,
    read_segments,
    resolving_power,
    air_to_vac,
    vac_to_air,
    MRS_RED_END_UM,
)

from ..epochs import (
    EpochGroup,
    group_paths_by_epoch,
    group_segments_by_epoch,
    check_single_epoch,
)

__all__ = [
    "check_single_epoch",
    "group_segments_by_epoch",
    "group_paths_by_epoch",
    "EpochGroup",
    "__version__",
    "Spectrum",
    "Segment",
    "read_spectrum",
    "read_segments",
    "resolving_power",
    "air_to_vac",
    "vac_to_air",
    "MRS_RED_END_UM",
]
