"""
jspec.pipelines
===============

The JWST Stage-1/2/3 reduction and re-extraction drivers, moved here from the
repository root without modification.

These modules were written as standalone top-level scripts and import each
other by bare module name (``import miri_mrs_auto_pipeline as mp``,
``from combine_nirspec_orders import get_mjd``). Rather than rewrite those
imports -- several of them are module-alias imports used for attribute
monkey-patching, and several modules do their own ``sys.path`` manipulation --
this package simply puts its own directory on ``sys.path`` at import time, so
the bare-name imports resolve exactly as they did when every script sat in the
repository root.

That means all three of these work and refer to the same module object:

    import jspec.pipelines.miri_mrs_auto_pipeline as mp   # package path
    from jspec.pipelines import abi_reextract_pfpc        # package path
    import miri_mrs_auto_pipeline as mp                   # legacy bare name

The last form also keeps working from the repository root because of the thin
shim modules left behind there (``miri_mrs_auto_pipeline.py`` etc.), which
re-export from here. Existing notebooks therefore need no edits.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_PIPELINE_DIR = _os.path.dirname(_os.path.abspath(__file__))

if _PIPELINE_DIR not in _sys.path:
    # Append rather than prepend: a module the user has deliberately put
    # earlier on sys.path should still win.
    _sys.path.append(_PIPELINE_DIR)

#: Modules in this package, in rough dependency order. Useful for the shims
#: and for ``python -m jspec.pipelines`` style discovery.
MODULES = (
    "jwst_proposal_pipeline",
    "mast_uncal_download",
    "background_subtraction",
    "kde_bkg_correct",
    "astrobkginterp_thermal",
    "combine_nirspec_orders",
    "combine_epoch_spectra",
    "miri_mrs_auto_pipeline",
    "miri_mrs_refine_extract",
    "miri_lrs_fs_auto_pipeline",
    "miri_imaging_auto_pipeline",
    "nircam_imaging_auto_pipeline",
    "nirspec_fs_refining",
    "nirspec_fs_auto_pipeline",
    "nirspec_refine_extract",
    "nirspec_ifu_auto_pipeline",
    "nirspec_miri_calibration_utils",
    "abi_reextract_pfpc",
    "prepare_mrs_pfpc_epochs",
    "rscale_curve_of_growth",
    "get_source_radec",
    "check_source_position_consistency",
    "compare_channel_extraction",
    "compare_epoch_diagnostics",
    "make_jwst_table",
)

__all__ = ["MODULES"]
