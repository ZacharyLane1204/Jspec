#!/usr/bin/env python3
"""
jwst_proposal_pipeline.py
============================

Single entry point for reducing an entire JWST proposal: give it a
`--proposal` number (and optionally a `--tag`), and it figures out which of
the four instrument modes this repo has pipelines for are actually present
in that proposal -- MIRI LRS Fixed-Slit, MIRI MRS (IFU), MIRI imaging, and
NIRSpec Fixed-Slit -- and runs each one it finds, in its own subfolder.

It does this by querying MAST once per mode (a cheap metadata-only query)
to check whether the proposal has any observations of that mode, then
invokes the corresponding script as a subprocess only for the modes that
are actually present. Modes with no data in the proposal are skipped
entirely (and reported as skipped), rather than being launched and failing.

Usage
-----
    python jwst_proposal_pipeline.py --proposal 1523

    python jwst_proposal_pipeline.py --proposal 1523 --tag visit1 \\
        --outdir ./reductions --max-epoch-gap-days 1.0 \\
        --nirspec-fs-args "--scale-background" \\
        --miri-lrs-fs-args "--scale-background" \\
        --miri-mrs-args "--src-x 34.2 --src-y 51.7 --bkg-mode polynomial"

Layout produced
----------------
<outdir>/
    NIRSpec_FS/      -- output of nirspec_fs_auto_pipeline.py
    MIRI_LRS_FS/     -- output of miri_lrs_fs_auto_pipeline.py
    MIRI_MRS/        -- output of miri_mrs_auto_pipeline.py
    MIRI_Imaging/    -- output of miri_imaging_auto_pipeline.py

Each subfolder is exactly what that pipeline's own `--outdir` produces, so
everything downstream of this wrapper (final_spectrum/, stage3/, etc.)
looks the same as if you'd run that script directly with
`--outdir <outdir>/<Mode>`.

Passing arguments through to individual pipelines
--------------------------------------------------
The four pipeline scripts' CLIs have diverged as they've been tuned per
instrument, so there is no single set of "extra args" that's safe to hand
every pipeline verbatim -- e.g. `--scale-background` exists only on
NIRSpec FS and MIRI LRS FS, `--src-x`/`--src-y`/`--bkg-mode` only on MIRI MRS, and
`--max-epoch-gap-days` doesn't exist on MIRI imaging at all (there's no
final-spectrum epoch-splitting step for images). Passing a flag to a
pipeline that doesn't define it is a hard argparse error, so this wrapper
deliberately does NOT offer a single blanket passthrough. Instead:

- `--skip-download` and `--max-cores` are genuinely common to all four
  scripts and are forwarded to every mode that runs.
- `--max-epoch-gap-days` is forwarded to the three modes that have a
  final-spectrum epoch-splitting step (NIRSpec FS, MIRI LRS FS, MIRI MRS)
  and is silently not passed to MIRI imaging.
- Anything mode-specific goes through that mode's own `--<mode>-args`
  string, e.g. `--nirspec-fs-args "--scale-background --tag test"`. Each
  string is split with `shlex.split` and appended to that mode's command
  only.
- `--only MODE` restricts the run to a single instrument mode (choices:
  nirspec_fs, miri_lrs_fs, miri_mrs, miri_imaging) for testing or
  re-running just one piece -- when using `--only`, the corresponding
  `--<mode>-args` string is still the right place for flags specific to
  that pipeline.

Notes
-----
- All four `*_auto_pipeline.py` scripts, plus `combine_nirspec_orders.py`,
  must be in the same directory as this wrapper (or otherwise importable/
  on PATH) -- this script runs them as subprocesses via
  `sys.executable <script> ...`.
"""

import argparse
import os
import shlex
import subprocess
import sys

from astroquery.mast import Observations

MODES = {
    "nirspec_fs": {
        "label": "NIRSpec FS",
        "subdir": "NIRSpec_FS",
        "script": "nirspec_fs_auto_pipeline.py",
        "instrument_name": ["NIRSPEC/SLIT"],
        "supports_max_epoch_gap": True,
    },
    "miri_lrs_fs": {
        "label": "MIRI LRS FS",
        "subdir": "MIRI_LRS_FS",
        "script": "miri_lrs_fs_auto_pipeline.py",
        "instrument_name": ["MIRI/SLIT"],
        "supports_max_epoch_gap": True,
    },
    "miri_mrs": {
        "label": "MIRI MRS",
        "subdir": "MIRI_MRS",
        "script": "miri_mrs_auto_pipeline.py",
        "instrument_name": ["MIRI/IFU"],
        "supports_max_epoch_gap": True,
    },
    "miri_imaging": {
        "label": "MIRI Imaging",
        "subdir": "MIRI_Imaging",
        "script": "miri_imaging_auto_pipeline.py",
        "instrument_name": ["MIRI/IMAGE"],
        "supports_max_epoch_gap": False,
    },
}


def mode_present(proposal_num, instrument_name):
    """Cheap MAST metadata query: does this proposal have any observations
    of this instrument mode at all?"""
    obs_table = Observations.query_criteria(
        instrument_name=instrument_name,
        provenance_name=["CALJWST"],
        proposal_id=[str(proposal_num)],
    )
    return len(obs_table) > 0


def run_mode(mode_key, proposal_num, outdir, tag, common_args, max_epoch_gap_days,
             mode_args_str, script_dir):
    info = MODES[mode_key]
    script_path = os.path.join(script_dir, info["script"])
    mode_outdir = os.path.join(outdir, info["subdir"])
    os.makedirs(mode_outdir, exist_ok=True)

    cmd = [sys.executable, script_path,
           "--proposal", str(proposal_num),
           "--outdir", mode_outdir]
    if tag:
        cmd += ["--tag", tag]
    cmd += common_args
    if max_epoch_gap_days is not None and info["supports_max_epoch_gap"]:
        cmd += ["--max-epoch-gap-days", str(max_epoch_gap_days)]
    if mode_args_str:
        cmd += shlex.split(mode_args_str)

    print(f"\n{'=' * 70}\nRunning {info['label']}: {' '.join(cmd)}\n{'=' * 70}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proposal", required=True, help="JWST proposal/program number, e.g. 1523")
    parser.add_argument("--tag", default=None,
                         help="Optional string passed through to every pipeline that runs -- "
                              "included in each pipeline's output directory/file names.")
    parser.add_argument("--outdir", default=None,
                         help="Base output directory (default: ./jwst_data_<proposal>[_<tag>]); "
                              "each mode gets its own subfolder underneath it.")
    parser.add_argument("--skip-download", action="store_true",
                         help="Forwarded to every mode that runs: reuse each mode's existing "
                              "uncal/ files instead of querying MAST again.")
    parser.add_argument("--max-cores", default=None, choices=["none", "quarter", "half", "all"],
                         help="Forwarded to every mode that runs (cores used for the jump step). "
                              "If omitted, each pipeline's own default is used.")
    parser.add_argument("--max-epoch-gap-days", type=float, default=None,
                         help="Forwarded to NIRSpec FS, MIRI LRS FS, and MIRI MRS (the three modes "
                              "with a final-spectrum epoch-splitting step); has no effect on MIRI "
                              "imaging, which doesn't have one. If omitted, each pipeline's own "
                              "default (1.0 day) is used.")
    parser.add_argument("--nirspec-fs-args", default=None,
                         help="Extra arguments for nirspec_fs_auto_pipeline.py only, as a single "
                              "quoted string, e.g. \"--scale-background --trace-exclude-frac 0.2\"")
    parser.add_argument("--miri-lrs-fs-args", default=None,
                         help="Extra arguments for miri_lrs_fs_auto_pipeline.py only, e.g. "
                              "\"--scale-background --optimal-extraction\"")
    parser.add_argument("--miri-mrs-args", default=None,
                         help="Extra arguments for miri_mrs_auto_pipeline.py only, e.g. "
                              "\"--src-x 34.2 --src-y 51.7 --bkg-mode polynomial\"")
    parser.add_argument("--miri-imaging-args", default=None,
                         help="Extra arguments for miri_imaging_auto_pipeline.py only.")
    parser.add_argument("--only", choices=list(MODES.keys()), default=None,
                         help="Restrict the run to a single instrument mode.")
    args = parser.parse_args()

    proposal_num = int(args.proposal)
    proposal_padded = f"{proposal_num:05d}"
    tag_suffix = f"_{args.tag}" if args.tag else ""
    outdir = args.outdir or os.path.abspath(f"./jwst_data_{proposal_padded}{tag_suffix}")
    os.makedirs(outdir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    common_args = []
    if args.skip_download:
        common_args.append("--skip-download")
    if args.max_cores is not None:
        common_args += ["--max-cores", args.max_cores]

    mode_args = {
        "nirspec_fs": args.nirspec_fs_args,
        "miri_lrs_fs": args.miri_lrs_fs_args,
        "miri_mrs": args.miri_mrs_args,
        "miri_imaging": args.miri_imaging_args,
    }

    modes_to_check = [args.only] if args.only else list(MODES.keys())

    print(f"Checking which instrument modes are present in proposal {proposal_padded} ...")
    present = {}
    for mode_key in modes_to_check:
        info = MODES[mode_key]
        found = mode_present(proposal_num, info["instrument_name"])
        present[mode_key] = found
        status = "found" if found else "not found -- skipping"
        print(f"  {info['label']}: {status}")

    results = {}
    for mode_key in modes_to_check:
        if not present[mode_key]:
            results[mode_key] = "skipped (no data)"
            continue
        rc = run_mode(mode_key, proposal_num, outdir, args.tag, common_args,
                       args.max_epoch_gap_days, mode_args[mode_key], script_dir)
        results[mode_key] = "OK" if rc == 0 else f"FAILED (exit code {rc})"

    print(f"\n{'=' * 70}\nSummary for proposal {proposal_padded}\n{'=' * 70}")
    for mode_key in modes_to_check:
        print(f"  {MODES[mode_key]['label']}: {results[mode_key]}")
    print(f"\nOutput directory: {outdir}")


if __name__ == "__main__":
    main()