#!/usr/bin/env python3
"""
run_quiet.py
============

Run any jwst-based console script with the log output turned down.

``jspec``'s own pipelines quiet themselves (see ``jwst_logging.py``), but
third-party tools built on jwst -- MRS-PFPC's ``pfpc_proc`` / ``pfpc_cor``,
for instance -- do not, and they usually live in their own conda
environment where ``jspec`` is not installed.

This script is deliberately standalone: it loads ``jwst_logging.py`` from
its own directory by path, so it works by absolute path from any
environment that has ``jwst`` in it, with nothing installed::

    conda activate mrs_pfpc
    cd SN2024aecx
    python /path/to/JSpec/pipelines/run_quiet.py pfpc_proc SN2024aecx_2025-06-01

The command is looked up in the active environment's ``console_scripts``
entry points and called in-process, so logging is already configured
before the first pipeline step runs -- which is the whole point, since
stpipe only leaves the configuration alone if one was applied first.

Options (anything after the command name is passed through untouched)::

    --verbose            full INFO logging, as if run directly
    --log-file FILE      keep full detail on disk, quiet console
"""
import argparse
import importlib.util
import os
import sys


def _load_jwst_logging():
    """Import the sibling jwst_logging.py without needing jspec installed."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "jwst_logging.py")
    spec = importlib.util.spec_from_file_location("_jspec_jwst_logging", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve(command):
    """Find `command` among the environment's console_scripts."""
    from importlib.metadata import entry_points
    try:
        scripts = entry_points(group="console_scripts")
    except TypeError:                       # Python < 3.10
        scripts = entry_points().get("console_scripts", [])
    for ep in scripts:
        if ep.name == command:
            return ep.load()
    available = sorted(ep.name for ep in scripts)
    raise SystemExit(
        f"No console script named {command!r} in this environment.\n"
        f"Is the right conda environment active? Available: "
        + ", ".join(available[:40]) + (" ..." if len(available) > 40 else "")
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true",
                     help="Show the full jwst/stpipe INFO output.")
    ap.add_argument("--log-file", default=None,
                     help="Also write the log to this file.")
    ap.add_argument("command",
                     help="Console script to run, e.g. pfpc_proc")
    ap.add_argument("args", nargs=argparse.REMAINDER,
                     help="Arguments passed through to the command.")
    args = ap.parse_args()

    jlog = _load_jwst_logging()
    level = "INFO" if args.verbose else "WARNING"
    jlog.set_env_level(level)      # inherited by spawned worker processes
    jlog.configure_jwst_logging(level, log_file=args.log_file)

    func = _resolve(args.command)
    sys.argv = [args.command] + list(args.args)
    sys.exit(func())


if __name__ == "__main__":
    main()
