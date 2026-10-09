#!/usr/bin/env python3
"""
jwst_logging.py
===============

Turn off the jwst/stpipe INFO firehose.

Why this module exists
----------------------
The obvious thing::

    logging.getLogger("stpipe").setLevel(logging.WARNING)

does not work, and silently so. Two separate reasons:

1. It misses loggers. ``jwst`` configures
   ``("jwst", "stcal", "stdatamodels", "stpipe", "tweakwcs", "CRDS",
   "py.warnings")`` -- note ``stcal``, which is where the
   ``dark_current.dark_sub`` "Science data nints=..." lines come from.

2. More importantly, stpipe *overrides* whatever you set. On the first
   ``Step.call()``, ``stpipe.step`` runs::

       elif configure_log and _log.LogConfig.applied is None:
           log_cfg = _log.load_configuration(...)   # defaults to INFO

   and ``LogConfig.apply()`` then does ``log.setLevel(self.level)`` on
   every one of those names. So any level you set at import time is
   reset to INFO the moment the pipeline actually starts.

The fix is in that ``is None`` check: if a configuration has *already*
been applied, stpipe leaves it alone. So rather than fighting it, we
apply our own configuration first and stpipe stands down.

Note this reaches into ``stpipe._log``, which is private. It was public
as ``stpipe.log`` before stpipe 1.1.0. Both paths are tried, and if
neither exists we fall back to plain ``setLevel`` calls -- degraded (it
will be overridden again) but never fatal.

Usage
-----
    from jspec.pipelines.jwst_logging import configure_jwst_logging
    configure_jwst_logging("WARNING")      # quiet: our own output only
    configure_jwst_logging("INFO")         # the full jwst firehose
    configure_jwst_logging("WARNING", log_file="run.log")
"""
import logging
import os

# Level recorded here so spawned child processes inherit it.
_LEVEL_ENV = "JSPEC_JWST_LOG_LEVEL"

# Mirrors jwst.stpipe.core.JwstStep.get_stpipe_loggers(), used when jwst
# is not importable or has changed the list. "asdf" is ours, not jwst's.
_FALLBACK_LOG_NAMES = (
    "jwst", "stcal", "stdatamodels", "stpipe", "tweakwcs", "CRDS",
    "py.warnings", "asdf",
)

# Extra loggers of our own. Deliberately NOT astroquery/astropy: those
# install a custom logger class via logging.setLoggerClass(), and calling
# getLogger() on them first creates a plain Logger that is missing the
# methods they expect ("'Logger' object has no attribute '_set_defaults'"
# at import). Only touch libraries that use stdlib logging normally.
_EXTRA_LOG_NAMES = ("asdf", "s3fs", "botocore")

# The LogConfig we applied, kept so a second call can undo it first.
_APPLIED = None
_APPLIED_NAMES = ()


def _stpipe_log_module():
    """stpipe's logging module, private since 1.1.0, public before it."""
    try:
        from stpipe import _log
        return _log
    except ImportError:
        pass
    try:
        from stpipe import log
        return log
    except ImportError:
        return None


def log_names():
    """Logger names to configure: ask jwst, else use the known list."""
    try:
        from jwst.stpipe.core import JwstStep
        names = list(JwstStep.get_stpipe_loggers())
    except Exception:
        names = list(_FALLBACK_LOG_NAMES)
    # Only adopt an extra logger that already exists. Calling getLogger()
    # on a name before its library is imported creates a plain Logger and
    # can break libraries that install their own logger class.
    existing = logging.Logger.manager.loggerDict
    for extra in _EXTRA_LOG_NAMES:
        if extra not in names and extra in existing:
            names.append(extra)
    return names


def configure_jwst_logging(level="WARNING", log_file=None):
    """Set the console log level for jwst and friends.

    Parameters
    ----------
    level : str or int
        "WARNING" (default) keeps the console to our own print()s and
        tqdm bars, while still letting real warnings and errors through.
        "INFO" restores jwst's normal per-step chatter.
    log_file : str, optional
        Also write the log to this file. Useful with the default quiet
        console: a clean terminal, with the full detail kept on disk for
        when a run goes wrong.

    Returns
    -------
    bool
        True if the override-proof path was used. False means stpipe
        will reset these levels to INFO on the first step.
    """
    global _APPLIED, _APPLIED_NAMES

    if isinstance(level, str):
        level_no = logging.getLevelName(level.upper())
        if not isinstance(level_no, int):
            raise ValueError(f"Unknown log level: {level!r}")
    else:
        level_no = int(level)

    names = log_names()
    mod = _stpipe_log_module()

    if mod is not None and hasattr(mod, "load_configuration"):
        try:
            # Undo a previous call, otherwise apply() skips loggers it
            # has already recorded a level for and nothing changes.
            if _APPLIED is not None:
                _APPLIED.undo(list(_APPLIED_NAMES))
                if hasattr(mod, "LogConfig"):
                    mod.LogConfig.applied = None

            try:
                cfg = mod.load_configuration(log_level=level_no,
                                              log_file=log_file)
            except TypeError:
                # Pre-1.1.0 stpipe took a config file, not kwargs.
                cfg = mod.load_configuration(mod._find_logging_config_file())
                cfg.level = level_no
            cfg.apply(names)
            _APPLIED, _APPLIED_NAMES = cfg, tuple(names)
            return True
        except Exception:
            pass

    for name in names:
        logging.getLogger(name).setLevel(level_no)
    return False


def configure_from_env(default="WARNING"):
    """Apply the level recorded in ``JSPEC_JWST_LOG_LEVEL``.

    Call this at *module import* time, not just from ``main()``.

    On macOS and Windows, multiprocessing uses the "spawn" start method:
    jwst/stcal steps that parallelise (jump detection, ramp fitting) start
    children that re-import the module but never run ``main()``. Those
    children inherit the parent's stderr, so if they are left at stpipe's
    INFO default their chatter still reaches your terminal even though the
    parent is quiet. Environment variables *are* inherited across spawn,
    so recording the level there is what gets it into the children.
    """
    return configure_jwst_logging(os.environ.get(_LEVEL_ENV, default))


def set_env_level(level):
    """Record the level so spawned child processes pick it up."""
    os.environ[_LEVEL_ENV] = level if isinstance(level, str) else str(level)


def quiet_jwst_logging(verbose=False, log_file=None):
    """``configure_jwst_logging`` driven by a ``--verbose`` flag."""
    level = "INFO" if verbose else "WARNING"
    set_env_level(level)
    return configure_jwst_logging(level, log_file=log_file)
