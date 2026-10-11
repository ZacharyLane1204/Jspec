"""Run a proposal-driven pipeline once per proposal ID.

A target is often observed under several programs -- a DD program for the
discovery epoch, a GO program for the follow-up, and so on -- and all of
those exposures belong in the same output tree. Rather than teach every
pipeline to merge several MAST queries, this module runs the existing
single-proposal ``main()`` once per ID, in order, against the same
``--outdir``. Each run downloads and reduces only its own exposures, and
because epoch directories are keyed on observation date rather than on
program, the results interleave by date exactly as if they had come from
one program.

Usage at the bottom of a pipeline module::

    if __name__ == "__main__":
        main_per_proposal(main)

``main()`` itself is untouched and still parses a single ``--proposal``;
this wrapper splits the command line before argparse ever sees it.

Accepted spellings, all equivalent::

    --proposal 4217 6583 9105
    --proposal 4217,6583,9105
    --proposal=4217 --proposal 6583
"""

import os
import sys
import time
import traceback

__all__ = ["split_proposals", "main_per_proposal"]

_FLAG = "--proposal"


def _looks_like_id(tok):
    """True for a bare proposal ID or a comma-joined list of them.

    Used to decide how many tokens after a bare ``--proposal`` belong to
    it. Anything that is not digits-and-commas ends the run, which also
    stops a missing value from swallowing the next flag's argument.
    """
    tok = tok.strip()
    if not tok or tok.startswith("-"):
        return False
    return all(part.strip().isdigit() for part in tok.split(",") if part.strip())


def split_proposals(argv):
    """Split ``argv`` into (proposal IDs, everything else).

    Returns the IDs in the order given, with duplicates removed but the
    first occurrence kept, and the remaining arguments with every
    ``--proposal`` token stripped out. Order of the remainder is
    preserved; argparse does not care where the flag was.
    """
    props, rest = [], []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == _FLAG:
            i += 1
            # Consume every ID that follows, so `--proposal 1 2 3` works.
            while i < len(argv) and _looks_like_id(argv[i]):
                props.extend(p for p in argv[i].split(",") if p.strip())
                i += 1
            continue
        if tok.startswith(_FLAG + "="):
            value = tok[len(_FLAG) + 1:]
            props.extend(p for p in value.split(",") if p.strip())
            i += 1
            continue
        rest.append(tok)
        i += 1

    seen, unique = set(), []
    for p in props:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            unique.append(p)
    return unique, rest


def main_per_proposal(main_func, argv=None):
    """Call ``main_func`` once per proposal ID on the command line.

    With zero or one ID this is a transparent pass-through, so the
    single-proposal behaviour -- including argparse's own error messages
    and exit codes -- is completely unchanged.

    With several IDs, a failure in one does not abort the rest: the
    traceback is printed, the run is recorded, and the next proposal
    starts. Partial results are usually still worth having, and the
    summary at the end says exactly which IDs need re-running. The exit
    code is non-zero if any of them failed.
    """
    raw = list(sys.argv[1:]) if argv is None else list(argv)
    props, rest = split_proposals(raw)

    if len(props) <= 1:
        return main_func()

    prog = sys.argv[0]
    original = list(sys.argv)
    # sys.argv[0] is a file path under `python -m`, which is not a command
    # anyone can paste back in, so rebuild the invocation from the module.
    mod = getattr(main_func, "__module__", None)
    if mod == "__main__":
        # Under `python -m pkg.mod` the function's module is "__main__";
        # the dotted name only survives on the module spec.
        spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        mod = getattr(spec, "name", None)
    invocation = (f"python -m {mod}" if mod and mod != "__main__"
                  else os.path.basename(prog))
    results = []

    print(f"Running {len(props)} proposals sequentially: {', '.join(props)}")
    print("Each is downloaded and reduced on its own; they share one output "
          "tree and are separated by observation date, not by program.\n")

    try:
        for n, pid in enumerate(props, 1):
            banner = f"  proposal {pid}   ({n} of {len(props)})  "
            print("\n" + "=" * 78)
            print(banner.center(78, "="))
            print("=" * 78 + "\n", flush=True)

            sys.argv = [prog, _FLAG, pid] + rest
            t0 = time.time()
            try:
                main_func()
            except SystemExit as exc:
                # argparse and explicit sys.exit() both land here. A clean
                # exit is success; anything else is this proposal failing,
                # not a reason to abandon the ones after it.
                code = exc.code
                ok = code in (None, 0)
                results.append((pid, ok, time.time() - t0,
                                None if ok else f"exited with status {code}"))
                if not ok:
                    print(f"\nproposal {pid} exited with status {code}; "
                          f"continuing", file=sys.stderr, flush=True)
            except KeyboardInterrupt:
                print(f"\ninterrupted during proposal {pid}; "
                      f"{len(props) - n} not started", file=sys.stderr)
                results.append((pid, False, time.time() - t0, "interrupted"))
                break
            except Exception as exc:
                traceback.print_exc()
                print(f"\nproposal {pid} failed ({type(exc).__name__}); "
                      f"continuing with the rest", file=sys.stderr, flush=True)
                results.append((pid, False, time.time() - t0,
                                f"{type(exc).__name__}: {exc}"))
            else:
                results.append((pid, True, time.time() - t0, None))
    finally:
        sys.argv = original

    print("\n" + "=" * 78)
    print("  proposal summary  ".center(78, "="))
    print("=" * 78)
    for pid, ok, dt, msg in results:
        mins = dt / 60.0
        status = "ok" if ok else "FAILED"
        print(f"  {pid:>8s}  {status:>7s}  {mins:6.1f} min"
              + (f"   {msg}" if msg else ""))
    missing = [p for p in props if p not in {r[0] for r in results}]
    for pid in missing:
        print(f"  {pid:>8s}  {'skipped':>7s}")

    failed = [pid for pid, ok, _, _ in results if not ok] + missing
    if failed:
        print(f"\n{len(failed)} of {len(props)} proposals did not finish. "
              f"Re-run just those with:")
        print(f"  {invocation} --proposal {' '.join(failed)} "
              f"{' '.join(rest)}")
        sys.exit(1)

    print(f"\nAll {len(props)} proposals finished.")
