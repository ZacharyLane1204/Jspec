#!/usr/bin/env python3
"""
compare_epoch_diagnostics.py
=============================

Pull the miri_mrs_refine_extract.py per-epoch diagnostics that matter for a
NIRSpec/MIRI flux mismatch -- ch1 background-tuning results (wing_frac,
wing_ok, bias_over_sigma) and the chosen extraction aperture (RSCALE) --
out of each epoch's <name>_report.json and print them side by side.

ch1 (4.90-7.65 um) is the MIRI sub-band bordering the NIRSpec join, so it's
the one most relevant to a mismatch seen right at ~5 um.

Usage
-----
    python compare_epoch_diagnostics.py \\
        --report Jan=refined/17012025/SN2024aecx_17012025_report.json \\
        --report Jun=refined/01062025/SN2024aecx_01062025_report.json \\
        --report Dec=refined/28122025/SN2024aecx_28122025_report.json
"""
import argparse
import json


def summarize(label, path, channel=1, extra=False):
    rep = json.load(open(path))
    tune = rep.get("tune", {})
    t = tune.get(str(channel)) or tune.get(channel) or {}
    rscale = rep.get("rscale", {})
    err = rep.get("error_model", {})
    print(f"--- {label} ({path}) ---")
    print(f"  ch{channel}: k={t.get('k')}  f={t.get('f')}  wing_frac={t.get('wing_frac')}  "
          f"wing_ok={t.get('wing_ok')}  bias_over_sigma={t.get('bias_over_sigma')}  "
          f"sigma_native={t.get('sigma_native')}  n_slices={t.get('n')}")
    print(f"  aperture: RSCALE={rscale.get('chosen')}  grid={rscale.get('grid')}")
    print(f"  error_model: R_white_median={err.get('R_white_median')}  "
          f"R_sys_median={err.get('R_sys_median')}")

    if extra:
        pos = rep.get("position", {})
        print(f"  position: adopted={pos.get('adopted')}  n_knot_px={pos.get('n_knot_px')}  "
              f"consistent={pos.get('consistent')}")
        comp = rep.get("companion", {})
        print(f"  companion: verdict={comp.get('verdict')}  "
              f"lam_resolved_max_um={comp.get('lam_resolved_max_um')}")
        abi = rep.get("abi") or {}
        vs_fast = abi.get("vs_fast", {}) if isinstance(abi, dict) else {}
        ch_key = str(channel) if str(channel) in vs_fast else channel
        vf = vs_fast.get(ch_key) if isinstance(vs_fast, dict) else None
        print(f"  abi vs fast (ch{channel} mean_diff_over_flux): "
              f"{vf.get('mean_diff_over_flux') if isinstance(vf, dict) else vs_fast.get(ch_key)}")
        stitch = rep.get("stitch", [])
        for j in stitch:
            if j.get("join", "").startswith(("1A", "1B")):
                print(f"  internal join {j['join']} @ {j['centre']:.2f} um: "
                      f"step {j['step_pct']:+.1f}% +/- {j['err_pct']:.1f}%  "
                      f"({'featureless' if j['clean'] else 'curved'})")
    print()
    return dict(label=label, wing_frac=t.get("wing_frac"), wing_ok=t.get("wing_ok"),
                bias_over_sigma=t.get("bias_over_sigma"), rscale=rscale.get("chosen"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="append", required=True, metavar="LABEL=PATH",
                    help="e.g. --report Dec=refined/28122025/SN2024aecx_28122025_report.json (repeatable)")
    ap.add_argument("--channel", type=int, default=1, help="MRS channel to compare (default 1, borders NIRSpec)")
    ap.add_argument("--extra", action="store_true",
                    help="also print position, companion/deblend verdict, AstroBkgInterp-vs-fast agreement, "
                         "and the 1A|1B internal stitch step")
    a = ap.parse_args()
    rows = []
    for item in a.report:
        label, path = item.split("=", 1)
        rows.append(summarize(label, path, a.channel, a.extra))

    print("Summary (ch%d):" % a.channel)
    print(f"{'epoch':<6} {'RSCALE':>8} {'wing_frac':>10} {'wing_ok':>8} {'bias/sigma':>11}")
    for r in rows:
        print(f"{r['label']:<6} {str(r['rscale']):>8} {str(r['wing_frac']):>10} "
              f"{str(r['wing_ok']):>8} {str(r['bias_over_sigma']):>11}")