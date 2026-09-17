"""Climb the calibration ladder on a sampled terrain library and write a report.

By default the reference is synthetic: the engine itself at hidden "true" parameters. That
exercises the whole pipeline offline and lets you check that the method recovers a known truth.
Replace it with a TableReference (CSV of observed outcomes) when real reference data exists.

Usage:
  python scripts/calibrate_ladder.py --rungs 1 2 --terrains 6 --reps 300 --design 40 --waves 2
  python scripts/calibrate_ladder.py --reference data/reference.csv --rungs 1 2 3
Outputs go to runs/<timestamp>/ : state.json (calibrated P and NROY ranges), stability.json,
coverage.json, error_trend.json, manifest.json, and the terrain library index.
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.calibrate import (SyntheticReference, TableReference, coverage_report, error_trend,  # noqa: E402
                           run_ladder, save_state, stability_report)
from sim.descriptors import TerrainLibrary, select_terrains, synthetic_design               # noqa: E402
from sim.manifest import make_manifest, save_manifest                                          # noqa: E402
from sim.units import default_params, RUNG_ROSTERS                                            # noqa: E402
from sim.vignettes import FREE, apply_theta, rung_vignettes                                   # noqa: E402

TRUE_THETA = {1: [2.5e-4, 5e-4, 0.7, 0.5], 2: [3.0, 2.5, 1.2, 1.0, 0.5, 0.45],
              3: [3.0, 1.0, 0.85, 0.5, 0.55], 4: [0.5, 5.0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--terrains", type=int, default=6, help="candidate synthetic terrains to sample")
    ap.add_argument("--keep", type=int, default=5, help="terrains selected from the candidates")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--design", type=int, default=40, help="parameter points per wave")
    ap.add_argument("--waves", type=int, default=2)
    ap.add_argument("--half", type=float, default=1500.0)
    ap.add_argument("--reference", default=None, help="CSV of reference outcomes; default synthetic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or os.path.join("runs", dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    log_f = open(os.path.join(out, "log.txt"), "w")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    # 1. terrain library sampled across the descriptor space, split train / holdout
    lib = TerrainLibrary("cache/terrains")
    specs = [dict(kind="synthetic", seed=i, half=a.half, cell=25.0, **d)
             for i, d in enumerate(synthetic_design(a.terrains, seed=a.seed))]
    terrains, descs = {}, {}
    for i, spec in enumerate(specs):
        t, d = lib.get(spec)
        terrains[f"t{i}"], descs[f"t{i}"] = t, d
        log(f"terrain t{i}: {t.name}  openness500={d.openness_500:.2f} relief={d.relief:.0f}")
    keys = list(terrains)
    train_i, hold_i = select_terrains([descs[k].vector() for k in keys], a.keep, 0.2, a.seed)
    train = {keys[i]: terrains[keys[i]] for i in train_i}
    hold = {keys[i]: terrains[keys[i]] for i in hold_i}
    log(f"training terrains {list(train)}, held out {list(hold)}")

    # 2. reference
    if a.reference:
        import pandas as pd
        reference = TableReference(pd.read_csv(a.reference))
        truth = None
    else:
        P_true = default_params()
        for r in a.rungs:
            P_true = apply_theta(P_true, FREE[r], TRUE_THETA[r])
        reference = SyntheticReference(P_true, n=max(900, 3 * a.reps), seed=999)
        truth = {s.name: v for r in a.rungs for s, v in zip(FREE[r], TRUE_THETA[r])}
        log("synthetic reference with hidden truth: " + json.dumps(truth))

    # 3. climb
    state = run_ladder(a.rungs, train, reference, specs, n_reps=a.reps, m=a.design, waves=a.waves,
                       seed=a.seed, log=log, half=a.half)
    save_state(state, os.path.join(out, "state.json"))

    # 4. diagnostics
    stab = stability_report(state.waves)
    json.dump(stab, open(os.path.join(out, "stability.json"), "w"), indent=1, default=float)
    if truth:
        log("NROY ranges vs truth:")
        for name, (lo, hi) in state.fixed.items():
            inside = lo <= truth[name] <= hi
            log(f"  {name:16s} [{lo:.4g}, {hi:.4g}]  truth {truth[name]:.4g}  {'inside' if inside else 'OUTSIDE'}")
    if hold:
        rows, cov = coverage_report(state, {r: FREE[r] for r in a.rungs}, hold,
                                    {r: rung_vignettes(r, a.half) for r in a.rungs}, reference,
                                    n_reps=a.reps, k_theta=10, seed=a.seed, log=log)
        json.dump(rows, open(os.path.join(out, "coverage.json"), "w"), indent=1, default=float)
        trend = error_trend(rows, descs)
        json.dump(trend, open(os.path.join(out, "error_trend.json"), "w"), indent=1, default=float)
        flagged = [t for t in trend if t["flag"]]
        log(f"error-trend flags: {len(flagged)} of {len(trend)}" + (
            " -> " + ", ".join(f"{t['metric']}~{t['descriptor']}" for t in flagged) if flagged else ""))
    t0 = next(iter(train.values()))
    save_manifest(make_manifest(state.P, a.seed, t0, RUNG_ROSTERS[a.rungs[-1]],
                                extra=dict(rungs=a.rungs, reps=a.reps, design=a.design, waves=a.waves,
                                           terrains=lib.entries())),
                  os.path.join(out, "manifest.json"))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
