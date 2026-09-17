"""Discretization convergence check: halve the tick and the grid cell and see what moves.

Usage: python scripts/convergence.py [--reps 2000] [--out runs/convergence.json]
If P(win) or mean losses shift by more than their intervals between resolutions, the results at
the default resolution are partly discretization artifacts and should be reported as such.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.engine import simulate                      # noqa: E402
from sim.scenario import Scenario, ScenarioConfig    # noqa: E402
from sim.stats import mean_ci, wilson                # noqa: E402
from sim.terrain import demo_terrain                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default="runs/convergence.json")
    a = ap.parse_args()
    rows = []
    for cell in (25.0, 12.5):
        t = demo_terrain(cell=cell)
        sc = Scenario(t, ScenarioConfig())
        tab = sc.tables("SBF")
        for tick in (30.0, 15.0, 7.5, 3.75):
            r = simulate(tab, "SBF", True, True, "Late", a.reps, seed=[7], tick_s=tick, n_log=0)
            lo, hi = wilson(r.win.sum(), r.n)
            bl = mean_ci(r.blue_losses)
            rows.append(dict(cell_m=cell, tick_s=tick, p_win=float(r.win.mean()), p_win_lo=float(lo),
                             p_win_hi=float(hi), blue_losses=bl[0], blue_losses_lo=bl[1], blue_losses_hi=bl[2],
                             minutes=float(r.minutes.mean())))
            print(f"cell {cell:5.1f} m  tick {tick:5.2f} s  P(win) {r.win.mean():.3f} [{lo:.3f}, {hi:.3f}]  "
                  f"Blue losses {bl[0]:.2f} [{bl[1]:.2f}, {bl[2]:.2f}]  minutes {r.minutes.mean():.1f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
