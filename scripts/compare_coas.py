"""Compare several Blue COAs with a statistical guarantee: Kim-Nelson ranking and selection.

Usage: python scripts/compare_coas.py --delta 0.03 --alpha 0.05
Finds the COA with the highest P(win) against a fixed Red COA, sampling until the best is
separated from the rest by the indifference zone delta with confidence 1 - alpha. Runs the
Red-plan outer loop (several sampled defenses) so the selection is not conditional on one plan.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.engine import simulate                          # noqa: E402
from sim.scenario import Scenario, ScenarioConfig        # noqa: E402
from sim.stats import kn_select, wilson                  # noqa: E402
from sim.terrain import demo_terrain                     # noqa: E402

COAS = [(a, r, f) for a in ("SBF", "Maneuver") for r in (False, True) for f in (False, True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.03)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n0", type=int, default=100)
    ap.add_argument("--plans", type=int, default=3)
    ap.add_argument("--red", default="Late")
    ap.add_argument("--max-n", type=int, default=6000)
    a = ap.parse_args()
    t = demo_terrain()
    scens = [Scenario(t, ScenarioConfig(), seed=p, placement_temperature=0.5) for p in range(a.plans)]
    tabs = {armor: [s.tables(armor) for s in scens] for armor in ("SBF", "Maneuver")}
    counters = [0] * len(COAS)

    def sampler(i):
        armor, rc, fi = COAS[i]

        def draw(n):
            counters[i] += 1
            per = max(1, n // a.plans)
            out = [simulate(tabs[armor][p], armor, rc, fi, a.red, per, seed=[i, p, counters[i]], n_log=0).win
                   for p in range(a.plans)]
            return np.concatenate(out)[:n] if n >= a.plans else np.concatenate(out)[:n]
        return draw

    res = kn_select([sampler(i) for i in range(len(COAS))], delta=a.delta, alpha=a.alpha, n0=a.n0, max_n=a.max_n)
    print(f"KN selection (delta={a.delta}, alpha={a.alpha}, terminated={res['terminated']}):")
    for i, (armor, rc, fi) in enumerate(COAS):
        x = res["samples"][i]
        lo, hi = wilson(x.sum(), len(x))
        tag = "  <- selected" if i == res["best"] else (f"  eliminated at n={res['rounds'][i]}" if res["rounds"][i] else "")
        print(f"  {armor:9s} R{int(rc)}F{int(fi)}  n={len(x):5d}  P(win)={x.mean():.3f} [{lo:.3f}, {hi:.3f}]{tag}")


if __name__ == "__main__":
    main()
