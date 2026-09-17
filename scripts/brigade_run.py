"""Run one brigade-vs-brigade pairing and report intervals.

Usage: python scripts/brigade_run.py --attacker US --defender RU --att-coa two_up --def-coa forward
       --att-bdes 2 --reps 200
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.brigade import ATTACK_COAS, DEFEND_COAS, BrigadeSim, synthetic_brigade_terrain   # noqa: E402
from sim.nations import CLASSES, NATIONS                                                  # noqa: E402
from sim.stats import aalen_johansen, mean_ci, wilson                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacker", default="US", choices=list(NATIONS))
    ap.add_argument("--defender", default="RU", choices=list(NATIONS))
    ap.add_argument("--att-coa", default="two_up", choices=ATTACK_COAS)
    ap.add_argument("--def-coa", default="forward", choices=DEFEND_COAS)
    ap.add_argument("--att-bdes", type=int, default=1)
    ap.add_argument("--def-bdes", type=int, default=1)
    ap.add_argument("--att-strength", type=float, default=1.0)
    ap.add_argument("--def-strength", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--terrain-seed", type=int, default=1)
    ap.add_argument("--relief", type=float, default=120.0)
    ap.add_argument("--cover", type=float, default=0.2)
    a = ap.parse_args()
    t = synthetic_brigade_terrain(seed=a.terrain_seed, relief=a.relief, cover=a.cover)
    sim = BrigadeSim(t, NATIONS[a.attacker], NATIONS[a.defender], a.att_coa, a.def_coa,
                     att_bdes=a.att_bdes, def_bdes=a.def_bdes, att_strength=a.att_strength, def_strength=a.def_strength)
    r = sim.run(a.reps, seed=a.seed, n_log=0)
    lo, hi = wilson(r.win.sum(), r.n)
    print(f"{NATIONS[a.attacker].name} x{a.att_bdes} ({a.att_coa}) attacks {NATIONS[a.defender].name} x{a.def_bdes} ({a.def_coa})")
    print(f"  attacker platforms {int(r.att_n0.sum())}, defender {int(r.def_n0.sum())}, {a.reps} replications")
    print(f"  P(objective seized)  {r.win.mean():.3f}  [{lo:.3f}, {hi:.3f}]")
    m, l, h = mean_ci(r.att_loss_frac()); print(f"  attacker maneuver losses  {m:.3f} [{l:.3f}, {h:.3f}]")
    m, l, h = mean_ci(r.def_loss_frac()); print(f"  defender maneuver losses  {m:.3f} [{l:.3f}, {h:.3f}]")
    m, l, h = mean_ci(r.minutes); print(f"  minutes to decision       {m:.0f} [{l:.0f}, {h:.0f}]  (censored: {(r.cause == 0).mean():.2f})")
    print("  attacker losses by class: " + ", ".join(f"{c} {v:.1f}/{int(n0)}" for c, v, n0 in zip(CLASSES, r.att_losses.mean(0), r.att_n0)))
    print("  defender losses by class: " + ", ".join(f"{c} {v:.1f}/{int(n0)}" for c, v, n0 in zip(CLASSES, r.def_losses.mean(0), r.def_n0)))
    ts, cif = aalen_johansen(r.minutes, r.cause, 2)
    if len(ts):
        print(f"  by the end: attacker culminated {cif[-1, 0]:.2f}, objective seized {cif[-1, 1]:.2f} (Aalen-Johansen)")


if __name__ == "__main__":
    main()
