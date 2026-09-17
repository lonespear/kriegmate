"""The 3x3 brigade COA game: attacker schemes vs defender schemes, paired by common random
numbers, solved as a zero-sum game on P(objective seized) with a bootstrap interval on the value.

Usage: python scripts/brigade_game.py --attacker US --defender RU --att-bdes 2 --reps 100
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.brigade import ATTACK_COAS, DEFEND_COAS, BrigadeSim, synthetic_brigade_terrain   # noqa: E402
from sim.nations import NATIONS                                                           # noqa: E402
from sim.stats import bootstrap_game_value, wilson                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacker", default="US", choices=list(NATIONS))
    ap.add_argument("--defender", default="RU", choices=list(NATIONS))
    ap.add_argument("--att-bdes", type=int, default=2)
    ap.add_argument("--def-bdes", type=int, default=1)
    ap.add_argument("--att-strength", type=float, default=1.0)
    ap.add_argument("--def-strength", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--terrain-seed", type=int, default=1)
    a = ap.parse_args()
    t = synthetic_brigade_terrain(seed=a.terrain_seed)
    cells = {}
    for i, ac in enumerate(ATTACK_COAS):
        for j, dc in enumerate(DEFEND_COAS):
            sim = BrigadeSim(t, NATIONS[a.attacker], NATIONS[a.defender], ac, dc, att_bdes=a.att_bdes,
                             def_bdes=a.def_bdes, att_strength=a.att_strength, def_strength=a.def_strength)
            cells[i, j] = sim.run(a.reps, seed=a.seed, n_log=0).win        # same seed: paired across cells
            lo, hi = wilson(cells[i, j].sum(), a.reps)
            print(f"  {ac:12s} vs {dc:8s}  P(win) {cells[i, j].mean():.3f} [{lo:.3f}, {hi:.3f}]")
    v, lo, hi, x, y, xb, yb, M = bootstrap_game_value(cells, B=300, seed=a.seed)
    print("\nPayoff matrix (rows attacker, cols defender), P(objective seized):")
    print("               " + "  ".join(f"{d:>8s}" for d in DEFEND_COAS))
    for i, ac in enumerate(ATTACK_COAS):
        print(f"  {ac:12s} " + "  ".join(f"{M[i, j]:8.3f}" for j in range(3)))
    print(f"\nGame value {v:.3f}  (bootstrap 95% [{lo:.3f}, {hi:.3f}])")
    print("Attacker optimal mix: " + ", ".join(f"{c} {p:.2f}" for c, p in zip(ATTACK_COAS, x)))
    print("Defender optimal mix: " + ", ".join(f"{c} {p:.2f}" for c, p in zip(DEFEND_COAS, y)))
    print("Bootstrap-mean mixes: attacker " + ", ".join(f"{p:.2f}" for p in xb) + "; defender " + ", ".join(f"{p:.2f}" for p in yb))


if __name__ == "__main__":
    main()
