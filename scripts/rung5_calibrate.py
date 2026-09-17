"""Rung 5: calibrate the aggregate engine against the entity engine on company fights.

Fits two multipliers on the aggregate engine's direct-fire kill rates (attacker->defender and
defender->attacker) so that the aggregate engine reproduces the entity engine's mean loss
fractions on the same company fight, averaged over several synthetic terrains and two nation
pairings. Alternating bisection on each multiplier; the other metrics (P(win), minutes) are then
reported as residual gaps, not fitted. Writes data/aggregation_correction.json, which
BrigadeSim loads by default.

Usage: python scripts/rung5_calibrate.py --terrains 3 --reps 300 --rounds 2
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.aggregation_check import company_forces                                  # noqa: E402
from sim.brigade import CORRECTION_PATH, BrigadeSim, brigade_terrain_from            # noqa: E402
from sim.engine import simulate                                                       # noqa: E402
from sim.nations import NATIONS, entity_params                                        # noqa: E402
from sim.scenario import Scenario, ScenarioConfig                                     # noqa: E402
from sim.terrain import synthetic_terrain                                             # noqa: E402
from sim.units import COMPANY                                                         # noqa: E402

PAIRS = [("US", "RU"), ("CN", "IR")]


def cases(n_terrains, reps, seed):
    """Entity-engine references and the matching aggregate set-ups."""
    out = []
    cfg = ScenarioConfig()
    for k in range(n_terrains):
        t = synthetic_terrain(half=2000.0, cell=25.0, seed=4 + k, building_density=[0.003, 0.001, 0.008][k % 3],
                              forest_share=[0.07, 0.03, 0.15][k % 3], relief=[30, 15, 60][k % 3])
        sc = Scenario(t, cfg, COMPANY, seed=0)
        tab = sc.tables("Maneuver")
        bt = brigade_terrain_from(t)
        for a, d in PAIRS:
            att, dfn = NATIONS[a], NATIONS[d]
            e = simulate(tab, "Maneuver", True, False, "Early", reps, seed=[seed, k], P=entity_params(att, dfn), n_log=0)
            A, D = company_forces(att, dfn)
            plans = {0: dict(pos=np.array([[-cfg.start_dist, 0.0]]), wps=[[np.array([0.0, 0.0])]]),
                     1: dict(pos=np.array([[float(sc.rpos[:, 0].mean()), float(sc.rpos[:, 1].mean())]]), wps=[[]])}
            out.append(dict(terrain=bt, att=att, dfn=dfn, plans=plans, cfg=cfg,
                            ref=dict(p_win=e.win.mean(), att_loss=(e.blue_losses / 10).mean(),
                                     def_loss=(e.red_losses / 10).mean(), minutes=e.minutes.mean())))
    return out


def aggregate(case, scale, reps, seed):
    sim = BrigadeSim(case["terrain"], case["att"], case["dfn"], forces=company_forces(case["att"], case["dfn"]),
                     plans=case["plans"], objective=(0.0, 0.0), objective_companies=1, objective_radius=300.0,
                     max_minutes=case["cfg"].max_minutes, kill_scale=scale)
    g = sim.run(reps, seed=seed, n_log=0)
    return dict(p_win=g.win.mean(), att_loss=(g.att_losses[:, :4].sum(1) / 16.0).mean(),
                def_loss=g.def_loss_frac().mean(), minutes=g.minutes.mean())


def gap(cs, scale, reps, seed, key):
    return float(np.mean([aggregate(c, scale, reps, seed)[key] - c["ref"][key] for c in cs]))


def bisect(cs, scale, idx, key, reps, seed, lo=0.3, hi=4.0, iters=7, log=print):
    """Find the multiplier at position idx that zeroes the mean gap in `key` (monotone in the scale)."""
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        sc = list(scale)
        sc[idx] = mid
        g = gap(cs, tuple(sc), reps, seed, key)
        log(f"    scale[{idx}] = {mid:.3f}: mean gap in {key} = {g:+.3f}")
        if g > 0:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrains", type=int, default=3)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=CORRECTION_PATH)
    a = ap.parse_args()
    cs = cases(a.terrains, a.reps, a.seed)
    print(f"{len(cs)} company-fight cases ({a.terrains} terrains x {len(PAIRS)} pairings), {a.reps} replications each")
    scale = (1.0, 1.0)
    print("raw gaps: " + ", ".join(f"{k} {gap(cs, scale, a.reps, a.seed, k):+.3f}" for k in ("p_win", "att_loss", "def_loss", "minutes")))
    for r in range(a.rounds):
        print(f"round {r + 1}: defender->attacker kill scale (matches attacker losses)")
        d = bisect(cs, scale, 1, "att_loss", a.reps, a.seed)
        scale = (scale[0], d)
        print(f"round {r + 1}: attacker->defender kill scale (matches defender losses)")
        att = bisect(cs, scale, 0, "def_loss", a.reps, a.seed)
        scale = (att, scale[1])
    res = {k: gap(cs, scale, a.reps, a.seed, k) for k in ("p_win", "att_loss", "def_loss", "minutes")}
    print(f"\nfitted kill_scale = (attacker {scale[0]:.3f}, defender {scale[1]:.3f})")
    print("residual gaps (aggregate - entity): " + ", ".join(f"{k} {v:+.3f}" for k, v in res.items()))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(kill_scale_att=scale[0], kill_scale_def=scale[1], residual_gaps=res, terrains=a.terrains,
                   reps=a.reps, pairs=PAIRS), open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
