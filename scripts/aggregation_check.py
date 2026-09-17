"""Aggregation check: run the same company fight in the entity engine and in the aggregate
(brigade) engine on the same ground with the same nation catalog, and report the gap.

The aggregate engine is a mean-field version of the entity engine, so the two should agree on
the company fight up to the aggregation error. That error is what a rung-5 calibration would have
to absorb (or what a correction factor on the aggregate kill rates should remove). Report the gap
honestly; do not tune it away by hand.

Usage: python scripts/aggregation_check.py --attacker US --defender RU --reps 300
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.brigade import CLASS_INDEX, NC, BrigadeSim, Force, brigade_terrain_from     # noqa: E402
from sim.engine import simulate                                                     # noqa: E402
from sim.nations import CLASSES, NATIONS, entity_params                             # noqa: E402
from sim.scenario import Scenario, ScenarioConfig                                   # noqa: E402
from sim.stats import mean_ci, wilson                                               # noqa: E402
from sim.terrain import synthetic_terrain                                           # noqa: E402
from sim.units import COMPANY                                                       # noqa: E402


def company_forces(att, dfn):
    """The entity engine's company team (4 tanks, 6 IFV/inf, 2 scouts) vs its AT-heavy platoon
    (4 AT teams, 6 inf teams), expressed as aggregate companies."""
    def P_of(nation):
        eq = nation.equipment
        return {key: np.array([getattr(eq[c], key) for c in CLASSES], float)
                for key in ("area", "theta0", "theta_s", "theta_t", "sensor", "signature", "fire_rate", "p_supp",
                            "supp_vuln", "value", "speed", "ind_lethality", "rounds_per_min", "rmax")}
    # one aggregate entity per side, so breakpoint semantics match the entity engine (the team breaks
    # on the fraction of its fighting units lost; scouts do not count)
    a = np.zeros((1, NC))
    a[0, CLASS_INDEX["tank"]], a[0, CLASS_INDEX["ifv"]], a[0, CLASS_INDEX["inf"]], a[0, CLASS_INDEX["recon"]] = 4, 6, 6, 2
    A = Force(att, 0, ["TEAM"], np.array([0]), np.array([0]), a, P_of(att))
    d = np.zeros((1, NC)); d[0, CLASS_INDEX["atgm"]] = 4; d[0, CLASS_INDEX["inf"]] = 6
    D = Force(dfn, 1, ["AT"], np.array([0]), np.array([0]), d, P_of(dfn))
    return A, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacker", default="US", choices=list(NATIONS))
    ap.add_argument("--defender", default="RU", choices=list(NATIONS))
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw", action="store_true", help="ignore data/aggregation_correction.json (kill_scale = 1, 1)")
    a = ap.parse_args()
    att, dfn = NATIONS[a.attacker], NATIONS[a.defender]
    t = synthetic_terrain(half=2000.0, cell=25.0, seed=4)
    cfg = ScenarioConfig()

    # entity engine, Maneuver + recon, no fires, Early Red
    P = entity_params(att, dfn)
    sc = Scenario(t, cfg, COMPANY, seed=0)
    e = simulate(sc.tables("Maneuver"), "Maneuver", True, False, "Early", a.reps, seed=[a.seed], P=P, n_log=0)

    # aggregate engine on the same ground: one tank company, one mech company, a scout section vs one AT platoon
    bt = brigade_terrain_from(t)
    A, D = company_forces(att, dfn)
    start = -cfg.start_dist
    plans = {0: dict(pos=np.array([[start, 0.0]]), wps=[[np.array([0.0, 0.0])]]),
             1: dict(pos=np.array([[float(sc.rpos[:, 0].mean()), float(sc.rpos[:, 1].mean())]]), wps=[[]])}
    sim = BrigadeSim(bt, att, dfn, forces=(A, D), plans=plans, objective=(0.0, 0.0), objective_companies=1,
                     objective_radius=300.0, max_minutes=cfg.max_minutes, kill_scale=(1.0, 1.0) if a.raw else None)
    g = sim.run(a.reps, seed=a.seed, n_log=0)
    print(f"aggregate kill_scale in use: attacker {sim.kill_scale[0]:.3f}, defender {sim.kill_scale[1]:.3f}")

    def row(name, ent, agg, prop=False):
        if prop:
            lo1, hi1 = wilson(ent.sum(), len(ent)); lo2, hi2 = wilson(agg.sum(), len(agg))
            print(f"  {name:28s} entity {ent.mean():.3f} [{lo1:.3f}, {hi1:.3f}]   aggregate {agg.mean():.3f} [{lo2:.3f}, {hi2:.3f}]   gap {agg.mean() - ent.mean():+.3f}")
        else:
            m1, lo1, hi1 = mean_ci(ent); m2, lo2, hi2 = mean_ci(agg)
            print(f"  {name:28s} entity {m1:.3f} [{lo1:.3f}, {hi1:.3f}]   aggregate {m2:.3f} [{lo2:.3f}, {hi2:.3f}]   gap {m2 - m1:+.3f}")

    print(f"Aggregation check: {att.name} company team attacks a {dfn.name} AT platoon, {a.reps} replications each")
    row("P(win)", e.win, g.win, prop=True)
    row("attacker loss fraction", e.blue_losses / 10.0, g.att_losses[:, :4].sum(1) / 16.0)
    row("defender loss fraction", e.red_losses / 10.0, g.def_loss_frac())
    row("minutes to decision", e.minutes, g.minutes)
    print("  Gaps outside both intervals are aggregation error: the mean-field fire allocation and the\n"
          "  company-level breakpoints do not reproduce the entity engine, and a rung-5 calibration\n"
          "  (or a correction factor on aggregate kill rates) has to absorb them.")


if __name__ == "__main__":
    main()
