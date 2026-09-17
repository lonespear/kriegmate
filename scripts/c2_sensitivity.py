"""C2 sensitivity study: how much do decision latency, initiative, latitude, net reliability and
synchronization move the brigade outcome, for one side of one pairing?

This is the discipline behind every C2 claim in the model: the nation profiles are notional, so
their effect is reported as Sobol indices over the plausible ranges (c2.C2_RANGES), not as a
result. Latin-hypercube design over the five C2 factors of the attacker (or defender), GP
metamodel with the Monte Carlo noise floor, Saltelli/Jansen indices on P(objective seized).

Usage: python scripts/c2_sensitivity.py --attacker US --defender RU --side attacker --design 24 --reps 24
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.brigade import BrigadeSim, synthetic_brigade_terrain           # noqa: E402
from sim.c2 import C2_RANGES, PROFILES                                    # noqa: E402
from sim.doe import design, fit_metamodel, loo_r2, sobol_indices          # noqa: E402
from sim.nations import NATIONS                                           # noqa: E402
from sim.vignettes import ParamSpec                                       # noqa: E402

FACTORS = ["tau", "p_init", "latitude", "net", "sync"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacker", default="US", choices=list(NATIONS))
    ap.add_argument("--defender", default="RU", choices=list(NATIONS))
    ap.add_argument("--side", default="attacker", choices=["attacker", "defender"])
    ap.add_argument("--att-bdes", type=int, default=2)
    ap.add_argument("--design", type=int, default=24)
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/c2_sensitivity.json")
    a = ap.parse_args()
    specs = [ParamSpec(f, *C2_RANGES[f], f == "tau") for f in FACTORS]
    U, X = design(specs, a.design, seed=a.seed)
    t = synthetic_brigade_terrain(seed=1)
    code = a.attacker if a.side == "attacker" else a.defender
    lineages = ["IR", "IRGC"] if code == "IR" else [code]
    ys, vs = [], []
    for i, x in enumerate(X):
        prof = {ln: PROFILES[ln].scaled(tau_mult=x[0] / PROFILES[ln].tau, p_init=x[1], latitude=x[2], net=x[3], sync=x[4])
                for ln in lineages}
        sim = BrigadeSim(t, NATIONS[a.attacker], NATIONS[a.defender], "two_up", "mobile", att_bdes=a.att_bdes,
                         c2_profiles=prof)
        r = sim.run(a.reps, seed=a.seed, n_log=0)
        ys.append(r.win.mean())
        vs.append(r.win.var(ddof=1) / a.reps)
        print(f"  point {i + 1}/{a.design}: tau {x[0]:5.1f} p_init {x[1]:.2f} lat {x[2]:.2f} net {x[3]:.2f} sync {x[4]:.2f}"
              f"  -> P(win) {ys[-1]:.3f}")
    ys, vs = np.array(ys), np.array(vs)
    mm = fit_metamodel(U, ys, vs, seed=a.seed)
    r2 = loo_r2(U, ys, vs, seed=a.seed)
    si = sobol_indices(mm, len(specs), N=2048, seed=a.seed)
    print(f"\n{a.side} C2 factors ({code}): LOO R2 {r2:.2f}, MC noise share {mm.noise_share:.2f}"
          + ("" if r2 > 0.7 else "   (metamodel too noisy: indices indicative only; raise --reps/--design)"))
    for j in np.argsort(-si["ST"]):
        print(f"   {FACTORS[j]:9s} ST = {si['ST'][j]:.2f} ± {si['ST_se'][j]:.2f}   S = {si['S'][j]:.2f}")
    print(f"P(win) range over the design: {ys.min():.2f} .. {ys.max():.2f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(side=a.side, code=code, loo_r2=r2, noise_share=mm.noise_share, y=ys.tolist(),
                   ST=dict(zip(FACTORS, si["ST"].tolist())), S=dict(zip(FACTORS, si["S"].tolist())),
                   X=X.tolist()), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
