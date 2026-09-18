"""Space-filling design, GP metamodel, and Sobol indices for one rung's free parameters.

Usage: python scripts/doe_sobol.py --rung 2 --design 60 --reps 600
Prints, per output metric, the metamodel's leave-one-out R^2 and noise share, then total-effect
Sobol indices. Indices are only meaningful when LOO R^2 is high; if it is not, raise --reps
(less Monte Carlo noise) or --design (more points).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.manifest import code_version                 # noqa: E402
from sim.doe import design, fit_metamodel, loo_r2, run_design, sobol_indices   # noqa: E402
from sim.terrain import synthetic_terrain                                       # noqa: E402
from sim.units import default_params                                            # noqa: E402
from sim.vignettes import FREE, rung_vignettes                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, default=2)
    ap.add_argument("--design", type=int, default=60)
    ap.add_argument("--reps", type=int, default=600)
    ap.add_argument("--out", default="runs/sobol.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    specs = FREE[a.rung]
    t = synthetic_terrain(half=1500.0, seed=a.seed)
    vig = rung_vignettes(a.rung, 1500.0)[0]
    U, X = design(specs, a.design, seed=a.seed)
    Y, V, names = run_design(specs, X, t, vig, default_params(), n_reps=a.reps, seed=a.seed, log=print)
    report = {}
    for k, name in enumerate(names):
        mm = fit_metamodel(U, Y[:, k], V[:, k], seed=a.seed)
        r2 = loo_r2(U, Y[:, k], V[:, k], seed=a.seed)
        si = sobol_indices(mm, len(specs), N=2048, seed=a.seed)
        report[name] = dict(loo_r2=r2, noise_share=mm.noise_share,
                            ST={s.name: float(v) for s, v in zip(specs, si["ST"])},
                            S={s.name: float(v) for s, v in zip(specs, si["S"])},
                            ST_se={s.name: float(v) for s, v in zip(specs, si["ST_se"])})
        order = np.argsort(-si["ST"])
        print(f"\n{name}: LOO R2 = {r2:.2f}, MC noise share = {mm.noise_share:.2f}"
              + ("" if r2 > 0.7 else "  (metamodel too noisy: indices are indicative only)"))
        for j in order:
            print(f"   {specs[j].name:16s} ST = {si['ST'][j]:.2f} ± {si['ST_se'][j]:.2f}   S = {si['S'][j]:.2f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(code_version=code_version(), args=vars(a), outputs=report),
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
