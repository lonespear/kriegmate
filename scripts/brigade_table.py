"""Regenerate the brigade pairing table (docs technical reference, section 19.4) with intervals.

Runs a fixed set of attacker/defender pairings on the same synthetic ground and writes every
rate with its Wilson or t interval plus the replication count, so nothing in the table is
quoted without uncertainty.

Usage: python scripts/brigade_table.py [--reps 100] [--out runs/brigade_table.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.brigade import BrigadeSim, synthetic_brigade_terrain   # noqa: E402
from sim.manifest import code_version                           # noqa: E402
from sim.nations import CLASSES, NATIONS                        # noqa: E402
from sim.stats import mean_ci, wilson                           # noqa: E402

PAIRINGS = [
    ("US", 1, "RU", 1), ("US", 2, "RU", 1),
    ("RU", 1, "US", 1), ("RU", 2, "US", 1),
    ("CN", 1, "IR", 1), ("IR", 2, "CN", 1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--terrain-seed", type=int, default=1)
    ap.add_argument("--att-coa", default="two_up")
    ap.add_argument("--def-coa", default="forward")
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="indices into PAIRINGS to run (default: all); merges into an existing --out")
    ap.add_argument("--out", default="runs/brigade_table.json")
    a = ap.parse_args()

    t = synthetic_brigade_terrain(seed=a.terrain_seed)
    todo = PAIRINGS if a.only is None else [PAIRINGS[i] for i in a.only]
    rows = []
    if a.only is not None and os.path.exists(a.out):
        rows = [r for r in json.load(open(a.out))["rows"]
                if (r["attacker"], r["att_bdes"], r["defender"], r["def_bdes"]) not in todo]
    for att, nb, dfd, ndb in todo:
        sim = BrigadeSim(t, NATIONS[att], NATIONS[dfd], a.att_coa, a.def_coa,
                         att_bdes=nb, def_bdes=ndb)
        r = sim.run(a.reps, seed=a.seed, n_log=0)
        lo, hi = wilson(r.win.sum(), r.n)
        al, all_, ahi = mean_ci(r.att_loss_frac())
        dl, dlo, dhi = mean_ci(r.def_loss_frac())
        mm, mlo, mhi = mean_ci(r.minutes)
        row = dict(attacker=att, att_bdes=nb, defender=dfd, def_bdes=ndb,
                   att_coa=a.att_coa, def_coa=a.def_coa, reps=int(r.n),
                   p_seized=float(r.win.mean()), p_seized_lo=float(lo), p_seized_hi=float(hi),
                   att_loss_frac=al, att_loss_frac_lo=all_, att_loss_frac_hi=ahi,
                   def_loss_frac=dl, def_loss_frac_lo=dlo, def_loss_frac_hi=dhi,
                   minutes=mm, minutes_lo=mlo, minutes_hi=mhi,
                   censored_frac=float((r.cause == 0).mean()),
                   att_platforms=int(r.att_n0.sum()), def_platforms=int(r.def_n0.sum()),
                   att_losses_by_class={c: float(v) for c, v in zip(CLASSES, r.att_losses.mean(0))},
                   att_n0_by_class={c: int(v) for c, v in zip(CLASSES, r.att_n0)},
                   def_losses_by_class={c: float(v) for c, v in zip(CLASSES, r.def_losses.mean(0))},
                   def_n0_by_class={c: int(v) for c, v in zip(CLASSES, r.def_n0)})
        rows.append(row)
        print(f"{att} x{nb} -> {dfd} x{ndb}  n={r.n}  P(seized) {row['p_seized']:.3f} "
              f"[{lo:.3f}, {hi:.3f}]  att {al:.3f} [{all_:.3f}, {ahi:.3f}]  "
              f"def {dl:.3f} [{dlo:.3f}, {dhi:.3f}]  min {mm:.0f} [{mlo:.0f}, {mhi:.0f}]  "
              f"censored {row['censored_frac']:.2f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    key = {p: i for i, p in enumerate(PAIRINGS)}
    rows.sort(key=lambda r: key[(r["attacker"], r["att_bdes"], r["defender"], r["def_bdes"])])
    json.dump(dict(code_version=code_version(), args=vars(a), terrain="synthetic_brigade_terrain",
                   rows=rows), open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
