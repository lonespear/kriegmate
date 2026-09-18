"""The README's front-page worked result: one paired COA comparison, end to end.

Runs Blue COA A (support-by-fire) against Blue COA B (mounted maneuver) on the offline demo
ground with common random numbers, and reports P(win) with Wilson intervals, the paired
difference with its interval, and the CRN variance-reduction diagnostic. Writes the numbers and
a run manifest to JSON so the README can cite a committed artifact.

Usage: python scripts/headline.py [--reps 2000] [--out runs/headline.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.engine import TICK_DEFAULT, simulate           # noqa: E402
from sim.manifest import make_manifest, save_manifest   # noqa: E402
from sim.scenario import Scenario, ScenarioConfig       # noqa: E402
from sim.stats import crn_gain, mean_ci, paired_ci, wilson   # noqa: E402
from sim.terrain import demo_terrain                    # noqa: E402
from sim.units import default_params                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tick", type=float, default=TICK_DEFAULT)
    ap.add_argument("--out", default="runs/headline.json")
    a = ap.parse_args()

    t = demo_terrain()
    cfg = ScenarioConfig()
    sc = Scenario(t, cfg)
    out = {}
    res = {}
    for armor in ("SBF", "Maneuver"):
        r = simulate(sc.tables(armor), armor, True, True, "Late", a.reps, seed=[a.seed],
                     tick_s=a.tick, n_log=0)
        res[armor] = r
        lo, hi = wilson(r.win.sum(), r.n)
        bl = mean_ci(r.blue_losses)
        mn = mean_ci(r.minutes)
        out[armor] = dict(reps=int(r.n), p_win=float(r.win.mean()), p_win_lo=float(lo),
                          p_win_hi=float(hi), blue_losses=bl[0], blue_losses_lo=bl[1],
                          blue_losses_hi=bl[2], minutes=mn[0], minutes_lo=mn[1], minutes_hi=mn[2])
        print(f"{armor:9s} n={r.n}  P(win) {r.win.mean():.3f} [{lo:.3f}, {hi:.3f}]  "
              f"Blue losses {bl[0]:.2f} [{bl[1]:.2f}, {bl[2]:.2f}]  minutes {mn[0]:.1f}")

    d, dlo, dhi = paired_ci(res["Maneuver"].win, res["SBF"].win)
    rho, vratio = crn_gain(res["SBF"].win, res["Maneuver"].win)
    out["paired"] = dict(metric="P(win) SBF - Maneuver", diff=float(d), diff_lo=float(dlo),
                         diff_hi=float(dhi), crn_rho=rho, crn_var_ratio=vratio, reps=a.reps,
                         tick_s=a.tick)
    print(f"paired  P(win) SBF - Maneuver = {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")
    print(f"CRN     rho = {rho:.3f}, paired/independent variance = {vratio:.3f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    m = make_manifest(default_params(), [a.seed], t, sc.tables("SBF")["roster"], scenario_cfg=cfg,
                      extra=dict(script="scripts/headline.py", args=vars(a),
                                 coas=["SBF", "Maneuver"], recon=True, fires=True, red_coa="Late"))
    mpath = os.path.splitext(a.out)[0] + "_manifest.json"
    save_manifest(m, mpath)
    print(f"wrote {a.out} and {mpath}")


if __name__ == "__main__":
    main()
