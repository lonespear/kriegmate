"""The calibration ladder: four rungs of increasing size, each with its own roster, scenario
geometry, free parameters, and output metrics.

Rung 1 (duel) calibrates the physical parameters: aim dispersion and kill probabilities.
Rung 2 (section) calibrates detection and target selection.
Rung 3 (platoon) calibrates suppression, fires, and breakpoints.
Rung 4 (company) leaves only the synergy mechanics free.
Parameters calibrated at a lower rung are frozen (used as priors) at the rungs above.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .engine import concat_results, simulate
from .scenario import Scenario, ScenarioConfig
from .units import R_AT, R_INF, RUNG_ROSTERS, SCOUT, TANK, copy_params, default_params

BANDS = ["lt500", "500_1000", "gt1000"]
METRICS = (["p_win", "blue_loss_frac", "red_loss_frac", "minutes"]
           + [f"{side}_hit_{b}" for side in ("blue", "red") for b in BANDS]
           + ["blue_kill_per_hit", "red_kill_per_hit"])


@dataclass(frozen=True)
class ParamSpec:
    """One calibratable scalar: where it lives in the parameter dict, its prior range, and
    whether the range is on a log scale."""
    name: str
    lo: float
    hi: float
    log: bool = False

    def unit_to_value(self, u):
        return float(np.exp(np.log(self.lo) + u * (np.log(self.hi) - np.log(self.lo)))) if self.log \
            else float(self.lo + u * (self.hi - self.lo))

    def value_to_unit(self, v):
        return float((np.log(v) - np.log(self.lo)) / (np.log(self.hi) - np.log(self.lo))) if self.log \
            else float((v - self.lo) / (self.hi - self.lo))


# Setters map a spec name to a function that writes the value into the parameter dict.
def _set_pk(s, t):
    return lambda P, v: P["PK"].__setitem__((s, t), v)


def _set_vec(key, idx):
    return lambda P, v: P[key].__setitem__(idx, v)


def _set_scalar(key):
    return lambda P, v: P.__setitem__(key, v)


def _set_bp(key):
    """Breakpoint mean m with fixed concentration 20: Beta(20m, 20(1-m))."""
    return lambda P, v: P.__setitem__(key, (20 * v, 20 * (1 - v)))


SETTERS = {
    "theta0_tank": _set_vec("theta0", TANK), "theta0_at": _set_vec("theta0", R_AT),
    "pk_tank_at": _set_pk(TANK, R_AT), "pk_at_tank": _set_pk(R_AT, TANK),
    "pk_inf_rinf": _set_pk(2, R_INF), "pk_rinf_inf": _set_pk(R_INF, 2),
    "lam_blue": _set_scalar("lam_blue"), "lam_red": _set_scalar("lam_red"),
    "k_det": _set_scalar("k_det"), "b_threat": _set_scalar("b_threat"),
    "kappa_supp": _set_scalar("kappa_supp"), "fires_supp": _set_scalar("fires_supp"),
    "supp_scale": lambda P, v: P.__setitem__("p_supp", default_params()["p_supp"] * v),
    "bp_blue_mean": _set_bp("bp_blue"), "bp_red_mean": _set_bp("bp_red"),
    "p_mission_recon": _set_scalar("p_mission_recon"), "phi_fire": _set_scalar("phi_fire"),
}

FREE = {
    1: [ParamSpec("theta0_tank", 1.5e-4, 8e-4, True), ParamSpec("theta0_at", 2e-4, 1.5e-3, True),
        ParamSpec("pk_tank_at", 0.4, 0.9), ParamSpec("pk_at_tank", 0.3, 0.85)],
    2: [ParamSpec("lam_blue", 1.0, 8.0, True), ParamSpec("lam_red", 1.0, 8.0, True),
        ParamSpec("k_det", 0.4, 2.5, True), ParamSpec("b_threat", 0.0, 2.5),
        ParamSpec("pk_inf_rinf", 0.3, 0.7), ParamSpec("pk_rinf_inf", 0.3, 0.7)],
    3: [ParamSpec("kappa_supp", 1.0, 6.0), ParamSpec("supp_scale", 0.5, 1.5, True),
        ParamSpec("fires_supp", 0.5, 0.95), ParamSpec("bp_blue_mean", 0.3, 0.7),
        ParamSpec("bp_red_mean", 0.35, 0.75)],
    4: [ParamSpec("p_mission_recon", 0.2, 0.8), ParamSpec("phi_fire", 1.0, 10.0, True)],
}


@dataclass(frozen=True)
class Vignette:
    rung: int
    armor: str = "Maneuver"
    recon: bool = False
    fires: bool = False
    red_coa: str = "Early"
    cfg: ScenarioConfig = field(default_factory=ScenarioConfig)
    red_trigger: float = 600.0

    @property
    def roster(self):
        return RUNG_ROSTERS[self.rung]


def rung_config(rung, half=1500.0):
    """Scenario geometry sized to the rung and the available ground (start line inside the area)."""
    start = min({1: 900.0, 2: 1100.0, 3: 1300.0, 4: 1500.0}[rung], half - 100)
    return ScenarioConfig(start_dist=start, sbf_min=min(600.0, start - 300), sbf_max=min(1000.0, start - 50),
                          op_min=min(700.0, start - 250), op_max=min(1200.0, start - 50),
                          dismount_dist=350.0 if rung < 4 else 450.0,
                          red_radius=[0, 60, 120, 200, 300][rung], danger_close=250.0,
                          max_minutes=30.0 if rung < 4 else 60.0)


def rung_vignettes(rung, half=1500.0):
    """The standard vignette set at each rung: what a calibration wave runs on every terrain."""
    cfg = rung_config(rung, half)
    if rung == 1:
        return [Vignette(1, "Maneuver", False, False, "Early", cfg)]
    if rung == 2:
        return [Vignette(2, "Maneuver", False, False, r, cfg) for r in ("Early", "Late")]
    if rung == 3:
        return [Vignette(3, a, True, f, "Early", cfg) for a in ("SBF", "Maneuver") for f in (False, True)]
    return [Vignette(4, a, rc, fi, "Late", cfg)
            for a in ("SBF", "Maneuver") for rc in (False, True) for fi in (False, True)]


def apply_theta(P, specs, theta):
    """Return a copy of P with the spec values written in (theta in natural units)."""
    P = copy_params(P)
    for spec, v in zip(specs, theta):
        SETTERS[spec.name](P, float(v))
    return P


def _ratio(num, den):
    """Ratio of means and its delta-method variance; NaN when the denominator never occurs."""
    if den.sum() < 5:
        return float("nan"), float("nan")
    from .stats import ratio_ci
    r, lo, hi = ratio_ci(num, den)
    return float(r), float(max(((hi - lo) / 3.92) ** 2, 1e-8))


def _all_metrics(res):
    nb = res.blue_strength[:, 0]
    n = res.n
    means = dict(p_win=(float(res.win.mean()), float(res.win.var(ddof=1) / n)),
                 blue_loss_frac=(float((res.blue_losses / np.maximum(nb, 1)).mean()),
                                 float((res.blue_losses / np.maximum(nb, 1)).var(ddof=1) / n)),
                 red_loss_frac=(float((res.red_losses / res.red_strength[:, 0]).mean()),
                                float((res.red_losses / res.red_strength[:, 0]).var(ddof=1) / n)),
                 minutes=(float(res.minutes.mean()), float(res.minutes.var(ddof=1) / n)))
    for q, b in enumerate(BANDS):
        means[f"blue_hit_{b}"] = _ratio(res.blue_hits[:, q], res.blue_shots[:, q])
        means[f"red_hit_{b}"] = _ratio(res.red_hits[:, q], res.red_shots[:, q])
    means["blue_kill_per_hit"] = _ratio(res.red_losses, res.blue_hits.sum(1))
    means["red_kill_per_hit"] = _ratio(res.blue_losses, res.red_hits.sum(1))
    return means


def metrics_of(res):
    """Point estimate of every metric (NaN where the vignette produced no such shots)."""
    return {k: v[0] for k, v in _all_metrics(res).items()}


def metric_vars(res):
    """Monte Carlo variance of each metric estimate."""
    return {k: v[1] for k, v in _all_metrics(res).items()}


def run_vignette(terrain, vig, P, n, seed, n_plans=3, temperature=0.5, tick_s=None):
    """Run a vignette on one terrain over n_plans sampled Red plans (n replications total)."""
    from .engine import TICK_DEFAULT
    tick_s = TICK_DEFAULT if tick_s is None else tick_s
    per = max(20, n // n_plans)
    out = []
    for p in range(n_plans):
        sc = Scenario(terrain, vig.cfg, vig.roster, seed=p, placement_temperature=temperature)
        out.append(simulate(sc.tables(vig.armor), vig.armor, vig.recon and vig.roster.is_scout.any(),
                            vig.fires, vig.red_coa, per, seed=[seed, p], P=P, n_log=0, tick_s=tick_s,
                            red_trigger=vig.red_trigger))
    return concat_results(out)
