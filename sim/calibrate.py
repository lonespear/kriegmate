"""History matching against reference outcomes, with stability and coverage diagnostics.

History matching (Craig, Goldstein, Seheult & Smith 1997; Vernon, Goldstein & Bower 2010) rules
out the parameter settings that cannot reproduce the reference outcomes, rather than searching
for a single best fit. For each candidate theta and each reference output z, the implausibility
is

    I(theta) = |E[f(theta)] - z| / sqrt(Var_mc(f) + Var_ref(z) + Var_disc)

where Var_mc is the simulator's Monte Carlo variance, Var_ref the reference's uncertainty, and
Var_disc the model-discrepancy variance you are willing to tolerate. Theta is "not ruled out yet"
(NROY) when the largest I over all outputs is below a threshold (3 by convention, from Pukelsheim's
3-sigma rule). Waves re-sample inside the NROY box until it stops shrinking.

Two reference sources are provided:
- SyntheticReference: the same engine run with hidden "true" parameters at high replication count.
  A stand-in for a higher-fidelity model; it lets the whole pipeline be exercised and tested offline.
- TableReference: a CSV/DataFrame of observed metrics with variances, keyed by terrain and vignette,
  for instrumented-training or higher-fidelity-model data.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import stats as sps
from scipy.stats import qmc

from .descriptors import DESCRIPTOR_NAMES
from .units import default_params
from .vignettes import METRICS, apply_theta, metric_vars, metrics_of, run_vignette


# ---------------------------------------------------------------- reference sources
class SyntheticReference:
    """Reference outcomes from the engine itself at hidden parameter values."""

    def __init__(self, P_true, n=1500, seed=999, n_plans=3):
        self.P, self.n, self.seed, self.n_plans = P_true, n, seed, n_plans
        self._cache = {}

    def outcomes(self, terrain_key, terrain, vig):
        key = (terrain_key, vig)
        if key not in self._cache:
            res = run_vignette(terrain, vig, self.P, self.n, seed=self.seed, n_plans=self.n_plans)
            self._cache[key] = (metrics_of(res), metric_vars(res))
        return self._cache[key]


class TableReference:
    """Reference outcomes from a table with columns
    terrain_key, rung, armor, recon, fires, red_coa, <metric>, <metric>_var for each metric."""

    def __init__(self, df):
        self.df = df

    def outcomes(self, terrain_key, terrain, vig):
        d = self.df
        row = d[(d.terrain_key == terrain_key) & (d.rung == vig.rung) & (d.armor == vig.armor)
                & (d.recon == vig.recon) & (d.fires == vig.fires) & (d.red_coa == vig.red_coa)]
        if row.empty:
            return None
        row = row.iloc[0]
        return ({m: float(row[m]) for m in METRICS if m in row},
                {m: float(row[f"{m}_var"]) for m in METRICS if f"{m}_var" in row})


# ---------------------------------------------------------------- history matching
@dataclass
class Wave:
    rung: int
    unit_box: np.ndarray          # (d, 2) box in unit coordinates sampled this wave
    thetas: np.ndarray            # (m, d) natural units
    implaus: np.ndarray           # (m,) max implausibility
    per_output: dict              # (terrain, vignette, metric) -> implausibility vector (m,)
    nroy: np.ndarray              # bool (m,)
    nroy_box: np.ndarray          # (d, 2) natural units, hull of NROY points
    n_reps: int


@dataclass
class LadderState:
    """Calibrated parameters carried up the ladder, and the waves that produced them."""
    P: dict
    waves: list = field(default_factory=list)
    fixed: dict = field(default_factory=dict)      # spec name -> (lo, hi) NROY range after its rung


def _box_sample(box, m, seed):
    d = box.shape[0]
    u = qmc.LatinHypercube(d=d, seed=seed, optimization="random-cd" if m <= 300 else None).random(m)
    return box[:, 0] + u * (box[:, 1] - box[:, 0])


def history_match(rung, specs, P_base, terrains, vignettes, reference, n_reps=300, m=60,
                  waves=3, threshold=3.0, disc_frac=0.05, seed=0, log=print, shrink_floor=0.05,
                  n_plans=3):
    """Run `waves` waves of history matching at one rung.

    terrains: dict terrain_key -> Terrain (training set). vignettes: list of Vignette at this rung.
    reference: object with .outcomes(terrain_key, terrain, vig) -> (metrics, variances) or None.
    disc_frac: model-discrepancy SD as a fraction of the reference value (plus 0.02 absolute for
    proportions), the tolerance for the model being wrong even at the best theta.
    Returns list of Wave.
    """
    d = len(specs)
    box = np.array([[0.0, 1.0]] * d)
    out = []
    refs = {}
    for tk, t in terrains.items():
        for vig in vignettes:
            r = reference.outcomes(tk, t, vig)
            if r is not None:
                refs[tk, vig] = r
    if not refs:
        raise ValueError("No reference outcomes match the vignettes at this rung.")
    for w in range(waves):
        U = _box_sample(box, m, seed + w)
        thetas = np.array([[s.unit_to_value(u) for s, u in zip(specs, row)] for row in U])
        per_output = {}
        implaus = np.zeros(m)
        for i, th in enumerate(thetas):
            P = apply_theta(P_base, specs, th)
            worst = 0.0
            for (tk, vig), (z, zv) in refs.items():
                res = run_vignette(terrains[tk], vig, P, n_reps, seed=seed * 1000 + w * 100 + i,
                                   n_plans=n_plans)
                mu, mv = metrics_of(res), metric_vars(res)
                for met in METRICS:
                    if met not in z or not np.isfinite(z[met]) or not np.isfinite(mu[met]):
                        continue
                    disc = (disc_frac * abs(z[met]) + (0.02 if met != "minutes" else 0.1)) ** 2
                    I = abs(mu[met] - z[met]) / np.sqrt(mv[met] + zv.get(met, 0.0) + disc)
                    per_output.setdefault((tk, vig, met), np.zeros(m))[i] = I
                    worst = max(worst, I)
            implaus[i] = worst
        nroy = implaus < threshold
        if nroy.sum() == 0:                      # keep the least implausible fifth so the wave can continue
            nroy = implaus <= np.quantile(implaus, 0.2)
            log(f"  wave {w + 1}: nothing under threshold {threshold}; keeping the best 20%")
        U_n = U[nroy]
        new_box = np.c_[U_n.min(0), U_n.max(0)]
        width = np.maximum(new_box[:, 1] - new_box[:, 0], shrink_floor)
        centre = new_box.mean(1)
        new_box = np.c_[np.clip(centre - width / 2, 0, 1), np.clip(centre + width / 2, 0, 1)]
        nroy_box = np.array([[s.unit_to_value(lo), s.unit_to_value(hi)] for s, (lo, hi) in zip(specs, new_box)])
        out.append(Wave(rung, box.copy(), thetas, implaus, per_output, nroy, nroy_box, n_reps))
        log(f"  rung {rung} wave {w + 1}: {nroy.sum()}/{m} NROY, min I = {implaus.min():.2f}, "
            f"box volume {np.prod(new_box[:, 1] - new_box[:, 0]):.3g}")
        box = new_box
    return out


def nroy_samples(wave, specs, k, seed=0):
    """Draw k parameter vectors uniformly from the last wave's NROY box (for prediction)."""
    rng = np.random.default_rng(seed)
    box = wave.nroy_box
    u = rng.random((k, len(specs)))
    return box[:, 0] + u * (box[:, 1] - box[:, 0])


def run_ladder(rungs, terrains, reference, lib_specs, *, n_reps=300, m=60, waves=3, seed=0,
               log=print, P0=None, half=1500.0):
    """Climb the ladder: at each rung, free only that rung's parameters, history-match them, then
    fix them at the centre of the NROY box for the rungs above. Returns LadderState."""
    from .vignettes import FREE, rung_vignettes
    state = LadderState(P=P0 or default_params())
    for rung in rungs:
        specs = FREE[rung]
        log(f"Rung {rung}: {len(specs)} free parameters, {len(terrains)} training terrains")
        ws = history_match(rung, specs, state.P, terrains, rung_vignettes(rung, half), reference,
                           n_reps=n_reps, m=m, waves=waves, seed=seed + rung, log=log)
        state.waves.extend(ws)
        last = ws[-1]
        centre = last.nroy_box.mean(1)
        state.P = apply_theta(state.P, specs, centre)
        for s, (lo, hi) in zip(specs, last.nroy_box):
            state.fixed[s.name] = (float(lo), float(hi))
        log("  fixed at: " + ", ".join(f"{s.name}={c:.3g}" for s, c in zip(specs, centre)))
    return state


# ---------------------------------------------------------------- diagnostics
def stability_report(waves):
    """Per parameter, how the NROY range moved across waves. Stable when the last two ranges
    overlap heavily and the width stopped shrinking."""
    rows = []
    by_rung = {}
    for w in waves:
        by_rung.setdefault(w.rung, []).append(w)
    for rung, ws in by_rung.items():
        d = ws[0].thetas.shape[1]
        for j in range(d):
            widths = [w.nroy_box[j, 1] - w.nroy_box[j, 0] for w in ws]
            centres = [w.nroy_box[j].mean() for w in ws]
            last, prev = ws[-1].nroy_box[j], ws[-2].nroy_box[j] if len(ws) > 1 else ws[-1].nroy_box[j]
            overlap = max(0.0, min(last[1], prev[1]) - max(last[0], prev[0])) / max(prev[1] - prev[0], 1e-12)
            rows.append(dict(rung=rung, param=j, width_first=widths[0], width_last=widths[-1],
                             shrink=widths[-1] / max(widths[0], 1e-12), centre_shift=centres[-1] - centres[0],
                             overlap_last_two=overlap, stable=bool(overlap > 0.7 and widths[-1] / max(widths[0], 1e-12) > 0.5)))
    return rows


def coverage_report(state, specs_by_rung, holdout, vignettes_by_rung, reference, n_reps=300,
                    k_theta=20, level=0.9, seed=0, log=print):
    """On held-out terrains, predict each reference output with k_theta draws from the final NROY
    boxes (parameter uncertainty) plus Monte Carlo noise, and report how often the central
    `level` prediction interval covers the reference. Target: coverage close to `level`."""
    rows = []
    rng = np.random.default_rng(seed)
    for rung, vigs in vignettes_by_rung.items():
        specs = specs_by_rung[rung]
        box = np.array([state.fixed[s.name] for s in specs])
        for tk, t in holdout.items():
            for vig in vigs:
                ref = reference.outcomes(tk, t, vig)
                if ref is None:
                    continue
                z, zv = ref
                preds = {m: [] for m in METRICS}
                for j in range(k_theta):
                    th = box[:, 0] + rng.random(len(specs)) * (box[:, 1] - box[:, 0])
                    res = run_vignette(t, vig, apply_theta(state.P, specs, th), n_reps, seed=seed + j)
                    mu, mv = metrics_of(res), metric_vars(res)
                    for m in METRICS:
                        if np.isfinite(mu[m]):
                            preds[m].append(rng.normal(mu[m], np.sqrt(mv[m] + zv.get(m, 0.0))))
                for m in METRICS:
                    if m not in z or not np.isfinite(z[m]) or not np.all(np.isfinite(preds[m])):
                        continue
                    lo, hi = np.quantile(preds[m], [(1 - level) / 2, 1 - (1 - level) / 2])
                    rows.append(dict(rung=rung, terrain=tk, vignette=f"{vig.armor}/R{int(vig.recon)}F{int(vig.fires)}/{vig.red_coa}",
                                     metric=m, ref=z[m], pred_lo=float(lo), pred_hi=float(hi),
                                     covered=bool(lo <= z[m] <= hi), error=float(np.mean(preds[m]) - z[m])))
    cov = np.mean([r["covered"] for r in rows]) if rows else np.nan
    log(f"Held-out coverage at {level:.0%}: {cov:.2f} over {len(rows)} outputs")
    return rows, float(cov)


def error_trend(rows, descriptors):
    """Regress prediction error on each terrain descriptor. A significant slope means the model's
    error depends on the kind of ground, which parameter tuning cannot fix: a mechanic is missing.
    rows: coverage_report rows; descriptors: terrain_key -> Descriptors."""
    out = []
    for m in METRICS:
        sub = [r for r in rows if r["metric"] == m]
        if len(sub) < 4:
            continue
        err = np.array([r["error"] for r in sub])
        if not np.all(np.isfinite(err)):
            continue
        for name in DESCRIPTOR_NAMES:
            x = np.array([getattr(descriptors[r["terrain"]], name) for r in sub])
            if x.std() == 0:
                continue
            res = sps.linregress(x, err)
            out.append(dict(metric=m, descriptor=name, slope=float(res.slope), p_value=float(res.pvalue),
                            flag=bool(res.pvalue < 0.05)))
    return out


def save_state(state, path):
    P = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in state.P.items()}
    json.dump(dict(P=P, fixed=state.fixed,
                   waves=[dict(rung=w.rung, nroy_box=w.nroy_box.tolist(), n_nroy=int(w.nroy.sum()),
                               m=len(w.implaus), min_implaus=float(w.implaus.min()), n_reps=w.n_reps)
                          for w in state.waves]), open(path, "w"), indent=1)
