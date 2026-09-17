"""Time-to-event analysis of engagements.

Timeouts are censored observations: the fight had not been decided when the clock ran out.
`kaplan_meier` handles that for time to decision; `aalen_johansen` treats Blue breaking and Red
breaking as competing risks and gives the cumulative incidence of each. `sup_t_band` turns the
pointwise strength bands into a simultaneous band by bootstrapping the maximum standardized
deviation across time.
"""
from __future__ import annotations

import numpy as np


def kaplan_meier(time, event):
    """Survival of 'undecided' status. time in minutes; event 1 = decided, 0 = censored."""
    t = np.asarray(time, float)
    e = np.asarray(event, bool)
    ts = np.unique(t[e])
    S, var_s, out = 1.0, 0.0, []
    for u in ts:
        at_risk = (t >= u).sum()
        d = (e & (t == u)).sum()
        if at_risk == 0:
            continue
        S *= 1 - d / at_risk
        var_s += d / (at_risk * (at_risk - d)) if at_risk > d else 0.0
        se = S * np.sqrt(var_s)
        out.append((u, S, max(S - 1.96 * se, 0), min(S + 1.96 * se, 1)))
    return np.array(out) if out else np.zeros((0, 4))


def aalen_johansen(time, cause):
    """Cumulative incidence per cause. cause: 0 censored, 1 Blue wins (Red breaks), 2 Blue fails."""
    t = np.asarray(time, float)
    c = np.asarray(cause, int)
    ts = np.unique(t[c > 0])
    S, F1, F2, rows = 1.0, 0.0, 0.0, []
    for u in ts:
        at_risk = (t >= u).sum()
        d1, d2 = ((t == u) & (c == 1)).sum(), ((t == u) & (c == 2)).sum()
        F1 += S * d1 / at_risk
        F2 += S * d2 / at_risk
        S *= 1 - (d1 + d2) / at_risk
        rows.append((u, F1, F2))
    return np.array(rows) if rows else np.zeros((0, 3))


def sup_t_band(Y, alpha=0.05, B=500, seed=0):
    """Simultaneous 1-alpha band for the mean curve of Y (replications x time)."""
    rng = np.random.default_rng(seed)
    n, T = Y.shape
    m = Y.mean(0)
    se = Y.std(0, ddof=1) / np.sqrt(n)
    se = np.where(se > 0, se, np.inf)
    sup = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        sup[b] = np.max(np.abs(Y[idx].mean(0) - m) / se)
    c = np.quantile(sup, 1 - alpha)
    se = np.where(np.isfinite(se), se, 0.0)
    return m, m - c * se, m + c * se, float(c)
