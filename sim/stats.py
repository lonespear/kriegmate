"""Interval estimators, sequential sampling, ranking-and-selection, survival analysis,
simultaneous bands, and shrinkage used by the app and the calibration tools.

Every estimator here treats one replication as one observation. Replications within a run are
independent; across two COAs run on the same seed they are paired (common random numbers).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as student_t


# ---------------------------------------------------------------- basic intervals
def wilson(x, n, a=0.05):
    """Wilson score interval for a binomial proportion. Stays inside [0, 1] and keeps its
    coverage near 0 and 1, where the Wald interval collapses."""
    z = norm.ppf(1 - a / 2)
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mean_ci(y, a=0.05):
    y = np.asarray(y, float)
    m = y.mean()
    se = y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.nan
    z = norm.ppf(1 - a / 2)
    return m, m - z * se, m + z * se


def ratio_ci(num, den, a=0.05):
    """Ratio of means with a delta-method interval."""
    mn, md = np.mean(num), np.mean(den)
    if md == 0:
        return np.nan, np.nan, np.nan
    r = mn / md
    C = np.cov(num, den) / len(num)
    var = r * r * (C[0, 0] / mn ** 2 + C[1, 1] / md ** 2 - 2 * C[0, 1] / (mn * md)) if mn else C[0, 0] / md ** 2
    z = norm.ppf(1 - a / 2)
    se = np.sqrt(max(var, 0))
    return r, r - z * se, r + z * se


def paired_ci(a, b, alpha=0.05):
    """Mean of b - a over common-random-number replications."""
    return mean_ci(np.asarray(b, float) - np.asarray(a, float), alpha)


def crn_gain(a, b):
    """Variance-reduction diagnostic for a paired comparison: the achieved correlation between
    the two COAs' outputs and the ratio of paired to independent variance of the difference
    (1 means CRN bought nothing; 0.5 means it halved the variance)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return 0.0, 1.0
    rho = np.corrcoef(a, b)[0, 1]
    v_ind = a.var(ddof=1) + b.var(ddof=1)
    return float(rho), float((a - b).var(ddof=1) / v_ind)


def band(Y, a=0.05):
    """Pointwise mean and CI for each column of Y (replications x time)."""
    m = Y.mean(0)
    se = Y.std(0, ddof=1) / np.sqrt(Y.shape[0])
    z = norm.ppf(1 - a / 2)
    return m, m - z * se, m + z * se


def sup_t_band(Y, a=0.05, B=2000, seed=0):
    """Simultaneous confidence band for the mean curve, by the sup-t bootstrap: the band covers
    the whole mean curve with probability 1 - a, not each time point separately."""
    Y = np.asarray(Y, float)
    n, T = Y.shape
    m = Y.mean(0)
    se = Y.std(0, ddof=1) / np.sqrt(n)
    se_safe = np.where(se > 0, se, np.inf)
    rng = np.random.default_rng(seed)
    sup = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        mb = Y[idx].mean(0)
        sup[b] = np.max(np.abs(mb - m) / se_safe)
    c = np.quantile(sup, 1 - a)
    return m, m - c * se, m + c * se, float(c)


# ---------------------------------------------------------------- sample-size planning
def n_for_half_width(half_width, p=0.5, a=0.05):
    """Replications needed for a Wald half-width on a proportion (worst case p = 0.5)."""
    z = norm.ppf(1 - a / 2)
    return int(np.ceil(z * z * p * (1 - p) / half_width ** 2))


def run_until(sampler, half_width, batch=200, max_n=20000, a=0.05, min_n=200):
    """Sequential sampling: call sampler(batch) for fresh replications until the 95% (or 1 - a)
    half-width of the mean is at or below half_width. Returns (samples, n, achieved half-width).

    Stopping on the observed half-width makes the final interval slightly optimistic; the bias is
    small for min_n >= 200 (Law, Simulation Modeling and Analysis, sec. 9.4)."""
    ys = [sampler(min_n)]
    n = min_n
    while True:
        y = np.concatenate(ys)
        hw = norm.ppf(1 - a / 2) * y.std(ddof=1) / np.sqrt(n)
        if hw <= half_width or n >= max_n:
            return y, n, float(hw)
        ys.append(sampler(batch))
        n += batch


# ---------------------------------------------------------------- ranking and selection
def kn_select(samplers, delta, alpha=0.05, n0=50, max_n=10000, c=1.0):
    """Kim-Nelson (KN) fully sequential indifference-zone procedure.

    samplers: list of callables sampler(n) -> n fresh observations of each system (larger is
    better). Guarantees P(correct selection) >= 1 - alpha whenever the best system beats the
    others by at least delta. Returns dict with the selected index, the elimination round of
    each system, and the samples used. Reference: Kim & Nelson, ACM TOMACS 11(3), 2001.
    """
    k = len(samplers)
    if k == 1:
        return dict(best=0, rounds=[0], n=[0], samples=[np.zeros(0)])
    eta = 0.5 * ((2 * alpha / (k - 1)) ** (-2 / (n0 - 1)) - 1)
    h2 = 2 * c * eta * (n0 - 1)
    X = [np.asarray(s(n0), float) for s in samplers]
    S2 = np.zeros((k, k))
    for i in range(k):
        for l in range(k):
            if i != l:
                S2[i, l] = np.var(X[i] - X[l], ddof=1)
    alive = list(range(k))
    elim_round = [0] * k
    r = n0
    while len(alive) > 1 and r < max_n:
        means = {i: X[i].mean() for i in alive}
        keep = []
        for i in alive:
            ok = True
            for l in alive:
                if l == i:
                    continue
                W = max(0.0, (delta / (2 * c * r)) * (h2 * S2[i, l] / delta ** 2 - r))
                if means[i] < means[l] - W:
                    ok = False
                    break
            if ok:
                keep.append(i)
            else:
                elim_round[i] = r
        alive = keep if keep else alive[:1]
        if len(alive) == 1:
            break
        for i in alive:
            X[i] = np.concatenate([X[i], np.asarray(samplers[i](1), float)])
        r += 1
    best = max(alive, key=lambda i: X[i].mean())
    return dict(best=best, rounds=elim_round, n=[len(x) for x in X], samples=X,
                terminated=len(alive) == 1)


# ---------------------------------------------------------------- survival analysis
def kaplan_meier(times, event):
    """Kaplan-Meier survival estimate with Greenwood standard errors.
    times: decision times; event: 1 if the event happened, 0 if censored (time limit reached)."""
    times, event = np.asarray(times, float), np.asarray(event, bool)
    ts = np.unique(times[event])
    S, var, out_t = 1.0, 0.0, []
    surv, se = [], []
    for tt in ts:
        at_risk = (times >= tt).sum()
        d = (event & (times == tt)).sum()
        if at_risk == 0:
            continue
        S *= 1 - d / at_risk
        var += d / (at_risk * (at_risk - d)) if at_risk > d else 0
        out_t.append(tt)
        surv.append(S)
        se.append(S * np.sqrt(var))
    return np.array(out_t), np.array(surv), np.array(se)


def aalen_johansen(times, cause, n_causes=2):
    """Cumulative incidence of each competing cause (Aalen-Johansen estimator).
    cause: 0 = censored, 1..n_causes = the cause. Returns (times, CIF array of shape
    (len(times), n_causes)). Treating a competing event as censoring would overstate each cause's
    incidence; this estimator does not."""
    times, cause = np.asarray(times, float), np.asarray(cause, int)
    ts = np.unique(times[cause > 0])
    S = 1.0
    cif = np.zeros(n_causes)
    out_t, out = [], []
    for tt in ts:
        at_risk = (times >= tt).sum()
        if at_risk == 0:
            continue
        d = np.array([((cause == c) & (times == tt)).sum() for c in range(1, n_causes + 1)])
        cif = cif + S * d / at_risk
        S *= 1 - d.sum() / at_risk
        out_t.append(tt)
        out.append(cif.copy())
    return np.array(out_t), np.array(out).reshape(-1, n_causes)


# ---------------------------------------------------------------- shrinkage
def shrink_rates(x, n, prior_strength=None):
    """Empirical-Bayes beta-binomial shrinkage for per-cell rates x / n.

    Cells with few trials are pulled toward the overall mean; cells with many keep their raw rate.
    The prior (a, b) is fitted by the method of moments across cells; prior_strength overrides
    a + b. Returns posterior mean and a 90% posterior interval."""
    x, n = np.asarray(x, float), np.asarray(n, float)
    ok = n > 0
    N, k = n[ok].sum(), ok.sum()
    m = x[ok].sum() / N if N > 0 else 0.5
    p = np.where(ok, x / np.maximum(n, 1), m)
    if prior_strength is None:
        # weighted moment estimate of the between-cell variance (DerSimonian-Laird form)
        q = (n[ok] * (p[ok] - m) ** 2).sum() - (k - 1) * m * (1 - m)
        denom = N - (n[ok] ** 2).sum() / N
        tau2 = max(q / denom, 1e-6) if denom > 0 else 1e-6
        strength = float(np.clip(m * (1 - m) / tau2 - 1, 1.0, 1e4))
    else:
        strength = prior_strength
    a, b = m * strength, (1 - m) * strength
    post_a, post_b = a + x, b + (n - x)
    from scipy.stats import beta as beta_dist
    mean = post_a / (post_a + post_b)
    lo, hi = beta_dist.ppf(0.05, post_a, post_b), beta_dist.ppf(0.95, post_a, post_b)
    return mean, lo, hi, float(strength)


def t_ci(y, a=0.05):
    y = np.asarray(y, float)
    m, s, n = y.mean(), y.std(ddof=1), len(y)
    h = student_t.ppf(1 - a / 2, n - 1) * s / np.sqrt(n)
    return m, m - h, m + h


# ---------------------------------------------------------------- zero-sum games
def solve_zero_sum(M):
    """Value and optimal mixed strategies of the zero-sum game with payoff matrix M (row player
    maximizes). Linear programming (scipy.optimize.linprog). Returns (value, row_mix, col_mix)."""
    from scipy.optimize import linprog
    M = np.asarray(M, float)
    r, c = M.shape
    shift = M.min() - 1.0
    Ms = M - shift
    # row: maximize v s.t. Ms^T x >= v, sum x = 1
    res = linprog(c=np.r_[np.zeros(r), -1.0], A_ub=np.c_[-Ms.T, np.ones(c)], b_ub=np.zeros(c),
                  A_eq=np.r_[np.ones(r), 0.0][None, :], b_eq=[1.0],
                  bounds=[(0, None)] * r + [(None, None)], method="highs")
    x, v = res.x[:r], res.x[r]
    res2 = linprog(c=np.r_[np.zeros(c), 1.0], A_ub=np.c_[Ms, -np.ones(r)], b_ub=np.zeros(r),
                   A_eq=np.r_[np.ones(c), 0.0][None, :], b_eq=[1.0],
                   bounds=[(0, None)] * c + [(None, None)], method="highs")
    y = res2.x[:c]
    return float(v + shift), np.clip(x, 0, 1), np.clip(y, 0, 1)


def bootstrap_game_value(cells, B=500, seed=0):
    """cells: dict (i, j) -> array of per-replication payoffs (same length across cells when paired
    by common random numbers). Resamples replications, re-solves the game each time, and returns
    (value, lo, hi, row_mix_mean, col_mix_mean)."""
    keys = sorted(cells)
    r = max(k[0] for k in keys) + 1
    c = max(k[1] for k in keys) + 1
    n = min(len(v) for v in cells.values())
    rng = np.random.default_rng(seed)
    M = np.array([[cells[i, j][:n].mean() for j in range(c)] for i in range(r)])
    v0, x0, y0 = solve_zero_sum(M)
    vals, xs, ys = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        Mb = np.array([[cells[i, j][idx].mean() for j in range(c)] for i in range(r)])
        v, x, y = solve_zero_sum(Mb)
        vals.append(v)
        xs.append(x)
        ys.append(y)
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return v0, float(lo), float(hi), x0, y0, np.mean(xs, 0), np.mean(ys, 0), M
