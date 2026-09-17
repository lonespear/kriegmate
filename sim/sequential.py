"""Sequential sampling and ranking-and-selection.

`run_until` replaces a fixed replication count with a target half-width. `kn_select` is the
Kim & Nelson (2001) fully sequential indifference-zone procedure: it returns the COA whose
true P(win) is within delta of the best with probability at least 1 - alpha, screening COAs out
as evidence accumulates so the sampling budget goes to the close contenders.
"""
from __future__ import annotations

import numpy as np

from .engine import simulate
from .stats import wilson


def run_until(sim_fn, target_halfwidth, n0=200, batch=200, n_max=20_000):
    """Call sim_fn(n, seed_offset) -> array of 0/1 wins until the Wilson half-width is under target."""
    ys, off = [], 0
    while True:
        ys.append(sim_fn(n0 if off == 0 else batch, off))
        off += 1
        y = np.concatenate(ys)
        lo, hi = wilson(y.sum(), len(y))
        if (hi - lo) / 2 <= target_halfwidth or len(y) >= n_max:
            return y, (lo, hi)


def kn_select(sim_fns, delta=0.05, alpha=0.05, n0=50, batch=50, n_max=5000, crn=True):
    """Kim-Nelson KN procedure over k systems given by sim_fns[i](n, seed_offset) -> outcomes.

    With crn=True every system draws the same seed offsets, which KN allows and which sharpens
    the pairwise variance estimates. Returns (winner index, n used per system, history)."""
    k = len(sim_fns)
    eta = 0.5 * (((2 * alpha) / (k - 1)) ** (-2.0 / (n0 - 1)) - 1)
    h2 = 2 * eta * (n0 - 1)
    Y = [list(f(n0, 0)) for f in sim_fns]
    alive = list(range(k))
    S2 = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                d = np.asarray(Y[i]) - np.asarray(Y[j])
                S2[i, j] = d.var(ddof=1)
    hist = []
    off = 1
    while len(alive) > 1:
        r = len(Y[alive[0]])
        means = {i: np.mean(Y[i]) for i in alive}
        keep = []
        for i in alive:
            ok = True
            for j in alive:
                if i == j:
                    continue
                W = max(0.0, (delta / (2 * r)) * (h2 * S2[i, j] / delta ** 2 - r))
                if means[i] < means[j] - W:
                    ok = False
                    break
            if ok:
                keep.append(i)
        alive = keep if keep else [max(alive, key=lambda i: means[i])]
        hist.append((r, list(alive)))
        if len(alive) == 1 or r >= n_max:
            break
        for i in alive:
            Y[i].extend(sim_fns[i](batch, off))
        off += 1
    best = max(alive, key=lambda i: np.mean(Y[i]))
    return best, {i: len(Y[i]) for i in range(k)}, hist


def coa_fn(tab_for, coa, red_coa, seed, P=None, red_trigger=600.0):
    armor, recon, fires = coa
    return lambda n, off: simulate(tab_for(armor), armor, recon, fires, red_coa, n, seed=[seed, off],
                                   P=P, n_log=0, red_trigger=red_trigger).win
