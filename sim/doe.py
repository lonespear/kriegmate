"""Design of experiments, Gaussian-process metamodel, and Sobol sensitivity indices.

Workflow: sample the parameter (and scenario-factor) space with a space-filling design, run the
simulator at each point, fit a GP to the outputs, and compute first-order and total Sobol indices
on the GP with the Saltelli/Jansen estimators. The indices say which inputs move each output; the
ones with large total indices are the ones worth calibrating or measuring.

The GP is stochastic-kriging-like: a fitted white-noise term absorbs Monte Carlo error, and the
known per-point MC variance is used as the noise floor. With scipy's quasi-random tools the design
is a maximin-optimized Latin hypercube; a nearly orthogonal Latin hypercube (NOLH, Cioppa &
Lucas 2007) can be dropped in by replacing `design()` with a loader for the SEED Center tables.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from .vignettes import ParamSpec, apply_theta, metric_vars, metrics_of, run_vignette


def design(specs, m, seed=0):
    """m-point Latin hypercube in the unit cube, mapped to natural units through the specs."""
    U = qmc.LatinHypercube(d=len(specs), seed=seed, optimization="random-cd").random(m)
    X = np.array([[s.unit_to_value(u) for s, u in zip(specs, row)] for row in U])
    return U, X


def run_design(specs, X, terrain, vig, P_base, n_reps=300, seed=0, log=None, metrics=None):
    """Simulate every design point. Returns (Y, V): metric means and their MC variances,
    arrays of shape (m, n_metrics)."""
    metrics = metrics or ["p_win", "blue_loss_frac", "red_loss_frac", "minutes"]
    Y, V = [], []
    for i, th in enumerate(X):
        res = run_vignette(terrain, vig, apply_theta(P_base, specs, th), n_reps, seed=seed + i)
        mu, mv = metrics_of(res), metric_vars(res)
        Y.append([mu[k] for k in metrics])
        V.append([mv[k] for k in metrics])
        if log and (i + 1) % 10 == 0:
            log(f"  design point {i + 1}/{len(X)}")
    return np.array(Y), np.array(V), metrics


@dataclass
class Metamodel:
    gp: GaussianProcessRegressor
    y_mean: float
    y_sd: float
    noise_share: float = 0.0     # MC variance / total output variance at the design points

    def predict(self, U, return_std=False):
        out = self.gp.predict(np.atleast_2d(U), return_std=return_std)
        if return_std:
            m, s = out
            return m * self.y_sd + self.y_mean, s * self.y_sd
        return out * self.y_sd + self.y_mean


def fit_metamodel(U, y, v=None, seed=0):
    """Fit an anisotropic RBF GP with a white-noise term on unit-cube inputs. v (MC variances)
    sets the noise floor so the GP does not overfit simulation noise."""
    y = np.asarray(y, float)
    ym, ys = y.mean(), y.std() + 1e-12
    z = (y - ym) / ys
    noise_floor = float(np.mean(v) / ys ** 2) if v is not None else 1e-5
    nf = float(np.clip(noise_floor, 1e-6, 0.99))
    kernel = (ConstantKernel(1.0, (1e-2, 1e2)) * RBF(np.full(U.shape[1], 0.3), (0.03, 10.0))
              + WhiteKernel(nf, (nf, 1.0)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=3,
                                      random_state=seed).fit(U, z)
    return Metamodel(gp, ym, ys, float(min(noise_floor, 1.0)))


def loo_r2(U, y, v=None, seed=0):
    """Leave-one-out R^2 of the metamodel: below ~0.7, the Sobol indices are not trustworthy."""
    y = np.asarray(y, float)
    pred = np.empty_like(y)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        mm = fit_metamodel(U[keep], y[keep], None if v is None else v[keep], seed)
        pred[i] = mm.predict(U[i:i + 1])[0]
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / max(ss_tot, 1e-12))


def sobol_indices(model, d, N=4096, seed=0):
    """First-order (S) and total (ST) Sobol indices on the unit cube with the Saltelli scheme and
    Jansen's estimators (Saltelli et al. 2010). Returns dict with S, ST and bootstrap SEs."""
    sob = qmc.Sobol(d=2 * d, seed=seed, scramble=True)
    AB = sob.random(N)
    A, B = AB[:, :d], AB[:, d:]
    fA, fB = model.predict(A), model.predict(B)
    var = np.var(np.r_[fA, fB], ddof=1)
    S, ST = np.zeros(d), np.zeros(d)
    fAB = []
    for j in range(d):
        ABj = A.copy()
        ABj[:, j] = B[:, j]
        f = model.predict(ABj)
        fAB.append(f)
        S[j] = np.mean(fB * (f - fA)) / var
        ST[j] = 0.5 * np.mean((fA - f) ** 2) / var
    # bootstrap SEs over the sample rows
    rng = np.random.default_rng(seed)
    bs, bst = [], []
    for _ in range(100):
        idx = rng.integers(0, N, N)
        vb = np.var(np.r_[fA[idx], fB[idx]], ddof=1)
        bs.append([np.mean(fB[idx] * (fAB[j][idx] - fA[idx])) / vb for j in range(d)])
        bst.append([0.5 * np.mean((fA[idx] - fAB[j][idx]) ** 2) / vb for j in range(d)])
    return dict(S=S, ST=ST, S_se=np.std(bs, 0), ST_se=np.std(bst, 0))
