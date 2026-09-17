"""Estimator tests against known truths."""
import numpy as np

from sim import stats


def test_wilson_inside_unit_interval_and_covers():
    rng = np.random.default_rng(0)
    cover = 0
    for _ in range(400):
        x = rng.binomial(50, 0.05)
        lo, hi = stats.wilson(x, 50)
        assert 0 <= lo <= hi <= 1
        cover += lo <= 0.05 <= hi
    assert cover / 400 > 0.9


def test_kaplan_meier_exponential():
    rng = np.random.default_rng(0)
    T = rng.exponential(10, 5000)
    obs = np.minimum(T, 12.0)
    t, S, se = stats.kaplan_meier(obs, T <= 12.0)
    assert abs(S[np.searchsorted(t, 5) - 1] - np.exp(-0.5)) < 0.02


def test_aalen_johansen_competing_exponentials():
    rng = np.random.default_rng(0)
    T1, T2 = rng.exponential(10, 20000), rng.exponential(5, 20000)
    tt = np.minimum(T1, T2)
    cause = np.where(tt < 30, np.where(T1 < T2, 1, 2), 0)
    ts, cif = stats.aalen_johansen(np.minimum(tt, 30), cause)
    assert abs(cif[-1, 0] - (1 / 3) * (1 - np.exp(-9))) < 0.015
    assert abs(cif[-1, 1] - (2 / 3) * (1 - np.exp(-9))) < 0.015


def test_kn_selects_best_system():
    rng = np.random.default_rng(3)
    means = [0.0, 0.0, 0.0, 1.0]
    hits = 0
    for trial in range(10):
        res = stats.kn_select([lambda n, m=m: rng.normal(m, 2.0, n) for m in means], delta=0.5,
                              alpha=0.05, n0=30)
        hits += res["best"] == 3
    assert hits >= 9


def test_shrinkage_pulls_small_cells_toward_mean():
    x = np.array([0, 1, 5, 50, 90])
    n = np.array([1, 2, 10, 100, 100])
    post, lo, hi, strength = stats.shrink_rates(x, n)
    assert post[0] > 0.3                      # 0/1 is not believed to be zero
    assert abs(post[3] - 0.5) < 0.03          # 50/100 barely moves
    assert post[4] > 0.85
    assert np.all(lo <= post) and np.all(post <= hi)


def test_sup_t_band_wider_than_pointwise():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(300, 40))
    m, lo, hi, c = stats.sup_t_band(Y, B=300)
    _, plo, phi = stats.band(Y)
    assert c > 1.96 and np.all(lo <= plo) and np.all(hi >= phi)


def test_run_until_stops_at_half_width():
    rng = np.random.default_rng(0)
    y, n, hw = stats.run_until(lambda k: rng.random(k), 0.02, batch=100)
    assert hw <= 0.02 and n == len(y)


def test_crn_gain_detects_pairing():
    rng = np.random.default_rng(0)
    a = rng.random(500)
    rho, ratio = stats.crn_gain(a, a + 0.05 * rng.random(500))
    assert rho > 0.95 and ratio < 0.1
    rho2, ratio2 = stats.crn_gain(rng.random(500), rng.random(500))
    assert abs(rho2) < 0.15 and ratio2 > 0.8


def test_ratio_ci_covers():
    rng = np.random.default_rng(1)
    cover = 0
    for _ in range(300):
        num, den = rng.poisson(4, 200), rng.poisson(2, 200)
        r, lo, hi = stats.ratio_ci(num, den)
        cover += lo <= 2.0 <= hi
    assert cover / 300 > 0.9
