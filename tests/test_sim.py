"""Engine, terrain, scenario, and roster tests."""
import numpy as np
import pytest

from sim.engine import choice_prob, concat_results, gumbel_choice, hit_prob, simulate
from sim.scenario import Scenario, ScenarioConfig
from sim.terrain import (BUILDING, FOREST, ROAD, WATER, Terrain, blank_terrain, demo_terrain, los_fraction,
                         rasterize, synthetic_terrain)
from sim.units import COMPANY, DUEL, R_AT, R_INF, RUNG_ROSTERS, default_params
from sim.vignettes import rung_config


@pytest.fixture(scope="module")
def scen():
    return Scenario(demo_terrain(), ScenarioConfig())


def test_los_wall_blocks_and_tall_target_clears():
    t = blank_terrain(0, 0, 500, 10, "test")
    i, j = t.xy_to_ij(0, np.arange(-50, 50, 5))
    t.surface[i, j] = 5.0
    obs = np.array([[-100.0, 0.0]] * 2)
    tgt = np.array([[100.0, 0.0]] * 2)
    v = los_fraction(t, obs, [1.7, 1.7], tgt, np.array([[1.0, 1.5, 2.0], [9.0, 9.5, 10.0]]))
    assert v[0] == 0.0 and v[1] == 1.0


def test_hit_prob_matches_brute_force():
    rng = np.random.default_rng(0)
    d, th, A, vis = 800.0, 1e-3, 6.0, 0.5
    shots = rng.normal(0, d * th, (300_000, 2))
    mc = (np.hypot(*shots.T) <= np.sqrt(vis * A / np.pi)).mean()
    assert abs(mc - hit_prob(d, vis, A, th, 0, 0, 0, 0, 0, 0)) < 0.005


def test_gumbel_choice_matches_logit():
    rng = np.random.default_rng(1)
    V = np.array([np.log(.2), np.log(.05), -np.inf])
    N = 200_000
    ch = gumbel_choice(np.broadcast_to(V, (N, 3)), np.full(N, np.log(.03)), np.full(N, 2.0), rng.random((N, 4)))
    z = 2.0 * np.r_[np.log(.03), V]
    p = np.exp(z - z.max()); p /= p.sum()
    assert np.allclose(np.bincount(ch + 1, minlength=4) / N, p, atol=0.005)
    assert np.isclose(choice_prob(V[None], np.array([np.log(.03)]), np.array([2.0]), np.array([0]))[0], p[1])


def test_rasterize_osm_like_json():
    t = blank_terrain(41.0, -74.0, 500, 10, "test")

    def ring(x0, y0, x1, y1):
        lon, lat = t.xy_to_lonlat(np.array([x0, x1, x1, x0, x0]), np.array([y0, y0, y1, y1, y0]))
        return [{"lon": a, "lat": b} for a, b in zip(lon, lat)]

    def line(pts):
        lon, lat = t.xy_to_lonlat(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
        return [{"lon": a, "lat": b} for a, b in zip(lon, lat)]

    els = [
        {"type": "way", "tags": {"building": "yes", "building:levels": "3"}, "geometry": ring(0, 0, 40, 40)},
        {"type": "way", "tags": {"landuse": "forest"}, "geometry": ring(-300, -300, -100, -100)},
        {"type": "way", "tags": {"waterway": "river"}, "geometry": line([(200, -500), (200, 500)])},
        {"type": "way", "tags": {"highway": "primary"}, "geometry": line([(-500, 200), (500, 200)])},
        {"type": "way", "tags": {"highway": "footway"}, "geometry": line([(-500, -200), (500, -200)])},
    ]
    counts = rasterize(t, els)
    assert counts["building"] == 1 and counts["road"] == 1
    assert t.at(t.cls, 20, 20) == BUILDING and np.isclose(t.at(t.surface, 20, 20), 9.6)
    assert t.at(t.cls, -200, -200) == FOREST
    assert t.at(t.cls, 200, 0) == WATER
    assert t.at(t.cls, 200, 200) == ROAD


def test_terrain_save_load_roundtrip(tmp_path):
    t = synthetic_terrain(half=800, building_density=0.01, forest_share=0.2, relief=40, seed=3)
    t.fetched_at = "2026-09-16T00:00:00+00:00"
    t.save(tmp_path / "t.npz")
    t2 = Terrain.load(tmp_path / "t.npz")
    assert np.array_equal(t.surface, t2.surface) and t2.name == t.name and t2.fetched_at == t.fetched_at


def test_synthetic_terrain_descriptor_targets():
    lo = synthetic_terrain(half=1500, building_density=0.0005, forest_share=0.02, relief=10, seed=1)
    hi = synthetic_terrain(half=1500, building_density=0.02, forest_share=0.25, relief=80, seed=1)
    assert (hi.cls == FOREST).mean() > 3 * (lo.cls == FOREST).mean()
    assert (hi.cls == BUILDING).mean() > 5 * (lo.cls == BUILDING).mean()
    assert np.ptp(hi.ground) > 3 * np.ptp(lo.ground)


def test_scenario_tables(scen):
    tab = scen.tables("SBF")
    assert tab["visR"].shape[0] == 12 and tab["visR"].shape[2] == 10
    assert np.all(np.diff(tab["cum"], axis=1) > 0)
    assert scen.sbf_best_vis > 0


def test_every_rung_roster_builds_and_runs():
    t = demo_terrain()
    for rung, R in RUNG_ROSTERS.items():
        sc = Scenario(t, rung_config(rung, t.half), R, seed=1, placement_temperature=0.5)
        assert sc.rpos.shape == (R.nr, 2)
        r = simulate(sc.tables("Maneuver"), "Maneuver", bool(R.is_scout.any()), rung >= 3, "Early", 60,
                     seed=[rung], n_log=3)
        assert r.frames["xy"].shape[1:] == (3, R.nb, 2)
        assert r.blue_shots.shape == (60, 3) and r.cause.shape == (60,)


def test_stochastic_red_placement_varies_with_seed():
    t = demo_terrain()
    a = Scenario(t, ScenarioConfig(), COMPANY, seed=1, placement_temperature=0.5)
    b = Scenario(t, ScenarioConfig(), COMPANY, seed=2, placement_temperature=0.5)
    c = Scenario(t, ScenarioConfig(), COMPANY, seed=2, placement_temperature=0.0)
    d = Scenario(t, ScenarioConfig(), COMPANY, seed=2, placement_temperature=0.0)
    assert (a.rpos != b.rpos).any()
    assert np.array_equal(c.rpos, d.rpos)          # same seed -> same plan
    assert (b.rpos != c.rpos).any()                 # temperature moves the plan off the best-scored cells


def test_engine_deterministic_and_bounded(scen):
    a = simulate(scen.tables("SBF"), "SBF", True, True, "Early", 200, seed=[3], n_log=5)
    b = simulate(scen.tables("SBF"), "SBF", True, True, "Early", 200, seed=[3], n_log=5)
    assert np.array_equal(a.win, b.win) and np.array_equal(a.blue_losses, b.blue_losses)
    assert a.frames["xy"].shape[1:] == (5, 12, 2)
    assert 0 <= a.win.mean() <= 1 and a.blue_losses.max() <= 10
    assert set(np.unique(a.cause)) <= {0, 1, 2}
    assert np.all((a.cause == 2) == (a.win == 1))


def test_substreams_keep_crn_when_a_mechanic_is_switched():
    """With named substreams, switching fires on/off must not change the detection draws: the
    first tick's detections are identical across the two runs."""
    t = demo_terrain()
    sc = Scenario(t, ScenarioConfig())
    tab = sc.tables("SBF")
    a = simulate(tab, "SBF", True, False, "Early", 100, seed=[11], n_log=100)
    b = simulate(tab, "SBF", True, True, "Early", 100, seed=[11], n_log=100)
    assert np.array_equal(a.frames["red_seen"][1], b.frames["red_seen"][1])
    assert np.array_equal(a.frames["blue_seen"][1], b.frames["blue_seen"][1])


def test_tick_rescaling_keeps_units_sane(scen):
    tab = scen.tables("SBF")
    r15 = simulate(tab, "SBF", True, True, "Late", 300, seed=[5], tick_s=15.0, n_log=0)
    r7 = simulate(tab, "SBF", True, True, "Late", 300, seed=[5], tick_s=7.5, n_log=0)
    assert r7.blue_strength.shape[1] > r15.blue_strength.shape[1]        # more ticks, same time limit
    assert abs(r7.minutes.mean() - r15.minutes.mean()) < 2.0               # minutes, not ticks
    assert r7.tick_s == 7.5


def test_concat_results_stacks_plans():
    t = demo_terrain()
    parts = []
    for p in range(2):
        sc = Scenario(t, ScenarioConfig(), COMPANY, seed=p, placement_temperature=0.5)
        parts.append(simulate(sc.tables("SBF"), "SBF", True, True, "Early", 50, seed=[p], n_log=2 if p == 0 else 0))
    c = concat_results(parts)
    assert c.n == 100 and c.n_plans == 2 and c.frames["xy"].shape[1] == 2
    assert c.blue_strength.shape[0] == 100 and c.blue_shots.shape == (100, 3)


def test_harmless_red_never_kills(scen):
    P = default_params()
    P["PK"][R_AT] = 0
    P["PK"][R_INF] = 0
    r = simulate(scen.tables("Maneuver"), "Maneuver", True, True, "Early", 200, seed=[4], P=P)
    assert r.blue_losses.max() == 0


def test_duel_monotone_in_red_lethality():
    """Regression vignette: lowering Red's kill probability must raise Blue's win rate."""
    t = synthetic_terrain(half=1500.0, seed=0)
    sc = Scenario(t, rung_config(1, 1500.0), DUEL, seed=0, placement_temperature=0.5)
    tab = sc.tables("Maneuver")
    wins = []
    for pk in (0.8, 0.5, 0.2):
        P = default_params()
        P["PK"][R_AT, 0] = pk
        wins.append(simulate(tab, "Maneuver", False, False, "Early", 600, seed=[1], P=P, n_log=0).win.mean())
    assert wins[0] < wins[1] < wins[2]
