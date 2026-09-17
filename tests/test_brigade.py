"""Nation catalogs, the aggregate brigade engine, and the game solver."""
import numpy as np

from sim.brigade import (ATTACK_COAS, DEFEND_COAS, ENG, HELO, MANEUVER, ROLE_AVIATION, ROLE_ENGINEER, ROLE_SUPPORT,
                         BrigadeSim, BrigadeTerrain, build_force, load_correction, rasterize_coarse,
                         synthetic_brigade_terrain, brigade_terrain_from)
from sim.engine import simulate
from sim.nations import CLASSES, NATIONS, entity_params, pk_matrix
from sim.scenario import Scenario, ScenarioConfig
from sim.stats import bootstrap_game_value, solve_zero_sum
from sim.terrain import synthetic_terrain
from sim.units import COMPANY


def test_pk_matrix_orders_protection_and_warhead():
    us, ir = NATIONS["US"], NATIONS["IR"]
    PK, RM = pk_matrix(us, ["tank", "atgm", "inf"], ir, ["tank", "ifv", "inf"])
    assert PK[0, 1] > PK[0, 0]                     # tank gun kills IFV more often than a tank
    assert PK[1, 0] > PK[2, 0]                     # ATGM beats small arms against a tank
    assert RM[2, 0] <= 500
    PK2, _ = pk_matrix(ir, ["tank"], us, ["tank"])
    assert PK2[0, 0] < PK[0, 0]                    # older gun and lower quality vs modern armor


def test_entity_params_schema_runs_in_entity_engine():
    P = entity_params(NATIONS["CN"], NATIONS["IR"])
    assert P["PK"].shape == (6, 6) and P["theta0"].shape == (6,) and P["PK"][:4, :4].sum() == 0
    t = synthetic_terrain(half=1500.0, seed=0)
    sc = Scenario(t, ScenarioConfig(start_dist=1300), COMPANY, seed=0)
    r = simulate(sc.tables("SBF"), "SBF", True, True, "Late", 40, seed=[1], P=P, n_log=0)
    assert 0 <= r.win.mean() <= 1


def test_force_structure_and_scaling():
    f1 = build_force(NATIONS["RU"], 1)
    f2 = build_force(NATIONS["RU"], 1, copies=2)
    f3 = build_force(NATIONS["RU"], 1, strength=0.5)
    assert f2.N == 2 * f1.N and f2.bn.max() == 2 * (f1.bn.max() + 1) - 1
    assert (f3.n0 * MANEUVER).sum() < 0.6 * (f1.n0 * MANEUVER).sum()
    assert {ROLE_ENGINEER, ROLE_SUPPORT, ROLE_AVIATION} <= set(f1.role)
    assert f1.n0[f1.role == ROLE_AVIATION, HELO].sum() == 2 and f1.n0[f1.role == ROLE_ENGINEER, ENG].sum() > 0


def test_brigade_engine_runs_deterministically_and_force_ratio_matters():
    t = synthetic_brigade_terrain(seed=1)
    a = BrigadeSim(t, NATIONS["US"], NATIONS["RU"], "two_up", "forward").run(6, seed=3, n_log=1)
    b = BrigadeSim(t, NATIONS["US"], NATIONS["RU"], "two_up", "forward").run(6, seed=3, n_log=1)
    assert np.array_equal(a.win, b.win) and np.array_equal(a.att_losses, b.att_losses)
    assert a.tracks and a.tracks[0][1].shape[1] == len(a.att_names)
    assert set(np.unique(a.cause)) <= {0, 1, 2} and np.all((a.cause == 2) == (a.win == 1))
    strong = BrigadeSim(t, NATIONS["US"], NATIONS["RU"], "two_up", "forward", att_bdes=3).run(12, seed=3, n_log=0)
    weak = BrigadeSim(t, NATIONS["US"], NATIONS["RU"], "two_up", "forward", att_strength=0.5).run(12, seed=3, n_log=0)
    assert strong.win.mean() > weak.win.mean()
    assert a.obstacles.sum() > 0 and a.sorties[0].mean() >= 0 and a.ammo_left[0].mean() >= 0


def test_enablers_have_effects():
    t = synthetic_brigade_terrain(seed=1)
    base = BrigadeSim(t, NATIONS["US"], NATIONS["RU"], "two_up", "forward", att_bdes=2)
    r = base.run(10, seed=5, n_log=0)
    assert r.att_losses[:, HELO].mean() <= 2 and (r.def_losses[:, ENG].sum() >= 0)
    # a defender with no engineers lays no obstacles
    from sim.brigade import Force
    D = build_force(NATIONS["RU"], 1)
    D.n0[D.role == ROLE_ENGINEER] = 0
    sim = BrigadeSim(t, NATIONS["US"], NATIONS["RU"], forces=(build_force(NATIONS["US"], 0, 2), D))
    assert sim.plans[1]["obstacles"].sum() == 0


def test_correction_loaded_and_rasterize_coarse():
    att, dfn = load_correction()
    assert 0.2 < att < 5 and 0.2 < dfn < 5
    n = 10
    t = BrigadeTerrain(41.0, -74.0, 1500.0, 300.0, np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n), np.int8), "test")
    lon, lat = t.xy_to_lonlat(np.array([-1400, -400, -400, -1400, -1400]), np.array([-1400, -1400, -400, -400, -1400]))
    poly = [{"lon": a, "lat": b} for a, b in zip(lon, lat)]
    counts = rasterize_coarse(t, [{"type": "way", "tags": {"natural": "wood"}, "geometry": poly}])
    assert counts["forest"] == 1 and t.at(t.cover, -900, -900) > 0.9 and t.at(t.cls, -900, -900) == 2


def test_all_coa_pairs_and_nation_pairs_run():
    t = synthetic_brigade_terrain(seed=2, half=12000, cell=400)
    for ac in ATTACK_COAS:
        for dc in DEFEND_COAS:
            r = BrigadeSim(t, NATIONS["CN"], NATIONS["IR"], ac, dc).run(2, seed=1, n_log=0)
            assert r.n == 2
    for code in NATIONS:
        r = BrigadeSim(t, NATIONS[code], NATIONS["US"], "penetration", "depth").run(2, seed=1, n_log=0)
        assert r.att_n0.sum() > 0


def test_brigade_terrain_from_entity_terrain():
    t = synthetic_terrain(half=1000.0, cell=25.0, seed=1)
    bt = brigade_terrain_from(t)
    assert bt.cover.max() == 1.0 and bt.exposure(0.0, 0.0) <= 1.0 and bt.ground.shape == t.ground.shape


def test_zero_sum_solver_and_bootstrap():
    v, x, y = solve_zero_sum([[1, -1], [-1, 1]])
    assert abs(v) < 1e-9 and np.allclose(x, 0.5) and np.allclose(y, 0.5)
    v, x, y = solve_zero_sum([[3, 1], [0, 2]])
    assert abs(v - 1.5) < 1e-9 and np.allclose(x, [0.5, 0.5]) and np.allclose(y, [0.25, 0.75])
    rng = np.random.default_rng(0)
    cells = {(i, j): (rng.random(200) < p).astype(float) for (i, j), p in
             {(0, 0): 0.6, (0, 1): 0.3, (1, 0): 0.2, (1, 1): 0.7}.items()}
    v0, lo, hi, *_ = bootstrap_game_value(cells, B=100)
    assert lo <= v0 <= hi and 0.3 < v0 < 0.6


def test_c2_ideal_matches_static_limit_and_doctrine_runs():
    from sim.c2 import IDEAL, PROFILES, latency
    rng = np.random.default_rng(0)
    assert latency(IDEAL, rng) == 0.0 and latency(PROFILES["RU"], rng) > 0
    t = synthetic_brigade_terrain(seed=1)
    ideal = BrigadeSim(t, NATIONS["US"], NATIONS["RU"], "two_up", "mobile", att_bdes=2, c2="ideal")
    doc = BrigadeSim(t, NATIONS["US"], NATIONS["RU"], "two_up", "mobile", att_bdes=2, c2="doctrine")
    assert all(p.tau == 0 and p.p_init == 1 for p in ideal.profiles.values())
    r = doc.run(4, seed=2, n_log=2)
    kinds = {e[2] for rep in r.c2_events for e in rep}
    assert kinds <= {"reserve", "fires_shift", "ammo_pause", "bypass", "fratricide"}
    assert r.sectors[0] and all(lo < hi for lo, hi in r.sectors[0].values())


def test_iran_has_two_lineages_and_hq_company():
    from sim.brigade import ROLE_HQ
    f = build_force(NATIONS["IR"], 0)
    assert {"IR", "IRGC"} == set(f.lineage) and (f.role == ROLE_HQ).sum() == 1
    g = build_force(NATIONS["US"], 0, copies=2)
    assert (g.role == ROLE_HQ).sum() == 2 and set(g.lineage) == {"US"}


def test_c2_profile_override_changes_behaviour():
    from sim.c2 import PROFILES
    t = synthetic_brigade_terrain(seed=1)
    slow = {"US": PROFILES["US"].scaled(tau_mult=10.0, p_init=0.05, sync=0.3)}
    a = BrigadeSim(t, NATIONS["US"], NATIONS["RU"], "two_up", "mobile", att_bdes=2, c2_profiles=slow).run(3, seed=4, n_log=3)
    b = BrigadeSim(t, NATIONS["US"], NATIONS["RU"], "two_up", "mobile", att_bdes=2).run(3, seed=4, n_log=3)
    da = [e[4] - e[0] for rep in a.c2_events for e in rep if e[2] == "reserve" and e[3] == "observed"]
    db = [e[4] - e[0] for rep in b.c2_events for e in rep if e[2] == "reserve" and e[3] == "observed"]
    if da and db:
        assert np.mean(da) > np.mean(db)          # ten times the latency
