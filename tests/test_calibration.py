"""Terrain descriptors, DOE/metamodel, and history-matching tests (kept small to run fast)."""
import numpy as np

from sim.calibrate import SyntheticReference, coverage_report, history_match, stability_report
from sim.descriptors import TerrainLibrary, describe, select_terrains, synthetic_design
from sim.doe import design, fit_metamodel, sobol_indices
from sim.terrain import synthetic_terrain
from sim.units import default_params
from sim.vignettes import FREE, ParamSpec, apply_theta, metrics_of, run_vignette, rung_vignettes


def test_descriptors_track_terrain():
    open_t = synthetic_terrain(half=1200, building_density=0.0005, forest_share=0.02, relief=5, seed=1)
    closed = synthetic_terrain(half=1200, building_density=0.02, forest_share=0.3, relief=80, seed=1)
    d_open, d_closed = describe(open_t), describe(closed)
    assert d_open.openness_500 > d_closed.openness_500
    assert d_closed.forest_share > d_open.forest_share and d_closed.relief > d_open.relief


def test_design_and_selection_split():
    design_pts = synthetic_design(8, seed=0)
    assert len(design_pts) == 8 and all(0.0005 <= d["building_density"] <= 0.02 for d in design_pts)
    vecs = [np.r_[d["building_density"], d["forest_share"], d["relief"], d["road_density"]] for d in design_pts]
    train, hold = select_terrains(vecs, 6, 0.2, seed=0)
    assert len(hold) >= 1 and not set(train) & set(hold)


def test_terrain_library_caches(tmp_path):
    lib = TerrainLibrary(str(tmp_path))
    spec = dict(kind="synthetic", seed=0, half=800.0, cell=25.0, building_density=0.005, forest_share=0.1, relief=20.0)
    t1, d1 = lib.get(spec)
    t2, d2 = lib.get(spec)
    assert np.array_equal(t1.surface, t2.surface) and d1 == d2 and len(lib.entries()) == 1


def test_param_spec_roundtrip_and_apply():
    s = ParamSpec("k_det", 0.4, 2.5, True)
    assert abs(s.value_to_unit(s.unit_to_value(0.3)) - 0.3) < 1e-9
    P = apply_theta(default_params(), FREE[1], [3e-4, 5e-4, 0.66, 0.44])
    assert P["theta0"][0] == 3e-4 and P["PK"][0, 4] == 0.66 and P["PK"][4, 0] == 0.44
    assert default_params()["PK"][0, 4] != 0.66          # apply_theta copies


def test_vignette_metrics_shape():
    t = synthetic_terrain(half=1500.0, seed=0)
    r = run_vignette(t, rung_vignettes(2)[0], default_params(), 90, seed=1)
    m = metrics_of(r)
    assert 0 <= m["p_win"] <= 1 and np.isfinite(m["blue_loss_frac"]) and r.n_plans == 3


def test_history_match_keeps_truth_and_shrinks():
    t = synthetic_terrain(half=1500.0, seed=0)
    specs = FREE[1]
    truth = [2.5e-4, 5e-4, 0.7, 0.5]
    ref = SyntheticReference(apply_theta(default_params(), specs, truth), n=600, seed=9)
    waves = history_match(1, specs, default_params(), {"t": t}, rung_vignettes(1), ref,
                          n_reps=150, m=16, waves=2, seed=2, log=lambda *_: None)
    box = waves[-1].nroy_box
    assert all(lo <= v <= hi for (lo, hi), v in zip(box, truth))
    unit_vol = np.prod([(s.value_to_unit(hi) - s.value_to_unit(lo)) for s, (lo, hi) in zip(specs, box)])
    assert unit_vol < 0.9
    rows = stability_report(waves)
    assert len(rows) == len(specs)


def test_metamodel_and_sobol_on_known_function():
    specs = [ParamSpec("a", 0, 1), ParamSpec("b", 0, 1), ParamSpec("c", 0, 1)]
    U, X = design(specs, 60, seed=1)
    y = 3 * U[:, 0] + 0.1 * U[:, 1] + 0.0 * U[:, 2]       # only 'a' matters
    mm = fit_metamodel(U, y, np.full(len(y), 1e-6))
    si = sobol_indices(mm, 3, N=1024)
    assert si["ST"][0] > 0.9 and si["ST"][2] < 0.05
