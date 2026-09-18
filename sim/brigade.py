"""Aggregate brigade-on-brigade engine.

Resolution: the entity is a company or battery (a vector of platform counts by equipment class),
the tick is one minute, the grid is coarse (300 m cells over a 30 km box). It is a mean-field
version of the entity engine: every kill rate is computed from the same kernel (detection hazard,
circular-normal P_H on the visible fraction, P_K from the nation catalog, rate of fire), summed
over the platforms in a company and thinned with Poisson draws. That is what ties the two levels
together, and scripts/aggregation_check.py measures how well the tie holds.

Plans are templates: the attacker picks an axis scheme (two-up, penetration, envelopment) and
the defender a scheme (forward, depth, mobile); each maneuver battalion gets waypoints, batteries
get gun lines, recon gets a screen. Companies follow Dijkstra routes on the coarse cost grid,
detect through their side's network, allocate fire in proportion to expected value (the λ = 1
logit), take Poisson-thinned kills, get suppressed, and break on a Beta breakpoint. The fight
ends when the attacker seizes the objective, culminates, or the clock runs out.

All numbers in sim/nations.py are synthetic; see that module's docstring.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .engine import hit_prob
from .c2 import CROSS_LINEAGE_SYNC, GRAPHS, IDEAL, PROFILES, C2Profile, latency, lineages_for
from .nations import CLASS_PK, CLASSES, CLASS_INDEX, DIRECT_FIRE, INDIRECT, Nation, pk_matrix
from .terrain import Terrain, los_fraction

NC = len(CLASSES)
TANK, IFV, INF, ATGM, RECON, HOW, ROCK, ENG, SUP, HELO = (CLASS_INDEX[c] for c in CLASSES)
MANEUVER = np.array([True, True, True, True, False, False, False, False, False, False])   # counts toward breakpoints
IND_VULN = np.array([0.12, 0.35, 1.0, 1.0, 0.8, 0.3, 0.3, 0.35, 0.9, 0.0])              # artillery lethality by class
ATTACK_COAS = ["two_up", "penetration", "envelopment"]
DEFEND_COAS = ["forward", "depth", "mobile"]
STREAMS = ["det_a", "det_d", "kill_a", "kill_d", "ind_a", "ind_d", "supp", "bp", "place", "eng_a", "eng_d",
           "avn_a", "avn_d", "c2_a", "c2_d"]
ROLE_MANEUVER, ROLE_RECON, ROLE_BATTERY, ROLE_ENGINEER, ROLE_SUPPORT, ROLE_AVIATION, ROLE_HQ = range(7)
CROWD_K = 2                      # companies per cell above which crowding penalties apply
FRATRICIDE_RATE = 0.003          # per platform per minute out of sector in contact, times (1 - sync)
PAUSE_MIN = 20.0                 # tactical pause when the fires stock falls below the graph's threshold
PURSUIT_MIN = 30.0               # exploitation sequel after the objective is seized
ROUNDS_PER_TUBE = 100.0          # initial indirect-fire stock per tube
RESUPPLY_PER_TUBE_HOUR = 20.0    # rounds per tube per hour at full support strength
BREACH_MIN = (40.0, 12.0)        # minutes to breach without / with an engineer company within 1500 m
MINE_KILL_FRAC = 0.04            # expected fraction of a company's vehicles lost on first contact with a minefield
SORTIE_MIN, TURNAROUND_MIN = 20.0, 60.0
HELO_LOSS_PER_MIN = 0.012        # per helicopter per minute on station, times enemy shorad


# ------------------------------------------------------------------ coarse ground
class BrigadeTerrain(Terrain):
    """A Terrain (ground + cls for routing) plus continuous cover and urban fractions per cell.
    LOS is cast on the bare ground; cover reduces exposure instead of blocking rays."""

    def __init__(self, *args, cover=None, urban=None, **kw):
        super().__init__(*args, **kw)
        self.cover = cover if cover is not None else np.zeros_like(self.ground)
        self.urban = urban if urban is not None else np.zeros_like(self.ground)
        self._los = {}

    def exposure(self, x, y):
        return 1.0 - 0.6 * self.at(self.cover, x, y) - 0.2 * self.at(self.urban, x, y)


def brigade_terrain_from(t: Terrain):
    """Wrap a fine entity-engine Terrain: cover from forest/building cells, LOS on bare ground."""
    cover = ((t.cls == 2) | (t.cls == 3)).astype(float)
    urban = (t.cls == 3).astype(float)
    bt = BrigadeTerrain(t.lat0, t.lon0, t.half, t.cell, t.ground.copy(), t.ground.copy(), t.cls.copy(),
                        t.source, t.name, cover=cover, urban=urban)
    return bt


def synthetic_brigade_terrain(half=15000.0, cell=300.0, seed=0, relief=120.0, cover=0.2, urban=0.05,
                              river=True, name=None):
    """Rolling ground with ridges across the axis of advance, wood and village patches, a river
    with bridges, and a road net. cover/urban are target fractions of the area."""
    rng = np.random.default_rng(seed)
    n = int(round(2 * half / cell))
    t = BrigadeTerrain(0.0, 0.0, half, cell, np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n), np.int8),
                       "Synthetic brigade", name or f"Synthetic brigade ground (relief {relief:.0f} m, cover {cover:.2f})")
    x, y = t.cell_centres()
    g = 200 + relief * 0.5 * np.sin(x / 5000.0) * np.cos(y / 7000.0)
    g += relief * gaussian_filter(rng.normal(size=(n, n)), sigma=6) / 0.12
    t.ground = g
    field_c = gaussian_filter(rng.normal(size=(n, n)), sigma=3)
    t.cover = (field_c > np.quantile(field_c, 1 - cover)).astype(float) * rng.uniform(0.6, 1.0, (n, n))
    field_u = gaussian_filter(rng.normal(size=(n, n)), sigma=2)
    t.urban = (field_u > np.quantile(field_u, 1 - urban)).astype(float)
    cls = np.zeros((n, n), np.int8)
    cls[t.cover > 0.5] = 2
    cls[t.urban > 0.5] = 3
    if river:
        rx = 1500 + 1200 * np.sin(y[:, 0] / 4000.0)
        water = np.abs(x - rx[:, None]) < cell * 0.6
        cls[water] = 4
        for by in (-9000, -3000, 3000, 9000):
            i, j = t.xy_to_ij(rx[np.argmin(np.abs(y[:, 0] - by))], by)
            cls[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2] = 1
    for yy in (-6000, 0, 6000):
        i, j = t.xy_to_ij(x[0], np.full(n, yy))
        cls[i, j] = np.where(cls[i, j] == 4, 1, 1)
    t.cls = cls
    t.surface = g.copy()
    t.notes.append("Synthetic brigade ground; buildings and woods are cover fractions, not obstacles.")
    return t


def rasterize_coarse(t: BrigadeTerrain, elements, fine_cell=100.0):
    """Burn coarse OSM features (landuse/natural polygons, waterways, major roads) into cover, urban,
    water and road fractions per coarse cell by rasterizing on a finer temporary grid and block-averaging.
    Buildings are not fetched at this scale; residential/industrial/commercial landuse stands in for them."""
    from .terrain import _fill, _stroke, blank_terrain
    f = blank_terrain(t.lat0, t.lon0, t.half, fine_cell, "tmp")
    n = f.n
    forest, urban, water, road = (np.zeros((n, n), np.int8) for _ in range(4))
    dummy = np.zeros((n, n))
    counts = dict(forest=0, urban=0, water=0, road=0)
    for el in elements:
        if el.get("type") != "way" or not el.get("geometry"):
            continue
        tags = el.get("tags", {})
        lon = np.array([p["lon"] for p in el["geometry"]])
        lat = np.array([p["lat"] for p in el["geometry"]])
        x, y = f.lonlat_to_xy(lon, lat)
        xy = np.c_[x, y]
        lu, nat, ww, hw = tags.get("landuse"), tags.get("natural"), tags.get("waterway"), tags.get("highway")
        if nat in ("wood", "scrub") or lu in ("forest", "orchard"):
            _fill(f, forest, dummy, xy, 1, 0.0); counts["forest"] += 1
        elif lu in ("residential", "industrial", "commercial", "retail") or tags.get("place") in ("town", "village", "city"):
            _fill(f, urban, dummy, xy, 1, 0.0); counts["urban"] += 1
        elif nat == "water" or lu in ("reservoir", "basin") or ww == "riverbank":
            _fill(f, water, dummy, xy, 1, 0.0); counts["water"] += 1
        elif ww in ("river", "canal"):
            _stroke(f, water, dummy, xy, 1, 0.0, 100.0); counts["water"] += 1
        elif hw in ("motorway", "trunk", "primary", "secondary", "tertiary"):
            _stroke(f, road, dummy, xy, 1, 0.0, 30.0); counts["road"] += 1
    k = int(round(t.cell / fine_cell))
    m = (n // k) * k

    def block(a):
        b = a[:m, :m].astype(float).reshape(m // k, k, m // k, k).mean((1, 3))
        out = np.zeros((t.n, t.n))
        h = min(t.n, b.shape[0])
        out[:h, :h] = b[:h, :h]
        return out
    t.cover = block(forest)
    t.urban = block(urban)
    wat, rd = block(water), block(road)
    cls = np.zeros((t.n, t.n), np.int8)
    cls[t.cover > 0.5] = 2
    cls[t.urban > 0.5] = 3
    cls[wat > 0.25] = 4
    cls[rd > 0] = np.where(cls[rd > 0] == 4, 1, 1)         # roads cross water: bridges
    t.cls = cls
    t.surface = t.ground.copy()
    return counts


def fetch_osm_coarse(t, timeout=120):
    from .terrain import overpass_query
    w, s_, e, n_ = t.bounds_lonlat()
    bb = f"{s_},{w},{n_},{e}"
    q = f"""[out:json][timeout:90];
(way["natural"~"^(wood|scrub|water)$"]({bb});
 way["landuse"~"^(forest|orchard|residential|industrial|commercial|retail|reservoir|basin)$"]({bb});
 way["waterway"~"^(riverbank|river|canal)$"]({bb});
 way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]({bb}););
out tags geom;"""
    return overpass_query(q, timeout)


def osm_brigade_terrain(lat0, lon0, half=15000.0, cell=300.0, name=""):
    """Real ground for the brigade engine: AWS Terrain Tiles at zoom 11 for elevation, coarse OSM
    landuse/water/major roads for cover, urban, water and road fractions."""
    import datetime as _dt
    from .terrain import fetch_dem
    n = int(round(2 * half / cell))
    t = BrigadeTerrain(lat0, lon0, half, cell, np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n), np.int8),
                       "OpenStreetMap", name or f"{lat0:.3f}, {lon0:.3f}")
    try:
        t.ground = fetch_dem(t, z=11)
        t.notes.append("Elevation: AWS Terrain Tiles (zoom 11).")
    except Exception as ex:
        t.notes.append(f"Elevation unavailable ({ex}); ground treated as flat.")
    counts = rasterize_coarse(t, fetch_osm_coarse(t))
    t.notes.append("OSM ways: " + ", ".join(f"{v} {k}" for k, v in counts.items() if v))
    t.fetched_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    return t


VEH_COST = np.array([1.0, 0.7, 3.0, 2.0, np.inf])
VEH_SPEED = np.array([1.0, 1.3, 0.4, 0.5, 0.0])


def _grid_graph(cost, cell):
    n = cost.shape[0]
    idx = np.arange(n * n).reshape(n, n)
    I, J = np.mgrid[0:n, 0:n]
    rows, cols, w = [], [], []
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        I2, J2 = I + di, J + dj
        ok = (I2 >= 0) & (I2 < n) & (J2 >= 0) & (J2 < n)
        a, b = idx[I[ok], J[ok]], idx[I2[ok], J2[ok]]
        c = 0.5 * (cost.ravel()[a] + cost.ravel()[b]) * cell * np.hypot(di, dj)
        fin = np.isfinite(c)
        rows += [a[fin], b[fin]]
        cols += [b[fin], a[fin]]
        w += [c[fin], c[fin]]
    rows, cols, w = (np.concatenate(v) for v in (rows, cols, w))
    return coo_matrix((w, (rows, cols)), shape=(n * n, n * n)).tocsr()


# ------------------------------------------------------------------ aggregation correction
CORRECTION_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
                               "aggregation_correction.json")


def load_correction(path=CORRECTION_PATH):
    """(attacker, defender) direct-fire kill-rate multipliers from the rung-5 fit; (1, 1) if none."""
    try:
        d = json.load(open(path))
        return float(d["kill_scale_att"]), float(d["kill_scale_def"])
    except Exception:
        return 1.0, 1.0


# ------------------------------------------------------------------ force structure
@dataclass
class Force:
    nation: Nation
    side: int                      # 0 attacker, 1 defender
    names: list = field(default_factory=list)
    bn: np.ndarray = None          # battalion id per company (-1 for brigade assets)
    role: np.ndarray = None        # 0 maneuver, 1 recon, 2 battery, 3 engineer, 4 support, 5 aviation, 6 HQ
    n0: np.ndarray = None          # (N, NC) initial platform counts
    P: dict = None                 # per-class attribute arrays
    lineage: list = None           # C2 lineage code per company ("US", "RU", "CN", "IR", "IRGC")

    @property
    def N(self):
        return len(self.names)


def build_force(nation: Nation, side: int, copies: int = 1, strength: float = 1.0):
    """One or more brigades (copies) of a nation's template as company-level entities.
    strength scales platform counts (0.7 = a brigade at 70% strength)."""
    b = nation.brigade
    names, bn, role, rows = [], [], [], []
    rd = b["readiness"] * strength
    eq = nation.equipment

    def co(name, k, counts, r):
        names.append(name)
        bn.append(k)
        role.append(r)
        v = np.zeros(NC)
        for c, m in counts.items():
            v[CLASS_INDEX[c]] = round(m * rd)
        rows.append(v)

    bns = b.get("battalions") or [dict(tank=b["tank_cos_per_bn"], ifv=b["ifv_cos_per_bn"])] * b["maneuver_bns"]
    for c in range(copies):
        pre = f"{c + 1}-" if copies > 1 else ""
        for kk, comp in enumerate(bns):
            k = c * len(bns) + kk
            for i in range(comp["tank"]):
                co(f"T{pre}{kk + 1}/{i + 1}", k, dict(tank=eq["tank"].per_company), ROLE_MANEUVER)
            for i in range(comp["ifv"]):
                co(f"M{pre}{kk + 1}/{i + 1}", k, dict(ifv=eq["ifv"].per_company, inf=b["inf_platoons_per_ifv_co"],
                                                       atgm=b["atgm_teams_per_bn"] / comp["ifv"]), ROLE_MANEUVER)
        for i in range(b.get("at_cos", 0)):                   # anti-tank battalion companies (brigade assets)
            co(f"AT{pre}{i + 1}", -1, dict(atgm=eq["atgm"].per_company), ROLE_MANEUVER)
        for i in range(b["recon_cos"]):                       # cavalry / reconnaissance troops
            co(f"R{pre}{i + 1}", -1, dict(recon=eq["recon"].per_company, ifv=b.get("cav_ifv_per_troop", 0)), ROLE_RECON)
        for i in range(b["howitzer_batteries"]):              # fires battalion
            co(f"H{pre}{i + 1}", -1, dict(howitzer=eq["howitzer"].per_company), ROLE_BATTERY)
        for i in range(b["rocket_batteries"]):
            co(f"K{pre}{i + 1}", -1, dict(rocket=eq["rocket"].per_company), ROLE_BATTERY)
        for i in range(b.get("engineer_cos", 0)):             # engineer battalion
            co(f"E{pre}{i + 1}", -1, dict(engineer=eq["engineer"].per_company), ROLE_ENGINEER)
        for i in range(b.get("support_cos", 0)):              # support battalion
            co(f"S{pre}{i + 1}", -1, dict(support=eq["support"].per_company), ROLE_SUPPORT)
        for i in range(b.get("aviation_teams", 0)):           # attack weapons team
            co(f"A{pre}{i + 1}", -1, dict(attack_helo=eq["attack_helo"].per_company), ROLE_AVIATION)
        co(f"HQ{pre}", -1, dict(support=6), ROLE_HQ)              # brigade command post (soft, high value)
    P = force_params(nation)
    bn_arr = np.array(bn)
    n_bns = len(bns)
    per_bn = lineages_for(nation.code, n_bns)
    lineage = [per_bn[k % n_bns] if k >= 0 else per_bn[0] for k in bn_arr]
    return Force(nation, side, names, bn_arr, np.array(role), np.array(rows), P, lineage)


def force_params(nation):
    eq = nation.equipment
    P = {key: np.array([getattr(eq[c], key) for c in CLASSES], float)
         for key in ("area", "theta0", "theta_s", "theta_t", "sensor", "signature", "fire_rate", "p_supp",
                     "supp_vuln", "value", "speed", "ind_lethality", "rounds_per_min", "rmax", "ammo")}
    return P


# ------------------------------------------------------------------ plans
def _waypoints(coa, side, bn_count, half):
    """Per battalion: list of waypoints (x, y) in metres; attacker moves west->east."""
    xa, xf, xd, xo = -half + 4000, 2000.0, 5500.0, 6500.0      # assembly, forward line, depth line, objective
    if side == 0:
        if coa == "two_up":
            axes = [3000.0, -3000.0, 0.0]
            wps = [[(xa + 3000, axes[0]), (xo - 1500, axes[0]), (xo, 0.0)],
                   [(xa + 3000, axes[1]), (xo - 1500, axes[1]), (xo, 0.0)],
                   [(xa, 0.0), (xo - 4500, 0.0), (xo, 0.0)]]
        elif coa == "penetration":
            wps = [[(xa + 3000, 0.0), (xo, 0.0)],
                   [(xa, 0.0), (xo, 0.0)],
                   [(xa - 3000, 0.0), (xo, 0.0)]]
        else:  # envelopment: bn 1 fixes short of the line, bns 2-3 swing north
            wps = [[(xa + 3000, 0.0), (xf - 3500, 0.0)],
                   [(xa + 3000, 6000.0), (xf - 1000, 9500.0), (xo, 6000.0), (xo, 0.0)],
                   [(xa, 6000.0), (xf - 3000, 9500.0), (xo - 2000, 6000.0), (xo, 0.0)]]
        while len(wps) < bn_count:
            k = len(wps)
            wps.append([(xa - 3000 * (k - 2), 4000.0 * (-1) ** k), (xo - 1500, 2000.0 * (-1) ** k), (xo, 0.0)])
        return wps[:bn_count]
    if coa == "forward":
        ys = [-5000.0, 0.0, 5000.0, 0.0]
        xs = [xf, xf, xf, xd]
    elif coa == "depth":
        ys = [-3000.0, 3000.0, 0.0, 0.0]
        xs = [xf, xf, xd, xd + 1500]
    else:  # mobile: two forward, reserve back and ready to counterattack
        ys = [-3000.0, 3000.0, 0.0, 0.0]
        xs = [xf, xf, xd + 2500, xd + 3500]
    return [[(xs[k % 4], ys[k % 4])] for k in range(bn_count)]


def _company_offsets(k, spread=1200.0):
    """Spread k companies of a battalion in a 2-row box."""
    cols = int(np.ceil(k / 2))
    out = []
    for i in range(k):
        r, c = divmod(i, cols)
        out.append(((c - (cols - 1) / 2) * spread, (0.5 - r) * spread))
    return np.array(out).reshape(-1, 2)


# ------------------------------------------------------------------ results
@dataclass
class BrigadeResult:
    win: np.ndarray                # objective seized before culmination
    cause: np.ndarray              # 0 censored, 1 attacker culminated, 2 objective seized
    minutes: np.ndarray
    att_losses: np.ndarray         # (n, NC)
    def_losses: np.ndarray         # (n, NC)
    att_strength: np.ndarray       # (n, T+1) maneuver platforms alive
    def_strength: np.ndarray
    att_n0: np.ndarray             # (NC,)
    def_n0: np.ndarray
    tracks: list                   # per logged rep: (minutes, att_xy (T,Na,2), def_xy, att_alive, def_alive)
    att_names: list
    def_names: list
    coa: tuple
    tick_min: float = 1.0
    sorties: tuple = None          # per-replication attack-aviation sorties (attacker, defender)
    ammo_left: tuple = None        # indirect-fire rounds remaining at the end (attacker, defender)
    obstacles: np.ndarray = None   # defender obstacle cells
    c2_events: list = None         # per logged replication: (minute, side, trigger, stage, detail)
    sectors: tuple = None          # battalion sector y-bands per side

    @property
    def n(self):
        return len(self.win)

    def att_loss_frac(self):
        m = MANEUVER
        return (self.att_losses[:, m].sum(1)) / max(self.att_n0[m].sum(), 1)

    def def_loss_frac(self):
        m = MANEUVER
        return (self.def_losses[:, m].sum(1)) / max(self.def_n0[m].sum(), 1)


# ------------------------------------------------------------------ the engine
class BrigadeSim:
    def __init__(self, terrain: BrigadeTerrain, attacker: Nation, defender: Nation, att_coa="two_up",
                 def_coa="forward", max_minutes=480.0, objective=(6500.0, 0.0), att_bdes=1, def_bdes=1,
                 att_strength=1.0, def_strength=1.0, forces=None, plans=None, objective_companies=3,
                 objective_radius=1500.0, kill_scale=None, kill_draw="poisson", c2="doctrine", c2_profiles=None):
        """forces / plans override the templates: forces = (Force, Force), plans = {0: dict(pos, wps),
        1: dict(pos, wps)} with pos (N, 2) and wps a list of waypoint lists. Used by the aggregation
        check to run the aggregate engine on a company fight."""
        self.t = terrain
        if forces is None:
            self.A = build_force(attacker, 0, att_bdes, att_strength)
            self.D = build_force(defender, 1, def_bdes, def_strength)
        else:
            self.A, self.D = forces
        self.att_coa, self.def_coa = att_coa, def_coa
        self.max_minutes = max_minutes
        self.objective = np.array(objective, float)
        self.objective_companies, self.objective_radius = objective_companies, objective_radius
        # aggregation correction: multipliers on (attacker->defender, defender->attacker) direct-fire kill
        # rates, fitted by scripts/rung5_calibrate.py against the entity engine; None loads the shipped fit
        self.kill_scale = tuple(kill_scale) if kill_scale is not None else load_correction()
        self.kill_draw = kill_draw
        # C2: "doctrine" uses the nation profiles (Iran splits into Artesh and IRGC lineages); "ideal" is the
        # zero-latency, full-initiative limit that reproduces the static templates. c2_profiles overrides
        # individual lineage codes -> C2Profile (used by the sensitivity study).
        self.c2 = c2
        prof = dict(PROFILES)
        if c2_profiles:
            prof.update(c2_profiles)
        self.profiles = {code: (IDEAL if c2 == "ideal" else prof[code]) for code in prof}
        self.graphs = {code: GRAPHS[code] for code in GRAPHS}
        if self.A.lineage is None:
            self.A.lineage = [attacker.code] * self.A.N
        if self.D.lineage is None:
            self.D.lineage = [defender.code] * self.D.N
        self.cost = VEH_COST[terrain.cls]
        self.G = _grid_graph(self.cost, terrain.cell)
        self.cx, self.cy = (a.ravel() for a in terrain.cell_centres())
        self._pred = {}
        self.PK_ad, self.RM_ad = pk_matrix(attacker, CLASSES, defender, CLASSES)
        self.PK_da, self.RM_da = pk_matrix(defender, CLASSES, attacker, CLASSES)
        self.plans = plans or {0: self._plan(self.A, att_coa), 1: self._plan(self.D, def_coa)}
        for side in (0, 1):
            self.plans[side].setdefault("obstacles", np.zeros((terrain.n, terrain.n), bool))
            self.plans[side].setdefault("sectors", {0: (-1e9, 1e9)})
            self.plans[side].setdefault("axes", {0: 0.0})

    # ---- geometry
    def _snap(self, x, y):
        d = np.where(np.isfinite(self.cost.ravel()), np.hypot(self.cx - x, self.cy - y), np.inf)
        return int(np.argmin(d))

    def _route(self, src, dst):
        if dst not in self._pred:
            _, pred = dijkstra(self.G, indices=dst, return_predecessors=True)
            self._pred[dst] = pred
        pred, path, u = self._pred[dst], [], src
        while u != dst and u >= 0:
            path.append(u)
            u = pred[u]
        path.append(dst)
        return np.array(path)

    def _plan(self, F: Force, coa):
        """Initial positions and waypoint lists per company."""
        half = self.t.half
        bn_count = int(F.bn.max()) + 1
        wps = _waypoints(coa, F.side, bn_count, half)
        pos, wp = np.zeros((F.N, 2)), {}                # waypoints keyed by company index
        for k in range(bn_count):
            idx = np.flatnonzero(F.bn == k)
            off = _company_offsets(len(idx))
            start = np.array(wps[k][0])
            for q, i in enumerate(idx):
                pos[i] = start + off[q]
                wp[i] = [np.array(w) + off[q] * 0.6 for w in wps[k][1:]] if F.side == 0 else []
                if F.side == 1:
                    pos[i] = start + off[q] * np.array([0.8, 1.6])
        at = np.flatnonzero((F.bn < 0) & (F.role == ROLE_MANEUVER))
        for q, i in enumerate(at):                      # anti-tank battalion: brigade asset
            y = (q - (len(at) - 1) / 2) * 3000.0
            if F.side == 0:                             # follows the lead battalion as a fixing element
                pos[i] = np.array(wps[0][0]) + np.array([-800.0, y])
                wp[i] = [np.array(w) + np.array([-800.0, y * 0.5]) for w in wps[0][1:]]
            else:                                       # defender: on the forward line between sectors
                pos[i] = np.array([wps[0][0][0] + 400.0, y])
                wp[i] = []
        rec = np.flatnonzero(F.role == ROLE_RECON)
        for q, i in enumerate(rec):
            y = (q - (len(rec) - 1) / 2) * 4000.0
            if F.side == 0:
                pos[i] = np.array([-half + 8000, y])
                wp[i] = [np.array([1000.0, y]), np.array([5500.0, y * 0.5])]
            else:
                pos[i] = np.array([-3000.0, y])
                wp[i] = []
        bat = np.flatnonzero(F.role == ROLE_BATTERY)
        for q, i in enumerate(bat):
            y = (q - (len(bat) - 1) / 2) * 2500.0
            pos[i] = np.array([-half + 5000 - 1500 * (F.n0[i, ROCK] > 0), y]) if F.side == 0 \
                else np.array([9500.0 + 2000 * (F.n0[i, ROCK] > 0), y])
            wp[i] = []
        eng = np.flatnonzero(F.role == ROLE_ENGINEER)
        lead = [k for k in range(bn_count)][:max(1, len(eng))]
        for q, i in enumerate(eng):                     # attacker engineers travel with the lead battalions
            k = lead[q % len(lead)]
            if F.side == 0:
                pos[i] = np.array(wps[k][0]) + np.array([-1200.0, 0.0])
                wp[i] = [np.array(w) for w in wps[k][1:]]
            else:                                       # defender engineers sit behind the forward line
                pos[i] = np.array([wps[k][0][0] + 1500.0, wps[k][0][1]])
                wp[i] = []
        sup = np.flatnonzero(F.role == ROLE_SUPPORT)
        for q, i in enumerate(sup):
            y = (q - (len(sup) - 1) / 2) * 2000.0
            pos[i] = np.array([-half + 2500, y]) if F.side == 0 else np.array([half - 2500, y])
            wp[i] = []
        avn = np.flatnonzero(F.role == ROLE_AVIATION)
        for q, i in enumerate(avn):                     # holding area well to the rear; moves per sortie
            pos[i] = np.array([-half + 1500, 0.0]) if F.side == 0 else np.array([half - 1500, 0.0])
            wp[i] = []
        for i in np.flatnonzero(F.role == ROLE_HQ):     # command post: behind the lead battalions / at the depth line
            pos[i] = np.array([-half + 1500, 500.0]) if F.side == 0 else np.array([7000.0, 500.0])
            wp[i] = [np.array([-half + 8000, 500.0])] if F.side == 0 else []
        # battalion sectors: y-bands around each maneuver battalion's axis (attacker: its first waypoint;
        # defender: its position), used for fires clearance, boundary discipline and crowding
        axes = {k: float(wps[k][0][1]) for k in range(bn_count)}
        ys = sorted(set(axes.values()))
        halfwidth = 3000.0 if len(ys) < 2 else max(1500.0, min(4000.0, 0.5 * float(np.min(np.diff(ys)))))
        sectors = {k: (axes[k] - halfwidth, axes[k] + halfwidth) for k in axes}
        waypoints = [wp.get(i, []) for i in range(F.N)]
        # keep everything on passable ground
        for i in range(F.N):
            pos[i] = self._xy(self._snap(*pos[i]))
        obstacles = self._obstacles(F, wps, bn_count) if F.side == 1 else np.zeros((self.t.n, self.t.n), bool)
        return dict(pos=pos, wps=waypoints, obstacles=obstacles, sectors=sectors, axes=axes)

    def _obstacles(self, F: Force, wps, bn_count):
        """Defender obstacle belts (minefield + wire) 700 m in front of each occupied line, lengths
        set by the engineer platforms available (one 300 m cell per 0.7 engineer vehicle)."""
        grid = np.zeros((self.t.n, self.t.n), bool)
        n_eng = float(F.n0[:, ENG].sum())
        cells_total = int(n_eng / 0.7)
        if cells_total == 0:
            return grid
        lines = {}
        for k in range(bn_count):
            x, y = wps[k][0]
            lines.setdefault(round(x), []).append(y)
        per_line = cells_total // max(len(lines), 1)
        for x, ys in lines.items():
            per_bn = per_line // max(len(ys), 1)
            for y in ys:
                span = per_bn * self.t.cell / 2
                yy = np.arange(y - span, y + span, self.t.cell)
                i, j = self.t.xy_to_ij(np.full(len(yy), x - 700.0), yy)
                ok = (i >= 0) & (i < self.t.n) & (j >= 0) & (j < self.t.n)
                grid[i[ok], j[ok]] = True
        return grid

    def _xy(self, node):
        return np.array([self.cx[node], self.cy[node]])

    def _gap_near(self, blocked, cell, max_cells):
        """Nearest passable, unblocked cell in the same column (same x) within max_cells rows."""
        ci, cj = divmod(int(cell), self.t.n)
        for d in range(1, max_cells + 1):
            for cj2 in (cj - d, cj + d):
                if 0 <= cj2 < self.t.n and not blocked[ci, cj2] and np.isfinite(self.cost[ci, cj2]):
                    return ci * self.t.n + cj2
        return None

    # ---- LOS with a persistent cache
    def _vis(self, xa, xd):
        ia, ja = self.t.xy_to_ij(xa[:, 0], xa[:, 1])
        idd, jd = self.t.xy_to_ij(xd[:, 0], xd[:, 1])
        ca, cd = ia * self.t.n + ja, idd * self.t.n + jd
        keys = ca[:, None] * (self.t.n * self.t.n) + cd[None, :]
        D = np.hypot(xa[:, None, 0] - xd[None, :, 0], xa[:, None, 1] - xd[None, :, 1])
        out = np.zeros(keys.shape)
        need = (D <= 5000.0)
        flat = keys[need]
        cached = np.array([self.t._los.get(int(k), np.nan) for k in flat])
        miss = np.isnan(cached)
        if miss.any():
            pa, pd_ = np.nonzero(need)
            pa, pd_ = pa[miss], pd_[miss]
            v = los_fraction(self.t, xa[pa], 2.0, xd[pd_], np.array([[1.0, 2.0, 3.0]]), samples=48)
            for k, val in zip(flat[miss], v):
                self.t._los[int(k)] = float(val)
            cached[miss] = v
        out[need] = cached
        return out, D

    # ---- one replication
    def run(self, n=100, seed=0, n_log=2, log_every=5):
        kids = np.random.SeedSequence(seed).spawn(len(STREAMS))
        rs = {k: np.random.default_rng(c) for k, c in zip(STREAMS, kids)}
        T = int(self.max_minutes)
        A, D, t = self.A, self.D, self.t
        Na, Nd = A.N, D.N
        out = dict(win=np.zeros(n), cause=np.zeros(n, int), minutes=np.zeros(n), att_losses=np.zeros((n, NC)),
                   def_losses=np.zeros((n, NC)), att_strength=np.zeros((n, T + 1)), def_strength=np.zeros((n, T + 1)),
                   att_sorties=np.zeros(n), def_sorties=np.zeros(n), att_ammo=np.zeros(n), def_ammo=np.zeros(n))
        tracks, events = [], []
        for rep in range(n):
            res, track = self._one(rs, T, rep < n_log, log_every)
            ev = res.pop("c2_events")
            if rep < n_log:
                events.append(ev)
            for k, v in res.items():
                out[k][rep] = v
            if track is not None:
                tracks.append(track)
        return BrigadeResult(out["win"], out["cause"], out["minutes"], out["att_losses"], out["def_losses"],
                             out["att_strength"], out["def_strength"], A.n0.sum(0), D.n0.sum(0), tracks,
                             A.names, D.names, (self.att_coa, self.def_coa), sorties=(out["att_sorties"], out["def_sorties"]),
                             ammo_left=(out["att_ammo"], out["def_ammo"]), obstacles=self.plans[1]["obstacles"],
                             c2_events=events, sectors=(self.plans[0]["sectors"], self.plans[1]["sectors"]))

    def _one(self, rs, T, log, log_every):
        A, D, t = self.A, self.D, self.t
        F = [A, D]
        pos = [self.plans[0]["pos"].copy(), self.plans[1]["pos"].copy()]
        wps = [[list(w) for w in self.plans[0]["wps"]], [list(w) for w in self.plans[1]["wps"]]]
        nn = [A.n0.astype(float).copy(), D.n0.astype(float).copy()]
        supp = [np.zeros(A.N), np.zeros(D.N)]
        broken = [np.zeros(A.N, bool), np.zeros(D.N, bool)]
        last_seen = [np.full(A.N, -1e9), np.full(D.N, -1e9)]     # by the enemy
        fired_at = [np.full(A.N, -1e9), np.full(D.N, -1e9)]
        mission_left = [np.zeros(A.N), np.zeros(D.N)]
        cooldown = [np.zeros(A.N), np.zeros(D.N)]
        mission_tgt = [np.full(A.N, -1), np.full(D.N, -1)]
        path = [[None] * A.N, [None] * D.N]
        path_k = [np.zeros(A.N, int), np.zeros(D.N, int)]
        bp_co = [rs["bp"].beta(8, 8, A.N), rs["bp"].beta(8, 8, D.N)]        # company breakpoint, mean 0.5
        bp_bde = rs["bp"].beta(10, 10), rs["bp"].beta(11, 9)                # brigade: attacker 0.5, defender 0.55
        n0m = [(A.n0 * MANEUVER).sum(), (D.n0 * MANEUVER).sum()]
        strength = [np.zeros(T + 1), np.zeros(T + 1)]
        strength[0][0], strength[1][0] = n0m
        track = ([], [], [], [], []) if log else None
        # ---- C2 state: per-side triggers go observed -> issued -> delivered; subordinates may act on intent
        prof = [{code: self.profiles[code] for code in set(F[s].lineage)} for s in range(2)]
        main = [F[s].lineage[0] for s in range(2)]
        hq_idx = [np.flatnonzero(F[s].role == ROLE_HQ) for s in range(2)]
        c2_rng = [rs["c2_a"], rs["c2_d"]]
        triggers = [dict(), dict()]           # name -> dict(observed=None, deliver=None, done=False)
        for s in range(2):
            for name in ("reserve", "fires_shift", "ammo_pause"):
                triggers[s][name] = dict(observed=None, deliver=None, done=False)
        reserve_target = [None, None]         # bn index the reserve was sent to reinforce
        fires_sector = [None, None]           # sector index with priority of fires
        pause_left = [0.0, 0.0]
        pursuit_left = 0.0
        objective_taken = False
        sectors = [self.plans[s]["sectors"] for s in range(2)]
        axes = [self.plans[s]["axes"] for s in range(2)]
        c2_events = []
        committed_flag = [np.zeros(A.N, bool), np.zeros(D.N, bool)]
        man_bns_att = np.unique(A.bn[(A.bn >= 0) & (A.role == ROLE_MANEUVER)])
        lead_bns_att = set(man_bns_att[:2]) if self.att_coa != "envelopment" else set(man_bns_att[:3])

        def sector_of(s, y):
            ks = list(axes[s])
            return ks[int(np.argmin([abs(axes[s][k] - y) for k in ks]))] if ks else 0

        def own_sector(s, i):
            k = F[s].bn[i]
            return sectors[s].get(k, (-1e9, 1e9))

        def hq_alive(s):
            return bool(len(hq_idx[s]) == 0 or (nn[s][hq_idx[s], SUP].sum() > 0))

        def net(s):
            return prof[s][main[s]].net * (1.0 if hq_alive(s) else 0.5)

        def advance(s, name, seen, m):
            """Move a trigger through observe -> issue; returns True the minute the order is delivered."""
            tr = triggers[s][name]
            if tr["done"]:
                return False
            if tr["observed"] is None:
                if seen and c2_rng[s].random() < net(s):
                    tr["observed"] = m
                    tr["deliver"] = m + latency(prof[s][main[s]], c2_rng[s], hq_alive(s))
                    c2_events.append((m, s, name, "observed", tr["deliver"]))
                return False
            if m >= tr["deliver"]:
                tr["done"] = True
                c2_events.append((m, s, name, "delivered", m))
                return True
            return False
        cause, minutes, win = 0, float(T), 0.0
        moving = [np.zeros(A.N, bool), np.zeros(D.N, bool)]
        breaching = [np.zeros(A.N), np.zeros(D.N)]                # minutes of breaching left (0 = not breaching)
        breached = np.zeros_like(self.plans[1]["obstacles"])      # obstacle cells already breached (lanes)
        obstacles = self.plans[1]["obstacles"]
        tubes0 = [float(F[s].n0[:, [HOW, ROCK]].sum()) for s in range(2)]
        ammo = [ROUNDS_PER_TUBE * tubes0[s] for s in range(2)]
        sup0 = [max(float(F[s].n0[:, SUP].sum()), 1.0) for s in range(2)]
        on_station = [np.zeros(A.N), np.zeros(D.N)]               # sortie minutes left
        turnaround = [np.zeros(A.N), np.zeros(D.N)]
        helo_ammo = [np.zeros(A.N), np.zeros(D.N)]
        sorties = [0, 0]
        PKs = [(self.PK_ad, self.RM_ad), (self.PK_da, self.RM_da)]
        DIRECT = np.array([c in DIRECT_FIRE for c in CLASSES])
        det_streams = ["det_a", "det_d"]
        kill_streams = ["kill_a", "kill_d"]
        ind_streams = ["ind_a", "ind_d"]

        for m in range(T):
            alive = [(~broken[s]) & (nn[s].sum(1) > 0.5) for s in range(2)]
            # ---- intervisibility and ranges
            vis, Dm = self._vis(pos[0], pos[1])
            expo = [np.array([t.exposure(*p) for p in pos[s]]) * (1 + 0.3 * (breaching[s] > 0)) for s in range(2)]
            moving_prev = moving
            moving = [np.zeros(A.N, bool), np.zeros(D.N, bool)]
            cell_count = []
            for s in range(2):
                ci, cj = t.xy_to_ij(pos[s][:, 0], pos[s][:, 1])
                key = ci * t.n + cj
                man = alive[s] & (F[s].role == ROLE_MANEUVER)
                _, inv, cnt = np.unique(np.where(man, key, -1 - np.arange(F[s].N)), return_inverse=True, return_counts=True)
                cell_count.append(np.where(man, cnt[inv], 0))
            posture = np.where(moving_prev[1], 1.0, 0.6)                  # dug-in defender
            vis_ad = vis * (expo[1] * posture)[None, :]        # defender company as seen from attacker
            vis_da = vis.T * expo[0][None, :]                  # attacker company as seen from defender

            # ---- detection through each side's network (persistent 5 minutes)
            for s, (V, Dd, stream) in enumerate(((vis_ad, Dm, "det_a"), (vis_da, Dm.T, "det_d"))):
                me, en = F[s], F[1 - s]
                S = ((nn[s] * me.P["sensor"]).sum(1) / 10.0) * alive[s]                       # sensing per company
                sig = (nn[1 - s] * en.P["signature"]).sum(1) / np.maximum(nn[1 - s].sum(1), 1)
                recent_fire = (m - fired_at[1 - s]) < 5
                rng_fac = (1000.0 / np.maximum(Dd, 300.0)) ** 2
                hazard = me.nation.recon_uas * (S[:, None] * V * rng_fac).sum(0) * sig * (1 + 4.0 * recent_fire) * 0.6
                hazard += 0.004 * me.nation.recon_uas * alive[1 - s] * (1 + 2.0 * moving_prev[1 - s])   # wide-area UAS trickle
                u = rs[stream].random(en.N)
                seen = (u < 1 - np.exp(-hazard)) & alive[1 - s]
                last_seen[1 - s] = np.where(seen, m, last_seen[1 - s])
            detected = [(m - last_seen[s]) <= 5 for s in range(2)]

            # ---- direct fire, both sides simultaneously
            kills = [np.zeros((A.N, NC)), np.zeros((D.N, NC))]
            shots_on = [np.zeros(A.N), np.zeros(D.N)]
            for s, (V, Dd) in enumerate(((vis_ad, Dm), (vis_da, Dm.T))):
                me, en = F[s], F[1 - s]
                PK, RM = PKs[s]
                can = alive[s][:, None] & alive[1 - s][None, :] & detected[1 - s][None, :] & (V > 0.02)
                if not can.any():
                    continue
                # shots per minute by class a of company i; preference over (target company j, class b)
                Dv = Dd[:, None, :, None]
                Vv = V[:, None, :, None]
                ph = hit_prob(Dv, Vv, en.P["area"][None, None, None, :], me.P["theta0"][None, :, None, None],
                              me.P["theta_s"][None, :, None, None], 0.0, 0.0, 0.0,
                              supp[s][:, None, None, None] > 0.5, 3.0)
                inrange = Dv <= RM[None, :, None, :]
                # acquisition: the share of class-a shooters that have a class-b target in company j in
                # their sights this minute, from the entity engine's detection hazard (per 15 s, x4)
                acq_rate = (me.P["sensor"][None, :, None, None] * en.P["signature"][None, None, None, :] * Vv
                            * (1000.0 / np.maximum(Dv, 50.0)) ** 2 * 0.5 * (me.nation.recon_uas + en.nation.recon_uas) / 1.5)
                p_acq = 1 - np.exp(-4.0 * acq_rate)
                lethal = ph * PK[None, :, None, :] * inrange * can[:, None, :, None] * p_acq   # P(kill) per shot
                pref = lethal * nn[1 - s][None, None, :, :] * en.P["value"][None, None, None, :]
                tot = pref.sum((2, 3), keepdims=True)
                p_alloc = np.where(tot > 0, pref / np.maximum(tot, 1e-12), 0.0)             # logit with lambda = 1
                shots = (4.0 * me.P["fire_rate"] * me.nation.quality)[None, :] * nn[s] * (1 - 0.7 * supp[s])[:, None]
                shots = shots * alive[s][:, None] * DIRECT[None, :]                          # (i, a) per minute
                exp_jb = (shots[:, :, None, None] * p_alloc * lethal).sum((0, 1))          # (j, b) expected kills
                exp_jb = np.minimum(exp_jb * self.kill_scale[s], nn[1 - s])
                if self.kill_draw == "poisson":
                    kills[1 - s] = np.minimum(nn[1 - s], rs[kill_streams[s]].poisson(exp_jb))
                else:                                                                        # dithered rounding
                    kills[1 - s] = np.minimum(nn[1 - s], np.floor(exp_jb + rs[kill_streams[s]].random((en.N, NC))))
                shots_on[1 - s] = (shots[:, :, None, None] * p_alloc).sum((0, 1, 3))
                shot_any = (shots[:, :, None, None] * p_alloc).sum((1, 2, 3)) > 0
                fired_at[s] = np.where(shot_any, m, fired_at[s])

            # ---- indirect fire (draws on the brigade ammunition stock, refilled by the support battalion)
            for s in range(2):
                me, en = F[s], F[1 - s]
                sup_frac = float(nn[s][:, SUP].sum()) / sup0[s]
                ammo[s] += RESUPPLY_PER_TUBE_HOUR / 60.0 * tubes0[s] * sup_frac
                bat = np.flatnonzero((me.role == ROLE_BATTERY) & alive[s])
                if len(bat) == 0 or ammo[s] < 1:
                    continue
                Dd = np.hypot(pos[s][bat, None, 0] - pos[1 - s][None, :, 0], pos[s][bat, None, 1] - pos[1 - s][None, :, 1])
                u = rs[ind_streams[s]].random((len(bat), 2))
                for q, i in enumerate(bat):
                    cls = HOW if nn[s][i, HOW] > 0 else ROCK
                    tubes = nn[s][i, cls]
                    if mission_left[s][i] > 0:
                        j = mission_tgt[s][i]
                        mission_left[s][i] -= 1
                        if mission_left[s][i] == 0:
                            cooldown[s][i] = me.nation.c2_delay_min       # displace / re-target / resupply
                    else:
                        if cooldown[s][i] > 0:
                            cooldown[s][i] -= 1
                            continue
                        cand = detected[1 - s] & alive[1 - s] & (Dd[q] <= me.P["rmax"][cls])
                        if not cand.any() or u[q, 0] > 1.0 / me.nation.c2_delay_min:
                            continue
                        val = (nn[1 - s] * en.P["value"]).sum(1) * (1 + 2.0 * ((m - fired_at[1 - s]) < 5) * (en.role == ROLE_BATTERY))
                        if fires_sector[s] is not None:                                   # priority of fires
                            lo_y, hi_y = sectors[s][fires_sector[s]]
                            val = val * np.where((pos[1 - s][:, 1] >= lo_y) & (pos[1 - s][:, 1] <= hi_y), 2.0, 1.0)
                        val = np.where(cand, val * (0.5 + rs[ind_streams[s]].random(en.N)), -1.0)  # noisy priority
                        j = int(np.argmax(val))
                        # clearance of fires: the sector owner must clear the mission; cross-lineage is harder
                        owner = sector_of(s, pos[1 - s][j, 1])
                        owner_lineage = next((F[s].lineage[c] for c in np.flatnonzero(F[s].bn == owner)), main[s])
                        sync = prof[s][owner_lineage].sync * (1.0 if owner_lineage == F[s].lineage[i] else CROSS_LINEAGE_SYNC)
                        if c2_rng[s].random() > sync:
                            cooldown[s][i] = max(1.0, latency(prof[s][main[s]], c2_rng[s], hq_alive(s)) / 3.0)
                            continue
                        mission_tgt[s][i], mission_left[s][i] = j, 2
                        fired_at[s][i] = m
                    rounds = min(tubes * me.P["rounds_per_min"][cls], ammo[s])
                    ammo[s] -= rounds
                    cov = 1 - 0.5 * t.at(t.cover, *pos[1 - s][j])
                    crowd = 1.0 + 0.3 * max(0, cell_count[1 - s][j] - CROWD_K)                # massed targets
                    exp_b = rounds * me.P["ind_lethality"][cls] * IND_VULN * cov * crowd * (nn[1 - s][j] / np.maximum(nn[1 - s][j].sum(), 1))
                    draw = rs[ind_streams[s]].random(NC)
                    kills[1 - s][j] += np.minimum(nn[1 - s][j] - kills[1 - s][j], np.floor(exp_b + draw))
                    supp[1 - s][j] = min(1.0, supp[1 - s][j] + 0.6)

            # ---- attack aviation: sorties against detected concentrations, losses to enemy SHORAD
            for s in range(2):
                me, en = F[s], F[1 - s]
                avn = np.flatnonzero((me.role == ROLE_AVIATION) & (nn[s][:, HELO] > 0))
                if len(avn) == 0:
                    continue
                PKh = np.array([CLASS_PK[3, int(en.nation.equipment[c].protection)] if c != "attack_helo" else 0.0
                                for c in CLASSES]) * me.nation.quality
                tgt_mask = detected[1 - s] & alive[1 - s] & ((nn[1 - s] * MANEUVER).sum(1) + nn[1 - s][:, SUP] > 0)
                for i in avn:
                    if on_station[s][i] > 0:
                        on_station[s][i] -= 1
                        Dh = np.hypot(pos[s][i, 0] - pos[1 - s][:, 0], pos[s][i, 1] - pos[1 - s][:, 1])
                        inr = tgt_mask & (Dh <= me.P["rmax"][HELO])
                        n_helo = nn[s][i, HELO]
                        if inr.any() and helo_ammo[s][i] > 0:
                            shots = min(helo_ammo[s][i], 4.0 * me.P["fire_rate"][HELO] * n_helo)
                            helo_ammo[s][i] -= shots
                            pref = np.where(inr[:, None], nn[1 - s] * en.P["value"] * PKh, 0.0)
                            p = pref / max(pref.sum(), 1e-12)
                            ph = 1 - np.exp(-0.5)                                     # guided missile, ~0.4 per shot at standoff
                            exp_k = shots * p * ph * PKh
                            kills[1 - s] += np.minimum(nn[1 - s] - kills[1 - s], rs["avn_" + "ad"[s]].poisson(exp_k))
                            hit_cos = p.sum(1) > 0
                            supp[1 - s] = np.where(hit_cos, np.minimum(1.0, supp[1 - s] + 0.3), supp[1 - s])
                            fired_at[s][i] = m
                        near = float(((Dh <= 8000) & alive[1 - s] & (en.role == ROLE_MANEUVER)).sum())
                        p_loss = HELO_LOSS_PER_MIN * en.nation.shorad * min(1.5, near / 5.0)
                        lost = rs["avn_" + "ad"[s]].binomial(int(n_helo), min(1.0, p_loss))
                        kills[s][i, HELO] += lost
                        if on_station[s][i] == 0 or helo_ammo[s][i] <= 0:
                            on_station[s][i], turnaround[s][i] = 0, TURNAROUND_MIN
                            pos[s][i] = self.plans[s]["pos"][i].copy()
                    elif turnaround[s][i] > 0:
                        turnaround[s][i] -= 1
                    elif tgt_mask.any() and m >= 10:
                        cen = pos[1 - s][tgt_mask].mean(0)
                        toward = np.array([-1.0, 0.0]) if s == 0 else np.array([1.0, 0.0])   # stand off toward own side
                        pos[s][i] = cen + 6000.0 * toward
                        on_station[s][i], helo_ammo[s][i] = SORTIE_MIN, me.P["ammo"][HELO] * nn[s][i, HELO] / 2
                        sorties[s] += 1

            # ---- apply kills, suppression
            for s in range(2):
                kills[s] = np.minimum(kills[s], nn[s])
                nn[s] = nn[s] - kills[s]
                plat = np.maximum(nn[s].sum(1), 1)
                supp[s] = np.clip(supp[s] * 0.7 + 0.03 * shots_on[s] / plat, 0, 1)

            # ---- breakpoints
            for s in range(2):
                lost = 1 - (nn[s] * MANEUVER).sum(1) / np.maximum((F[s].n0 * MANEUVER).sum(1), 1)
                newly = (lost >= bp_co[s]) & (F[s].role == ROLE_MANEUVER) & ~broken[s]
                broken[s] |= newly
                for k in np.unique(F[s].bn[F[s].bn >= 0]):
                    idx = (F[s].bn == k) & (F[s].role == ROLE_MANEUVER)
                    if idx.any() and broken[s][idx].mean() >= 0.5:
                        broken[s][idx] = True

            # ---- decisions: triggers -> observe (net) -> latency -> order; subordinates may act on intent
            for s in range(2):
                me = F[s]
                g = self.graphs[main[s]]
                man_bns = np.unique(me.bn[(me.bn >= 0) & (me.role == ROLE_MANEUVER)])
                bn_frac = {k: (nn[s][me.bn == k] * MANEUVER).sum() / max((me.n0[me.bn == k] * MANEUVER).sum(), 1)
                           for k in man_bns}
                # reserve commitment (attacker: reserve bns; defender 'mobile': counterattack force)
                if s == 0:
                    lead = list(man_bns[:2]) if self.att_coa != "envelopment" else list(man_bns[:1])
                    reserve = [k for k in man_bns if k not in lead]
                    seen = m >= 120 or any(bn_frac[k] < 0.6 or broken[0][(me.bn == k) & (me.role == ROLE_MANEUVER)].all() for k in lead)
                else:
                    lead = list(man_bns[:2])
                    reserve = [k for k in man_bns if k not in lead] if self.def_coa == "mobile" else []
                    seen = any(bn_frac[k] < 0.6 for k in lead) and alive[0].any()
                if reserve and not triggers[s]["reserve"]["done"]:
                    delivered = advance(s, "reserve", seen, m)
                    # initiative: the reserve commander sees the same trigger and may act before the order arrives
                    res_lineage = me.lineage[int(np.flatnonzero(me.bn == reserve[0])[0])]
                    acts = delivered or (seen and triggers[s]["reserve"]["observed"] is not None
                                         and c2_rng[s].random() < prof[s][res_lineage].p_init / max(prof[s][res_lineage].tau, 1.0))
                    if acts:
                        triggers[s]["reserve"]["done"] = True
                        if s == 0:
                            if g["reserve"] == "reinforce_success":
                                tgt = max(lead, key=lambda k: pos[0][(me.bn == k) & alive[0]][:, 0].max() if (alive[0] & (me.bn == k)).any() else -1e9)
                            elif g["reserve"] == "reinforce_failure":
                                tgt = min(lead, key=lambda k: bn_frac[k])
                            else:
                                tgt = None                           # planned axis: keep own waypoints
                            reserve_target[0] = tgt
                            for k in reserve:
                                for i in np.flatnonzero((me.bn == k) & (me.role == ROLE_MANEUVER)):
                                    if tgt is not None:
                                        y = axes[0][tgt]
                                        wps[0][i] = [np.array([2000.0, y]), np.array([self.objective[0] - 1500, y]),
                                                     self.objective.copy()]
                                        path[0][i] = None
                                    committed_flag[0][i] = True
                        else:
                            aim = pos[0][alive[0]][pos[0][alive[0]][:, 0].argmax()]      # most advanced attacker
                            for k in reserve:
                                for i in np.flatnonzero((me.bn == k) & (me.role == ROLE_MANEUVER)):
                                    wps[1][i] = [aim.copy()]
                                    path[1][i] = None
                                    committed_flag[1][i] = True
                        c2_events.append((m, s, "reserve", "acted" if not delivered else "ordered", reserve_target[s]))
                # priority of fires to the sector where the enemy masses
                if g["fires_shift"] and not triggers[s]["fires_shift"]["done"]:
                    en_man = alive[1 - s] & (F[1 - s].role == ROLE_MANEUVER) & detected[1 - s]
                    counts = {k: int(((pos[1 - s][:, 1] >= lo) & (pos[1 - s][:, 1] <= hi) & en_man).sum())
                              for k, (lo, hi) in sectors[s].items()}
                    seen = bool(counts) and max(counts.values()) >= 4
                    if advance(s, "fires_shift", seen, m):
                        fires_sector[s] = max(counts, key=counts.get)
                # sustainment: tactical pause when the fires stock runs low
                if not triggers[s]["ammo_pause"]["done"] and ROUNDS_PER_TUBE * tubes0[s] > 0:
                    seen = ammo[s] < g["ammo_pause"] * ROUNDS_PER_TUBE * tubes0[s]
                    if advance(s, "ammo_pause", seen, m):
                        pause_left[s] = PAUSE_MIN
                if pause_left[s] > 0:
                    pause_left[s] -= 1

            # ---- movement (toward waypoints; obstacles; boundary discipline; crowding)
            for s in range(2):
                me = F[s]
                for i in range(me.N):
                    if not alive[s][i] or not wps[s][i] or supp[s][i] > 0.6 or me.role[i] == ROLE_AVIATION:
                        continue
                    if breaching[s][i] > 0:                     # halted at an obstacle
                        breaching[s][i] -= 1
                        if breaching[s][i] <= 0:
                            ci, cj = t.xy_to_ij(*pos[s][i])
                            breached[ci, cj] = True
                        continue
                    if s == 0 and me.role[i] == ROLE_MANEUVER and me.bn[i] not in lead_bns_att and not committed_flag[0][i]:
                        continue                     # reserve waits for its order
                    if pause_left[s] > 0 and me.role[i] == ROLE_MANEUVER:
                        continue                     # tactical pause
                    if path[s][i] is None:
                        path[s][i] = self._route(self._snap(*pos[s][i]), self._snap(*wps[s][i][0]))
                        path_k[s][i] = 0
                    p = path[s][i]
                    spd = 4.0 * (nn[s][i] * me.P["speed"]).sum() / max(nn[s][i].sum(), 1)
                    cell_i, cell_j = t.xy_to_ij(*pos[s][i])
                    spd *= VEH_SPEED[t.cls[cell_i, cell_j]] * (1 - 0.7 * supp[s][i])
                    if cell_count[s][i] > CROWD_K:
                        spd *= 0.5                   # congestion at lanes, bridges and the objective
                    if s == 0 and (detected[0][i] and (Dm[i][alive[1]] <= 2500).any() if alive[1].any() else False):
                        spd *= 0.6                  # in contact
                    steps = spd / t.cell
                    new_k = min(len(p) - 1, path_k[s][i] + steps)
                    if s == 0 and obstacles.any():           # attacker: stop at the first unbreached obstacle cell
                        for cell in p[int(path_k[s][i]) + 1:int(new_k) + 1]:
                            ci, cj = divmod(int(cell), t.n)
                            if obstacles[ci, cj] and not breached[ci, cj]:
                                new_k = float(list(p).index(cell))
                                eng_near = ((me.role == ROLE_ENGINEER) & alive[s] & (supp[s] < 0.6)
                                            & (np.hypot(pos[s][:, 0] - pos[s][i, 0], pos[s][:, 1] - pos[s][i, 1]) <= 1500)).any()
                                lin = me.lineage[i]
                                if (self.graphs[lin]["obstacle"] == "bypass" and c2_rng[s].random() < prof[s][lin].latitude):
                                    gap = self._gap_near(obstacles | breached, cell, 10)
                                    if gap is not None:
                                        wps[s][i].insert(0, self._xy(gap))    # detour through the nearest gap
                                        path[s][i] = None
                                        c2_events.append((m, s, "bypass", me.names[i], None))
                                        break
                                breaching[s][i] = BREACH_MIN[1] if eng_near else BREACH_MIN[0]
                                veh = nn[s][i] * (np.arange(NC) != INF) * (np.arange(NC) != ATGM)
                                mine = rs["eng_a"].poisson(MINE_KILL_FRAC * veh)
                                nn[s][i] = np.maximum(nn[s][i] - mine, 0)
                                break
                    path_k[s][i] = new_k
                    pos[s][i] = self._xy(p[int(path_k[s][i])])
                    moving[s][i] = breaching[s][i] == 0
                    if int(path_k[s][i]) >= len(p) - 1 and breaching[s][i] == 0:
                        wps[s][i].pop(0)
                        path[s][i] = None

            # ---- boundary discipline: friendly fire on units out of sector while in contact
            for s in range(2):
                me = F[s]
                for i in np.flatnonzero(alive[s] & (me.role == ROLE_MANEUVER) & (me.bn >= 0)):
                    lo_y, hi_y = own_sector(s, i)
                    if lo_y <= pos[s][i, 1] <= hi_y:
                        continue
                    Dc = np.hypot(pos[s][:, 0] - pos[s][i, 0], pos[s][:, 1] - pos[s][i, 1])
                    others = alive[s] & (me.bn != me.bn[i]) & (me.role == ROLE_MANEUVER) & (Dc <= 2000)
                    if not others.any() or not detected[s][i]:
                        continue
                    lin_i = me.lineage[i]
                    lin_o = me.lineage[int(np.flatnonzero(others)[0])]
                    sync = prof[s][lin_i].sync * (1.0 if lin_i == lin_o else CROSS_LINEAGE_SYNC)
                    ff = c2_rng[s].poisson(FRATRICIDE_RATE * (1 - sync) * nn[s][i] * MANEUVER)
                    if ff.sum() > 0:
                        nn[s][i] = np.maximum(nn[s][i] - ff, 0)
                        c2_events.append((m, s, "fratricide", me.names[i], int(ff.sum())))

            # ---- bookkeeping and end conditions
            for s in range(2):
                strength[s][m + 1] = (nn[s] * MANEUVER * (~broken[s])[:, None]).sum()
            if log and m % log_every == 0:
                track[0].append(m)
                track[1].append(pos[0].copy())
                track[2].append(pos[1].copy())
                track[3].append(alive[0].copy())
                track[4].append(alive[1].copy())
            near_obj = lambda s: np.hypot(pos[s][:, 0] - self.objective[0], pos[s][:, 1] - self.objective[1]) <= self.objective_radius
            att_on = (alive[0] & (A.role == ROLE_MANEUVER) & near_obj(0)).sum()
            def_on = (alive[1] & (D.role == ROLE_MANEUVER) & near_obj(1)).sum()
            att_lost = 1 - strength[0][m + 1] / max(n0m[0], 1)
            def_lost = 1 - strength[1][m + 1] / max(n0m[1], 1)
            if not objective_taken and ((att_on >= self.objective_companies and def_on == 0) or def_lost >= bp_bde[1]):
                objective_taken, cause, minutes, win = True, 2, m + 1, 1.0
                if self.graphs[main[0]]["objective"] == "exploit" and hq_alive(0):
                    pursuit_left = PURSUIT_MIN                # sequel: keep pressing while the defender is broken
                    for i in np.flatnonzero(alive[0] & (A.role == ROLE_MANEUVER)):
                        wps[0][i] = [np.array([self.objective[0] + 4000.0, pos[0][i, 1] * 0.5])]
                        path[0][i] = None
                else:
                    break
            elif objective_taken:
                pursuit_left -= 1
                if pursuit_left <= 0:
                    break
            if att_lost >= bp_bde[0] or (broken[0] | ~alive[0])[A.role == ROLE_MANEUVER].all():
                cause, minutes = 1, m + 1
                break
        for s in range(2):
            strength[s][int(minutes) + 1:] = strength[s][int(minutes)]
        res = dict(win=win, cause=cause, minutes=minutes,
                   att_losses=(A.n0 - nn[0]).sum(0), def_losses=(D.n0 - nn[1]).sum(0),
                   att_strength=strength[0], def_strength=strength[1],
                   att_sorties=sorties[0], def_sorties=sorties[1], att_ammo=ammo[0], def_ammo=ammo[1],
                   c2_events=c2_events)
        if track is not None:
            track = (np.array(track[0]), np.array(track[1]), np.array(track[2]), np.array(track[3]), np.array(track[4]))
        return res, track
