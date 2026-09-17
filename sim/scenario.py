"""Scenario geometry: Red positions, Blue start/SBF/OP positions, routes, and LOS tables."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .terrain import (BUILDING, CONCEAL, INF_COST, INF_SPEED, VEH_COST, VEH_SPEED, WATER,
                      los_fraction)
from .units import (BUILDING_FLOOR, COMPANY, DISMOUNT_HEIGHTS, EYE_DISMOUNT, EYE_RED, EYE_VEHICLE,
                    R_AT, RED_HEIGHTS, VEHICLE_HEIGHTS)


@dataclass(frozen=True)
class ScenarioConfig:
    bearing_deg: float = 90.0      # direction from the objective toward Blue's start (0 = north)
    start_dist: float = 1700.0
    sbf_min: float = 900.0
    sbf_max: float = 1400.0
    op_min: float = 1000.0
    op_max: float = 1500.0
    dismount_dist: float = 450.0
    red_radius: float = 300.0
    danger_close: float = 250.0
    max_minutes: float = 60.0


def _grid_graph(cost, cell):
    n = cost.shape[0]
    idx = np.arange(n * n).reshape(n, n)
    flat = cost.ravel()
    I, J = np.mgrid[0:n, 0:n]
    rows, cols, w = [], [], []
    for di, dj in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        I2, J2 = I + di, J + dj
        ok = (I2 >= 0) & (I2 < n) & (J2 >= 0) & (J2 < n)
        a, b = idx[I[ok], J[ok]], idx[I2[ok], J2[ok]]
        c = 0.5 * (flat[a] + flat[b]) * cell * np.hypot(di, dj)
        fin = np.isfinite(c)
        rows += [a[fin], b[fin]]
        cols += [b[fin], a[fin]]
        w += [c[fin], c[fin]]
    return coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n * n, n * n)).tocsr()


def _bearing(x, y):
    return np.degrees(np.arctan2(x, y)) % 360


def _ang_diff(a, b):
    return np.abs((np.asarray(a) - b + 180) % 360 - 180)


def _greedy(scores, xy, k, spacing, exclude=(), rng=None, temperature=0.0):
    """Pick k well-spaced cells in score order. With temperature > 0, scores get Gumbel noise
    scaled by temperature, so repeated calls sample different good positions (a logit over cells)."""
    if k <= 0:
        return []
    if temperature > 0 and rng is not None:
        scores = scores + temperature * rng.gumbel(size=len(scores))
    order = np.argsort(-scores)
    picks = [np.asarray(e) for e in exclude]
    chosen = []
    for o in order:
        if all(np.hypot(*(xy[o] - p)) >= spacing for p in picks):
            chosen.append(o)
            picks.append(xy[o])
            if len(chosen) == k:
                break
    return chosen


class Scenario:
    """Places both forces on the ground, routes Blue, and precomputes LOS along every route.

    seed drives the candidate-cell sampling. placement_temperature > 0 makes Red, SBF and OP
    placement stochastic (a logit over scored cells); route_noise > 0 perturbs the movement
    cost grids with a lognormal random field so routes vary between seeds. Both default to 0 for
    the deterministic, best-scored plan; the outer-loop tools sample them.
    """

    def __init__(self, terrain, cfg: ScenarioConfig, roster=COMPANY, seed=0, placement_temperature=0.0,
                 route_noise=0.0):
        self.t, self.cfg, self.R = terrain, cfg, roster
        self.seed, self.route_noise = seed, float(route_noise)
        self.rng = np.random.default_rng(seed)
        self.temperature = float(placement_temperature)
        t = terrain
        if cfg.start_dist > t.half - t.cell:
            raise ValueError(f"Start distance {cfg.start_dist:.0f} m is outside the {2 * t.half:.0f} m area. "
                             "Shorten it or load a larger area.")
        self.vcost, self.icost = VEH_COST[t.cls].copy(), INF_COST[t.cls].copy()
        if route_noise > 0:
            from scipy.ndimage import gaussian_filter
            field = gaussian_filter(self.rng.normal(size=t.cls.shape), sigma=3)
            field = np.exp(route_noise * field / field.std())
            self.vcost, self.icost = self.vcost * field, self.icost * field
        self.Gv, self.Gi = _grid_graph(self.vcost, t.cell), _grid_graph(self.icost, t.cell)
        self.cx, self.cy = (a.ravel() for a in t.cell_centres())
        self.dir = np.array([np.sin(np.radians(cfg.bearing_deg)), np.cos(np.radians(cfg.bearing_deg))])
        self.obj_i = self._snap(0.0, 0.0, self.icost)
        self.obj_v = self._snap(0.0, 0.0, self.vcost)
        self.dv, self.pv = dijkstra(self.Gv, indices=self.obj_v, return_predecessors=True)
        self.di, self.pi = dijkstra(self.Gi, indices=self.obj_i, return_predecessors=True)
        self._place_red()
        self._place_blue_starts()
        self._place_sbf_op()
        self._unit_cache = {}
        self._tables = {}

    # ---------------------------------------------------------------- placement
    def _snap(self, x, y, cost, reach=None):
        ok = np.isfinite(cost.ravel())
        if reach is not None:
            ok &= np.isfinite(reach)
        if not ok.any():
            raise ValueError("No passable ground in the area.")
        d = np.where(ok, np.hypot(self.cx - x, self.cy - y), np.inf)
        return int(np.argmin(d))

    def _xy(self, node):
        return np.array([self.cx[node], self.cy[node]])

    def _sample(self, mask, k):
        idx = np.flatnonzero(mask)
        if len(idx) > k:
            idx = self.rng.choice(idx, k, replace=False)
        return idx

    def _place_red(self):
        t, cfg = self.t, self.cfg
        r = np.hypot(self.cx, self.cy)
        cls = t.cls.ravel()
        building = (cls == BUILDING)
        conceal = CONCEAL[cls]

        def score(cands, dists, spread, heights):
            pts = np.array([d * np.array([np.sin(np.radians(cfg.bearing_deg + a)),
                                          np.cos(np.radians(cfg.bearing_deg + a))])
                            for d in dists for a in spread])
            P, Q = len(cands), len(pts)
            obs = np.repeat(np.c_[self.cx[cands], self.cy[cands]], Q, 0)
            eye = np.repeat(EYE_RED + BUILDING_FLOOR * building[cands], Q)
            vis = los_fraction(t, obs, eye, np.tile(pts, (P, 1)), heights).reshape(P, Q).mean(1)
            return vis

        at_c = self._sample((r <= cfg.red_radius) & (cls != WATER), 250)
        at_s = score(at_c, np.linspace(600, cfg.start_dist, 8), [-25, -12, 0, 12, 25], VEHICLE_HEIGHTS)
        at_s = at_s + 0.4 * (1 - conceal[at_c])
        xy = np.c_[self.cx[at_c], self.cy[at_c]]
        n_at = int((self.R.rtype == R_AT).sum())
        n_inf = self.R.nr - n_at
        at = [at_c[k] for k in _greedy(at_s, xy, n_at, 60, rng=self.rng, temperature=self.temperature)]
        inf_c = self._sample((r <= min(cfg.red_radius, 180)) & (cls != WATER), 250)
        inf_s = score(inf_c, np.linspace(150, 800, 6), [-25, -12, 0, 12, 25], DISMOUNT_HEIGHTS)
        inf_s = inf_s + 0.6 * (1 - conceal[inf_c])
        xy2 = np.c_[self.cx[inf_c], self.cy[inf_c]]
        inf = [inf_c[k] for k in _greedy(inf_s, xy2, n_inf, 40, exclude=[self._xy(a) for a in at],
                                         rng=self.rng, temperature=self.temperature)]
        nodes = at + inf
        if len(nodes) < self.R.nr:
            raise ValueError("Not enough dry ground near the objective to place the defense.")
        self.red_nodes = np.array(nodes)
        self.rpos = np.c_[self.cx[self.red_nodes], self.cy[self.red_nodes]]
        self.r_building = building[self.red_nodes]
        self.r_conceal = conceal[self.red_nodes]
        self.r_eye = EYE_RED + BUILDING_FLOOR * self.r_building
        self.r_heights = RED_HEIGHTS[None, :] + BUILDING_FLOOR * self.r_building[:, None]

    def _place_blue_starts(self):
        cfg = self.cfg
        c = cfg.start_dist * self.dir
        perp = np.array([self.dir[1], -self.dir[0]])
        offsets = self.R.start_offsets()
        self.start_nodes = []
        for u, off in enumerate(offsets):
            p = c + off * perp
            if self.R.is_scout[u]:
                self.start_nodes.append(self._snap(*p, self.icost, reach=self.di))
            else:
                self.start_nodes.append(self._snap(*p, self.vcost, reach=self.dv))

    def _place_sbf_op(self):
        t, cfg = self.t, self.cfg
        r = np.hypot(self.cx, self.cy)
        ang = _ang_diff(_bearing(self.cx, self.cy), cfg.bearing_deg)
        conceal = CONCEAL[t.cls.ravel()]

        def pick(mask, eye, k, spacing, w_conceal, label):
            cands = self._sample(mask, 300)
            if len(cands) < k:
                raise ValueError(f"No usable {label} positions; widen the {label} ring or change the bearing.")
            P, NR = len(cands), self.R.nr
            obs = np.repeat(np.c_[self.cx[cands], self.cy[cands]], NR, 0)
            tgt = np.tile(self.rpos, (P, 1))
            vis = los_fraction(t, obs, eye, tgt, np.tile(self.r_heights, (P, 1))).reshape(P, NR).mean(1)
            s = vis + w_conceal * (1 - conceal[cands])
            xy = np.c_[self.cx[cands], self.cy[cands]]
            return ([cands[i] for i in _greedy(s, xy, k, spacing, rng=self.rng, temperature=self.temperature)],
                    float(vis.max()))

        sbf_mask = (r >= cfg.sbf_min) & (r <= cfg.sbf_max) & (ang <= 45) & np.isfinite(self.dv)
        n_sbf = max(1, int(self.R.is_tank.sum()))
        self.sbf_nodes, self.sbf_best_vis = pick(sbf_mask, EYE_VEHICLE, n_sbf, 60, 0.2, "support-by-fire")
        n_op = int(self.R.is_scout.sum())
        if n_op:
            op_mask = (r >= cfg.op_min) & (r <= cfg.op_max) & (ang <= 60) & np.isfinite(self.di)
            self.op_nodes, _ = pick(op_mask, EYE_DISMOUNT, n_op, 150, 0.5, "observation post")
        else:
            self.op_nodes = []

    # ---------------------------------------------------------------- routes
    @staticmethod
    def _path(u, pred, src):
        cells = [u]
        while u != src:
            u = pred[u]
            if u < 0:
                raise ValueError("A unit has no route to its destination. Change the bearing or start distance.")
            cells.append(u)
        return cells

    def _route(self, u, armor):
        s = self.start_nodes[u]
        R = self.R
        if R.is_tank[u]:
            if armor == "SBF":
                dst = self.sbf_nodes[int(R.is_tank[:u].sum()) % len(self.sbf_nodes)]
                _, pred = dijkstra(self.Gv, indices=dst, return_predecessors=True)
                return self._path(s, pred, dst), None
            return self._path(s, self.pv, self.obj_v), None
        if R.is_scout[u]:
            dst = self.op_nodes[int(R.is_scout[:u].sum()) % len(self.op_nodes)]
            _, pred = dijkstra(self.Gi, indices=dst, return_predecessors=True)
            return self._path(s, pred, dst), 0
        veh = self._path(s, self.pv, self.obj_v)
        cut = next((k for k, c in enumerate(veh)
                    if np.hypot(self.cx[c], self.cy[c]) <= self.cfg.dismount_dist), len(veh) - 1)
        foot = self._path(veh[cut], self.pi, self.obj_i)
        return veh[:cut + 1] + foot[1:], cut + 1

    def _unit(self, u, armor):
        key = (u, armor if self.R.is_tank[u] else None)
        if key in self._unit_cache:
            return self._unit_cache[key]
        t, NR = self.t, self.R.nr
        cells, dis_k = self._route(u, armor)
        K = len(cells)
        xy = np.c_[self.cx[cells], self.cy[cells]]
        seg = np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T)]
        cum = np.cumsum(seg)
        vehicle = np.ones(K, bool) if dis_k is None else (np.arange(K) < dis_k)
        cls = t.cls.ravel()[cells]
        speed = np.where(vehicle, VEH_SPEED[cls], INF_SPEED[cls])
        obs = np.repeat(xy, NR, 0)
        red = np.tile(self.rpos, (K, 1))
        eye = np.repeat(np.where(vehicle, EYE_VEHICLE, EYE_DISMOUNT), NR)
        visR = los_fraction(t, obs, eye, red, np.tile(self.r_heights, (K, 1))).reshape(K, NR)
        hts = np.where(vehicle[:, None], VEHICLE_HEIGHTS, DISMOUNT_HEIGHTS)
        visB = los_fraction(t, red, np.tile(self.r_eye, K), obs, np.repeat(hts, NR, 0)).reshape(K, NR)
        dist = np.hypot(obs[:, 0] - red[:, 0], obs[:, 1] - red[:, 1]).reshape(K, NR)
        rec = dict(xy=xy, cum=cum, vehicle=vehicle, speed=speed, conceal=CONCEAL[cls],
                   visR=visR, visB=visB, dist=dist, K=K)
        self._unit_cache[key] = rec
        return rec

    def tables(self, armor):
        """Padded per-unit route tables for the engine (shapes (NB, K, ...))."""
        if armor in self._tables:
            return self._tables[armor]
        recs = [self._unit(u, armor) for u in range(self.R.nb)]
        K = max(r["K"] for r in recs)

        def pad(key):
            out = []
            for r in recs:
                a = r[key]
                reps = K - len(a)
                if reps:
                    a = np.concatenate([a, np.repeat(a[-1:], reps, 0)])
                out.append(a)
            return np.stack(out)

        cum = pad("cum")
        for u, r in enumerate(recs):                    # keep padded distances strictly increasing
            cum[u, r["K"]:] = r["cum"][-1] + 1e-3 * np.arange(1, K - r["K"] + 1)
        tab = dict(xy=pad("xy"), cum=cum, length=np.array([r["cum"][-1] for r in recs]),
                   vehicle=pad("vehicle"), speed_fac=pad("speed"), conceal=pad("conceal"),
                   visR=pad("visR"), visB=pad("visB"), dist=pad("dist"),
                   rpos=self.rpos, r_conceal=self.r_conceal, cfg=self.cfg, roster=self.R)
        self._tables[armor] = tab
        return tab

    def routes_xy(self, armor):
        return [self._unit(u, armor)["xy"] for u in range(self.R.nb)]

    def dismount_points(self):
        out = []
        for u in np.flatnonzero(self.R.is_inf):
            rec = self._unit(u, "SBF")
            k = int(np.argmin(rec["vehicle"])) if not rec["vehicle"].all() else rec["K"] - 1
            out.append(rec["xy"][k])
        return np.array(out).reshape(-1, 2)


__all__ = ["Scenario", "ScenarioConfig"]
