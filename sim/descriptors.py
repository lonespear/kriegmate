"""Terrain descriptors, terrain-space sampling, and a terrain library with disk cache.

The point is to make "tuned across varied terrain" a testable claim: every terrain gets a short
numeric description, calibration terrains are chosen to cover that space evenly, and a share of
them is held out and never used for tuning.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.stats import qmc

from .terrain import BUILDING, FOREST, ROAD, WATER, Terrain, los_fraction, synthetic_terrain
from .units import DISMOUNT_HEIGHTS, EYE_DISMOUNT

DESCRIPTOR_NAMES = ["building_density", "forest_share", "road_density", "water_share", "relief",
                    "openness_500", "openness_1500"]


@dataclass(frozen=True)
class Descriptors:
    building_density: float   # share of cells that are building
    forest_share: float       # share of cells that are wood or scrub
    road_density: float       # share of cells that are road
    water_share: float
    relief: float             # 5th-95th percentile spread of ground elevation, m
    openness_500: float       # mean LOS fraction between random points 400-600 m apart (dismount eye/target)
    openness_1500: float      # same for 1400-1600 m

    def vector(self):
        return np.array([getattr(self, k) for k in DESCRIPTOR_NAMES])


def describe(t: Terrain, n_pairs=300, seed=0) -> Descriptors:
    rng = np.random.default_rng(seed)
    cls = t.cls
    share = lambda c: float((cls == c).mean())

    def openness(lo, hi):
        ang = rng.uniform(0, 2 * np.pi, n_pairs)
        d = rng.uniform(lo, hi, n_pairs)
        a = rng.uniform(-t.half, t.half, (n_pairs, 2))
        b = a + d[:, None] * np.c_[np.cos(ang), np.sin(ang)]
        b = np.clip(b, -t.half + 1, t.half - 1)      # pairs near the edge get a shorter baseline
        vis = los_fraction(t, a, EYE_DISMOUNT, b, DISMOUNT_HEIGHTS)
        return float(vis.mean())

    g = t.ground
    return Descriptors(share(BUILDING), share(FOREST), share(ROAD), share(WATER),
                       float(np.percentile(g, 95) - np.percentile(g, 5)),
                       openness(400, 600), openness(1400, 1600))


# ---------------------------------------------------------------- sampling the terrain space
SYNTHETIC_RANGES = dict(building_density=(0.0005, 0.02), forest_share=(0.02, 0.30),
                        relief=(5.0, 80.0), road_density=(0.01, 0.04))


def synthetic_design(k, seed=0, ranges=SYNTHETIC_RANGES):
    """Latin-hypercube sample of synthetic-terrain generator settings (log scale on density)."""
    names = list(ranges)
    u = qmc.LatinHypercube(d=len(names), seed=seed, optimization="random-cd").random(k)
    out = []
    for row in u:
        d = {}
        for name, v in zip(names, row):
            lo, hi = ranges[name]
            d[name] = float(np.exp(np.log(lo) + v * (np.log(hi) - np.log(lo)))) if name != "relief" \
                else float(lo + v * (hi - lo))
        out.append(d)
    return out


def select_terrains(desc_vectors, k, holdout_frac=0.2, seed=0):
    """Choose k representative terrains from a candidate pool by k-means on standardized
    descriptors (one nearest to each centroid), and split them into train/holdout.

    Returns (train_idx, holdout_idx) into the candidate pool. Holdout terrains are the ones
    farthest from their cluster's other members, so they test extrapolation, not memorization."""
    X = np.asarray(desc_vectors, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    k = min(k, len(Z))
    rng = np.random.default_rng(seed)
    cent, lab = kmeans2(Z, k, minit="++", seed=rng)
    picks = []
    for c in range(k):
        members = np.flatnonzero(lab == c)
        if len(members) == 0:
            continue
        picks.append(int(members[np.argmin(np.linalg.norm(Z[members] - cent[c], axis=1))]))
    picks = sorted(set(picks))
    n_hold = max(1, int(round(holdout_frac * len(picks)))) if len(picks) > 2 else 0
    # hold out the picks whose descriptors are farthest from the centroid of the rest
    if n_hold:
        d_all = np.linalg.norm(Z[picks] - Z[picks].mean(0), axis=1)
        order = np.argsort(-d_all)
        hold = [picks[i] for i in order[:n_hold]]
    else:
        hold = []
    train = [p for p in picks if p not in hold]
    return train, hold


# ---------------------------------------------------------------- terrain library
class TerrainLibrary:
    """Terrains plus their descriptors, cached under a directory as .npz + index.json."""

    def __init__(self, root="cache/terrains"):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.index_path = os.path.join(root, "index.json")
        self.index = json.load(open(self.index_path)) if os.path.exists(self.index_path) else {}

    @staticmethod
    def key(spec):
        return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]

    def get(self, spec, builder=None):
        """spec: dict describing the terrain (synthetic settings, or lat/lon/half/cell for OSM).
        builder: callable(spec) -> Terrain, used on a cache miss."""
        k = self.key(spec)
        path = os.path.join(self.root, f"{k}.npz")
        if k in self.index and os.path.exists(path):
            return Terrain.load(path), Descriptors(**self.index[k]["descriptors"])
        if builder is None:
            builder = build_from_spec
        t = builder(spec)
        d = describe(t)
        t.save(path)
        self.index[k] = dict(spec=spec, descriptors=asdict(d), source=t.source, name=t.name,
                             fetched_at=t.fetched_at)
        json.dump(self.index, open(self.index_path, "w"), indent=1)
        return t, d

    def entries(self):
        return dict(self.index)


def build_from_spec(spec):
    spec = dict(spec)
    kind = spec.pop("kind", "synthetic")
    if kind == "synthetic":
        return synthetic_terrain(**spec)
    if kind == "osm":
        from .terrain import osm_terrain
        return osm_terrain(**spec)
    raise ValueError(f"Unknown terrain spec kind {kind!r}")
