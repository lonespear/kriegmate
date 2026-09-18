"""Ground model: OpenStreetMap features and elevation tiles rasterized to a local grid.

Coordinates are local metres (x east, y north) around (lat0, lon0). Grid row i runs south to
north, column j runs west to east.
"""
from __future__ import annotations

import datetime as _dt
import io
import math
from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path

OPEN, ROAD, FOREST, BUILDING, WATER = range(5)
CLASS_NAMES = ["Open", "Road", "Forest", "Building", "Water"]
CONCEAL = np.array([1.0, 1.0, 0.5, 0.35, 1.0])      # exposure multiplier for a unit in the cell
VEH_COST = np.array([1.0, 0.7, 4.0, np.inf, np.inf])
INF_COST = np.array([1.0, 0.9, 1.5, 2.5, np.inf])
VEH_SPEED = np.array([1.0, 1.3, 0.4, 0.0, 0.0])
INF_SPEED = np.array([1.0, 1.0, 0.8, 0.6, 0.0])

USER_AGENT = "adversarial-mc-sim/0.1 (educational wargaming model)"
M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON_EQ = 111_320.0
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter",
                 "https://overpass.private.coffee/api/interpreter"]
ROAD_WIDTH = {"motorway": 24, "trunk": 20, "primary": 14, "secondary": 12, "tertiary": 10}
SKIP_HIGHWAY = {"footway", "path", "steps", "cycleway", "bridleway", "pedestrian",
                "corridor", "proposed", "construction", "elevator", "platform"}


@dataclass
class Terrain:
    lat0: float
    lon0: float
    half: float
    cell: float
    ground: np.ndarray
    surface: np.ndarray
    cls: np.ndarray
    source: str
    name: str = ""
    notes: list = field(default_factory=list)
    fetched_at: str = ""          # ISO timestamp of the OSM/elevation fetch (for run manifests)

    def save(self, path):
        np.savez_compressed(path, lat0=self.lat0, lon0=self.lon0, half=self.half, cell=self.cell,
                            ground=self.ground, surface=self.surface, cls=self.cls, source=self.source,
                            name=self.name, notes=np.array(self.notes, dtype=object),
                            fetched_at=self.fetched_at)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        return cls(float(z["lat0"]), float(z["lon0"]), float(z["half"]), float(z["cell"]),
                   z["ground"], z["surface"], z["cls"].astype(np.int8), str(z["source"]),
                   str(z["name"]), list(z["notes"]), str(z["fetched_at"]))

    @property
    def n(self):
        return self.cls.shape[0]

    @property
    def kx(self):
        return M_PER_DEG_LON_EQ * math.cos(math.radians(self.lat0))

    def xy_to_ij(self, x, y):
        j = np.clip(np.floor((np.asarray(x, float) + self.half) / self.cell).astype(int), 0, self.n - 1)
        i = np.clip(np.floor((np.asarray(y, float) + self.half) / self.cell).astype(int), 0, self.n - 1)
        return i, j

    def ij_to_xy(self, i, j):
        return (-self.half + (np.asarray(j) + 0.5) * self.cell,
                -self.half + (np.asarray(i) + 0.5) * self.cell)

    def xy_to_lonlat(self, x, y):
        return self.lon0 + np.asarray(x) / self.kx, self.lat0 + np.asarray(y) / M_PER_DEG_LAT

    def lonlat_to_xy(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * M_PER_DEG_LAT

    def bounds_lonlat(self):
        w, s = self.xy_to_lonlat(-self.half, -self.half)
        e, n = self.xy_to_lonlat(self.half, self.half)
        return float(w), float(s), float(e), float(n)

    def at(self, arr, x, y):
        i, j = self.xy_to_ij(x, y)
        return arr[i, j]

    def cell_centres(self):
        ii, jj = np.mgrid[0:self.n, 0:self.n]
        return self.ij_to_xy(ii, jj)


def blank_terrain(lat0, lon0, half, cell, source, name=""):
    n = int(round(2 * half / cell))
    half = n * cell / 2
    z = np.zeros((n, n))
    return Terrain(lat0, lon0, half, cell, z.copy(), z.copy(), np.zeros((n, n), np.int8), source, name)


# ------------------------------------------------------------------ rasterization helpers
def _fill(t, cls, fh, poly, c, h):
    if len(poly) < 3:
        return
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    if x1 < -t.half or y1 < -t.half or x0 > t.half or y0 > t.half:
        return
    i0, j0 = (int(v) for v in t.xy_to_ij(x0, y0))
    i1, j1 = (int(v) for v in t.xy_to_ij(x1, y1))
    ii, jj = np.mgrid[i0:i1 + 1, j0:j1 + 1]
    cx, cy = t.ij_to_xy(ii, jj)
    inside = Path(poly).contains_points(np.c_[cx.ravel(), cy.ravel()]).reshape(ii.shape)
    if inside.any():
        si, sj = ii[inside], jj[inside]
    else:                                   # footprint smaller than one cell: mark its centroid cell
        mx, my = poly[:, 0].mean(), poly[:, 1].mean()
        if abs(mx) > t.half or abs(my) > t.half:
            return
        ci, cj = t.xy_to_ij(mx, my)
        si, sj = np.atleast_1d(ci), np.atleast_1d(cj)
    cls[si, sj] = c
    fh[si, sj] = np.maximum(fh[si, sj], h) if c == BUILDING else h


def _stroke(t, cls, fh, line, c, h, width):
    r = max(0, int(round(width / 2 / t.cell)))
    for (xa, ya), (xb, yb) in zip(line[:-1], line[1:]):
        m = max(2, int(math.hypot(xb - xa, yb - ya) / (t.cell / 2)) + 1)
        xs, ys = np.linspace(xa, xb, m), np.linspace(ya, yb, m)
        keep = (np.abs(xs) <= t.half) & (np.abs(ys) <= t.half)
        if not keep.any():
            continue
        i, j = t.xy_to_ij(xs[keep], ys[keep])
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                ii, jj = np.clip(i + di, 0, t.n - 1), np.clip(j + dj, 0, t.n - 1)
                cls[ii, jj] = c
                fh[ii, jj] = h


def _height(tags):
    for k in ("height", "building:height"):
        if k in tags:
            try:
                return float(str(tags[k]).split()[0].rstrip("m"))
            except ValueError:
                pass
    if "building:levels" in tags:
        try:
            return 3.2 * float(tags["building:levels"])
        except ValueError:
            pass
    return 8.0


def _classify(tags):
    if "building" in tags:
        return "building"
    if tags.get("natural") == "wood" or tags.get("landuse") == "forest":
        return "forest"
    if tags.get("natural") == "scrub":
        return "scrub"
    if (tags.get("natural") == "water" or tags.get("landuse") in ("reservoir", "basin")
            or tags.get("waterway") == "riverbank"):
        return "water_area"
    if tags.get("waterway") in ("river", "canal"):
        return "water_line"
    hw = tags.get("highway")
    if hw and hw not in SKIP_HIGHWAY:
        return "road"
    return None


def rasterize(t, elements):
    """Burn OSM ways (Overpass `out geom` JSON) into the class and feature-height grids."""
    cls = np.zeros((t.n, t.n), np.int8)
    fh = np.zeros((t.n, t.n))
    groups = {k: [] for k in ("forest", "scrub", "water_area", "water_line", "road", "building")}
    for el in elements:
        if el.get("type") != "way" or not el.get("geometry"):
            continue
        tags = el.get("tags", {})
        kind = _classify(tags)
        if kind is None:
            continue
        lon = np.array([p["lon"] for p in el["geometry"]])
        lat = np.array([p["lat"] for p in el["geometry"]])
        x, y = t.lonlat_to_xy(lon, lat)
        groups[kind].append((np.c_[x, y], tags))
    for poly, _ in groups["forest"]:
        _fill(t, cls, fh, poly, FOREST, 15.0)
    for poly, _ in groups["scrub"]:
        _fill(t, cls, fh, poly, FOREST, 3.0)
    for poly, _ in groups["water_area"]:
        _fill(t, cls, fh, poly, WATER, 0.0)
    for line, tags in groups["water_line"]:
        _stroke(t, cls, fh, line, WATER, 0.0, 20 if tags.get("waterway") == "river" else 10)
    for line, tags in groups["road"]:          # roads after water, so crossings act as bridges
        _stroke(t, cls, fh, line, ROAD, 0.0, ROAD_WIDTH.get(tags.get("highway"), 8))
    for poly, tags in groups["building"]:
        _fill(t, cls, fh, poly, BUILDING, _height(tags))
    t.cls = cls
    t.surface = t.ground + fh
    return {k: len(v) for k, v in groups.items()}


# ------------------------------------------------------------------ network sources
def geocode(query, timeout=20):
    import requests
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params=dict(q=query, format="json", limit=1),
                     headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    if not js:
        raise ValueError(f"No place matched '{query}'. Try a more specific name, or enter coordinates.")
    return float(js[0]["lat"]), float(js[0]["lon"]), js[0].get("display_name", query)


def fetch_osm(t, timeout=90):
    w, s, e, n = t.bounds_lonlat()
    bb = f"{s},{w},{n},{e}"
    q = f"""[out:json][timeout:60];
(way["building"]({bb});
 way["natural"~"^(wood|scrub|water)$"]({bb});
 way["landuse"~"^(forest|reservoir|basin)$"]({bb});
 way["waterway"~"^(riverbank|river|canal)$"]({bb});
 way["highway"]({bb}););
out tags geom;"""
    return overpass_query(q, timeout)


def overpass_query(q, timeout=90):
    """Send an Overpass query to every mirror at once and return the first successful answer.

    Public mirrors swing between seconds and timeouts from one request to the next, so racing
    them costs little and avoids waiting out a slow mirror before trying the next.
    """
    import concurrent.futures as cf
    import requests

    def post(url):
        r = requests.post(url, data={"data": q}, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        r.raise_for_status()
        return r.json()["elements"]

    pool = cf.ThreadPoolExecutor(len(OVERPASS_URLS))
    futures = [pool.submit(post, url) for url in OVERPASS_URLS]
    last = None
    try:
        for f in cf.as_completed(futures):
            try:
                return f.result()
            except Exception as ex:  # wait for the other mirrors
                last = ex
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    raise RuntimeError(f"OpenStreetMap request failed on every Overpass mirror: {last}")


def _tile_xy(lon, lat, z):
    N = 2 ** z
    return (np.asarray(lon) + 180) / 360 * N, (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * N


def fetch_dem(t, z=13, timeout=30):
    """Sample AWS Terrain Tiles (Terrarium encoding) at every cell centre."""
    import requests
    from PIL import Image
    from scipy.ndimage import map_coordinates
    w, s, e, n = t.bounds_lonlat()
    x0f, y0f = _tile_xy(w, n, z)
    x1f, y1f = _tile_xy(e, s, z)
    tx0, ty0, tx1, ty1 = int(x0f), int(y0f), int(x1f), int(y1f)
    if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > 16:
        raise ValueError("Area needs more than 16 elevation tiles; reduce the area size.")
    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{tx}/{ty}.png"
            r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            a = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB")).astype(float)
            mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = (
                a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768)
    x, y = t.cell_centres()
    lon, lat = t.xy_to_lonlat(x, y)
    px, py = _tile_xy(lon, lat, z)
    px = (px - tx0) * 256 - 0.5
    py = (py - ty0) * 256 - 0.5
    return map_coordinates(mosaic, [py.ravel(), px.ravel()], order=1, mode="nearest").reshape(t.n, t.n)


def osm_terrain(lat0, lon0, half=2000.0, cell=25.0, name=""):
    t = blank_terrain(lat0, lon0, half, cell, "OpenStreetMap", name)
    try:
        t.ground = fetch_dem(t)
        t.notes.append("Elevation: AWS Terrain Tiles (zoom 13).")
    except Exception as ex:
        t.notes.append(f"Elevation unavailable ({ex}); ground treated as flat.")
    counts = rasterize(t, fetch_osm(t))
    t.notes.append("OSM ways: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in counts.items() if v))
    t.fetched_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    return t


# ------------------------------------------------------------------ synthetic ground
def synthetic_terrain(half=2000.0, cell=25.0, seed=4, building_density=0.003, forest_share=0.07,
                      relief=30.0, road_density=0.016, river=True, name=None):
    """Synthetic ground at Null Island with controllable descriptors, for offline work and for
    sampling the terrain space when calibrating (see sim.descriptors).

    building_density: target share of cells that are building; forest_share: share of cells that
    are wood; relief: peak height (m) of the main ridge and hill; road_density: target share of
    road cells; river: a north-south river with a road bridge.
    """
    rng = np.random.default_rng(seed)
    t = blank_terrain(0.0, 0.0, half, cell, "Synthetic",
                      name or f"Synthetic (b={building_density:.3f}, f={forest_share:.2f}, relief={relief:.0f} m)")
    x, y = t.cell_centres()
    g = 120 + relief * np.exp(-((x - 1050) / 180) ** 2)
    g += relief * np.exp(-(((x - 600) ** 2 + (y - 650) ** 2) / 250 ** 2))
    g += 0.6 * relief * np.exp(-(((x + 700) ** 2 + (y + 500) ** 2) / 400 ** 2))
    for _ in range(12):
        cx, cy = rng.uniform(-half, half, 2)
        g += rng.uniform(0.1, 0.3) * relief * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / rng.uniform(150, 350) ** 2))
    t.ground = g
    cls = np.zeros((t.n, t.n), np.int8)
    fh = np.zeros((t.n, t.n))
    # woods: blobs placed until the target share is met
    target = forest_share * t.n * t.n
    tries = 0
    while (cls == FOREST).sum() < target and tries < 60:
        cx, cy = rng.uniform(-half * 0.9, half * 0.9, 2)
        rad = rng.uniform(150, 450)
        blob = np.hypot(x - cx, (y - cy) * rng.uniform(0.6, 1.4)) < rad
        cls[blob], fh[blob] = FOREST, 15.0
        tries += 1
    if river:
        _stroke(t, cls, fh, np.array([[1350, -half], [1300, 0], [1380, half]]), WATER, 0.0, 30)
    roads = [np.array([[-half, 0], [half, 0]]), np.array([[0, -half], [0, half]])]
    n_extra = int(max(0, road_density * t.n * t.n / (2 * half / cell) - 2))
    for _ in range(n_extra):
        a = rng.uniform(-half, half, 2)
        b = rng.uniform(-half, half, 2)
        roads.append(np.array([a, b]))
    for k, line in enumerate(roads):
        _stroke(t, cls, fh, line, ROAD, 0.0, 12 if k < 2 else 8)
    # buildings: a village around the objective, sized to the target density
    n_b = int(building_density * t.n * t.n * (cell / 25.0) ** 2)
    placed, tries = 0, 0
    rmax = 250 + 40 * np.sqrt(n_b)
    while placed < n_b and tries < 20 * n_b + 50:
        tries += 1
        r, a = rng.uniform(40, rmax), rng.uniform(0, 2 * np.pi)
        cx, cy = r * np.cos(a), r * np.sin(a)
        if abs(cy) < 20 or abs(cx) < 15 or abs(cx) > half - 50 or abs(cy) > half - 50:
            continue
        w, h = rng.uniform(12, 22, 2)
        poly = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                         [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
        _fill(t, cls, fh, poly, BUILDING, rng.uniform(6, 12))
        placed += 1
    t.cls = cls
    t.surface = t.ground + fh
    t.notes.append("Synthetic ground for offline use; the basemap under it is empty ocean.")
    return t


def demo_terrain(half=2000.0, cell=25.0, seed=4):
    """The default offline ground: a village objective, a ridge, a river with a bridge, woods."""
    t = synthetic_terrain(half, cell, seed, name="Synthetic valley (offline)")
    return t


# ------------------------------------------------------------------ line of sight
def los_fraction(t, obs_xy, eye_h, tgt_xy, tgt_h, samples=160, chunk=4000):
    """Fraction of sampled target heights visible from each observer.

    obs_xy, tgt_xy: (P, 2); eye_h: (P,) metres above ground; tgt_h: (P, H) heights above ground.
    Cells within ~one cell of either end are ignored so a unit's own building or wood line
    doesn't block its own sight.
    """
    obs_xy, tgt_xy = np.asarray(obs_xy, float), np.asarray(tgt_xy, float)
    P = len(obs_xy)
    eye_h = np.broadcast_to(np.asarray(eye_h, float), (P,))
    tgt_h = np.asarray(tgt_h, float)
    if tgt_h.ndim == 1:
        tgt_h = np.broadcast_to(tgt_h, (P, len(tgt_h)))
    tk = (np.arange(samples) + 0.5) / samples
    out = np.empty(P)
    for a in range(0, P, chunk):
        b = min(P, a + chunk)
        o, g = obs_xy[a:b], tgt_xy[a:b]
        d = np.hypot(*(g - o).T)
        px = o[:, :1] + (g[:, :1] - o[:, :1]) * tk
        py = o[:, 1:] + (g[:, 1:] - o[:, 1:]) * tk
        surf = t.at(t.surface, px, py)
        along = d[:, None] * tk
        valid = (along > 0.75 * t.cell) & (along < d[:, None] - 0.75 * t.cell)
        z0 = t.at(t.ground, o[:, 0], o[:, 1]) + eye_h[a:b]
        g1 = t.at(t.ground, g[:, 0], g[:, 1])
        req = np.where(valid, (surf - z0[:, None]) / tk + z0[:, None] - g1[:, None], -np.inf).max(1)
        out[a:b] = (tgt_h[a:b] > req[:, None]).mean(1)
    return out
