"""Run manifests: everything needed to reproduce a result."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess

import numpy as np
import scipy


def _pkg_root():
    return os.path.dirname(os.path.abspath(__file__))


SOURCE_PATHS = ["sim", "scripts", "requirements-lock.txt"]   # the code a script's result depends on


def dirty_source():
    """Tracked files under SOURCE_PATHS that differ from HEAD, as a sorted list of paths.

    The Streamlit apps and the tests are deliberately excluded: no script imports them, so
    editing them cannot change a result. Untracked files and regenerated artifacts under runs/
    are excluded for the same reason.
    """
    root = os.path.dirname(_pkg_root())
    try:
        out = subprocess.check_output(["git", "-C", root, "status", "--porcelain", "-uno", "--"]
                                      + SOURCE_PATHS, stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return []
    return sorted(line[3:].strip() for line in out.splitlines() if line.strip())


def code_version():
    """Git commit if the package lives in a repository, else a hash of the source files.

    "+dirty" means the *simulation source* differs from the commit; dirty_source() names which
    files. See SOURCE_PATHS for what counts.
    """
    root = os.path.dirname(_pkg_root())
    try:
        sha = subprocess.check_output(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL, text=True).strip()
        return f"git:{sha}{'+dirty' if dirty_source() else ''}"
    except Exception:
        h = hashlib.sha1()
        for name in sorted(os.listdir(_pkg_root())):
            if name.endswith(".py"):
                h.update(open(os.path.join(_pkg_root(), name), "rb").read())
        return "src:" + h.hexdigest()[:12]


def params_hash(P):
    parts = []
    for k in sorted(P):
        v = P[k]
        parts.append(k + "=" + (np.asarray(v).tobytes().hex() if isinstance(v, np.ndarray) else json.dumps(v)))
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def make_manifest(P, seed, terrain, roster, scenario_cfg=None, extra=None):
    m = dict(created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             code_version=code_version(), dirty_source=dirty_source(),
             params_hash=params_hash(P), seed=seed,
             roster=roster.name, roster_blue=roster.btype.tolist(), roster_red=roster.rtype.tolist(),
             terrain=dict(source=terrain.source, name=terrain.name, lat0=terrain.lat0, lon0=terrain.lon0,
                          half=terrain.half, cell=terrain.cell, fetched_at=terrain.fetched_at,
                          surface_hash=hashlib.sha1(terrain.surface.tobytes()).hexdigest()[:12]),
             versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    if scenario_cfg is not None:
        m["scenario"] = {k: getattr(scenario_cfg, k) for k in scenario_cfg.__dataclass_fields__}
    if extra:
        m.update(extra)
    return m


def save_manifest(m, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(m, open(path, "w"), indent=1, default=str)
