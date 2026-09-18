"""APP-6 style unit symbols, drawn to a single icon atlas for deck.gl.

deck.gl's IconLayer wants one image holding every symbol plus a mapping from name to box, so
the whole set is drawn once with Pillow and cached. Symbols follow APP-6 / MIL-STD-2525 closely
enough to read at a glance without claiming to be a compliant renderer: a frame carries
affiliation (rectangle friendly, diamond hostile) and the fill carries the type (crossed
diagonals infantry, oval armour, single diagonal recon, chevron anti-tank).

Sizes are in pixels on screen, so a symbol stays the same size as the map zooms.
"""
from __future__ import annotations

import base64
import io
from functools import lru_cache

from PIL import Image, ImageDraw

from .units import IFV, INF, R_AT, R_INF, SCOUT, TANK

CELL = 64                      # atlas cell, px
SS = 4                         # supersample factor, for smooth edges
FRIEND_RGB = (31, 95, 168)
HOSTILE_RGB = (194, 59, 34)
DEAD_RGB = (122, 122, 126)
WHITE = (255, 255, 255)

# name -> (affiliation, type, state); state is "ok", "dead" or "unseen"
TYPE_KEY = {TANK: "tank", IFV: "ifv", INF: "inf", SCOUT: "scout", R_AT: "at", R_INF: "inf"}


def symbol_name(btype, hostile, dead=False, unseen=False):
    """The atlas key for one unit's current state."""
    state = "dead" if dead else ("unseen" if unseen else "ok")
    return f"{'hos' if hostile else 'fr'}-{TYPE_KEY[btype]}-{state}"


NAMES = [f"{aff}-{ty}-{st}"
         for aff in ("fr", "hos")
         for ty in ("tank", "ifv", "inf", "scout", "at")
         for st in ("ok", "dead", "unseen")]


def _frame_points(hostile, w, h, pad):
    """Outline of the affiliation frame inside a cell of w x h."""
    if hostile:                                     # diamond, point up
        cx, cy = w / 2, h / 2
        rx, ry = w / 2 - pad, h / 2 - pad
        return [(cx, cy - ry), (cx + rx, cy), (cx, cy + ry), (cx - rx, cy)]
    x0, y0 = pad, h / 2 - (h / 2 - pad) * 0.74      # rectangle, wider than tall
    return [(x0, y0), (w - pad, y0), (w - pad, h - y0), (x0, h - y0)]


def _fill_marks(draw, kind, box, colour, width):
    """The type mark drawn inside the frame."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    inset_x, inset_y = w * 0.16, h * 0.16
    a, b = (x0 + inset_x, y0 + inset_y), (x1 - inset_x, y1 - inset_y)
    if kind in ("inf", "ifv"):                      # crossed diagonals
        draw.line([a, b], fill=colour, width=width)
        draw.line([(a[0], b[1]), (b[0], a[1])], fill=colour, width=width)
    if kind in ("tank", "ifv"):                     # armour oval
        cy = (y0 + y1) / 2
        ry = h * 0.22
        draw.ellipse([x0 + w * 0.2, cy - ry, x1 - w * 0.2, cy + ry], outline=colour, width=width)
    if kind == "scout":                             # single diagonal, recon
        draw.line([(a[0], b[1]), (b[0], a[1])], fill=colour, width=width)
    if kind == "at":                                # chevron, anti-tank
        cx = (x0 + x1) / 2
        draw.line([(a[0], b[1]), (cx, a[1]), (b[0], b[1])], fill=colour, width=width, joint="curve")


def _draw_symbol(name):
    """One cell, drawn large and shrunk down so the strokes are smooth."""
    aff, kind, state = name.split("-")
    hostile = aff == "hos"
    size = CELL * SS
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    edge = DEAD_RGB if state == "dead" else (HOSTILE_RGB if hostile else FRIEND_RGB)
    stroke = max(2, int(2.6 * SS))
    pad = 5 * SS

    alpha = 130 if state == "unseen" else 255      # a contact Blue has not seen yet is ghosted
    pts = _frame_points(hostile, size, size, pad)
    fill = (*WHITE, 90) if state == "unseen" else (*WHITE, 235 if state != "dead" else 200)
    d.polygon(pts, fill=fill, outline=(*edge, alpha), width=stroke)

    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    box = (min(xs), min(ys), max(xs), max(ys))
    if hostile:                                     # keep the mark inside the diamond
        cx, cy = size / 2, size / 2
        r = (box[2] - box[0]) / 2 * 0.62
        box = (cx - r, cy - r, cx + r, cy + r)
    _fill_marks(d, kind, box, (*edge, alpha), stroke)

    if state == "dead":                             # struck through, inside the frame
        d.line([(box[0], box[3]), (box[2], box[1])], fill=(*DEAD_RGB, 255), width=stroke)
    return img.resize((CELL, CELL), Image.LANCZOS)


# ------------------------------------------------------------------ map geometry
# The same symbols as vector geometry, for drawing on the map. deck.gl's IconLayer renders
# nothing in the deck build Streamlit ships, so the frames go out as polygons and the type
# marks as paths, which the Path and Polygon layers draw reliably. Coordinates below are in a
# unit square centred on (0, 0); the caller scales them to metres and offsets them to the unit.
HALF_W, HALF_H = 0.5, 0.37                       # friendly rectangle
DIAG = 0.5                                       # hostile diamond half-diagonal


def _unit_frame(hostile):
    if hostile:
        return [(0, DIAG), (DIAG, 0), (0, -DIAG), (-DIAG, 0)]
    return [(-HALF_W, -HALF_H), (HALF_W, -HALF_H), (HALF_W, HALF_H), (-HALF_W, HALF_H)]


def _unit_marks(kind, hostile):
    """Type marks as a list of polylines in the unit square."""
    x, y = (DIAG * 0.44, DIAG * 0.44) if hostile else (HALF_W * 0.62, HALF_H * 0.62)
    out = []
    if kind in ("inf", "ifv"):                   # crossed diagonals
        out += [[(-x, -y), (x, y)], [(-x, y), (x, -y)]]
    if kind in ("tank", "ifv"):                  # armour oval, as a closed polyline
        import math
        rx, ry = x * 1.05, y * 0.62
        out.append([(rx * math.cos(a), ry * math.sin(a))
                    for a in [i * math.pi / 8 for i in range(17)]])
    if kind == "scout":                          # single diagonal, recon
        out.append([(-x, -y), (x, y)])
    if kind == "at":                             # chevron, anti-tank
        out.append([(-x, -y), (0, y), (x, -y)])
    return out


def geometry(btype, hostile, size_m):
    """(frame polygon, mark polylines) in metres around the unit, ready to offset and project."""
    kind = TYPE_KEY[btype]
    s = size_m
    frame = [(px * s, py * s) for px, py in _unit_frame(hostile)]
    marks = [[(px * s, py * s) for px, py in line] for line in _unit_marks(kind, hostile)]
    return frame, marks


def strike(size_m):
    """The line struck through a destroyed unit."""
    h = size_m * 0.5
    return [(-h, -h), (h, h)]


@lru_cache(maxsize=64)
def icon(name):
    """One symbol as a PNG data URI (used by the HTML legend, not by the map)."""
    buf = io.BytesIO()
    _draw_symbol(name).save(buf, format="PNG")
    return {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            "width": CELL, "height": CELL, "anchorX": CELL // 2, "anchorY": CELL // 2, "mask": False}


@lru_cache(maxsize=1)
def atlas():
    """(data URI, icon mapping) for every symbol, laid out in one row."""
    sheet = Image.new("RGBA", (CELL * len(NAMES), CELL), (0, 0, 0, 0))
    mapping = {}
    for i, name in enumerate(NAMES):
        sheet.paste(_draw_symbol(name), (i * CELL, 0))
        mapping[name] = {"x": i * CELL, "y": 0, "width": CELL, "height": CELL,
                         "anchorX": CELL // 2, "anchorY": CELL // 2, "mask": False}
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), mapping
