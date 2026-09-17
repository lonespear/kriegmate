"""Streamlit front end: real ground from OpenStreetMap, COA comparison, and replay."""
import base64
import io
import math
import time

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from PIL import Image

from sim import stats
from sim.engine import TICK_DEFAULT, concat_results, simulate
from sim.manifest import make_manifest
from sim.scenario import Scenario, ScenarioConfig
from sim.terrain import CLASS_NAMES, demo_terrain, geocode, osm_terrain
from sim.units import (COMPANY, TANK, B_LABELS, BTYPE, IS_INF, IS_SCOUT, NB, NR, R_LABELS, RTYPE,
                       TYPE_NAMES, default_params)

st.set_page_config(page_title="Company-team attack on real ground", page_icon="🗺️", layout="wide")

FRIEND = [31, 95, 168]
HOSTILE = [194, 59, 34]
FIRES = [217, 154, 0]
OLIVE = [94, 107, 50]
DEAD = [128, 128, 128]
CLASS_RGB = np.array([[226, 229, 214], [250, 250, 244], [118, 148, 92], [92, 92, 104], [110, 160, 200]])
ARMOR = {"SBF": "Armor supports by fire", "Maneuver": "Armor maneuvers with infantry",
         "Unsupported": "Armor attacks alone"}
RED_COA = {"Early": "Engage at maximum range", "Late": "Hold fire until trigger range"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea, label { font-family: 'Barlow', system-ui, sans-serif; }
h1, h2, h3, h4 { font-family: 'Barlow Condensed', 'Barlow', sans-serif !important; letter-spacing: .01em; }
h1 { font-weight: 700 !important; font-size: 2.2rem !important; padding-bottom: 0 !important; }
.lede { color: #4A544B; max-width: 72ch; margin: .1rem 0 1rem; line-height: 1.5; }
.key { display: inline-flex; align-items: center; gap: .35rem; margin-right: 1.1rem; font-size: .92rem; }
.swatch { width: .8rem; height: .8rem; border-radius: 50%; display: inline-block; }
.ring { width: .8rem; height: .8rem; border-radius: 50%; display: inline-block; border: 2px solid rgb(217,154,0); }
.clock { font-family: 'Barlow Condensed', sans-serif; font-size: 1.7rem; font-weight: 600;
         font-variant-numeric: tabular-nums; line-height: 1.1; }
.coa { color: #4A544B; font-size: .95rem; }
.feed { font-variant-numeric: tabular-nums; font-size: .88rem; line-height: 1.55; min-height: 12rem; }
.feed .k { color: rgb(194,59,34); font-weight: 600; }
.fine { color: #5d665e; font-size: .82rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- cached loaders
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def cached_geocode(q):
    return geocode(q)


@st.cache_data(show_spinner=False, ttl=24 * 3600, max_entries=8)
def cached_osm(lat, lon, half, cell, name):
    return osm_terrain(lat, lon, half, cell, name)


@st.cache_data(show_spinner=False, max_entries=8)
def terrain_png(_t, version):
    gy, gx = np.gradient(_t.ground, _t.cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt_ = np.radians(315), np.radians(45)
    shade = np.clip(np.sin(alt_) * np.cos(slope) + np.cos(alt_) * np.sin(slope) * np.cos(az - aspect), 0, 1)
    rgb = CLASS_RGB[_t.cls] * (0.55 + 0.45 * shade[..., None])
    img = np.dstack([np.clip(rgb, 0, 255), np.full(_t.cls.shape, 215)]).astype(np.uint8)[::-1]
    buf = io.BytesIO()
    Image.fromarray(img, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def state_default(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


state_default("terrain", None)
state_default("terrain_version", 0)
state_default("runs", None)

if st.session_state.terrain is None:
    st.session_state.terrain = demo_terrain()
    st.session_state.terrain_version += 1


# ---------------------------------------------------------------- sidebar: the order
with st.sidebar:
    st.header("Ground")
    source = st.radio("Terrain source", ["OpenStreetMap", "Offline demo"], horizontal=True)
    place, lat, lon, mode = None, None, None, None
    if source == "OpenStreetMap":
        mode = st.radio("Find the area by", ["Place name", "Coordinates"], horizontal=True)
        if mode == "Place name":
            place = st.text_input("Place", "Camp Buckner, West Point, NY")
        else:
            lat = st.number_input("Latitude", -85.0, 85.0, 41.3915, format="%.5f")
            lon = st.number_input("Longitude", -180.0, 180.0, -73.9560, format="%.5f")
    half = st.slider("Area half-width (m)", 1000, 3000, 2000, 250)
    cell = st.select_slider("Grid cell (m)", [20, 25, 30, 40], 25)
    if st.button("Load ground", type="primary", width="stretch"):
        try:
            with st.spinner("Loading ground and elevation…"):
                if source == "Offline demo":
                    t_new = demo_terrain(half, cell)
                else:
                    if mode == "Place name":
                        lat, lon, name = cached_geocode(place.strip())
                    else:
                        name = f"{lat:.4f}, {lon:.4f}"
                    t_new = cached_osm(float(lat), float(lon), float(half), float(cell), name)
            st.session_state.terrain = t_new
            st.session_state.terrain_version += 1
            st.session_state.runs = None
            st.session_state.pop("load_error", None)
        except Exception as ex:
            st.session_state.load_error = str(ex)
    if st.session_state.get("load_error"):
        st.error(f"Ground didn't load: {st.session_state.load_error}\n\n"
                 "Check the place name, try a smaller area, or switch to the offline demo.")

    t = st.session_state.terrain
    max_start = int(t.half - t.cell)

    def ring(label, lo, hi, key):
        lo, hi = min(lo, max_start - 100), min(hi, max_start)
        return st.slider(label, 200, max_start, (lo, hi), 50, key=f"{key}_{max_start}")

    st.header("Scenario")
    bearing = st.slider("Blue attacks from (bearing, degrees)", 0, 355, 90, 5,
                        help="Direction from the objective to Blue's start line. 90 means Blue comes from the east.")
    start_dist = st.slider("Start line distance (m)", 400, max_start, min(1700, max_start), 50,
                           key=f"start_{max_start}")
    sbf_ring = ring("Support-by-fire ring (m)", 900, 1400, "sbf")
    op_ring = ring("Observation-post ring (m)", 1000, 1500, "op")
    dismount = st.slider("Dismount distance (m)", 100, 1200, 450, 50)
    red_radius = st.slider("Defense radius (m)", 100, 600, 300, 25)
    danger = st.slider("Danger close (m)", 100, 600, 250, 25)
    minutes = st.slider("Time limit (min)", 10, 120, 60, 5)

    st.header("Courses of action")

    def coa_inputs(tag, armor, recon, fires):
        st.subheader(f"Blue COA {tag}")
        a = st.selectbox("Armor role", list(ARMOR), index=list(ARMOR).index(armor),
                         format_func=ARMOR.get, key=f"armor_{tag}")
        c1, c2 = st.columns(2)
        r = c1.toggle("Recon", recon, key=f"recon_{tag}")
        f = c2.toggle("Fires", fires, key=f"fires_{tag}")
        return a, r, f

    coa_a = coa_inputs("A", "SBF", True, True)
    coa_b = coa_inputs("B", "Maneuver", True, True)
    st.subheader("Red")
    red_coa = st.radio("Fire discipline", list(RED_COA), format_func=RED_COA.get)
    trigger = st.slider("Trigger range (m)", 150, 2000, 600, 50, disabled=red_coa == "Early")

    st.header("Run")
    n_reps = st.select_slider("Replications per COA", [200, 500, 1000, 2000, 5000], 1000)
    n_plans = st.slider("Red plans sampled", 1, 5, 1,
                        help="With 1, the defense always takes its best-scored positions. With more, each run "
                             "draws the defense from the good positions, so results average over Red's plan "
                             "instead of conditioning on one. The map and replay show the first plan.")
    tick_s = st.select_slider("Time step (s)", [3.75, 7.5, 15.0], TICK_DEFAULT,
                              help="7.5 s is the converged default; 15 s runs twice as fast but is not converged "
                                   "(see README, Discretization).")
    n_log = st.slider("Replays kept per COA", 5, 50, 20, 5)
    seed = int(st.number_input("Seed", 0, 1_000_000, 2026))
    run_clicked = st.button("Run both COAs", type="primary", width="stretch")

cfg = ScenarioConfig(bearing_deg=float(bearing), start_dist=float(start_dist),
                     sbf_min=float(sbf_ring[0]), sbf_max=float(sbf_ring[1]),
                     op_min=float(op_ring[0]), op_max=float(op_ring[1]),
                     dismount_dist=float(dismount), red_radius=float(red_radius),
                     danger_close=float(danger), max_minutes=float(minutes))

scen_key = (st.session_state.terrain_version, cfg, n_plans)
scen, scen_error, plans = None, None, []
if st.session_state.get("scen_key") == scen_key:
    plans = st.session_state.plans
    scen = plans[0]
else:
    try:
        with st.spinner("Placing units and computing lines of sight…"):
            plans = [Scenario(t, cfg, COMPANY, seed=p, placement_temperature=0.0 if n_plans == 1 else 0.5)
                     for p in range(n_plans)]
        scen = plans[0]
        st.session_state.plans, st.session_state.scen_key = plans, scen_key
    except Exception as ex:
        scen_error = str(ex)

run_key = (scen_key, coa_a, coa_b, red_coa, trigger, n_reps, n_log, seed, tick_s)
if run_clicked and scen is not None:
    with st.spinner("Running both courses of action…"):
        res = {}
        per = max(20, n_reps // n_plans)
        for tag, (a, r, f) in (("A", coa_a), ("B", coa_b)):
            parts = [simulate(pl.tables(a), a, r, f, red_coa, per, seed=[seed, p], red_trigger=float(trigger),
                              n_log=n_log if p == 0 else 0, tick_s=float(tick_s))
                     for p, pl in enumerate(plans)]
            res[tag] = concat_results(parts)
    manifest = make_manifest(default_params(), seed, t, COMPANY, cfg,
                             extra=dict(coa_a=coa_a, coa_b=coa_b, red_coa=red_coa, red_trigger=trigger,
                                        replications=per * n_plans, red_plans=n_plans, tick_s=tick_s))
    st.session_state.runs = dict(res=res, coas={"A": coa_a, "B": coa_b}, key=run_key,
                                 red=red_coa, n=per * n_plans, manifest=manifest)
    st.session_state.pop("tick", None)


# ---------------------------------------------------------------- map helpers
def to_lonlat(xy):
    lon, lat = t.xy_to_lonlat(np.asarray(xy)[..., 0], np.asarray(xy)[..., 1])
    return np.stack([lon, lat], -1)


def view_state():
    mpp = 2 * t.half / 760
    zoom = math.log2(156543.03 * math.cos(math.radians(t.lat0)) / mpp)
    return pdk.ViewState(latitude=t.lat0, longitude=t.lon0, zoom=zoom, pitch=0)


def make_deck(layers):
    w, s, e, n = t.bounds_lonlat()
    base = pdk.Layer("BitmapLayer", data=None, image=terrain_png(t, st.session_state.terrain_version),
                     bounds=[w, s, e, n], opacity=0.9)
    return pdk.Deck(layers=[base] + layers, initial_view_state=view_state(),
                    map_provider="carto", map_style="light", tooltip={"text": "{tip}"})


def points(xy, color, tips, radius, name, stroked=False, line_color=None):
    ll = to_lonlat(xy)
    data = [{"pos": [float(a), float(b)], "tip": tip} for (a, b), tip in zip(ll, tips)]
    return pdk.Layer("ScatterplotLayer", data=data, id=name, get_position="pos",
                     get_fill_color=color, get_radius=radius, radius_min_pixels=4, pickable=True,
                     stroked=stroked, filled=not stroked or line_color is None,
                     get_line_color=line_color or [30, 30, 30], line_width_min_pixels=1.5)


def labels(xy, texts, name, color=(28, 35, 33)):
    ll = to_lonlat(xy)
    data = [{"pos": [float(a), float(b)], "text": s, "tip": s} for (a, b), s in zip(ll, texts)]
    return pdk.Layer("TextLayer", data=data, id=name, get_position="pos", get_text="text",
                     get_size=13, get_color=list(color), get_pixel_offset=[0, -16],
                     font_family="Barlow Condensed, Barlow, sans-serif", font_weight=600)


def coa_label(coa):
    a, r, f = coa
    extras = [x for x, on in (("recon", r), ("fires", f)) if on]
    return ARMOR[a] + (", with " + " and ".join(extras) if extras else ", no recon or fires")


def mmss(tick, tick_s):
    s = int(tick * tick_s)
    return f"{s // 60:02d}:{s % 60:02d}"


def legend(items):
    parts = []
    for c, txt, kind in items:
        if kind == "ring":
            sw = '<span class="ring"></span>'
        else:
            op = ";opacity:.35" if kind == "faint" else ""
            sw = f'<span class="swatch" style="background:rgb({c[0]},{c[1]},{c[2]}){op}"></span>'
        parts.append(f'<span class="key">{sw}{txt}</span>')
    st.markdown("".join(parts), unsafe_allow_html=True)


# ---------------------------------------------------------------- page
st.title("Company-team attack on real ground")
st.markdown(
    '<p class="lede">A Blue company team attacks a dug-in, anti-tank-heavy defense. Ground from '
    'OpenStreetMap sets line of sight, cover, and routes. Each course of action runs as a Monte Carlo '
    'experiment on shared random numbers, so COA A and COA B can be compared replication by replication '
    'and replayed side by side.</p>', unsafe_allow_html=True)

if scen_error:
    st.error(f"The scenario can't be built on this ground: {scen_error}")
    st.stop()

runs = st.session_state.runs
stale = runs is not None and runs["key"] != run_key
tab_ground, tab_results, tab_losses, tab_play = st.tabs(["Ground", "Results", "Where losses happen", "Playback"])

# ---------------------------------------------------------------- Ground
with tab_ground:
    left, right = st.columns([3, 1.2])
    with left:
        armor_shown = coa_a[0]
        routes = scen.routes_xy(armor_shown)
        path_data = []
        for u, xy in enumerate(routes):
            if (IS_SCOUT[u] and not coa_a[1]) or (IS_INF[u] and armor_shown == "Unsupported"):
                continue
            col = OLIVE if IS_SCOUT[u] else FRIEND
            path_data.append({"path": to_lonlat(xy).tolist(), "color": col + [170],
                              "tip": f"{B_LABELS[u]} route ({TYPE_NAMES[BTYPE[u]]})"})
        layers = [
            pdk.Layer("PathLayer", data=path_data, id="routes", get_path="path", get_color="color",
                      width_min_pixels=2, pickable=True),
            points(np.array([scen._xy(nd) for nd in scen.sbf_nodes]), OLIVE + [230],
                   ["Support-by-fire position"] * len(scen.sbf_nodes), 30, "sbf"),
            points(np.array([scen._xy(nd) for nd in scen.op_nodes]), [255, 255, 255, 230],
                   ["Observation post"] * len(scen.op_nodes), 30, "op", stroked=True, line_color=OLIVE),
            points(scen.dismount_points(), [255, 255, 255, 220], ["Dismount point"] * 6, 14, "dis",
                   stroked=True, line_color=FRIEND),
            points(scen.rpos, HOSTILE + [235],
                   [f"{R_LABELS[j]} {TYPE_NAMES[RTYPE[j]]}{', in a building' if scen.r_building[j] else ''}"
                    for j in range(NR)], 18, "red"),
            labels(scen.rpos, R_LABELS, "red_lbl", HOSTILE),
            labels(np.array([[0.0, 0.0]]), ["OBJ"], "obj"),
        ]
        st.pydeck_chart(make_deck(layers), height=620)
        legend([(FRIEND, "Blue routes (COA A)", "solid"), (OLIVE, "Support-by-fire and OP", "solid"),
                (HOSTILE, "Red positions", "solid")])
    with right:
        st.subheader(t.name or t.source)
        st.caption(t.source)
        for note in t.notes:
            st.markdown(f'<p class="fine">{note}</p>', unsafe_allow_html=True)
        comp = pd.DataFrame({"Ground": CLASS_NAMES,
                             "Share": np.bincount(t.cls.ravel(), minlength=5) / t.cls.size})
        st.dataframe(comp, hide_index=True, column_config={
            "Share": st.column_config.ProgressColumn("Share", min_value=0.0, max_value=1.0, format="percent")})
        st.metric("Best support-by-fire view of the defense", f"{scen.sbf_best_vis:.0%}",
                  help="Mean visible fraction of the ten Red positions from the best SBF cell.")
        st.metric("Red positions in buildings", f"{int(scen.r_building.sum())} of {NR}")
        st.caption(f"Relief {t.ground.min():.0f} to {t.ground.max():.0f} m. Grid {t.n} x {t.n} at {t.cell:.0f} m.")

if runs is None:
    for tab in (tab_results, tab_losses, tab_play):
        with tab:
            st.info("Set the two courses of action in the sidebar, then select **Run both COAs**.")
    st.stop()

res, coas, n = runs["res"], runs["coas"], runs["n"]
if stale:
    for tab in (tab_results, tab_losses, tab_play):
        with tab:
            st.warning("These results come from an earlier setup. Select **Run both COAs** to update them.")

# ---------------------------------------------------------------- Results
with tab_results:
    A, B = res["A"], res["B"]
    st.markdown(f'<p class="coa"><b>A:</b> {coa_label(coas["A"])}<br><b>B:</b> {coa_label(coas["B"])}<br>'
                f'<b>Red:</b> {RED_COA[runs["red"]].lower()}. {n:,} replications each, shared random numbers.</p>',
                unsafe_allow_html=True)

    def fmt(m, lo, hi, d=2):
        return f"{m:.{d}f}  [{lo:.{d}f}, {hi:.{d}f}]"

    rows = []
    wa, wb = stats.wilson(A.win.sum(), n), stats.wilson(B.win.sum(), n)
    rows.append(("P(Blue win)", fmt(A.win.mean(), *wa, 3), fmt(B.win.mean(), *wb, 3),
                 fmt(*stats.paired_ci(A.win, B.win), 3)))
    for name, key in (("Blue losses", "blue_losses"), ("Red losses", "red_losses"), ("Minutes to decision", "minutes")):
        a, b = getattr(A, key), getattr(B, key)
        rows.append((name, fmt(*stats.mean_ci(a)), fmt(*stats.mean_ci(b)), fmt(*stats.paired_ci(a, b))))
    ea, eb = stats.ratio_ci(A.red_losses, A.blue_losses), stats.ratio_ci(B.red_losses, B.blue_losses)
    rows.append(("Exchange ratio (Red lost per Blue lost)", fmt(*ea), fmt(*eb), ""))
    rows.append(("Time limit reached", f"{A.timeout.mean():.1%}", f"{B.timeout.mean():.1%}", ""))
    st.dataframe(pd.DataFrame(rows, columns=["Measure", "COA A", "COA B", "B minus A (paired)"]),
                 hide_index=True, width="stretch")
    rho, ratio = stats.crn_gain(A.win, B.win)
    st.caption("95% intervals. Wilson for P(win), normal for means and paired differences, delta method for "
               "the exchange ratio. These cover simulation error only; the parameters are notional and uncalibrated. "
               f"Shared random numbers gave a correlation of {rho:.2f} between the two COAs' outcomes, so the paired "
               f"difference has {ratio:.0%} of the variance an independent comparison would have.")

    st.subheader("How the fights end")
    st.caption("Cumulative incidence of each ending over time (Aalen-Johansen). Fights still running at the time "
               "limit are censored, not counted as either ending.")
    inc = []
    for tag, r in (("A", A), ("B", B)):
        ts, cif = stats.aalen_johansen(r.minutes, r.cause, 2)
        for k, name in ((0, "Blue broke"), (1, "Red broke")):
            inc.append(pd.DataFrame({"minute": np.r_[0.0, ts], "p": np.r_[0.0, cif[:, k]],
                                     "series": f"COA {tag}: {name}"}))
    idf = pd.concat(inc)
    inc_chart = alt.Chart(idf).mark_line(interpolate="step-after").encode(
        x=alt.X("minute:Q", title="Minutes"), y=alt.Y("p:Q", title="Share of fights ended this way"),
        color=alt.Color("series:N", scale=alt.Scale(
            domain=["COA A: Red broke", "COA B: Red broke", "COA A: Blue broke", "COA B: Blue broke"],
            range=["#1F5FA8", "#7FA7D6", "#C23B22", "#E39A8A"]), title=None)).properties(height=260)
    st.altair_chart(inc_chart, width="stretch")

    st.subheader("Strength over time")
    simultaneous = st.toggle("Simultaneous band (covers the whole curve at once)", False,
                             help="Pointwise bands cover each time point separately; a simultaneous sup-t "
                                  "bootstrap band covers the entire mean curve with 95% confidence and is wider.")
    ticks = A.blue_strength.shape[1]
    recs = []
    for tag, r in (("A", A), ("B", B)):
        for side, Y in (("Blue", r.blue_strength), ("Red", r.red_strength)):
            if simultaneous:
                m, lo, hi, _ = stats.sup_t_band(Y, B=400)
            else:
                m, lo, hi = stats.band(Y)
            recs.append(pd.DataFrame({"minute": np.arange(ticks) * r.tick_s / 60, "mean": m, "lo": lo, "hi": hi,
                                      "series": f"COA {tag}: {side}"}))
    sdf = pd.concat(recs)
    sdf = sdf[sdf.minute <= max(A.minutes.max(), B.minutes.max()) + 1]
    color = alt.Color("series:N", scale=alt.Scale(
        domain=["COA A: Blue", "COA B: Blue", "COA A: Red", "COA B: Red"],
        range=["#1F5FA8", "#7FA7D6", "#C23B22", "#E39A8A"]), title=None)
    base = alt.Chart(sdf).encode(x=alt.X("minute:Q", title="Minutes"))
    chart = (base.mark_area(opacity=0.25).encode(y=alt.Y("lo:Q", title="Units still fighting"), y2="hi:Q", color=color)
             + base.mark_line().encode(y="mean:Q", color=color)).properties(height=320)
    st.altair_chart(chart, width="stretch")
    st.caption(("Shaded bands are simultaneous 95% sup-t bands on the mean curve." if simultaneous else
                "Shaded bands are pointwise 95% intervals on the mean.")
               + " Decided replications hold their final strength.")
    with st.expander("Run manifest"):
        st.json(runs.get("manifest", {}))

# ---------------------------------------------------------------- Losses
with tab_losses:
    which = st.segmented_control("Course of action", ["A", "B"], default="A", key="loss_coa",
                                 format_func=lambda k: f"COA {k}") or "A"
    r = res[which]
    c1, c2 = st.columns([3, 1.3])
    with c1:
        heat = []
        if len(r.death_xy):
            ii, jj = t.xy_to_ij(r.death_xy[:, 0], r.death_xy[:, 1])
            cells, counts = np.unique(ii * t.n + jj, return_counts=True)
            cx, cy = t.ij_to_xy(cells // t.n, cells % t.n)
            # empirical-Bayes shrinkage: cells with a handful of deaths are pulled toward the mean
            # rate, so the map shows where losses concentrate rather than where noise landed
            post, _, _, strength = stats.shrink_rates(counts, np.full(len(counts), float(n)))
            heat = [{"pos": p, "w": float(w)} for p, w in zip(to_lonlat(np.c_[cx, cy]).tolist(), post)]
        rk = r.red_killed.mean(0)
        layers = [
            pdk.Layer("HeatmapLayer", data=heat, id="heat", get_position="pos", get_weight="w",
                      radius_pixels=45, opacity=0.75,
                      color_range=[[255, 237, 160], [254, 178, 76], [240, 59, 32], [140, 20, 20]]),
            pdk.Layer("ScatterplotLayer", id="rk",
                      data=[{"pos": p, "rad": 10 + 60 * float(rk[j]),
                             "tip": f"{R_LABELS[j]} destroyed in {rk[j]:.0%} of runs"}
                            for j, p in enumerate(to_lonlat(scen.rpos).tolist())],
                      get_position="pos", get_radius="rad", get_fill_color=HOSTILE + [200],
                      stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1,
                      pickable=True),
        ]
        st.pydeck_chart(make_deck(layers), height=600)
        legend([((240, 59, 32), "Where Blue units were destroyed (shrunk per-cell rate)", "solid"),
                (HOSTILE, "Red position, sized by how often it was destroyed", "solid")])
        if r.n_plans > 1:
            st.caption(f"Deaths pooled over {r.n_plans} sampled Red plans; Red markers show the first plan's positions.")
    with c2:
        present = r.frames["alive_b"][0, 0] if r.n_log else np.ones(NB, bool)
        bk = []
        for i in range(NB):
            if not present[i]:
                continue
            x = r.blue_killed[:, i].sum()
            lo, hi = stats.wilson(x, n)
            bk.append((B_LABELS[i], TYPE_NAMES[BTYPE[i]], x / n, f"[{lo:.2f}, {hi:.2f}]"))
        st.markdown("**Blue units destroyed**")
        st.dataframe(pd.DataFrame(bk, columns=["Unit", "Type", "Rate", "95% CI"]), hide_index=True,
                     column_config={"Rate": st.column_config.NumberColumn(format="%.2f")})
        rows = []
        for j in range(NR):
            x = r.red_killed[:, j].sum()
            lo, hi = stats.wilson(x, n)
            rows.append((R_LABELS[j], TYPE_NAMES[RTYPE[j]], x / n, f"[{lo:.2f}, {hi:.2f}]"))
        st.markdown("**Red positions destroyed**")
        st.dataframe(pd.DataFrame(rows, columns=["Unit", "Type", "Rate", "95% CI"]), hide_index=True,
                     column_config={"Rate": st.column_config.NumberColumn(format="%.2f")})

# ---------------------------------------------------------------- Playback
with tab_play:
    A, B = res["A"], res["B"]
    nl = min(A.n_log, B.n_log)
    if nl == 0:
        st.info("No replays were kept. Raise **Replays kept per COA** and run again.")
        st.stop()
    outcome = lambda r, i: "win" if r.win[i] else "loss"
    summ = pd.DataFrame({"rep": np.arange(nl), "a_win": A.win[:nl], "b_win": B.win[:nl],
                         "a_loss": A.blue_losses[:nl], "b_loss": B.blue_losses[:nl]})

    def typical(mask, col):
        sub = summ[mask]
        if sub.empty:
            return None
        med = sub[col].median()
        return int(sub.iloc[(sub[col] - med).abs().argsort().iloc[0]].rep)

    picks = {
        "A and B split": next((int(i) for i in summ.rep if A.win[i] != B.win[i]), None),
        "Typical A win": typical(summ.a_win == 1, "a_loss"),
        "Typical A loss": typical(summ.a_win == 0, "a_loss"),
        "Costliest A run": int(summ.sort_values("a_loss").iloc[-1].rep),
    }
    avail = [k for k, v in picks.items() if v is not None]
    c1, c2 = st.columns([2, 3])
    with c1:
        quick = st.segmented_control("Jump to", avail, key="quick_pick")
        if quick and st.session_state.get("last_quick") != quick:
            st.session_state.rep_pick = picks[quick]
            st.session_state.last_quick = quick
            st.session_state.pending_tick = 0
    with c2:
        def rep_label(i):
            pa = (A.blue_losses <= A.blue_losses[i]).mean()
            return (f"Run {i + 1}: A {outcome(A, i)} with {A.blue_losses[i]:.0f} lost "
                    f"({pa:.0%} of A runs lost as few or fewer); B {outcome(B, i)} with {B.blue_losses[i]:.0f} lost")
        rep = st.selectbox("Replay", list(range(nl)), format_func=rep_label, key="rep_pick")

    max_t = max(len(A.frames["xy"]), len(B.frames["xy"])) - 1
    if "pending_tick" in st.session_state:
        st.session_state.tick = min(st.session_state.pop("pending_tick"), max_t)
    if st.session_state.get("tick", 0) > max_t:
        st.session_state.tick = max_t
    c1, c2, c3 = st.columns([5, 1.2, 1])
    tick = c1.slider("Time", 0, max_t, key="tick", format="%d", help=f"Each step is {A.tick_s:g} seconds.")
    speed = c2.select_slider("Speed", ["Slow", "Normal", "Fast"], "Normal")
    play = c3.button("Play", icon=":material/play_arrow:", width="stretch")
    st.caption("Both panels replay the same replication number. With shared random numbers, the runs "
               "match until the plans pull them apart, so the split shows where the plan changed the outcome.")
    legend([(FRIEND, "Blue", "solid"), (HOSTILE, "Red, detected by Blue", "solid"),
            (HOSTILE, "Red, not yet detected", "faint"), (DEAD, "Destroyed", "solid"),
            (FIRES, "Suppressed", "ring"), (FIRES, "Fire mission", "solid")])

    def events_by_rep(r):
        out = {}
        for ev in r.events:
            out.setdefault(ev[1], []).append(ev)
        return out

    ev_cache = st.session_state.setdefault("ev_cache", {})
    if ev_cache.get("key") != runs["key"]:
        ev_cache.clear()
        ev_cache.update(key=runs["key"], A=events_by_rep(A), B=events_by_rep(B))

    def frame_layers(r, tag, i, tt):
        f = r.frames
        tt = min(tt, len(f["xy"]) - 1)
        xy = f["xy"][tt, i]
        present = f["alive_b"][0, i]
        alive_b, supp_b, mounted = f["alive_b"][tt, i], f["supp_b"][tt, i], f["mounted"][tt, i]
        alive_r, supp_r, seen = f["alive_r"][tt, i], f["supp_r"][tt, i], f["red_seen"][tt, i]
        blue = []
        for u in np.flatnonzero(present):
            kind = "IFV with infantry" if mounted[u] else TYPE_NAMES[BTYPE[u]]
            status = "destroyed" if not alive_b[u] else ("suppressed" if supp_b[u] else "fighting")
            blue.append({"pos": to_lonlat(xy[u]).tolist(),
                         "color": (FRIEND if alive_b[u] else DEAD) + [240],
                         "rad": 30 if (BTYPE[u] == TANK or mounted[u]) else 18,
                         "tip": f"{B_LABELS[u]}: {kind}, {status}"})
        red = []
        for j in range(NR):
            status = "destroyed" if not alive_r[j] else ("suppressed" if supp_r[j] else "fighting")
            col = DEAD + [230] if not alive_r[j] else HOSTILE + ([240] if seen[j] else [90])
            red.append({"pos": to_lonlat(scen.rpos[j]).tolist(), "color": col, "rad": 20,
                        "tip": f"{R_LABELS[j]}: {TYPE_NAMES[RTYPE[j]]}, {status}"
                               f"{'' if seen[j] or not alive_r[j] else ', not yet detected'}"})
        rings = [{"pos": to_lonlat(xy[u]).tolist(), "tip": f"{B_LABELS[u]} suppressed"}
                 for u in np.flatnonzero(present & alive_b & supp_b)]
        rings += [{"pos": to_lonlat(scen.rpos[j]).tolist(), "tip": f"{R_LABELS[j]} suppressed"}
                  for j in np.flatnonzero(alive_r & supp_r)]
        shots, impacts = [], []
        for ev in ev_cache[tag].get(i, []):
            if ev[0] != tt:
                continue
            _, _, side, sh, tg, hit, kill, p, ph = ev
            if side == "F":
                impacts.append({"pos": to_lonlat(scen.rpos[tg]).tolist(),
                                "tip": f"Fire mission on {R_LABELS[tg]}{', destroyed' if kill else ''}"})
                continue
            a_xy, b_xy = (xy[sh], scen.rpos[tg]) if side == "B" else (scen.rpos[sh], xy[tg])
            col = (FRIEND if side == "B" else HOSTILE) + ([255] if kill else [150])
            shots.append({"from": to_lonlat(a_xy).tolist(), "to": to_lonlat(b_xy).tolist(),
                          "color": col, "w": 4 if kill else (2 if hit else 1),
                          "tip": f"P(chosen) {p:.2f}, P(hit) {ph:.2f}"})
        text = [{"pos": to_lonlat(xy[u]).tolist(), "text": B_LABELS[u], "tip": B_LABELS[u]}
                for u in np.flatnonzero(present & alive_b)]
        text += [{"pos": to_lonlat(scen.rpos[j]).tolist(), "text": R_LABELS[j], "tip": R_LABELS[j]}
                 for j in np.flatnonzero(alive_r)]
        return [
            pdk.Layer("ScatterplotLayer", data=impacts, id=f"imp{tag}", get_position="pos", get_radius=55,
                      get_fill_color=FIRES + [120], stroked=True, get_line_color=FIRES,
                      line_width_min_pixels=2, pickable=True),
            pdk.Layer("LineLayer", data=shots, id=f"shot{tag}", get_source_position="from",
                      get_target_position="to", get_color="color", get_width="w", width_units="pixels",
                      pickable=True),
            pdk.Layer("ScatterplotLayer", data=red, id=f"red{tag}", get_position="pos", get_radius="rad",
                      get_fill_color="color", radius_min_pixels=4, pickable=True,
                      stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1),
            pdk.Layer("ScatterplotLayer", data=blue, id=f"blue{tag}", get_position="pos", get_radius="rad",
                      get_fill_color="color", radius_min_pixels=4, pickable=True,
                      stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1),
            pdk.Layer("ScatterplotLayer", data=rings, id=f"ring{tag}", get_position="pos", get_radius=42,
                      filled=False, stroked=True, get_line_color=FIRES, line_width_min_pixels=2),
            pdk.Layer("TextLayer", data=text, id=f"txt{tag}", get_position="pos", get_text="text",
                      get_size=12, get_color=[28, 35, 33], get_pixel_offset=[0, -14],
                      font_family="Barlow Condensed, Barlow, sans-serif", font_weight=600),
        ]

    def feed_html(r, tag, i, tt):
        name = lambda side, k: B_LABELS[k] if side == "B" else R_LABELS[k]
        lines = []
        for ev in reversed([e for e in ev_cache[tag].get(i, []) if e[0] <= tt][-9:]):
            et, _, side, sh, tg, hit, kill, p, ph = ev
            if side == "F":
                what = f"Fire mission on {R_LABELS[tg]}" + (', <span class="k">destroyed</span>' if kill else ", suppressing")
            else:
                other = "R" if side == "B" else "B"
                res_txt = '<span class="k">destroyed</span>' if kill else ("hit" if hit else "miss")
                what = (f"{name(side, sh)} fires on {name(other, tg)}: {res_txt} "
                        f"(chose it at p={p:.2f}, P(hit) {ph:.2f})")
            lines.append(f"{mmss(et, r.tick_s)}&nbsp;&nbsp;{what}")
        body = "<br>".join(lines) if lines else "No shots yet."
        return f'<div class="feed">{body}</div>'

    def status_line(r, i, tt):
        f = r.frames
        last = len(f["xy"]) - 1
        tt2 = min(tt, last)
        bl = int((f["alive_b"][0, i] & ~f["alive_b"][tt2, i] & ~IS_SCOUT).sum())
        rl = int((~f["alive_r"][tt2, i]).sum())
        done = tt >= r.minutes[i] * 60 / r.tick_s
        verdict = (" Blue took the objective." if r.win[i] else " Blue's attack failed.") if done else ""
        return f"Blue lost {bl}, Red lost {rl}.{verdict}"

    cols = st.columns(2)
    slots = {}
    for col, tag in zip(cols, ("A", "B")):
        with col:
            st.markdown(f'<p class="coa"><b>COA {tag}:</b> {coa_label(coas[tag])}</p>', unsafe_allow_html=True)
            slots[tag] = dict(clock=st.empty(), map=st.empty(), feed=st.empty())

    def render(tt):
        for tag in ("A", "B"):
            r = res[tag]
            slots[tag]["clock"].markdown(
                f'<div class="clock">{mmss(tt, r.tick_s)}</div><div class="coa">{status_line(r, rep, tt)}</div>',
                unsafe_allow_html=True)
            slots[tag]["map"].pydeck_chart(make_deck(frame_layers(r, tag, rep, tt)), height=470)
            slots[tag]["feed"].markdown(feed_html(r, tag, rep, tt), unsafe_allow_html=True)

    if play:
        delay = {"Slow": 0.6, "Normal": 0.3, "Fast": 0.08}[speed]
        start = 0 if tick >= max_t else tick
        for tt in range(start, max_t + 1):
            render(tt)
            time.sleep(delay)
        st.session_state.pending_tick = max_t
        st.rerun()
    else:
        render(tick)

st.markdown(
    '<p class="fine">Map data © OpenStreetMap contributors (ODbL). Basemap © CARTO. Elevation from AWS Terrain '
    'Tiles (Mapzen). Model parameters are notional and uncalibrated; use the outputs to compare plans and '
    'explore sensitivity, not to predict outcomes.</p>', unsafe_allow_html=True)
