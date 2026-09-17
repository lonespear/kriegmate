"""Brigade-on-brigade page: pick two nations, schemes of maneuver, force ratio and ground; run
paired replications; solve the 3x3 COA game. Run with `streamlit run app_brigade.py`.
Every number behind this page is synthetic (see sim/nations.py)."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sim import stats
from sim.brigade import ATTACK_COAS, DEFEND_COAS, MANEUVER, BrigadeSim, synthetic_brigade_terrain
from sim.nations import CLASSES, NATIONS

st.set_page_config(page_title="Brigade COA game", layout="wide")
st.title("Brigade against brigade")
st.caption("Company-level aggregate engine driven by the same kernels as the entity model. All equipment "
           "and force-structure numbers are notional placeholders; see README section 19 and sim/nations.py.")

with st.sidebar:
    st.header("Forces")
    att = st.selectbox("Attacker", list(NATIONS), format_func=lambda c: NATIONS[c].name, index=0)
    dfn = st.selectbox("Defender", list(NATIONS), format_func=lambda c: NATIONS[c].name, index=1)
    att_bdes = st.slider("Attacking brigades", 1, 3, 2)
    def_bdes = st.slider("Defending brigades", 1, 3, 1)
    att_str = st.slider("Attacker strength", 0.5, 1.0, 1.0, 0.05)
    def_str = st.slider("Defender strength", 0.5, 1.0, 1.0, 0.05)
    st.header("Schemes")
    att_coa = st.selectbox("Attacker scheme", ATTACK_COAS)
    def_coa = st.selectbox("Defender scheme", DEFEND_COAS)
    st.header("Ground (synthetic)")
    relief = st.slider("Relief (m)", 20, 300, 120, 10)
    cover = st.slider("Cover share", 0.0, 0.5, 0.2, 0.05)
    tseed = st.number_input("Terrain seed", 0, 999, 1)
    st.header("Run")
    reps = st.select_slider("Replications", [10, 20, 50, 100, 200, 400], 100)
    seed = int(st.number_input("Seed", 0, 1_000_000, 7))
    run = st.button("Run this pairing", type="primary", width="stretch")
    run_game = st.button("Solve the 3×3 COA game", width="stretch")

with st.expander("Notional catalogs in play"):
    for code in (att, dfn):
        n = NATIONS[code]
        st.markdown(f"**{n.name}** — {n.notes}")
        st.dataframe(pd.DataFrame([dict(cls=c, platform=e.name, theta0_mrad=e.theta0 * 1e3, sensor=e.sensor,
                                        fire_rate=e.fire_rate, protection=e.protection, warhead=e.warhead,
                                        rmax_m=e.rmax, per_company=e.per_company)
                                   for c, e in n.equipment.items()]), hide_index=True, width="stretch")
        b = n.brigade
        bns = ", ".join(f"{k['tank']}T+{k['ifv']}M" for k in b["battalions"])
        st.caption(f"Structure: {len(b['battalions'])} maneuver bns [{bns} companies], {b.get('at_cos', 0)} AT cos, "
                   f"{b['recon_cos']} cav/recon troops, {b['howitzer_batteries']} tube + {b['rocket_batteries']} rocket batteries, "
                   f"{b['engineer_cos']} engineer cos, {b['support_cos']} support cos, {b['aviation_teams']} attack weapons team; "
                   f"SHORAD factor {n.shorad}, C2 delay {n.c2_delay_min} min, quality {n.quality}, readiness {b['readiness']}.")


@st.cache_resource(show_spinner=False)
def ground(seed, relief, cover):
    return synthetic_brigade_terrain(seed=int(seed), relief=float(relief), cover=float(cover))


t = ground(tseed, relief, cover)

if run:
    with st.spinner("Running…"):
        sim = BrigadeSim(t, NATIONS[att], NATIONS[dfn], att_coa, def_coa, att_bdes=att_bdes, def_bdes=def_bdes,
                         att_strength=att_str, def_strength=def_str)
        st.session_state.bres = sim.run(int(reps), seed=seed, n_log=1, log_every=5)
        st.session_state.bmeta = dict(att=att, dfn=dfn, att_coa=att_coa, def_coa=def_coa)

if run_game:
    cells, prog = {}, st.progress(0.0, "COA matrix…")
    for i, ac in enumerate(ATTACK_COAS):
        for j, dc in enumerate(DEFEND_COAS):
            sim = BrigadeSim(t, NATIONS[att], NATIONS[dfn], ac, dc, att_bdes=att_bdes, def_bdes=def_bdes,
                             att_strength=att_str, def_strength=def_str)
            cells[i, j] = sim.run(int(reps), seed=seed, n_log=0).win
            prog.progress((i * 3 + j + 1) / 9, f"{ac} vs {dc}")
    prog.empty()
    st.session_state.bgame = stats.bootstrap_game_value(cells, B=300, seed=seed)

res = st.session_state.get("bres")
if res is not None:
    meta = st.session_state.bmeta
    st.subheader(f"{NATIONS[meta['att']].name} ({meta['att_coa']}) attacks {NATIONS[meta['dfn']].name} ({meta['def_coa']})")
    lo, hi = stats.wilson(res.win.sum(), res.n)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(objective seized)", f"{res.win.mean():.2f}", f"[{lo:.2f}, {hi:.2f}]", delta_color="off")
    m, l, h = stats.mean_ci(res.att_loss_frac()); c2.metric("Attacker maneuver losses", f"{m:.2f}", f"[{l:.2f}, {h:.2f}]", delta_color="off")
    m, l, h = stats.mean_ci(res.def_loss_frac()); c3.metric("Defender maneuver losses", f"{m:.2f}", f"[{l:.2f}, {h:.2f}]", delta_color="off")
    m, l, h = stats.mean_ci(res.minutes); c4.metric("Minutes to decision", f"{m:.0f}", f"[{l:.0f}, {h:.0f}]", delta_color="off")
    st.caption(f"{res.n} replications; {(res.cause == 0).mean():.0%} censored at the time limit. Wilson interval on the "
               "proportion, normal intervals on means. Simulation error only. "
               f"Attack-aviation sorties per fight: attacker {res.sorties[0].mean():.1f}, defender {res.sorties[1].mean():.1f}; "
               f"indirect-fire rounds left at the end: attacker {res.ammo_left[0].mean():.0f}, defender {res.ammo_left[1].mean():.0f}; "
               f"defender obstacle belt: {int(res.obstacles.sum())} cells.")

    st.subheader("Losses by class")
    df = pd.DataFrame({"class": CLASSES, "attacker start": res.att_n0.astype(int), "attacker lost": res.att_losses.mean(0).round(1),
                       "defender start": res.def_n0.astype(int), "defender lost": res.def_losses.mean(0).round(1)})
    st.dataframe(df, hide_index=True, width="stretch")

    st.subheader("How the fights end")
    ts, cif = stats.aalen_johansen(res.minutes, res.cause, 2)
    if len(ts):
        idf = pd.concat([pd.DataFrame({"minute": np.r_[0.0, ts], "p": np.r_[0.0, cif[:, k]], "ending": name})
                         for k, name in ((0, "attacker culminated"), (1, "objective seized"))])
        st.altair_chart(alt.Chart(idf).mark_line(interpolate="step-after").encode(
            x=alt.X("minute:Q", title="Minutes"), y=alt.Y("p:Q", title="Cumulative incidence"),
            color=alt.Color("ending:N", title=None)).properties(height=240), width="stretch")

    st.subheader("Maneuver strength over time")
    recs = []
    T = res.att_strength.shape[1]
    for side, Y in (("attacker", res.att_strength), ("defender", res.def_strength)):
        m_, lo_, hi_ = stats.band(Y)
        recs.append(pd.DataFrame({"minute": np.arange(T), "mean": m_, "lo": lo_, "hi": hi_, "side": side}))
    sdf = pd.concat(recs)
    sdf = sdf[sdf.minute <= res.minutes.max() + 5]
    base = alt.Chart(sdf).encode(x=alt.X("minute:Q", title="Minutes"), color=alt.Color("side:N", title=None))
    st.altair_chart((base.mark_area(opacity=0.25).encode(y=alt.Y("lo:Q", title="Maneuver platforms in the fight"), y2="hi:Q")
                     + base.mark_line().encode(y="mean:Q")).properties(height=260), width="stretch")

    if res.tracks:
        st.subheader("Positions (first logged replication)")
        mins, axy, dxy, aal, dal = res.tracks[0]
        k = st.slider("Minute", 0, int(mins[-1]), 0, 5)
        idx = int(np.searchsorted(mins, k, side="right") - 1)
        pts = pd.DataFrame({"x": np.r_[axy[idx, :, 0], dxy[idx, :, 0]] / 1000, "y": np.r_[axy[idx, :, 1], dxy[idx, :, 1]] / 1000,
                            "side": ["attacker"] * axy.shape[1] + ["defender"] * dxy.shape[1],
                            "name": res.att_names + res.def_names,
                            "alive": np.r_[aal[idx], dal[idx]]})
        x, y = t.cell_centres()
        step = max(1, t.n // 50)
        bg = pd.DataFrame({"x": x[::step, ::step].ravel() / 1000, "y": y[::step, ::step].ravel() / 1000,
                           "cover": t.cover[::step, ::step].ravel(), "water": (t.cls[::step, ::step] == 4).ravel()})
        oi, oj = np.nonzero(res.obstacles)
        ox, oy = t.ij_to_xy(oi, oj)
        obs_df = pd.DataFrame({"x": np.asarray(ox) / 1000, "y": np.asarray(oy) / 1000})
        ch = (alt.Chart(bg).mark_square(size=28).encode(x=alt.X("x:Q", title="km east"), y=alt.Y("y:Q", title="km north"),
                                                         color=alt.Color("cover:Q", scale=alt.Scale(scheme="greens"), legend=None),
                                                         opacity=alt.value(0.5))
              + alt.Chart(bg[bg.water]).mark_square(size=28, color="#4a90d9").encode(x="x:Q", y="y:Q")
              + alt.Chart(obs_df).mark_square(size=20, color="#222222").encode(x="x:Q", y="y:Q")
              + alt.Chart(pts).mark_point(size=90, filled=True).encode(
                  x="x:Q", y="y:Q", color=alt.Color("side:N", scale=alt.Scale(domain=["attacker", "defender"], range=["#1F5FA8", "#C23B22"])),
                  opacity=alt.condition(alt.datum.alive, alt.value(1.0), alt.value(0.2)),
                  shape=alt.Shape("side:N", legend=None), tooltip=["name", "side", "alive"])
              ).properties(height=520)
        st.altair_chart(ch, width="stretch")
        st.caption("Faded markers are broken or destroyed companies; black squares are the defender's obstacle belts. "
                   "Objective at (6.5, 0). Names: T tank co, M mech co, R cav troop, H/K tube/rocket battery, "
                   "E engineer co, S support co, A attack weapons team.")

game = st.session_state.get("bgame")
if game is not None:
    v, lo, hi, x, y, xb, yb, M = game
    st.subheader("COA game: P(objective seized)")
    mdf = pd.DataFrame([dict(attacker=ac, defender=dc, p=M[i, j]) for i, ac in enumerate(ATTACK_COAS) for j, dc in enumerate(DEFEND_COAS)])
    st.altair_chart(alt.Chart(mdf).mark_rect().encode(x=alt.X("defender:N", title="Defender scheme"), y=alt.Y("attacker:N", title="Attacker scheme"),
                                                     color=alt.Color("p:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]), title="P(win)"),
                                                     tooltip=["attacker", "defender", alt.Tooltip("p:Q", format=".3f")])
                    .properties(height=220) + alt.Chart(mdf).mark_text().encode(x="defender:N", y="attacker:N", text=alt.Text("p:Q", format=".2f")),
                    width="stretch")
    st.markdown(f"**Game value {v:.3f}** (bootstrap 95% interval [{lo:.3f}, {hi:.3f}]). Attacker mix: "
                + ", ".join(f"{c} {p:.2f}" for c, p in zip(ATTACK_COAS, x)) + ". Defender mix: "
                + ", ".join(f"{c} {p:.2f}" for c, p in zip(DEFEND_COAS, y)) + ".")
    st.caption("Cells share random numbers (paired). The value is what the attacker can guarantee if the defender "
               "also plays its minimax mix; the bootstrap resamples replications and re-solves the game.")
