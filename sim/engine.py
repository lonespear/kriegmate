"""Stochastic engagement engine on precomputed route/LOS tables, with replay logging.

Time advances in ticks. Each tick every live unit may detect, choose a target (multinomial logit
with a hold-fire option), fire, and move; both sides resolve simultaneously. A replication ends
when either side reaches its stochastic breakpoint or the time limit runs out.

Random numbers come from named substreams (STREAMS) spawned from one SeedSequence, and every
draw has a fixed shape set by the roster. Two courses of action run with the same seed on the
same roster therefore share every random number: that is what makes paired (common random
number) comparisons valid, and it survives adding new mechanics as long as new draws get new
stream names appended at the end of STREAMS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .units import IFV, default_params

TICK_S = 15                      # the tick length the parameters are stated for
TICK_DEFAULT = 7.5               # the tick the engine runs at by default (see README, "Discretization")
RANGE_BANDS = np.array([500.0, 1000.0])   # edges of the shot/hit range bands
STREAMS = ["det_b", "det_r", "choice_b", "choice_r", "ready_b", "ready_r", "hit_b", "hit_r",
           "kill_b", "kill_r", "supp_b", "supp_r", "mission", "mission_tgt", "mission_kill",
           "mission_supp", "breakpoints"]          # never reorder; append new streams at the end


def gumbel_choice(V, V0, lam, u):
    """Multinomial-logit choice via Gumbel-max. Returns -1 for hold fire, else target index."""
    Vall = np.concatenate([V0[..., None], V], axis=-1)
    G = -np.log(-np.log(np.clip(u, 1e-12, 1 - 1e-12)))
    return np.argmax(lam[..., None] * Vall + G, axis=-1) - 1


def choice_prob(V, V0, lam, ch):
    """Model probability of the option actually chosen (for replay annotation)."""
    z = lam[..., None] * np.concatenate([V0[..., None], V], axis=-1)
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(-1, keepdims=True)
    return np.take_along_axis(p, (ch + 1)[..., None], -1)[..., 0]


def hit_prob(D, vis, area, theta0, theta_s, theta_t, m_shooter, m_target, supp, kappa):
    """Circular-normal aim error against the visible fraction of a circular-equivalent target."""
    sig2 = D ** 2 * (theta0 ** 2 + theta_s ** 2 * m_shooter + theta_t ** 2 * m_target) * (1 + kappa * supp)
    return 1 - np.exp(-vis * area / (2 * np.pi * np.maximum(sig2, 1e-9)))


@dataclass
class SimResult:
    win: np.ndarray               # 1 if Red broke first
    blue_losses: np.ndarray       # fighting units (not scouts) lost
    red_losses: np.ndarray
    minutes: np.ndarray           # time of decision (time limit if censored)
    timeout: np.ndarray           # 1 if the time limit was reached
    blue_strength: np.ndarray     # (n, T+1) fighting units alive per tick
    red_strength: np.ndarray
    blue_killed: np.ndarray       # (n, NB) bool
    red_killed: np.ndarray        # (n, NR) bool
    death_xy: np.ndarray          # (m, 2) local metres where Blue units died
    death_rep: np.ndarray         # (m,) replication of each death
    frames: dict = field(default_factory=dict)   # per-tick snapshots of the first n_log reps
    events: list = field(default_factory=list)   # (tick, rep, side, shooter, target, hit, kill, p_choice, p_hit)
    n_log: int = 0
    ticks: int = 0
    tick_s: float = TICK_S
    cause: np.ndarray = None      # 0 = time limit (censored), 1 = Blue broke, 2 = Red broke
    n_plans: int = 1
    blue_shots: np.ndarray = None # (n, 3) direct-fire shots by range band (<500, 500-1000, >1000 m)
    blue_hits: np.ndarray = None  # (n, 3) hits by range band
    red_shots: np.ndarray = None
    red_hits: np.ndarray = None

    @property
    def n(self):
        return len(self.win)


def _pad_cat(arrs):
    T = max(a.shape[1] for a in arrs)
    return np.concatenate([np.pad(a, ((0, 0), (0, T - a.shape[1])), mode="edge") for a in arrs])


def concat_results(results):
    """Stack replications from runs on different Red plans (outer loop). Replay frames and events
    are kept from the first run only, because unit positions differ between plans."""
    r0 = results[0]
    cat = lambda key: np.concatenate([getattr(r, key) for r in results])
    return SimResult(
        win=cat("win"), blue_losses=cat("blue_losses"), red_losses=cat("red_losses"),
        minutes=cat("minutes"), timeout=cat("timeout"),
        blue_strength=_pad_cat([r.blue_strength for r in results]),
        red_strength=_pad_cat([r.red_strength for r in results]),
        blue_killed=cat("blue_killed"), red_killed=cat("red_killed"),
        death_xy=cat("death_xy"), death_rep=cat("death_rep"),
        frames=r0.frames, events=r0.events, n_log=r0.n_log, ticks=max(r.ticks for r in results),
        tick_s=r0.tick_s, cause=cat("cause"), n_plans=sum(r.n_plans for r in results),
        blue_shots=cat("blue_shots"), blue_hits=cat("blue_hits"),
        red_shots=cat("red_shots"), red_hits=cat("red_hits"))


def make_streams(seed):
    kids = np.random.SeedSequence(seed).spawn(len(STREAMS))
    return {name: np.random.default_rng(k) for name, k in zip(STREAMS, kids)}


def simulate(tab, armor, recon, fires, red_coa, n, seed, P=None, red_trigger=600.0, n_log=20,
             tick_s=TICK_DEFAULT):
    """Run n replications of one course-of-action pair on precomputed route tables.

    tab: Scenario.tables(armor); armor in {'SBF','Maneuver','Unsupported'}; red_coa in
    {'Early','Late'}. seed: int or list, the run's SeedSequence entropy. tick_s: time step in
    seconds; per-tick parameters are stated for 15 s and rescaled (hazards and speeds linearly,
    per-tick probabilities as 1 - (1 - p)^(tick_s / 15)).
    """
    P = P or default_params()
    R = tab["roster"]
    NB, NR = R.nb, R.nr
    BTYPE, RTYPE = R.btype, R.rtype
    IS_INF, IS_SCOUT, IS_TANK = R.is_inf, R.is_scout, R.is_tank
    rs = make_streams(seed)
    PK, RMAX = P["PK"], P["RMAX"]
    cfg = tab["cfg"]
    f = tick_s / TICK_S
    per_tick = lambda p: 1 - (1 - np.asarray(p, float)) ** f
    T = int(round(cfg.max_minutes * 60 / tick_s))
    K = tab["cum"].shape[1]
    L, cum = tab["length"], tab["cum"]
    ar_b, ar_r = np.arange(NB), np.arange(NR)
    n_log = min(n_log, n)
    fire_rate = per_tick(P["fire_rate"])
    p_mission_recon, p_mission_norecon = per_tick(P["p_mission_recon"]), per_tick(P["p_mission_norecon"])
    supp_direct = max(1, int(round(P["supp_ticks_direct"] / f)))
    supp_fires = max(1, int(round(P["supp_ticks_fires"] / f)))

    b_alive = np.ones((n, NB), bool)
    if armor == "Unsupported":
        b_alive[:, IS_INF] = False
    if not recon:
        b_alive[:, IS_SCOUT] = False
    present = b_alive.copy()
    denom = (b_alive & ~IS_SCOUT).sum(1)
    r_alive = np.ones((n, NR), bool)
    s = np.zeros((n, NB))
    supp_b = np.zeros((n, NB), int)
    supp_r = np.zeros((n, NR), int)
    fired_b = np.zeros((n, NB), bool)
    fired_r = np.zeros((n, NR), bool)
    last_b = np.full((n, NB), -1)
    last_r = np.full((n, NR), -1)
    det_b = np.zeros((n, NB, NR), bool)
    det_r = np.zeros((n, NR, NB), bool)
    sprung = np.zeros(n, bool)
    ammo_r = np.tile(P["ammo"][RTYPE], (n, 1))
    bp_b = rs["breakpoints"].beta(*P["bp_blue"], n)
    bp_r = rs["breakpoints"].beta(*P["bp_red"], n)
    done = np.zeros(n, bool)
    win = np.zeros(n, bool)
    t_end = np.full(n, T)
    cause = np.zeros(n, int)
    shots_b, hits_b, shots_r, hits_r = (np.zeros((n, 3)) for _ in range(4))
    red_expo0 = P["red_posture"] * tab["r_conceal"]
    blue_str = np.zeros((n, T + 1))
    red_str = np.zeros((n, T + 1))
    blue_str[:, 0], red_str[:, 0] = denom, NR
    death_xy, death_rep = [], []
    frames = {k: [] for k in ("xy", "alive_b", "supp_b", "mounted", "moving",
                              "alive_r", "supp_r", "red_seen", "blue_seen")}
    events = []

    def locate(s):
        k = np.stack([np.searchsorted(cum[u], s[:, u], side="right") - 1 for u in range(NB)], 1)
        return np.clip(k, 0, K - 1)

    def snapshot(k, moving, union_b, union_r):
        g = slice(0, n_log)
        frames["xy"].append(tab["xy"][ar_b, k[g]])
        frames["alive_b"].append(b_alive[g] & present[g])
        frames["supp_b"].append(supp_b[g] > 0)
        frames["mounted"].append(tab["vehicle"][ar_b, k[g]] & IS_INF)
        frames["moving"].append(moving[g])
        frames["alive_r"].append(r_alive[g].copy())
        frames["supp_r"].append(supp_r[g] > 0)
        frames["red_seen"].append(union_b[g])
        frames["blue_seen"].append(union_r[g])

    t = -1
    for t in range(T):
        u = dict(db=rs["det_b"].random((n, NB, NR)), dr=rs["det_r"].random((n, NR, NB)),
                 gb=rs["choice_b"].random((n, NB, NR + 1)), gr=rs["choice_r"].random((n, NR, NB + 1)),
                 rb=rs["ready_b"].random((n, NB)), rr=rs["ready_r"].random((n, NR)),
                 hb=rs["hit_b"].random((n, NB)), hr=rs["hit_r"].random((n, NR)),
                 kb=rs["kill_b"].random((n, NB)), kr=rs["kill_r"].random((n, NR)),
                 sb=rs["supp_b"].random((n, NB)), sr=rs["supp_r"].random((n, NR)),
                 m=rs["mission"].random(n), ft=rs["mission_tgt"].random((n, NR)),
                 fk=rs["mission_kill"].random(n), fs=rs["mission_supp"].random(n))
        live = ~done
        k = locate(s)
        veh = tab["vehicle"][ar_b, k]
        mounted = veh & IS_INF
        etype = np.where(mounted, IFV, BTYPE)
        xy = tab["xy"][ar_b, k]
        D = tab["dist"][ar_b, k]
        Dt = D.transpose(0, 2, 1)

        # ---- movement intent
        rem_inf = np.where(b_alive & IS_INF, L - s, np.inf).min(1)
        stop = np.broadcast_to(L, (n, NB)).copy()
        if armor == "Maneuver" and IS_TANK.any():
            Lt = L[IS_TANK][None, :]
            stop[:, IS_TANK] = np.where(np.isfinite(rem_inf)[:, None],
                                        np.clip(Lt - rem_inf[:, None], 0, Lt), Lt)
        moving = b_alive & (s < stop - 1e-6) & (supp_b == 0)

        # ---- exposure (terrain LOS fraction x concealment x posture)
        b_expo = tab["conceal"][ar_b, k] * np.where(supp_b > 0, 0.5, 1.0)
        r_expo = red_expo0[None, :] * np.where(supp_r > 0, 0.5, 1.0)
        visR = tab["visR"][ar_b, k] * r_expo[:, None, :]                       # Red j as seen by Blue i
        visB = (tab["visB"][ar_b, k] * b_expo[:, :, None]).transpose(0, 2, 1)  # Blue i as seen by Red j
        rng_fac = (1000.0 / np.maximum(D, 50.0)) ** 2
        rng_fac_t = rng_fac.transpose(0, 2, 1)

        # ---- detection (per-tick hazard, persistent once made)
        rate_b = (P["sensor"][etype][:, :, None] * P["signature"][RTYPE] * visR
                  * (1 + P["phi_fire"] * fired_r)[:, None, :] * rng_fac * P["k_det"])
        det_b |= (u["db"] < 1 - np.exp(-f * rate_b)) & b_alive[:, :, None]
        det_b &= b_alive[:, :, None] & r_alive[:, None, :]
        rate_r = (P["sensor"][RTYPE][None, :, None] * P["signature"][etype][:, None, :] * visB
                  * (1 + P["mu_move"] * moving + P["phi_fire"] * fired_b)[:, None, :] * rng_fac_t * P["k_det"])
        det_r |= (u["dr"] < 1 - np.exp(-f * rate_r)) & r_alive[:, :, None]
        det_r &= r_alive[:, :, None] & b_alive[:, None, :]

        # ---- information sharing
        net_up = recon & (b_alive & IS_SCOUT).any(1)
        dbb = np.linalg.norm(xy[:, :, None] - xy[:, None, :], axis=-1)
        S = (dbb <= 150.0) & b_alive[:, :, None] & b_alive[:, None, :]
        local = (S.astype(np.float32) @ det_b.astype(np.float32)) > 0
        union_b = det_b.any(1)
        union_r = det_r.any(1)
        E_b = np.where(net_up[:, None, None], union_b[:, None, :] & b_alive[:, :, None], local)
        E_r = union_r[:, None, :] & r_alive[:, :, None]

        snapshot(k, moving, union_b, union_r)

        # ---- Blue target selection
        PH_b = hit_prob(D, visR, P["area"][RTYPE], P["theta0"][etype][:, :, None],
                        P["theta_s"][etype][:, :, None], 0.0, moving[:, :, None], 0.0,
                        (supp_b > 0)[:, :, None], P["kappa_supp"])
        PK_b = PK[etype[:, :, None], RTYPE[None, None, :]]
        ok_b = E_b & (D <= RMAX[etype[:, :, None], RTYPE[None, None, :]]) & (PK_b > 0) & (visR > 0)
        threat_b = last_r[:, None, :] == ar_b[None, :, None]
        V_b = np.log(PH_b * PK_b * P["value"][RTYPE] + 1e-12) + P["b_threat"] * threat_b
        V_b = np.where(ok_b, V_b, -np.inf)
        V0_b = P["v0"] + P["dv0_supp"] * (supp_b > 0) + np.where(IS_SCOUT, 50.0, 0.0)
        lam_b = P["lam_blue"] * np.where(supp_b > 0, P["lam_supp"], 1.0)
        ch_b = gumbel_choice(V_b, V0_b, lam_b, u["gb"])
        shoot_b = (ch_b >= 0) & b_alive & (u["rb"] < fire_rate[etype]) & live[:, None]

        # ---- Red target selection
        PH_r = hit_prob(Dt, visB, P["area"][etype][:, None, :], P["theta0"][RTYPE][None, :, None],
                        0.0, P["theta_t"][RTYPE][None, :, None], 0.0, moving[:, None, :],
                        (supp_r > 0)[:, :, None], P["kappa_supp"])
        PK_r = PK[RTYPE[None, :, None], etype[:, None, :]]
        ok_r = E_r & (Dt <= RMAX[RTYPE[None, :, None], etype[:, None, :]]) & (PK_r > 0) & (visB > 0)
        threat_r = last_b[:, None, :] == ar_r[None, :, None]
        V_r = np.log(PH_r * PK_r * P["value"][etype][:, None, :] + 1e-12) + P["b_threat"] * threat_r
        V_r = np.where(ok_r, V_r, -np.inf)
        V0_r = P["v0"] + P["dv0_supp"] * (supp_r > 0)
        if red_coa == "Late":
            dmin = np.where(b_alive[:, None, :], Dt, np.inf).min(2)
            V0_r = V0_r + np.where((dmin > red_trigger) & ~sprung[:, None], 50.0, 0.0)
        lam_r = P["lam_red"] * np.where(supp_r > 0, P["lam_supp"], 1.0)
        ch_r = gumbel_choice(V_r, V0_r, lam_r, u["gr"])
        shoot_r = (ch_r >= 0) & r_alive & (ammo_r > 0) & (u["rr"] < fire_rate[RTYPE]) & live[:, None]

        # ---- shot resolution (simultaneous)
        cb = np.clip(ch_b, 0, None)[:, :, None]
        ph_b = np.take_along_axis(PH_b, cb, 2)[..., 0]
        pk_b = np.take_along_axis(PK_b, cb, 2)[..., 0]
        d_b = np.take_along_axis(D, cb, 2)[..., 0]
        hit_b = shoot_b & (u["hb"] < ph_b)
        kill_b = hit_b & (u["kb"] < pk_b)
        sup_b = shoot_b & ~kill_b & (u["sb"] < P["p_supp"][etype] * np.exp(-d_b / 1000)
                                     * P["supp_vuln"][RTYPE][cb[..., 0]])
        oh_b = ch_b[:, :, None] == ar_r
        red_killed = (oh_b & kill_b[:, :, None]).any(1)
        red_supp = (oh_b & sup_b[:, :, None]).any(1)

        cr = np.clip(ch_r, 0, None)[:, :, None]
        ph_r = np.take_along_axis(PH_r, cr, 2)[..., 0]
        pk_r = np.take_along_axis(PK_r, cr, 2)[..., 0]
        d_r = np.take_along_axis(Dt, cr, 2)[..., 0]
        tgt_type = np.take_along_axis(etype, cr[..., 0], 1)
        hit_r = shoot_r & (u["hr"] < ph_r)
        kill_r = hit_r & (u["kr"] < pk_r)
        sup_r = shoot_r & ~kill_r & (u["sr"] < P["p_supp"][RTYPE] * np.exp(-d_r / 1000)
                                     * P["supp_vuln"][tgt_type])
        oh_r = ch_r[:, :, None] == ar_b
        blue_killed = (oh_r & kill_r[:, :, None]).any(1)
        blue_supp = (oh_r & sup_r[:, :, None]).any(1)

        # ---- indirect fires (a mission needs a shared detection outside danger close)
        fire_kill = np.zeros((n, NR), bool)
        fire_supp = np.zeros((n, NR), bool)
        mission = np.zeros(n, bool)
        tgt = np.zeros(n, int)
        if fires:
            dmin_all = np.where(b_alive[:, :, None], D, np.inf).min(1)
            cand = union_b & r_alive & (dmin_all > cfg.danger_close)
            p_m = np.where(net_up, p_mission_recon, p_mission_norecon)
            mission = (u["m"] < p_m) & cand.any(1) & live
            tgt = np.argmax(np.where(cand, u["ft"], -1.0), 1)
            oh = (ar_r == tgt[:, None]) & mission[:, None]
            fire_kill = oh & (u["fk"] < P["fires_kill"])[:, None]
            fire_supp = oh & (u["fs"] < P["fires_supp"])[:, None]

        # ---- replay log (first n_log replications)
        if n_log:
            g = slice(0, n_log)
            pb = choice_prob(V_b[g], V0_b[g], lam_b[g], ch_b[g])
            pr = choice_prob(V_r[g], V0_r[g], lam_r[g], ch_r[g])
            for rep, i in zip(*np.nonzero(shoot_b[g])):
                events.append((t, int(rep), "B", int(i), int(ch_b[rep, i]), bool(hit_b[rep, i]),
                               bool(kill_b[rep, i]), float(pb[rep, i]), float(ph_b[rep, i])))
            for rep, j in zip(*np.nonzero(shoot_r[g])):
                events.append((t, int(rep), "R", int(j), int(ch_r[rep, j]), bool(hit_r[rep, j]),
                               bool(kill_r[rep, j]), float(pr[rep, j]), float(ph_r[rep, j])))
            for rep in np.flatnonzero(mission[g]):
                events.append((t, int(rep), "F", -1, int(tgt[rep]), True,
                               bool(fire_kill[rep].any()), float("nan"), float("nan")))

        # ---- state update (frozen once a replication is decided)
        band_b, band_r = np.digitize(d_b, RANGE_BANDS), np.digitize(d_r, RANGE_BANDS)
        for q in range(3):
            shots_b[:, q] += (shoot_b & (band_b == q)).sum(1)
            hits_b[:, q] += (hit_b & (band_b == q)).sum(1)
            shots_r[:, q] += (shoot_r & (band_r == q)).sum(1)
            hits_r[:, q] += (hit_r & (band_r == q)).sum(1)
        red_killed = (red_killed | fire_kill) & live[:, None]
        blue_killed &= live[:, None]
        if blue_killed.any():
            rr, ii = np.nonzero(blue_killed)
            death_xy.append(xy[rr, ii])
            death_rep.append(rr)
        supp_b = np.maximum(supp_b - 1, 0)
        supp_r = np.maximum(supp_r - 1, 0)
        supp_b = np.where(blue_supp & live[:, None], np.maximum(supp_b, supp_direct), supp_b)
        supp_r = np.where(red_supp & live[:, None], np.maximum(supp_r, supp_direct), supp_r)
        supp_r = np.where(fire_supp, np.maximum(supp_r, supp_fires), supp_r)
        r_alive &= ~red_killed
        b_alive &= ~blue_killed
        fired_b, fired_r = shoot_b, shoot_r
        last_b = np.where(shoot_b, ch_b, -1)
        last_r = np.where(shoot_r, ch_r, -1)
        sprung |= shoot_r.any(1)
        ammo_r = ammo_r - shoot_r
        spd = f * P["speed"][etype] * tab["speed_fac"][ar_b, k]
        s = np.where(moving & live[:, None], np.minimum(s + spd, stop), s)

        # ---- breakpoints
        bl = (denom - (b_alive & ~IS_SCOUT).sum(1)) / np.maximum(denom, 1)
        rl = (~r_alive).sum(1) / NR
        b_brk, r_brk = bl >= bp_b, rl >= bp_r
        newly = live & (b_brk | r_brk)
        win |= live & r_brk & ~b_brk                 # simultaneous break scored as Blue failure
        cause = np.where(newly, np.where(r_brk & ~b_brk, 2, 1), cause)
        t_end = np.where(newly, t + 1, t_end)
        done |= newly
        blue_str[:, t + 1] = (b_alive & ~IS_SCOUT).sum(1)
        red_str[:, t + 1] = r_alive.sum(1)
        if done.all():
            break

    ticks = t + 1
    k = locate(s)
    snapshot(k, np.zeros((n, NB), bool), det_b.any(1), det_r.any(1))
    blue_str[:, ticks + 1:] = blue_str[:, ticks:ticks + 1]
    red_str[:, ticks + 1:] = red_str[:, ticks:ticks + 1]
    frames = {key: np.stack(v) for key, v in frames.items()}
    return SimResult(
        win=win.astype(float),
        blue_losses=(denom - (b_alive & ~IS_SCOUT).sum(1)).astype(float),
        red_losses=(~r_alive).sum(1).astype(float),
        minutes=t_end * tick_s / 60.0,
        timeout=(~done).astype(float),
        blue_strength=blue_str, red_strength=red_str,
        blue_killed=present & ~b_alive, red_killed=~r_alive,
        death_xy=np.concatenate(death_xy) if death_xy else np.zeros((0, 2)),
        death_rep=np.concatenate(death_rep) if death_rep else np.zeros(0, int),
        frames=frames, events=events, n_log=n_log, ticks=ticks, tick_s=tick_s, cause=cause,
        blue_shots=shots_b, blue_hits=hits_b, red_shots=shots_r, red_hits=hits_r,
    )
