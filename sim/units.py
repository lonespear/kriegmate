"""Unit types, rosters, and default parameters.

A `Roster` fixes the array shapes the engine works with. Every course of action run on the same
roster draws the same random numbers in the same order, which is what keeps common random
numbers (CRN) synchronized. Units a COA leaves out (no scouts, no infantry) start the run dead
rather than being removed from the arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TANK, IFV, INF, SCOUT, R_AT, R_INF = range(6)
TYPE_NAMES = ["Tank", "IFV (mounted inf)", "Inf team", "Scout", "Red AT team", "Red inf team"]

# Heights (m above ground) sampled for visible-fraction calculations
VEHICLE_HEIGHTS = np.array([0.5, 1.5, 2.4])
DISMOUNT_HEIGHTS = np.array([0.3, 1.0, 1.7])
RED_HEIGHTS = np.array([0.3, 0.6, 0.9])
EYE_VEHICLE, EYE_DISMOUNT, EYE_RED = 2.5, 1.7, 1.0
BUILDING_FLOOR = 3.0   # Red in a building fights from the second floor


@dataclass(frozen=True)
class Roster:
    name: str
    btype: np.ndarray
    rtype: np.ndarray

    @property
    def nb(self):
        return len(self.btype)

    @property
    def nr(self):
        return len(self.rtype)

    @property
    def is_tank(self):
        return self.btype == TANK

    @property
    def is_inf(self):
        return self.btype == INF

    @property
    def is_scout(self):
        return self.btype == SCOUT

    @property
    def b_labels(self):
        out, counts = [], {}
        for ty in self.btype:
            counts[ty] = counts.get(ty, 0) + 1
            out.append({TANK: "T", INF: "I", SCOUT: "S"}[ty] + str(counts[ty]))
        return out

    @property
    def r_labels(self):
        out, counts = [], {}
        for ty in self.rtype:
            counts[ty] = counts.get(ty, 0) + 1
            out.append({R_AT: "AT", R_INF: "R"}[ty] + str(counts[ty]))
        return out

    def start_offsets(self, spacing_vehicle=100.0, spacing_foot=100.0):
        """Lateral offsets (m) from the axis of advance at the start line, by group."""
        off = np.zeros(self.nb)
        for mask, sp in ((self.is_tank, spacing_vehicle), (self.is_inf, spacing_foot), (self.is_scout, 200.0)):
            k = mask.sum()
            if k:
                off[mask] = (np.arange(k) - (k - 1) / 2) * sp
        return off

    def __hash__(self):
        return hash((self.name, self.btype.tobytes(), self.rtype.tobytes()))

    def __eq__(self, other):
        return isinstance(other, Roster) and hash(self) == hash(other)


def make_roster(name, tanks, inf, scouts, at, rinf):
    return Roster(name, np.array([TANK] * tanks + [INF] * inf + [SCOUT] * scouts, int),
                  np.array([R_AT] * at + [R_INF] * rinf, int))


# The four rungs of the calibration ladder (see docs/technical-reference.md, "Calibration ladder")
DUEL = make_roster("duel", 1, 0, 0, 1, 0)                 # rung 1: one tank vs one AT team
SECTION = make_roster("section", 2, 2, 0, 1, 2)          # rung 2: tank section + 2 teams vs AT + 2 teams
PLATOON = make_roster("platoon", 2, 4, 1, 2, 4)          # rung 3: platoon(+) vs reinforced squad
COMPANY = make_roster("company", 4, 6, 2, 4, 6)          # rung 4: company team vs AT-heavy platoon
RUNG_ROSTERS = {1: DUEL, 2: SECTION, 3: PLATOON, 4: COMPANY}

# Module-level aliases for the company roster, used by the app
BTYPE, RTYPE = COMPANY.btype, COMPANY.rtype
B_LABELS, R_LABELS = COMPANY.b_labels, COMPANY.r_labels
NB, NR = COMPANY.nb, COMPANY.nr
IS_TANK, IS_INF, IS_SCOUT = COMPANY.is_tank, COMPANY.is_inf, COMPANY.is_scout


def default_params():
    """All model parameters. Per-tick quantities are stated for a 15 s tick; the engine rescales
    them when it is run with another tick length."""
    P = dict(
        area=np.array([6.0, 6.0, 1.0, 1.0, 1.0, 1.0]),            # m^2 presented at full exposure
        theta0=np.array([3e-4, 8e-4, 2e-3, 2e-3, 4e-4, 2e-3]),    # rad, stationary aim dispersion
        theta_s=np.array([5e-4, 1e-3, 4e-3, 4e-3, 4e-3, 4e-3]),   # rad, added when the shooter moves
        theta_t=np.array([1e-3, 1e-3, 1e-3, 1e-3, 3e-4, 1e-3]),   # rad, lead error vs a moving target
        sensor=np.array([0.5, 0.4, 0.8, 2.0, 0.8, 0.6]),          # detection hazard multiplier (shooter)
        signature=np.array([1.0, 1.0, 0.5, 0.2, 0.3, 0.3]),       # detection hazard multiplier (target)
        fire_rate=np.array([0.6, 0.6, 0.8, 0.5, 0.3, 0.8]),        # P(ready to fire) per 15 s tick
        p_supp=np.array([0.4, 0.4, 0.3, 0.1, 0.1, 0.3]),           # suppressiveness of a non-killing shot
        supp_vuln=np.array([0.1, 0.1, 1.0, 1.0, 1.0, 1.0]),        # susceptibility to suppression
        value=np.array([1.0, 0.9, 0.5, 0.4, 1.0, 0.6]),            # target value in the choice model
        speed=np.array([75.0, 75.0, 20.0, 15.0, 0.0, 0.0]),       # m per 15 s tick on open ground
        ammo=np.array([np.inf, np.inf, np.inf, np.inf, 3, np.inf]),
        lam_blue=4.0, lam_red=3.0, lam_supp=0.3,                   # choice rationality; multiplier when suppressed
        v0=np.log(0.01), dv0_supp=3.0, b_threat=1.0,               # hold-fire utility; its bump when suppressed; threat weight
        kappa_supp=3.0, mu_move=1.0, phi_fire=5.0, k_det=1.0,      # dispersion x(1+kappa s); detection bonuses; global detection scale
        red_posture=0.6,                                            # prepared-position exposure multiplier
        supp_ticks_direct=2, supp_ticks_fires=4,                   # suppression duration (15 s ticks)
        p_mission_recon=0.5, p_mission_norecon=0.2,                # P(fire mission) per 15 s tick
        fires_supp=0.85, fires_kill=0.08,
        bp_blue=(10, 10), bp_red=(11, 9),                          # breakpoint ~ Beta(a, b) on fraction lost
    )
    PK = np.zeros((6, 6))
    RMAX = np.zeros((6, 6))
    for s, pk_at, pk_inf, r in [(TANK, .7, .6, 2500), (IFV, .6, .6, 1500), (INF, .5, .5, 500), (SCOUT, .4, .4, 400)]:
        PK[s, R_AT], PK[s, R_INF] = pk_at, pk_inf
        RMAX[s, [R_AT, R_INF]] = r
    PK[R_AT, [TANK, IFV, INF, SCOUT]] = [.6, .7, .05, .05]
    RMAX[R_AT, :4] = 2500                                           # ATGM
    PK[R_INF, [TANK, IFV]] = [.3, .45]
    RMAX[R_INF, [TANK, IFV]] = 300                                  # RPG
    PK[R_INF, [INF, SCOUT]] = [.5, .5]
    RMAX[R_INF, [INF, SCOUT]] = 500                                 # small arms / MG
    P["PK"], P["RMAX"] = PK, RMAX
    return P


def copy_params(P):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in P.items()}
