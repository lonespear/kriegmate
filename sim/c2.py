"""Command and control: decision latency, subordinate initiative, latitude, net reliability,
synchronization, and nation-specific decision graphs (branches and sequels).

EVERYTHING HERE IS SYNTHETIC. The profiles encode widely reported but contested doctrinal
characterizations as ranges, not facts: mission command with commander's intent (US), directive
control with weak junior-leader initiative (Russia), phase-locked systems planning (China), and a
split Iranian force (Artesh, Soviet-style; IRGC, decentralized). scripts/c2_sensitivity.py exists
so that any claim built on these numbers is reported as a sensitivity, not a result.

Each battalion carries a C2Profile. Every decision in the brigade engine goes through the same
four steps: a trigger is OBSERVED by the brigade with probability net per minute (halved when the
HQ is destroyed); an order is ISSUED after a lognormal latency with mean tau (doubled without HQ);
meanwhile the subordinate may ACT ON INTENT with probability p_init as soon as it sees the trigger
itself; and the action is bounded by LATITUDE (bypass, exploit) and taxed by SYNC (fires clearance
across a boundary, fratricide risk out of sector).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class C2Profile:
    name: str
    tau: float          # mean decision latency, minutes (observe -> order at the companies)
    p_init: float       # probability a subordinate acts on intent when it sees a trigger and orders are stale
    latitude: float     # 0..1: permitted deviation from the task (bypass obstacles, exploit past the objective)
    net: float          # per-minute probability the brigade picture reaches / comes from a unit
    sync: float         # probability a cross-boundary coordination (fires clearance, boundary change) succeeds at once
    tau_sd: float = 0.5 # lognormal sigma of the latency

    def scaled(self, tau_mult=1.0, p_init=None, latitude=None, net=None, sync=None):
        return replace(self, tau=self.tau * tau_mult, p_init=self.p_init if p_init is None else p_init,
                       latitude=self.latitude if latitude is None else latitude, net=self.net if net is None else net,
                       sync=self.sync if sync is None else sync)


IDEAL = C2Profile("ideal", tau=0.0, p_init=1.0, latitude=1.0, net=1.0, sync=1.0, tau_sd=0.0)

# Notional profiles (see module docstring). Ranges for the sensitivity study are in C2_RANGES.
PROFILES = {
    "US": C2Profile("mission command", tau=5.0, p_init=0.80, latitude=0.80, net=0.90, sync=0.85),
    "RU": C2Profile("directive control", tau=18.0, p_init=0.25, latitude=0.25, net=0.70, sync=0.75),
    "CN": C2Profile("phase-locked systems C2", tau=9.0, p_init=0.40, latitude=0.40, net=0.85, sync=0.85),
    "IR": C2Profile("Artesh", tau=20.0, p_init=0.30, latitude=0.30, net=0.55, sync=0.55),
    "IRGC": C2Profile("IRGC", tau=5.0, p_init=0.85, latitude=0.90, net=0.50, sync=0.35),
}
C2_RANGES = dict(tau=(2.0, 30.0), p_init=(0.1, 0.95), latitude=(0.1, 1.0), net=(0.4, 1.0), sync=(0.3, 1.0))

# Cross-lineage synchronization multiplier (Artesh <-> IRGC coordination is the weak link)
CROSS_LINEAGE_SYNC = 0.5

# Decision graphs: what each nation's plan does at each trigger. Keys are triggers the engine
# evaluates every minute; values are the branch or sequel taken.
GRAPHS = {
    "US": dict(reserve="reinforce_success",   # commit the reserve where progress is best (exploit)
               objective="exploit",           # sequel: push to the depth objective if the defender is breaking
               obstacle="bypass",             # lead units look for a gap if latitude allows
               ammo_pause=0.15,               # tactical pause when the fires stock drops below this fraction
               fires_shift=True),             # shift priority of fires to the sector where the enemy masses
    "RU": dict(reserve="reinforce_failure",   # reinforce the axis that stalled, per the plan
               objective="consolidate", obstacle="breach", ammo_pause=0.10, fires_shift=True),
    "CN": dict(reserve="planned_axis",        # commit on the pre-planned axis regardless of progress
               objective="consolidate", obstacle="breach", ammo_pause=0.15, fires_shift=True),
    "IR": dict(reserve="reinforce_failure", objective="consolidate", obstacle="breach", ammo_pause=0.10,
               fires_shift=False),
    "IRGC": dict(reserve="reinforce_success", objective="exploit", obstacle="bypass", ammo_pause=0.05,
                 fires_shift=False),
}


def latency(profile: C2Profile, rng, hq_alive=True):
    """One decision latency draw in minutes (lognormal with mean tau; doubled without the HQ)."""
    if profile.tau <= 0:
        return 0.0
    mean = profile.tau * (1.0 if hq_alive else 2.0)
    mu = np.log(mean) - 0.5 * profile.tau_sd ** 2
    return float(rng.lognormal(mu, profile.tau_sd))


def lineages_for(nation_code, bn_count):
    """Which C2 profile each battalion runs under. Iran splits: the last maneuver battalion and
    its supporting elements are IRGC; everything else is Artesh."""
    if nation_code == "IR":
        return ["IR"] * (bn_count - 1) + ["IRGC"]
    return [nation_code] * bn_count
