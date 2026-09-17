"""Notional equipment catalogs and brigade templates for four nations.

EVERYTHING HERE IS SYNTHETIC. The numbers are proof-of-concept placeholders chosen to be
internally consistent and to reflect open-source qualitative characterizations (modern thermal
sights vs older optics, fire-and-forget vs wire-guided AT, artillery-heavy vs maneuver-heavy
force design). They are not measured performance data and must not be read as such. Their job
is to give the calibration ladder something to calibrate and the brigade model a documented
starting point. Every value is meant to be replaced through the ladder in sim/calibrate.py.

Each nation fills the same ten equipment classes:

  tank, ifv, inf (dismount platoon-equivalent), atgm (AT team), recon, howitzer (tube battery),
  rocket (MLRS battery), engineer (breach/obstacle vehicles), support (sustainment trucks),
  attack_helo (attack aviation)

and the same brigade functional structure: three maneuver battalions, a cavalry/reconnaissance
element, a fires battalion, an engineer battalion, a support battalion, an attack-aviation
enabler (a two-ship team), and an air-defence posture expressed as one `shorad` factor.

with the same attribute schema the entity engine uses (area, theta0, theta_s, theta_t, sensor,
signature, fire_rate per 15 s, p_supp, supp_vuln, value, speed m per 15 s, protection tier,
warhead tier, rmax_at / rmax_ap). Kill probabilities come from a warhead-vs-protection table
(CLASS_PK) times a nation-specific quality factor, so a nation's PK matrix is derived, not
typed in, and the derivation is documented in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CLASSES = ["tank", "ifv", "inf", "atgm", "recon", "howitzer", "rocket", "engineer", "support", "attack_helo"]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
DIRECT_FIRE = ["tank", "ifv", "inf", "atgm", "recon", "engineer", "support"]   # ground direct-fire shooters
INDIRECT = ["howitzer", "rocket"]
AVIATION = ["attack_helo"]                                     # handled by the aviation block, not direct fire

# Warhead tier (rows) vs protection tier (cols): notional P(kill | hit).
# Warhead tiers: 0 small arms / MG, 1 autocannon, 2 RPG-class, 3 ATGM, 4 tank gun.
# Protection tiers: 0 dismount, 1 light vehicle, 2 IFV, 3 MBT (older), 4 MBT (modern, APS/ERA).
CLASS_PK = np.array([
    [0.50, 0.15, 0.00, 0.00, 0.00],
    [0.55, 0.60, 0.40, 0.03, 0.02],
    [0.40, 0.55, 0.50, 0.30, 0.15],
    [0.08, 0.80, 0.80, 0.70, 0.50],
    [0.30, 0.80, 0.85, 0.75, 0.60],
])


@dataclass(frozen=True)
class Equipment:
    name: str
    cls: str
    area: float               # m^2 presented at full exposure
    theta0: float             # rad, stationary aim dispersion (lower = better sights/fire control)
    theta_s: float            # rad, added while moving
    theta_t: float            # rad, lead error vs a moving target
    sensor: float             # detection hazard multiplier as observer
    signature: float          # detection hazard multiplier as target
    fire_rate: float          # P(ready to fire) per 15 s
    p_supp: float             # suppressiveness of a non-killing shot
    supp_vuln: float
    value: float              # target value in the choice model
    speed: float              # m per 15 s on open ground
    protection: int           # protection tier 0..4
    warhead: int              # main weapon warhead tier 0..4
    rmax: float               # max direct-fire range (m); artillery: max range
    ammo: float = np.inf
    secondary_warhead: int = -1     # e.g. an IFV's ATGM; used vs tanks when better
    secondary_rmax: float = 0.0
    per_company: int = 0            # platforms in one company / battery of this class
    ind_lethality: float = 0.0      # artillery: notional kills per round vs dismounts in the open
    rounds_per_min: float = 0.0     # artillery: rounds per tube per minute in a mission


@dataclass(frozen=True)
class Nation:
    name: str
    code: str
    quality: float                  # crew training / maintenance multiplier on PK and fire rate
    c2_delay_min: float             # notional minutes from detection to a fire mission
    recon_uas: float                # detection hazard multiplier from organic UAS coverage
    shorad: float                   # 0..1 organic short-range air defence density; drives attack-helo losses
    equipment: dict                 # cls -> Equipment
    brigade: dict                   # template, see below
    notes: str = ""

    def eq(self, cls):
        return self.equipment[cls]


def _e(**kw):
    return Equipment(**kw)


def _enablers(eng_name, eng_n, sup_name, sup_n, helo_name, helo_theta, helo_rate, helo_rmax, helo_ammo):
    """Engineer, support and attack-aviation classes with the same shape for every nation."""
    return dict(
        engineer=_e(name=eng_name, cls="engineer", area=7.0, theta0=2e-3, theta_s=3e-3, theta_t=1e-3, sensor=0.5,
                    signature=1.1, fire_rate=0.3, p_supp=0.2, supp_vuln=0.3, value=0.7, speed=60, protection=2,
                    warhead=0, rmax=800, per_company=eng_n),
        support=_e(name=sup_name, cls="support", area=9.0, theta0=3e-3, theta_s=4e-3, theta_t=1e-3, sensor=0.3,
                   signature=1.3, fire_rate=0.2, p_supp=0.1, supp_vuln=0.8, value=0.6, speed=60, protection=1,
                   warhead=0, rmax=400, per_company=sup_n),
        attack_helo=_e(name=helo_name, cls="attack_helo", area=10.0, theta0=helo_theta, theta_s=helo_theta,
                       theta_t=2e-4, sensor=1.5, signature=1.0, fire_rate=helo_rate, p_supp=0.3, supp_vuln=0.0,
                       value=1.2, speed=800, protection=1, warhead=3, rmax=helo_rmax, ammo=helo_ammo, per_company=2),
    )


# ------------------------------------------------------------------ United States (ABCT-like)
US = Nation(
    "United States", "US", quality=1.00, c2_delay_min=4.0, recon_uas=1.6, shorad=0.25,
    equipment=dict(
        **_enablers("M1150 ABV / M1 breacher (notional)", 14, "BSB truck platoon (notional)", 20,
                    "AH-64E pair (notional)", 3e-4, 0.5, 8000, 16),
        tank=_e(name="M1A2 (notional)", cls="tank", area=6.5, theta0=2.2e-4, theta_s=3.5e-4, theta_t=8e-4,
                sensor=1.0, signature=1.0, fire_rate=0.65, p_supp=0.4, supp_vuln=0.1, value=1.0, speed=80,
                protection=4, warhead=4, rmax=3000, per_company=14),
        ifv=_e(name="M2 Bradley (notional)", cls="ifv", area=6.0, theta0=6e-4, theta_s=9e-4, theta_t=1e-3,
               sensor=0.9, signature=1.0, fire_rate=0.7, p_supp=0.45, supp_vuln=0.15, value=0.85, speed=80,
               protection=2, warhead=1, rmax=1800, secondary_warhead=3, secondary_rmax=3500, per_company=14),
        inf=_e(name="Rifle platoon (notional)", cls="inf", area=1.0, theta0=1.8e-3, theta_s=4e-3, theta_t=1e-3,
               sensor=0.8, signature=0.5, fire_rate=0.8, p_supp=0.3, supp_vuln=1.0, value=0.5, speed=20,
               protection=0, warhead=0, rmax=500, secondary_warhead=2, secondary_rmax=300, per_company=3),
        atgm=_e(name="Javelin team (notional)", cls="atgm", area=1.0, theta0=2.5e-4, theta_s=4e-3, theta_t=1.5e-4,
                sensor=0.9, signature=0.3, fire_rate=0.3, p_supp=0.1, supp_vuln=1.0, value=1.0, speed=15,
                protection=0, warhead=3, rmax=2500, ammo=3, per_company=6),
        recon=_e(name="Scout section (notional)", cls="recon", area=1.0, theta0=2e-3, theta_s=4e-3, theta_t=1e-3,
                 sensor=2.2, signature=0.2, fire_rate=0.5, p_supp=0.1, supp_vuln=1.0, value=0.4, speed=15,
                 protection=0, warhead=0, rmax=400, per_company=6),
        howitzer=_e(name="M109A7 battery (notional)", cls="howitzer", area=8.0, theta0=1e-3, theta_s=1e-3,
                    theta_t=1e-3, sensor=0.3, signature=1.2, fire_rate=0.0, p_supp=0.0, supp_vuln=0.5, value=0.9,
                    speed=60, protection=1, warhead=0, rmax=22000, per_company=6, ind_lethality=0.030,
                    rounds_per_min=3.0),
        rocket=_e(name="HIMARS battery (notional)", cls="rocket", area=8.0, theta0=1e-3, theta_s=1e-3,
                  theta_t=1e-3, sensor=0.3, signature=1.3, fire_rate=0.0, p_supp=0.0, supp_vuln=0.5, value=1.0,
                  speed=70, protection=1, warhead=0, rmax=70000, per_company=6, ind_lethality=0.120,
                  rounds_per_min=1.0),
    ),
    brigade=dict(
        battalions=[dict(tank=2, ifv=2)] * 3,                    # three combined-arms battalions
        inf_platoons_per_ifv_co=3, atgm_teams_per_bn=6,
        recon_cos=3, cav_ifv_per_troop=4,                        # cavalry squadron: 3 troops, scouts + Bradleys
        howitzer_batteries=3, rocket_batteries=0,                # fires battalion: 3 x 6 M109A7 (HIMARS is a division asset)
        at_cos=0, engineer_cos=2, support_cos=3, aviation_teams=1, readiness=0.95),
    notes="ABCT-like: three combined-arms battalions, a cavalry squadron, a fires battalion, a brigade engineer "
          "battalion, a support battalion, and a two-ship AH-64 attack weapons team. Modern fire control and "
          "thermals (low theta0, high sensor), fire-and-forget AT, fast C2, thin organic SHORAD.",
)

# ------------------------------------------------------------------ Russia (motor-rifle brigade-like)
RU = Nation(
    "Russia", "RU", quality=0.85, c2_delay_min=8.0, recon_uas=1.4, shorad=0.70,
    equipment=dict(
        **_enablers("IMR-2 / UR-77 engineer vehicles (notional)", 10, "Material support truck platoon (notional)", 20,
                    "Ka-52 pair (notional)", 4e-4, 0.45, 8000, 12),
        tank=_e(name="T-90M / T-72B3M mix (notional)", cls="tank", area=5.5, theta0=3.2e-4, theta_s=5.5e-4,
                theta_t=1e-3, sensor=0.75, signature=1.0, fire_rate=0.55, p_supp=0.4, supp_vuln=0.1, value=1.0,
                speed=75, protection=3, warhead=4, rmax=2500, per_company=10),
        ifv=_e(name="BMP-3 (notional)", cls="ifv", area=5.5, theta0=7e-4, theta_s=1.1e-3, theta_t=1e-3,
               sensor=0.7, signature=1.0, fire_rate=0.65, p_supp=0.45, supp_vuln=0.2, value=0.8, speed=75,
               protection=2, warhead=1, rmax=1500, secondary_warhead=3, secondary_rmax=3000, per_company=10),
        inf=_e(name="Motor-rifle platoon (notional)", cls="inf", area=1.0, theta0=2.2e-3, theta_s=4.5e-3,
               theta_t=1e-3, sensor=0.65, signature=0.55, fire_rate=0.75, p_supp=0.3, supp_vuln=1.0, value=0.5,
               speed=18, protection=0, warhead=0, rmax=500, secondary_warhead=2, secondary_rmax=300, per_company=3),
        atgm=_e(name="Kornet team (notional)", cls="atgm", area=1.0, theta0=3.5e-4, theta_s=4e-3, theta_t=3e-4,
                sensor=0.75, signature=0.35, fire_rate=0.3, p_supp=0.1, supp_vuln=1.0, value=1.0, speed=15,
                protection=0, warhead=3, rmax=4000, ammo=4, per_company=6),
        recon=_e(name="Recon section (notional)", cls="recon", area=1.0, theta0=2e-3, theta_s=4e-3, theta_t=1e-3,
                 sensor=1.8, signature=0.25, fire_rate=0.5, p_supp=0.1, supp_vuln=1.0, value=0.4, speed=15,
                 protection=0, warhead=0, rmax=400, per_company=6),
        howitzer=_e(name="2S19 Msta battery (notional)", cls="howitzer", area=8.0, theta0=1e-3, theta_s=1e-3,
                    theta_t=1e-3, sensor=0.3, signature=1.2, fire_rate=0.0, p_supp=0.0, supp_vuln=0.5, value=0.9,
                    speed=55, protection=1, warhead=0, rmax=24000, per_company=6, ind_lethality=0.024,
                    rounds_per_min=4.0),
        rocket=_e(name="BM-21 / Tornado-G battery (notional)", cls="rocket", area=8.0, theta0=1e-3, theta_s=1e-3,
                  theta_t=1e-3, sensor=0.3, signature=1.3, fire_rate=0.0, p_supp=0.0, supp_vuln=0.6, value=0.9,
                  speed=60, protection=1, warhead=0, rmax=30000, per_company=6, ind_lethality=0.060,
                  rounds_per_min=6.0),
    ),
    brigade=dict(
        battalions=[dict(tank=1, ifv=3)] * 3 + [dict(tank=3, ifv=0)],   # three MR battalions (as BTGs) + a tank battalion
        inf_platoons_per_ifv_co=3, atgm_teams_per_bn=6,
        recon_cos=2, cav_ifv_per_troop=3,                        # reconnaissance battalion (recon + UAV company)
        howitzer_batteries=6, rocket_batteries=3,                # two tube battalions + a rocket battalion
        at_cos=3,                                                # anti-tank battalion: 3 Kornet companies
        engineer_cos=2, support_cos=3, aviation_teams=1, readiness=0.85),
    notes="Motor-rifle brigade fielding three BTG-sized motor-rifle battalions (each with a tank company) and a "
          "tank battalion, a reconnaissance battalion, two tube artillery battalions and a rocket battalion (the "
          "fires weight a BTG draws on), an anti-tank battalion, engineer-sapper battalion, material support "
          "battalion, a Ka-52 pair, and a dense organic air-defence battalion.",
)

# ------------------------------------------------------------------ China (heavy combined-arms brigade-like)
CN = Nation(
    "China", "CN", quality=0.90, c2_delay_min=6.0, recon_uas=1.7, shorad=0.70,
    equipment=dict(
        **_enablers("GSL-130 / GCZ-110 engineer vehicles (notional)", 12, "Service support truck platoon (notional)", 20,
                    "Z-10 pair (notional)", 3.5e-4, 0.45, 7000, 12),
        tank=_e(name="Type 99A / 96A mix (notional)", cls="tank", area=6.0, theta0=2.8e-4, theta_s=4.5e-4,
                theta_t=9e-4, sensor=0.85, signature=1.0, fire_rate=0.6, p_supp=0.4, supp_vuln=0.1, value=1.0,
                speed=75, protection=3, warhead=4, rmax=2800, per_company=10),
        ifv=_e(name="ZBD-04A (notional)", cls="ifv", area=5.8, theta0=6.5e-4, theta_s=1e-3, theta_t=1e-3,
               sensor=0.8, signature=1.0, fire_rate=0.65, p_supp=0.45, supp_vuln=0.2, value=0.8, speed=75,
               protection=2, warhead=1, rmax=1600, secondary_warhead=3, secondary_rmax=3000, per_company=10),
        inf=_e(name="Mechanized platoon (notional)", cls="inf", area=1.0, theta0=2e-3, theta_s=4.2e-3,
               theta_t=1e-3, sensor=0.7, signature=0.5, fire_rate=0.75, p_supp=0.3, supp_vuln=1.0, value=0.5,
               speed=18, protection=0, warhead=0, rmax=500, secondary_warhead=2, secondary_rmax=300, per_company=3),
        atgm=_e(name="HJ-12 team (notional)", cls="atgm", area=1.0, theta0=2.8e-4, theta_s=4e-3, theta_t=2e-4,
                sensor=0.85, signature=0.3, fire_rate=0.3, p_supp=0.1, supp_vuln=1.0, value=1.0, speed=15,
                protection=0, warhead=3, rmax=3000, ammo=3, per_company=6),
        recon=_e(name="Recon section (notional)", cls="recon", area=1.0, theta0=2e-3, theta_s=4e-3, theta_t=1e-3,
                 sensor=2.0, signature=0.22, fire_rate=0.5, p_supp=0.1, supp_vuln=1.0, value=0.4, speed=15,
                 protection=0, warhead=0, rmax=400, per_company=6),
        howitzer=_e(name="PCL-181 battery (notional)", cls="howitzer", area=8.0, theta0=1e-3, theta_s=1e-3,
                    theta_t=1e-3, sensor=0.3, signature=1.2, fire_rate=0.0, p_supp=0.0, supp_vuln=0.5, value=0.9,
                    speed=65, protection=1, warhead=0, rmax=25000, per_company=6, ind_lethality=0.027,
                    rounds_per_min=4.0),
        rocket=_e(name="PHL-03 battery (notional)", cls="rocket", area=8.0, theta0=1e-3, theta_s=1e-3,
                  theta_t=1e-3, sensor=0.3, signature=1.3, fire_rate=0.0, p_supp=0.0, supp_vuln=0.5, value=1.0,
                  speed=60, protection=1, warhead=0, rmax=70000, per_company=6, ind_lethality=0.090,
                  rounds_per_min=2.0),
    ),
    brigade=dict(
        battalions=[dict(tank=2, ifv=2)] * 4,                    # four combined-arms battalions
        inf_platoons_per_ifv_co=3, atgm_teams_per_bn=6,
        recon_cos=3, cav_ifv_per_troop=3,                        # reconnaissance battalion
        howitzer_batteries=3, rocket_batteries=2,                # artillery battalion: PCL-181 batteries + PHL-03
        at_cos=0, engineer_cos=2, support_cos=3, aviation_teams=1, readiness=0.90),
    notes="Heavy combined-arms brigade: four combined-arms battalions, reconnaissance battalion, artillery "
          "battalion, engineer and chemical-defence battalion, service support battalion, a Z-10 pair, and an "
          "organic air-defence battalion. Blended new/old fleet, heavy UAS use.",
)

# ------------------------------------------------------------------ Iran (armored brigade-like)
IR = Nation(
    "Iran", "IR", quality=0.70, c2_delay_min=12.0, recon_uas=1.5, shorad=0.40,
    equipment=dict(
        **_enablers("Older dozer/bridging engineer vehicles (notional)", 10, "Support truck platoon (notional)", 16,
                    "AH-1J / Toufan pair (notional)", 6e-4, 0.4, 5000, 8),
        tank=_e(name="T-72S / Karrar mix (notional)", cls="tank", area=5.5, theta0=4.5e-4, theta_s=8e-4,
                theta_t=1.2e-3, sensor=0.55, signature=1.0, fire_rate=0.45, p_supp=0.4, supp_vuln=0.15, value=1.0,
                speed=65, protection=3, warhead=4, rmax=2200, per_company=10),
        ifv=_e(name="BMP-2 / Boragh (notional)", cls="ifv", area=5.5, theta0=9e-4, theta_s=1.4e-3, theta_t=1.1e-3,
               sensor=0.55, signature=1.0, fire_rate=0.55, p_supp=0.4, supp_vuln=0.25, value=0.75, speed=65,
               protection=2, warhead=1, rmax=1300, secondary_warhead=2, secondary_rmax=1500, per_company=10),
        inf=_e(name="Mechanized platoon (notional)", cls="inf", area=1.0, theta0=2.6e-3, theta_s=5e-3, theta_t=1e-3,
               sensor=0.6, signature=0.55, fire_rate=0.7, p_supp=0.3, supp_vuln=1.0, value=0.5, speed=17,
               protection=0, warhead=0, rmax=450, secondary_warhead=2, secondary_rmax=300, per_company=3),
        atgm=_e(name="Toophan team (notional)", cls="atgm", area=1.0, theta0=4.5e-4, theta_s=4e-3, theta_t=4e-4,
                sensor=0.6, signature=0.35, fire_rate=0.25, p_supp=0.1, supp_vuln=1.0, value=1.0, speed=15,
                protection=0, warhead=3, rmax=3500, ammo=4, per_company=8),
        recon=_e(name="Recon section (notional)", cls="recon", area=1.0, theta0=2e-3, theta_s=4e-3, theta_t=1e-3,
                 sensor=1.4, signature=0.3, fire_rate=0.5, p_supp=0.1, supp_vuln=1.0, value=0.4, speed=15,
                 protection=0, warhead=0, rmax=400, per_company=6),
        howitzer=_e(name="M109 / Raad-2 battery (notional)", cls="howitzer", area=8.0, theta0=1e-3, theta_s=1e-3,
                    theta_t=1e-3, sensor=0.3, signature=1.2, fire_rate=0.0, p_supp=0.0, supp_vuln=0.6, value=0.9,
                    speed=50, protection=1, warhead=0, rmax=18000, per_company=6, ind_lethality=0.020,
                    rounds_per_min=2.5),
        rocket=_e(name="Fajr / HM-20 battery (notional)", cls="rocket", area=8.0, theta0=1e-3, theta_s=1e-3,
                  theta_t=1e-3, sensor=0.3, signature=1.3, fire_rate=0.0, p_supp=0.0, supp_vuln=0.6, value=0.9,
                  speed=55, protection=1, warhead=0, rmax=25000, per_company=6, ind_lethality=0.045,
                  rounds_per_min=4.0),
    ),
    brigade=dict(
        battalions=[dict(tank=3, ifv=0)] * 2 + [dict(tank=0, ifv=3)],   # two tank battalions + a mechanised battalion
        inf_platoons_per_ifv_co=3, atgm_teams_per_bn=8,
        recon_cos=2, cav_ifv_per_troop=2,
        howitzer_batteries=3, rocket_batteries=2,                # artillery group: M109/Raad-2 + Fajr rockets
        at_cos=1, engineer_cos=2, support_cos=3, aviation_teams=1, readiness=0.75),
    notes="Armoured brigade: two tank battalions and a mechanised battalion, reconnaissance company-plus, "
          "artillery group, engineer battalion, support battalion, an AH-1J pair, and mixed-age air defence. "
          "Older fleet and optics (highest theta0, lowest sensor), lower readiness and slower C2, heavier "
          "dismounted AT allocation.",
)

NATIONS = {n.code: n for n in (US, RU, CN, IR)}
NATION_NAMES = {n.code: n.name for n in NATIONS.values()}


# ------------------------------------------------------------------ derived kill probabilities
def pk_between(shooter: Equipment, target: Equipment, quality=1.0):
    """P(kill | hit) and the range it applies at, using the better of the main and secondary
    weapons against the target's protection tier. Returns (pk, rmax)."""
    main = CLASS_PK[shooter.warhead, target.protection]
    best, rng = main, shooter.rmax
    if shooter.secondary_warhead >= 0:
        sec = CLASS_PK[shooter.secondary_warhead, target.protection]
        if sec > main:
            best, rng = sec, shooter.secondary_rmax
    return float(np.clip(best * quality, 0, 1)), float(rng)


def pk_matrix(nation_a: Nation, classes_a, nation_b: Nation, classes_b):
    """PK and RMAX matrices for shooters (nation_a, classes_a) against targets (nation_b, classes_b)."""
    PK = np.zeros((len(classes_a), len(classes_b)))
    RM = np.zeros_like(PK)
    for i, ca in enumerate(classes_a):
        for j, cb in enumerate(classes_b):
            if ca in INDIRECT or ca in AVIATION or cb in AVIATION:
                continue
            PK[i, j], RM[i, j] = pk_between(nation_a.eq(ca), nation_b.eq(cb), nation_a.quality)
    return PK, RM


# ------------------------------------------------------------------ bridge to the entity engine
ENTITY_BLUE = ["tank", "ifv", "inf", "recon"]     # entity-engine slots 0..3 (Tank, IFV, Inf, Scout)
ENTITY_RED = ["atgm", "inf"]                      # entity-engine slots 4..5 (Red AT, Red inf)


def entity_params(blue: Nation, red: Nation):
    """Parameter dict in the entity engine's six-slot schema: Blue's tank/IFV/inf/recon attack,
    Red's AT teams and infantry defend. Lets the calibration ladder run any nation pair."""
    from .units import default_params
    P = default_params()
    eqs = [blue.eq(c) for c in ENTITY_BLUE] + [red.eq(c) for c in ENTITY_RED]
    for key in ("area", "theta0", "theta_s", "theta_t", "sensor", "signature", "fire_rate", "p_supp",
                "supp_vuln", "value", "speed", "ammo"):
        P[key] = np.array([getattr(e, key) for e in eqs], float)
    P["fire_rate"][:4] *= blue.quality
    P["fire_rate"][4:] *= red.quality
    P["speed"][4:] = 0.0                                   # the defense is static in the entity engine
    PK = np.zeros((6, 6))
    RM = np.zeros((6, 6))
    PK[:4, 4:], RM[:4, 4:] = pk_matrix(blue, ENTITY_BLUE, red, ENTITY_RED)
    PK[4:, :4], RM[4:, :4] = pk_matrix(red, ENTITY_RED, blue, ENTITY_BLUE)
    P["PK"], P["RMAX"] = PK, RM
    P["k_det"] = 0.5 * (blue.recon_uas + red.recon_uas) / 1.5
    return P
