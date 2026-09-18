# Technical reference

The full specification of the model, the statistics, the calibration ladder and the brigade
engine. The project overview, the headline result and the verification status table are in the
[README](../README.md); the derivations are in
[Appendix A: mathematical methodology](appendix-a-methodology.md).

Every quantitative statement below names the script that produces it. Numbers marked
*(committed artifact)* are reproduced by a JSON file under `runs/`; numbers marked
*(not committed)* have to be regenerated with the command given.

## Contents

- [2. Repository layout](#2-repository-layout)
- [3. Ground: terrain acquisition and representation](#3-ground-terrain-acquisition-and-representation)
- [4. Scenario: placing both forces and routing Blue](#4-scenario-placing-both-forces-and-routing-blue)
- [5. Engine: the combat mechanics](#5-engine-the-combat-mechanics)
- [6. Parameters](#6-parameters)
- [7. Statistics](#7-statistics)
- [8. Calibration ladder](#8-calibration-ladder)
- [9. Sensitivity analysis: DOE, metamodel, Sobol](#9-sensitivity-analysis-doe-metamodel-sobol)
- [10. Discretization: tick and cell convergence](#10-discretization-tick-and-cell-convergence)
- [11. Reproducibility: seeds, substreams, manifests](#11-reproducibility-seeds-substreams-manifests)
- [12. Scripts](#12-scripts)
- [13. The app](#13-the-app)
- [14. Tests](#14-tests)
- [15. Known limits and findings](#15-known-limits-and-findings)
- [16. Roadmap](#16-roadmap)
- [17. References](#17-references)
- [18. Data attribution and licences](#18-data-attribution-and-licences)
- [19. Brigade against brigade](#19-brigade-against-brigade)

---

## 2. Repository layout

```
app.py                    Streamlit app: company team (Ground / Results / Where losses happen / Playback)
app_brigade.py            Streamlit app: brigade vs brigade, nation pick, COA game
sim/
  units.py                unit types, Roster, rung rosters, default parameters
  terrain.py              Terrain grid, OSM + DEM fetch, rasterization, synthetic ground, LOS
  scenario.py             Red/SBF/OP placement, Blue routes, per-route LOS tables
  engine.py               the tick-based stochastic engagement engine, SimResult, substreams
  stats.py                intervals, sequential sampling, KN selection, survival, bands, shrinkage
  descriptors.py          terrain descriptors, terrain-space sampling, TerrainLibrary cache
  vignettes.py            the calibration ladder: rungs, free parameters, metrics
  calibrate.py            history matching, reference sources, stability/coverage/trend reports
  doe.py                  Latin hypercube design, GP metamodel, Sobol indices
  manifest.py             run manifests (code version, parameter hash, terrain snapshot)
  nations.py              notional equipment catalogs and brigade templates for US, CN, RU, IR
  brigade.py              aggregate brigade engine: coarse ground, company entities, plans, fires, C2
  c2.py                   command-and-control profiles, decision graphs, latency, lineages (Artesh / IRGC)
scripts/
  headline.py             the README's worked result: one paired COA comparison with CRN
  convergence.py          tick/cell convergence sweep
  brigade_table.py        the six-pairing brigade table, every rate with an interval
  calibrate_ladder.py     climb the ladder on a sampled terrain library, write a report
  doe_sobol.py            sensitivity analysis for one rung
  compare_coas.py         ranking-and-selection over Blue COAs with a Red-plan outer loop
  brigade_run.py          one brigade pairing with intervals
  brigade_game.py         3x3 attacker/defender scheme game, LP solution, bootstrap value
  aggregation_check.py    same company fight in both engines; reports the aggregation gap
  rung5_calibrate.py      fits the aggregate kill-rate correction against the entity engine
  c2_sensitivity.py       Sobol indices of the five C2 factors on P(objective seized)
data/aggregation_correction.json   the shipped rung-5 fit (multipliers and residual gaps)
tests/                    pytest suite (engine, statistics, calibration, headless app)
data/reference_template.csv   column schema for real reference outcomes
notebooks/                the earlier lane-model notebook (no terrain), kept for the derivations
cache/terrains/           (created at run time) terrain library with descriptors and fetch dates
runs/                     committed result artifacts (headline, convergence, sobol_rung1,
                          brigade_table) plus their manifests; scripts write new outputs here
docs/                     technical-reference.md and appendix-a-methodology.md
requirements-lock.txt     exact versions the committed artifacts were produced with
```

Data flow: `terrain.py` produces a `Terrain`; `scenario.py` turns it and a `ScenarioConfig` into
per-unit route tables (`Scenario.tables(armor)`); `engine.simulate` runs replications on those
tables and returns a `SimResult`; `stats.py` turns results into intervals; the calibration and
DOE modules wrap `run_vignette` (scenario + engine on a rung roster) in loops over parameters and
terrains.

---

## 3. Ground: terrain acquisition and representation

### 3.1 The grid

`Terrain` is a square grid centred on the objective: `half` metres each way, `cell` metres per
cell (default 2000 m / 25 m, giving 160 × 160 cells). Local coordinates are metres east (x) and
north (y) of the objective; `xy_to_lonlat` / `lonlat_to_xy` use a local equirectangular
projection, accurate to well under a metre at these scales.

Per cell:

| Field | Meaning |
|---|---|
| `ground` | bare-earth elevation (m) |
| `surface` | ground + obstruction height: building height or 15 m canopy |
| `cls` | 0 Open, 1 Road, 2 Forest, 3 Building, 4 Water |
| `source`, `name`, `notes`, `fetched_at` | provenance for the manifest |

`Terrain.save(path)` / `Terrain.load(path)` write and read a compressed `.npz`.

### 3.2 Real ground (`osm_terrain`)

1. `geocode(query)` → Nominatim.
2. `fetch_dem(t)` → AWS Terrain Tiles (Terrarium PNG, zoom 13, about 19 m at mid latitudes),
   bilinearly resampled onto the grid.
3. `fetch_osm(t)` → one Overpass query for buildings, landuse/natural (wood, scrub, water),
   waterways, and highways with geometry.
4. `rasterize(t, elements)` classifies each way (`_classify`) and paints it:
   - buildings: polygon fill, height from `height`, `building:levels` × 3.2 m, or 6 m default
   - forest/scrub: polygon fill, 15 m canopy
   - water polygons and waterways (stroked 30 m wide, or from `width`)
   - roads stroked by class (motorway 24 m … tertiary 10 m, others 8 m); footways, paths and
     similar are skipped; **bridges** (`bridge=yes`) are painted over water so the routing graph
     sees a crossing.
   `fetched_at` is stamped with the UTC time of the fetch.

Multipolygon relations (courtyards, complex woods) are not handled; only ways are. See the
roadmap.

### 3.3 Synthetic ground (`synthetic_terrain`)

For offline work and for sampling the terrain space, `synthetic_terrain(half, cell, seed,
building_density, forest_share, relief, road_density, river)` generates a village around the
objective, a ridge and hills scaled by `relief`, wood blobs placed until `forest_share` of the
grid is forest, a north-south river with a road bridge, and a road network. The four named
settings are the knobs the terrain sampler turns (section 8.3). `demo_terrain()` is the default
offline valley used by the app and the tests.

### 3.4 Line of sight (`los_fraction`)

`los_fraction(t, obs_xy, eye_h, tgt_xy, tgt_h, samples=160)` casts a ray from observer eye
height to each of several target heights (three per unit type: hull/turret for vehicles, prone /
kneeling / standing for dismounts), samples the surface along the ray, and returns the fraction
of target heights that clear every intermediate sample. That fraction is the **visible fraction**
used everywhere below: a 0.33 means one of three sample heights is visible, so a unit is
one-third exposed. A building's own footprint does not block a unit standing in it; Red in a
building fights from 3 m up (`BUILDING_FLOOR`).

Cost: one ray per (observer point, target unit, target height). Scenario construction on the
default grid takes a few seconds, dominated by these rays.

---

## 4. Scenario: placing both forces and routing Blue

`Scenario(terrain, cfg, roster=COMPANY, seed=0, placement_temperature=0.0, route_noise=0.0)`

### 4.1 `ScenarioConfig`

| Field | Default | Meaning |
|---|---|---|
| `bearing_deg` | 90 | direction from the objective toward Blue's start (0 = north) |
| `start_dist` | 1700 | Blue start line distance (m); must fit inside the area |
| `sbf_min`, `sbf_max` | 900, 1400 | ring (m from objective) searched for support-by-fire positions |
| `op_min`, `op_max` | 1000, 1500 | ring searched for scout observation posts |
| `dismount_dist` | 450 | infantry dismounts once its vehicle route is this close to the objective |
| `red_radius` | 300 | Red positions are chosen within this radius of the objective |
| `danger_close` | 250 | no fire mission on a Red unit closer than this to any Blue unit |
| `max_minutes` | 60 | time limit; replications still running are censored |

### 4.2 Red placement (`_place_red`)

Candidate cells are sampled within `red_radius` (up to 250 for AT teams, 250 within
min(`red_radius`, 180 m) for infantry). Each candidate is scored by the mean visible fraction
toward a fan of points along Blue's likely approach (six ranges × five bearings around
`bearing_deg`; AT teams look 600 m to `start_dist`, infantry 150–800 m). Infantry scores add
0.6 × (1 − concealment) so buildings and woods are favoured. Positions are then picked in score
order with a minimum spacing (60 m AT, 40 m infantry).

`placement_temperature` makes this stochastic: scores get `temperature × Gumbel` noise before
ordering, which is a Plackett–Luce sample over the good positions. Temperature 0 is the
deterministic best-scored plan; 0.5 (the app's multi-plan setting, and the calibration default)
draws a different plausible defense per seed. Running several seeds and concatenating results
averages over Red's plan instead of conditioning on one; see section 7.9.

### 4.3 Blue placement and routes

- **Start line:** `start_dist` out along the bearing, units spread laterally by
  `Roster.start_offsets()` (100 m between vehicles, 200 m between scouts).
- **Routes:** Dijkstra on an 8-connected grid graph with class-dependent costs
  (`VEH_COST` = open 1, road 0.7, forest 4, building ∞, water ∞; `INF_COST` = 1, 0.9, 1.5, 2.5,
  ∞). Bridges are road cells, so vehicles cross rivers there and nowhere else. `route_noise > 0`
  multiplies the cost grids by a smooth lognormal field so routes vary between seeds.
- **Tanks:** to the objective (Maneuver, Unsupported) or to a support-by-fire position (SBF).
- **Infantry:** vehicle route to `dismount_dist`, then the foot route to the objective; the
  hand-off index is stored so the engine knows when the unit is mounted (fights as IFV) or
  dismounted (fights as Inf team).
- **Scouts:** foot route to an observation post.
- **SBF and OP positions** (`_place_sbf_op`): candidates in the rings, scored by mean visible
  fraction of the Red positions plus a concealment bonus (0.2 SBF, 0.5 OP), picked with spacing
  (60 m, 150 m); one SBF position per tank, one OP per scout.

### 4.4 Route tables (`Scenario.tables(armor)`)

For each Blue unit and every cell of its route: position, cumulative distance, mounted flag,
speed factor (`VEH_SPEED` / `INF_SPEED` by class), concealment, distance to every Red unit,
visible fraction of each Red unit from there (`visR`), and visible fraction of the Blue unit from
each Red position (`visB`). The engine only interpolates along these tables; it never casts a ray
at run time, which is what makes 2000 replications take a couple of seconds.

---

## 5. Engine: the combat mechanics

Derivations for every mechanic are in Appendix A.1–A.7.

`simulate(tab, armor, recon, fires, red_coa, n, seed, P=None, red_trigger=600, n_log=20, tick_s=7.5)`

Time advances in ticks of `tick_s` seconds (default 7.5, see section 10). All `n` replications
run in lockstep as vectorized arrays. Each tick, for every live unit, in this order:

### 5.1 Movement intent

A unit moves if alive, not at its stop point, and not suppressed. In the Maneuver COA tanks
throttle so they do not outrun the slowest live infantry unit (stop point = own route length −
infantry's remaining distance). Speed per tick is `f · speed[type] · speed_fac(cell)`, with
`f = tick_s / 15`.

### 5.2 Exposure

A unit's exposure is the terrain visible fraction from the tables, times concealment of its cell
(open 1, forest 0.5, building 0.35), times 0.5 if suppressed (it goes to ground). Red in a
prepared position additionally has posture exposure `red_posture` (0.6). These products give
`visR[i, j]` (Red j as seen by Blue i) and `visB[i, j]` (Blue i as seen by Red j), each in [0, 1].

### 5.3 Detection

A persistent per-pair detection with a per-tick hazard

```
rate = sensor[shooter] · signature[target] · vis · (1 + φ_fire · target_fired_last_tick)
       · (1 + μ_move · target_moving)        [Red detecting Blue only]
       · (1000 / max(D, 50))²  · k_det
P(detect this tick) = 1 − exp(−f · rate)
```

Detections are dropped when either unit dies. **Information sharing:** with a live scout and
`recon=True` the whole Blue force sees the union of detections (the "net"); otherwise a unit sees
its own detections plus those of Blue units within 150 m. Red always shares (a prepared defense
with wire and radios).

### 5.4 Target selection: random-utility multinomial logit

Each shooter chooses among its detected, in-range, visible, killable targets plus a hold-fire
option, by Gumbel-max sampling (`gumbel_choice`), which is exactly a multinomial logit:

```
V_target = log( P_H · P_K · value[target] ) + b_threat · [target shot at me last tick]
V_hold   = v0 + dv0_supp · [I am suppressed]    (+ a large constant for scouts, who never shoot,
                                                 and for Red in a Late COA before the trigger)
P(choose k) ∝ exp(λ · V_k),   λ = lam_blue or lam_red, × lam_supp if suppressed
```

`choice_prob` reports the model probability of the option actually chosen; the playback tab shows
it for every shot in the logged replications.

**Red COAs.** `Early`: Red engages whenever it has a shot. `Late`: Red holds fire until any Blue
unit is within `red_trigger` metres, then the ambush is sprung for that replication.

### 5.5 Hit probability: circular-normal aim error

```
σ² = D² · (θ0² + θs² · shooter_moving + θt² · target_moving) · (1 + κ_supp · shooter_suppressed)
P_H = 1 − exp( − vis · A / (2π σ²) )
```

`A` is the target's presented area at full exposure; multiplying by the visible fraction `vis`
shrinks the target rather than the shot. `test_hit_prob_matches_brute_force` checks this against
a two-dimensional normal draw. Kill given hit is `PK[shooter, target]`; `RMAX` caps range.

### 5.6 Fire, suppression, ammunition

A chosen target is fired on with probability `fire_rate[type]` per 15 s (rescaled to the tick).
A non-killing shot suppresses with probability `p_supp[shooter] · exp(−D/1000) · supp_vuln[target]`.
Suppression lasts `supp_ticks_direct` (2 × 15 s) and blocks movement, raises dispersion, raises
the hold-fire utility, and lowers λ. Both sides resolve simultaneously within a tick. Red AT
teams carry 3 missiles.

### 5.7 Indirect fires

With `fires=True`, each tick a fire mission happens with probability `p_mission_recon` (net up)
or `p_mission_norecon` (net down), against a uniformly chosen Red unit that some Blue unit has
detected and that is outside `danger_close` of every Blue unit. The mission kills with
`fires_kill` (0.08) and suppresses with `fires_supp` (0.85) for `supp_ticks_fires` (4 × 15 s).
Recon therefore raises the mission rate and widens the candidate list: that is the modelled
recon × fires synergy.

### 5.8 Breakpoints and outcomes

Each replication draws a breakpoint for each side from `Beta(bp_blue)` / `Beta(bp_red)` on the
fraction of fighting units lost (scouts do not count). When a side's losses reach its
breakpoint the replication ends; if both break in the same tick it is scored as a Blue failure.
`SimResult.cause` records 0 = time limit (censored), 1 = Blue broke, 2 = Red broke.

### 5.9 `SimResult`

| Field | Shape | Meaning |
|---|---|---|
| `win`, `timeout` | (n,) | Red broke first; time limit reached |
| `blue_losses`, `red_losses` | (n,) | units lost |
| `minutes` | (n,) | decision time (time limit if censored) |
| `cause` | (n,) | 0 censored, 1 Blue broke, 2 Red broke |
| `blue_strength`, `red_strength` | (n, T+1) | fighting units alive per tick |
| `blue_killed`, `red_killed` | (n, NB) / (n, NR) | which units died |
| `death_xy`, `death_rep` | (m, 2), (m,) | where Blue units died, and in which replication |
| `blue_shots`, `blue_hits`, `red_shots`, `red_hits` | (n, 3) | direct-fire counts by range band (<500, 500–1000, >1000 m) |
| `frames`, `events` | | replay snapshots and the annotated shot log for the first `n_log` replications |
| `tick_s`, `ticks`, `n_plans` | | resolution, ticks run, Red plans pooled |

`concat_results([...])` stacks results from runs on different Red plans (section 7.9), keeping
the replay from the first.

### 5.10 Rosters

`Roster(name, btype, rtype)` fixes the array shapes. Units a COA omits (no scouts without recon,
no infantry in Unsupported) start the run dead so shapes, and therefore random-number streams,
never change between COAs. The four rung rosters:

| Rung | Name | Blue | Red |
|---|---|---|---|
| 1 | `DUEL` | 1 tank | 1 AT team |
| 2 | `SECTION` | 2 tanks, 2 inf teams | 1 AT team, 2 inf teams |
| 3 | `PLATOON` | 2 tanks, 4 inf teams, 1 scout | 2 AT, 4 inf |
| 4 | `COMPANY` | 4 tanks, 6 inf teams, 2 scouts | 4 AT, 6 inf |

`make_roster(name, tanks, inf, scouts, at, rinf)` builds others.

---

## 6. Parameters

`default_params()` returns one dict. Per-tick quantities are stated for a **15 s tick**; the
engine rescales them to whatever `tick_s` it runs at (probabilities as `1 − (1 − p)^f`, hazards
and speeds by `f`, durations by `1/f`). Type index: 0 Tank, 1 IFV (mounted inf), 2 Inf team,
3 Scout, 4 Red AT team, 5 Red inf team.

| Key | Type | Default | Role |
|---|---|---|---|
| `area` | per type, m² | 6, 6, 1, 1, 1, 1 | presented area at full exposure |
| `theta0` | per type, rad | 3e-4, 8e-4, 2e-3, 2e-3, 4e-4, 2e-3 | stationary aim dispersion |
| `theta_s` | per type, rad | 5e-4, 1e-3, 4e-3, 4e-3, 4e-3, 4e-3 | added when the shooter moves |
| `theta_t` | per type, rad | 1e-3 ×4, 3e-4, 1e-3 | lead error vs a moving target |
| `sensor` | per type | .5, .4, .8, 2, .8, .6 | detection hazard multiplier (as shooter) |
| `signature` | per type | 1, 1, .5, .2, .3, .3 | detection hazard multiplier (as target) |
| `fire_rate` | per type, per 15 s | .6, .6, .8, .5, .3, .8 | P(ready to fire) |
| `p_supp` | per type | .4, .4, .3, .1, .1, .3 | suppressiveness of a non-killing shot |
| `supp_vuln` | per type | .1, .1, 1, 1, 1, 1 | susceptibility to suppression |
| `value` | per type | 1, .9, .5, .4, 1, .6 | target value in the choice model |
| `speed` | per type, m per 15 s | 75, 75, 20, 15, 0, 0 | open-ground speed |
| `ammo` | per type | ∞ except AT 3 | rounds |
| `PK` | 6×6 | see `units.py` | kill given hit, shooter × target |
| `RMAX` | 6×6 m | tank 2500, IFV 1500, inf 500, scout 400, ATGM 2500, RPG 300, small arms 500 | max range |
| `lam_blue`, `lam_red`, `lam_supp` | | 4, 3, 0.3 | choice rationality; multiplier when suppressed |
| `v0`, `dv0_supp`, `b_threat` | | log 0.01, 3, 1 | hold-fire utility, its bump when suppressed, threat weight |
| `kappa_supp` | | 3 | dispersion multiplier (1 + κ) when suppressed |
| `mu_move`, `phi_fire` | | 1, 5 | detection bonus for moving / firing targets |
| `k_det` | | 1 | global detection scale |
| `red_posture` | | 0.6 | exposure of a prepared position |
| `supp_ticks_direct`, `supp_ticks_fires` | 15 s ticks | 2, 4 | suppression duration |
| `p_mission_recon`, `p_mission_norecon` | per 15 s | .5, .2 | fire-mission probability |
| `fires_supp`, `fires_kill` | | .85, .08 | mission effects |
| `bp_blue`, `bp_red` | Beta(a, b) | (10, 10), (11, 9) | breakpoint on fraction lost |

`copy_params(P)` deep-copies the arrays; `vignettes.apply_theta` writes calibration values in.

---

## 7. Statistics

Derivations and the reasons for each estimator are in Appendix A.8–A.12 and A.18.

All in `sim/stats.py`. One replication is one observation. Nothing here corrects for the fact
that the parameters are notional: intervals describe simulation error, and, in the calibration
tools, parameter uncertainty inside the NROY region. They never describe model error.

### 7.1 Proportions: Wilson score interval (`wilson`)

Used for P(win) and every other proportion. Stays in [0, 1] and keeps its coverage near 0 and 1,
where the Wald interval collapses (a 0/200 result gets [0, 0.019], not [0, 0]).

### 7.2 Means, paired differences, ratios (`mean_ci`, `paired_ci`, `ratio_ci`)

Normal intervals on means (n ≥ 200 in the app). Paired differences between COAs use the
per-replication difference, which is where common random numbers pay off. The exchange ratio
(Red losses / Blue losses) is a ratio of means with a delta-method interval, including the
covariance between numerator and denominator.

### 7.3 Common random numbers and the CRN diagnostic (`crn_gain`)

Two COAs run with the same seed on the same roster consume identical random numbers in
identical order (section 11). `crn_gain(a, b)` reports the achieved correlation between their
per-replication outcomes and the ratio of paired-difference variance to what independent runs
would give. The app prints both: a correlation of 0.6 means the paired comparison needed 2.5×
fewer replications than an independent one.

### 7.4 Curves: pointwise and simultaneous bands (`band`, `sup_t_band`)

Strength-over-time curves get pointwise 95% bands by default. The app's toggle switches to a
**sup-t bootstrap band** that covers the whole mean curve at once with 95% confidence
(bootstrap the maximum standardized deviation over time; the band is wider by that quantile
instead of 1.96). Use the simultaneous band when the claim is about the curve ("Blue strength
stays above Red's throughout") rather than one time point.

### 7.5 Sequential sampling (`n_for_half_width`, `run_until`)

`n_for_half_width(h)` gives the worst-case replications for a Wald half-width `h` on a
proportion. `run_until(sampler, h)` keeps drawing batches until the observed half-width is at
or below `h`. Stopping on the observed half-width makes the final interval slightly optimistic;
the bias is negligible for the 200-replication minimum enforced (Law, *Simulation Modeling and
Analysis*, sec. 9.4).

### 7.6 Ranking and selection: Kim–Nelson (`kn_select`)

When the question is "which COA is best", pairwise intervals are the wrong tool: eight COAs give
28 comparisons and the winner's interval is biased upward by selection. `kn_select` implements
the fully sequential indifference-zone procedure of Kim & Nelson (2001): after `n0`
replications per system it computes pairwise difference variances, then adds one replication at
a time to the survivors and eliminates any system that falls below another by more than a
shrinking continuation region. It guarantees P(correct selection) ≥ 1 − α whenever the best
beats every other by at least δ, and it spends replications where the contest is close.
`scripts/compare_coas.py` runs it over the eight Blue COAs. `test_kn_selects_best_system`
checks it on known Gaussian systems.

### 7.7 Survival analysis with censoring (`kaplan_meier`, `aalen_johansen`)

Replications that reach the time limit are **censored**, not failures. Time-to-decision curves
therefore use Kaplan–Meier (with Greenwood errors). Since a fight can end two ways (Blue broke,
Red broke) that compete, the app's "How the fights end" chart uses the Aalen–Johansen
cumulative incidence estimator: the share of fights that have ended each way by time t,
accounting for censoring and for the competing ending. Treating a competing event as censoring
would overstate both incidences. Both are tested against exponential truths.

### 7.8 Heatmap shrinkage (`shrink_rates`)

Per-cell death rates on the loss map are noisy where few deaths occurred. `shrink_rates(x, n)`
fits a beta-binomial prior across cells by a weighted method of moments (DerSimonian–Laird
form) and returns posterior means and 90% intervals. Cells with a handful of deaths are pulled
toward the overall rate; cells with many keep their raw rate. The map's weights are these
posterior means, so hot spots reflect where losses concentrate rather than where noise landed.

### 7.9 The Red-plan outer loop

A fixed defense makes every result conditional on one plan Red may not choose. The app's
"Red plans sampled" control and `run_vignette` build several `Scenario`s with
`placement_temperature = 0.5` and different seeds, run the same COA on each with a
plan-specific seed, and concatenate (`concat_results`). Intervals then include Red-plan
variation. With the outer loop the pairing across COAs is still exact within each plan, so CRN
holds.

### 7.10 The zero-sum COA game: LP value and bootstrap (`solve_zero_sum`, `bootstrap_game_value`)

`stats.solve_zero_sum` solves the Blue-vs-Red zero-sum matrix game by linear programming and
returns the value and both minimax mixes; `bootstrap_game_value` resamples the replications
behind each payoff cell to put an interval on the value. This came from the lane-model notebook
in `notebooks/` and now drives the brigade 3x3 scheme game (`scripts/brigade_game.py`,
section 19.5). It is **not yet wired to the company-team app**, where the natural Red strategy
set is the sampled-defense outer loop of 7.9; that is the open item in section 16.
Derivation: Appendix A.18.

---

## 8. Calibration ladder

The mathematics of history matching, sampling, coverage and identifiability is in Appendix A.13–A.15.

The parameters in section 6 are notional. The ladder is how they stop being notional: calibrate
the mechanics where they can be observed, small first, freezing what each rung learns before
climbing.

### 8.1 Rungs

| Rung | Engagement | Free parameters (`vignettes.FREE`) | Prior range | What identifies them |
|---|---|---|---|---|
| 1 | tank vs AT team | `theta0_tank`, `theta0_at` (log), `pk_tank_at`, `pk_at_tank` | 1.5e-4–8e-4, 2e-4–1.5e-3 rad; 0.4–0.9, 0.3–0.85 | hit fraction by range band (dispersion), kills per hit (P_K) |
| 2 | section vs team, Early and Late Red | `lam_blue`, `lam_red`, `k_det` (log), `b_threat`, `pk_inf_rinf`, `pk_rinf_inf` | 1–8, 1–8, 0.4–2.5, 0–2.5, 0.3–0.7 ×2 | time to decision, loss fractions; λ and threat weight need choice data (see 15) |
| 3 | platoon vs squad, SBF/Maneuver × fires | `kappa_supp`, `supp_scale` (log), `fires_supp`, `bp_blue_mean`, `bp_red_mean` | 1–6, 0.5–1.5, 0.5–0.95, 0.3–0.7, 0.35–0.75 | outcome and duration differences between fires on/off, loss fractions at decision |
| 4 | company team, all 8 COAs, Late Red | `p_mission_recon`, `phi_fire` (log) | 0.2–0.8, 1–10 | recon × fires interaction |

`ParamSpec(name, lo, hi, log)` maps a unit-cube coordinate to a natural value; `SETTERS` writes
each name into the parameter dict (breakpoint means become `Beta(20m, 20(1−m))`). Everything
not in a rung's `FREE` list is fixed at its default or at the value fixed by a lower rung.

### 8.2 Vignettes and metrics

`rung_vignettes(rung, half)` returns the standard vignette set per rung (`Vignette(rung, armor,
recon, fires, red_coa, cfg, red_trigger)`), with geometry from `rung_config` scaled to the rung
(start 900 / 1100 / 1300 / 1500 m, Red radius 60 / 120 / 200 / 300 m, 30 or 60 minute limit).
`run_vignette(terrain, vig, P, n, seed, n_plans=3, temperature=0.5)` runs it over three sampled
Red plans.

`metrics_of(result)` returns, and `metric_vars` gives the Monte Carlo variance of:

| Metric | Definition |
|---|---|
| `p_win` | mean of `win` |
| `blue_loss_frac`, `red_loss_frac` | mean fraction of the starting fighting units lost |
| `minutes` | mean decision time |
| `blue_hit_<band>`, `red_hit_<band>` | hits / shots in each range band (ratio of means, delta-method variance); NaN if fewer than 5 shots in the band |
| `blue_kill_per_hit`, `red_kill_per_hit` | enemy losses / own hits |

The band and kills-per-hit metrics exist because the first version of the duel could not
separate dispersion from lethality: with only outcome metrics the NROY box did not shrink at
all. Hit fractions by range identify θ; kills per hit identify P_K.

### 8.3 Terrain sampling with a holdout

"Tuned across varied terrain" must be testable, so terrains are chosen, not hand-picked.

- `descriptors.describe(terrain)` computes seven numbers: building density, forest share, road
  density, water share, relief (5th–95th percentile elevation spread), and openness at 500 m and
  1500 m (mean visible fraction between random point pairs at that separation, dismount heights).
- `synthetic_design(k)` draws a Latin-hypercube sample of generator settings (log scale on
  densities) over `SYNTHETIC_RANGES`; for real ground, build the candidate pool from a list of
  `dict(kind="osm", lat0, lon0, half, cell)` specs instead.
- `select_terrains(vectors, k, holdout_frac=0.2)` runs k-means on the standardized descriptors,
  picks the terrain nearest each centroid, and holds out the picks farthest from the centre of
  the set, so the holdout tests extrapolation rather than memorization.
- `TerrainLibrary(root)` caches terrains as `.npz` with descriptors, source, and `fetched_at` in
  `index.json`, keyed by a hash of the spec.

### 8.4 History matching (`calibrate.history_match`)

History matching (Craig et al. 1997; Vernon, Goldstein & Bower 2010) rules out the parameter
settings that cannot reproduce the reference outcomes instead of hunting for one best fit, which
suits a noisy simulator with few runs and possibly non-identifiable parameters. For each
candidate θ, terrain, vignette, and metric:

```
I(θ) = |E[f(θ)] − z| / sqrt( Var_mc + Var_ref + Var_disc )
```

with `Var_mc` the simulator's Monte Carlo variance at θ, `Var_ref` the reference's, and
`Var_disc` the model-discrepancy variance you tolerate (`disc_frac` of the reference value plus
0.02 absolute for proportions, 0.1 minute for times). θ is **not ruled out yet (NROY)** when the
largest I over all outputs is below `threshold` (3, the Pukelsheim three-sigma rule). Each wave
samples a Latin hypercube inside the current box, evaluates I, and shrinks the box to the hull
of the NROY points (never below `shrink_floor` per side). If nothing survives, the best 20% are
kept so the wave can continue and the log says so.

`run_ladder(rungs, terrains, reference, ...)` climbs: at each rung only that rung's `FREE`
parameters vary; after its waves the centre of the NROY box is written into `P` and the box is
recorded in `state.fixed` for the rungs above. `save_state` writes `state.json`.

`nroy_samples(wave, specs, k)` draws parameter vectors from the final box for prediction.

### 8.5 Reference sources

- `SyntheticReference(P_true, n=1500)`: the engine itself at hidden parameters, run at high
  replication count and cached per (terrain, vignette). A stand-in for a higher-fidelity model,
  and the way the pipeline is tested: the truth must land inside the NROY box.
- `TableReference(df)`: a table with columns
  `terrain_key, rung, armor, recon, fires, red_coa, <metric>, <metric>_var, ...`
  (see `data/reference_template.csv`). Any metric column may be omitted; missing metrics are
  skipped. Variances are the reference's own uncertainty: for instrumented data, the sampling
  variance of the observed rate; for a higher-fidelity model, its Monte Carlo variance.

To use real data: build one row per (terrain, vignette) from NTC/JRTC/JMRC instrumented
engagements or OneSAF/JCATS runs of the same vignettes, key the terrains to entries in the
terrain library (fetch the real ground for the training area), and run
`scripts/calibrate_ladder.py --reference your.csv`.

### 8.6 Stability, coverage, and error-trend diagnostics

"Stabilized" is defined, not felt:

- `stability_report(waves)`: per parameter, NROY width first vs last wave, centre shift, and
  the overlap between the last two waves' ranges. Flagged stable when overlap > 0.7 and the
  width stopped shrinking (last / first > 0.5).
- `coverage_report(state, ...)`: on **held-out** terrains, predict each reference output with
  `k_theta` draws from the final NROY boxes plus Monte Carlo and reference noise, form the
  central 90% prediction interval, and report the fraction of reference outputs it covers.
  Target: close to 0.90. Much lower means the model or its discrepancy term is wrong; much
  higher means the NROY region is too loose to be useful.
- `error_trend(rows, descriptors)`: regress prediction error on each terrain descriptor. A
  significant slope means the error depends on the kind of ground; parameter tuning cannot fix
  that, a mechanic is missing. Needs at least two held-out terrains with different descriptors.

Identifiability is read off the NROY volume: a box that stays near the prior volume after
several waves means those outputs do not constrain those parameters.

### 8.7 What the demo run showed

`python scripts/calibrate_ladder.py --rungs 1 2 --terrains 5 --keep 4 --reps 200 --design 20 --waves 2`
(three training terrains, one held out, synthetic reference). **These figures are not committed
and were not regenerated in the 2026-09-18 pass**; rerun the command before quoting them. Note
also what the run checks: the reference is produced by this same engine, so a successful recovery
validates the calibration pipeline, not the combat model.

- Rung 1: NROY volume 0.086 → 0.072 of the prior after two waves; all four truths inside.
- Rung 2: NROY volume 0.76 → 0.56; truths inside, but `lam_blue`, `lam_red`, `b_threat` spanned
  almost their whole prior. Aggregate outcomes do not identify choice-model parameters; see 15.
- Held-out 90% coverage: 0.90 over 31 outputs.
- Error trend: not computable with one held-out terrain (needs ≥ 2).

---

## 9. Sensitivity analysis: DOE, metamodel, Sobol

Estimators and the GP formulation are in Appendix A.16–A.17.

`sim/doe.py`, driven by `scripts/doe_sobol.py`.

1. `design(specs, m)`: a maximin-optimized Latin hypercube (`scipy.stats.qmc`) over a rung's free
   parameters. A nearly orthogonal Latin hypercube (Cioppa & Lucas 2007) can replace it by
   loading the SEED Center tables into the same unit-cube interface.
2. `run_design`: simulate every point; returns metric means and Monte Carlo variances.
3. `fit_metamodel(U, y, v)`: an anisotropic-RBF Gaussian process with a white-noise term whose
   floor is the mean Monte Carlo variance (stochastic-kriging style), so the GP does not fit
   noise. `Metamodel.noise_share` is MC variance / total output variance at the design points.
4. `loo_r2`: leave-one-out R². **Trust rule:** indices are reported as trustworthy only when
   LOO R² > 0.7. Below that, raise `--reps` (less noise) or `--design` (more points).
5. `sobol_indices(model, d, N)`: first-order and total-effect indices by the Saltelli scheme
   with Jansen's estimators (Saltelli et al. 2010) on scrambled Sobol samples, with bootstrap
   standard errors.

Demo (rung 1, 24 points × 300 replications, `--out runs/sobol_rung1.json`, committed): LOO R²
0.75–0.86, noise share 0.04–0.05; `pk_at_tank` dominates every output (total index 0.61–0.79),
`theta0_at` next (0.14–0.28), the tank's own parameters matter little. The per-output table is
in the README. At rung 2 with 200 replications the noise share was ~0.7 and LOO R² near zero
(**not committed**; regenerate with `python scripts/doe_sobol.py --rung 2 --reps 200`): the tool
correctly refused to trust those indices.

---

## 10. Discretization: tick and cell convergence

`scripts/convergence.py` reruns one COA pair (SBF, recon + fires, Late Red, offline demo ground,
seed 7) at four tick lengths and two cell sizes, and writes the table plus a run manifest.

Regenerated at commit `325b74d` with `python scripts/convergence.py --reps 1500`; committed as
`runs/convergence.json` and `runs/convergence_manifest.json`.

| cell (m) | tick (s) | n | P(win) [95%] | Blue losses [95%] | minutes |
|---|---|---|---|---|---|
| 25.0 | 30.00 | 1500 | 0.373 [0.349, 0.397] | 4.70 [4.61, 4.79] | 5.4 |
| 25.0 | 15.00 | 1500 | 0.565 [0.539, 0.590] | 3.81 [3.71, 3.91] | 4.9 |
| 25.0 | 7.50 | 1500 | 0.698 [0.674, 0.721] | 3.11 [3.01, 3.21] | 4.5 |
| 25.0 | 3.75 | 1500 | 0.702 [0.678, 0.725] | 3.02 [2.92, 3.12] | 4.3 |
| 12.5 | 30.00 | 1500 | 0.236 [0.215, 0.258] | 5.20 [5.12, 5.29] | 5.4 |
| 12.5 | 15.00 | 1500 | 0.401 [0.376, 0.426] | 4.41 [4.31, 4.51] | 5.0 |
| 12.5 | 7.50 | 1500 | 0.529 [0.504, 0.554] | 3.80 [3.70, 3.90] | 4.7 |
| 12.5 | 3.75 | 1500 | 0.567 [0.542, 0.592] | 3.60 [3.50, 3.71] | 4.5 |

**Tick.** At 25 m cells the 15 s tick is not converged: P(win) moves 0.13 from 15 s to 7.5 s,
while 7.5 s and 3.75 s agree within their intervals (0.698 vs 0.702). The mechanism is
simultaneity: within a tick every live unit fires, including units that die that tick, so longer
ticks let more of Red's first volley land before Blue's kills take effect. `TICK_DEFAULT = 7.5`
as a result; `TICK_S = 15` stays the reference tick for parameter statements, and the app exposes
the tick so the check can be repeated on any ground. Report results at 7.5 s; treat 15 s runs as
fast previews. At 12.5 m cells even 7.5 s has not settled (0.529 -> 0.567 at 3.75 s, intervals
touching): a finer grid needs a finer tick.

**Cell size is not converged.** P(win) differs by 0.169 at 7.5 s and 0.135 at 3.75 s between
25 m and 12.5 m cells, with non-overlapping intervals at every tick length; Blue losses differ by
0.6–0.7 of a vehicle. This is the largest single numerical artifact in the model - larger than
the effects the simulation is built to measure. The mechanism is line-of-sight sampling: a finer
grid resolves small terrain features that break sight lines. Consequence: absolute probabilities
from this model are cell-size artifacts as much as they are model outputs, and paired comparisons
at a fixed cell size are the only outputs with a defensible reading - and their stability across
cell sizes has not been tested. Carrying the sweep to 6.25 m and 3.125 m, or replacing
grid-sampled LOS with an exact vector method, is the top open verification item (section 16).

> **Correction (2026-09-18).** This section previously reported the 12.5 m rows as
> 0.261 / 0.487 / 0.621 / 0.655 and concluded that "cell size matters less (about 0.05 in P(win)
> between 25 m and 12.5 m at fine ticks)". Those numbers do not reproduce from
> `scripts/convergence.py` at the stated seed, and the conclusion was wrong in magnitude
> (0.14-0.17, not 0.05) and in consequence. The superseded numbers had no surviving provenance
> and have been deleted rather than reconciled.

---

## 11. Reproducibility: seeds, substreams, manifests

- **Seed.** `simulate(..., seed=[a, b, ...])` feeds a `numpy.random.SeedSequence`. Same seed,
  same tables, same tick → identical results (`test_engine_deterministic_and_bounded`).
- **Named substreams.** The engine draws every random number from one of the streams in
  `engine.STREAMS` (`det_b`, `det_r`, `choice_b`, ..., `breakpoints`), each an independent child
  of the seed. Because every draw has a fixed shape set by the roster, two COAs on the same
  roster consume the same numbers for the same purposes tick by tick: that is what makes the
  paired comparison valid, and it survives switching a mechanic on or off
  (`test_substreams_keep_crn_when_a_mechanic_is_switched`). Rule for engine changes: new draws
  get a **new stream name appended at the end** of `STREAMS`; never reorder it.
- **Manifests.** `manifest.make_manifest(P, seed, terrain, roster, cfg, extra)` records the
  UTC time, code version (git commit and dirty flag, or a hash of `sim/*.py` outside a repo), a
  hash of the parameter dict, the roster, the terrain's source, centre, extent, cell, fetch time
  and a hash of its surface, library versions, the scenario config, and anything passed in
  `extra` (the app passes both COAs, the Red COA, replications, plans, and tick). The app shows
  the manifest under an expander on the Results tab; `calibrate_ladder.py` writes one per run.
  OSM changes over time, so a result is only reproducible with the cached terrain (`cache/`),
  which the manifest identifies.

---

## 12. Scripts

All scripts add the repository root to `sys.path`, so run them from anywhere.

| Script | Purpose | Main options | Output |
|---|---|---|---|
| `scripts/headline.py` | the README's worked result: one paired COA comparison with CRN | `--reps 2000 --seed 1 --tick 7.5 --out runs/headline.json` | printed table + JSON + manifest |
| `scripts/convergence.py` | tick/cell sweep (section 10) | `--reps 2000 --seed 7 --out runs/convergence.json` | printed table + JSON + manifest |
| `scripts/brigade_table.py` | the six-pairing brigade table with an interval on every rate (19.4) | `--reps 100 --seed 0 --terrain-seed 1 --out runs/brigade_table.json` | printed table + JSON |
| `scripts/calibrate_ladder.py` | sample terrains, climb rungs, diagnostics | `--rungs 1 2 [3 4] --terrains 6 --keep 5 --reps 300 --design 40 --waves 2 --half 1500 --reference data.csv --seed 0 --out runs/<ts>` | `state.json`, `stability.json`, `coverage.json`, `error_trend.json`, `manifest.json`, `log.txt` |
| `scripts/doe_sobol.py` | design, GP, Sobol for one rung | `--rung 1 --design 60 --reps 600 --out runs/sobol.json` | printed indices + JSON |
| `scripts/compare_coas.py` | KN selection over the 8 Blue COAs | `--delta 0.03 --alpha 0.05 --n0 100 --plans 3 --red Late --max-n 6000` | printed table |

The brigade and entity scripts are also listed in 19.x. Run times measured on a laptop
(Python 3.13, `requirements-lock.txt`): `convergence.py --reps 1500` about 2 min;
`headline.py --reps 2000` under 1 min; `doe_sobol --rung 1 --design 24 --reps 300` about 1 min;
`brigade_table.py --reps 100` about 25 min (the brigade engine runs 1.5–3 s per replication);
the ladder demo in section 8.7 about 4 min; `compare_coas` with defaults a few minutes.
Rungs 3–4 and larger designs scale linearly in replications × points × terrains × vignettes.

**Committed artifacts.** `runs/headline.json`, `runs/convergence.json`,
`runs/sobol_rung1.json`, `runs/brigade_table.json` and the matching `*_manifest.json` files are
committed, and the README and this document cite them. Every other quoted number is marked
**not committed** with the command that regenerates it.

---

## 13. The app

`streamlit run app.py`. The sidebar drives everything; the four tabs read the last run.

### Sidebar

| Section | Control | Effect |
|---|---|---|
| Ground | Offline demo / Real ground | choose synthetic valley or OSM + DEM fetch |
| | place name, area half-width, cell size, Load ground | fetch and rasterize |
| Blue COA A / B | armor (SBF, Maneuver, Unsupported), recon, fires | the two COAs compared, paired by CRN |
| Scenario | bearing, start distance, SBF ring, OP ring, dismount distance, Red radius, danger close, time limit | `ScenarioConfig` |
| Red COA | Early / Late, trigger range | Red's fire discipline |
| Run | replications per COA | 200–5000 |
| | Red plans sampled | 1 = best-scored defense; 2–5 = outer loop over sampled defenses (7.9) |
| | Time step (s) | 3.75 / 7.5 / 15; 7.5 is the converged default (10) |
| | replays kept, seed, Run both COAs | |

### Tabs

- **Ground**: the map with terrain classes, Red positions (AT and infantry, in buildings or
  cover), Blue start line, routes for the current armor choice, SBF positions and OPs, and the
  scenario notes (OSM way counts, fetch time, warnings such as a start line outside the area).
- **Results**: P(win), losses, exchange ratio, decision time for each COA with 95% intervals and
  the paired difference; the CRN diagnostic sentence; "How the fights end" (Aalen–Johansen
  cumulative incidence of Blue-broke vs Red-broke, censoring at the time limit); strength over
  time with pointwise or simultaneous bands (toggle); the run manifest (expander).
- **Where losses happen**: per COA, a heatmap of where Blue units were destroyed with
  empirical-Bayes-shrunk per-cell rates, Red positions sized by how often each was destroyed;
  a quick-pick of which COA or the split view; note when deaths were pooled over several Red plans.
- **Playback**: side-by-side replay of a chosen replication in each COA on the same random
  numbers, with a time slider, play button, unit states (alive, suppressed, mounted, moving,
  seen by the enemy), the event feed for that replication (shooter, target, hit, kill, the
  model's probability of that target choice and of the hit), and a status line.

`tests/test_app.py` drives the app headlessly through a load, a run, the sliders and toggles,
and a multi-plan run.

---

## 14. Tests

`python -m pytest -q` collects **47 tests**. Measured on 2026-09-18 (Python 3.13,
`requirements-lock.txt`, laptop under load):

- `pytest tests/test_sim.py tests/test_stats.py tests/test_calibration.py tests/test_brigade.py`
  → **43 passed in 10 min 55 s**. This is the simulation and statistics suite; it backs every
  verified claim in the README.
- `pytest -q` earlier the same day → **46 passed in 9 min 21 s**, before
  `tests/test_deck_json.py` was added. The four headless Streamlit app tests dominate the
  wall-clock time.

`pytest.ini` sets `pythonpath = .`.

| File | What it checks |
|---|---|
| `tests/test_sim.py` | LOS blocking, hit probability vs brute force, Gumbel-max vs logit probabilities, OSM rasterization on synthetic Overpass JSON, terrain save/load, synthetic descriptor targets, scenario tables, every rung roster builds and runs, stochastic vs deterministic placement, engine determinism and bounds, substream CRN under a mechanic switch, tick rescaling, result concatenation, harmless Red never kills, duel monotone in Red lethality (a regression vignette) |
| `tests/test_stats.py` | Wilson coverage, Kaplan–Meier and Aalen–Johansen against exponential truths, KN selects the best of four Gaussian systems, shrinkage behaviour, sup-t wider than pointwise, sequential stopping, CRN diagnostic, ratio interval coverage |
| `tests/test_calibration.py` | descriptors track terrain, design and train/holdout split, terrain library caching, ParamSpec round trip and apply, vignette metrics, history matching keeps a synthetic truth and shrinks, GP + Sobol recover a known additive function |
| `tests/test_app.py` | headless app: load, run, sliders, segmented controls, simultaneous-band toggle, play; multi-plan run |
| `tests/test_brigade.py` | PK derivation ordering, nation catalogs run in the entity engine, force structure with all functional battalions and scaling, brigade engine determinism and force-ratio monotonicity, obstacles depend on engineers, aviation losses bounded, correction loads, coarse OSM rasterizer, every COA and nation pair runs, terrain wrapping, zero-sum LP and bootstrap |
| `tests/test_app_brigade.py` | headless brigade app: run a pairing, move the playback slider, solve the game |
| (in `test_brigade.py`) | ideal C2 is the zero-latency limit, doctrine runs and logs decision events, Iran carries two lineages and every brigade an HQ, a slowed profile lengthens observed-to-order latency |

Regression policy: when a mechanic changes, `test_duel_monotone_in_red_lethality` and the
rung-roster test guard small-scale behaviour, and the ladder should be rerun on the cached
terrain library to check that fixed parameters do not drift (section 15).

---

## 15. Known limits and findings

**Modelling**

- Parameters are notional. Nothing here is validated against real engagements.
- Red is static: it does not displace, counterattack, or call fires. Its only decisions are
  position (scored, optionally sampled) and fire discipline (Early/Late).
- The **entity** engine has no obstacles, breaching, smoke, drones, C2 delays, damage states
  or resupply. The **brigade** engine (section 19) does model obstacle belts, breaching and
  C2 latency; it still has no smoke, drones, damage states, EW, fixed-wing air or
  air-defence units. Brigade-specific limits are in 19.6.
- OSM multipolygon relations are ignored; complex woods and courtyards rasterize incompletely.
- Every unit is checked against every enemy each tick (O(NB × NR)); fine to a company team,
  not to a battalion. Range-gated engagement lists are needed above that.
- The lane-model notebook found that parameter uncertainty accounted for about 96% of outcome
  variance versus Monte Carlo error (MC interval ±0.02 on P(win), parameter draws spanning
  0.33–0.72). Expect the same here until the ladder has real reference data.

**Statistics and calibration**

- Intervals cover simulation error (and, in coverage reports, NROY parameter uncertainty).
  They do not cover model error.
- The 15 s tick is not converged (section 10). The default is 7.5 s.
- **The grid cell size is not converged at all** (section 10): P(win) differs by 0.14–0.17
  between 25 m and 12.5 m cells at fine ticks, with non-overlapping intervals. Absolute
  output levels are partly a discretization artifact; only paired comparisons at a fixed
  cell size are defensible, and even their stability across cell sizes is untested.
- Rung 2's choice-model parameters (`lam_blue`, `lam_red`, `b_threat`) are not identifiable
  from aggregate outcomes. Identifying them needs choice-level data (who shot whom, when, with
  what alternatives available), fitted by conditional-logit maximum likelihood; that is the
  "structured elicitation" route in the plan, not the outcome-matching route.
- Rung 1 identifies `theta0_at` weakly because ATGM shots in the duel vignette cluster in one
  range band where P_H is near 1 for most of the prior; a longer-range duel vignette would help.
- The error-trend diagnostic needs at least two held-out terrains; the demo used one.
- `select_terrains` with few candidates can leave a cluster empty; the demo with five candidates
  kept four. Use 12+ candidates for real work.
- History matching's NROY box is axis-aligned; correlated parameter constraints (e.g. θ × P_K
  trade-offs) are not captured by the box, only by the sample inside it. `nroy_samples` should
  be replaced by resampling the actual NROY points when the box is far larger than they occupy.

**Parameter drift as a signal.** If a parameter fixed at a lower rung wants to move when a
mechanic is added or the rung above is calibrated, it was compensating for something missing
(λ absorbing C2 effects, breakpoints absorbing suppression). Treat drift as evidence of a
missing mechanic, not as something to re-tune away. The cached terrain library and the run
manifests exist so that the same vignettes can be rerun after every engine change.

---

## 16. Roadmap

In the order the plan set: rigor first, then complexity.

**Rigor, remaining**
- Choice-level data path: log (shooter, chosen target, alternatives, utilities) per shot for
  every replication (not only the replay subset) and add a conditional-logit fitter for λ and
  the utility weights.
- The LP game solution and game-value bootstrap live in `sim/stats.py` and drive the brigade
  COA game (7.10, 19.5); wiring them to the company-team app with the Red-plan outer loop as
  Red's strategy set is still open.
- Cell-size convergence: carry the section 10 sweep to 6.25 m and 3.125 m, or replace
  grid-sampled LOS with an exact vector method.
- Replace box refocusing in history matching with emulator-based implausibility (GP per output)
  so later waves need fewer simulator runs.
- Real-ground terrain library: candidate pool from a list of training-area coordinates,
  descriptors from OSM ground, holdout by descriptor distance.

**Complexity, deferred until rungs 1–3 are stable**
- Obstacles and breaching (minefields, wire, ditches) with breach time and breach-under-fire.
- Smoke (obscuration as a time-varying visible-fraction multiplier).
- Drones as recon (detection from above, ignoring most LOS) and as strike.
- Red agency: displacement to alternate positions, local counterattack, Red fires.
- C2: order delays and decision latency, so the net's effect is not instantaneous.
- Damage states (mobility kill, firepower kill) instead of binary alive/dead.
- Multipolygon OSM relations; seasonal foliage; ditches and embankments from finer DEMs.
- Battalion scale: range-gated engagement lists, echeloned units.

---

## 17. References

- Craig, P. S., Goldstein, M., Seheult, A. H., & Smith, J. A. (1997). Pressure matching for
  hydrocarbon reservoirs: a case study in the use of Bayes linear strategies for large computer
  experiments. *Case Studies in Bayesian Statistics*, 3, 37–93. (history matching)
- Vernon, I., Goldstein, M., & Bower, R. G. (2010). Galaxy formation: a Bayesian uncertainty
  analysis. *Bayesian Analysis*, 5(4), 619–669. (history matching with emulators)
- Kim, S.-H., & Nelson, B. L. (2001). A fully sequential procedure for indifference-zone
  selection in simulation. *ACM TOMACS*, 11(3), 251–273. (KN)
- Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., & Tarantola, S. (2010).
  Variance based sensitivity analysis of model output. *Computer Physics Communications*,
  181(2), 259–270. (Sobol estimators)
- Cioppa, T. M., & Lucas, T. W. (2007). Efficient nearly orthogonal and space-filling Latin
  hypercubes. *Technometrics*, 49(1), 45–55. (NOLH designs)
- Ankenman, B., Nelson, B. L., & Staum, J. (2010). Stochastic kriging for simulation
  metamodeling. *Operations Research*, 58(2), 371–382.
- Law, A. M. (2015). *Simulation Modeling and Analysis*, 5th ed. McGraw-Hill. (sequential
  interval estimation, CRN)
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference.
  *JASA*, 22(158), 209–212.
- Aalen, O. O., & Johansen, S. (1978). An empirical transition matrix for non-homogeneous
  Markov chains based on censored observations. *Scandinavian Journal of Statistics*, 5, 141–150.
- DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical
  Trials*, 7(3), 177–188. (moment estimator used for shrinkage)
- McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In
  *Frontiers in Econometrics*. (random-utility target selection)

---

## 18. Data attribution and licences

- Map data © OpenStreetMap contributors, available under the Open Database Licence (ODbL).
  Fetched through the Overpass API; respect its usage policy (the app identifies itself and
  caches results).
- Elevation: AWS Terrain Tiles (Terrarium), derived from open elevation sources including SRTM,
  ETOPO1, and national datasets; see the Terrain Tiles attribution page.
- Geocoding: Nominatim (OSM Foundation usage policy applies).
- Basemap: © CARTO, © OpenStreetMap contributors.

The code in this repository is provided for educational and analytical use. The model is a
teaching and methodology instrument; do not use its outputs for operational decisions.

---

## 19. Brigade against brigade

### 19.1 Why a second engine

A brigade fight is roughly 100 companies a side, 30 km of ground, and 8 hours. The entity engine
is O(units²) per tick with a ray-cast LOS table per route cell; it would take hours per
replication at that size and the per-vehicle detail would be fiction anyway (the parameters are
notional). The standard answer is a hierarchy of models: an aggregate engine whose company-level
kill rates are derived from the entity engine, plus a check that the two agree where they
overlap. That is what `sim/brigade.py` and `scripts/aggregation_check.py` do.

### 19.2 Nation catalogs and brigade structure (`sim/nations.py`)

Four nations, each filling the same ten classes: `tank`, `ifv`, `inf` (dismount platoon),
`atgm` (AT team), `recon`, `howitzer` (tube battery), `rocket` (MLRS battery), `engineer`
(breach/obstacle vehicles), `support` (sustainment trucks), `attack_helo`. Every class has the
entity engine's attribute schema (area, θ0, θs, θt, sensor, signature, fire rate, suppression
terms, value, speed) plus a **protection tier** (0 dismount … 4 modern MBT), a **warhead tier**
(0 small arms … 4 tank gun), an optional secondary weapon (an IFV's ATGM), range, ammo,
platforms per company, and for artillery a notional kills-per-round lethality and rate of fire.
Nation-level: `quality` (multiplies P_K and fire rate), `c2_delay_min` (fire-mission latency),
`recon_uas` (detection multiplier), `shorad` (organic short-range air-defence density, 0–1).

Every brigade has the same functional elements, filled from the nation's template with its own
weights (a Russian brigade is fires-heavy, a Chinese one has four maneuver battalions, an Iranian
one is tank-heavy and thin on enablers):

| Element | US (ABCT-like) | Russia (MR brigade / BTGs) | China (heavy CA brigade) | Iran (armoured brigade) |
|---|---|---|---|---|
| maneuver battalions | 3 CABs × (2 tank + 2 mech cos) | 3 MR bns × (1 tank + 3 MR cos) as BTGs + 1 tank bn (3 cos) | 4 CA bns × (2 tank + 2 mech cos) | 2 tank bns (3 cos) + 1 mech bn (3 cos) |
| anti-tank battalion | — | 3 Kornet companies | — | 1 Toophan company |
| cavalry / reconnaissance | squadron: 3 troops (scouts + 4 Bradleys) | recon bn: 2 troops (+ UAV) | recon bn: 3 troops | 2 troops |
| fires | 3 × 6 M109A7 | 6 tube + 3 rocket batteries (two howitzer bns + a rocket bn) | 3 tube + 2 rocket batteries | 3 tube + 2 rocket batteries |
| engineer battalion | 2 cos × 14 breachers | 2 × 10 IMR/UR-77 | 2 × 12 | 2 × 10 |
| support battalion | 3 cos × 20 trucks | 3 × 20 | 3 × 20 | 3 × 16 |
| attack aviation enabler | 2 × AH-64E, 16 missiles/sortie | 2 × Ka-52, 12 | 2 × Z-10, 12 | 2 × AH-1J, 8 |
| SHORAD factor | 0.25 | 0.70 | 0.70 | 0.40 |
| tank θ0 (mrad) / protection / sensor | 0.22 / 4 / 1.00 | 0.32 / 3 / 0.75 | 0.28 / 3 / 0.85 | 0.45 / 3 / 0.55 |
| quality / readiness / C2 delay (min) | 1.00 / 0.95 / 4 | 0.85 / 0.85 / 8 | 0.90 / 0.90 / 6 | 0.70 / 0.75 / 12 |
| platforms (maneuver) / companies | 325 (204) / 24 | 310 (186) / 35 | 319 (201) / 30 | 176 (94) / 23 |

Templates are lists of battalion compositions (`battalions=[dict(tank=1, ifv=3)] * 3 +
[dict(tank=3, ifv=0)]`), so any structure can be expressed. HIMARS is a division asset and is
left out of the US brigade; add `rocket_batteries=1` to attach one.

P(kill | hit) is **derived**, not typed in: `CLASS_PK[warhead, protection] × quality`, taking the
better of the main and secondary weapon against each target. `entity_params(blue, red)` maps any
nation pair onto the entity engine's six slots so the calibration ladder can run any pairing.

**Every number is synthetic.** They encode open-source qualitative characterizations and
nothing else; the app labels them as such.

### 19.3 The aggregate engine (`sim/brigade.py`)

The mean-field derivation that ties this engine to the entity engine is Appendix A.19; the correction fit is A.20.

- **Ground.** `synthetic_brigade_terrain(half=15000, cell=300, relief, cover, urban, river)`: a
  `BrigadeTerrain` with bare-ground elevation for LOS, continuous cover and urban fractions that
  reduce exposure (`exposure = 1 − 0.6·cover − 0.2·urban`), a river with four bridges, and roads.
  `brigade_terrain_from(terrain)` wraps a fine entity-engine terrain for the aggregation check.
  LOS is cast on the ground with `los_fraction` and cached per cell pair in the terrain object, so
  the cache is shared across replications and COAs.
- **Entities.** `build_force(nation, side, copies, strength)` turns a template into companies:
  a vector of platform counts by class, battalion id, and role (maneuver / recon / battery).
  `copies` gives multi-brigade attacks; `strength` scales counts (a brigade at 70%).
- **Plans.** Attacker schemes: `two_up` (two battalions abreast, reserve committed at 2 h or when
  a lead battalion breaks), `penetration` (column on one axis), `envelopment` (one battalion
  fixes short of the line, two swing north). Defender schemes: `forward` (battalions on line at
  +2 km), `depth` (two forward, the rest on a second line at +5.5 km), `mobile` (two forward, a
  reserve that counterattacks toward the most advanced attacker when a forward battalion falls
  under 60%). Recon screens ahead of both; batteries sit on gun lines. Objective at (6.5 km, 0).
- **Detection.** Side-level network: each enemy company's hazard per minute sums, over own
  companies with LOS, sensing strength × signature × visible fraction × (1000/D)², times the
  nation's UAS factor and a firing bonus, plus a small wide-area trickle; detections persist five
  minutes.
- **Direct fire.** For each (shooter company, class a, target company, class b): P_H from the
  circular-normal kernel on the visible fraction, P_K from the derived matrix, range gating, and an
  **acquisition fraction** `1 − exp(−4·hazard_ab)` from the entity detection hazard so fire is
  acquisition-limited at long range exactly as it is in the entity engine. Shots per minute are
  `4 × fire_rate × quality × count × (1 − 0.7·suppression)`; each company allocates them across
  (target, class) in proportion to expected value (the λ = 1 logit). Expected kills are drawn by
  dithered rounding (`floor(E + U)`, unbiased), never exceeding the platforms present. Dug-in
  defenders get posture exposure 0.6 while stationary.
- **Indirect fire.** A battery with a detected target in range starts a three-minute mission with
  probability `1/c2_delay` per minute, then waits `c2_delay` minutes (displace, re-target).
  Kills per class = rounds × lethality × class vulnerability (`IND_VULN`: dismounts 1.0, IFV
  0.35, tank 0.12) × cover reduction; the target is suppressed. Batteries that fired are easier
  to detect for five minutes, so counter-battery fire emerges from the same rules.
- **Engineers, obstacles, breaching.** The defender's engineer battalion lays obstacle belts
  (minefield + wire) 700 m in front of each occupied line, one 300 m cell per 0.7 engineer
  vehicle, split across the battalion sectors. An attacking company that reaches an unbreached
  obstacle cell halts, takes mine losses (Poisson, 4% of its vehicles), and breaches in 12 minutes
  if an attacker engineer company is alive, unsuppressed and within 1.5 km, else 40 minutes; while
  halted its exposure is ×1.3. A breached cell becomes a lane for everyone behind. Attacker
  engineer companies travel 1.2 km behind the lead battalions.
- **Support battalion and ammunition.** Each side has an indirect-fire stock of 100 rounds per
  tube; missions draw it down; the support battalion refills it at 20 rounds per tube per hour
  scaled by the fraction of support platforms still alive. Support companies sit in the rear as
  soft, high-value targets for artillery and aviation, so interdicting sustainment shows up as a
  fires battalion that goes quiet.
- **Cavalry.** Recon troops carry both scouts (high sensor, low signature) and a few vehicles
  (`cav_ifv_per_troop`), so they screen, fight for information, and can be killed.
- **Attack aviation.** A two-ship team flies 20-minute sorties with a 60-minute turnaround once
  the side has detected an enemy concentration: it takes station 6 km toward its own side from the
  centroid of detected targets, engages detected companies within its missile range (guided
  missile, P_H ≈ 0.4 per shot at standoff, P_K from warhead tier 3 vs the target's protection),
  prefers high-value targets, and suppresses what it hits. Each minute on station each
  helicopter is lost with probability `0.012 × enemy shorad × min(1.5, enemy maneuver companies
  within 8 km / 5)`. Helicopters are not targets for ground direct fire or artillery.
- **Breakpoints.** Company: Beta(8, 8) on fraction of maneuver platforms lost; a battalion with
  half its companies broken is broken. Brigade: attacker culminates when effective maneuver
  strength (unbroken companies) falls below Beta(10, 10); defender breaks below Beta(11, 9).
  Outcome: `cause` 2 (objective seized: three unbroken attacker maneuver companies within 1.5 km
  and no unbroken defender company there, or the defender broke), 1 (attacker culminated), 0
  (censored at 8 h).
- **Kill draws.** Direct-fire, artillery and aviation kills are Poisson-thinned expected counts,
  capped at the platforms present (`kill_draw="poisson"`; `"dither"` restores the lower-variance
  rounding draw). The direct-fire expectation is multiplied by the rung-5 correction (19.7).
- **CRN.** Thirteen named substreams; draw shapes are fixed by the force structure, so all nine
  COA pairs of a nation matchup share random numbers (Poisson draws weaken the pairing slightly
  compared with uniform draws).
- **Output** `BrigadeResult`: win, cause, minutes, losses by class per side, maneuver strength
  per minute per side, initial counts, aviation sorties, rounds left, the obstacle grid, position
  tracks for logged replications.

Runtime: 0.3–2 s per replication on a laptop depending on how long the fight runs.

### 19.4 What it shows

Synthetic ground (seed 1), full brigade combat teams, corrected kill rates (19.7), 20 replications:

| Pairing (two-up vs forward) | P(objective seized) | attacker / defender maneuver losses | minutes |
|---|---|---|---|
| US ×1 → RU | 0.00 | 0.26 / 0.15 | 66 |
| US ×2 → RU | 0.70 | 0.19 / 0.28 | 85 |
| RU ×1 → US | 0.05 | 0.28 / 0.14 | 135 |
| RU ×2 → US | 0.80 | 0.22 / 0.33 | 175 |
| CN ×1 → IR | 0.55 | 0.16 / 0.32 | 101 |
| IR ×2 → CN | 0.00, 55% censored | 0.32 / 0.19 | 411 |

Force ratio dominates at equal quality. Enabler effects are visible in the class-level losses:
Iran's Cobra pair does not survive eight hours against Chinese air defence (1.95 of 2 lost per
fight); Russian obstacle belts (36 cells from 20 engineer vehicles at 85% readiness) cost the US
attack roughly 15 minutes and 4% of the vehicles that hit them; the Russian defender against
US ×2 ends with 470 rounds of a 5400-round stock after the support battalion is hit. The scheme
game only discriminates schemes near even odds; at 2:1 the cells saturate.

### 19.5 App (`app_brigade.py`)

Sidebar: attacker and defender nation, brigades per side, strength per side, schemes, synthetic
ground knobs, replications, seed. "Run this pairing" gives P(objective seized) with a Wilson
interval, loss fractions, decision time, losses by class, the Aalen–Johansen incidence of
culmination vs seizure, maneuver strength bands, and a minute-by-minute position map of the
first logged replication. "Solve the 3×3 COA game" runs all nine scheme pairs on shared random
numbers and shows the payoff heatmap, the LP value with its bootstrap interval, and both sides'
minimax mixes. An expander shows the notional catalogs in play.

### 19.6 Limits specific to the brigade level

- Defender battalions are static except the mobile reserve; no withdrawal to depth positions,
  no counter-mobility beyond the initial belts, no EW, no fixed-wing air, no air-defence
  attrition (SHORAD is a factor, not a unit).
- Detection is a side-level network, not per company; the acquisition fraction stands in for
  per-pair detection.
- Breakpoint parameters at company, battalion and brigade level are notional and untested
  against anything; so are the obstacle, breach, ammunition and aviation constants at the top of
  `sim/brigade.py`.
- The rung-5 correction (19.7) was fitted on company fights and is applied to brigade fights;
  that is an extrapolation with no check above company scale.

### 19.7 Rung 5: the aggregation correction (`scripts/rung5_calibrate.py`)

The aggregate engine is calibrated against the entity engine the same way the entity engine is
meant to be calibrated against data: run the same company fight in both (one company team vs
one AT platoon, three synthetic terrains, US→RU and CN→IR), then fit two multipliers on the
aggregate direct-fire kill rates by alternating bisection so the mean loss fractions match.
P(win) and duration are not fitted; they are reported as residuals.

Result (`data/aggregation_correction.json`, committed, loaded by `BrigadeSim` by default). The
corrected column is read from that file; the raw column is **not committed** — regenerate it with
`python scripts/aggregation_check.py --raw`:

| | raw aggregate − entity | corrected − entity |
|---|---|---|
| P(win) | +0.41 | +0.048 |
| attacker loss fraction | −0.22 | +0.015 |
| defender loss fraction | +0.23 | +0.010 |
| minutes | +1.1 | +0.91 |
| kill-rate multipliers | (1, 1) | attacker→defender **0.63**, defender→attacker **2.54** |

Two things the fit taught: the first attempt split the entity company team into three aggregate
companies with their own breakpoints, and no multiplier could close the attacker-loss gap (it
stuck at −0.16 at 4×) because the attacker culminated on company breakpoints at 25% actual
losses; one entity per side fixed that. And the uncorrected aggregate over-favours the attacker
by 0.41 in P(win) at company scale, which is why brigade results in 19.4 differ from the earlier
uncorrected runs. `scripts/aggregation_check.py --raw` shows the uncorrected gap;
without `--raw` it shows the residual.

### 19.8 Real ground at brigade scale

`brigade.osm_brigade_terrain(lat, lon, half=15000, cell=300)` fetches AWS Terrain Tiles at zoom
11 and a coarse OSM query (wood/scrub/forest, residential/industrial/commercial landuse, water
and rivers, motorway–tertiary roads), rasterizes on a 100 m temporary grid, and block-averages
into cover, urban, water and road fractions per 300 m cell (`rasterize_coarse`). Residential and
industrial landuse stands in for buildings, which are not fetched at this scale. Roads crossing
water become bridges. The parser is tested offline on synthetic Overpass JSON; the network path
is untested in this sandbox, like the entity-scale fetch.

### 19.9 Command and control, decision graphs, and deconfliction (`sim/c2.py`)

Until this layer existed every unit executed its template perfectly and instantly. Now every
decision passes through four steps that differ by nation, and the differences are ranges to be
tested, not facts.

**Five C2 variables per battalion lineage** (`C2Profile`):

| | τ latency (min) | p_init initiative | latitude | net reliability | sync |
|---|---|---|---|---|---|
| US mission command | 5 | 0.80 | 0.80 | 0.90 | 0.85 |
| Russia directive control | 18 | 0.25 | 0.25 | 0.70 | 0.75 |
| China phase-locked systems C2 | 9 | 0.40 | 0.40 | 0.85 | 0.85 |
| Iran, Artesh | 20 | 0.30 | 0.30 | 0.55 | 0.55 |
| Iran, IRGC | 5 | 0.85 | 0.90 | 0.50 | 0.35 |

Meanings: a trigger is **observed** by the brigade with probability `net` per minute (halved if
the HQ company is destroyed); the order arrives after a lognormal **latency** with mean τ
(doubled without the HQ); meanwhile the subordinate that sees the trigger itself may **act on
intent** each minute with hazard `p_init/τ`; the action is bounded by **latitude** (bypass an
obstacle, exploit past the objective) and taxed by **sync** (clearance of fires across a
boundary succeeds at once with probability sync, else waits about τ/3; friendly fire out of
sector scales with 1 − sync). Iran fields two lineages: the last maneuver battalion is IRGC,
the rest Artesh, and coordination across the two multiplies sync by 0.5. `c2="ideal"` sets
τ = 0, p_init = latitude = net = sync = 1 and reproduces the static templates.

**Decision graphs** (`GRAPHS`): each nation's plan answers the same triggers differently.

| Trigger | US | Russia | China | Artesh / IRGC |
|---|---|---|---|---|
| lead battalion under 60% or broken, or 2 h elapsed → commit reserve | to the axis making the most progress | to the axis that stalled | on the pre-planned axis | stalled axis / most progress |
| objective seized (sequel) | exploit 30 min toward the depth objective | consolidate | consolidate | consolidate / exploit |
| obstacle contact | bypass through the nearest gap if latitude allows, else breach | breach | breach | breach / bypass |
| fires stock below threshold | pause 20 min at 15% | 10% | 15% | 10% / 5% |
| ≥ 4 enemy maneuver companies detected in one sector | shift priority of fires (×2) | shift | shift | no shift |
| defender (mobile scheme): forward battalion under 60% | counterattack toward the most advanced attacker, through the same observe–latency–initiative chain | | | |

Sectors, clearance and crowding: each maneuver battalion owns a y-band around its axis. A fire
mission on a target inside another battalion's sector needs that battalion's clearance; more
than two companies in a cell halves movement and raises artillery lethality by 30% per extra
company; a company out of its sector and in contact with friendlies nearby takes friendly-fire
losses at 0.003 per platform per minute × (1 − sync). Every decision, bypass and fratricide
event is logged for the replay replications (`BrigadeResult.c2_events`).

**The sensitivity study** (`scripts/c2_sensitivity.py`) is the guard against narrative
parameters: Latin hypercube over the five factors of one side across `C2_RANGES` (τ 2–30 min,
p_init 0.1–0.95, latitude 0.1–1, net 0.4–1, sync 0.3–1), GP with the Monte Carlo noise floor,
Saltelli/Jansen indices on P(objective seized). Report C2 claims as "τ moves P(win) by X over
its range" rather than "Russia loses because of its NCOs." A demonstration run (US attacker vs RU,
16 design points × 12 replications, about 10 minutes; **not committed** — regenerate with
`python scripts/c2_sensitivity.py --attacker US --defender RU`) spanned P(win) 0.33–0.75 across
the design but its metamodel had LOO R² below zero with a Monte Carlo noise share of 1.0, so the
tool refused to certify its indices (the raw ranking put `sync` first, `tau` second). That is
the correct behaviour: a study that can support a claim about C2 needs roughly 40 points ×
100 replications (about 3 hours at current speed). Do not quote the demonstration indices.

Runtime rose to 1.5–3 s per replication with this layer (per-company Python loops for sectors
and fratricide); vectorizing them is the first performance item.

