# kriegmate — adversarial Monte Carlo COA analysis

**Which course of action is better, by how much, and how confident can we be?** This repository
answers that question the way an operations-research shop has to answer it: with a stochastic
combat simulation on real ground, paired comparisons on common random numbers, ranking-and-selection
with a stated error rate, variance-based sensitivity analysis, and a rung-by-rung calibration
procedure with an explicit path from notional parameters to fitted ones.

**Status: verified, not validated.** The estimators, the mechanics and the calibration machinery
are unit-tested and reproduce known answers. The combat model itself is *notional*: every
parameter, nation catalog and command-and-control profile is invented or derived from open
sources, and nothing here has been fitted to, or checked against, a real engagement or a
higher-fidelity reference model. The one calibration that has been run (rung 5) fits one of this
repository's engines against the other. **Do not use any number here operationally.** The
contribution on offer is the analysis method, not the answers.

The repository is new (first published 2026-09-17) and under active development.

---

## The worked result

Two Blue courses of action for a company-team attack on a prepared defense, on the offline demo
ground, run on **the same random numbers** so the comparison is paired:

| Blue COA | n | P(win) [95% Wilson] | Blue losses [95%] | minutes |
|---|---|---|---|---|
| **SBF** — armor supports by fire | 2000 | **0.678** [0.658, 0.699] | 3.18 [3.09, 3.27] | 4.5 |
| **Maneuver** — armor moves with the infantry | 2000 | **0.649** [0.628, 0.670] | 3.42 [3.34, 3.50] | 4.5 |

**Paired difference, P(win) SBF − Maneuver: +0.029, 95% CI [+0.009, +0.050].** SBF is better on
this ground and this scenario, but by about three points of win probability, not by a landslide —
and the interval only just excludes zero at n = 2000 per COA.

**Common random numbers earned their keep:** the achieved correlation between the two COAs was
ρ = 0.50, and the paired variance of the difference was 0.499 of the independent variance. CRN
halved the variance, which is the same as doubling the replication budget.

Reproduce: `python scripts/headline.py --reps 2000`
(seed 1, tick 7.5 s, recon and fires on, Red "Late"). Committed artifacts:
[`runs/headline.json`](runs/headline.json) and
[`runs/headline_manifest.json`](runs/headline_manifest.json) (git SHA, parameter hash, terrain
hash, scenario config, library versions).

---

## The model at a glance

The entity engine is a tick-based stochastic **kill chain**. It is not a Lanchester model; there
are no Lanchester equations anywhere in this repository.

```
        ┌─ exposure ──────┐   terrain class, concealment, mounted/dismounted
        │                 v
 move ──┼─> DETECT ──> CHOOSE ──> HIT ──> KILL ──> SUPPRESS ──> BREAK
        │   hazard      target     P_H      P_K      state        Beta
        │   ~ visible   random-   circular-  by     for a few   breakpoint
        │   fraction    utility   normal    type    ticks       per element
        │               logit     aim error
        └─ line of sight over a DEM + OSM raster ─────────────────┘
```

Ground comes from OpenStreetMap ways and AWS Terrain Tiles, or from a synthetic generator for
offline work. Line of sight is sampled over the elevation grid with terrain-class height
contributions. Every random number is drawn from a **named substream**, so two COAs can be run on
identical draws for every mechanic they share.

The **aggregate brigade engine** (`sim/brigade.py`) fights brigades of four notional nations on a
30 km box. It is derived as a **mean-field (Poisson-thinned) limit of the same kill chain**
(see [Appendix A.19](docs/appendix-a-methodology.md)), then bias-corrected against the entity
engine (rung 5, below). It is not Lanchester either.

---

## What is verified and what is not validated

| Claim | Evidence | Status |
|---|---|---|
| Hit probability is the circular-normal (Rayleigh) probability of hitting a target of area *A* | `test_hit_prob_matches_brute_force` | **verified** |
| Per-tick probabilities rescale correctly as 1 − (1 − p)^f | `test_tick_rescaling_keeps_units_sane` | **verified** |
| Wilson intervals hold their nominal coverage | `test_wilson_inside_unit_interval_and_covers` | **verified** |
| Kim–Nelson selects the best of four known Gaussian systems | `test_kn_selects_best_system` | **verified** |
| Kaplan–Meier and Aalen–Johansen recover exponential truths under censoring | `test_kaplan_meier_exponential`, `test_aalen_johansen_competing_exponentials` | **verified** |
| GP metamodel + Sobol recover a known additive function | `test_metamodel_and_sobol_on_known_function` | **verified** |
| History matching retains a hidden synthetic truth and shrinks the NROY volume | `test_history_match_keeps_truth_and_shrinks` | **verified** |
| Common random numbers reduce variance in a real paired comparison | `runs/headline.json` (ρ = 0.50, variance ratio 0.499) | **verified** |
| Discretization: 15 s ticks are not converged; 7.5 s and 3.75 s agree at 25 m cells | `runs/convergence.json` | **verified** |
| The aggregate engine, corrected, tracks the entity engine to ≈0.01 in loss fractions | `data/aggregation_correction.json` | **verified against the entity engine only** |
| Any parameter value, nation catalog, C2 profile, or resulting P(win) | — | **not validated — notional** |
| Any statement about a real weapon system, unit, or army | — | **out of scope — do not quote** |

`pytest tests/test_sim.py tests/test_stats.py tests/test_calibration.py tests/test_brigade.py`
→ **43 passed in 10 min 55 s** (2026-09-18, Python 3.13, `requirements-lock.txt`). That is the
suite behind every claim in the table above: the engine, the estimators, the calibration
machinery and the brigade engine. The full `pytest -q` adds four headless Streamlit app tests
(47 collected) and ran 46 passed / 0 failed in 9 min 21 s earlier the same day, before the
pydeck-serialization regression test was added.

---

## Verification evidence

### Discretization: tick and cell convergence

`python scripts/convergence.py --reps 1500` (seed 7, SBF, recon and fires, Red "Late", offline
demo ground). Regenerated at commit `325b74d`; committed as
[`runs/convergence.json`](runs/convergence.json) with
[`runs/convergence_manifest.json`](runs/convergence_manifest.json).

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

**The tick is converged at 7.5 s; the cell size is not converged at all.**

- *Tick.* At 25 m cells, P(win) moves 0.13 from 15 s to 7.5 s and only 0.004 from 7.5 s to 3.75 s
  (intervals overlap). The mechanism is simultaneity: within a tick every live unit fires,
  including units that die that tick, so longer ticks let more of Red's first volley land.
  `TICK_DEFAULT = 7.5` as a result, and 15 s runs are fast previews only.
  At 12.5 m cells the tick has *not* settled: 0.529 → 0.567 from 7.5 s to 3.75 s, intervals
  touching. A finer grid needs a finer tick.
- *Cell size.* P(win) differs by **0.169** at 7.5 s (0.698 vs 0.529) and **0.135** at 3.75 s
  (0.702 vs 0.567) between 25 m and 12.5 m cells, with non-overlapping intervals at every tick
  length. Halving the cell also costs Blue 0.6–0.7 of a vehicle at those ticks. This is the **largest single
  numerical artifact in the model** — larger than the SBF-vs-Maneuver effect the simulation is
  built to measure, and larger than the tick effect at the converged tick. The mechanism is line-of-sight
  sampling: a finer grid resolves small terrain features that break sight lines, so Blue is
  detected and engaged differently.

  **Consequence for every result in this repository: absolute probabilities are cell-size
  artifacts as much as they are model outputs, and must not be read as predictions.** Paired
  comparisons at a fixed cell size (like the worked result above) are the only outputs with a
  defensible interpretation, and even they have not been shown to be cell-size-stable. A
  cell-size convergence study on the demo ground — carrying the sweep down to 6.25 m and 3.125 m,
  or replacing grid-sampled LOS with an exact vector method — is the top open verification item.

> **Correction.** Through 2026-09-17 this README reported the 12.5 m rows as
> 0.261 / 0.487 / 0.621 / 0.655 and concluded that "cell size matters less (about 0.05 in P(win)
> between 25 m and 12.5 m at fine ticks)". Those numbers do not reproduce from
> `scripts/convergence.py` at the stated seed, and the conclusion drawn from them was wrong in
> both magnitude (0.14–0.17, not 0.05) and direction of consequence (cell size is the dominant
> artifact, not a minor one). The table above is what the committed script actually produces; the
> superseded numbers have no surviving provenance and have been deleted.

### Sensitivity: design, metamodel, Sobol indices

`python scripts/doe_sobol.py --rung 1 --design 24 --reps 300 --out runs/sobol_rung1.json`
(24-point maximin Latin hypercube, 300 replications per point, anisotropic-RBF GP with a
Monte Carlo noise floor, Saltelli sampling with Jansen's estimators). Committed as
[`runs/sobol_rung1.json`](runs/sobol_rung1.json).

| output | LOO R² | MC noise share | top total-effect index | second |
|---|---|---|---|---|
| P(win) | 0.86 | 0.04 | `pk_at_tank` 0.79 ± 0.02 | `theta0_at` 0.14 |
| Blue loss fraction | 0.86 | 0.04 | `pk_at_tank` 0.79 ± 0.02 | `theta0_at` 0.14 |
| Red loss fraction | 0.86 | 0.04 | `pk_at_tank` 0.74 ± 0.02 | `theta0_at` 0.17 |
| minutes to decision | 0.75 | 0.05 | `pk_at_tank` 0.61 ± 0.02 | `theta0_at` 0.28 |

The **trust rule is enforced in code**: `doe_sobol.py` prints
`(metamodel too noisy: indices are indicative only)` next to any output whose LOO R² is below
0.7, so an untrustworthy index cannot be quoted without the warning attached. At rung 2 with 200
replications the noise share rises to ≈0.7 and LOO R² falls near zero, and every output is
flagged. That refusal to certify is the point of the machinery.
(Rung-2 figures **not committed**; regenerate with `python scripts/doe_sobol.py --rung 2 --reps 200`.)

Reading: one parameter — the AT team's probability of killing a tank — carries most of the
variance in every outcome at this rung. If this model were ever to be calibrated against data,
that is the parameter the data has to pin down first.

### Rung 5: correcting the aggregate engine against the entity engine

`python scripts/rung5_calibrate.py` fits two multipliers on the aggregate direct-fire kill rates
so that mean loss fractions match the entity engine on the same company fight (three synthetic
terrains, US→RU and CN→IR, 200 replications). P(win) and duration are not fitted and are reported
as residuals. The shipped fit is committed as
[`data/aggregation_correction.json`](data/aggregation_correction.json):

| | corrected − entity |
|---|---|
| P(win) | +0.048 |
| attacker loss fraction | +0.015 |
| defender loss fraction | +0.010 |
| minutes | +0.91 |
| kill-rate multipliers | attacker→defender **0.628**, defender→attacker **2.537** |

The uncorrected gap (P(win) +0.41, loss fractions −0.22 / +0.23) is regenerated with
`python scripts/aggregation_check.py --raw`; **that output is not committed.** Note what this
calibration is and is not: it makes one of this repository's engines agree with the other. It is
not validation, and it is fitted at company scale and applied at brigade scale.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.10+; results produced on 3.13
pip install -r requirements-lock.txt                  # exact versions; requirements.txt has bounds
python -m pytest -q                                   # 46 tests

streamlit run app.py                                  # company team, real or synthetic ground
streamlit run app_brigade.py                          # brigade vs brigade (US / CN / RU / IR)

python scripts/headline.py                            # the worked result above
python scripts/convergence.py --reps 1500             # the convergence table above
python scripts/doe_sobol.py --rung 1 --design 24 --reps 300   # the Sobol table above
python scripts/brigade_table.py --reps 100            # the brigade pairing table
python scripts/compare_coas.py                        # Kim-Nelson selection over 8 Blue COAs
python scripts/calibrate_ladder.py                    # history matching, rungs 1-2
```

The apps work fully offline on synthetic ground. Real ground needs network access to Nominatim,
Overpass and AWS Terrain Tiles; if those are blocked the app says so and falls back offline.

Twenty lines of library use:

```python
from sim.terrain import demo_terrain
from sim.scenario import Scenario, ScenarioConfig
from sim.engine import simulate
from sim.stats import wilson, paired_ci, crn_gain

t = demo_terrain()
sc = Scenario(t, ScenarioConfig())
a = simulate(sc.tables("SBF"), "SBF", recon=True, fires=True, red_coa="Late", n=2000, seed=[1])
b = simulate(sc.tables("Maneuver"), "Maneuver", True, True, "Late", 2000, seed=[1])  # same seed
print(wilson(a.win.sum(), a.n), paired_ci(a.win, b.win), crn_gain(a.win, b.win))
```

---

## The analysis toolkit

Every method below is implemented, unit-tested, and derived in
[Appendix A](docs/appendix-a-methodology.md).

| Method | Where | What it buys |
|---|---|---|
| Wilson, delta-method and sup-t intervals | `sim/stats.py` | every rate and curve carries uncertainty |
| Common random numbers, named substreams, CRN diagnostic | `sim/engine.py`, `stats.crn_gain` | paired COA comparison at half the variance |
| Kim–Nelson ranking and selection | `stats.kn_select` | "COA *k* is best" with a stated error rate and indifference zone |
| Kaplan–Meier and Aalen–Johansen | `sim/stats.py` | fights that hit the time limit are censored, not dropped |
| Sequential sampling to a half-width | `stats.run_until` | stop when the answer is precise enough, not at a round number |
| Empirical-Bayes shrinkage | `stats.shrink_rates` | loss heatmaps that are not driven by one-replication cells |
| Latin hypercube design + stochastic-kriging GP | `sim/doe.py` | a metamodel that fits signal, not Monte Carlo noise |
| Saltelli/Jansen Sobol indices with a LOO R² > 0.7 trust rule | `doe.sobol_indices` | which parameter to measure first — and a refusal when the data cannot say |
| History matching: implausibility, NROY waves, held-out terrain coverage | `sim/calibrate.py` | a calibration procedure that reports what it cannot identify |
| Zero-sum LP game value with a bootstrap interval | `stats.solve_zero_sum`, `bootstrap_game_value` | a scheme-vs-scheme game value with an interval, not one COA pair |
| Red-plan outer loop over sampled defenses | `sim/scenario.py`, `scripts/compare_coas.py` | selection that is not conditional on one Red plan |
| Run manifests: git SHA, parameter hash, terrain hash, library versions | `sim/manifest.py` | any table can be traced to the code that made it |

---

## Limitations

**The model.**

- Every parameter is notional. Nothing is validated against a real engagement, a field trial, or
  a higher-fidelity model. The calibration ladder in
  [technical reference §8](docs/technical-reference.md#8-calibration-ladder) is the procedure for
  fixing this; no rung has ever been run against real reference data.
- **Grid cell size is not converged** (above): halving the cell moves P(win) by 0.14–0.17 on the
  demo ground. Absolute output levels are partly a discretization artifact.
- In the entity engine, Red is static: it does not displace, counterattack, or call its own fires.
  Its only decisions are position and fire discipline (Early/Late).
- The entity engine has no obstacles, breaching, smoke, drones, damage states or resupply. The
  brigade engine adds obstacles, breaching and C2 latency, but has no EW, no fixed-wing air and
  no air-defence units (SHORAD is a lethality factor, not a unit). See
  [§15](docs/technical-reference.md#15-known-limits-and-findings) and
  [§19.6](docs/technical-reference.md#196-limits-specific-to-the-brigade-level).
- Every unit is checked against every enemy each tick, O(N_B × N_R); fine for a company team, not
  for a battalion.
- OSM multipolygon relations are ignored, so complex woods and courtyards rasterize incompletely.

**The statistics.**

- Intervals cover **Monte Carlo error only**. They do not cover model error, and they do not
  cover parameter uncertainty. In the predecessor lane-model notebook, parameter uncertainty
  accounted for about 96% of outcome variance against a ±0.02 Monte Carlo half-width; expect the
  same here. More replications do not touch that term — only calibration does.
  (Source: `notebooks/adversarial_combined_arms_mc.ipynb`; **not re-executed** — regenerate before
  quoting.)
- Rung 2's choice-model parameters (`lam_blue`, `lam_red`, `b_threat`) are **not identifiable**
  from aggregate outcomes. Identifying them needs choice-level data (who shot whom, when, with
  what alternatives available).
- The history-matching NROY box is axis-aligned, so correlated parameter constraints are captured
  only by the samples inside it, not by the box.
- Brigade results rest on 100 replications per pairing; the Wilson intervals are wide
  (roughly ±0.10 on a rate near 0.5) and the table separates only large effects.

**Public release.** The adversary-nation force templates and command-and-control profiles in
`sim/nations.py` and `sim/c2.py` are invented for methodological demonstration and labelled as
such throughout. They are not intelligence products and carry no claim about any real army.

---

## Roadmap

**Verification, first.**

1. Carry the cell-size sweep down to 6.25 m and 3.125 m, or replace grid-sampled LOS with an
   exact vector method, and settle whether the grid is the dominant artifact it currently looks
   like.
2. Re-run the paired COA comparison at two cell sizes to see whether the *ranking* of COAs is
   stable even though the levels are not.
3. Commit a result artifact and manifest for every remaining quoted number (the ladder demo, the
   raw aggregation gap, the C2 study).

**Validation: the first rung against real data.** Nothing here is validated, and the cheapest
credible first step is rung 1 against published probability-of-hit versus range curves for a
generic direct-fire weapon. Until a rung is fitted to something outside this repository, the
project is a methods demonstration.

**Complexity, deferred until the rungs are stable.** Smoke, drones, Red agency, damage states,
choice-level logging with a conditional-logit fitter for the target-choice parameters,
emulator-based implausibility in history matching, and battalion-scale range-gated engagement
lists. The full list is in
[technical reference §16](docs/technical-reference.md#16-roadmap).

---

## Documentation

- **[docs/technical-reference.md](docs/technical-reference.md)** — the full specification:
  repository layout, terrain, scenario, engine mechanics, parameters, statistics, the calibration
  ladder, sensitivity, reproducibility, scripts, apps, tests, limits, roadmap, references, and the
  brigade engine.
- **[docs/appendix-a-methodology.md](docs/appendix-a-methodology.md)** — derivations for every
  mechanic and estimator, keyed to the code.

## Attribution and licence

Code under the [Apache License 2.0](LICENSE). Map data © OpenStreetMap contributors (ODbL),
fetched through the Overpass API. Elevation from AWS Terrain Tiles (Terrarium), derived from open
sources including SRTM and ETOPO1.

Author: Jonathan Day. Personal research project; the views and the model are the author's own and
represent no organization. **Not for operational use.**
