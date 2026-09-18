# Appendix A: mathematical methodology

Derivations for every mechanic and estimator, keyed to the code. The model specification is in
the [technical reference](technical-reference.md); the project overview is in the
[README](../README.md).

Each dropdown states the model or estimator precisely, gives the derivation or the argument
for why it is the right tool, and names where it lives in the code. Notation: $n$ replications,
a tick of length $\Delta$ seconds ($\Delta_0 = 15$ s is the reference tick, $f = \Delta/\Delta_0$),
$D$ range in metres, $v \in [0,1]$ visible fraction.

<details>
<summary><b>A.0 The estimand, replications, and what an interval covers</b></summary>

A course-of-action pair $(c_B, c_R)$, a scenario $s$ (ground, placement, geometry) and a
parameter vector $\theta$ define a distribution over outcomes $Y$ (win indicator, losses,
decision time). One replication is a draw $Y_r \sim P(Y \mid c_B, c_R, s, \theta)$, produced by
the engine from an i.i.d. stream of uniforms. All estimates are functionals of the empirical
distribution of $Y_1,\dots,Y_n$, so the intervals in the app cover **Monte Carlo error** only:
they say how well $\bar Y$ estimates $E[Y\mid c_B,c_R,s,\theta]$, not how well that expectation
describes any real fight.

By the law of total variance, if $\theta$ is itself uncertain with distribution $\pi$,
$$\operatorname{Var}(Y)=E_\pi[\operatorname{Var}(Y\mid\theta)]+\operatorname{Var}_\pi(E[Y\mid\theta]).$$
The nested experiment in the notebook estimated the second term at about 96% of the total
(MC half-width $\pm0.02$ on $P(\text{win})$ against a spread of 0.33–0.72 across parameter
draws). This is why calibration precedes complexity: the second term is not reduced by more
replications, only by narrowing $\pi$.
</details>

<details>
<summary><b>A.1 Line of sight and the visible fraction</b> (<code>terrain.los_fraction</code>)</summary>

For an observer at $\mathbf{o}$ with eye height $h_o$ and a target at $\mathbf{t}$ with $K$ sample
heights $h_1<\dots<h_K$, the ray to height $h_k$ is $z_k(u) = (g(\mathbf{o})+h_o)(1-u) + (g(\mathbf{t})+h_k)\,u$,
$u\in(0,1)$. Height $h_k$ is visible if $z_k(u_m) > s(\mathbf{x}(u_m))$ at every interior sample
$u_m$, where $s$ is the surface (ground + obstruction height). The visible fraction is
$v = \frac{1}{K}\sum_k \mathbf{1}[h_k \text{ visible}]$. With $K=3$, $v\in\{0,\tfrac13,\tfrac23,1\}$: a
hull-down vehicle showing only its turret has $v=\tfrac13$. The observer's own cell and the
target's own cell are excluded from the blocking test so a unit inside a building or wood is not
blocked by its own cover; cover instead multiplies exposure (A.2).

Why a fraction rather than a Boolean: it feeds both the detection hazard (a partly visible target
is detected more slowly) and the hit probability (A.4), where it scales the presented area. That
gives a single, physically motivated way for terrain to act on both links of the kill chain.
</details>

<details>
<summary><b>A.2 Detection as a hazard process, and why per-tick rescaling is exact</b> (<code>engine.simulate</code>)</summary>

Detection of target $j$ by observer $i$ is a first-passage time with hazard (per $\Delta_0$)

$$\lambda_{ij} = k_{\det}\, s_i\, g_j\, v_{ij}\, \big(1 + \varphi\,\mathbf 1[j \text{ fired}]\big)\big(1 + \mu\,\mathbf 1[j \text{ moving}]\big)\Big(\frac{1000}{\max(D_{ij},50)}\Big)^2,$$

where $s_i$ is sensor quality, $g_j$ signature, $v_{ij}$ the visible fraction times exposure
(concealment $\times$ posture $\times$ suppression). Over one tick of length $f\Delta_0$ the
survival probability of "not yet detected" is $e^{-f\lambda_{ij}}$, so
$P(\text{detect this tick}) = 1 - e^{-f\lambda_{ij}}$. Detections persist while both units live.
The inverse-square term is the small-target limit of an angular-size sensor model; $k_{\det}$ is
the global scale that rung 2 calibrates.

**Tick rescaling.** A per-tick probability $p$ stated for $\Delta_0$ corresponds to a hazard
$-\ln(1-p)/\Delta_0$. Keeping the hazard fixed and changing the tick to $f\Delta_0$ gives
$p_f = 1-(1-p)^f$; rates and speeds scale linearly with $f$; durations in ticks scale by $1/f$.
These are the transformations `simulate` applies, so parameter *statements* are tick-invariant.
What is not tick-invariant is the simultaneity of resolution within a tick (A.7).
</details>

<details>
<summary><b>A.3 Random-utility target selection: Gumbel-max is the multinomial logit</b> (<code>engine.gumbel_choice</code>, <code>choice_prob</code>)</summary>

A shooter faces alternatives $k\in\{0,1,\dots,K\}$ (0 = hold fire) with deterministic utilities
$V_k$ and i.i.d. Gumbel(0,1) shocks $\varepsilon_k$; it chooses $\arg\max_k(\lambda V_k+\varepsilon_k)$.
The Gumbel-max theorem gives
$$P(\text{choose } k) = \frac{e^{\lambda V_k}}{\sum_{k'} e^{\lambda V_{k'}}},$$
the multinomial logit. Sampling is exact: draw $u_k\sim U(0,1)$, set
$\varepsilon_k = -\ln(-\ln u_k)$, take the argmax. Because the draws are per (shooter, alternative)
with a fixed shape, this preserves common random numbers across COAs (A.8).

Utilities: $V_k = \ln(P_{H,k}P_{K,k}\,\text{value}_k) + b_\text{threat}\mathbf 1[k \text{ shot at me}]$
for targets; $V_0 = v_0 + \delta v_0\,\mathbf 1[\text{suppressed}]$ for hold fire. Using the log of
expected value as utility means that at $\lambda=1$ the shooter is a proportional-value
allocator and as $\lambda\to\infty$ it is a greedy best-shot allocator; $\lambda$ is therefore
"rationality" and is what rung 2 tries (and fails, from aggregates) to identify. `choice_prob`
evaluates the logit probability of the realized choice for the replay feed.
</details>

<details>
<summary><b>A.4 Circular-normal hit probability</b> (<code>engine.hit_prob</code>)</summary>

Let the aim-point error be bivariate normal $\mathcal N(0,\sigma^2 I)$ in the target plane and let
the visible part of the target be a disc of area $A_{\text{eff}} = vA$, radius
$R=\sqrt{A_{\text{eff}}/\pi}$. The miss distance $\rho$ is Rayleigh, so
$$P_H = P(\rho\le R) = 1-\exp\!\Big(-\frac{R^2}{2\sigma^2}\Big) = 1-\exp\!\Big(-\frac{vA}{2\pi\sigma^2}\Big).$$
The dispersion is angular and compounds independent error sources in quadrature:
$$\sigma^2 = D^2\big(\theta_0^2 + \theta_s^2\,m_{\text{shooter}} + \theta_t^2\,m_{\text{target}}\big)(1+\kappa\,\text{supp}).$$
$\theta_0$ is stationary aim error (sights, fire control), $\theta_s$ the penalty for firing on
the move, $\theta_t$ lead error against a mover, $\kappa$ the suppression penalty.
`test_hit_prob_matches_brute_force` checks the closed form against 300 000 sampled shots. Scaling
$A$ by $v$ rather than reducing $P_H$ multiplicatively is what makes "one-third exposed" mean
one-third of the target area, which is the physically right effect of a hull-down position.
</details>

<details>
<summary><b>A.5 The kill chain and suppression as competing outcomes</b></summary>

Per tick a live shooter with a chosen target fires with probability $p_{\text{ready}}$ (the
rate-of-fire term), hits with $P_H$, kills with $P_K$. A shot that does not kill suppresses with
probability $p_{\text{supp}}(\text{shooter})\,e^{-D/1000}\,\text{vuln}(\text{target})$, so per tick
$P(\text{kill}) = p_{\text{ready}}P_HP_K$ and
$P(\text{suppress only}) = p_{\text{ready}}(1-P_HP_K)\,p_{\text{supp}}e^{-D/1000}\,\text{vuln}$.
Suppression is a countdown state (2 ticks for direct fire, 4 for artillery, at $\Delta_0$) that
halves exposure, blocks movement, multiplies dispersion by $(1+\kappa)$, raises $V_0$ by
$\delta v_0$, and multiplies $\lambda$ by $\lambda_{\text{supp}}$. This is the mechanism through
which "fires" and "recon" interact: recon widens the artillery target list and raises the
mission rate, artillery suppression lowers the defender's fire, and the attacker closes.
Both sides resolve simultaneously within a tick from the state at the start of the tick.
</details>

<details>
<summary><b>A.6 Breakpoints, outcomes and censoring</b></summary>

Each replication draws breakpoints $\beta_B\sim\text{Beta}(a_B,b_B)$, $\beta_R\sim\text{Beta}(a_R,b_R)$
on the fraction of fighting units lost. The replication ends at the first tick where either
fraction reaches its breakpoint; a tie counts against Blue. Blue "wins" iff Red broke first.
Replications reaching the time limit $T$ are right-censored at $T$: their decision time is
$\ge T$ and unknown. This is why time-to-decision statistics use survival estimators (A.11)
and why `cause` records 0/1/2. The Beta form gives a stochastic break threshold with a
mean $a/(a+b)$ and a concentration $a+b$ that the ladder (rung 3) can move.
</details>

<details>
<summary><b>A.7 Discretization error and the convergence test</b> (<code>scripts/convergence.py</code>)</summary>

The engine is a fixed-step scheme for a continuous-time marked point process. Within a tick all
events are drawn from the state at the tick start, so a unit killed in tick $t$ still fires in
tick $t$. The probability that two duelling units both die in the same tick is $O(\Delta^2)$ per
pair, but summed over pairs and ticks its effect on P(win) is $O(\Delta)$, i.e. first order in
the tick length, like an explicit Euler scheme. The check runs the same seed at
$\Delta\in\{30,15,7.5,3.75\}$ s and cell $\in\{25,12.5\}$ m and asks whether successive halvings
move the outputs by more than their intervals. Results: 15 s $\to$ 7.5 s moved P(win) by 0.13;
7.5 s $\to$ 3.75 s by 0.007 (inside the interval), so 7.5 s is the converged default.
Cell halving moved P(win) by about 0.05: an LOS-sampling effect, not a time-stepping one.
</details>

<details>
<summary><b>A.8 Common random numbers, substreams and the paired estimator</b> (<code>engine.STREAMS</code>, <code>stats.paired_ci</code>, <code>stats.crn_gain</code>)</summary>

For two COAs with per-replication outputs $Y^A_r, Y^B_r$,
$$\operatorname{Var}(\bar Y^A-\bar Y^B) = \frac{\sigma_A^2+\sigma_B^2-2\rho\,\sigma_A\sigma_B}{n}.$$
Independent runs have $\rho=0$; running both on the same random numbers induces $\rho>0$ and
shrinks the variance by the factor $1-\frac{2\rho\sigma_A\sigma_B}{\sigma_A^2+\sigma_B^2}$, which
`crn_gain` reports. The pairing is only valid if the same draw is used for the same purpose in
both runs, so every draw type has its own generator (spawned child of one `SeedSequence`) and
every draw has a fixed array shape set by the roster: a kill that happens in one COA cannot
shift later draws in the other. New mechanics get new stream names appended to `STREAMS`;
reordering would silently break the pairing. `test_substreams_keep_crn_when_a_mechanic_is_switched`
verifies the first tick's detections are identical with fires on and off.
</details>

<details>
<summary><b>A.9 Interval estimators: Wilson, delta method, Greenwood, sup-t</b> (<code>stats.wilson</code>, <code>ratio_ci</code>, <code>kaplan_meier</code>, <code>sup_t_band</code>)</summary>

**Wilson.** Invert the score test $\{p: |\hat p-p|/\sqrt{p(1-p)/n}\le z\}$; solving the quadratic
gives centre $\frac{\hat p + z^2/2n}{1+z^2/n}$ and half-width
$\frac{z}{1+z^2/n}\sqrt{\hat p(1-\hat p)/n + z^2/4n^2}$. Unlike Wald, it never collapses at
$\hat p\in\{0,1\}$ and its coverage stays near nominal for small $np$; `test_wilson_inside_unit_interval_and_covers`
checks coverage at $p=0.05$, $n=50$.

**Delta method for a ratio of means** $R=\bar X/\bar Y$: with $C$ the covariance of $(\bar X,\bar Y)$,
$$\operatorname{Var}(R)\approx R^2\Big(\frac{C_{XX}}{\bar X^2}+\frac{C_{YY}}{\bar Y^2}-\frac{2C_{XY}}{\bar X\bar Y}\Big).$$

**Greenwood** for the Kaplan–Meier variance: $\widehat{\operatorname{Var}}(\hat S(t)) = \hat S(t)^2\sum_{t_j\le t}\frac{d_j}{n_j(n_j-d_j)}$.

**Simultaneous band.** For a curve $\bar Y_t$ with pointwise SE $s_t$, bootstrap replications of
$\sup_t |\bar Y^*_t-\bar Y_t|/s_t$ give the $1-\alpha$ quantile $c_\alpha$; the band
$\bar Y_t\pm c_\alpha s_t$ covers the whole mean curve with probability $\approx 1-\alpha$.
$c_\alpha > z_{1-\alpha/2}$ always (3.2 vs 1.96 in the test), which is the price of a claim about
the entire curve instead of one time point.
</details>

<details>
<summary><b>A.10 Sequential sampling and Kim–Nelson ranking-and-selection</b> (<code>stats.run_until</code>, <code>stats.kn_select</code>)</summary>

**Sequential stopping.** Draw batches until $z_{1-\alpha/2}\,s/\sqrt n \le h$. Stopping on an
observed half-width makes the final interval slightly liberal; with a minimum of 200
replications the effect is negligible (Law 2015, §9.4).

**KN procedure** (Kim & Nelson 2001), $k$ systems, indifference zone $\delta$, confidence $1-\alpha$,
first stage $n_0$: compute $S^2_{il}$, the sample variance of the pairwise differences
$X_{ir}-X_{lr}$, and
$$\eta = \tfrac12\Big[\big(\tfrac{2\alpha}{k-1}\big)^{-2/(n_0-1)}-1\Big],\qquad h^2 = 2c\,\eta\,(n_0-1).$$
At stage $r$, system $i$ survives iff for every surviving $l$
$$\bar X_i(r) \ge \bar X_l(r) - W_{il}(r),\qquad W_{il}(r)=\max\Big\{0,\ \frac{\delta}{2cr}\Big(\frac{h^2S^2_{il}}{\delta^2}-r\Big)\Big\}.$$
The continuation region $W$ shrinks linearly to zero at $r = h^2S^2_{il}/\delta^2$, so the
procedure terminates. The guarantee is $P(\text{select the best})\ge1-\alpha$ whenever the best
system's mean exceeds every other by at least $\delta$ (the indifference-zone assumption); it is
what a "which COA is best" question needs, since pairwise intervals on the winner are biased by
selection. `test_kn_selects_best_system` checks 9/10 correct selections on Gaussian systems
separated by $2\delta$.
</details>

<details>
<summary><b>A.11 Survival and competing risks: Kaplan–Meier and Aalen–Johansen</b> (<code>stats.kaplan_meier</code>, <code>stats.aalen_johansen</code>)</summary>

With distinct event times $t_1<t_2<\dots$, $n_j$ at risk just before $t_j$ and $d_j$ events at $t_j$,
the product-limit estimator is $\hat S(t)=\prod_{t_j\le t}(1-d_j/n_j)$; censored replications
contribute to $n_j$ until their censoring time and never to $d_j$. That is the correct treatment
of fights that hit the time limit: they are known to last at least $T$.

Two endings compete (Blue broke, Red broke). With $d_{kj}$ events of cause $k$ at $t_j$, the
cumulative incidence of cause $k$ is
$$\hat F_k(t)=\sum_{t_j\le t}\hat S(t_j^-)\,\frac{d_{kj}}{n_j},$$
where $\hat S$ is the all-cause survivor. Treating the other cause as censoring would estimate
$1-\prod(1-d_{kj}/n_j)$, which overstates $F_k$ because it pretends the competing ending could
still happen later. `test_aalen_johansen_competing_exponentials` verifies $F_1(30)\to
\frac{\lambda_1}{\lambda_1+\lambda_2}(1-e^{-(\lambda_1+\lambda_2)30})$ for two exponential causes.
</details>

<details>
<summary><b>A.12 Empirical-Bayes shrinkage of per-cell rates</b> (<code>stats.shrink_rates</code>)</summary>

Cell $i$ has $x_i$ deaths in $n_i$ trials with rate $p_i\sim\text{Beta}(a,b)$; then
$x_i\mid p_i\sim\text{Bin}(n_i,p_i)$ and the posterior mean is
$$\hat p_i = \frac{a+x_i}{a+b+n_i} = w_i\,\frac{x_i}{n_i} + (1-w_i)\,\frac{a}{a+b},\qquad w_i=\frac{n_i}{n_i+a+b}.$$
The prior is estimated by a weighted method of moments. With $m=\sum x_i/\sum n_i$,
$q=\sum_i n_i(\hat p_i^{\text{raw}}-m)^2-(k-1)m(1-m)$ and $\text{den}=\sum n_i-\sum n_i^2/\sum n_i$
(the DerSimonian–Laird form), the between-cell variance is $\tau^2=\max(q/\text{den},\epsilon)$
and the prior strength is $a+b = m(1-m)/\tau^2-1$, clipped to $[1,10^4]$. Cells with few trials
shrink toward $m$, cells with many keep their raw rate, and when the cells show no excess
variation over binomial noise the map collapses to a flat rate (which is the honest answer).
</details>

<details>
<summary><b>A.13 History matching: implausibility and the NROY region</b> (<code>calibrate.history_match</code>)</summary>

For parameters $\theta$, reference output $z$ with variance $\sigma_{\text{ref}}^2$, simulator mean
$\hat f(\theta)$ with Monte Carlo variance $\sigma_{\text{mc}}^2(\theta)$ and a model-discrepancy
variance $\sigma_{\text{disc}}^2$ (the error you tolerate at the best $\theta$),
$$I(\theta)=\frac{|\hat f(\theta)-z|}{\sqrt{\sigma_{\text{mc}}^2+\sigma_{\text{ref}}^2+\sigma_{\text{disc}}^2}},\qquad
I_{\max}(\theta)=\max_{\text{outputs}} I(\theta).$$
$\theta$ is ruled out when $I_{\max}>3$. The threshold is Pukelsheim's three-sigma rule: for any
unimodal distribution, $P(|X-\mu|>3\sigma)\le 4/81\approx0.05$, so a 3-sigma miss is implausible
without assuming normality. Waves sample a Latin hypercube (A.14) in the current box, compute
$I_{\max}$, and refocus the box on the hull of the non-implausible points, never narrowing a side
below `shrink_floor`. The output is a *region* (NROY), not a point: if several outputs cannot
distinguish two parameters, the region stays wide in that direction, which is how
non-identifiability is detected (rung 2's $\lambda$). The discrepancy term is essential; without
it, as $n\to\infty$ the simulator's own noise vanishes and any imperfect model rules out
everything.
</details>

<details>
<summary><b>A.14 Sampling the terrain and parameter spaces; holdout selection</b> (<code>descriptors</code>, <code>doe.design</code>)</summary>

A Latin hypercube of $m$ points in $d$ dimensions places one point in each of $m$ equal slices of
every axis, giving marginal stratification; `scipy.stats.qmc.LatinHypercube(optimization="random-cd")`
then improves the centred discrepancy so the points are also spread jointly. Log-scale axes
(densities, $\theta_0$, $\lambda$) are sampled uniformly in $\log$ so the design spends points
evenly across orders of magnitude.

Terrain descriptors $\mathbf d\in\mathbb R^7$ are standardized; $k$-means picks $k$ centroids and
the terrain nearest each; the holdout is the fraction of picks farthest from the centre of the
picked set, so it tests extrapolation to unusual ground, not interpolation. Openness at 500 m
and 1500 m is the mean visible fraction over random point pairs at those separations, i.e. a
Monte Carlo estimate of the terrain's intervisibility function at two ranges.
</details>

<details>
<summary><b>A.15 Held-out coverage and the error-trend test</b> (<code>calibrate.coverage_report</code>, <code>error_trend</code>)</summary>

For each held-out output $z$, draw $\theta^{(1..K)}$ uniformly from the final NROY box, simulate,
and form predictive samples $y^{(k)}\sim\mathcal N\big(\hat f(\theta^{(k)}),\ \sigma_{\text{mc}}^2+\sigma_{\text{ref}}^2\big)$;
the central $90\%$ interval of $\{y^{(k)}\}$ is the prediction interval and coverage is the share
of held-out $z$ inside it. The composition (parameter uncertainty $\times$ simulation noise
$\times$ reference noise) is the same one that will be quoted when the model is used, so
coverage near 0.90 is a direct check that the quoted uncertainty is calibrated. The error-trend
test regresses $\hat f-z$ on each descriptor across held-out terrains; a slope with $p<0.05$
means the error depends on the ground and no parameter setting will remove it, which is the
signature of a missing mechanic.
</details>

<details>
<summary><b>A.16 Gaussian-process metamodel with a Monte Carlo noise floor</b> (<code>doe.fit_metamodel</code>, <code>loo_r2</code>)</summary>

Outputs are standardized and modelled as $y(\mathbf u)=\mu(\mathbf u)+\varepsilon$ with
$\mu\sim\mathcal{GP}(0,\ \sigma_f^2\,k(\mathbf u,\mathbf u'))$, anisotropic RBF
$k=\exp\!\big(-\tfrac12\sum_j (u_j-u'_j)^2/\ell_j^2\big)$, and white noise $\varepsilon\sim\mathcal N(0,\sigma_n^2)$.
Hyperparameters maximize the log marginal likelihood
$-\tfrac12 \mathbf y^\top K^{-1}\mathbf y-\tfrac12\ln|K|-\tfrac m2\ln 2\pi$ with
$K=\sigma_f^2\,k+\sigma_n^2 I$, from three restarts. The lower bound on $\sigma_n^2$ is the mean
Monte Carlo variance of the design points (in standardized units), the stochastic-kriging idea:
the GP cannot fit below the simulator's own noise. `noise_share` $=\sigma_{\text{mc}}^2/\sigma_y^2$
says how much of the output variation is noise; LOO $R^2=1-\sum(y_i-\hat y_{-i})^2/\sum(y_i-\bar y)^2$
refits without each point. Below $R^2\approx0.7$ the metamodel is not a trustworthy stand-in for
the simulator and the Sobol indices from it are indicative only.
</details>

<details>
<summary><b>A.17 Sobol indices: Saltelli sampling with Jansen's estimators</b> (<code>doe.sobol_indices</code>)</summary>

For $Y=f(\mathbf U)$ with independent inputs, $\operatorname{Var}(Y)=\sum_i V_i+\sum_{i<j}V_{ij}+\dots$;
the first-order index $S_i=V_i/V=\operatorname{Var}(E[Y\mid U_i])/V$ and the total index
$S_{T_i}=E[\operatorname{Var}(Y\mid \mathbf U_{\sim i})]/V$. With two scrambled Sobol matrices
$A,B$ and $A^{(i)}_B$ ($A$ with column $i$ from $B$):
$$\hat S_i=\frac{1}{NV}\sum_{r} f(B)_r\big(f(A^{(i)}_B)_r-f(A)_r\big),\qquad
\hat S_{T_i}=\frac{1}{2NV}\sum_r\big(f(A)_r-f(A^{(i)}_B)_r\big)^2,$$
which are the estimators Saltelli et al. (2010) recommend. Bootstrap over rows gives standard
errors. $S_{T_i}\approx0$ means input $i$ can be fixed anywhere in its range without changing the
output; $S_{T_i}\gg S_i$ means it acts mainly through interactions. Rung 1 shows
$S_T(p_{K,\text{AT}\to\text{tank}})\approx0.6$–$0.74$: Red's lethality dominates every output.
</details>

<details>
<summary><b>A.18 The zero-sum COA game: LP solution and bootstrap</b> (<code>stats.solve_zero_sum</code>, <code>bootstrap_game_value</code>)</summary>

With payoff matrix $M$ (attacker rows, maximizing), shift $M'=M-\min M+1>0$ and solve
$$\max_{\mathbf x,v}\ v\quad\text{s.t.}\quad M'^\top\mathbf x\ge v\mathbf 1,\ \sum x_i=1,\ \mathbf x\ge0,$$
and the dual for the defender $\min_{\mathbf y,w} w$ s.t. $M'\mathbf y\le w\mathbf 1$. Strong LP
duality gives $v=w$ (the minimax theorem); subtracting the shift restores the value. The value is
what the attacker can guarantee against a defender who also plays minimax; a pure saddle point
(as in the demo) means neither side gains from randomizing. Because every cell is estimated
from the same replications (CRN), the bootstrap resamples *replications* jointly across cells,
re-solves the game, and reports the percentile interval on the value and the mean mixes.
</details>

<details>
<summary><b>A.19 The aggregate brigade engine as a mean-field limit of the entity engine</b> (<code>brigade.BrigadeSim</code>)</summary>

Consider company $i$ with $n_{ia}$ platforms of class $a$ and enemy company $j$ with $n_{jb}$ of
class $b$. In the entity engine each platform of class $a$ fires per minute at rate
$4p_{\text{ready},a}$ (four reference ticks per minute), chooses a target by the logit (A.3), and
kills with $P_HP_K$. Summing Bernoulli kill events over platforms and taking expectations gives
the aggregate rate

$$E[\text{kills}_{jb}] = \sum_{i,a}\underbrace{4p_{\text{ready},a}\,q\,n_{ia}(1-0.7\,\text{supp}_i)}_{\text{shots}_{ia}}\ \pi_{ia\to jb}\ \ell_{ia\to jb},$$

with $\ell = P_H(D_{ij},v_{ij})P_K(a,b)\,\mathbf 1[D\le R_{\max}]\,\text{acq}_{ia\to jb}$ the kill
probability per shot and $\pi$ the allocation. The allocation is the $\lambda=1$ logit with
utility $\ln(n_{jb}\,\text{value}_b\,\ell)$, i.e. $\pi\propto n_{jb}\,\text{value}_b\,\ell$: a shooter
spreads fire in proportion to expected value, the mean-field version of A.3 (taking $\lambda=1$
sidesteps the un-identified $\lambda$ of rung 2). The acquisition fraction
$\text{acq}=1-\exp(-4\,\lambda^{\det}_{ab})$ is the per-minute detection probability of A.2 and
stands in for per-pair persistent detection; without it every platform would fire at full rate
the instant its side's network saw a target, which the first brigade runs showed to be wrong by
a factor of several in tempo.

Kills are drawn as $\min\{n_{jb},\ \text{Poisson}(E[\text{kills}_{jb}])\}$: the Poisson law is the
limit of many small-probability Bernoulli kills (Le Cam), and the cap is the physical constraint.
The "dither" option $\lfloor E+U\rfloor$ is unbiased but has variance $\le\tfrac14$ instead of $E$.
Artillery kills use expected rounds $\times$ lethality $\times$ class vulnerability $\times$
cover; aviation losses are $\text{Binomial}(\text{helicopters},\ 0.012\cdot\text{shorad}\cdot\min(1.5,\text{near}/5))$
per minute on station. Ammunition follows $\dot S = -\text{consumption} + 20\,T_0\,\phi_{\text{sup}}/60$
rounds per minute with $T_0$ initial tubes and $\phi_{\text{sup}}$ the surviving support fraction.
Breakpoints act on *effective* strength (unbroken companies only), so a brigade culminates when
its battalions come apart, not only when its vehicles are destroyed.
</details>

<details>
<summary><b>A.20 Rung 5: calibrating the aggregate engine to the entity engine</b> (<code>scripts/rung5_calibrate.py</code>)</summary>

Let $g_A(\kappa_A,\kappa_D)$ and $g_D(\kappa_A,\kappa_D)$ be the aggregate engine's mean attacker
and defender loss fractions on the calibration cases when its attacker$\to$defender and
defender$\to$attacker kill expectations are multiplied by $\kappa_A,\kappa_D$, and $e_A,e_D$ the
entity engine's. Each $g$ is monotone in its own multiplier (more lethal defender fire cannot
lower attacker losses), so the root of $g_D(\kappa_A,\cdot)-e_A$ in $\kappa_D$ is found by bisection
on a log scale (geometric-mean midpoint, since the multiplier is a ratio), then the root in
$\kappa_A$ for $g_A-e_D$, alternating for two rounds. P(win) and duration are held out of the fit
and reported as residuals. The first fit failed to converge in $\kappa_D$ because the aggregate
side had three companies with independent breakpoints while the entity side had one team: no
value of $\kappa_D$ can raise losses that the breakpoint structure caps. Matching the entity
structure (one aggregate entity per side) made the objective solvable. Fitted
$(\kappa_A,\kappa_D)=(0.63,\ 2.54)$ with residuals $+0.05$ in P(win), $\approx0$ in losses,
$+0.9$ min in duration. The fit is on company fights; applying it at brigade scale is an
extrapolation that a future battalion-scale check should test.
</details>

<details>
<summary><b>A.21 Command and control as a latency–initiative process</b> (<code>c2.py</code>, <code>brigade.BrigadeSim</code>)</summary>

A trigger event $E$ at time $t_E$ is observed by the brigade at
$T_{obs}=t_E+G$, $G\sim\text{Geometric}(\nu)$ with $\nu=\text{net}\cdot h$ and $h\in\{1,\tfrac12\}$
for HQ alive/destroyed. The order is delivered at $T_{obs}+L$ with
$L\sim\text{LogNormal}(\mu,\sigma^2)$, $\mu=\log(\tau h^{-1})-\sigma^2/2$ so that $E[L]=\tau/h$
($\sigma=0.5$). The subordinate who sees the trigger acts on intent at the first success of a
Bernoulli$(p_{\text{init}}/\tau)$ trial per minute after $T_{obs}$, so the effective time to
action is $\min(L,\ \text{Geometric}(p_{\text{init}}/\tau))$. The initiative clock has mean
$\tau/p_{\text{init}}$ minutes against the order's mean $\tau/h$, so the two are a race on the
same time scale: with $p_{\text{init}}\to1$ the initiative clock's mean falls to $\tau$, it
usually fires first and the order is pre-empted; with $p_{\text{init}}\to0$ its mean diverges
and nothing happens before the order arrives. This is the
formal content of "mission command" here: not better decisions, but a race between an order
and a subordinate's initiative, with the parameters of the race differing by nation.

Clearance of fires is a Bernoulli$(\text{sync}\cdot c)$ gate per mission attempt with
$c=1$ within a lineage and $c=0.5$ across Artesh/IRGC; a failure costs a delay of $L/3$.
Friendly-fire losses out of sector are $\text{Poisson}\big(0.003\,(1-\text{sync}\,c)\,n\big)$
per minute while in contact. Crowding multiplies artillery lethality by
$1+0.3\,(k-2)^+$ for $k$ companies in a cell and halves speed for $k>2$. All of these are
sensitivities, not measurements; the study in 19.9 exists so that any conclusion drawn from
them is reported with its Sobol index.
</details>

