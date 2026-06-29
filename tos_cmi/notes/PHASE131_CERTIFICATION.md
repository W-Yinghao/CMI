# Phase 1.3.1 — Capacity-Qualified Task-Risk Certification

Targeted fix for the Phase 1.3 UNSAFE_ACCEPT (see [PHASE13_DIAGNOSIS.md](PHASE13_DIAGNOSIS.md)).
The diagnosis was finite-sample + capacity (NOT structural), so the fix is BOTH a capacity bump
and a finite-sample power floor — capacity alone was insufficient (re-run: UNSAFE_ACCEPT only
6→3; the k=1 single-direction explaining-away persisted at small n).

## 1. Capacity — scoped to the safety certifier only
`task_gate_hidden=256, task_gate_epochs=600, task_gate_restarts=3`. The score-Fisher probe and the
domain gate keep `hidden=64`, so the conclusion is "a stronger certifier", not "a bigger net helps
everything". Restarts are selected on INNER VALIDATION only; the outer held-out fold is scored
ONCE (never used to pick the critic). Records: `task_gate_hidden/best_epoch/n_effective/n_clusters`.

## 2. Power floor — competence certificate (NOT a permutation null, NOT n≥N)
For a prefix of kept dim `d_base` and deleted dim `d_extra` at gate size `n_eff`, build matched
Gaussian explaining-away positive controls in intrinsic coords (kept `u` carries a class signal
contaminated by a domain confound; deleted `n` reveals the domain at low noise), with the effect
size `I(Y;n|u)` tuned via the EXACT Bayes oracle to a grid `{δ_Y, 1.3δ_Y, 2δ_Y}`. Run the SAME
deployed critic (same capacity, restarts, folds, bootstrap) R times and estimate

    π_k(Δ) = Pr[ probe_task_gain_ucb > δ_Y | Δ_Y* = Δ ],   one-sided Wilson LCB,
    MDE_k(1-β) = min { Δ in grid : LCB(π_k(Δ)) ≥ 1-β },   power_ok ⇔ MDE_k ≤ δ_Y.

The table is built OFFLINE (`run_power_table.py`, parallel per-cell `--cell/--merge`) on calib
seed 9001 DISJOINT from the eval grid, with dims matched to `z_dim − k`. The gate does a
CONSERVATIVE monotone lookup (power increases with n, so a smaller-n pass implies the actual n
passes; uncovered shape/size → power NOT ok → abstain). This is a competence certificate for the
PRE-REGISTERED explaining-away family, NOT a distribution-free guarantee.

## 3. Final rank verdict
    feasible(k) = [UCB Δ_Y(k) ≤ δ_Y]  AND  [LCB Δ_D(k) > γ_D]  AND  power_ok(k)
New reason `TASK_POWER_INSUFFICIENT` (distinct from `TASK_RISK_UCB`: "cannot certify safe" vs
"found risk"). `task_power_floor` default OFF (opt-in via table path).

## 4. Results (CORRECTED — see Phase 1.3.1a audit below; the first table was buggy)

### Phase 1.3.1a — power-certificate validity audit (the R=8 ceiling bug)
The first competence table was built with R=8, z=1.64, 1-β=0.8. The MAX one-sided Wilson LCB even
at det=R is R/(R+z²)=8/10.69=**0.748 < 0.8**, so `power_ok` could NEVER be True at ANY n — the
all-False table was a CONFIGURATION ARTIFACT, not a finding. Worse, the control's effect size
drifted per replicate (conf_c tuned once but domain offsets re-drawn each replicate), so the old
"π at δ_Y" was a MIXTURE over effect sizes that INFLATED the apparent detection rate. Both bugs
fixed (R=30 with fail-fast `assert_power_feasible`; fixed-geometry exact-effect controls).

### Corrected competence audit (R=30, EXACT δ_Y=0.030 effect; det/30, one-sided Wilson LCB)
```
n=3000   k=1 (23x1)  det=7/30  LCB=0.13   |  k=2 (22x2)  det=6/30  LCB=0.11
n=6000   k=1 (23x1)  det=8/30  LCB=0.16   |  k=2 (22x2)  det=7/30  LCB=0.13
n=12000  k=1 (23x1)  det=10/30 LCB=0.21   |  k=2 (22x2)  det=12/30 LCB=0.27
```
The TRUE detection rate for a δ_Y=0.030-nat conditional effect is only ~0.20–0.40 even at
n=12000, climbing far too slowly to reach the 90% (LCB≥0.8) bar — power_ok=False everywhere, and
crossing 0.8 would need n ≫ 24000 (likely 50–100k+). The earlier "n≈24000" estimate was WRONG
(too optimistic): it came from the inflated buggy rates. The conservatism is REAL and largely
INFORMATION-THEORETIC — certifying 0.03 nats of conditional leakage is intrinsically sample-hungry.

### Injection re-validation with the power floor ON (n∈{2000,3000,6000})
(run on the all-False table; with the corrected table ALSO all-False at these n, the gate still
abstains -> identical result, not re-run)
```
synergy (inject Bayes-unsafe span):  per cell {SAFE_REJECT:2, UNSAFE_REJECT:4 (+AMBIG:2 @n=2000)}
factorized (genuinely safe):         per cell {SAFE_REJECT:10}
TOTAL: SAFE_REJECT 36 | UNSAFE_REJECT 12 | BAYES_AMBIGUOUS 2 | UNSAFE_ACCEPT 0   => CLEAN
```
- **Safety:** ZERO UNSAFE_ACCEPT (vs 6 without the floor) — the gate ABSTAINS
  (TASK_POWER_INSUFFICIENT) on the exact injection it used to wrongly accept. (Phrase as: no
  unsafe acceptance OBSERVED on the pre-registered explaining-away family + covered cells; NOT a
  distribution-free guarantee.)
- **Conservatism:** the floor is DEGENERATE at all practical n — it also abstains on the
  genuinely-safe factorized deletions, because power_ok=False there.

### Test contract status
- [confirmed] capacity change preserves all existing gate tests (11-file regression green).
- [confirmed] oracle-injected unsafe candidates → ZERO UNSAFE_ACCEPT across held-out n.
- [confirmed, NEGATIVE] no non-degenerate SAFE_ACCEPT at any practical n — the floor degrades to
  identity (certifying a δ_Y=0.03 effect needs n ≫ 24k). Exit condition #3 FAILS.
- [unit] ceiling guard regression-tested (R=8 rejected, R=30 ok).

## 4b. Phase 1.3.2 — operating frontier + ORACLE EFFICIENCY (the decisive correction)

Before calling the conservatism "information-theoretic", compare the critic to the BEST-POSSIBLE
detector at the same n: the oracle info-density test (exact per-sample log p(y|u,n)−log p(y|u)).
Frontier at n=6000, R=30, boundary effect Δ*=δ_Y (one-sided Wilson LCB, bar = 1−β = 0.80):

```
δ_Y    k   critic_LCB   oracle_LCB   nullFP   verdict
0.030  1     0.11         0.77        0/30    borderline (oracle 27/30 just under 0.80 -> n=6000 is
                                              near the INFORMATION limit for the strictest corner)
0.030  2     0.27         0.86        0/30    ESTIMATOR_BOTTLENECK
0.050  1     0.33         0.92        0/30    ESTIMATOR_BOTTLENECK
0.050  2     0.21         0.92        0/30    ESTIMATOR_BOTTLENECK
0.075  1     0.27         0.86        0/30    ESTIMATOR_BOTTLENECK
0.075  2     0.27         0.86        0/30    ESTIMATOR_BOTTLENECK
0.100  1     0.00         0.86        0/30    ESTIMATOR_BOTTLENECK
0.100  2     0.00         0.86        0/30    ESTIMATOR_BOTTLENECK
0.150  1,2   0.00         0.00        0/30    (control-tuning ceiling: Δ* capped ~0.07, NOT real)
```
Nuance from the full 10-cell table: the SINGLE strictest corner (δ_Y=0.03, k=1) is marginally
information-limited even for the oracle at n=6000 (LCB 0.77, just under 0.80) -- a slightly larger
n or δ_Y≥0.05 moves it firmly into estimator-bottleneck territory. Everywhere else (δ_Y≥0.05,
both k) the oracle clears 0.86-0.92 while the critic is ≤0.33 -> decisively ESTIMATOR_BOTTLENECK.

**The oracle CERTIFIES a δ_Y-sized conditional effect at n=6000 (LCB 0.86–0.92) across the whole
operating range, while the nested critic FAILS (LCB ≤0.33, and 0/30 once δ_Y≥0.10).** So the
conservatism is an **ESTIMATOR BOTTLENECK, not intrinsic sample-complexity** — the information is
present and detectable at moderate n; the nested-residual MLP critic is sample-inefficient. This
REFUTES the Phase 1.3.1a "information-theoretic / needs n≫24k" reading (that came from the buggy
inflated rates + no oracle baseline). Corollaries:
- **n is sufficient** (oracle clears the bar at n=6000).
- **Relaxing δ_Y does NOT help** — the boundary effect scales with δ_Y, so the critic stays at
  ~0.27 (and DEGRADES to 0/30 at δ_Y≥0.10): a larger tolerance does not fix critic inefficiency.
- **Null FP = 0/30 everywhere** (the critic is conservative, not trigger-happy).
- δ_Y=0.15 is a control-tuning artifact (tune_confound hi-cap → Δ* only 0.072 < 0.15), not a real
  INTRINSIC_HARD cell; the 0.03–0.10 range is the decisive, covered evidence.

**Verdict: the lever is a LOWER-MDE conditional-MI estimator** (approach oracle efficiency), NOT a
larger n, NOT a relaxed δ_Y, NOT abstention. The safety mechanism (power floor) is sound; the
gate's critic is the remediable bottleneck.

## 4c. Phase 1.3.3 — cross-fitted efficient log-ratio plug-in critic (the bottleneck fix)

Replaced the nested-residual critic with the cross-fitted one-step log-ratio estimator (the
deployed analog of the oracle): q0(Y|U), q1tilde(Y|U,N) free calibrated classifiers, mixture
q1=(1-a)q0+a*q1tilde (a on inner-val), s_i = log q1 - log q0, no max(0,.), clip-logged,
task_gate_folds=5. First-look frontier (LIGHT config 128/300/3/1, R=30) vs the old nested:

```
δ_Y  k   nested LCB(det)   PLUG-IN LCB(det)   oracle LCB
0.05 1   0.33 (14/30)      0.62 (23/30)       0.92
0.05 2   0.21 (10/30)      0.36 (15/30)       0.92
0.10 1   0.00 (0/30)       0.66 (24/30)       0.86
0.10 2   0.00 (0/30)       0.66 (24/30)       0.86
```

The plug-in ~doubles detection and at δ_Y=0.10 goes from total failure (0/30) to 24/30 -- the
estimator change is decisively the right lever. At the LIGHT config it reaches point power
0.50-0.80 (LCB 0.36-0.66), narrowing but not yet clearing the stringent LCB≥0.80 bar (needs
~28/30 at R=30). The full DEPLOYMENT config (256/600/5/3 -- more capacity, more folds = more data
per fit, restarts) is expected to close the remaining gap; that certification is running
(results/frontier_deploy). null FP stays 0/30. Full 11-file regression green with the plug-in as
the deployed decision (nested kept as a diagnostic).

## 4d. Phase 1.3.3 deployment-config certification (256/600/5/3, R=30) + 1.3.3a hygiene

Deployment-config plug-in vs oracle (boundary Δ*=δ_Y, bar = 80% certified-power LCB ≥ 0.80):
```
δ_Y=0.10 k=1 (23x1):  plug-in 28/30 (LCB 0.82)  oracle 29/30 (0.86)  null 0/30  -> BOTH_OK ✓
δ_Y=0.10 k=2 (22x2):  plug-in 27/30 (LCB 0.77)  oracle 29/30 (0.86)  null 0/30  -> borderline (1 short)
δ_Y=0.05 k=1:         (running)
```
The plug-in CLEARS the bar at δ_Y=0.10 k=1 (LCB 0.82, ~matching the oracle 0.86) -- from the
nested critic's 0/30. k=2 is one detection short (0.77). The estimator bottleneck is resolved at
δ_Y=0.10; δ_Y=0.05 (smaller, harder) pending.

Hygiene (Phase 1.3.3a): NO silent fold fallback (-> TASK_GATE_COVERAGE_FAILURE); the competence
table carries an estimator FINGERPRINT and the gate ABSTAINS on mismatch; outer-held-out NLL
recorded (q0/q1 final mixture). Name: "cross-fitted one-step log-ratio / efficiency-targeting"
(efficiency is conditional on classifier convergence), not unconditionally "efficient".

## 4e. Phase 1.3.4 — independent-seed decision-level certification (VERDICT: NOT certified)

Four disjoint seed groups, R=50, frozen deployment config (tag plugin-logratio-v1-cert-candidate),
exact-cell + scope lookup. Boundary effect Δ*=δ_Y; bar = one-sided Wilson LCB ≥ 0.80 (≥45/50).
```
G1 cert table (calib seeds 20000)  -- the table the gate USES:
  δ_Y=0.10 k=1: boundary 34/50 (LCB 0.57) -> MDE 0.13 -> power_ok=FALSE
  δ_Y=0.10 k=2: boundary 38/50 (LCB 0.65) -> MDE 0.13 -> power_ok=FALSE
G2 boundary confirmation (independent seeds 30000):
  δ_Y=0.10 k=1: plug-in 45/50 (LCB 0.81) PASS  | oracle 48/50 (LCB 0.89)
  δ_Y=0.10 k=2: plug-in 43/50 (LCB 0.76) fail  | oracle 48/50 (LCB 0.89)
  δ_Y=0.05 k=1: plug-in 46/50 (LCB 0.83) PASS  | oracle 48/50 (LCB 0.89)
```
**VERDICT: default-on NOT certified.** The seed groups DISAGREE at the primary cell (calib 34/50
LCB 0.57 vs confirm 45/50 LCB 0.81): the plug-in's TRUE boundary detection rate at δ_Y=0.10 is
≈0.80 -- right at the bar, which (at R=50) effectively needs a true rate ≈0.90 -- so it straddles
pass/fail across independent control-family draws. The earlier single-seed deploy result (28/30,
LCB 0.82) was a favorable draw; independent-seed confirmation (exactly the protocol's purpose)
exposes it as borderline. k=2 fails both groups. The ORACLE robustly clears (0.89) -> the
information is present with headroom; the residual gap is ESTIMATOR inefficiency, not n / δ_Y /
intrinsic. (Aside: the 2δ_Y=0.20 control target saturated at Δ*≈0.09 -- the control family's max
Δ* for these dims/base_sep is ~0.13; the boundary 0.10 target is reliable and is what decides
power_ok.) G3/G4 not run: the calib table is power_ok=False, so the gate abstains everywhere ->
trivially 0 UNSAFE_ACCEPT but degenerate; running them would not certify default-on.

Per the pre-registered protocol, the borderline/mixed primary is SEALED. Next = a stronger
estimator (cross-fit MODEL LIBRARY + convex STACKING) on a 3rd development seed set to push the
true detection rate ~0.80 -> ~0.90, then re-certify on a 4th fresh set. NOT generic AIPW. The
estimator+config remain FROZEN for these confirmatory seeds (no tuning on them).

## 4f. Phase 1.3.5 — stacked log-ratio (model library + convex stacking): does NOT clearly help

Development sweep (DEV seeds 70000, R=30, moderate config; plug-in vs STACKED at boundary Δ*=δ_Y):
```
δ_Y=0.10 k=2 (22x2):  plug-in 30/30 (0.92)  STACKED 28/30 (0.82)  oracle 30/30 (0.92)
δ_Y=0.10 k=1 (23x1):  plug-in 22/30 (0.59)  STACKED 24/30 (0.66)  oracle 28/30 (0.82)
```
Stacking does NOT clearly beat the plug-in: k=1 marginal (+2/30, both << bar), k=2 worse (-2/30).
The EM concentrates weight on the MLP-256 member (= the plug-in); the added linear/quad/MLP-64
learners are strictly weaker for this smooth nonlinear synergy, so the stack ~ plug-in. Per the
protocol (no clear dev improvement -> do NOT run the heavy fresh-seed cert) the stacked cert is
NOT launched.

CRUCIAL: the ORACLE itself is borderline on the hard k=1 DEV geometry (0.82) -- on some
control-family draws even a perfect detector barely clears, so the headroom for ANY estimator at
(δ_Y=0.10, n=6000, k=1) is small. Combined with the plug-in cert (true rate ~0.80, borderline),
this points to: at δ_Y=0.10, n=6000, the task-protected gate is SAFE-BUT-CONSERVATIVE and cannot
robustly default-on with this conditional-estimator family. Levers left (user-gated): a
fundamentally different conditional estimator (different inductive bias, not more MLPs); a larger-n
operating point; a task-justified δ_NI; or accept the honest negative (gate abstains -> safe).

## 5. Default-on gate (task_protect) — STAYS OFF (default-on NOT certified; stacking did not help)
The deployment cert is the estimator-direction proof (plug-in clears at δ_Y=0.10 k=1), NOT the
default-on certification. Per the pre-registered rules, flipping task_protect + task_power_floor
ON TOGETHER requires, on seeds/cells NOT used for estimator/config selection:
- pre-registered boundary cells meet the bar on INDEPENDENT seeds (held-out confirmation);
- oracle-injected unsafe grid -> ZERO UNSAFE_ACCEPT;
- factorized safe controls -> non-degenerate SAFE_ACCEPT;
- per-cell only (no extrapolation to uncovered dims / n / cluster structure);
- if k=2 / δ_Y=0.05 lag -> R=50 recheck or cross-fit model library + convex stacking (NOT
  generic AIPW).
Default-on exit conditions NOT met (zero UNSAFE_ACCEPT ✓, non-degenerate SAFE_ACCEPT ✗ at all
practical n). `task_protect` stays default OFF; `task_power_floor` stays opt-in.

The honest, AUDITED operating-regime conclusion: the power floor makes the selective method SAFE
(no unsafe acceptance observed) but, with δ_Y=0.03 and this critic, DEGENERATE at practical n —
certifying a 0.03-nat conditional-leakage deletion is information-theoretically sample-hungry
(needs n ≫ 24k). Levers, in order of leverage: (b) RELAX δ_Y to a task-justified non-inferiority
margin (the dominant lever — larger effects certify at far smaller n); (c) a lower-MDE estimator
(helps, but cannot beat the information floor for 0.03 nats); (a) accept a large-n-only regime;
(d) keep the floor opt-in and report the conservatism. δ_Y must be set by a task non-inferiority
argument, NOT to make the table pass.
