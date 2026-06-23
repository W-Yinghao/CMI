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

## 5. Default-on gate (task_protect) — STAYS OFF
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
