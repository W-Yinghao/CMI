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
