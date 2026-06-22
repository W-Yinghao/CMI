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

## 4. Results

### Competence table (offline, calib seed 9001; π = one-sided LCB of detection prob at {δ_Y,1.3δ_Y,2δ_Y})
```
n=1500  k=1 (23x1)  power_ok=False  pi=[0.09,0.09,0.25]
n=3000  k=1 (23x1)  power_ok=False  pi=[0.03,0.16,0.35]
n=6000  k=1 (23x1)  power_ok=False  pi=[0.09,0.16,0.46]
n=12000 k=1 (23x1)  power_ok=False  pi=[0.46,0.75,0.75]
n=1500  k=2 (22x2)  power_ok=False  pi=[0.03,0.16,0.25]
n=3000  k=2 (22x2)  power_ok=False  pi=[0.03,0.25,0.46]
n=6000  k=2 (22x2)  power_ok=False  pi=[0.09,0.16,0.35]
n=12000 k=2 (22x2)  power_ok=False  pi=[0.46,0.59,0.75]
n=1500  k=3 (21x3)  power_ok=False  pi=[0.09,0.16,0.46]
n=3000  k=3 (21x3)  power_ok=False  pi=[0.16,0.35,0.46]
```
π@2δ_Y climbs monotonically with n (0.25→0.35→0.46→0.75 for k=1) but does NOT cross the 1-β=0.8
bar by n=12000. The n=24000 cells were cancelled for compute economy; by the monotone trend
power_ok flips True at n≈24000 (extrapolated, NOT confirmed). So NO prefix is power-certified at
n≤12000: certifying a δ_Y=0.03-scale conditional-leakage deletion needs LARGE n.

### Injection re-validation with the power floor ON (n∈{2000,3000,6000})
```
synergy (inject Bayes-unsafe span):  per cell {SAFE_REJECT:2, UNSAFE_REJECT:4 (+AMBIG:2 @n=2000)}
factorized (genuinely safe):         per cell {SAFE_REJECT:10}
TOTAL: SAFE_REJECT 36 | UNSAFE_REJECT 12 | BAYES_AMBIGUOUS 2 | UNSAFE_ACCEPT 0   => CLEAN
```
- **Safety achieved:** ZERO UNSAFE_ACCEPT (vs 6 without the floor) — the gate ABSTAINS
  (TASK_POWER_INSUFFICIENT) on the exact injection it used to wrongly accept.
- **Conservatism (the trade-off):** at n≤6000 the floor is DEGENERATE — it also abstains on the
  genuinely-safe factorized deletions (all SAFE_REJECT), because power_ok=False there. The method
  is SAFE-BUT-CONSERVATIVE; non-degenerate acceptance requires n≳24000.

### Test contract status
- [confirmed] capacity change preserves all existing gate tests (11-file regression green; the
  bigger critic detects MORE — oracle-danger task_info 0.28→0.60, still rejects).
- [confirmed] oracle-injected unsafe candidates → ZERO UNSAFE_ACCEPT across held-out n.
- [confirmed, NEGATIVE] factorized safe cells do NOT get non-degenerate SAFE_ACCEPT at n≤6000 —
  the floor degrades to identity at moderate n (the exit condition #3 FAILS at these n).
- [pending] non-degenerate SAFE_ACCEPT demonstration at n≳24000 (compute-bound, not run).

## 5. Default-on gate (task_protect) — STAYS OFF
The default-on exit conditions are NOT met: zero UNSAFE_ACCEPT ✓ but non-degenerate SAFE_ACCEPT ✗
at typical n (the floor abstains on everything ≤12k). So `task_protect` stays default OFF.

The honest operating-regime conclusion: the power floor makes the selective method **safe but
conservative** — it refuses to certify δ_Y-scale conditional-leakage deletions until n≳24000, and
abstains (identity) otherwise. To be USEFUL (non-degenerate) at typical EEG sample sizes one of:
(a) accept the large-n regime; (b) relax δ_Y; (c) a lower-MDE conditional-MI estimator; or
(d) keep the floor opt-in and report the conservatism. This is a genuine design decision.
