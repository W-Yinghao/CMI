# Freeze protocol — PRE-REGISTERED decision rules (fixed BEFORE results; 2026-06-21)

All thresholds/criteria below are set before seeing the corresponding results. No post-hoc invention.

## 0. Dependency chain (with the closed-loop pilot inserted before Freeze B)
```
r12 deterministic re-run → layered path-equivalence → continuous decomposition (cohort-level CIs)
→ selection stability → Freeze A (A1) → P1.5 saturation + collapse/utility decision
→ passive gate falsification → MINIMAL CLOSED-LOOP PILOT → Freeze B
→ full stress suite + adapter matrix → one confirmatory matrix → frozen-guard reruns → TUAB (one-time)
```

## 1. Δ_implementation naming (§4) — NOT "Δ_init" until audited
`Δ_implementation = g(enc,current) − g(enc,old)`. Call it **init effect only after** a code/config/artifact diff
confirms the ONLY difference between r8sel and current is the P2.2 init (else it bundles other changes). Diff:
source checkpoint, serialized state, preprocessing, candidate grid, batch order, metric impl, initialization. (No
GPU; a hash/diff check.) Until then: "Δ_implementation (dominant known change: P2.2 init)".

## 2. Equivalence semantics (§2) — units + interval, not point
`EPS_EQ_PP = 0.5` **percentage points** (bAcc stored on [0,1] ⇒ 0.005). Unit asserted in `r12_decomp.py`.
- **equivalent**: the whole cohort-level CI ⊂ [−0.5,+0.5].
- **inconclusive**: |point| ≤ 0.5 but CI extends outside the band (do NOT call equivalent).
- Uncertainty unit = **cohort** (aggregate seeds within cohort first); LOCO is the key small-n diagnostic; NO
  plain fold-bootstrap. Run-path tolerance is separate: **TOL_PATH = 1e-4 pp** (near machine precision), via
  per-config **prediction hash** (predictions, not bAcc).

## 3. Freeze A — branches LOCKED to avoid the LPC dependency conflict (§5)
Freeze A (A1) freezes the adaptation+accuracy **process** AND **both legal branches**: `CITA+LPC` and
`CITA-no-LPC`. P1.5 decides which enters Freeze B. If neither/other change is needed, emit **Freeze A2 explicitly**
(never silently modify A1). Frozen in A1: current-code+deterministic training, source checkpoints+serialized state,
outer/inner splits, candidate grid {0,0.1,0.3}, full-pipeline selection objective, target batch construction/order,
seeds, eval code + immutable schema + the fail-fast gate. **Magnitude accepted as-is (1.2/1.8/3.0).**

## 4. P1.5 retain/drop criteria for LPC (§5) — pre-registered, decided AFTER P1.5
Retain LPC into Freeze B **iff ALL hold**:
1. leakage reduction **persists on the validation-selected strong probe** (grouped split);
2. an **alternative-inductive-bias probe** (GBT/kNN) does NOT recover most of the leakage (held-out attacker);
3. task-utility loss ≤ **2.0 pp** held-out task-probe bAcc (pre-set band);
4. NOT collapse: effective-rank drop ≤ **15%**, between/within scatter drop ≤ **15%**, feature-var not ≈0.
Else drop LPC → Freeze B uses CITA-no-LPC. **Capacity-saturation STOP rule:** stop adding probe tiers when two
consecutive tiers improve held-out attacker leakage by < **δ=0.02** (KL) AND the improvement CI covers 0.

## 5. Falsification slice — leakage guards (§6), all pre-registered
- **Deployment-computable only:** every candidate implements `score_target(source_state, z_target, batch_metadata=None)`
  — **no `y_target` in the signature.** Allowed inputs: serialized source class moments, predicted classes,
  unlabeled target geometry, label-free clustering. (Target separability with true labels = post-hoc explanatory
  ONLY, not a controller candidate.)
- **No density self-inclusion:** `P(target|z)` scored leave-one-out / cross-fitted / on an independent reference
  subset (small-batch optimism guard).
- **Held-out by GENERATOR FAMILY** (not severity): leave-one-cohort-out × leave-one-generator-family-out.
- **Clustered CIs:** by batch/cohort/generator (sample harm is NOT independent across a shared-statistics batch).

## 6. Harm target (§7) — incremental loss primary
**Primary target = `Δℓ_i = ℓ_i^adapted − ℓ_i^base`** (continuous; captures confidence change). Binary
`1[correct→wrong]` = interpretive endpoint. Always record: harmful (C→W), beneficial (W→C), unchanged-correct,
unchanged-wrong, gate-PREVENTED harm, gate-LOST benefit. A harm-controller identifies **incremental adaptation
damage**, not just hard samples.

## 7. Gate degrees-of-freedom FROZEN before the slice (§8)
Per candidate, fix BEFORE results: formula; **direction (high score = high risk)**; normalization; sample→batch
aggregation; threshold calibration (dev cohorts); missing-value/fallback; whether a monotonic transform is allowed.
**No post-hoc direction flip.** Oracle-flipped `max(AUC,1−AUC)` may be REPORTED as a diagnostic but MUST NOT pick
the architecture.

## 8. Architecture decision table (§8) — pre-registered
| condition (held-out cohort × generator-family) | architecture |
|---|---|
| same frozen score+direction+aggregation predicts BOTH sample `Δℓ` AND batch `ΔL`, AND closed-loop pilot shows net benefit | **single deployable gate** |
| different scores stably carry batch-eligibility vs sample-abstention, and the COMBINATION beats either alone | **two-level controller** |
| prediction unstable across held-out, OR predicts but no net benefit after control | **diagnostic-only** (not a controller) |
`pure_conditional` (margin-uncorrelated relabel) is EXPECTED to fail for every label-free density score — this
**defines the identifiability boundary**, not an anomaly.

## 9. Minimal closed-loop pilot (§1) — REQUIRED before Freeze B
Passive falsification is necessary but NOT sufficient (screening can also remove beneficial samples / shift batch
stats). Before Freeze B, run a small pilot:
- 1 adapter (**CITA**); generators {clean, low-margin, high-margin, pure_conditional}; **2–3 natural batch sizes**;
- compare **{no-gate, pre-screen, post-abstain}**; thresholds + directions FROZEN from the development split.
- **Freeze-B condition:** the chosen architecture (a) predicts harm AND (b) **≥1 deployment action improves
  held-out net loss / selective risk** WITHOUT unacceptable clean harm (clean false-abstention ≤ pre-set bound,
  e.g. 5%). Else → diagnostic-only; do not claim a controller.

## 10. Invariants (unchanged)
TUAB sealed until the very end. Source-free deployment guard + serialized/online equivalence + no-target-label
guard re-run against the frozen state before TUAB. Unified baseline protocol (same checkpoint/batches/seeds).
