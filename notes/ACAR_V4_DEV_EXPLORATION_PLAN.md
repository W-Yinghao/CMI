# ACAR v4 — DEV EXPLORATION PLAN (Phase 1)

```
STATUS         : NON-BINDING
LINEAGE        : POST-V3 DEV_STOP
LOCKBOX        : NO LOCKBOX ACCESS
EXTERNAL ARM   : NO EXTERNAL ARM (Arm B not authorized)
THIS FILE      : NOT ACAR_FROZEN_v4.md  (exploration plan only; freezes nothing, selects nothing)
DATE           : 2026-06-29
```

Companion to `notes/ACAR_V4_DESIGN_DRAFT.md`. This is the **Phase-1 development** plan: how directions A (policy-risk
budgeting), B (hierarchical calibration), and C (information frontiers) are run on the **seven DEV cohorts only**, what
each emits, and the order of implementation. Everything here is **exploratory / model-selection**. No held-out cohort,
no lockbox, no external Arm B. v4 never edits v3.

---

## 0. Phases

```
Phase 0  Design draft (this + the draft); NO numbers run.                              ← current
Phase 1  DEV exploration on seven cohorts; A, B, C may run in parallel; exploratory.
Phase 2  v4 DEV-design lock (ONLY if a candidate passes G0–G6): freeze + tag.
Phase 3  External Arm B (ONLY after Phase 2; separately authorized; re-audited cohorts).
```

Phase 1 may begin only after the design draft is committed. Phase 1 produces **no** external evidence — its sole
outputs are exploratory frontiers and candidate diagnostics that decide whether Phase 2 is even attempted.

---

## 1. Shared substrate (reused from v3, unchanged)

- **Splits.** v3 subject-disjoint outer folds (K=5, `seed_outer=0`) → FIT/CAL (`seed_fitcal=1`, 0.70) → TRAIN/VAL
  (`seed_es=2`, 0.80); canonical-SubjectKey permutation-independent hashing. A/B/C share identical cells.
- **Execution cache.** v3 single-execution `BatchActionExecutionRecord` / `disease_exec_cache`: per-batch per-action
  embeddings and `ΔR_a(B)` computed **once**, shared across A/B/C. No re-fit, no re-leak.
- **Features.** Bit-for-bit v2 `acar.features.feature_vector` (11-D paired+context) is the shared label-free observable
  for harm/benefit scores; v4 may add *derived* score heads on top but does not change the captured features.
- **Metrics.** Subject-macro (subject-equal-weighted) red / coverage / harm-rate / center-AUROC; fallback identity
  batches in all denominators; reduction `red(π) = −mean_B ΔR_{π(B)}(B)`. **Contract (enforced in `frontiers.py` /
  `policies.py`):** every accounting/frontier primitive takes `weights=None`; weights=None is the batch-level primitive
  (NOT a DEV conclusion), and the DEV path passes `subject_macro_weights(subject_ids)` = `1/(n_subjects·n_batches_of_s)`
  so coverage/red/harm are subject-equal weighted means with fallback rows kept in the weighted denominator. All
  primitives are **fail-closed**: non-finite values, empty / zero-action arrays, out-of-range `choice`/`action_idx`, and
  non-finite / non-positive-sum weights raise — nothing is silently coerced.
- **C0 / v2 replay comparator.** The v2 recipe (ActionRegressor HGB≥40 / Ridge≥8 / constant, seed 0) reproduced as the
  baseline reference point, exactly as in v3.

---

## 2. Direction A outputs (policy-risk budgeting)

For each disease and disease-macro, over a finite `λ` grid:

```
coverage(λ)          adaptation coverage of π_λ (non-identity fraction; fallback in denominator)
red(λ)               deployed NLL reduction −mean ΔR_{π_λ(B)}
harm_rate(λ)         P(ΔR_{π_λ(B)} > 0 | π_λ(B) ≠ identity)
CAL_risk(λ)          chosen subject-level CAL loss ℓ ∈ {mean, pos, harm}
λ*                   most aggressive λ whose CAL risk passes the budget (LTT/RCPS-style grid selection)
```

Policy families compared (`acar/v4/policies.py`): safe-set (primary, A1), benefit-ranked, direct-selective. Calibration
candidates (`acar/v4/risk_control.py`): `select_ltt_grid` over a FIXED finite λ grid with subject-level CAL losses
(`subject_losses_from_policy`, losses ∈ {mean, positive, harm_indicator}; fallback identity rows realized 0, kept in
the subject denominator). One-sided risk test of `H0: E[L] ≥ budget` (method ∈ {ttest, hoeffding}), an FWER correction
(Holm / Bonferroni), and selection of the **most aggressive PASSING λ** (`passer = adjusted_p ≤ alpha`). The λ-risk
curve may be **non-monotone**, so NO monotone conformal-risk-control theorem is claimed — the FIXED grid + multiple
testing is the guarantee; the primary method is fixed only at a future v4 freeze. `< 2` CAL subjects ⇒ `NOT_EVALUABLE`;
no passing λ ⇒ `NO_PASS`; malformed inputs RAISE. Output: per-family `(coverage, red, harm)` operating points + the
selected `λ*` per loss. **Exploratory only.**

## 3. Direction B outputs (hierarchical calibration)

The three calibration OBJECTS (`acar/v4/hierarchy.py`, returning a per-subject `SubjectRisk`) on the **same** policy
family:

```
B0  all_action_joint_max     v3-style Z_s = max_{B} max_a score_{B,a}   (legacy comparator; responds to ALL actions)
B1  policy_subject_risk       Z_s = mean_B ℓ(ΔR_{π(B)}, π(B))            (executed policy only; ignores unexecuted)
B2  hierarchical_policy_risk  batch-level ℓ first → subject summary       (executed policy only; two-stage extension pt)
```

The decisive invariant (guarded): **B0 responds to a change in an UNEXECUTED action's risk; B1/B2 do not** — that is
V4-B's object difference from v3 (calibrate the deployed policy's risk, not the all-action simultaneous risk). `loss ∈
{mean, positive, harm_indicator}`; identity/fallback realize 0 and stay in the subject denominator; subject-equal,
canonical/permutation-independent order. The DEV question is whether calibrating B1/B2 (the executed-policy object)
moves the calibrated point materially up the frontier vs B0 (the v3 object). NOT a coverage theorem.

## 4. Direction C outputs (information frontiers — implement FIRST)

`acar/v4/frontiers.py`, all on subject-disjoint OOF cells:

```
F_true_oracle         (coverage → red, + harm)   true ΔR for batch & action selection (global-max ceiling)
F_single_score_oracle (coverage → red, + harm)   ONE label-free score ranking + action, best hindsight coverage
F_score_oracle (union)(coverage → red, + harm)   UPPER ENVELOPE over a pre-listed set of single-score frontiers =
                                                 the information ceiling of those observables (one score ≠ ceiling)
F_policy_family       (coverage → red, + harm)   v4-A/B π_λ family operating points on OOF (+ Pareto upper envelope)
F_calibrated          discrete points            calibrated frozen policies; v2 router + v3 C1/C2/C3 as references
```

Plus the **gap decomposition** per disease, with TWO parallel (never-mixed) outputs:

```
mode="ceiling"  (MAIN diagnostic — exact telescoping; answers "can any operating point work?")
  info_gap        = ceiling(F_true_oracle)  − ceiling(F_score_oracle_union)      (guaranteed ≥ 0)
  policy_gap      = ceiling(F_score_union)   − ceiling(F_policy_family)            (signed)
  calibration_gap = ceiling(F_policy_family) − red(F_calibrated)                   (signed)
  info_gap + policy_gap + calibration_gap = ceiling(F_true_oracle) − red(F_calibrated)   [exact]

mode="auc"      (DESCRIPTIVE only — NOT pass/fail; never replaces the ceiling diagnostic)
  area-under-frontier (over coverage, on the Pareto envelope) for true / score-union / policy, and the auc info/policy
  gaps. The calibrated point has no area, so calibration stays the ceiling reference.
```

**Implement Direction C first** — the frontier audit most quickly answers whether v3's failure was information-,
policy-, or calibration-limited, which determines whether A/B are even worth pushing.

---

## 5. Implementation order

```
1. acar/v4/frontiers.py   + synthetic guards   ← FIRST (answers the why)
2. acar/v4/policies.py    + synthetic guards       (π_λ families feed F_policy_family)
3. acar/v4/risk_control.py + synthetic guards      (LTT/RCPS λ* selection)
4. acar/v4/hierarchy.py   + synthetic guards       (B1/B2 subject-level calibration)
5. acar/v4/develop.py                              (DEV orchestration over the v3 cache; exploratory)
6. acar/v4/manifest.py                             (provenance + hashes; reuse v3 loader)
```

Each module ships with synthetic-fixture guards (no real data) before any DEV orchestration. The first two commits are
synthetic-only: they read no real cohort, select nothing, and freeze nothing.

---

## 6. Candidate gate (recap; full criteria in the design draft §8)

```
G0 provenance/leakage/split guards pass    G4 harmful adapted-batch rate within CAL risk rule
G1 per-disease coverage ≥ 0.15             G5 fallback identity in denominator
G2 per-disease red > 0                     G6 PD and SCZ both non-vacuous
G3 disease-macro red > C0/v2 replay
```

Direction-C frontiers are explanatory, not pass/fail.

---

## 7. Stop rules

```
No candidate passes G0–G6      ⇒  V4-DEV-NEGATIVE / NO LOCKBOX CONSUMED  (+ frontier gap decomposition)
A candidate passes G0–G6       ⇒  proceed to Phase 2 (write ACAR_FROZEN_v4.md, freeze + tag); only THEN Arm B
Killed / partial run           ⇒  operationally aborted (NOT a scientific verdict)
```

No threshold / seed / `λ` / loss / candidate search after a DEV read. The old seven cohorts cannot be written up as
external validation. Any continuation past v4 is a NEW dated, separately-tagged protocol.
