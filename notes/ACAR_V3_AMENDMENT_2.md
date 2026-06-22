# ACAR v3 — Amendment 2 (DEV-lock completion, DESIGN-ONLY, NON-BINDING)

**Date:** 2026-06-22 · **Status:** `NON-BINDING / NO DEV NUMERICAL RUN / NO LOCKBOX ENDPOINT ACCESSED / NOT TAGGED`
**Applies to:** `ACAR_V3_FREEZE_SKELETON.md` (updated in place), `ACAR_V3_DESIGN_DRAFT.md` (banner strengthened),
`acar/v3/set_features.py` + tests (hardened, commit `685a526`). Closes the eight DEV-lock statistical gaps + the
six set-contract blockers from the `93f417c`/skeleton review. Design + synthetic engineering only; touches no v2
executable code or result; v2 endpoint `1528a94` / tag `acar-v2-protocol` @ `9b2f0c1` unchanged. **This is the last
content needed before `acar-v3-dev-design-v1` can be tagged** — pending the C1/C2/C3 + `upper_bound()` + conformal
implementation and a green re-run of all DEV-design guards.

## A. Eight DEV-lock statistical gaps (now closed in the skeleton)

1. **C3 tail gate pinned pre-DEV** — positive-excess 95th pct ≤ **2.0 × (OOF ΔR SD per disease×action)**; no
   post-DEV fill (S2; removed from S11 TBD).
2. **C1 S2 eligibility defined** — C1 is selectable only if it passes the **raw-residual `max_a share ≤ 0.60`**
   dominance gate; else C1 is ablation-only. C0 stays comparator-only (diagnostic dominance) (S2).
3. **DEV OOF estimator written as ONE algorithm** — outer K subject-disjoint folds (EVAL); non-EVAL hash-split →
   FIT/CAL; FIT hash-split → TRAIN/VAL (early-stop only); predictor sees FIT, `q` sees CAL, S2/S4 aggregate on outer
   EVAL; fallback `<MIN_BATCH` retained→identity; three frozen seeds + K + ratios (S5).
4. **"all eligible DEV subjects, no residual/scale/candidate-based exclusion"** — refit wording fixed; eligibility is
   by the frozen split/inclusion rule only (S5).
5. **Harmful-rate = one executable statistic** — restrict to **router-adapted** batches; per-subject router vs
   best-fixed harmful rate; zero-adapt subjects excluded+reported; one-sided Wilcoxon signed-rank, `zero_method=
   "wilcox"`, ≥10 paired subjects else not_evaluable, exact for n<25, **Holm across sites**; **PASS = Holm p<0.05**,
   no secondary alternative (S6).
6. **C2 `q⁺=max(q,0)`** at deployment — removes uncertainty inversion in `U=μ+qσ`; coverage-safe (only raises U) (S1).
7. **Disease-specific models** — train PD and SCZ separately; two weights hashes; per-disease×action `σ_min` (S5/S9).
8. **S7 label-access wording fixed** — external **CAL** labels compute only `q`; external **EVAL** labels are
   invisible to the deployment path and read **once** at endpoint scoring (so G1/G2 are computable). Design-draft
   banner strengthened to mark V3.0/V3.8/V3.10 (and all normative rules) superseded by the skeleton.

## B. Six set-contract blockers (now implemented + tested, `685a526`; documented in skeleton §S13)

1. **`canonical_digest` = full 64-char SHA-256**, raw float64-LE + uint8 bytes, schema-versioned header, **no
   rounding** → single-ULP sensitive (was 32-char + 12-digit rounding).
2. **Canonical row order BEFORE adapters** → permutation invariance is byte-identical at the execution path (tested
   with `np.array_equal` on values/masks/context), not a hash tolerance.
3. **Canonical action execution order** (ACTION_VOCAB) + action-selection validation (unknown/dup/identity/empty).
4. **Validated + immutable `WindowActionSet`** (shapes, binary masks, masked-slots-exactly-0, finiteness,
   action_name/index consistency, unique keys; read-only arrays).
5. **Action capability map** asserted vs adapter (drift guard); **probability/shape validation**; fallback proven to
   make **no** adapter call (monkeypatch); missing-zero vs genuine-zero tested via the validated constructor.
6. **Structured `WindowKey`** `(dataset_id, subject_id, recording_id, window_index)` canonical serialization (the v3
   data loader must emit these; v2 `Batch` lacks per-row window identity).

15 synthetic guards pass; digest length confirmed 64. No DEV cohorts, no candidate/width/AUROC/router metrics, no
labels beyond the toy source state used to build the frozen SourceState.

## C. Remaining before `acar-v3-dev-design-v1` (still no DEV/lockbox access)

1. Implement `acar/v3/predictors.py` (C1 mean-Huber, C2 β-NLL mean/scale, C3 additive CQR) with a unified
   candidate-specific `upper_bound()` (incl. C2 `q⁺`), and `acar/v3/conformal.py` (subject joint score per candidate,
   per disease). DeepSets architecture/optimizer constants from S5 are fixed in code.
2. Add synthetic guards for predictors/conformal (set-invariance of μ̂/σ̂/U/action; CAL-vs-EVAL split isolation
   generalized to the standardized/CQR score; C2 q⁺ monotone-coverage; serialization).
3. Add the v3 data layer emitting structured `WindowKey` (derive from `recording_id`+`window_index`).
4. Re-run ALL DEV-design guards green, fix the architecture/seeds/failure-handling, THEN commit + light-tag
   `acar-v3-dev-design-v1` (the DEV_DESIGN_LOCK) on a clean worktree.
5. Only after the tag: run the first DEV numerical gate (S2 + S4). DEV no-pass = `DEV_STOP / NO_LOCKBOX_CONSUMED`.

Metadata-only, in parallel: source a second independent PD lockbox; verify ds007526 primary version/mapping/overlap;
clarify ASZED acquisition units + integrity flag (any payload check = separate content-blind pre-registration).
