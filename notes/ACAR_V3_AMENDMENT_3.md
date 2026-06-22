# ACAR v3 — Amendment 3 (predictors/conformal + residual hardening; DESIGN-ONLY, NON-BINDING)

**Date:** 2026-06-22 · **Status:** `NON-BINDING / NO DEV NUMERICAL RUN / NO LOCKBOX ENDPOINT ACCESSED / NOT TAGGED`
**Code:** commit `2303d5c` (`acar/v3/predictors.py`, `conformal.py`, hardened `set_features.py`, +tests).
**Docs:** `ACAR_V3_FREEZE_SKELETON.md` updated in place (S1/S5/S6/S13). v2 endpoint `1528a94` / tag
`acar-v2-protocol` @ `9b2f0c1` unchanged. Closes the six residual blockers from the `685a526` review and adds the
candidate predictors + conformal layer, all synthetic-only.

## A. Six residual blockers (closed in code `2303d5c`)

1. **Object-level immutability** — `WindowActionSet` is now `@dataclass(frozen=True, slots=True)`; field rebind
   raises `FrozenInstanceError` (previously only the array buffers were read-only). Guard added.
2. **Disambiguated `WindowKey` encoding** — structured `WindowKey`→`"WK"+canonical-JSON`, string→`"S"+s`, anything
   else→`TypeError`. No delimiter collision; toy strings stay toy-only.
3. **Immutable `FallbackBatchRecord`** — `<MIN_BATCH` returns `(forced_identity, reason, window_keys,
   canonical_input_digest[64], n_windows)`; actions/keys validated; **no adapter called** (guard).
4. **Identity reference computed once** — `build_action_sets` canonicalizes + computes identity once, then each
   action once; call-count guard asserts identity×1 + each action×1.
5. **C1/C3 negative-q rule frozen** — C1/C3 use raw additive `q` (NO clamp); only **C2** uses `q⁺=max(q,0)` (the
   sole candidate with the negative-q×scale uncertainty inversion). Guard.
6. **Harmful-rate test tie-aware** — drop zero diffs (recorded); ≥10 nonzero else `NOT_EVALUABLE`; exact Wilcoxon
   only when `|d|` distinct and n<25 (SciPy exact-null valid), else a deterministic sign-flip permutation
   (`seed=0, n_perm=20000`); all-zero ⇒ `NOT_EVALUABLE`. Guard.

## B. Candidate predictors + conformal (new, `2303d5c`)

- **`predictors.py`** — frozen `CandidatePrediction(candidate, disease, action, point, upper_center, scale)`;
  disease-specific DeepSets (pinned `HP`, S5); per-candidate `score()`/`upper_bound()` exactly per skeleton S1:
  C1 `U=μ̂+q, s=ΔR−μ̂`; C2 `U=μ̂+max(q,0)·max(σ̂,σmin), s=(ΔR−μ̂)/σ̃`; C3 `U=q̂₉₀+q, s=ΔR−q̂₉₀` with
  `q̂₉₀=q̂₅₀+softplus(d)+ε` (no crossing). Raw-ΔR-unit outputs via FIT-only target standardization.
- **`conformal.py`** — `subject_joint_score` (max over a subject's eligible batches × all non-identity actions;
  missing action raises); `conformal_rank` `k=⌈(m+1)(1−α)⌉` + `conformal_q` strict `+∞` when `k>m`; `route`
  (argmin `U<−δ` else identity, canonical ties); `harmful_rate_test` (A.6).
- **Guards (toy only):** 19 set-feature + 11 predictor/conformal — incl. prediction permutation/action invariance,
  mask-consumption, target-standardization round-trip, C2 scale>0/floor/`q⁺` + C1/C3 no-clamp, C3 no-crossing,
  conformal rank/`+∞`, joint-max + missing-action, CAL-vs-EVAL isolation (CAL→only q; EVAL→nothing), serialize
  round-trip + full-64 weights SHA-256, disease-tag propagation, tie-aware harmful test. ALL PASS.

## C. Skeleton pins added (S1/S5/S6/S13)

DeepSets/training HP block (S5); final all-DEV refit epoch = **median of outer folds' best-epoch, round half-up,
fixed in advance**; harmful-rate tie-aware rule (S6); C1/C3 no-clamp (S1); object-immutability + FallbackRecord +
identity-once + implemented predictors/conformal (S13).

## D. Remaining before `acar-v3-dev-design-v1` (still no DEV/lockbox)

1. **v3 data layer** emitting structured `WindowKey` (from `recording_id`+`window_index`; v2 `Batch` lacks per-row
   window identity) — synthetic/contract tests only.
2. **`develop.py`** wiring S5's split-as-ONE-algorithm (outer EVAL / non-EVAL→FIT/CAL / FIT→TRAIN/VAL; predictor=FIT,
   q=CAL, S2/S4 on outer EVAL; disease-specific; v2 replay) — as code, with synthetic guards; **no DEV cohort read**.
3. Pin SciPy version + env lock; finalize the `<<TBD-after-DEV>>` placeholders that are *procedural* (the *numeric*
   ones stay TBD until the gate runs).
4. Full green re-run of ALL DEV-design guards → commit + light-tag `acar-v3-dev-design-v1` on a clean worktree.
5. Only then: first DEV numerical gate (S2+S4). No-pass = `DEV_STOP / NO_LOCKBOX_CONSUMED`.

Metadata-only, in parallel: second PD lockbox; ds007526 primary verification; ASZED unit/integrity clarification
(any payload check = separate content-blind pre-registration).
