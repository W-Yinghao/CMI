# Target-X observability audit — PRE-REG AMENDMENT 02 (F2.0d full-run readiness; supersedes conflicting parts)

Transparent amendment to `TARGETX_OBSERVABILITY_PREREG.md` + `_AMENDMENT_01.md` per PM code review of the
full runner. Fixes four P0 bugs that would change the primary result, plus formula/definition corrections.
Branch `agent/cmi-trace-targetx-observability`. BLOCKING for the F2.0d engineering smoke. Manuscript FROZEN.

## B1 — informed actions vs controls are SEPARATED; random control = ambient same-rank projector (P0.1)
The old `eligible_actions` drew random subsets from the candidate basis and deduped by `frozenset(S)`, so every
random action collapsed into an already-present informed subset (delta_random_mean ≡ 0). FIX:
- An action is `{name, kind ∈ {informed, random, baseline}, rank, dirs (orthonormal [k,D] to delete) | apply_fn,
  eligible_for_selection}`.
- **informed** (eligible): identity; source-fitted-basis singletons; all rank-≤3 subsets; source-greedy prefixes.
- **random** (NOT eligible): AMBIENT same-rank random orthonormal projectors `P = Q_k Q_kᵀ` in the full latent
  space (NOT coordinate subsets of the candidate basis), **≥50 per informed rank present**. Comparators only.
- **baseline** (NOT eligible): whitening, target-mean-centering.
- Only `eligible_for_selection = True` actions can be chosen by the target-X selector.

## B2 — G1 selector gets a task-safety gate + a random-specificity gate (P0.2)
G1(S)=‖P_S d‖² (d=μ_s−μ_{t,cal}) is ≥0 and monotone non-decreasing in added orthogonal directions, so an
un-gated argmax always picks the maximum allowed rank and identity never triggers. FIX — S_TX = argmax G1 over
the SAFE & SPECIFIC informed set A, else identity, where S∈A iff BOTH:
1. **source task-safety**: source-LOSO held-out bAcc drop from deleting S ≤ 0.02 (i.e. gain ≥ −0.02).
2. **random-specificity**: G1(S) > Q₀.₉₅ over the ≥50 ambient same-rank random projectors of that rank.
Observability ρ is reported PER RANK and as a rank-stratified / rank-partial correlation (pooled ρ may be a
rank artifact and is secondary).

## B3 — BNCI2015 outcome is session-MACRO (P0.3)
`utility` takes `session_query` and returns BOTH:
- `delta_query_session_macro` (PRIMARY) = mean over query sessions of [bAcc_session(delete S) − bAcc_session(∅)];
- `delta_query_pooled` (SENSITIVITY) = pooled-trial bAcc gain.
Subjects with a single query session: macro = that session. BNCI2014 (single query session `1test`): macro=pooled.

## B4 — full runner PRESERVES all per-action rows + emits the analysis artifacts (P0.4)
Save every action's scores + utilities. Emitted artifacts:
`targetx_action_rows.jsonl, targetx_fold_summary.csv, targetx_cluster_intervals.csv,
targetx_observability_by_rank.csv, targetx_negative_controls.csv, targetx_gate_verdict.json,
targetx_completeness_matrix.csv`. The five-gate verdict is deterministic from these.

## B5 — observable-suite corrections
- **G2 is REMOVED from the statistical/Holm family** (P1.2): on an orthonormal basis G2(S)=‖P_S d‖²≡G1(S); it is
  kept only as an algebraic sanity check, never independent evidence. Secondary family is 10 observables
  (G3,G4,G5,P1,P2,P3,P4,C1,C2,C3), implemented and run ONLY after the G1 primary result is frozen (F2.2).
- **G5** (P1.3): condition number computed in the RETAINED (complement) subspace `Q_⊥ᵀ Z`, not the ambient space
  (where deleted directions create ~0 eigenvalues → mechanical blow-up). Effective-rank stays a separate
  secondary diagnostic.
- **C3** (P1.3): source AND target pseudo-task contrasts use the SAME post-deletion transform, v⁻=(I−P_S)v.

## B6 — certification (F2.0b) details to fix before Gate 5 (do NOT change the NOT-CERTIFIED verdict)
- The `linear` capacity is a `hidden_dim=8` neural posterior → renamed **`tiny_mlp`** (a true linear logistic
  posterior may be added later); no capacity is described as "linear" until then.
- Random controls are matched to EACH split-specific ticket's EXACT rank (not the average `kbar`), paired per split.
- An empty split-specific ticket stays **identity** (never force-delete direction 0).
- Gate 5 (in F2.1) restores the FULLY-RETRAINED within-label permutation null; the current paired raw-KL
  difference is a fast screening statistic, not a null-calibrated ruler, and is labelled as such.

## B7 — required implementation tests (synthetic, engineering only)
`test_random_controls_survive_dedup, test_random_controls_not_selectable, test_random_rank_matches_informed_rank,
test_g1_selector_has_task_safety_gate, test_g1_identity_fallback_against_random_null,
test_bnci2015_session_macro_not_pooled, test_full_runner_preserves_action_rows, test_rank_stratified_observability,
test_gate_verdict_deterministic`.

## B8 — scope discipline
Synthetic + smoke are for gradients/indices/data-firewall/numerical-stability/implementation-consistency ONLY.
No method GO/NO-GO from any synthetic/smoke/single-fold/single-seed result. Full F2, adaptation, TTE, and
manuscript remain HELD/FROZEN. F2.0d ends at a 2-subject/dataset engineering smoke reported for PM review.
