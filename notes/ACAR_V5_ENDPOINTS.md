# ACAR V5 — Endpoints, Gates & Statistical Control **(DRAFT — UNTAGGED — NON-BINDING; pinned at sign-off)**

Companion to `ACAR_FROZEN_v5.md`, `ACAR_V5_CANDIDATE_SPACE.md`, `ACAR_V5_SPLITS.md`. **No runs authorized by this draft.**
Subject is the statistical unit everywhere. All quantities are subject-macro per disease unless stated. "v2_replay" = the
bit-for-bit v2-recipe comparator recomputed on the SAME pool/substrate (as in v3/v4; on the v4 regenerated substrate its red was
≈ 0 / −0.0008 — so a utility margin must clear real signal, not noise).

## 1. Primary gates G1–G6 (ALL must hold, per disease, to be eligible)
- **G1 — coverage floor.** `LCB[coverage_disease] ≥ 0.15` (subject-clustered lower bound). Macro coverage ≥ 0.20 PREFERRED (soft,
  reported, not a hard fail). `coverage = n_adapted_batches / n_eval_batches` with fallback/identity batches IN the denominator.
- **G2 — deployed utility (PRIMARY = effect size, NOT CI-only).** Per disease: `red_disease > 0` AND
  `red_disease − v2_replay_red_disease ≥ 0.02 NLL`; AND macro `red − v2_replay_red ≥ 0.02 NLL`. **ε = 0.02 NLL** (pinned). The
  point-estimate margin is the pass; CI is NOT the primary gate (avoids turning G2 into a power test). `red = −mean_subject(chosen
  ΔR over ALL eval batches)` (fallback/identity batches included; identity ΔR = 0).
  - *Supporting (reported, not a separate hard gate):* the subject-clustered lower bound of `(red − v2_replay)`. A point estimate
    that clears 0.02 but whose robustness (G6) fails across seeds/substrates is treated as noise — **G6 is what guards against
    noise-gaming, not a utility CI co-gate.**
  - *Near-miss descriptor (DESCRIPTIVE ONLY, never a pass):* `0.01 ≤ (red − v2_replay) < 0.02`.
- **G3 — all-batch harm.** `UCB[L_harm_all_disease] ≤ 0.10` (LTT-certified upper bound). `L_harm_all` = subject-mean harmful-risk
  over ALL eval batches (non-adapted batches contribute 0).
- **G4 — conditional adapted-harm (THE new gate v4 lacked).** `UCB[harm_among_adapted_disease] ≤ 0.30` (LTT-certified upper
  bound over the ADAPTED batches only). This is what makes "adapt rarely but badly" FAIL even when G3 passes — exactly the v4
  failure mode (v4 replay: adapted-harm 0.73 PD / 1.00 SCZ behind 2–5% coverage; see `notes/ACAR_V4_CLOSEOUT.md`).
- **G5 — benefit retention.** `red(policy) ≥ 0.25 × red(upper-envelope)` (oracle / score-union envelope) OR
  `red(policy) ≥ red(best-fixed-action-with-abstention)`. Guards against a router that passes G2's floor but captures a trivial
  slice of the achievable benefit.
- **G6 — robustness (BUILT-IN, not post-hoc).** G1–G5 must ALL hold on EVERY pre-registered substrate/seed/cohort stress test
  S1–S3 (`ACAR_V5_SPLITS.md` §Robustness). A candidate that passes on one regenerated substrate but not across S1–S3 is NOT
  eligible. This is the gate v4 never had as a precondition.

## 2. Statistical certification (LTT for safety/coverage; effect-size for utility)
Per (candidate, disease) the following are tested; safety/coverage use one-sided CIs (that is what LTT is for), utility uses the
effect-size gate G2 plus the robustness guard G6:
```
H1 (G3): UCB[L_harm_all]          ≤ 0.10
H2 (G4): UCB[harm_among_adapted]  ≤ 0.30
H3 (G1): LCB[coverage]            ≥ 0.15
H4 (G2): point[red − v2_replay]   ≥ 0.02   (effect size; report LCB[red − v2_replay] as supporting)
```
- **Estimators (subject is the cluster):** subject-level empirical-Bernstein bounds for bounded losses (harm in [0,1]); or
  subject cluster-bootstrap / permutation (sign-flip) for `red − v2_replay`. The exact estimator + α are pinned at sign-off; do
  NOT switch estimators after seeing results. **No batch-level p-values** (v2/v3 discipline).
- **Multiple testing:** Holm correction across (candidate × disease × hypothesis) within the pre-registered space; the LTT
  family-wise level is pinned at sign-off (proposed α = 0.05).
- **λ / threshold certification:** for grid families, a config is eligible only if H1–H3 hold at its operating point AND G2's
  effect-size margin holds; the finite-grid search itself is covered by the multiple-testing correction.

## 3. Selection objective (Stage-2)
Among G1–G5-eligible configs, select:
```
maximize   min_disease ( red_disease − v2_replay_red_disease )
subject to LCB[coverage_disease]        ≥ 0.15      (G1)
           UCB[L_harm_all_disease]       ≤ 0.10      (G3)
           UCB[harm_among_adapted_disease] ≤ 0.30    (G4)
           red_disease − v2_replay_red_disease ≥ 0.02 both diseases (G2)
           G5 benefit retention
```
Tie-break (deterministic, pre-registered): lower `harm_among_adapted` → higher coverage → more conservative family
(P3 ≺ P1/P2/P4 ≺ P5). The winner is FIXED before Stage-4/5 (no reselection).

## 4. Model form (objective correction vs v4)
V5 does NOT frame the primary task as "fit a regressor for ΔR_a and conformalize it" (v3 set-conformal collapsed; v4 direct-policy
risk transferred poorly). V5 selects, from the bounded policy class, the abstaining router that maximizes the min-disease utility
margin under the safety/coverage/robustness constraints above. ΔR/feature regressions, if used inside a family, are means to the
policy, not the certified endpoint.

## 5. Outcomes & stop rules
- **PASS (rare, the high bar):** some candidate satisfies G1–G6 on DEV + S1–S3 → eligible for Stage-5 external (ONCE, single-site
  per disease). External report = "single-site held-out confirmation", not cross-site generalization.
- **DEV_STOP / NO_LOCKBOX_CONSUMED:** no candidate satisfies G1–G6 → STOP, no external read, no tag. (Expected baseline outcome.)
- **SAFE-BUT-USELESS is NOT a pass:** safety/coverage certified but G2/G5 utility not met → FAIL (not a partial pass).
- **NEGATIVE RESULT (pre-committed):** if even the most conservative family (P3) cannot hold G2 utility under G6 robustness, record
  the honest negative conclusion (label-free adaptation-risk features measurable but insufficient for stable deployment-utility
  control here). No tuning to escape it.
- **No post-replay / post-external tuning** of candidate / score / policy / loss / λ-grid / comparator / thresholds. Continuation =
  a NEW dated protocol.
