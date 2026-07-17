# Mechanism-Subspace Oracle — Stage-C real-EEG ENGINEERING smoke: findings (NO scientific weight)

Smoke = 2 datasets × 2 backbones × 2 held-out subjects × seed 0 × 4 candidate families; random ambient 2×8,
shared-overlap pool 200. Engineering-only (n=2 → degenerate small-n bootstrap; NO scientific verdict). The smoke
surfaced defects that must be resolved BEFORE the full M1 (126-cell) run. **This note was CORRECTED after a
4-agent adversarial verification panel (V1/V2/V3 + adjudicator, run wf_964b74d4); the first draft overstated
Blocker 2 and proposed a fix (total-energy matching) that the panel FALSIFIED.** Contract-level items are the
PM's decision (amendment 03). Manuscript FROZEN. Only the project owner may explicitly stop a scientific line.

## What ran cleanly (EEGNet primary confirmatory path)
16/16 EEGNet cells executed end-to-end on real EEG for BOTH datasets, all 4 families: source-only construction →
exhaustive rank≤3 selection (unconstrained + source-LOSO-safe) → session-macro T_query outcome → ambient random +
shared-overlap-matched controls → firewall + audit + provenance. Aggregator produced tri-state verdicts, Holm on
the confirmatory family, q95-exceedance, reason-coded skips. The aggregator STATISTICS were independently
reproduced and verified correct by the panel (Holm, cluster bootstrap LCB/UCB, one-sided p, tri-state, AMBIENT_ONLY
flag, no CLOSED, no silent ambient substitution). No scientific weight (n=2).

## BLOCKER 1 — DGCNN secondary path has no session axis (data-artifact gap) — STANDS
DGCNN forward-graph dumps (`results/cmi_trace_p0p1/objective_comparison/*/audit/*erm*seed*.audit.npz`) carry
`Z_source/Z_target/subj_*/outer_fold` but **no `session_target`** (0/63). The oracle's evaluation (P0.7 step 3) is
a session-macro T_query split, so DGCNN cannot be evaluated from existing artifacts. The runner now REASON-CODES
this (`NO_SESSION_AXIS_FOR_QUERY_SPLIT`, 4/4 DGCNN smoke cells) instead of silently dropping cells (the earlier
glob produced zero DGCNN rows — a false "absent"). Blocks only the SECONDARY DGCNN route (BACKBONE_SPECIFIC, which
per P0.6 cannot unlock M2 without a confirmatory rerun). EEGNet confirmatory PRIMARY unaffected.
PM options: (a) re-infer the frozen DGCNN adapter persisting session metadata (genuine re-inference); (b) approve a
session-free target split for the DGCNN secondary only (contract touch); (c) M1 EEGNet-primary-only, defer DGCNN.

## BLOCKER 2 — P0.5 shared-overlap-matched control fails on real cells — REAL, but characterization CORRECTED
**Structural fact (verified, keep):** the mechanism dictionary B (rank-8 generalized eigenbasis of
`G_dis v = ρ(G_shared+ηI)v`) has per-direction shared-overlap `q_j = b_jᵀ G_shared b_j / tr(G_shared) ≈ 0` for ALL
directions — because `rank(G_shared)=C−1` (3 for the 4-class BNCI2014, 1 for the 2-class BNCI2015) and D=16, so B
is forced into G_shared's rank-(C−1) NULL space (avg ‖P_null b_j‖²≈1.0). The generalized eig extremizes
shared-overlap toward 0 by construction. This is a real property, not a bug.

**CORRECTIONS to the first draft (panel-caught, load-bearing):**
- The claim "no Haar rank-8 dict at ANY pool reaches rmse≤0.02" is **FALSE for the 2-class BNCI2015 cells** — a
  single dict reaches 0.003–0.008 at pool ≈2e5. The "plateaus far above 0.02" text was an artifact of stopping the
  sweep at pool 5000. What actually matters is the code's GATE METRIC (`matching_rmse` over the kept-`n_random`
  dictionaries, plus the `gap≤0.01` gate), not a single best dict.
- Matching the SELECTED rank≤3 deletion subspace (the object actually deleted at score time) instead of the full
  rank-8 dictionary is a within-spirit repair: it yields `verdict=OK` for BNCI2015 and passes the rmse gate
  everywhere; the ONLY residual obstruction for the 4-class BNCI2014 (the confirmatory PRIMARY) is the `gap≤0.01`
  gate (0.014–0.025), which is a **tunable contract pin**, not a proof of impossibility. (Caveat: that repair
  selects S using Y_cal → firewall reading needs PM judgement.)
- Net: as FROZEN, P0.5 fail-closes on all real cells → without a fix M1's confirmatory gate would silently fall
  back to AMBIENT_ONLY. But the failure is dataset-split (2-class repairable, 4-class blocked at the gap gate), not
  a universal impossibility.

**The total-energy fix I proposed is FALSIFIED — do NOT adopt.** Matching the total task-contrast energy
`t_j = b_jᵀ(G_shared+G_dis)b_j/tr(·)` is correctly firewalled but fails as a specificity control:
(1) WRONG INVARIANT — it confounds deployable shared content `G_shared` (source-LOSO deletion HURTS +0.09…+0.13
    bAcc) with subject-disagreement `G_dis` (deletion is SAFE, ≈−0.005). B carries **0% shared**; the total-energy
    matched control carries **27–48% shared** → the two arms are NOT equally task-bearing, so ENRICHED would be
    confounded.
(2) DEGENERATE in D=16 — an 8-number profile barely constrains an 8-dim subspace, so the matched control collapses
    onto plain ambient-random (subspace overlap with B 0.50–0.53, at the geometric floor 0.495; oracle dU
    indistinguishable from ambient) → adds no specificity the ambient control already lacks.
(3) MISREPORTED FEASIBILITY — my "rmse 0.003–0.006 @5000" was the best-single-dict distance (dist_min/√r), NOT the
    code's gate metric. The real gate metric for total-energy is 0.012–0.018 at n_keep=10 and **0.019–0.031 at
    n_keep=100** (the powered setting) → FAILS the 0.02 gate on 3/4 cells.

### The real decision (ESTIMAND FORK — PM adjudicates for amendment 03)
Both skeptics converge on using **G_shared's null structure** for the control but disagree on whether that is
"exactly right" or "degenerate":
- **Candidate control (V2, firewall-clean, feasible):** Haar-random WITHIN G_shared's numerical null space →
  `q≈0` by construction (matches B on shared-overlap, no plateau), still captures 0.37–0.45 of `tr(G_dis)` (NOT
  task-null), source-LOSO-safe like B. Uses only source `G_shared` — no target outcome.
- **V1 caution:** any shared-null-matched control is degenerate — it forces the random control into the SAME
  zero-shared region as B, so the control is confounded with the property under test.
These are the same geometric region viewed two ways. **The fork the PM must pick:** is the scientific question
*"is B's alignment to G_dis special vs. a random LOSO-safe (shared-null) subspace?"* (then V2's control is correct)
— OR must the control be matched on deployable/shared content while differing from B in shared composition (then
shared-null is degenerate and neither shared-null nor total-energy works, and the estimand itself needs
restatement)? Amendment 03 should (i) resolve this estimand, (ii) pre-register a null-rank threshold for the
numerical null space, (iii) report the code GATE metric at the powered n_keep=100 (not best-single-dict).

## Code-correctness fixes made this pass (NOT contract changes)
- **FAIL-OPEN routing gated** (was load-bearing): `route_stage_result` now takes `specificity_control`; if the
  stats clear route A but only AMBIENT_ONLY ran, verdict = `ENRICHED_VS_AMBIENT_ONLY_MATCHED_CONTROL_UNAVAILABLE`
  (does NOT unlock M2). Aggregator passes MATCHED only when the per-cell matched vector exists.
- **n<2 cluster guard**: single-subject cells → `INSUFFICIENT_CLUSTERS` (no spurious p~1e-4 / LCB=UCB).
- **Unexpected-status surfacing**: any row whose status is neither ok nor skipped is reported, never dropped.
- **Aggregator unit tests added** (`test_mechanism_aggregator.py`: _holm, _cluster_ci one-sided p, _cell_specific
  fail-closed matched-vs-ambient) + routing-gate test. 22 mechanism tests green.

## Disposition
M1 (126 cells) remains HOLD pending PM decisions on BLOCKER 1 (DGCNN session axis) and BLOCKER 2 (the estimand fork
+ amendment 03 control redefinition). Stage-B code fail-closed CORRECTLY and, together with the adversarial panel,
surfaced both the structural control problem AND the flaws in my own first-draft fix — no result was fabricated or
silently degraded.
