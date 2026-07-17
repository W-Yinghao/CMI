# Mechanism-Subspace Oracle — Stage-C real-EEG ENGINEERING smoke: findings (NO scientific weight)

Smoke = 2 datasets × 2 backbones × 2 held-out subjects × seed 0 × 4 candidate families; random ambient 2×8,
shared-overlap pool 200. Engineering-only (n=2 → degenerate small-n bootstrap; NO scientific verdict). The smoke
did its job: it surfaced two blockers that must be resolved BEFORE the full M1 (126-cell) run. Both are
contract-level (P0.4 pipeline / P0.5 control) questions — only the project owner/PM may change the frozen
contract. Manuscript FROZEN. Only the project owner may explicitly stop a scientific line.

## What ran cleanly (EEGNet primary confirmatory path)
16/16 EEGNet cells executed end-to-end on real EEG for BOTH datasets, all 4 families: source-only construction →
exhaustive rank≤3 selection (unconstrained + source-LOSO-safe) → session-macro T_query outcome → ambient random
controls → shared-overlap-matched control (fail-closed) → firewall + audit fields + provenance hashes. The
aggregator produced tri-state verdicts (significance / interval / practical), Holm across the confirmatory family,
q95-exceedance vs control, reason-coded skips surfaced. No scientific weight (n=2), but the M1 EEGNet primary
pipeline is engineering-validated.

## BLOCKER 1 — DGCNN secondary path has no session axis (data-artifact gap)
The DGCNN forward-graph dumps (`results/cmi_trace_p0p1/objective_comparison/*/audit/*erm*seed*.audit.npz`) carry
`Z_source/Z_target/subj_*/outer_fold` but **no `session_target`** (0/63 files). The mechanism oracle's evaluation
(P0.7 pipeline step 3) is a session-macro T_query split, so DGCNN cannot be evaluated from existing artifacts. The
runner now REASON-CODES this (`NO_SESSION_AXIS_FOR_QUERY_SPLIT`, 4/4 DGCNN smoke cells) instead of silently
dropping the cells (the earlier glob silently produced zero DGCNN rows — a false "absent"). Scope: this blocks only
the SECONDARY DGCNN route (BACKBONE_SPECIFIC), which per P0.6 cannot unlock M2 without a confirmatory rerun anyway.
The EEGNet confirmatory PRIMARY is unaffected.
Options (PM decision): (a) re-infer the frozen DGCNN forward-graph adapter persisting session metadata (genuine
re-inference, consistent with the "prefer real re-inference over proxy" discipline); (b) approve a session-free
target split (subject/fold macro) for the DGCNN secondary only — a contract touch; (c) run M1 EEGNet-primary-only
and defer DGCNN. No unilateral change made.

## BLOCKER 2 — P0.5 shared-overlap-matched control is INFEASIBLE BY CONSTRUCTION
The mechanism dictionary B (rank-8 generalized eigenbasis of `G_dis v = ρ(G_shared+ηI)v`) has a per-direction
shared-overlap profile `q_j = b_jᵀ G_shared b_j / tr(G_shared) ≈ 0` for ALL directions — because the generalized
eigenproblem SELECTS directions that maximize disagreement RELATIVE to shared, i.e. it extremizes shared-overlap
toward 0 by construction. Haar-random rank-8 dictionaries carry moderate shared-overlap and essentially never
reach q≈0, so the P0.5 fail-closed match (`rmse ≤ 0.02`) is unreachable at ANY pool:

| dataset (real EEGNet, seed0) | shared-overlap match rmse @ pool 200 / 1000 / 5000 |
|---|---|
| BNCI2014_001 sub1 | 0.138 / 0.114 / **0.098** |
| BNCI2014_001 sub2 | 0.128 / 0.105 / **0.092** |
| BNCI2015_001 sub10 | 0.099 / 0.061 / **0.038** |
| BNCI2015_001 sub11 | 0.098 / 0.059 / **0.038** |

rmse decreases with pool but PLATEAUS far above 0.02 (a measure-near-zero event: no random rank-8 subspace has all
8 directions at q≈0). As frozen, the P0.5 PRIMARY specificity control would ALWAYS fail-closed → M1 confirmatory
gate silently falls back to AMBIENT_ONLY (the smoke aggregator flags exactly this: `ctrl=AMBIENT_ONLY` on every
contrast cell).

### Proposed fix (PM approval required — do NOT change P0.5 unilaterally)
Match the **TOTAL task-contrast energy** profile `t_j = b_jᵀ (G_shared+G_dis) b_j / tr(G_shared+G_dis)` instead of
shared-overlap-only `q_j`. Rationale: the mechanism object is DEFINED by minimizing shared-overlap, so matching
shared-overlap is degenerate (asks random dicts to also live in the shared-null region, where they become
task-null, not a neutral control). Matching TOTAL task-contrast energy holds "how task-bearing" fixed and lets the
mechanism dict differ only in the shared-vs-disagreement COMPOSITION — which is exactly the scientific question
(does the disagreement-heavy subspace enrich future-subject harm more than an equally-task-bearing random
subspace?). Verified feasible on the same 6 real cells: best match rmse @ pool 5000 = **0.003–0.006** (≪ 0.02),
both datasets. The firewall is preserved (energy profile uses only source G_shared/G_dis, no target outcome).

## Disposition
M1 (126 cells) remains HOLD pending PM decisions on BLOCKER 1 (DGCNN session axis) and BLOCKER 2 (P0.5 control
redefinition → amendment 03). Stage-B code (module + hardened runner + aggregator + CPU sbatch + 17 tests) is
committed; it fail-closed CORRECTLY and surfaced both defects — no result was fabricated or silently degraded.
