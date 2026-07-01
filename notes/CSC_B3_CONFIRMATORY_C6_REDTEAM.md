# CSC Route B3 Confirmatory — C6 INDEPENDENT RED-TEAM RE-AGGREGATION NOTE

**Role.** Independent, separate agent re-aggregating the frozen confirmatory artifact from raw `per_cluster`
(5376 records) — NOT trusting the runner's `verdict` block, using its OWN scipy `Beta.ppf` Clopper-Pearson
(not the runner's bisection). This is the C6 gate; the final verdict is contingent on it.

Artifact: `csc/results/b3_confirmatory_result.json` @ worktree tag `csc-b3-confirmatory-v1` / `0595f64`.

## Structure (PASS)
- 5376 records; 112 cells × 48 replicates; **every scenario×kind×budget cell has exactly n=48** (0 deviations,
  0 unexpected cells). Phase counts control 4032, primary 768, secondary 576; budget balanced 2688/2688.
  Truth sane (control→NO_CONCEPT, primary+secondary→CONCEPT).
- `base_seed==3000000`; **all per-cell seeds match** `base + 100000*cell_index + replicate` (0 mismatches);
  5376 cluster seeds unique; cluster ∪ target-offset(+10000) = 10752 with **0 collisions**; **0 overlap** with
  smoke <100000, A src 900000–65, A tgt 1800000–65, B dev blocks.

## Provenance (PASS)
- Recomputed canonical sha256 over `frozen_payload` = `a96dc0de…65cf62` == manifest file == artifact
  `manifest_hash` (all three equal). `git_head==0595f64` == tag commit; `git_status_clean==True`. All 3 pinned
  code hashes (paired_calibrated / paired_conditional_test / paired_sim) match on disk.

## Criteria (independently recomputed)
- **C1 PASS** — missing_pair + unequal_epochs_extreme false confirms = 0 (both 100% INVALID_PAIR_STRUCTURE,
  576/576 each).
- **C2 PASS** — all 14 control kind×budget cells (n=288, each verified pooled over 6 scenarios) CP-up ≤ 0.05.
  Worst `clean|m30` = 3/288, CP-up **0.026701**. Total control confirms = 6.
- **C3 PASS** — only 5 control cells have ≥1 confirm; max 2/48 (clean|baseline|m30). No cell ≥6/48; none even
  ≥3/48; no kind×budget with ≥2 cells at ≥3.
- **C4 PASS** — paired_concept & paired_concept_plus_cov, both m20 and m30, pooled 4 primary scenarios:
  power 192/192 = 1.0000, CP-lo **0.984518**. Every per-scenario cell = 1.00.
- **C5 PASS** — n_sampler_failures 0, n_boot_invalid 0, 0 states outside the 5-state set.

## Secondary (non-gating)
paired_pure_conditional: m20 42/288 = 0.146 (CP-lo 0.113); m30 81/288 = 0.281 (CP-lo 0.238).

## Red-team integrity checks
- Own scipy CP matches runner values to machine precision (max |Δ| cp_up 2.8e-17; cp_lo 0). No CP discrepancy.
- **No denominator inflation/deflation** — INVALID_PAIR_STRUCTURE and NEED_MORE_LABELS retained in both control
  (n=288) and power (n=192) denominators. `confirmed` flag == (state==CONCEPT_CONFIRMED) for all 5376 (0 mismatches).
- Confirmed accounting reconciles: 6 control + 768 primary + 123 secondary = 897.
- Controls confirmed evaluated over all 6 scenarios (incl. label_noise + few_epochs), not just the 4 strong.

## C6 VERDICT
**C6 CONFIRMS the runner's preliminary verdict WITHOUT correction.** Every gating number reproduced exactly
and independently; no discrepancy, hidden failure, denominator inflation, or overclaim.
**Overall C6-inclusive verdict: PASS.**
