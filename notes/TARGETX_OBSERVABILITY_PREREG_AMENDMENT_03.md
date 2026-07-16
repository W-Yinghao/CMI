# Target-X observability audit — PRE-REG AMENDMENT 03 (F2.1a analysis-and-gate closure)

Transparent amendment closing the F2.1 analysis pipeline per PM review of `81fdf650`. Fixes the five unclosed
full-run items (baselines, hindsight denominator, promised artifacts, phase enforcement, rule-level Gate 5) so
the frozen G1 selector becomes a fully-aggregable, reconstructable, control-compared, five-gate real-EEG
system. Branch `agent/cmi-trace-targetx-observability`. Manuscript FROZEN. No full F2.1 run in F2.1a.

## C1 — complete action / comparator system (P0.1, F2.1a-1)
Action carries THREE transforms so transductive baselines are honest: `apply_source`, `apply_target_cal`,
`apply_target_query` (projector-deletion actions use the same (I−P) for all three).
- **Selectable (informed)**: identity; cond-basis singletons; cond-basis rank-2 and rank-3 subsets;
  source-greedy prefixes. **Primary selectable rank ≤ 3.**
- **Non-selectable controls**: 50 ambient same-rank random projectors per informed rank; **whitening**
  (fit on source, applied to source & target); **target-mean-centering** (Z_s−μ_s ; Z_{t}−μ_{t,cal}, a genuine
  transductive alignment baseline — CRITICAL because G1 IS target-mean-discrepancy reduction); **source-greedy
  standalone**; **target-hindsight calibration oracle**. (CORAL/EA alignment optional secondary baseline, F2.2.)
- Target-hindsight and random NEVER enter the G1 selector's eligible set.

## C2 — target-hindsight denominator (P0.2, F2.1a-1)
Session-aware oracle: greedily select a deletion on `T_cal` **using T_cal target labels** (non-deployable),
score on `T_query`. Δ_target_hindsight is the recovery-ratio denominator. Also report Δ_source_greedy (its own
independent T_query gain) and oracle_recovery_ratio = Δ_TX / Δ_target_hindsight.

## C3 — full selection audit trail (P0.5 reconstruction, F2.1a-2)
Every action row saves: `basis_family, basis_rank, basis_hash, projector_hash, basis_indices, action_kind,
eligible, rank, G1, source_task_drop, random_q95_same_rank, safe_gate_pass, specificity_gate_pass,
utility_macro, utility_pooled, config_hash, git_sha`. Selected fold summary saves: `delta_tx,
delta_source_greedy, delta_whitening, delta_mean_centering, delta_random_same_rank, delta_target_hindsight,
oracle_recovery_ratio`. The selected projector must be byte-reconstructable from (basis_hash, basis_indices).

## C4 — aggregation + deterministic 5-gate engine (P0.3, F2.1a-3)
`scripts/aggregate_targetx_observability.py` emits ALL seven artifacts: `targetx_action_rows.jsonl,
targetx_fold_summary.csv, targetx_cluster_intervals.csv, targetx_observability_by_rank.csv,
targetx_negative_controls.csv, targetx_gate_verdict.json, targetx_completeness_matrix.csv`. Stats: inference
unit = target subject / outer fold; the 3 seeds of a fold are bootstrapped together; subject-cluster 10,000
bootstrap; BNCI2015 primary = query-session MACRO (pooled = sensitivity); G1 observability reported PER RANK
(pooled rank-correlation secondary); all control comparisons are PAIRED differences (Δ_TX − Δ_control per
fold, then cluster CI), never two independent CIs. The five gates are PURE functions in
`tos_cmi/eval/targetx_gates.py` with unit tests:
1. Observability: LCB95(median within-subject Spearman ρ(G1, U), rank-stratified) > 0.
2. Actionability: LCB95(Δ_TX macro) > 0 AND LCB95(Δ_TX − Δ_random) > 0 AND > source-greedy AND > whitening AND
   > mean-centering (all paired).
3. Oracle recovery: LCB95(oracle_recovery_ratio) ≥ 0.25.
4. Cross-dataset safety: no dataset with LCB95(Δ_TX) clearly < 0 while the other is positive.
5. Subject-leakage specificity: rule-level cross-fitted ΔÎ_specific LCB95 > 0 (C5).

## C5 — Gate 5 = rule-level cross-fitted certification (P0.5, F2.1a-4)
Certify the SAME target-X selection RULE, not a conveniently re-measured projector. Per outer fold: source →
eraser-fit / posterior-train / posterior-eval; rebuild cond basis on eraser-fit; apply the SAME G1 + safety +
random-specificity rule on `T_cal X` to pick the action; apply it to posterior train/eval; compare exact-rank
ambient random; capacities = TRUE-LINEAR (logistic) + MLP-small + MLP-large; FULLY-RETRAINED within-label
permutation null; report ΔÎ_specific = ΔÎ_selected − ΔÎ_random with familywise capacity handling.

## C6 — phase enforcement (P0.4)
`run_targetx_observability.py --phase {primary,secondary}`. `primary` = G1 ONLY (default for the full F2.1
run). `secondary` (the 9 remaining observables, Holm-corrected; G2 excluded as ≡G1) runs ONLY after the G1
primary result is frozen (F2.2). A full run must never silently execute secondaries.

## C7 — strengthened tests
`test_session_macro_differs_from_pooled` (fixture with unequal per-session gains; assert macro ≠ pooled and
equals the hand-computed macro); `test_end_to_end_artifacts_and_reconstruction` (temp dir: action-row count ==
manifest; random rows present; selected projector rebuildable from basis_hash+indices; gate verdict fully
regenerated from written artifacts); `test_gate5_rule_matches_main_selector`; the five gate pure functions each
unit-tested on constructed inputs.

## C8 — third (final) pre-full smoke (F2.1a-5)
Real EEG, 2 datasets × 2 subjects, seed 0, cond, G1 only, 50 random/rank, ALL baselines, ALL artifacts, Gate 5
= true-linear + MLP-small + few permutations (engineering only). Smoke checks: baselines non-empty; oracle
denominator present; recovery ratio computable; paired-CI code runs; selected projector reconstructable; Gate-5
rule == main selector; secondaries NOT run; `paper/` unchanged (0 files). No scientific conclusion from smoke.

## C9 — scope
Synthetic/smoke = engineering only. Method GO/NO-GO requires full LOSO × 3 seeds × both datasets, subject/
fold-cluster CI, full negative controls. Full F2.1, adaptation, TTE, manuscript remain HELD/FROZEN.
