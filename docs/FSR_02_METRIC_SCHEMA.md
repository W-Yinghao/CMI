# FSR_02 — Unified Metric Schema

**Project FSR — Step 2A.** Machine-readable definition of the six-level Functional Shortcut Reliance ladder as concrete, direction-signed metrics. This document is the contract that `scripts/fsr/build_phase2_tables.py` implements and `scripts/fsr/analyze_*.py` consume. It defines metrics only — no results, no interpretation.

Companion: `docs/FSR_03_CPU_ANALYSIS_PREREG.md` (the analysis pre-registration), `results/fsr_phase2/route_metric_table.csv` (the normalized instantiation).

## 0. Conventions

- **Ladder levels.** `L1` detectability · `L2` reducibility · `L3` erasability · `L4` task coupling · `L5` functional reliance · `L6` target consequence. Predictor-side = {L1,L2,L3,L4}; endpoint-side = {L5,L6}.
- **Direction.** Each metric declares the sign of its *expected* relationship under the hypothesis being tested. `higher_is_better` / `lower_is_better` for endpoints; `expected_corr_with_<X>` for predictors. A metric whose realized sign contradicts the "old hypothesis" direction is itself the finding (that is the FSR thesis).
- **Missing-value policy** (closed vocabulary): `not_measured`, `not_applicable`, `artifact_missing`, `needs_cpu_recompute`, `needs_small_frozen_probe`. No meaning-bearing field is blank.
- **Target-label policy** per metric: `NONE` (metric uses no target labels), `EVAL_ONLY` (target y used only to score the reported endpoint), `AUDIT_ONLY` (target X used at adapt-time, target y eval-only), `FORBIDDEN_RETRACTED` (a metric whose historical computation used target y for fit — retained only as a retracted/disclosed record, never in an RQ table).
- **primary_or_exploratory.** `primary` metrics carry a pre-registered RQ test; `exploratory` metrics are descriptive/robustness only.

Each metric below is declared with: `metric_name · ladder_level · direction · unit · source_artifact · allowed_routes · missing_value_policy · target_label_policy · primary_or_exploratory`.

---

## 1. L1 — Detectability (can D be decoded from Z | Y?)

```yaml
domain_probe_acc:
  ladder_level: L1
  direction: {expected_corr_with_R3: positive}   # old hypothesis: more decodable => more reliance
  unit: balanced_accuracy [chance..1]
  source_artifact: oaci C10/C17 probe; tos erasure_report subj_*_lin/mlp
  allowed_routes: [OACI_*, TOS_*]
  missing_value_policy: not_measured
  target_label_policy: NONE
  primary_or_exploratory: exploratory
posterior_KL:
  ladder_level: L1
  direction: {expected_corr_with_R3: positive}
  unit: nats (label-conditional posterior KL, plug-in proxy — NOT unbiased CMI)
  source_artifact: results/cigl_r123/final/r1_hardened_nperm1000.csv (observed_kl); multiseed_pareto.csv
  allowed_routes: [CIGL]
  missing_value_policy: artifact_missing   # per-fold graph_kl for seeds 1/2 pruned (raw npz/gate JSONs gone)
  target_label_policy: NONE
  primary_or_exploratory: primary
permutation_null_ratio:
  ladder_level: L1
  direction: {higher_means: more_detectable}
  unit: ratio observed_kl / perm_mean
  source_artifact: results/cigl_r123/final/r1_hardened_nperm1000.csv
  allowed_routes: [CIGL]
  missing_value_policy: not_applicable
  target_label_policy: NONE
  primary_or_exploratory: exploratory
graph_kl:
  ladder_level: L1
  direction: {old_hypothesis: higher_leakage_more_reliance, expected_corr_with_R3_task_drop: positive}
  unit: nats (posterior-KL proxy on graph_z)
  source_artifact: r1_hardened_nperm1000.csv (seed0, representation=graph); gap_correlations.csv (pooled rho, output)
  allowed_routes: [CIGL]
  missing_value_policy: artifact_missing   # per-fold seeds 1/2 pruned; seed0 recomputable at n=42
  target_label_policy: NONE
  primary_or_exploratory: primary
node_kl:
  ladder_level: L1
  direction: {old_hypothesis: higher_leakage_more_reliance, expected_corr_with_R3: positive}
  unit: nats (posterior-KL proxy on node_z)
  source_artifact: gap_correlations.csv; multiseed_pareto.csv
  allowed_routes: [CIGL]
  missing_value_policy: artifact_missing   # per-fold seeds 1/2 pruned
  target_label_policy: NONE
  primary_or_exploratory: exploratory
spatial_kl_if_available:
  ladder_level: L1
  direction: {expected_corr_with_R3: positive}
  unit: nats (per-branch posterior-KL on spatial_z)
  source_artifact: NONE ON DISK — requires frozen spatial_z + a trained per-branch probe
  allowed_routes: [FBCSP_LGG_branch_ablation]
  missing_value_policy: needs_small_frozen_probe
  target_label_policy: NONE
  primary_or_exploratory: primary   # (blocks RQ4)
```

## 2. L2 — Reducibility (does a method reduce measured leakage?)

```yaml
delta_KL:
  ladder_level: L2
  direction: {method_effect: reduce; negative_delta_is_reduction}
  unit: nats (method_KL - ERM_KL)
  source_artifact: results/cigl_r123/final/bootstrap_ci.csv
  allowed_routes: [CIGL, FCIGL, dCIGL]
  missing_value_policy: not_applicable
  target_label_policy: NONE
  primary_or_exploratory: primary
delta_probe_acc:
  ladder_level: L2
  direction: {method_effect: reduce}
  unit: balanced_accuracy delta
  source_artifact: r3_reliance.csv (subject_leak_drop) — present per fold, not summarized as a scalar
  allowed_routes: [CIGL]
  missing_value_policy: needs_cpu_recompute
  target_label_policy: NONE
  primary_or_exploratory: exploratory
leakage_reduction_pct:
  ladder_level: L2
  direction: {method_effect: reduce}
  unit: percent vs ERM
  source_artifact: gap_graph_node_mismatch.csv; CIGL_35 blueprint (-35..-77%)
  allowed_routes: [CIGL, CIGL_35_blueprint]
  missing_value_policy: not_applicable
  target_label_policy: NONE
  primary_or_exploratory: exploratory
feature_KL_delta:
  ladder_level: L2
  direction: {method_effect: reduce}
  unit: nats (feature_z KL delta vs baseline)
  source_artifact: results/cigl/metacmi_phase2/*_rows.json
  allowed_routes: [MetaCMI]
  missing_value_policy: not_applicable
  target_label_policy: NONE
  primary_or_exploratory: exploratory
```

## 3. L3 — Erasability (can an eraser remove subject signal?)

```yaml
subject_decode_drop:
  ladder_level: L3
  direction: {old_erasure_hypothesis: more_removal_more_target_benefit; expected_corr_with_target_benefit: positive}
  unit: balanced_accuracy drop (ERM_subject_acc - post_eraser_subject_acc)
  source_artifact: tos erasure_report.json (aggregate subj_*_lin/mlp)
  allowed_routes: [TOS_mean_scatter, TOS_LEACE, TOS_INLP, TOS_RLACE, TOS_random_k]
  missing_value_policy: not_applicable
  target_label_policy: NONE
  primary_or_exploratory: primary
LEACE_residual_subject_acc:
  ladder_level: L3
  direction: {lower_means_more_removed}
  unit: balanced_accuracy (lin + mlp residual)
  source_artifact: tos erasure_report.json (subj_LEACE_lin/mlp)
  allowed_routes: [TOS_LEACE]
  missing_value_policy: not_applicable
  target_label_policy: NONE
  primary_or_exploratory: primary
INLP_residual_subject_acc: {ladder_level: L3, direction: {lower_means_more_removed}, unit: balanced_accuracy, source_artifact: tos erasure_report.json (subj_INLP_lin/mlp), allowed_routes: [TOS_INLP], missing_value_policy: not_applicable, target_label_policy: NONE, primary_or_exploratory: primary}
RLACE_residual_subject_acc: {ladder_level: L3, direction: {lower_means_more_removed}, unit: balanced_accuracy, source_artifact: tos erasure_report.json (subj_RLACE_lin/mlp), allowed_routes: [TOS_RLACE], missing_value_policy: not_applicable, target_label_policy: NONE, primary_or_exploratory: primary}
random_k_subject_acc:
  ladder_level: L3
  direction: {control: should_NOT_remove}
  unit: balanced_accuracy (subj_random_k_lin/mlp)
  source_artifact: tos erasure_report.json
  allowed_routes: [TOS_random_k]
  missing_value_policy: not_applicable
  target_label_policy: NONE
  primary_or_exploratory: primary   # the falsifier control
```

## 4. L4 — Task coupling (is the subspace aligned with the task head?)

```yaml
align_k1: {ladder_level: L4, direction: {expected_corr_with_R3: positive}, unit: energy_fraction [0..1], source_artifact: gap_alignment.csv (k=1), allowed_routes: [CIGL, FCIGL, dCIGL], missing_value_policy: not_applicable, target_label_policy: NONE, primary_or_exploratory: exploratory}
align_k2:
  ladder_level: L4
  direction: {expected_corr_with_R3_task_drop: positive}   # the RIGHT-sign predictor (FSR thesis)
  unit: energy_fraction of linear task-head row-space in top-2 label-conditional subject subspace [0..1]
  source_artifact: results/cigl_r123/final/gap_alignment.csv (representation=graph_z, k=2)
  allowed_routes: [CIGL, FCIGL, dCIGL]
  missing_value_policy: not_applicable
  target_label_policy: NONE
  primary_or_exploratory: primary
task_head_alignment:
  ladder_level: L4
  direction: {expected_corr_with_R3: positive}
  unit: energy_fraction
  source_artifact: gap_diagnostic_summary.yaml (scalar per dataset/method)
  allowed_routes: [CIGL, FCIGL, dCIGL]
  missing_value_policy: not_applicable
  target_label_policy: NONE
  primary_or_exploratory: exploratory
branch_ablation_drop:
  ladder_level: L4
  direction: {larger_drop_more_load_bearing}
  unit: balanced_accuracy drop (mean - zero_branch)
  source_artifact: FBCSP_F0_AGGREGATE.csv; fblgg_f0/F0_DIAGNOSTICS
  allowed_routes: [FBCSP_LGG_branch_ablation, FBCSP_LGG_graph_starvation, FBCSP_LGG_bottleneck_analysis]
  missing_value_policy: not_applicable
  target_label_policy: NONE
  primary_or_exploratory: primary   # L4-only for FBCSP (no L1/L5 pairing -> SUPPORT_ONLY)
branch_gate_weight: {ladder_level: L4, direction: {higher_more_load_bearing}, unit: softmax_weight [0..1], source_artifact: FBCSP_F0_AGGREGATE.csv (gate_*), allowed_routes: [FBCSP_LGG_branch_ablation], missing_value_policy: not_applicable, target_label_policy: NONE, primary_or_exploratory: exploratory}
```

## 5. L5 — Functional reliance (does removing the subspace change the model?)

```yaml
R3_task_drop:
  ladder_level: L5
  direction: {higher_means_more_functional_reliance}
  unit: balanced_accuracy drop after removing top-k label-conditional subject subspace (head-replay)
  source_artifact: results/cigl_r123/final/r3_reliance.csv (conditioning=label_conditional, k)
  allowed_routes: [CIGL, FCIGL, dCIGL, MetaCMI]
  missing_value_policy: not_measured
  target_label_policy: NONE   # source-only fit, target appended eval-only
  primary_or_exploratory: primary
logit_SymKL:
  ladder_level: L5
  direction: {higher_means_more_reliance}
  unit: symmetric KL of logits before/after removal
  source_artifact: NOT in CIGL_62/65/66; may exist in cigl_direct_reliance R3 files (uncited)
  allowed_routes: [dCIGL]
  missing_value_policy: needs_cpu_recompute
  target_label_policy: NONE
  primary_or_exploratory: exploratory
CE_delta:
  ladder_level: L5
  direction: {higher_means_more_reliance}
  unit: cross-entropy delta after removal
  source_artifact: NOT summarized in frozen final/ CSVs
  allowed_routes: [CIGL, dCIGL]
  missing_value_policy: not_measured
  target_label_policy: NONE
  primary_or_exploratory: exploratory
NLL_delta_if_replay_only:
  ladder_level: L5
  direction: {higher_means_more_reliance}
  unit: NLL delta (ONLY when computed via exact head-replay, not deployment)
  source_artifact: none frozen
  allowed_routes: [CIGL]
  missing_value_policy: not_measured
  target_label_policy: NONE
  primary_or_exploratory: exploratory
```

## 6. L6 — Target consequence (harmful / benign / task-useful?)

```yaml
target_bAcc_delta:
  ladder_level: L6
  direction: higher_is_better
  unit: balanced_accuracy delta (method - baseline) on held-out target
  source_artifact: bootstrap_ci.csv (CIGL); tos erasure_target_deploy_summary.json (dtgt_bacc); cita gate rows
  allowed_routes: [CIGL, TOS_mean_scatter, TOS_LEACE, TOS_INLP, TOS_RLACE, TOS_random_k, CITA_*, TTA_Control_non_CMI]
  missing_value_policy: not_applicable
  target_label_policy: EVAL_ONLY   # (AUDIT_ONLY for CITA_*/TTA which use target X at adapt)
  primary_or_exploratory: primary
target_NLL_delta:
  ladder_level: L6
  direction: lower_is_better
  unit: NLL delta (dtgt_nll)
  source_artifact: tos erasure_target_deploy_summary.json
  allowed_routes: [TOS_LEACE, TOS_RLACE, TOS_INLP, TOS_mean_scatter, TOS_random_k]
  missing_value_policy: not_applicable   # (MISSING for the CMI cluster -> not_measured there)
  target_label_policy: EVAL_ONLY
  primary_or_exploratory: primary
target_ECE_delta:
  ladder_level: L6
  direction: lower_is_better
  unit: expected-calibration-error delta
  source_artifact: NONE (not reported anywhere)
  allowed_routes: [CIGL, TOS_*]
  missing_value_policy: not_measured
  target_label_policy: EVAL_ONLY
  primary_or_exploratory: exploratory
worst_subject_delta:
  ladder_level: L6
  direction: higher_is_better
  unit: worst-target-subject balanced_accuracy delta
  source_artifact: tos erasure_target_deploy_summary.json (worst_subject_tgt_bacc); OACI C10 (worst-domain)
  allowed_routes: [TOS_*, OACI_selection_leakage_not_target]
  missing_value_policy: not_measured   # (not in CIGL frozen CSVs)
  target_label_policy: EVAL_ONLY
  primary_or_exploratory: exploratory
harm_among_adapted:
  ladder_level: L6
  direction: lower_is_better
  unit: UCB of harm rate among adapted (ACAR G4)
  source_artifact: notes/ACAR_V5_STAGE2B_REAL_SELECTION_RESULT_*.md
  allowed_routes: [ACAR_stage2b_dev_stop]
  missing_value_policy: not_applicable
  target_label_policy: EVAL_ONLY
  primary_or_exploratory: exploratory
refusal_accept_decision:
  ladder_level: L6
  direction: {safe_default: refuse_when_uncertain}
  unit: categorical {accept, refuse, unsafe_accept} + counts
  source_artifact: tos notes/PHASE131_CERTIFICATION.md, PHASE2_EEG_FROZEN_PILOT.md
  allowed_routes: [TOS_refusal_gate]
  missing_value_policy: not_applicable
  target_label_policy: NONE
  primary_or_exploratory: exploratory   # decision endpoint, NOT a target-metric scalar
```

---

## 7. Direction rule summary (the sign contract Step 2 must not confuse)

```text
graph_kl              : old hypothesis => corr(graph_kl, R3_task_drop) POSITIVE   [FSR finding: it is NEGATIVE]
align_k2              : corr(align_k2, R3_task_drop) POSITIVE                       [FSR finding: it is POSITIVE]
subject_decode_drop   : old erasure hypothesis => corr(removal, target_benefit) POSITIVE  [FSR finding: ~0 / negative]
target_bAcc_delta     : higher_is_better
target_NLL_delta      : lower_is_better  (but a drop matched by random_k = non-specific, NOT a benefit)
R3_task_drop          : higher = more functional reliance
branch_ablation_drop  : larger = more load-bearing branch
```

## 8. Metric → RQ routing (implemented in build_phase2_tables.py)

| RQ | predictor metric(s) | endpoint metric | includable routes |
|---|---|---|---|
| RQ1 | graph_kl / node_kl (L1) | R3_task_drop (L5) | CIGL |
| RQ2 | subject_decode_drop (L3) | target_bAcc_delta, target_NLL_delta (L6) | TOS_mean_scatter, LEACE, INLP, RLACE, random_k |
| RQ3 | graph_kl (L1) vs align_k2 (L4) | R3_task_drop (L5) | CIGL |
| RQ4 | spatial_kl / per-branch leakage (L1) | per-branch R3 (L5) | none — `spatial_kl_if_available` + per-branch L5 are `needs_small_frozen_probe` |

All other routes (FCIGL/dCIGL/MetaCMI/CITA/TTA, OACI, ACAR, CSC, LPC, prior-decoupled TTA, FBCSP branch/bottleneck/scaffold) are `SUPPORT_ONLY` / `BOUNDARY_ONLY` / `PROTOCOL_ONLY` / `BACKGROUND_ONLY` under the revised Phase-1 gate and inform interpretation only. The exact per-route decision is in `results/fsr_phase2/analysis_inclusion_table.csv`.
