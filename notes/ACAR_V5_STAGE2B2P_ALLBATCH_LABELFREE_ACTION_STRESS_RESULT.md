# ACAR V5 — Stage-2B2P all-batch label-free stable-action stress (RESULT)

```
status: PASS

implementation_base_sha:   f079aca9570fef47b333e34cc238376e29fb6cc8
stage1b_run_id:            acar-v5-stage1b-c4412b4-r1
stage1b_registry_sha256:   2bbe55f4cdb4f1a18cee3b2c9e7583dba9fe9e84b9c563fb37781e98ebcbb76d
protocol_tag_target_sha:   4278435975a72b1127803dd2cffab420c083e430

scope:
  all-batch label-free real-package action stress
  no labels / no v2 replay / no thresholds / no scoring / no selection
```

Label-free stress of the Stage-2B2-amended action-provider seam (`stable_matched_coral_v1`) on the admitted Stage-1B package.
Not candidate selection. It opened EVERY full 32-window batch (window-ordered; tails < 32 excluded; batch size unchanged) across
`train/val/cal/eval` of the 10 seed-20260711 selection refs and ran `identity / matched_coral (stable) / spdim / t3a` through the
amended `real_action_provider`, validating the full output + paired-feature-finiteness contracts on every batch. Two independent
runs on **distinct nodes**. The frozen `pmct` path is not on this route (`spdim`/`t3a` still route through the frozen
`acar.actions`). Run at a detached git worktree pinned to exactly `f079aca` (guard: worktree HEAD == impl, clean).

## Run environment

```
Run A:  SLURM job 883465   node nodecpu05   PASS   elapsed 2418.0 s
Run B:  SLURM job 883471   node nodecpu04   PASS   elapsed 2470.1 s   (submitted --exclude=nodecpu05 to force a distinct node)

python 3.13.14   torch 2.6.0+cu124   numpy 2.4.4   device_kind cpu   torch_num_threads 1   (conda env acar-v4-regen)
```

## Per-ref table (identical on both nodes)

```
disease fold seed          n_train n_val n_cal n_eval  n_total | identity mc  spdim t3a  feat | class_order
PD      0    20260711          237    39   123    103      502 | PASS     PASS PASS  PASS PASS | [control,case]=[0,1]
PD      1    20260711          240    42   106    114      502 | PASS     PASS PASS  PASS PASS | [control,case]=[0,1]
PD      2    20260711          272    37   140     53      502 | PASS     PASS PASS  PASS PASS | [control,case]=[0,1]
PD      3    20260711          247    40   132     83      502 | PASS     PASS PASS  PASS PASS | [control,case]=[0,1]
PD      4    20260711          196    35   122    149      502 | PASS     PASS PASS  PASS PASS | [control,case]=[0,1]
SCZ     0    20260711          593    89   329    280     1291 | PASS     PASS PASS  PASS PASS | [control,case]=[0,1]
SCZ     1    20260711          605   125   343    218     1291 | PASS     PASS PASS  PASS PASS | [control,case]=[0,1]
SCZ     2    20260711          618   130   327    216     1291 | PASS     PASS PASS  PASS PASS | [control,case]=[0,1]
SCZ     3    20260711          627   131   295    238     1291 | PASS     PASS PASS  PASS PASS | [control,case]=[0,1]
SCZ     4    20260711          562   113   277    339     1291 | PASS     PASS PASS  PASS PASS | [control,case]=[0,1]
                                                        = 8965
```

(No labels. No candidate scores. `mc` = stable matched_coral; `feat` = paired-feature-finiteness contract.)

## Aggregate (identical on both nodes)

```
n_refs_checked                    10
n_batches_checked                 8965      (full 32-window batches only)
n_actions_checked                 35860     (= 8965 * 4)
n_matched_coral_checked           8965
n_spdim_checked                   8965
n_nonfinite_pa                    0
n_nonfinite_zpost                 0
n_feature_contract_failures       0
max_probability_rowsum_error      2.220446049250313e-16      (machine epsilon)
max_matched_coral_M_smax          10.000000000000039         (SVD operator-gain cap BINDING; holds at smax=10 to FP)
matched_coral_alpha_eff_min       0.05617101904149896
matched_coral_alpha_eff_max       0.0625                     (= 32/(2*256) = g_cov cap when g_unc~1; in [0,1])
max_matched_coral_cond_after_floor 2063.0330640107272        (< shrink bound (1-rho)*D/rho+1 ~= 2305; floor never fired)
label_loader_calls                0
v2_replay_calls                   0
selected_candidate                null
forbidden_modules_present         []        (label_loader / v2_replay / selection_engine / policy_eval / stage2b_authorization
                                             all ABSENT from sys.modules — hard proof of label-free)
first_fail                        None
```

## Two-run comparison (cross-node)

```
run_A_status                  PASS
run_B_status                  PASS
node_A                        nodecpu05
node_B                        nodecpu04
same_node_or_distinct_node    DISTINCT
aggregate_counts_match        true      (per_ref tables + all aggregate keys field-by-field identical)
any_batch_failure_in_either_run  false
```

## What this establishes (and its limits)

- The Stage-2B2 amendment removes the numerical blocker at scale: on all **8965** full 32-window / 256-D real batches across
  both diseases and all four split roles, stable `matched_coral` produced finite, contract-valid `p_a` / `z_post` / paired
  features — where the frozen `pmct` CORAL produced intermittent non-finite `p_a` (the blocker) on the same substrate.
- The two boundedness mechanisms behave exactly as designed on real data: the **SVD operator-gain cap** is *binding*
  (`M_smax` pinned at 10 to FP — the raw operator would exceed it), while the **shrink** is the primary conditioner
  (`cond_after_floor` peaks at 2063 < the 2305 shrink bound), so the eigenvalue floor is a genuinely inert safety net here.
- Reproducible across **distinct nodes** (cpu05 vs cpu04) to full precision — directly closing the original "FP-sensitive across
  nodes" concern that motivated the amendment.
- Label-free is proven, not asserted: the forbidden label / v2-replay / selection / policy / auth modules were never imported
  (`sys.modules` check). No thresholds, no scoring, no candidate selected.

Limits: this is an action-seam finiteness/boundedness/reproducibility stress ONLY. It certifies nothing about utility, harm,
coverage, or candidate ranking — no labels were read and no G1–G6 quantity was computed. It does not authorize Stage-2B.

## Next gate (SEPARATE authorization)

A new real **Stage-2B** DEV candidate-selection authorization pinned to `f079aca` (or a reviewed result-note-only successor).
Stage-2B real selection remains on HOLD until then.
