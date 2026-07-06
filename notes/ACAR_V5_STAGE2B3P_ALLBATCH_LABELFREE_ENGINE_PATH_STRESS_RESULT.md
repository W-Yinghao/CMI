# ACAR V5 — Stage-2B3P all-batch label-free engine-path stress, incl. forced tails (RESULT)

```
status: PASS

implementation_base_sha:   ba097775f239b2107210e1626a08603a2071790a
stage1b_run_id:            acar-v5-stage1b-c4412b4-r1
stage1b_registry_sha256:   2bbe55f4cdb4f1a18cee3b2c9e7583dba9fe9e84b9c563fb37781e98ebcbb76d
protocol_tag_target_sha:   4278435975a72b1127803dd2cffab420c083e430

scope:
  all-batch label-free real-package engine-path stress
  includes forced tails
  no labels / no v2 replay / no thresholds / no scoring / no selection
```

Label-free stress of the Stage-2 batching + action-provider path on the admitted Stage-1B package, pinned `ba09777`. Extends
Stage-2B2P (full 32-window batches only) to EVERY batch the shipped `PE.window_batches` emits — full 32, partial-but-eligible
(8 ≤ n < 32), and forced tails (n < STAGE2_MIN_BATCH = 8) — using the shipped `RAP.real_action_provider`. Routing is on the shipped
`forced` flag, so a forced tail is NEVER handed to a non-identity action (matched_coral/spdim/t3a). For forced n=1 tails (the exact
Stage-2B crash input) it additionally asserts `stable_matched_coral_v1` fails CLOSED. Two independent runs on distinct nodes.
Uses `PE.window_batches` only (never the label-consuming `evaluate_candidate_disease`); imports NO label loader / v2 replay /
selection engine / auth.

## Run environment

```
Run A:  SLURM job 885049   node nodecpu03   PASS   elapsed 1779.8 s
Run B:  SLURM job 885050   node nodecpu10   PASS   elapsed 2272.8 s   (submitted --exclude=nodecpu03 to force a distinct node)
Smoke:  SLURM job 885011   node nodecpu10   PASS   (PD/fold0 harness pre-flight)

python 3.13.14   torch 2.6.0+cu124   numpy 2.4.4   device_kind cpu   torch_num_threads 1   (conda env acar-v4-regen)
```

## Per-ref table (identical on both nodes)

```
disease fold seed        n_train n_val n_cal n_eval  n_total | full32 partial_8_31 forced_lt8 forced_prov | elig_action feat
PD      0    20260711        340    63   176    140      719 |    502          153         64          0 | PASS        PASS
PD      1    20260711        334    66   156    163      719 |    502          153         64          0 | PASS        PASS
PD      2    20260711        371    60   192     96      719 |    502          153         64          0 | PASS        PASS
PD      3    20260711        349    64   187    119      719 |    502          153         64          0 | PASS        PASS
PD      4    20260711        289    57   172    201      719 |    502          153         64          0 | PASS        PASS
SCZ     0    20260711        690   113   381    332     1516 |   1291          208         17          0 | PASS        PASS
SCZ     1    20260711        710   151   399    256     1516 |   1291          208         17          0 | PASS        PASS
SCZ     2    20260711        724   156   383    253     1516 |   1291          208         17          0 | PASS        PASS
SCZ     3    20260711        731   157   351    277     1516 |   1291          208         17          0 | PASS        PASS
SCZ     4    20260711        655   136   327    398     1516 |   1291          208         17          0 | PASS        PASS
                                                        11175 |   8965         1805        405          0
```

(No labels. No candidate scores. `forced_prov` = non-identity action-provider calls on forced tails; `elig_action` /
`feat` = eligible-batch action-output and paired-feature-finiteness contracts.)

## Aggregate (identical on both nodes)

```
n_refs_checked                       10
n_subjects_checked                   2280
n_total_batches_checked              11175
n_full_32_batches                    8965
n_partial_eligible_batches_8_to_31   1805       (NEW coverage vs Stage-2B2P, which tested full 32-window batches only)
n_forced_tail_batches_lt_8           405
n_forced_tail_provider_calls         0          (no forced tail routed through any non-identity action)
n_forced_n1_failclosed_verified      55         (every single-window tail — the exact Stage-2B crash input — fails CLOSED)
n_eligible_batches_action_checked    10770      (= 8965 full + 1805 partial)
n_actions_checked                    43080      (= 10770 * 4)
n_matched_coral_checked              10770
n_spdim_checked                      10770
n_nonfinite_pa                       0
n_nonfinite_zpost                    0
n_feature_contract_failures          0
max_probability_rowsum_error         2.220446049250313e-16      (machine epsilon)
min_eligible_batch_n                 8          (eligibility boundary = MIN_BATCH, confirmed)
max_eligible_batch_n                 32
forced_tail_n_values_observed        [1, 2, 3, 4, 5, 6, 7]      (all sub-MIN_BATCH sizes present)
max_matched_coral_M_smax             10.000000000000039 (A) / 10.000000000000048 (B)   (SVD gain cap binding)
label_loader_calls                   0
v2_replay_calls                      0
selected_candidate                   null
forbidden_modules_present            []         (label_loader / v2_replay / selection_engine / auth absent from sys.modules)
first_fail                           None
```

## Two-run comparison (cross-node)

```
run_A_status                  PASS
run_B_status                  PASS
node_A                        nodecpu03
node_B                        nodecpu10
same_node_or_distinct_node    DISTINCT
aggregate_counts_match        true      (per_ref tables + all aggregate COUNT keys field-by-field identical)
any_batch_failure_in_either_run  false
```

Full disclosure of the one non-identical cell: `max_matched_coral_M_smax` differs at the 1e-14 level
(`10.000000000000039` vs `10.000000000000048`) — floating-point noise in the largest SVD singular value across BLAS on the two
nodes. It is not a count and not a contract quantity; both values are pinned at the operator-gain cap (10.0) to 14 digits. Every
finiteness / contract / routing / count is identical.

## What this establishes (and its limits)

- The Stage-2B forced-tail crash is fully resolved on the real package: across all **405** forced tails in all 10 refs, the
  action provider was called **0** times, and every one of the **55** single-window tails (the exact inputs that crashed
  `acar-v5-stage2b-f079aca-r1`) now fails CLOSED (`stable_matched_coral_v1` raises) instead of producing a non-finite covariance.
- New coverage beyond Stage-2B2P: **1805 partial-but-eligible batches (8 ≤ n < 32)** — never exercised before — run all four
  actions with finite, contract-valid `p_a` / `z_post` / paired features. The eligibility boundary is confirmed to be
  `n < MIN_BATCH (8)`, not `n < 32` (`min_eligible_batch_n = 8`).
- Reproducible across distinct nodes (cpu03 vs cpu10): all per-ref and aggregate counts identical; the only difference is FP noise
  at 1e-14 in the SVD cap value.
- Label-free proven, not asserted: `label_loader` / `v2_replay` / `selection_engine` / `auth` never imported (`sys.modules`
  check); no thresholds, no scoring, no candidate selected.

Limits: this is a batching / action-seam finiteness / boundedness / forced-tail-contract / reproducibility stress ONLY. It reads
no labels and computes no utility, harm, coverage, or candidate ranking (no G1–G6). It does NOT authorize Stage-2B.

## Next gate (SEPARATE authorization)

A new real **Stage-2B** DEV candidate-selection authorization pinned to `ba09777` (or a reviewed result-note-only successor).
Stage-2B real selection remains on HOLD until then.
