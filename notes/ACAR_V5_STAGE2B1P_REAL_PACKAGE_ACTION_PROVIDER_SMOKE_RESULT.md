# ACAR V5 — Stage-2B1P Real-Package Action-Provider Smoke Result

```
status: PASS

implementation_base_sha:  a5c44c31607b0771e5367d6eb84bcaced7c71b37
stage1b_run_id:           acar-v5-stage1b-c4412b4-r1
stage1b_registry_sha256:  2bbe55f4cdb4f1a18cee3b2c9e7583dba9fe9e84b9c563fb37781e98ebcbb76d
protocol_tag_target_sha:  4278435975a72b1127803dd2cffab420c083e430

scope: label-free real-package action-provider smoke only
       no labels / no v2 replay / no scoring / no thresholds / no selection
```

A **read-only, label-free** preflight (NOT candidate selection). It admitted the Stage-1B package, and for ONLY the 10 canonical
seed-`20260711` selection refs read the per-ref `source_state_file` + real `feat_dump` embeddings and ran identity / matched_coral /
spdim / t3a on a deterministic 32-window sample per split role, validating the source-state-adapter, action-output, z_post, and
feature-finiteness contracts. No `participants.tsv` / label loader was invoked, no v2-replay was invoked, no threshold or route
decision was produced, `run_selection` was not called. Executed via SLURM job `883339` (nodecpu11, single-threaded, `ExitCode 0:0`,
30 s) with the sbatch guard enforcing `HEAD == a5c44c31607b0771e5367d6eb84bcaced7c71b37`.

## Sampling

For each of the 10 refs, for each split role {train, val, cal, eval}: the first subject_key (lexicographic) with a full 32-window
batch was chosen and its first 32-window batch run through identity / matched_coral / spdim / t3a. Every ref supplied all four
split roles with a full 32-window batch (no fail-closed sampling gap).

## Per-ref table

| disease | fold | seed | sampled split_roles | sampled n_batches | actions_checked | source_state_adapter | source_state_sha==registry | action_output | feature_finiteness |
|---|---|---|---|---|---|---|---|---|---|
| PD  | 0 | 20260711 | train, val, cal, eval | 4 | identity, matched_coral, spdim, t3a | OK | true | OK | OK |
| PD  | 1 | 20260711 | train, val, cal, eval | 4 | identity, matched_coral, spdim, t3a | OK | true | OK | OK |
| PD  | 2 | 20260711 | train, val, cal, eval | 4 | identity, matched_coral, spdim, t3a | OK | true | OK | OK |
| PD  | 3 | 20260711 | train, val, cal, eval | 4 | identity, matched_coral, spdim, t3a | OK | true | OK | OK |
| PD  | 4 | 20260711 | train, val, cal, eval | 4 | identity, matched_coral, spdim, t3a | OK | true | OK | OK |
| SCZ | 0 | 20260711 | train, val, cal, eval | 4 | identity, matched_coral, spdim, t3a | OK | true | OK | OK |
| SCZ | 1 | 20260711 | train, val, cal, eval | 4 | identity, matched_coral, spdim, t3a | OK | true | OK | OK |
| SCZ | 2 | 20260711 | train, val, cal, eval | 4 | identity, matched_coral, spdim, t3a | OK | true | OK | OK |
| SCZ | 3 | 20260711 | train, val, cal, eval | 4 | identity, matched_coral, spdim, t3a | OK | true | OK | OK |
| SCZ | 4 | 20260711 | train, val, cal, eval | 4 | identity, matched_coral, spdim, t3a | OK | true | OK | OK |

Each action's output satisfied: `p_a` shape [32,2], finite, in [0,1], rows sum to 1, class order [control,case]=[0,1]; z_post = None
for t3a and a finite [32,256] geometry array for matched_coral / spdim. Paired features: d_entropy/d_margin/flip_rate/JS/n_eff
finite for every action; Bures/post_sep finite for matched_coral/spdim and (allowed) NaN for t3a. Embedding dim on the real package
= 256.

## Aggregate

```
n_refs_checked:      10   (PD fold0-4, SCZ fold0-4; seed 20260711 only)
n_batches_checked:   40   (10 refs × 4 split roles)
n_actions_checked:  160   (40 batches × 4 actions)
spdim_checked:      true  (spdim ran on real 256-dim embeddings — not skipped)
torch_available:    true
label_loader_calls:  0
v2_replay_calls:     0
selected_candidate:  null
first_fail:          null
```

## Interpretation

The real action-provider seam — identity (source-state LDA f_0) + matched_coral / spdim / t3a via the frozen `acar.actions` and the
v5→old source-state adapter — is contract-valid on the admitted Stage-1B package's real embeddings for all 10 canonical selection
refs, with the per-ref `source_state_file` bytes matching the registry hash. No labels, no v2-replay, no scoring, no thresholds, no
selection were performed. Real Stage-2B candidate selection remains a SEPARATE authorization.
