# S2P_15 - Route B Channel Contract

**Phase:** 9D-1 Route B channel canonicalization and stratified sampling contract.

**Status:** PASS for B1 launch conditions. No training was run by the contract job.

## Scope

Route B is:

```text
CBraMod-only adaptive-channel high-budget calibration
corpus = TUEG processed exact-33ch subset
model = CBraMod
CodeBrain = excluded
```

This is not CodeBrain-compatible and not a 19-common-equivalent curve.

## Channel Contract

The exact-33ch subset contains six distinct channel-order/reference groups. A single global 33-name order is not
feasible without dropping or imputing channels, which is not authorized.

Selected contract:

```text
selected_sampling_contract = fixed_group_mix
canonical_channel_order_scope = group_specific_pinned_orders
group_id = channel_order_hash x reference_scheme
```

Within each group, records are reordered to that group's pinned 33-channel order. Across budgets, windows are sampled
with fixed group proportions using exact largest-remainder allocation.

This licenses:

```text
CBraMod 33ch heterogeneous-reference budget calibration with fixed group-mixture sampling.
```

It does not license:

```text
CBraMod is channel-invariant.
33ch Route B is CodeBrain-compatible.
33ch results can be merged into the 19-common curve.
```

## Feasibility

B1 budgets:

```text
H = {200, 500, 1000, 2000}
seeds = {0, 1}
```

All budgets pass exact group-mixture allocation:

```text
200h:  24000 / 24000 windows
500h:  60000 / 60000 windows
1000h: 120000 / 120000 windows
2000h: 240000 / 240000 windows
```

Largest single group fallback is not feasible for >=2000h under the same val requirement, so B1 uses fixed group mix.

## Downstream Sanity

Tiny SHU-MI sanity used source-only PCA/head selection and target labels only for final scoring.

Result:

```text
native32:
  random target bAcc = 0.520
  released target bAcc = 0.540
  released - random = +0.020
  sanity pass = true

19-common:
  random target bAcc = 0.492
  released target bAcc = 0.532
  released - random = +0.040
  sanity pass = true
```

Primary downstream for Route B is therefore:

```text
SHU_MI_native32
```

The tiny sanity is not the final downstream audit. Training outputs still require the full frozen SHU-MI audit.

## Go/No-Go

`route_b_training_go_nogo.json` reports:

```json
{
  "route": "B_33ch_cbramod_only",
  "canonical_channel_order_pinned": true,
  "canonical_channel_order_scope": "group_specific_pinned_orders",
  "reference_scheme_groups_pinned": true,
  "fixed_group_mix_feasible": true,
  "largest_single_group_feasible_ge2000h": false,
  "selected_sampling_contract": "fixed_group_mix",
  "budgets_h": [200, 500, 1000, 2000],
  "seeds": [0, 1],
  "h4000_included": false,
  "downstream_primary": "SHU_MI_native32",
  "released_checkpoint_downstream_sanity_pass": true,
  "target_labels_used_for_selection": false,
  "launch_route_b_b1": true
}
```

## Artifacts

```text
results/s2p_route_b_33ch_contract/route_b_channel_group_manifest.csv
results/s2p_route_b_33ch_contract/route_b_reference_scheme_manifest.csv
results/s2p_route_b_33ch_contract/route_b_canonical_channel_order.json
results/s2p_route_b_33ch_contract/route_b_group_mix_by_budget.csv
results/s2p_route_b_33ch_contract/route_b_sampling_contract_check.csv
results/s2p_route_b_33ch_contract/route_b_downstream_sanity.csv
results/s2p_route_b_33ch_contract/route_b_training_go_nogo.json
results/s2p_route_b_33ch_contract/route_b_b1_training_tasks.csv
```
