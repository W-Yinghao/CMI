TALOS_00N - Failure Taxonomy

Status: read-only taxonomy over existing TALOS_00B artifacts. No adapter was
rerun. No variant universe, trust-region bound, threshold, or red-team semantic
was changed.

Inputs

```text
run_manifest:       results/talos/talos00b_bnci2014_001_seed0/run_manifest.json
variant_table:      results/talos/talos00b_bnci2014_001_seed0/variant_table.csv
per_fold_metrics:   results/talos/talos00b_bnci2014_001_seed0/per_fold_metrics.csv
red_team:           results/talos/talos00b_bnci2014_001_seed0/red_team.json
scientific_gate:    results/talos/talos00b_bnci2014_001_seed0/scientific_gate.json
adapter_norms:      results/talos/talos00b_bnci2014_001_seed0/adapter_norms.json
collapse_guards:    results/talos/talos00b_bnci2014_001_seed0/collapse_guards.json
```

Top-level outcome

```text
scientific gate: FAIL
red-team failures: 0
red-team warnings: 0
P1 request: denied
P2 request: denied
```

Taxonomy labels

```text
failed_clean_effect_size:
  A clean, non-boundary TALOS variant did not reach the required +0.020 target
  bAcc improvement over ERM on a required backbone.

failed_boundary_hit_guard:
  Apparent target gains were produced by adapters that hit predeclared
  trust-region bounds, making them diagnostic-only.

failed_cross_backbone_consistency:
  The route did not produce a clean positive result on both EEGNetMini and
  EEGConformerMini.

passed_red_team:
  Handoff immutability, target-label quarantine, adapter determinism, and
  variant freeze passed with no failures.

passed_target_quarantine:
  Removing or permuting target labels did not change adapter state, predictions,
  probability hashes, variant ranking, reported variant, or pre-final metrics.

passed_determinism:
  Repeating the same source state, target features, seed, and config reproduced
  adapter and prediction hashes.

passed_variant_freeze:
  The variant universe stayed fixed:
  ERM_NO_ADAPT, TTA_CONTROL_REPLAY, TALOS_L, TALOS_D, TALOS_LD.
```

Aggregate taxonomy

```text
failed_clean_effect_size:       yes
failed_boundary_hit_guard:      yes
failed_cross_backbone_consistency: yes
passed_red_team:                yes
passed_target_quarantine:       yes
passed_determinism:             yes
passed_variant_freeze:          yes
```

Per-backbone taxonomy

```text
EEGConformerMini:
  best clean TALOS variant: TALOS_L
  TALOS_L mean bAcc: 0.4157
  ERM mean bAcc: 0.4136
  delta vs ERM: +0.0021
  boundary-hit count for TALOS_L: 0 / 9
  taxonomy: failed_clean_effect_size

EEGNetMini:
  TALOS_L mean bAcc: 0.4209
  ERM mean bAcc: 0.4005
  TALOS_L delta vs ERM: +0.0204
  TALOS_L boundary-hit count: 1 / 9
  TALOS_D boundary-hit count: 9 / 9
  TALOS_LD boundary-hit count: 9 / 9
  taxonomy: failed_boundary_hit_guard
```

Variant-level diagnostic table

```text
EEGConformerMini:
  ERM_NO_ADAPT        bAcc 0.4136  NLL 5.1844  ECE 0.5090  boundary 0/9
  TTA_CONTROL_REPLAY  bAcc 0.4136  NLL 4.8540  ECE 0.5015  boundary 0/9
  TALOS_L             bAcc 0.4157  NLL 4.7819  ECE 0.4965  boundary 0/9
  TALOS_D             bAcc 0.4180  NLL 5.8497  ECE 0.5201  boundary 9/9
  TALOS_LD            bAcc 0.4188  NLL 5.4760  ECE 0.5111  boundary 9/9

EEGNetMini:
  ERM_NO_ADAPT        bAcc 0.4005  NLL 2.4672  ECE 0.3441  boundary 0/9
  TTA_CONTROL_REPLAY  bAcc 0.4005  NLL 2.4182  ECE 0.3416  boundary 0/9
  TALOS_L             bAcc 0.4209  NLL 2.1447  ECE 0.3027  boundary 1/9
  TALOS_D             bAcc 0.4579  NLL 1.7499  ECE 0.2474  boundary 9/9
  TALOS_LD            bAcc 0.4581  NLL 1.7503  ECE 0.2458  boundary 9/9
```

Passed checks

```text
handoff_immutability: PASS
target_label_quarantine: PASS
adapter_determinism: PASS
variant_freeze: PASS
collapse guards: PASS across aggregate variants
```

Collapse note

Collapse did not drive the negative result:

```text
collapse_warning_count: 0 for every backbone x variant
```

The limiting conditions were insufficient clean effect and boundary-hit gains.

Forbidden interpretation

Do not describe TALOS_00B as "almost positive". Do not treat the EEGNetMini
boundary-hit gains as P1 evidence. Do not use target metrics to tune trust
region bounds, add TALOS-LR, add TALOS-full, or select a deployment variant.

Mechanism note

TALOS_00B narrows the mechanism question. It suggests that the historical
TTA-Control gain is not cleanly explained by the current bounded logit
temperature/bias adapter, diagonal feature affine adapter, or their combination.
It does not prove that all target-unlabeled adaptation is impossible.
