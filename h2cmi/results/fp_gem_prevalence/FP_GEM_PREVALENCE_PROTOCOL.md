# FP-GEM Fixed-Reservoir Prevalence-Stress Protocol

Status: `FROZEN_BEFORE_GPU_COMPUTE`

This is the final targeted method experiment. It asks whether FP-GEM is less sensitive than Joint-GEM, RCT, and official SPDIM to a controlled change in adaptation-batch class prevalence while the source checkpoint and real EEG trial reservoir remain fixed. It is not a broad or natural-transfer benchmark.

## Frozen Scope

- Dataset: `Lee2019_MI` only.
- Targets: all 54 repaired-split subjects, IDs 1 through 54.
- Source seeds: 0, 1, 2.
- Source state: the exact 162 persisted P12 checkpoints; fresh source training and fallback retraining are prohibited.
- Evaluation: the exact P12 50-trial evaluation block for each subject, unchanged, class counts `[25,25]`.
- Adaptation reservoir: the exact P12 50-trial adaptation block for each subject, class counts `[25,25]`.
- Methods: `source_only_tsmnet`, `rct`, `spdim_geodesic`, `spdim_bias`, `Joint-GEM`, `FP-GEM`.
- Backbone, classifier, source objective, repaired split, method parameters, optimizer, and regularization are exactly P12.

## Controlled Intervention

The frozen class-0 prevalence grid is `q in {0.1, 0.5, 0.9}` with a constant adaptation size of 50:

| q | class-0 | class-1 | construction |
|---:|---:|---:|---|
| 0.1 | 5 | 45 | deterministic class-order then modulo repeat/crop within the P12 reservoir |
| 0.5 | 25 | 25 | exact original P12 trial IDs and order |
| 0.9 | 45 | 5 | deterministic class-order then modulo repeat/crop within the P12 reservoir |

Every repeated occurrence and its repeat index are persisted in `fp_gem_prevalence_manifest.csv`. The builder uses labels only to construct this controlled intervention. Runtime methods receive ordered EEG/features but no labels, q value, or target performance. Evaluation labels are first accessed after all q-specific adaptations have finished.

P12 retained prediction/logits hashes rather than vectors. The q=0.5 methods are therefore deterministically replayed solely to recover vectors for prediction-disagreement analysis. All six prediction and logits hashes and both GEM geometry hashes must match P12 exactly; committed P12 q=0.5 metrics remain the accepted center rows.

## Frozen Method Details

- Feature hook: sole tensor input to `TSMNet.classifier`, dimension 210.
- Classifier: unchanged frozen `Linear(210,2)`.
- Density: P12 source class-conditional Student-t mixture density, reconstructed from the exact checkpoint source features and required to match the P12 density-state hash.
- Geometry: `T(z) = diag(exp(a)) z + b`.
- Initialization: `a=0`, `b=0`, `pi_fit=source empirical prior`.
- Geometry optimizer: Adam, learning rate 0.05.
- Iterations: 20 outer iterations, three transform steps per iteration; no performance-based rollback or stopping.
- Regularization: `logdet_weight=1`, `trust_region_a=1`, `trust_region_b=1`, `prior_anchor_strength=1`, `dirichlet=5`.
- FP-GEM: source empirical prior remains fixed.
- Joint-GEM: responsibility prior M-step enabled; all other settings identical.
- Official SPDIM: 30 epochs, learning rate 0.01, parameter t=1.

## Checkpoint-Reuse Gate

Before fleet launch, one V100 job is frozen to `Lee2019_MI / target 1 / source seed 0`. It computes no accuracy or bAcc and must reproduce:

- checkpoint file SHA-256: `8ae4e3d0dd15dbe22a6133e36f4672ac6ebec4923f4239feae95c7916548d36d`
- source-only prediction SHA-256: `eef2221a3c8a95b2631ab98ac334624163144564ae89a3cb143844fd409d8964`
- source-only logits SHA-256: `1c9035576ae7389fc04d7803b75c038893f3302f2c5b56d99b7780faabd4ed78`
- all six q=0.5 P12 prediction/logits hashes
- both P12 GEM geometry hashes
- P12 source-density state hash

Any mismatch blocks P13. Fresh training is not permitted.

## Frozen Endpoints

Source seeds are first averaged within target subject and method.

Primary endpoint:

`0.5 * (abs(bAcc(q=0.1) - bAcc(q=0.5)) + abs(bAcc(q=0.9) - bAcc(q=0.5)))`

Primary comparison: `FP-GEM sensitivity minus Joint-GEM sensitivity`. The primary design claim is supported only if the paired percentile 95% CI is entirely below zero.

External comparisons, all reported separately:

- FP-GEM minus RCT sensitivity
- FP-GEM minus SPDIM geodesic sensitivity
- FP-GEM minus SPDIM bias sensitivity

Secondary endpoints are endpoint mean bAcc, worst-prevalence bAcc, prediction disagreement from q=0.5, GEM geometry displacement from q=0.5, Joint-GEM fitted-prior movement, and per-q mean bAcc for every method.

The bootstrap has 10,000 paired replicates, seed 20260710, and cluster unit `target_subject`. Every sampled subject preserves all methods and q values. No equivalence or noninferiority claim is allowed.

## Expected Artifacts And Counts

- Target-seed units: 162.
- Manifest occurrence rows: 24,300.
- Final result rows: 2,916 = 54 subjects x 3 seeds x 3 q values x 6 methods.
- New endpoint adaptation rows: 1,620.
- Reused P12 q=0.5 rows: 972.
- Reused source-only endpoint rows: 324.
- Geometry rows: 972.
- Seed-averaged subject-method rows: 324.

All units require complete prediction/logits hashes, exact checkpoint hashes, no duplicate keys, no adaptation/evaluation overlap, clean launch provenance, and accepted jobs absent from `squeue`. Completion is determined with `squeue` plus stdout/stderr and artifact parse/count/checksum validation.

## Frozen Provenance

| artifact | SHA-256 |
|---|---|
| P12 result CSV | `f3e4ca699b81e4fa2cab404109aa2dfe7aa1fbe58f25e2779d3d11651e40d48d` |
| P12 runner | `720b91b1b43cdf6a983be1cb8413430a06b98d6f4923166fa14614041ec46abd` |
| P12 config | `d44fd98aa5913eb45908b7fd398b04e5a268dd4aaa75f15bcc96819f424bf165` |
| repaired split semantic manifest | `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e` |
| external SPDIM commit | `1b0de0ccd4c48a4ff28f087b866a0b671b029c39` |
| P13 manifest CSV | `fb15b1e616178ba542dec9a1be2a16508c66761c731a0874191552adcfcef7a9` |
| P13 manifest JSON | `8c5b160fcec5ffeaded7faaf196f9753d7e0f7f15e583f8a18a5651ddf1c5802` |
| P13 manifest semantic | `29febb846ab5935dfed398953b28cbc2da86862842edf4c851a21515df71263f` |
| P13 config | `12acd01fbad33cdc5feadf2fe54da0c7423960ab6f1bfa7c8a7005ff76b87e2f` |
| P13 runner | `e5b4450e4e8bb9f715fd9ef4e12b6f26d415c9cac4592ab154430f6550903d9e` |
| P13 analyzer | `9986d890a24483dc6b3dfcd365eb0e0acdb8709fc63a7e396e73c93bcdd7da53` |
| checkpoint-gate launcher | `69ee059874fe5a6a4f5a235f66fe6c4e58042772f31ef65aa6a5e07d34093111` |
| fleet launcher | `efea558ba672e694d13bdb4bd007d1ca9e922d5cdce38659f09f55f4298fe6b0` |

Hardware replay groups are fixed from P12: 134 V100 units and 28 A100 units, with maximum concurrent GPU tasks 8.
