# S2P_31 - Phase C Cross-Task Closure Results

**Status:** COMPUTE COMPLETE / INDEPENDENT VERIFIER PASS / READY FOR PM CLOSURE REVIEW.

**Code authority:** `c86e0a2158ef068198ce450e8fd735b86a94b013`.

This document is a technical result record. It is not a manuscript, abstract, or submission narrative. Phase C
uses the ten immutable Route-B representation contracts and performs no encoder training, fine-tuning, H4000,
CodeBrain Stage-2 work, or checkpoint selection.

## Closure checks

```text
Raw FACED / SEED-V / ISRUC fleet:       COMPLETE
SEED-V subject-cluster bootstrap:       PASS, 5,000 replicates
Independent fleet verifier:             PASS, failures=[]
Immutable representation hashes:        10/10 rechecked
Target-label selection firewall:        PASS
Best-seed / best-budget selection:       NONE
Cross-dataset pooled p-value:            NONE
Monotonic scaling law established:       NO
Project-close recommendation:            NO
Manuscript writing authorization:        NO
```

SEED-V bootstrap samples the 16 subjects as biological clusters. Sessions and trials remain nested inside each
sampled subject; windows are never inferential units. Pretraining seeds are averaged within each budget.

## SEED-V confirmatory inference

### Task endpoints

| Contrast | Kappa delta | 95% cluster CI | Holm p | NLL improvement | 95% cluster CI | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H500 vs H200 | +0.04521 | [-0.02344, 0.11436] | 0.10158 | +0.03496 | [-0.27129, 0.31365] | 0.37872 |
| H1000 vs H200 | +0.10372 | [0.04687, 0.16406] | 0.00160 | +0.25958 | [0.02495, 0.49398] | 0.04559 |
| H2000 vs H200 | +0.09043 | [0.03646, 0.14323] | 0.00180 | +0.40027 | [0.21176, 0.59831] | 0.00080 |
| Pooled higher vs H200 | +0.07979 | [0.03299, 0.12319] | 0.00180 | +0.23160 | [0.00837, 0.45255] | 0.04559 |

The prospective Kappa endpoint and target NLL both support task accessibility beyond H200 at the pooled-higher
contrast. H500 alone does not clear its cluster interval. The Kappa means are `0.14894`, `0.19415`, `0.25266`,
and `0.23936` for H200, H500, H1000, and H2000, respectively; the response is non-monotonic and does not license
an optimum or scaling-law claim.

### Subject and geometry endpoints

| Contrast | Point | 95% cluster CI | Holm p | Status |
| --- | ---: | ---: | ---: | --- |
| random - H200 subject NLL | +0.30653 | [0.21846, 0.39971] | 0.00040 | supported |
| H200 - pooled-higher subject NLL | +0.03231 | [0.02236, 0.04299] | 0.00040 | supported |
| pooled-higher - H200 rank-4 overlap | -0.000568 | [-0.003881, 0.000758] | 0.77025 | not established |

SEED-V therefore supports early and continuing subject strengthening, but it does not replicate the FACED
subject-task overlap decrease under the frozen rank-4 instrument.

### Functional reliance

All eight pretrained cells entered the frozen SEED-V L5 Holm family. Zero of eight subject-subspace
interventions exceeded the source-validation-energy-matched random-subspace null. This is a cohort/session
diagnostic, not an unseen-subject non-reliance claim.

## FACED anchor verification

The fleet verifier independently recomputed target Kappa and balanced accuracy from the saved target predictions,
and recomputed B1-Core subject NLL, target NLL, and rank-8 geometry from the saved support artifacts.

```text
Early subject structure (P1):                 SUPPORTED
Primary target-NLL later gain (P2):           NOT ESTABLISHED
Rank-8 subject-task overlap decrease (P3):    SUPPORTED
Task-gated L5 above matched null:              0/8
Kappa/bAcc budget gains:                       retained as sensitivity evidence
```

The Phase-C ledger does not use FACED Kappa sensitivity to overwrite the frozen primary target-NLL null.

## ISRUC_S3 verification

Prediction payloads reproduce the aggregate Kappa means:

```text
random      0.53137
H200        0.42362
H500        0.56457
H1000       0.60551
H2000       0.62349
released    0.69104
```

Rotation-mean subject NLL decreases from H200 to the higher budgets, and rotation-mean rank-4 subject-task
overlap also decreases. These are low-power directional findings over ten rotating test subjects, not pooled
high-powered inference.

The independent verifier reproduces three Holm-positive L5 cells:

```text
H500_s0
H1000_s0
H2000_s1
```

No budget has both training seeds positive. The frozen disposition is therefore
`TASK_AND_SEED_DEPENDENT_RELIANCE_SIGNAL`, not general sleep-stage subject reliance and not a budget trend.

## Cross-task disposition

```text
Subject strengthening:
  FACED supported; SEED-V supported; ISRUC directionally supported.

Subject-task overlap decrease:
  FACED supported; ISRUC directionally supported; SEED-V not established.

Task accessibility:
  FACED primary NLL null with positive Kappa/bAcc sensitivity;
  SEED-V Kappa and NLL pooled-higher contrasts supported;
  ISRUC directional budget emergence supported.

Functional subject reliance:
  not detected on FACED or SEED-V;
  task- and training-seed-dependent signal on ISRUC.
```

Consequently, Phase C does not license a universal geometric-separation claim, a universal non-reliance claim,
a monotonic scaling law, or a pooled EEG significance result.

## Numerical replay boundary

Independent SEED-V refitting preserved every discrete task prediction and every gate decision. CPU/BLAS replay
produced continuous differences below `1e-4` for probe outputs and `1e-5` for geometry. One H2000_s0 null draw
crossed an exact discrete-Kappa tie, shifting the empirical p-value by one of 201 count units; its Holm value and
cell inclusion were unchanged. The verifier records these fixed tolerances and still requires removed-energy
matching error at or below `1e-10`.

## Authoritative artifacts

The closure package is under `results/s2p_route_b_phase_c_closure/`:

```text
seedv_bootstrap_budget_metrics.csv
seedv_bootstrap_task_contrasts.csv
seedv_bootstrap_subject_contrasts.csv
seedv_bootstrap_geometry_contrasts.csv
seedv_bootstrap_l5.csv
seedv_bootstrap_point_reproduction.csv
seedv_bootstrap_support.npz
seedv_bootstrap_verification.json
phase_c_independent_checks.csv
phase_c_claim_ledger.csv
phase_c_dataset_verdicts.json
phase_c_closure_verdict.json
phase_c_closure_artifact_manifest.csv
```

`phase_c_closure_verdict.json` marks Phase C as ready for PM closure review. It does not close the overall project,
authorize a new experiment, or authorize manuscript work.
