# S2P Phase C raw cross-task fleet results

This is a technical result and provenance record, not a manuscript draft. The raw ten-object frozen-readout
fleets are complete for SEED-V and ISRUC_S3. Confirmatory uncertainty closure is still pending where stated.

## Execution integrity

SEED-V used the frozen native-62-channel trial-mean path. ISRUC used all ten Cohort-III rotations, three fixed
downstream-head seeds, and A40-only head fitting after mixed-hardware attempt `895321` was excluded. All ISRUC
objects report `NVIDIA A40`, compute capability `8.6`, frozen encoders, source-validation-only epoch selection,
and final-score-only target labels. The ISRUC aggregate independently reproduced prediction metrics, payload
hashes, 200-null L5 arithmetic, source-validation variance matching, and Holm correction.

## SEED-V descriptive fleet

Budget means over the two pretraining seeds were:

| Budget | Test Kappa | Test NLL |
| ---: | ---: | ---: |
| 200h | 0.1489 | 2.3807 |
| 500h | 0.1941 | 2.3458 |
| 1000h | 0.2527 | 2.1212 |
| 2000h | 0.2394 | 1.9805 |

The random and released Kappa references were `0.1383` and `0.2394`. Subject-probe NLL changed from `0.3771`
at random initialization to a mean of `0.0705` at H200 and approximately `0.037-0.040` at higher budgets.
Rank-4 subject-task overlap was low for all pretrained objects. All eight task-gated pretrained cells failed to
exceed the variance-matched random-subspace L5 null after Holm correction. The budget response is not monotonic,
SEED-V is not an unseen-subject replication, and the pre-registered 5,000-replicate subject-cluster bootstrap and
independent fleet verification remain pending.

## ISRUC_S3 descriptive fleet

Budget means over the two pretraining seeds were:

| Budget | Test Kappa | Test NLL | Balanced accuracy |
| ---: | ---: | ---: | ---: |
| 200h | 0.4236 | 1.0832 | 0.4988 |
| 500h | 0.5646 | 0.8734 | 0.6269 |
| 1000h | 0.6055 | 0.7869 | 0.6660 |
| 2000h | 0.6235 | 0.7660 | 0.6784 |

The random and released Kappa references were `0.5314` and `0.6910`. H200 remained below random, while the
descriptive budget means increased through H2000. Subject-probe NLL fell from `1.3296` at random initialization to
`0.5274` at H200, `0.3021` at H500, `0.2410` at H1000, and `0.2481` at H2000. Mean rank-4 subject-task overlap
fell from `0.0621` at H200 to `0.0281`, `0.0211`, and `0.0219` at H500, H1000, and H2000. All rotation-level
geometry stability checks passed.

H200 did not pass the task gate. Among the six task-gated pretrained cells, the subject-subspace intervention
exceeded the matched null after Holm correction for `H500_s0`, `H1000_s0`, and `H2000_s1`, but not for their
paired training seeds. Leave-one-test-subject-out subject-minus-null effects stayed positive for each of these
three cells, while individual test-subject effects were heterogeneous. This is evidence that detectable linear
subject-subspace reliance can be task- and seed-dependent on ISRUC; it does not establish universal reliance or
contradict the SEED-V/FACED null findings.

## Current boundary

Raw cross-task compute and ISRUC aggregate verification are complete. Phase C remains open until the frozen
confirmatory uncertainty and independent-verifier package is complete. No encoder training, encoder fine-tuning,
H4000, CodeBrain Stage-2, additional downstream dataset, Phase B2, or manuscript work is authorized by these
results.
