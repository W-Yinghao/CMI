# S2P_28 - CBraMod Cross-Task Red-Team Contract

**Phase:** C0-C2. **Status:** FROZEN. This document defines fail-closed checks, not alternative analyses.

## Provenance attacks

Phase C stops before endpoints if any of the following is true:

1. the immutable closure manifest does not contain the ordered ten-object contract;
2. a physical payload SHA256 differs before or after feature extraction;
3. a payload is writable, symlinked, or resolved through a mutable logical path;
4. strict model loading has missing or unexpected keys;
5. random initialization differs from the frozen architecture/seed contract;
6. released CBraMod is used to select preprocessing, rank, probe, head, layer, metric, or gate threshold;
7. feature extraction is not bitwise deterministic on a pinned dataset-specific checksum batch.

The historical mutable source paths in the closure manifest are provenance records only. Phase C runners may
read only `immutable_path`.

## SEED-V leakage attacks

The SEED-V gate fails if:

1. a key does not uniquely identify subject, session, trial, and window;
2. a trial appears in more than one train/val/test split;
3. a trial contains more than one class label;
4. windows from one trial are split across a fit/holdout fold;
5. long trials receive more endpoint weight merely because they contain more windows;
6. a window is treated as an independent bootstrap or inferential unit;
7. session is treated as independent of subject;
8. the same-subject trial split is described as unseen-subject transfer;
9. native62 channel order is absent, variable, or silently remapped.

Trial-mean features are the frozen analysis unit. A window-level sensitivity cannot replace the primary result.

## ISRUC leakage and chronology attacks

The ISRUC gate fails if:

1. any object other than official Cohort III is used;
2. a subject, epoch, label, sequence, or rotation differs from the accepted recovery manifests;
3. a 20-epoch sequence is reordered, overlapped, padded, or split across fit/holdout;
4. train, validation, and test subjects overlap in any rotation;
5. flat-epoch classification replaces the frozen sequence-aware head;
6. a failed head is followed by a post-result architecture or hyperparameter change;
7. validation or test labels enter encoder preprocessing, representation selection, or geometry rank selection;
8. ten test subjects are converted into thousands of independent epoch observations for inference;
9. native6 channel order is absent or changed.

With ten biological units, wide or unstable uncertainty is an allowed result. ISRUC_S1 or another sleep dataset
cannot silently replace a failed ISRUC_S3 gate.

## Endpoint and selection attacks

The prospective primary task metric for both new datasets is Cohen's Kappa. NLL, balanced accuracy, and weighted
F1 remain mandatory secondary endpoints. The analysis fails closed if:

1. Kappa is replaced after seeing NLL or accuracy;
2. the dataset test endpoint selects PCA dimension, probe C, sequence-head architecture, optimizer, epoch count,
   checkpoint, layer, rank, or normalization;
3. a best pretraining seed is reported in place of the budget mean;
4. SEED-V and ISRUC p-values are pooled into a global EEG significance claim;
5. FACED's retrospective primary target NLL status is rewritten because Phase C prospectively uses Kappa;
6. Kappa/NLL disagreement is hidden rather than reported as endpoint discordance.

## Geometry attacks

The equal-rank contract is rank four for both new five-class tasks. It fails closed if:

1. rank changes by model, budget, seed, fold, or observed result;
2. subject and task subspaces use unequal rank;
3. the primary overlap changes from normalized projection overlap;
4. target-test labels fit either subspace;
5. held-out self-captured energy is below 0.05 and the overlap is still interpreted;
6. absolute overlap is compared directly between native62 SEED-V and native6 ISRUC;
7. a failed overlap is rescued with CKA, RSA, a nonlinear probe, or a layer search.

Phase C contains no variance partition. The Phase-B variance family remains closed as unstable and cannot be
reintroduced under a new dataset without a separate protocol.

## Functional reliance attacks

L4/L5/L6 are interpretable only after the frozen task gate. The family fails closed if:

1. subject erasure and random-null erasure do not match removed source-validation energy;
2. target data determine null rank, direction, partial coefficient, or energy target;
3. fewer or different nulls are used selectively by object;
4. Holm correction excludes an unfavorable task-gated cell;
5. non-significant L5 is described as subject identity being harmless or unused in every possible sense;
6. SEED-V is described as unseen-subject reliance;
7. ISRUC low power is hidden or its test epochs are treated as independent subjects.

## Compute and launch attacks

Every SLURM entry must resolve the repository from explicit `S2P_REPO_ROOT` or a valid `SLURM_SUBMIT_DIR`, reject
`/var/spool`, pin the expected Git commit, require a clean tracked worktree, and expose a dry-run mode. It must not
contain an auto-chain to encoder training or fine-tuning.

Before any gate job:

```text
SLURM stability gate passes
submit/query/cancel canary passes
local HEAD equals remote project/s2p-subject-scaling
C0 go/no-go passes
```

Gate and fleet are distinct invocations. A gate pass cannot launch the other dataset. A gate failure cannot be
worked around by changing partition, data, head, metric, split, channel mapping, or object subset.

## Authorized claim boundary

Phase C may classify the FACED finding as FACED-specific, affective-domain general, or directionally cross-task.
It may report task-dependent L5 or endpoint discordance. It may not claim:

```text
general EEG scaling law
subject invariance
causal harmlessness of subject identity
CodeBrain replication
encoder fine-tuning performance
three independent unseen-subject replications
```

Any stop-rule event is persisted with the triggering object, dataset, check, expected value, observed value, and
code commit. It is not silently retried with altered scientific settings.
