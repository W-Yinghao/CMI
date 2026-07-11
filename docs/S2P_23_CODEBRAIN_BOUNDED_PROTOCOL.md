# S2P_23 - Bounded CodeBrain Cross-Architecture Protocol

**Status:** FROZEN candidate protocol; preflight only. Project OPEN. Manuscript writing OFF.

> Post-freeze asset closure: the official ISRUC Cohort III 10-subject payloads were recovered and validated, then
> processed into 425 paired 20-epoch sequences under the pinned native contract. The full tree hash is rechecked by
> every preflight. This closes the ISRUC asset blocker but does not authorize Stage-2 training.

## Scientific question

> Does the representation-emergence pattern observed with CBraMod remain detectable under a different EEG
> architecture and pretext objective?

The test is explicitly not phrased as replication of an established `subject-first/task-later` mechanism. CBraMod
established early and continuing subject strengthening, declining measured subject-task overlap, and no detected
task-gated L5 effect beyond a matched null. Its primary target NLL did not establish a later task gain.

## Scope

```text
name: S2P-CodeBrain-Bounded
tokenizer: released TFDual-Tokenizer, frozen
encoder: native EEGSSM Stage-2, 8 layers, hidden size 200
objective: native masked temporal-token CE + frequency-token CE
mask: native (B,19,30), ratio 0.5
budgets: 200 / 1000 / 2000 hours
pretraining initialization seeds: 0 / 1
downstream: FACED / SEED-V / ISRUC_S3
evaluation: frozen audit and unified full fine-tuning, kept separate
```

No tokenizer retraining, model-size scaling, masking rewrite, 33-channel substitution, H4000 CBraMod run, or full
CodeBrain 1k-9k replication is in scope.

The released tokenizer has already learned from an externally trained corpus. H200 therefore means `200h of
Stage-2 EEGSSM training conditional on the released tokenizer`, not a fully low-budget CodeBrain model.

## Source contract

- CodeBrain source authority: `${CODEBRAIN_ROOT}`, commit pinned by the preflight manifest.
- Frozen tokenizer: `CodeBrain_Tokenizer.pth`, SHA256 pinned before every gate/run.
- Released EEGSSM: `CodeBrain.pth`, external path-validity reference only.
- Native target API: `TFDual.get_codebook_indices` under `eval()` and `no_grad()`.
- Native Stage-2 output: two 4096-way logits at the positions selected by the 3-D mask.
- Native normalization: processed volts are converted to microvolts, then the model receives native `/100` input.

Local execution paths are runtime-only bindings and are never expanded into committed reports:

```text
S2P_REPO_ROOT
S2P_PYTHON
CODEBRAIN_ROOT
CODEBRAIN_TOKENIZER_PATH
CODEBRAIN_RELEASED_PATH
FACED_ROOT
SEEDV_ROOT
ISRUC_FLAT_ROOT
ISRUC_PROCESSED_ROOT
```

The processed TUEG substrate is not the paper corpus reproduction. It is the existing `4704743c` processed corpus,
whose preprocessing differs from the paper's stated 0.3-75 Hz/notch/raw-corpus contract. All claims are conditional
on this local substrate.

## Data-volume contract

The substrate is the canonical 19 common LE channel order, 200 Hz, 30 s windows, 30 one-second patches. The fixed
128-subject pretrain-validation pool contributes 24 windows per subject and is disjoint from every train budget.

Training budgets are exact prefixes of one PCG64 permutation over every unique train window after val-subject
exclusion:

```text
H200  subset H1000 subset H2000
no replacement
one fixed subset seed
the same data subset for initialization seeds 0 and 1
```

This makes the two training seeds optimization/init replicates rather than combined data-and-init replicates.
Population coverage, per-subject exposure, recording coverage, unique-window digests, and optimizer update counts are
reported for every budget. Subject count is not isolated.

Stage-2 uses 10 epochs. Larger budgets therefore imply both more unique windows and more optimizer updates. The
bounded test may report a `budget response`; it cannot assign that response solely to unique data or call it a
scaling law.

## Tokenizer target gate

Before any Stage-2 training, temporal and frequency targets are measured separately on 256 windows from each of:

```text
H200 prefix
H1000 \ H200 increment
H2000 \ H1000 increment
fixed pretrain validation
```

Every stream and stratum must satisfy all frozen thresholds:

```text
shape = [N,570]
ids in [0,4096)
unique codes >= 16
effective perplexity >= 8
dominant-code fraction <= 0.95
unique target sequences >= max(16, 0.05*N)
```

A second identical tokenizer forward must produce exactly equal targets. A native SSSM canary must show input
`[8,19,30,200]`, mask `[8,19,30]`, matching masked target/logit counts, 4096 logits, and finite separate temporal
and frequency CE. The canary performs no backward or optimizer step.

Temporal failure stops the bounded experiment. A frequency-only effective target is not accepted as CodeBrain.

## Stage-2 training gate

Training remains held until a later PM decision and all preflight blockers are closed. A future trainer adapter may
add subject-disjoint validation and immutable checkpoint closure, but it must preserve the native objective, mask,
tokenizer target semantics, and ten-epoch schedule. Selection is by pretrain-val dual CE only. FACED/SEED-V/ISRUC
labels cannot choose a checkpoint.

Every selected checkpoint must be copied to a no-overwrite content-addressed path, SHA256 pinned, strict-reloaded,
and feature-equivalence checked before downstream use.

## Downstream roles

### FACED

Primary cross-subject mechanism dataset. Native 32-channel, 10 s, nine classes; fixed subject split 1-80 / 81-100 /
101-123. Run frozen subject/task/geometry, task-gated L5, and the separate unified fine-tuning control.

Frozen confirmatory family mirrors CBraMod B1-Core:

```text
P1 random -> H200 continuous subject metric
P2 H200 -> pooled(H1000,H2000) target NLL
P3 H200 -> pooled(H1000,H2000) rank-8 subject-task overlap
```

Kappa and balanced accuracy are frozen sensitivities, not replacements for target NLL.

### SEED-V

Same-task-domain external validation. Its local split contains the same 16 subjects/sessions on all sides and divides
trials 0-4 / 5-9 / 10-14. It does not confirm unseen-subject reliance.

### ISRUC_S3

Cross-task sleep validation. Use the native six-channel, 20-consecutive-epoch context and a pre-frozen one-layer
Transformer sequence head. The native ten-fold contract rotates an 8:1:1 subject split: one fold subject is
validation and the next subject is test. Ten-subject clustering has low power and is interpreted accordingly. A
flat epoch store without auditable Group-III subject, chronology, and sequence boundaries is not a valid substitute.

## Evaluation tracks

### Track A - Frozen representation audit

Every dataset includes deterministic random init, released CodeBrain, H200, H1000, and H2000. Probe/subspace fitting
is source-only; target labels are final scoring only. FACED carries the full mechanism family. SEED-V and ISRUC_S3
carry dataset-appropriate diagnostics without inheriting FACED's cross-subject claim.

### Track B - Unified full fine-tuning control

Use one dataset-specific protocol across random, all bounded checkpoints, and released CodeBrain, with five
downstream seeds. Select checkpoints by source validation only and score test once after selection. Cohen's Kappa is
the multiclass primary endpoint. Frozen and fine-tuning endpoints remain separate.

## Cross-architecture hypotheses

```text
H-CB1 subject: random -> H200 improves the frozen continuous subject metric.
H-CB2 geometry: H200 -> higher budgets reduces rank-8 subject-task overlap.
H-CB3 task: frozen target NLL and fine-tuned Kappa are tested independently.
```

Architecture-general support requires H-CB1 and H-CB2 plus no detected task-specific L5 reliance, or an explicit
task boundary. Task emergence is not required because CBraMod's primary target NLL did not establish it.

## Claim boundary

Allowed:

> CodeBrain Stage-2 with a fixed released tokenizer shows, or does not show, the same measured subject strengthening
> and subject-task geometric separation as CBraMod on the local 19-channel processed corpus.

Forbidden:

```text
full CodeBrain scaling-law replication
full low-budget CodeBrain pretraining
1k-9k paper reproduction
tokenizer-plus-encoder emergence
subject invariance
causal subject reliance from overlap alone
monotonic scaling law
```

## Current launch rule

The feasibility-only entry may generate metadata, target-utilization, native-shape, and downstream-asset diagnostics.
It cannot launch Stage-2 or downstream training. Stage-2 remains PM-held even if every preflight item passes.
