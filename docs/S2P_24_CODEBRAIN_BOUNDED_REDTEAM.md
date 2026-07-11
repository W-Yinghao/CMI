# S2P_24 - Bounded CodeBrain Red-Team

**Status:** pretraining launch held pending SLURM stability and tokenizer-target preflight.

> Post-red-team closure: the missing ISRUC_S3 asset was recovered from the official Cohort III source. Ten source
> archives and ten extracted-channel MAT files are size/SHA pinned; 10/10 label/MAT pairs align after the official
> 30-epoch tail exclusion; and the processed 425-sequence tree passes a full content hash. The incompatible flat
> four-fold LMDB and Group-I tree remain forbidden. See S2P_25.

## Adversarial findings

### 1. A fixed released tokenizer weakens the low-budget interpretation

The tokenizer encodes information learned outside the bounded Stage-2 subsets. Any H200 representation is a
low-budget EEGSSM conditional on a mature target generator. It cannot establish when full CodeBrain first learns
subject structure.

### 2. Budget and optimizer updates are coupled

Ten epochs over larger subsets increase unique windows, updates, and total presentations. Results are budget
responses, not unique-data causal effects. Update counts must be reported, and no comparison may be called a data
scaling law.

### 3. Local TUEG is not the paper preprocessing substrate

The current processed corpus and paper differ in filtering and record inclusion. The bounded test is a controlled
cross-architecture audit on local 19-common data, not a 9246h/raw-corpus reproduction.

### 4. Temporal target collapse is a hard failure

Prior downstream datasets showed temporal-code collapse. Aggregate dual CE could hide a collapsed temporal stream
behind a healthy frequency stream. Utilization, entropy/perplexity, dominant share, and sequence diversity are gated
per stream and per budget stratum. Temporal failure stops the test.

### 5. Native Stage-2 lacks the required validation contract

The upstream trainer saves on training loss and has no subject-disjoint pretrain validation. The scientific runner
must add validation/checkpoint closure without altering target generation, masking, or loss. Downstream labels cannot
select the epoch.

### 6. The native implementation contains device and batch-size traps

`SSSM` has hard-coded CUDA operations, the upstream DataParallel IDs are fixed, and the codebook forward uses a
`squeeze` path that is unsafe at batch size one. The bounded runner must use explicit in-allocation `cuda:0`,
`drop_last` or an asserted batch size >=2, and no unverified multi-GPU rewrite.

### 7. Window sampling can silently change the population

Uniform unique-window sampling weights long recordings more heavily and does not isolate subject diversity. Nested
prefixes and per-budget subject/recording/exposure diagnostics make this visible. No subject-count claim is allowed.

### 8. FACED's upstream model wrapper can overwrite loaded weights

`Models/model_for_faced.py` loads the pretrained backbone and then calls `self.apply(init_weights)`, which reaches
backbone linear layers. A direct run can therefore reinitialize part of the loaded encoder. The bounded downstream
adapter must initialize only the new head or load the backbone after head initialization, then verify every backbone
parameter against the selected checkpoint before the first optimizer step.

### 9. The upstream fine-tuning trainer repeatedly evaluates test

The native trainer scores test whenever validation Kappa improves. Even if test does not directly select the model,
repeated observation violates this project's target firewall. The unified adapter must save by source validation and
evaluate test exactly once after selection.

### 10. Prefix stripping is not a provenance contract

Downstream wrappers blindly remove seven characters from every checkpoint key, assuming `module.`. Each checkpoint
must be schema-audited; prefix removal occurs only when every key has that prefix. Strict load with zero missing and
unexpected keys is mandatory.

### 11. SEED-V is not an unseen-subject replication

The local LMDB has the same 16 subjects and sessions in train/val/test, split by trial IDs. Subject/session geometry
there is descriptive. It cannot be pooled with FACED's cross-subject evidence.

### 12. ISRUC sequence construction cannot be inferred from a flat LMDB

The currently located flat ISRUC store declares 89,283 numeric epochs and a four-fold train/val/test index manifest.
That manifest is internally disjoint, but it is not the required Group-III ten-subject rotating 8:1:1 contract and
does not expose subject identity, epoch chronology, or 20-epoch sequence boundaries. The native loader expects
paired, subject-named sequence and label files. Unless a ten-subject ISRUC_S3 tree is found or rebuilt from
provenance-complete raw data, ISRUC_S3 remains a launch blocker. Group-I 100-subject assets cannot be silently
renamed as subgroup III.

The upstream ISRUC loader also zips unsorted sequence and label directory listings. A valid adapter must pair files
by exact stem and verify labels before splitting.

### 13. Released CodeBrain is only an external reference

Its tokenizer/encoder provenance, local preprocessing, and budget are unmatched. It validates paths and establishes
a descriptive reference band; it cannot define a budget threshold or choose metrics/hyperparameters.

### 14. Architecture comparison is substrate-conditional

CBraMod Route B used a fixed 33-channel heterogeneous-reference mixture; bounded CodeBrain uses 19 common LE
channels. Similar geometry supports robustness across these full pipelines, not an architecture-only causal effect.

## Fail-closed preflight

Stage-2 training remains NO-GO if any of these is false:

```text
CodeBrain source and tokenizer SHA pinned
200/1000/2000 exact unique-window nested budgets
subject-disjoint pretrain validation
temporal and frequency target gates
native target/logit shape and finite-loss canary
FACED split/shape contract
SEED-V split/shape contract
ISRUC_S3 subject/sequence contract
released/random downstream path gates designed and runnable
target-label firewall clean
```

The feasibility job has no trainer call, backward pass, optimizer step, downstream fit, or child submission.

## Training stop rules

If later authorized, stop immediately on any of:

```text
tokenizer or source SHA drift
target utilization drift below the frozen gate
wrong H/seed/subset mapping
window reuse or train/val subject overlap
input shape other than [B,19,30,200]
normalization other than microvolt then native /100
temporal/frequency objective weighting change
non-finite CE or gradients
checkpoint selection using train loss or downstream labels
mutable checkpoint downstream use
downstream backbone mismatch after load
test evaluation before final source-val selection
```

## Disposition

```text
full 1k-9k replication: deferred separate project
bounded Stage-2 training: held
preflight-only audit: approved
new CBraMod training: prohibited
B2 layerwise: prohibited
manuscript writing: prohibited
```
