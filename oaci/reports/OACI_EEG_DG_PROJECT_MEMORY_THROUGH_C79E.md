# OACI EEG-DG Project Memory Through C79E

## Current gate

```text
C79-E_seed4_does_not_replicate_either_core_pattern
```

C79E completed the post-seed-3 prospective seed-4 training-seed robustness
audit. It is outcome-informed by C78S but was locked before protected seed-4
checkpoint outcomes. It is not pre-C78S confirmation, new-subject replication,
target-population confirmation, new-raw-data confirmation, or external
validation.

## Binding objects

```text
replacement protocol commit: ec4834c
protocol SHA-256:             e350b7f0c4ee3dfcf6b4f5651c1c7a0e8beac72e478ffb6c1e98e12df814f587
field lock:                   35d0c65
analysis lock:                7cebf2e
authorization record:         b67ba6c
field freeze:                 50232df
label provisioning:           6c3dc91
primary-output freeze:        cfd57cc
scientific red team:          439c8c5
result report:                7dee4be
audit namespace repair:       e48edda
final validation:             a12dc8b
```

The historical C79 protocol hash remains preserved but timing-invalid for
pre-C78S confirmation. C79P superseded it additively; C79E used only the
replacement protocol above.

## Seed-4 field

```text
complete units:                 1,458
primary units:                  1,296
target-4 engineering units:       162
ERM / OACI / SRC:            18 / 720 / 720
strict-source rows:           6,718,464
target-unlabeled rows:          839,808
construction rows:                2,235
evaluation rows:                  2,373
construction/evaluation overlap:      0
failed retained units:                  0
```

Target 4 is engineering-only. It enters no primary estimand, null,
multiplicity family, target count, cross-seed primary summary, or success rule.
All checkpoint/state/sidecar, optimizer, genealogy, cadence, Wz+b/logit,
softmax, hook-z, repeat-logit, and repeat-z identities pass. Target rows and
labels used in training, source-audit training rows, selector reads,
outcome-driven retention, and outcome-driven retry are all zero. The same-label
oracle was neither created nor opened.

Successful phase accounting records 8.715907 GPU-hours. Frozen external
payload is 30,109,676,522 bytes (about 28.04 GiB). Raw caches, model weights,
and optimizer states remain external.

## Scientific decisions

### P1 information-conditioned transition

```text
reliability:                  0.756456
target-bootstrap 95% CI:    [0.681795, 0.833763]
raw / Holm p:                 0.011673 / 0.070039
P1-M measurement gate:       false
construction top-1/5/10:     0.1250 / 0.5000 / 0.6875
random top-1/5/10:           0.0123 / 0.0617 / 0.1235
construction regret:         0.110667
regret reduction:            0.686781
P1-A actionability gate:     true
P1 compound replication:     false
```

Construction-label information remains materially actionable in its diagnostic
information class, but the compound P1 gate requires both measurement and
actionability. This is not source-only or target-unlabeled selection, zero-label
DG, OACI/SRC rescue, checkpoint recommendation, or deployability.

### P2 local association and transport

```text
local effect:                 0.210137
positive trajectory cells:   32 / 32
worst-control raw p:          0.092
Holm p:                       0.368
P2-L local gate:              false
fixed-kernel LOTO R2:        -0.098497
fixed-kernel LORO R2:        -0.032944
transport qualification:     false / false
P2 compound replication:     false
```

The descriptive local effect remains positive. The identity-only matched null
sets the worst control and the fixed family is inactive. This is not proof of a
zero association, causal representation mechanism, W-versus-z origin,
validated target gauge, universal nontransport, or control.

### Mandatory secondary audits

```text
H2R deviance reduction:       -8.717406; p=0.862; does not qualify
H4R strict-source F2 R2:      -0.096288; adjusted p=1; does not qualify
H5R target-unlabeled F4 R2:    0.010450; adjusted p=1; does not qualify
H6R positive-control effect:   0.415635; raw/Holm p=0.011673/0.070039; inactive
```

H2R/H4R/H5R are exact-candidate nonqualification results only. They do not
establish universal irrelevance, universal impossibility, equivalence to zero,
or absence of representation information. H6R is descriptive and cannot be
rescued with its raw p-value.

## Cross-seed interpretation

All registered aggregate effects retain their seed-3 direction. P2-L and
P2-overall are the only gate-discordant paths. P2 local changes from 0.242656
to 0.210137; the paired-target difference is -0.032519 with 95% bootstrap CI
[-0.061609, 0.004581]. The interval includes zero, while the fixed gate changes.
Use "gate-level training-seed heterogeneity," not "significant effect-size
heterogeneity."

No p-values were combined across seeds. Seed 3 and seed 4 share targets and raw
trials and are paired/repeated measurements, not independent target samples.

## Provenance and validation

Six execution repairs and one regression audit-namespace repair are retained
additively. None changed a scientific registry, estimand, model, kernel,
threshold, null, multiplicity family, locked implementation, or scientific
outcome decision.

```text
authorization preflight: 24 / 24 PASS
scientific red team:      17 / 17 PASS
final-report red team:    27 / 27 PASS
focused regression:      38 passed
C65-C79E:               315 passed, 1 conditional skip
C23-C79E:               722 passed, 1 conditional skip
full OACI:            1,650 passed, 1 conditional skip
```

The first complete regression attempt retained three failures caused by C79E
sharing two Mode-R audit filenames. The additive repair restored exact Mode-R
hashes, moved C79E ledgers to `c79e_*`, and the complete rerun passed. No
scientific output changed. A third complete run on the final report/handoff
HEAD also passed; all earlier attempts are retained.

## Stop boundary

Stop for PM review. C79E authorizes nothing beyond itself. Do not automatically
start C80, seed 5, BNCI2014_004, new targets, same-label-oracle analysis, new
feature/kernel/model search, checkpoint recommendations, deployability work, or
manuscript drafting.
