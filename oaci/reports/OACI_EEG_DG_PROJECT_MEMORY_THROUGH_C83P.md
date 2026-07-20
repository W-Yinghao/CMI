# OACI EEG-DG Project Memory Through C83P

## Current Control Gate

```text
C83_AAAI_EVIDENCE_CLAIM_FIGURE_TABLE_FREEZE_READY_FOR_MANUSCRIPT_AUTHORIZATION
```

C83P is complete. This is a no-new-outcome evidence-freeze and manuscript-
readiness state, not manuscript authorization.

## Scientific Base

The latest valid scientific result remains:

```text
C82-D_zero_label_comparison_training_seed_method_identity_or_target_heterogeneous
```

Historical C81 remains:

```text
C81-E_protocol_input_implementation_or_provenance_blocker
```

C82 is a post-C81-outcome-access recovery. C83P does not alter either identity.

## C83P Chronology

```text
C82 final handoff/base:             34ae76a4a588059ab5a6d82b8116088a14af4ad5
C82 PM GitHub audit addendum:       5cee693132bc950d7e0ad9c3c9028e7cb1fcfcf7
C83 evidence-freeze implementation: c927b3a80bc1b75ed4d9ec0d7d2460342574ffe2
C83 readiness verification:         32abc29
```

The C82 PM addendum was committed and pushed before the C83 paper-facing
contracts. It is additive and changes no C82 result hash, table, number, or
gate.

## C82 Interpretation Addendum

### LOTO

The emitted C82 LOTO table is a seed-by-left-out-target panel. Its method
preservation rule starts from the full-panel cross-seed common method set.

```text
seed 3 category: B
seed 4 category: C
common B-method set: empty
global method-aware panels preserved: 7 / 16
required threshold: 12 / 16
```

Authoritative wording:

> The registered global method-aware stability rule preserved 7/16 panels. The
> full-panel categories already differed across seeds, and the common B-method
> set was empty.

The table is not a per-panel, per-method Q1 ledger. It does not by itself prove
that COTT independently lost Q1 in every seed-3 LOTO panel.

### Q5

Information-class membership was fixed. The displayed method is selected by
the observed minimum mean regret within that class. It must be called:

```text
descriptive best registered method within a fixed class
```

It is neither a prospectively fixed class representative nor an inferential
winner across methods.

## Frozen Scientific Result

```text
seed 3:
  COTT/U13 Q1 PASS
  no Q2 pass
  category B

seed 4:
  COTT positive and material mean direction
  7/8 favorable targets
  exact max-T p = 0.101167, Q1 FAIL
  no Q2 pass
  category C

A-method intersection: empty
B-method intersection: empty
global method-aware LOTO: 7 / 16
final: C82-D
```

Core values remain:

```text
S1 regret:          0.779476 / 0.804823
COTT regret:        0.338641 / 0.465335
COTT Q1 effect:     0.440835 / 0.339488
COTT Q1 max-T p:    0.015564 / 0.101167
Q0 B=1 regret:      0.353383 / 0.373705
COTT-Q0 difference: -0.014743 / 0.091630
Q2 upper bound:      0.144528 / 0.250901
COTT top-1:          0.1250 / 0.0000
COTT Spearman:       0.276605 / 0.184232
```

C80 remains:

```text
B*_seed3 = 1
B*_seed4 = 1
ordinal distance = 0
all 16 LOTO B* values are 2 or 4
```

The C80 frontier is full-panel, source-relative, Q0-policy-specific, and
leave-target sensitive.

## C83 Evidence Freeze

The authoritative machine artifacts are:

```text
claim contract:
  oaci/reports/C83_AAAI_CLAIM_CONTRACT.json

number registry:
  oaci/reports/c83p_tables/authoritative_number_registry.csv

figure registry and contracts:
  oaci/reports/c83p_tables/figure_data_registry.csv
  oaci/reports/C83_AAAI_FIGURE_CONTRACT.md

table registry:
  oaci/reports/c83p_tables/table_data_registry.csv

reference fidelity:
  oaci/reports/c83p_tables/baseline_reference_fidelity_appendix.csv

limitations/external validity:
  oaci/reports/C83_AAAI_LIMITATIONS_AND_EXTERNAL_VALIDITY_CONTRACT.json

reproducibility:
  oaci/reports/C83_AAAI_REPRODUCIBILITY_INDEX.json
```

Freeze counts:

```text
supported claims:                 10
forbidden expansions:             10
authoritative number identities: 928 / 928
main figures:                       4
publication figures rendered:      0
main tables:                        3
reference-fidelity rows:           34
input-unavailable methods:          5
reviewer risks:                    17
metadata checks:                   20 / 20 PASS
final red team:                    52 / 52 PASS
```

Every empirical figure/table value resolves to one number ID and then to a
committed source artifact, row key, and field. No paper-facing value is copied
from prose.

## Protected State

```text
new real-data statistics:       0
EEG arrays opened:              0
construction/evaluation views: 0
selection payload loads:        0
selection recomputation:        0
new methods or retuning:        0
target4 primary rows:           0
same-label oracle accesses:     0
training/forward/re-inference:  0
GPU:                            0
manuscript sections drafted:    0
```

No raw EEG, target-label array, model weight, optimizer state, or payload over
50 MiB was added to Git.

## Verification

```text
focused C83P:   26 passed                         job 895253
C65-C83P:      486 passed, 1 skip, 3 deselected job 895254
C23-C83P:      897 passed, 1 skip, 3 deselected job 895255
full OACI:   1,821 passed, 1 skip, 3 deselected job 895256
```

All four jobs used 48 CPU, 96 GiB, GPU 0 and completed with `ExitCode=0:0` and
empty stderr. The skip is the finalized C78F conditional test. The three
deselections are historical C79P preauthorization-state tests.

## Claim Boundary

C83P supports only field-, dataset-, split-, method-, and policy-specific
claims. It does not support:

```text
universal zero-label impossibility
universal one-label sufficiency
external validity or new-subject generalization
deployability
causal representation mechanism
COTT failure in general
OACI or SRC rescue
cross-regime selector transport
information-theoretic minimum label budget
training seed as independent population replication
```

## Next Authorization

C83P stops before manuscript drafting. A future direct PI statement must
separately authorize drafting from the C83 evidence freeze. No C84, new
experiment, seed 5, BNCI2014_004, active acquisition, target4 primary analysis,
or same-label oracle work is authorized.

