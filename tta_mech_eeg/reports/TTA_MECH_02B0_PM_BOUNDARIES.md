TTA_MECH_02B0 - PM Boundaries

Frozen prior decision:

```text
TTA_MECH_01S_SYNTHESIS: PASS
accepted_commit: 7e0ddc4
```

Allowed in 02B0:

```text
artifact inventory
condition registry
protocol
red-team contracts
read-only feasibility reporting
```

Forbidden in 02B0:

```text
TTA_MECH_02B real run
real EEG forward
BN refresh
target metric computation
new adaptation baseline
new adapter
target-informed condition selection
deployment baseline selection
P1/P2 training
TALOS-LR rescue
CEDAR surgery rescue
CMI objective
CutClean / pruning / mask / surgery
source-free deployment claim
```

CutClean boundary

TTA-MECH does not use privacy heads, structured pruning, sparsity grids,
MI-aware training objectives, admissible model selection, or CutClean-for-EEG
framing. These remain outside the active code path and outside the report
interpretation.

Interpretation boundary

`TTA_MECH_02B_NOT_FEASIBLE_FROM_CURRENT_ARTIFACTS` is not a scientific BN
negative result. It means the present artifacts cannot support a strict BN /
normalization audit without new PM approval for additional artifacts or a new
preflight.
