TALOS_00N - Negative Closeout

Status: TALOS_00B real frozen-feature adapter replay is accepted as a valid
real EEG negative / diagnostic-only readout. This closeout freezes the TALOS
low-degree-of-freedom frozen-feature route. It does not run a new experiment.

Frozen reference

```text
negative readout commit: 9ca1eb1 Add TALOS_00B real replay readout
preflight baseline:      b8dbb70 Add TALOS_00A preflight pipeline
design baseline:         96894a7 Add TALOS_00 project design
dataset:                 BNCI2014_001
backbones:               EEGNetMini, EEGConformerMini
seed:                    0
fold universe:           9 LOSO target folds per backbone
feature artifacts:       18/18 CEDAR_01F handoff artifacts
```

Gate summary

```text
TALOS_00A_PREFLIGHT: PASS
TALOS_00B_REAL_REPLAY: COMPLETE_NEGATIVE
feature artifacts: 18/18
handoff hash check: PASS
red-team failures: 0
red-team warnings: 0
target-label quarantine: PASS
adapter determinism: PASS
variant freeze: PASS
scientific gate: FAIL
P1: DENIED
P2: DENIED
source-free deployment claim: FORBIDDEN
generalization / safety claim: FORBIDDEN
```

Runtime artifacts are frozen under:

```text
results/talos/talos00b_bnci2014_001_seed0/
```

Canonical handoff hash:

```text
03d97199352892bf39afeed3ce826aa185735a766351b82aa2083587621d289c
```

TALOS_00B payload hash:

```text
68ae5140dd5131df69bcb8d17d6af4b973d3b56129480eeb1e2e31f81d3da199
```

Conclusion

TALOS_00B completed the approved real frozen-feature adapter replay over
BNCI2014_001 using EEGNetMini and EEGConformerMini CEDAR_01F feature artifacts.
All 18 feature artifacts passed handoff and hash validation. The approved
variant universe was limited to:

```text
ERM_NO_ADAPT
TTA_CONTROL_REPLAY
TALOS_L
TALOS_D
TALOS_LD
```

Target labels remained final-metric-only. Adapter fitting, predictions,
variant ordering, and pre-final artifacts were invariant to target-label
removal and permutation. The failure is therefore a scientific gate failure,
not a red-team or engineering failure.

Per-backbone conclusion

```text
EEGConformerMini:
  best clean TALOS variant: TALOS_L
  best clean TALOS delta vs ERM: +0.0021
  required clean delta: +0.020
  conclusion: insufficient clean effect

EEGNetMini:
  TALOS_L delta vs ERM: +0.0204, but boundary-hit in 1/9 folds
  TALOS_D delta vs ERM: +0.0575, boundary-hit in 9/9 folds
  TALOS_LD delta vs ERM: +0.0577, boundary-hit in 9/9 folds
  conclusion: gains are not clean under the trust-region contract
```

Scientific interpretation

The correct conclusion is narrow:

```text
current low-degree-of-freedom TALOS adapters do not cleanly explain or
reproduce the known TTA-Control positive signal on BNCI2014_001 frozen features
across EEGNetMini and EEGConformerMini.
```

Do not broaden this into:

```text
target-unlabeled EEG adaptation is impossible
TTA-Control is invalid
source-free adaptation is proven impossible
```

The historical TTA-Control positive signal remains an open mechanism question.
TALOS_00B shows that TALOS-L, TALOS-D, and TALOS-LD are not a clean sufficient
mechanistic explanation under the frozen trust-region contract.

Boundary-hit disposition

Boundary-hit gains are diagnostic-only. They cannot count as positive evidence
or justify P1:

```text
EEGConformerMini TALOS_D:  9/9 boundary-hit
EEGConformerMini TALOS_LD: 9/9 boundary-hit
EEGNetMini TALOS_L:        1/9 boundary-hit
EEGNetMini TALOS_D:        9/9 boundary-hit
EEGNetMini TALOS_LD:       9/9 boundary-hit
```

Frozen prohibitions

```text
No TALOS P1 from TALOS_00B.
No TALOS P2.
No TALOS-LR rescue.
No TALOS-full rescue.
No low-rank affine rescue.
No geometry-loss full variant rescue.
No trust-region relaxation on this run.
No target-informed hyperparameter rescue.
No CMI / pruning / mask / surgery / safety-gate reentry.
No source-free deployment claim.
No generalization, privacy, or safety claim.
```

Project state

```text
TALOS low-degree-of-freedom frozen-feature route: CLOSED_NEGATIVE
TALOS as P1 training method: NOT APPROVED
TALOS as source-free deployment method: NOT APPROVED
TALOS assets: RETAIN as diagnostic / red-team / adapter-replay infrastructure
```

Future work, if any, must be a new mechanism-audit protocol with a new charter.
It must not be framed as TALOS_01, CEDAR_02, or a rescue of TALOS_00B.
