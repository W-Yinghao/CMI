# OACI EEG-DG Project Memory Through C85VP

## Current State

```text
milestone:
  C85VP

gate:
  C85V_INDEPENDENT_PROOF_REVIEW_IMPLEMENTED_AND_LOCKED_READY_FOR_PI_AUTHORIZATION

C85V lock commit:
  3c732489407ebca7603e5fb65d03c1ae25d046b6

C85V lock SHA-256:
  35cd029ba9cf68599a53d3f23db7a7c0a721440d9fb79be88a084548e452b20f

C85V authorized:
  false

C85E authorized:
  false
```

C85VP is protocol and validator-lock readiness only. T1-T7 remain `OPEN`.

## Preserved C85T Evidence

The accepted C85T V3 bundle remains immutable at:

```text
/projects/EEG-foundation-model/yinghao/oaci-c85t-synthetic-v3/
  c85t-v3-3ee51a994969ebaa-9ec012bedbf24f1f
```

```text
result SHA-256:
  ecaff65e942dbb81d93a3bdb61589fa9f1f6590f7188947688e6b30617140cec

manifest SHA-256:
  a727beebcb45598ea0f92f37bed8ef32369b1c793ecad9efc2f5d9941bd5bb0e

semantic receipt SHA-256:
  735edf13a24c074cb3c18e56d168ebd905b3a7bcb29e3c273b3652bb1b7dcc6e

completion receipt SHA-256:
  418f74e4c3cf60847b11bf18a890ffebf870ed8adee1a75d304b01075646e65d
```

C85VP did not rerun S0-S10 or read Monte Carlo arrays as proof.

## Protocol Chronology

```text
C85VP protocol commit:
  436d6ff6a3710cd9a3c75cf2f22d0306a10f2d40

C85VP implementation commit:
  a0beda65ee7db2d0a68f9f04dc01af477c48cefb

C85V lock commit:
  3c732489407ebca7603e5fb65d03c1ae25d046b6
```

Candidate hashes and dispositions were known before protocol. Candidate bodies
were not opened for review until after the protocol commit.

## Review Architecture

```text
Stage A:
  candidate-blind independent derivation from statements, obligations,
  exact finite laws and primary literature

Stage B:
  candidate comparison and adversarial audit after Stage-A freeze

Stage C:
  deterministic theorem-specific adjudication with no majority vote
```

Each role produces separately hashed artifacts. Stage A has no candidate-path
argument. C85V modules import neither C85T proof generation nor C85T Monte
Carlo dispatchers.

## Theorem Rules

```text
T1/T3/T4/T5/T7:
  PROVED requires a complete general proof;
  finite enumeration supports at most PROVED_FINITE_MODEL_ONLY.

T2/T6:
  COUNTEREXAMPLE requires exact frozen construction replay.

T5:
  remains OPEN if decoder/disjoint-optimum/Fano assumptions are absent;
  C85V cannot repair the frozen statement.
```

Primary sources are registered for Blackwell, Le Cam, Fano and CVaR. Citation
alone never changes status.

## Lock And Validation

```text
runtime-bound repository objects:
  41

bound external objects:
  13

focused / C65 / C23 / full passes:
  395 / 1,040 / 1,451 / 2,375

accepted stderr:
  empty for all four runs

red team:
  56 / 56 PASS
```

The initial C65/C23 attempts exposed one obsolete C85TR2 readiness assertion
that the already consumed C85T authorization record was absent. Accepted
replays used the same post-C85T deselection already documented by C85T. The
failed and cancelled attempts are retained in `C85VP_REGRESSION_VERIFICATION.md`.

## Protected Boundary

```text
registered C85V executions:
  0

formal theorem-status transitions:
  0

C85T Monte Carlo reruns:
  0

real data / active acquisition / C85E / manuscript:
  0 / 0 / 0 / 0
```

Only a new standalone `授权 C85V` may authorize the unique current C85V lock.
No C85VP result authorizes C85E, real data, active acquisition, new data/model
zoos, or manuscript changes.
