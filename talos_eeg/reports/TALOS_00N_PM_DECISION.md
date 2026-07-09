TALOS_00N - PM Decision Record

Decision: TALOS_00B is accepted as a valid real EEG negative / diagnostic-only
readout. The TALOS low-degree-of-freedom frozen-feature route is closed
negative. P1, P2, source-free deployment claims, generalization claims, and
safety claims are denied.

Formal gate state

```text
TALOS_00_DESIGN_PACKAGE: PASS
TALOS_00A_PREFLIGHT: PASS
TALOS_00B_REAL_FROZEN_FEATURE_REPLAY: COMPLETE_NEGATIVE
TALOS_LOW_DOF_ADAPTER_TO_P1: DENIED
TALOS_P1: BLOCKED / DENIED_FROM_00B
TALOS_P2: BLOCKED
SOURCE_FREE_DEPLOYMENT_CLAIM: FORBIDDEN
GENERALIZATION_OR_SAFETY_CLAIM: FORBIDDEN
CMI / PRUNING / SURGERY RESCUE: FORBIDDEN
```

Accepted evidence

```text
feature_artifacts_loaded: 18/18
per_artifact_hash_check: PASS
red-team failures: 0
red-team warnings: 0
target-label quarantine: PASS
adapter determinism: PASS
variant freeze: PASS
scientific gate: FAIL
p1_training: false
source_free_deployment_claim: false
```

Decision rationale

P1 requires a stable, clean, non-leaking, non-boundary real EEG signal from the
approved low-degree-of-freedom adapter universe. TALOS_00B did not provide that
evidence:

```text
EEGConformerMini: effect too small
EEGNetMini: gains boundary-contaminated
```

Therefore TALOS cannot proceed to:

```text
P1 source-free serialized-state training
P2 streaming / clinical transfer
source-free deployment claim
generalization claim
safety claim
new method claim from TALOS_00B
```

Forbidden next work under TALOS_00B

```text
No P1 request.
No P2 request.
No TALOS-LR rescue.
No TALOS-full rescue.
No low-rank affine rescue.
No trust-region relaxation on this run.
No target-informed hyperparameter rescue.
No boundary-hit reinterpretation as acceptable.
No EEGNet-only selection while ignoring EEGConformerMini.
No CMI, pruning, mask, surgery, CutClean, safety-gate, or harm-router reentry.
No source-free deployment claim.
No generalized TTA improvement claim.
```

Relationship to CEDAR and CutClean

This decision does not reopen CEDAR or CutClean-style pruning. TALOS failed on
a different chain:

```text
target-unlabeled frozen features
-> low-degree-of-freedom adapter
-> clean non-boundary target gain
```

CutClean-style auxiliary privacy heads, MI-aware training, structured pruning,
fine-tuning, sparsity grids, and privacy-head thresholds are not authorized as
a TALOS rescue path.

TALOS archive state

```text
TALOS low-degree-of-freedom frozen-feature route: CLOSED_NEGATIVE
TALOS as P1 training method: NOT APPROVED
TALOS as source-free deployable method: NOT APPROVED
TALOS as red-team / adapter-replay infrastructure: RETAIN
TALOS as mechanism explanation for TTA-Control gain: NOT SUPPORTED
```

Allowed next work

Only closeout and project-state documentation are authorized under TALOS_00N.
If work continues, it must be a new mechanism-audit project with a new charter,
for example TTA-MECH-EEG, a target-unlabeled EEG adaptation mechanism audit. It
must not be called TALOS_01, CEDAR_02, or a method rescue.

Precise final wording

```text
Current low-degree-of-freedom TALOS adapters cannot cleanly explain or
reproduce the known TTA-Control positive signal on BNCI2014_001 frozen features.
```

Do not write:

```text
target-unlabeled adaptation is impossible
TTA-Control is invalid
source-free EEG adaptation is proven impossible
```
