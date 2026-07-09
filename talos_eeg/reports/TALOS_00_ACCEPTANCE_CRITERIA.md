TALOS_00 - Acceptance Criteria

Status: design-only criteria for the first TALOS experiment. These criteria do
not authorize TALOS_00 execution.

Gate purpose

TALOS_00 decides whether a low-dimensional non-CMI target-unlabeled adapter is
worth promoting to a source-free serialized-state training phase. It is not a
paper claim and not a deployment claim.

Mandatory preconditions

```text
CEDAR_01F handoff manifest validated
18/18 approved BNCI2014_001 feature artifacts available
variant universe frozen
hyperparameters frozen
trust-region bounds frozen
target labels quarantined until final metrics
red-team checks implemented before reporting PASS
```

Primary metrics

```text
target balanced accuracy
target NLL
target ECE
worst-fold target balanced accuracy
target prediction entropy
mean prediction label distribution
adapter norm
source-audit diagnostic metrics
```

Target labels are final metric inputs only. They must not affect adapter
training, variant selection, hyperparameter choice, collapse rescue, or
acceptance rescue.

PASS

TALOS_00 passes only if all conditions hold:

```text
1. Target-label noninterference red-team passes.
   Removing, shuffling, or replacing target labels leaves adapter parameters
   and target-unlabeled output hashes unchanged.

2. At least one TALOS variant improves over ERM on both backbones.
   target bAcc TALOS - ERM >= +0.02 for EEGNetMini and EEGConformerMini.

3. TALOS is noninferior to TTA-Control.
   TALOS >= TTA-Control - 0.005 in bAcc, or bAcc is effectively tied while
   NLL/ECE is materially better.

4. No collapse.
   Target entropy is non-degenerate, mean prediction is not single-class, and
   adapter norm stays inside the predeclared trust region.

5. Low-dimensional contribution is plausible.
   TALOS-L or TALOS-D is close to TALOS-LD, showing that a small operator
   explains the gain rather than a high-capacity transform.

6. Source-free discipline is not violated.
   If the run is declared source-free, the adapter reads serialized source
   statistics and not source examples.
```

PASS outcome:

```text
TALOS_00_PASS
PM may consider TALOS_01_SOURCE_FREE_SERIALIZED_STATE_TARGET_ADAPTATION
No SOTA, safety, privacy, or generalization claim is created by P0.
```

CONDITIONAL PASS

Allowed only if:

```text
one backbone satisfies PASS conditions
the other backbone is tied, low-signal, or inconclusive
there is no collapse
there is no target-label leakage
there is no source-free violation
TALOS is not clearly worse than TTA-Control / CORAL / SPDIM
```

Conditional outcome:

```text
TALOS_01_SINGLE_BACKBONE_MECHANISM may be requested
No full P1 matrix is approved
No method claim is approved
```

FAIL

TALOS_00 fails immediately if any hold:

```text
TALOS <= ERM on both backbones
TALOS is clearly worse than TTA-Control, CORAL, or SPDIM
target entropy collapse
mean prediction becomes effectively single-class
adapter norm hits or exceeds trust-region boundary
result only comes from an unapproved high-capacity transform
target labels influence adapter fitting, variant selection, or acceptance
hyperparameters are amended after target metrics are observed
source-free mode reads source examples
```

FAIL outcome:

```text
TALOS closes as diagnostic replay
No TALOS_01
No P2
No source-free deployment claim
No target-generalization claim
```

Red-team requirements

A TALOS_00 report must include:

```text
target_label_quarantine.json
adapter_determinism.json
source_free_guard.json
variant_universe_freeze.json
collapse_guards.json
adapter_norm_bounds.json
```

The report must distinguish:

```text
target-unlabeled adapter outputs
final target metrics
source-audit diagnostics
historical report-only references
```

Forbidden interpretations

```text
Do not describe TALOS as CMI control.
Do not describe TALOS as pruning or surgery.
Do not describe TALOS as a safety gate.
Do not use target metrics to rescue a variant.
Do not treat CEDAR_01N as "almost worked".
Do not call a target metric gain a certified generalization guarantee.
```

P1 entry requirements

Before TALOS_01 can be requested, TALOS_00 must produce:

```text
PASS or explicit CONDITIONAL PASS
complete frozen-feature run manifest
target-label noninterference pass
adapter determinism pass
source-free guard pass or explicit non-source-free limitation
collapse guard pass
adapter norm bound pass
variant universe freeze proof
```

Without those artifacts, TALOS remains design-only or diagnostic-only.
