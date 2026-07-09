TTA_MECH_00 - Project Charter

Status: docs-only design charter. TTA_MECH_00 does not run real EEG
experiments, introduce a new method, request P1/P2, or make a deployment claim.

Project name:

```text
TTA-MECH-EEG
Mechanistic Audit of Target-Unlabeled EEG Adaptation
```

One-line positioning:

```text
Benchmark and explain existing target-unlabeled EEG adaptation signals; do not
propose a new algorithm.
```

Background

The project portfolio now contains two recent clean negative boundaries:

```text
CEDAR_01N:
  source-only frozen-latent surgery closed negative on real EEG
  0 / 54 ACCEPT candidates
  no mask / pruning / surgery continuation

TALOS_00B:
  low-degree-of-freedom adapter route closed negative
  EEGConformerMini clean effect too small
  EEGNetMini gains boundary-hit dependent
  no TALOS-LR / TALOS-full / trust-region rescue
```

Older CMI/CIGL/CITA work supplies another boundary:

```text
CMI audit can be useful diagnostically
CMI control did not produce stable functional reliance reduction
CITA target gain was attributable to non-CMI TTA-Control behavior, not the CMI term
```

Therefore TTA-MECH is not a method rescue project. It is a mechanism audit /
benchmark project for existing target-unlabeled adaptation behavior.

Core question

```text
What mechanism explains the robust non-CMI target-unlabeled adaptation gain
observed in earlier CITA / TTA-Control experiments?
```

Scientific goals

```text
1. Reproduce or audit existing TTA baselines when artifacts and code support it.
2. Decompose entropy / balance adaptation, matched-CORAL / SPDIM geometry,
   T3A classifier-template adjustment, source replay, and normalization / BN
   contributions.
3. Separate target balanced-accuracy gain from calibration gain.
4. Output a mechanism taxonomy, not a new algorithm.
```

Baseline families in scope

```text
ERM-no-adapt
TTA-Control
matched-CORAL
SPDIM
T3A
```

Mechanism hypotheses to audit

```text
entropy / confidence sharpening
class-balance or target prior shift
feature recentering and covariance alignment
classifier-template adjustment
source replay regularization
normalization / BatchNorm dynamics
calibration-only improvement without bAcc gain
bAcc-only improvement with NLL / ECE degradation
```

Non-goals

```text
No SOTA claim.
No new adapter.
No new objective.
No CMI rescue.
No CEDAR mask / pruning / surgery.
No TALOS-LR / TALOS-full / low-rank rescue.
No CutClean-style privacy-head pruning.
No safety gate or harm router.
No target-label-based method selection.
No P1 / P2 training claim.
No source-free deployment claim.
```

Relationship to CutClean

CutClean is a privacy-preserving pruning pipeline using block-level auxiliary
privacy heads, MI-aware training, structured pruning, fine-tuning, sparsity
grids, and privacy-head thresholding. TTA-MECH does not import that framing and
does not authorize privacy-head pruning, sparsity search, or target-validation
selection.

Exit condition for TTA_MECH_00

TTA_MECH_00 succeeds only as a design package if it freezes a benchmark
protocol, red-team contract, and PM boundary for future replay audits. It does
not approve TTA_MECH_01 execution.
