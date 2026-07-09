# C51R - Conditioned Coverage Code Refactor

## Scope

This refactor reorganizes the stacked C49-C51 conditioned coverage audit code without changing scientific artifacts or audit semantics.

No training, GPU replay, re-inference, selector artifact, checkpoint-choice output, or new diagnostic claim was added.

## Refactor Units

1. Shared audit infrastructure:
   - added `audit_utils.py`
   - added `locked_witness.py`
   - moved CSV IO, frozen config checks, numeric helpers, forbidden-claim guard, stable query/group keys, and the C49 broad-witness registry behind shared utilities

2. Shared island morphology metrics:
   - added `island_metrics.py`
   - moved local-Bayes hit and group fragmentation out of C50 private code
   - kept C50 wrappers for compatibility while letting C51 call the shared implementation directly

3. Shared score diagnostics:
   - added `score_diagnostics.py`
   - moved rank, Spearman, AUC, AUPRC, decile diagnostic controls, and trajectory top-hit scoring out of C51 main flow
   - added direct utility regression tests for the newly public helpers

## Red-Team Checks

- Refactor 1: PASS - C51 no longer depends on C50's private locked-witness builder; C49-C51 focused tests passed.
- Refactor 2: PASS - group fragmentation has one shared implementation; C50 wrapper remains compatibility-only; C50-C51 focused tests passed.
- Refactor 3: PASS - score-shape diagnostics have one shared implementation; focused tests caught and fixed a public-helper list/array compatibility bug; C49-C51 focused tests passed.
- Artifact equivalence: PASS - C49, C50, and C51 artifacts were regenerated after refactor and produced no report/table diff.
- Guardrails: PASS - no artifact payload changed, no selector/checkpoint artifact emitted, no forbidden C49-C51 report/table claim introduced.

## Validation

- C49-C51 focused suite: 35 green after utility tests were added.
- Full C23-C51 regression plus refactor utility tests: 271 green.
