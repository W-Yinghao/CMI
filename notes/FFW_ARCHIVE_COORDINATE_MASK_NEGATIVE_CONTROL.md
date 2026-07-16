# FFW pilot — ARCHIVED as a coordinate-mask negative control (PM directive)

The "Finding Fantastic Weights" (FFW, arXiv:2403.14200) *weight/neuron* line is STOPPED. Only the abstract idea
survives — "in a trained model there may be a better functional substructure; learn what to keep vs delete" —
and it is redirected to a low-rank **subspace in representation space** (see
`notes/SUBSPACE_CANDIDATE_SPAN_AUDIT_SPEC.md`, method line = Subspace Ticket Selection), NOT weights, NOT
native neuron coordinates.

## Why weight/neuron FFW is dropped for this project
- Structured neuron masking assumes the bias is concentrated in a subset of neurons; when the information is
  spread across representation dimensions the neuron mask fails (the FFW paper says as much). Our evidence is
  that subject leakage is low-rank but NOT aligned with native neuron axes.
- Weight/neuron pruning is maximally base-model-dependent (architecture, channel counts, BN/residual, native
  coordinates); a mask does not transfer across backbones. A low-rank subspace restricted to a small
  source-only candidate span (rank r≤8–12) is far more backbone-portable and statistically cheaper
  (Grassmann d.o.f. k(r−k) ≪ k(d−k)).

## Disposition of the existing FFW pilot
- **Keep the code** (`tos_cmi/eeg/ffw.py`, `scripts/run_ffw.py` on branch `agent/cmi-trace-erasure-oracle`);
  do NOT delete.
- **Finish any already-running FFW jobs** for the record; do not launch new ones.
- **Do NOT extend** to the 63-cell full-EEG matrix, unstructured whole-network pruning, magnitude pruning of
  large matrices, random-rotation FFW, or oracle-aligned neuron pruning. The previously-approved "FFW full-EEG
  validation" is REVOKED.
- Archive role: a one-shot **coordinate-mask negative control** — evidence that native-coordinate / neuron
  masking is insufficiently expressive for the leakage geometry, motivating the oblique low-rank subspace
  approach. Any FFW result is reported only in that role, never as a standalone method success.

## Status
FFW weight/neuron: STOP-EXTEND, ARCHIVE. Subspace method line: see Track B/C/D in the candidate-span spec
(Track B = GO characterization; Track C learned selector + Track D TTE = HOLD until the F2 primary gate passes).
Manuscript FROZEN.
