# ACAR v3 — Amendment 8 (loader-binding correction; DEV-engineering, NON-BINDING)

**Date:** 2026-06-23 · **Status:** `NON-BINDING / NO DEV NUMERICAL RUN / NO LOCKBOX ENDPOINT ACCESSED / NOT TAGGED`
Closes the loader-binding (execution-record) cluster found in `ca796ab`. The structural direction stands; this fixes
five issues where `y_te`, an unbound `state`, a row-order mismatch, or a second adapter pass could corrupt a
deployment/execution record. v2 endpoint `1528a94` / tag `acar-v2-protocol` @ `9b2f0c1` untouched. SYNTHETIC ONLY.
Per directive: proceed directly to synthetic-only `develop.py` after these guards — no further open-ended review.

## Resolutions (1:1 with the review)

1. **`y_te` no longer reaches deployment identity through a whole-file hash.** Provenance is now FIELD-SEPARATED:
   `full_dump_sha256` (whole file — AUDIT ONLY), `source_fit_sha256` (z_ev,y_ev), `deployment_input_sha256`
   (z_te + canonical WindowKeys), `label_sha256` (WindowKey-aligned y_te), `subject_list_sha256`. `v3_source_state_ref`
   now folds `source_fit_sha256` (NOT the whole file), so `source_state_ref` → `DeploymentBatch.source_state_ref` →
   `deployment_batch_digest` are all independent of `y_te`. Guard: a full pipeline run on two dumps differing ONLY in
   `y_te` yields bit-identical `deployment_input`/`subject_list`/batch digests/execution hashes/predictions; only
   `full_dump`, `label`, and ΔR move.

2. **The executed source state is bound to the batch's declared ref.** New immutable `SourceStateArtifact` (disease,
   embedding_dim, source_fit_sha256, source_state_sha256, source_state_ref, env, opaque state). `assert_compatible()`
   checks `source_state_ref == batch.source_state_ref`, disease, and embedding_dim, and runs `verify_integrity()` —
   ALL before any adapter. Two entry points: DEV `fit_source_state_artifact*` (the only callers of `fit_source_state`)
   and external `load_frozen_source_state_artifact` (rebuilds f_0 from frozen params, never fits). Guard: a batch
   declaring state B against artifact A fails; disease/dim mismatch fails; monkeypatching `fit_source_state` to raise
   still lets the external path load (proves no fit) while the DEV path raises.

3. **Order-insensitive batch digest can no longer be paired with order-sensitive outputs.** `DeploymentBatch` now
   canonicalizes to ONE row order at construction (sort by WindowKey; z follows). `canonical_row_digest()` is an
   order-SENSITIVE digest over that stored order, and the execution record binds it. `labeled_risk_record` aligns
   labels by WindowKey in that canonical order and binds `action_outputs_sha256`.

4. **Features and ΔR now come from ONE execution.** `SourceStateArtifact.execute(batch)` runs identity + the 3 actions
   EXACTLY once into a `BatchActionExecutionRecord` (captures z0/p0/za/pa). From it: `window_action_sets()` builds the
   3 WindowActionSets via `set_features._build_was` on the CAPTURED outputs (no re-execution); `labeled_risk_record()`
   computes ΔR from the same captured p0/pa; `deployment_feature_record()` returns a `DeploymentFeatureRecord` carrying
   `execution_sha256` + `action_outputs_sha256` (both now fields on the record). Guard: an end-to-end count shows
   exactly 4 adapter calls (identity + 3) — the feature and ΔR paths do NOT re-execute; cross-pairing a record with the
   wrong batch (digest/row mismatch) or wrong source artifact fails.

5. **S9 provenance via `LoadedDumpManifest`** (dataset_id, disease, the 5 field-separated hashes, n_subjects /
   n_recordings / n_windows / embedding_dim, schema). `subject_list_sha256` is permutation-insensitive (incl.
   duplicates) but changes when a dataset-aware SubjectKey is added/removed. All `np.load` calls use context managers
   (handles closed) with `allow_pickle=False`.

## Guards (7 tests covering the reviewer's 10 requirements; all green)
field-separated 5-hash manifest; strict dtypes no-coercion; **label-firewall full-pipeline poison proxy** (incl.
`full_dump` change does not propagate); state-ref/disease/embedding_dim binding before any forward pass; external load
never fits; single-execution 4-call count + features/ΔR shared hashes + wrong-batch/cross-source rejection + captured-
probability immutability + canonical row identity; subject-list perm-insensitive/add-sensitive; PD-on-SCZ rejected
before prediction + fallback→None. All 5 v3 suites + the v2 guard suite pass.

## Next (authorized, synthetic-only): `develop.py`
S5 split orchestration on SYNTHETIC fixtures: canonical-`SubjectKey` SHA-256 splits (outer EVAL; non-EVAL→FIT/CAL;
FIT→TRAIN/VAL) with stage-specific seeds/ratios, permutation-independent; fallback-only subjects retained in
deployment/EVAL but never faked into a predictor or CAL score; exactly one joint CAL score per eligible CAL subject;
C0/v2 replay over identical subjects/batches/Arm-B CAL-EVAL; final refit only from OOF selection/epochs/C2 floors.
No real DEV value read. Then: env lock → single `ACAR_V3_DEV_DESIGN_SPEC.md` consolidation → clean-worktree verify →
`acar-v3-dev-design-v1` tag → first real DEV read.
