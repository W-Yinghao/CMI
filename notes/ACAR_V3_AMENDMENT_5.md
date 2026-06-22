# ACAR v3 — Amendment 5 (tag-prep correction; DESIGN-ONLY, NON-BINDING)

**Date:** 2026-06-22 · **Status:** `NON-BINDING / NO DEV NUMERICAL RUN / NO LOCKBOX ENDPOINT ACCESSED / NOT TAGGED`
Closes the 7 reproducible `DEV_DESIGN_LOCK` blockers found in the `2e32ff7`/`03fbf8b` review. Skeleton S5/S13 synced.
v2 endpoint `1528a94` / tag `acar-v2-protocol` @ `9b2f0c1` unchanged. All synthetic-only.

## Resolutions (1:1 with the review)

1. **Artifact cache bypass removed.** `_NET_CACHE` (a shared *mutable* `nn.Module`) is deleted. `verify_integrity()`
   recomputes the hash from the immutable bytes and **rebuilds a fresh net every call**; `predict()` uses that fresh
   net. No mutable module is stored or returned, so a mutated live module can no longer pass integrity.
2. **Training-side disease binding.** `TrainExample` now carries `disease`; `_validate(examples, disease)` raises if
   any example's disease ≠ the requested disease (rejecting a single set used as both PD and SCZ). Each example also
   validates that every `WindowActionSet.window_keys` entry is a `WindowKey` under the example's `SubjectKey`.
3. **target SD floor consistency.** `HP["target_sd_floor"]` is now **1e-3** (matching `TargetNormalizer` and the
   skeleton); a separate `HP["input_sd_floor"]=1e-6`. Since `HP` is hashed into the artifact, metadata is now correct.
4. **Canonical little-endian parameter bytes.** `state_items()` stores **explicit `<`-endian** dtype + bytes
   (precision preserved); `_validate_items` checks sorted-unique names, dtype `<…`, byte-length vs shape, and
   finiteness. Hash is platform-canonical.
5. **Structured identity + frozen batching.** `SubjectKey`/`RecordingKey`/`WindowKey` are **validated frozen
   dataclasses** (non-empty ids; `window_index` non-negative int, no bool/coercion). `DeploymentBatch` requires
   `SubjectKey`/`RecordingKey`/`WindowKey` instances (plain tuples rejected). `build_deployment_batches` dropped the
   `batch_size` arg (chunks at the frozen **B=32**), performs **no `str()/int()` coercion** (so `1` ≠ `"1"`), rejects
   empty input, and requires one consistent embedding dimension across all rows.
6. **Provenance epoch semantics.** Artifact field renamed `training_epochs → n_epochs_trained` = epochs actually
   trained (`best_epoch_zero_based + 1` for earlystop; `n_epochs` for refit). `fit_candidate_earlystop` returns
   `(artifact, best_epoch_zero_based)` for fold diagnostics. `TrainExample.deployment_batch_digest` validated as
   full **hex** SHA-256.
7. **Fail-closed boundaries completed.** `FallbackBatchRecord` validates `forced_identity is True`, full-hex digest,
   `n_windows == len(window_keys) < MIN_BATCH`. `WindowActionSet` requires `MIN_BATCH ≤ n ≤ B` (a `<MIN_BATCH` set is
   a `FallbackBatchRecord`, not a WAS). C2 `CandidatePrediction.scale_floor ≥ 0`. `conformal_rank` rejects float/bool
   `m` (non-negative integer only).

## Guards
All v3 suites green, with new guards for: cache-tamper → integrity failure, training disease binding + mixed-disease
+ TRAIN/VAL overlap, canonical LE bytes, 1e-13 floor → different hash, n_epochs_trained semantics, WAS n-bounds,
WindowKey/FallbackBatchRecord validation, frozen-B batching (no `batch_size`), id no-coercion, embedding-dim
consistency, plain-tuple key rejection, `scale_floor ≥ 0`, `conformal_rank` int-m. v2 guards still pass.

## Remaining before `acar-v3-dev-design-v1` (no DEV/lockbox)
**Structural real loader** (synthetic `.npz` fixtures only): deployment loader reads ONLY
`z_te/subject_id_te/recording_id_te/window_index_te`; a separate label loader reads `y_te` → `LabeledRiskRecord`
only after label-free outputs + deployment digests are fixed; a **label-poisoned proxy fixture** proves the
deployment loader never touches `y_te`; verify dump/source-state/subject-list SHA-256; enforce
`DeploymentBatch.disease == artifact.disease` (end-to-end PD-artifact-on-SCZ-batch → raise); frozen `B=32`,
`MIN_BATCH=8`; row/id/window-index/dim/finiteness/uniqueness checks → **then** `develop.py` (S5 split orchestration +
S2/S4 + C0/v2 replay + final refit) → env lock → full clean guard run → tag → first DEV gate.
