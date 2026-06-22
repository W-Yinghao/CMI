# ACAR v3 — Amendment 6 (artifact/data-object/guard fail-closed completeness; DESIGN-ONLY, NON-BINDING)

**Date:** 2026-06-22 · **Status:** `NON-BINDING / NO DEV NUMERICAL RUN / NO LOCKBOX ENDPOINT ACCESSED / NOT TAGGED`
Closes the 6 reproducible blockers from the `2526827`/`7188a2a` review. Skeleton header + S13 provenance corrected.
v2 endpoint `1528a94` / tag `acar-v2-protocol` @ `9b2f0c1` unchanged. All synthetic-only.

## Resolutions (1:1 with the review)

1. **Artifact hash covers a UNIQUE canonical representation.** `sigma_min` and `env` are canonicalized to sorted
   tuples with **no duplicate keys** (validated; previously `dict()`-collapsed → collision). Hash uses
   length-prefixed encoding (no `dict()`). `arch_schema` must `== SCHEMA_VERSION`. `state_items` must match the
   candidate's **frozen architecture exactly** (name/dtype=`<f4`/shape); integer/foreign dtypes rejected.
   `artifact_sha256` is `init=False`. `verify_integrity()` now returns **None**; `predict()` uses a private
   `_fresh_net()` that builds an ephemeral net (no module is returned from a public method).
2. **Strong immutability via bytes-backed arrays.** New `acar/v3/_util.frozen_array` rebuilds every stored array on an
   immutable `bytes` buffer, so `flags.writeable = True` **raises**. Applied to `WindowActionSet`, `DeploymentBatch.z`,
   and the normalizers. Sequence fields are canonicalized to tuples (`FallbackBatchRecord.window_keys`,
   `LabeledRiskRecord.delta_r_by_action`, `DeploymentBatch.window_keys`).
3. **Contract gaps closed.** `CandidatePrediction.disease ∈ {PD,SCZ}`; C2 `scale_floor > 0`; `DeploymentBatch`
   embedding dim `d ≥ 1`; the 3 actions of a `(SubjectKey, deployment_batch_digest)` must carry **identical
   `window_keys`** (same natural batch), enforced in `_validate` and by the new immutable `DeploymentFeatureRecord`
   that derives the 3 consistent `TrainExample`s.
4. **Epoch semantics + target weighting.** Artifact records `best_epoch_zero_based`, `checkpoint_epoch_count`
   (= best+1), and `n_epochs_executed` (optimizer epochs actually run; `≥ checkpoint`), all strict non-bool ints.
   `final_epochs` and `refit_candidate_fixed_epochs` reject bool/float/negative epochs (no silent `int()` truncation).
   **Target normalization is now subject-balanced** (subject-equal first/second moments), matching the loss.
5. **Two weak guards fixed.** Mixed-disease rejection now uses a **valid** candidate (`C1`) so it actually exercises
   the disease check. CAL/EVAL isolation no longer self-compares: it asserts CAL labels move `q`, that `route` takes
   **no** label argument, and that `U` shifts by exactly the `q⁺·scale` term (route/U are a function of `(preds,q)`
   only; EVAL labels are not in the path).
6. **Skeleton provenance.** Header now reads "incorporates Amendments 1–6" with the exact commit chain; S13 records
   Amendment 5/6 with real commits and drops the "41→all" count in favor of "all v3 suites pass".

## Guards
All v3 suites green with new guards: dup-`sigma_min`/`env` hash rejection, `<f4`-exact state bytes, `verify_integrity`
returns None + tamper→failure, bytes-backed arrays reject `writeable=True`, `CandidatePrediction.disease`, C2
`scale_floor>0`, `DeploymentBatch d≥1`, `DeploymentFeatureRecord` window-key consistency, epoch-field strictness +
`final_epochs` strict, subject-balanced target, real mixed-disease + real CAL/EVAL isolation. v2 guards still pass.

## Remaining before `acar-v3-dev-design-v1` (no DEV/lockbox)
Structural real loader on SYNTHETIC `.npz` fixtures: deployment loader reads ONLY
`z_te/subject_id_te/recording_id_te/window_index_te` with **strict integer normalization** (accept `np.integer`,
reject bool/float/str — not arbitrary `int()`); labels aligned by `WindowKey` (not row order); an immutable
`ActionOutputsRecord` hashing batch-digest + `p0` + all `pa` + action vocab + probability schema; `LabeledRiskRecord`
binds `deployment_batch_digest` **and** `action_outputs_sha256` (ΔR ↔ exact frozen action outputs); a v3 canonical
source-state hash (classifier classes/weights/moments/schema/versions/source-dump hash); all SHA-256 lowercase 64-hex;
a **label-poisoned proxy fixture** proving the deployment path never reads `y_te`; PD-artifact-on-SCZ-`DeploymentBatch`
rejected **before** any prediction. → then `develop.py` (S5 orchestration) → env lock → full clean guard run → tag →
first DEV gate (`DEV_STOP / NO_LOCKBOX_CONSUMED` if no candidate passes).
