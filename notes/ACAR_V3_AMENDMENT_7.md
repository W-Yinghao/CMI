# ACAR v3 — Amendment 7 (bounded pre-loader correction; DEV-engineering, NON-BINDING)

**Date:** 2026-06-22 · **Status:** `NON-BINDING / NO DEV NUMERICAL RUN / NO LOCKBOX ENDPOINT ACCESSED / NOT TAGGED`
Last correction window before the structural real loader (the tag `acar-v3-dev-design-v1` is still open, so the window
is still open). `4ef0045`'s substance stands; this closes five reproducible issues that touch artifact identity, hash
uniqueness, or fail-closed semantics — paths the green synthetic suites did not exercise. v2 endpoint `1528a94` / tag
`acar-v2-protocol` @ `9b2f0c1` untouched. Synthetic-only. **Per directive: no 7th open-ended design review — proceed
to the loader after this.**

## Resolutions (1:1 with the review)

1. **A C2 net could be mislabeled into a C3 artifact.** `make_artifact()` now requires `net.candidate == candidate`
   **and** `set(net.heads) == NON_IDENTITY` (C2/C3 share param names/shapes, so the type check is the only guard).
   Epoch fields are validated as **strict non-bool int** in the factory (no `int(0.9)→0` truncation; `True` rejected).
   Duplicate `sigma_min`/`env` keys are no longer silenced: dict input has unique keys; a sequence-of-pairs is passed
   **through unchanged** so `_canon_pairs` raises on duplicates.

2. **Artifact hash was not an injective encoding.** `_hash()` used NUL separators, so `("a","b\0c")` and `("a\0b","c")`
   collided. It now uses true **length-prefixed** encoding (`>Q` byte-length + bytes) for every `env`/`sigma_min`/
   `state_items`/meta field. `env` entries are validated `(str,str)`, `sigma_min` floors validated Python `float`. The
   artifact stores its **own immutable `hp_snapshot`** at construction and hashes that — mutating the global `HP` after
   construction can no longer change an existing artifact's hash or fail its `verify_integrity()`.

3. **`DeploymentFeatureRecord` was not actually immutable.** `__post_init__` now materializes `per_action` to a tuple
   and validates disease ∈ {PD,SCZ}, `SubjectKey`, lowercase-64-hex digest, the three actions in canonical order, each
   `window_action_set.action_name == action`, every `WindowKey` under the subject, identical `window_keys` across the
   three actions, and finite ΔR — all before it can be used by the loader/`develop.py`.

4. **Three fail-closed bypasses removed.** (a) `subject_joint_score(iter(()))` returned `−∞` (an empty generator is
   truthy) → now materialized to a tuple and rejected. (b) `extract_action_set()` ran the identity + action adapters
   before the size check → now rejects `< MIN_BATCH` (and `> B`) **before any adapter**, and validates identity `z0`
   shape/finiteness. (c) `build_action_sets()` now validates identity `z0` shape/finiteness before fanning out.

5. **Subject-balanced training given an exact, registered definition.** New `_epoch_optimize` does **one optimizer step
   per epoch via deterministic gradient accumulation**: each subject contributes `mean(its example losses)/n_subjects`,
   so the epoch gradient is the exact mean over **all** subjects regardless of minibatch boundaries (the old last-
   partial-minibatch over-weighting is gone). This is now the registered semantics: *per-subject-equal, epoch-level
   accumulation* — not "subject-level minibatch optimization."

Shared digest validator `_is_hex64()` now accepts **lowercase 64-hex only** (matches the loader spec; uppercase
SHA-256 rejected everywhere: `DeploymentBatch` source, `LabeledRiskRecord`, `TrainExample`, `DeploymentFeatureRecord`).

## Guards added (all green)
C2-net-as-C3 → raise; float/bool epoch at `make_artifact` → raise; duplicate `sigma_min`/`env` via factory → raise;
NUL-collision env tuples → **different** hashes; global `HP` mutation → existing artifact `verify_integrity` still
passes (own snapshot); empty subject generator → raise; `<MIN_BATCH` `extract_action_set` → **zero** adapter calls;
malformed identity `z0` → raise before non-identity; `_epoch_optimize` gradient **invariant to minibatch size**
(partial-batch safe); uppercase SHA-256 → raise (`DeploymentBatch` source + `LabeledRiskRecord`). All four v3 suites
+ the v2 guard suite pass.

## Next (immediate, no DEV/lockbox): structural real loader on SYNTHETIC `.npz` fixtures
Deployment loader reads ONLY `z_te/subject_id_te/recording_id_te/window_index_te` with strict `np.integer` window-index
(reject bool/float/str — not arbitrary `int()`); labels aligned by `WindowKey` (not row order) into `LabeledRiskRecord`;
an immutable `ActionOutputsRecord` hashing batch-digest + `p0` + all `pa` + action vocab + probability schema;
`LabeledRiskRecord` binds `deployment_batch_digest` **and** `action_outputs_sha256`; a v3 canonical source-state hash
(classifier classes/weights/moments/schema/versions/source-dump hash); lowercase-64-hex everywhere; a **label-poisoned
proxy fixture** proving the deployment path never reads `y_te`; PD-artifact-on-SCZ-`DeploymentBatch` rejected **before**
any prediction. → loader guards → `develop.py` (S5) → env lock → green re-run → `acar-v3-dev-design-v1` → first DEV gate.
Full documentation consolidation (single `ACAR_V3_DEV_DESIGN_SPEC.md`, Amendments 1–7 retained as changelog) is
deferred until the loader and `develop.py` are done — one consolidation, not a seventh design review.
