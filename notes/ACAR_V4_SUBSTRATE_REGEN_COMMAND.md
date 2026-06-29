# ACAR v4 — Option B substrate-regeneration COMMAND CONTRACT **(FROZEN FOR B1 REVIEW; NO TRAINING)**

```
STATUS : B1-PREFLIGHT. The ONLY admissible regeneration + compatibility commands, their inputs, outputs, runtime lock, and
         guards are frozen here as a reviewable object. Both CLIs are FAIL-CLOSED: they validate fully then RAISE before any
         torch/cmi import, DEV/raw read, or output write. Real training/replay needs the explicit B1 sign-off (last §).
DATE   : 2026-06-29 (machine UTC)
NOT AUTHORIZED (still): EEGNet/GPU training · DEV raw BIDS read · held-out raw read · external preprocessing · compatibility
         replay on real DEV · acar-v4-protocol tag · ACAR_FROZEN_v4 finalization · external Arm-B run.
```

## 1. The two frozen commands
```
# (1) regenerate the NEW all-DEV substrate for ONE disease (B1-gated; raises SubstrateTrainingNotAuthorizedError)
python -m acar.v4.run_regen_substrate --disease PD \
    --dev-input-manifest /abs/acar_v4_regen_pd_inputs.json --output /abs/new_pd_substrate_dir
python -m acar.v4.run_regen_substrate --disease SCZ \
    --dev-input-manifest /abs/acar_v4_regen_scz_inputs.json --output /abs/new_scz_substrate_dir

# (2) fixed-candidate DEV compatibility replay (B1-gated; raises SubstrateCompatibilityNotAuthorizedError)
python -m acar.v4.run_substrate_compatibility \
    --substrate-manifest /abs/substrate_manifest.json --output /abs/new_compat_dir
```
No arbitrary cohort list / seed / pipeline config / candidate is accepted — every degree of freedom is pinned below and
enforced by `acar.v4.regen_substrate` validators.

## 2. run_regen_substrate input manifest schema (validate_regen_manifest, fail-closed)
```
protocol_commit          40-hex; run aborts unless HEAD == this commit
repo_clean_required      MUST be true; git worktree must be clean (porcelain empty, incl. untracked)
disease                  "PD" | "SCZ"
dev_cohorts              EXACTLY DEV_SCOPE[disease]  (PD: ds002778,ds003490,ds004584 ; SCZ: ds003944,ds003947,ds004000,ds004367)
                         — any external/rejected id (zenodo14808296, ds007526, ds007020, 14178398, aszed) is REJECTED
source_kind              "raw_bids" | "canonical_features"
source_paths             {cohort: ABSOLUTE path}, keyed by EXACTLY dev_cohorts
seed                     STRICT int 0 (bool / "0" / 0.0 / 0.9 rejected — no silent coercion)
subject_list_sha256      64-hex   (provenance of the training subjects)
diagnosis_label_sha256   64-hex
pipeline_config_sha256   64-hex AND == canonical FROZEN_PIPELINE hash (regen_substrate.canonical_pipeline_config_sha256())
env_lock_path            non-empty path (must exist at run)
env_lock_sha256          64-hex AND == sha-256 of the env_lock_path file (verified in run_regen_substrate preflight; see §4)
```
(The in-memory `pipeline_config`, when passed to `validate_substrate_request`, must EXACTLY equal `FROZEN_PIPELINE` — extra
keys rejected.)

## 3. Output artifact schema (the authorized run writes; pinned now)
Per disease (`regen_substrate.expected_artifact_paths`): `v4_alldev_encoder_<D>.pt`, `v4_alldev_source_state_<D>.npz`,
`v4_alldev_substrate_<D>.provenance.json`. Plus, written into `--output`:
```
encoder checkpoint        + encoder_checkpoint_sha256 = sha256(canonical little-endian state_dict bytes)
source-state artifact     + source_state_sha256 = acar.v3 SourceStateArtifact full-bytes hash
encoder provenance JSON    (ENCODER_ARTIFACT_FIELDS: arch=EEGNet, training command, data scope, seed, determinism,
                            torch/braindecode versions, embedding_dim==16, source_state_ref)
source-state provenance JSON
manifest LAST (manifest_sha256; RESULT sentinel)   ← "output complete" iff this exists
```

## 4. Runtime lock (regen is torch/braindecode/GPU — heavier than the pure-numpy v4 code)
Schema + validator + canonical hasher: `acar/v4/regen_envlock.py` (PURE; no torch capture). Required fields
(`expected_regen_env_fields`):
```
schema_version (acar_v4_regen_env_lock/1) · status · python_version · torch_version · braindecode_version · numpy_version ·
scipy_version · sklearn_version · cuda_version · cudnn_version · device_kind ("cuda"|"cpu") · device_name · driver_version ·
torch_deterministic_algorithms (true) · seed (int 0) · torch_intraop_threads · torch_interop_threads · omp_num_threads ·
threadpool_backends · pipeline_config_sha256 · protocol_commit
```
- `status` ∈ {`SCHEMA_ONLY_NOT_CAPTURED`, `CAPTURED_AND_VERIFIED`}. `schema_only_template(...)` builds a reviewable
  skeleton (placeholder versions); a CAPTURED lock MUST fill real non-empty version/device fields (and, if `device_kind ==
  cuda`, cuda/cudnn/driver) — a skeleton cannot impersonate a captured runtime.
- **`run_regen_substrate` requires `status == CAPTURED_AND_VERIFIED`**, that the lock file's sha equals the manifest
  `env_lock_sha256`, and that the lock's `protocol_commit` + `pipeline_config_sha256` match the manifest. Capturing the real
  runtime (torch import on the chosen training node) is part of B1 — NOT done here.
- Device (GPU model / CPU) is PINNED in the lock — "node choice" is not a post-hoc degree of freedom (cf. the B0 finding
  that some DEV hashes came from a different CUDA device).

## 5. Atomic, no-overwrite output (matches v3/v4 runner discipline)
`--output` must be absent; the authorized run claims it atomically (os.mkdir) BEFORE any training; writes artifacts; writes
the manifest LAST; on any abort removes the claimed dir. A run is COMPLETE iff its RESULT/manifest sentinel exists. No
partial directory is ever interpreted as a usable artifact.

## 6. run_substrate_compatibility manifest schema (validate_substrate_manifest, fail-closed)
```
protocol_commit   40-hex (HEAD must match; clean worktree)
candidate         EXACTLY {score_family: shift_margin, policy: benefit_ranked, loss: harm_indicator}  (NO reselection)
alpha/budget/coverage_min  EXACTLY 0.10 / 0.10 / 0.15  (pinned operating point)
substrates        {PD:{...}, SCZ:{...}} each: encoder_checkpoint_path + sha256, source_state_path + sha256,
                  encoder_provenance_path, source_state_provenance_path
dev_cohorts       {PD: DEV_SCOPE[PD], SCZ: DEV_SCOPE[SCZ]}  (exact)
env_lock_sha256   64-hex
```
Before the (gated) replay, `run_substrate_compatibility` runs a PURE artifact file-hash preflight: each PD/SCZ
encoder_checkpoint + source_state file must EXIST and its sha-256 must equal the manifest — fail-closed (FileNotFoundError /
ValueError) before any torch import or DEV read. (Pre-B1 the trained artifacts do not exist, so this preflight fails by
design.)
Pass-line = `regen_substrate.compatibility_replay_pass` (pure, pre-registered): per disease — CAL LTT λ* certified ∧
coverage ≥ 0.15 ∧ red > 0 ∧ EVAL L_harm_all ≤ 0.10 ∧ **v2_replay EVALUABLE ∧ red > v2_replay_red (HARD — no waiver)**; macro
— disease-macro red > disease-macro v2_replay. **If v2_replay is not evaluable for either disease, the replay FAILS and
external Arm B is NOT authorized.** This is a substrate-COMPATIBILITY check for the already-fixed candidate, NOT a new DEV
selection run (the in-sample caveat in ACAR_V4_SUBSTRATE_REGEN_PLAN.md §7 applies).

## 7. Guards (acar/v4/tests/test_regen_substrate.py + test_regen_envlock.py — all green; NO training/torch)
wrong disease / wrong-or-external cohort list → fail · seed STRICT int 0 (bool/"0"/0.0/0.9 → fail) · pipeline_config drift
OR extra key → fail · pipeline_config_sha256 ≠ canonical → fail · missing/absent env lock → fail · env_lock_sha256 ≠ file
hash → fail · env lock SCHEMA_ONLY_NOT_CAPTURED → rejected (CAPTURED_AND_VERIFIED required) · env lock missing/extra field,
bad status/device, non-strict seed, deterministic flag false, CAPTURED-with-empty-versions → fail · output exists → fail
before any training call · the command cannot override candidate/score/grid/comparator · compatibility replay FAILS if
v2_replay not evaluable (HARD), if PD passes but SCZ fails, if red ≤ v2_replay, or if L_harm_all > 0.10; passes ONLY when
every pre-declared numeric gate passes · compatibility artifact file-hash preflight FAILS on missing path or sha mismatch ·
both CLIs fail-closed AFTER a full preflight (HEAD==commit, clean worktree, output absent, env-lock/artifact hashes) and
BEFORE any heavy import / DEV read / output write (tests assert `torch`/`cmi` never enter sys.modules).

## 8. B1 SIGN-OFF (the only thing that unlocks real training)
B1 authorizes all-DEV substrate regeneration EXACTLY as specified by: this file + `run_regen_substrate.py` +
`run_substrate_compatibility.py` + the fixed input manifests + the pinned regen env lock. Until B1 is signed:
`_require_b1_authorization` raises and `_train_and_write` is unreachable; no training occurs. Post-B1 sequence: train PD+SCZ
substrates → review artifact hashes → fixed-candidate compatibility replay → if PASS, update `ACAR_FROZEN_v4.md` with the new
substrate hashes → clean run + sign-off + tag `acar-v4-protocol` → THEN held-out preprocessing + the single external Arm-B
run. If B1 is declined or the replay fails → Option C (V4 stays DEV-only exploratory).
