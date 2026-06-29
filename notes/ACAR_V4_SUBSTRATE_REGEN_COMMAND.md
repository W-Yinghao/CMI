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
seed                     0 (only 0 admissible)
subject_list_sha256      64-hex   (provenance of the training subjects)
diagnosis_label_sha256   64-hex
pipeline_config_sha256   64-hex   (== sha of the canonical FROZEN_PIPELINE)
env_lock_path            non-empty path (must exist at run)
env_lock_sha256          64-hex   (the regen runtime lock; see §4)
```

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
A SEPARATE regen env lock (pinned in the input manifest as env_lock_sha256) must record:
```
torch version · braindecode version · numpy/scipy versions · CUDA/cuDNN + device (GPU model + driver, OR explicit CPU) ·
torch deterministic flags · seed 0 · intra/inter-op + OMP thread settings.
```
Device (GPU model / CPU) is PINNED in the lock — "node choice" is not a post-hoc degree of freedom (cf. the B0 finding that
some DEV hashes came from a different CUDA device).

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
Pass-line = `regen_substrate.compatibility_replay_pass` (pure, pre-registered): per disease — CAL LTT λ* certified ∧
coverage ≥ 0.15 ∧ red > 0 ∧ EVAL L_harm_all ≤ 0.10 ∧ **v2_replay EVALUABLE ∧ red > v2_replay_red (HARD — no waiver)**; macro
— disease-macro red > disease-macro v2_replay. **If v2_replay is not evaluable for either disease, the replay FAILS and
external Arm B is NOT authorized.** This is a substrate-COMPATIBILITY check for the already-fixed candidate, NOT a new DEV
selection run (the in-sample caveat in ACAR_V4_SUBSTRATE_REGEN_PLAN.md §7 applies).

## 7. Guards (acar/v4/tests/test_regen_substrate.py — all green; NO training/torch)
wrong disease / wrong-or-external cohort list → fail · seed≠0 → fail · pipeline_config drift → fail · missing/absent env
lock → fail · output exists → fail before any training call · dry-run never imports torch/cmi · dry-run returns the exact
expected artifact paths · the command cannot override candidate/score/grid/comparator · compatibility replay FAILS if
v2_replay not evaluable, if PD passes but SCZ fails, if red ≤ v2_replay, or if L_harm_all > 0.10; passes ONLY when every
pre-declared numeric gate passes · both CLIs fail-closed AFTER a full preflight (HEAD==commit, clean worktree, output absent)
and BEFORE any heavy import / DEV read / output write.

## 8. B1 SIGN-OFF (the only thing that unlocks real training)
B1 authorizes all-DEV substrate regeneration EXACTLY as specified by: this file + `run_regen_substrate.py` +
`run_substrate_compatibility.py` + the fixed input manifests + the pinned regen env lock. Until B1 is signed:
`_require_b1_authorization` raises and `_train_and_write` is unreachable; no training occurs. Post-B1 sequence: train PD+SCZ
substrates → review artifact hashes → fixed-candidate compatibility replay → if PASS, update `ACAR_FROZEN_v4.md` with the new
substrate hashes → clean run + sign-off + tag `acar-v4-protocol` → THEN held-out preprocessing + the single external Arm-B
run. If B1 is declined or the replay fails → Option C (V4 stays DEV-only exploratory).
