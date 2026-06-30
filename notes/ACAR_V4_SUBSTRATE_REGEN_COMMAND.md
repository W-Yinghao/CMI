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

# (2) fixed-candidate DEV compatibility replay (C1; executable but AUTHORIZATION-gated; omit --compat-authorization => fails closed)
python -m acar.v4.run_substrate_compatibility \
    --substrate-manifest /abs/substrate_manifest.json --output /abs/new_compat_dir \
    [--compat-authorization /abs/compat_auth.json]
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
source_kind              MUST be "raw_bids" (B1b trains raw→encoder; canonical_features is rejected by the validator —
                         SOURCE_KINDS=("raw_bids",), so a canonical_features bypass of raw→encoder training is impossible)
source_paths             {cohort: ABSOLUTE path}, keyed by EXACTLY dev_cohorts
source_file_manifest_sha256              64-hex   (overall raw file-list provenance; metadata only, no signal read)
per_cohort_source_file_manifest_sha256   {cohort: 64-hex}, keyed by EXACTLY dev_cohorts (per-cohort raw-file-set provenance)
eligible_subject_list_sha256             64-hex over the EXACT DEV-eligible namespaced subjects (from DEV subject_id_te)
per_cohort_eligible_subject_list_sha256  {cohort: 64-hex}
n_eligible_subjects                      STRICT int == EXACT_ELIGIBLE[disease] (PD 230 / SCZ 225)
excluded_subjects                        {"dsid/sub-xxx": reason} — every raw sub-* NOT eligible MUST be pinned here + is
                                         NEVER read (e.g. SCZ ds004000/sub-042 — in raw dirs but not in the DEV subject_id_te)
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
encoder checkpoint        + encoder_state_dict_sha256     = canonical semantic sha256(sorted name|dtype|shape|LE-bytes)
                          + encoder_checkpoint_file_sha256 = sha256(.pt file bytes)  [transport/integrity]
source-state artifact     + source_state_artifact_sha256  = acar.v3 SourceStateArtifact canonical (semantic) hash
                          + source_state_file_sha256       = sha256(.npz file bytes) [transport/integrity]
encoder provenance JSON    (ENCODER_ARTIFACT_FIELDS: arch=EEGNet, training command, data scope, seed, determinism,
                            torch/braindecode versions, embedding_dim==16, source_state_ref)
source-state provenance JSON
manifest LAST (manifest_sha256; RESULT sentinel)   ← "output complete" iff this exists
```

## 4. Runtime lock (regen is torch/braindecode/GPU — heavier than the pure-numpy v4 code)
Schema + validator + canonical hasher: `acar/v4/regen_envlock.py` (PURE; no torch capture). Required fields
(`expected_regen_env_fields`):
```
schema_version (acar_v4_regen_env_lock/1) · status · capture_note · python_version · torch_version ·
torchvision_version · torchaudio_version · braindecode_version · moabb_version · mne_version · skorch_version ·
numpy_version · scipy_version · sklearn_version · cuda_version · cudnn_version · device_kind ("cuda"|"cpu") · device_name ·
driver_version · torch_deterministic_algorithms (true) · seed (int 0) · torch_intraop_threads · torch_interop_threads ·
omp_num_threads · threadpool_backends · pipeline_config_sha256 · protocol_commit
```
- `status` ∈ {`SCHEMA_ONLY_NOT_CAPTURED`, `CAPTURED_AND_VERIFIED`, `CAPTURE_FAILED`}. `schema_only_template(...)` builds a
  reviewable skeleton (placeholder versions); `CAPTURE_FAILED` is an honest failure record. A **CAPTURED_AND_VERIFIED** lock
  MUST: fill real non-empty version/device fields incl. the **import-critical `torchvision_version`/`torchaudio_version`/
  `moabb_version`** (the eeg2025 failure was exactly a torchaudio↔torch + braindecode↔moabb mismatch — these are in the lock
  and therefore in its hash); if `device_kind==cuda`, fill cuda/cudnn/driver; and **pin `torch_intraop_threads ==
  torch_interop_threads == omp_num_threads == 1`** (the lock must capture the SAME deterministic single-thread runtime the
  training run uses — interop is set in a FRESH capture process before any inter-op work). A skeleton cannot impersonate a
  captured runtime. Capture tool: `acar/v4/capture_regen_envlock.py` (env introspection only; NO training/data).
- **Operational lock = repo-EXTERNAL, commit-consistent.** Capture against the CLEAN commit `H` that the preflight will run
  at (`--protocol-commit H`); write the lock to an absolute path OUTSIDE the repo (e.g.
  `/abs/acar_v4_regen_env_lock_<H>.json`); the manifest sets `env_lock_path`=that path + `env_lock_sha256`=its sha; the
  record commit only references the path+sha. This avoids the self-reference where committing the lock into the repo moves
  `HEAD` and makes `HEAD != protocol_commit` at preflight.
- **`run_regen_substrate` requires `status == CAPTURED_AND_VERIFIED`**, that the lock file's sha equals the manifest
  `env_lock_sha256`, and that the lock's `protocol_commit` + `pipeline_config_sha256` match the manifest. Capturing the real
  runtime (torch import on the chosen training node) is part of B1 — NOT done here.
- Device (GPU model / CPU) is PINNED in the lock — "node choice" is not a post-hoc degree of freedom (cf. the B0 finding
  that some DEV hashes came from a different CUDA device).

## 5. Atomic, no-overwrite output (matches v3/v4 runner discipline)
`--output` must be absent; the authorized run claims it atomically (os.mkdir) BEFORE any training; writes artifacts; writes
the manifest LAST; on any abort removes the claimed dir. A run is COMPLETE iff its RESULT/manifest sentinel exists. No
partial directory is ever interpreted as a usable artifact.

## 6. run_substrate_compatibility manifest schema (validate_substrate_manifest, fail-closed) — **C1 (executable, auth-gated)**
```
substrate_protocol_commit      40-hex — the FROZEN substrate-generation commit (b99fa4f); substrates were trained by it
compatibility_protocol_commit  40-hex — the C1 replay code commit; HEAD must == THIS (NOT the substrate commit)  ← two-commit split
                               (the retired bare `protocol_commit` is REJECTED)
candidate         EXACTLY {score_family: shift_margin, policy: benefit_ranked, loss: harm_indicator}  (NO reselection)
alpha/budget/coverage_min  EXACTLY 0.10 / 0.10 / 0.15  (pinned operating point)
substrates        {PD:{...}, SCZ:{...}} each (H5 4 unambiguous hashes; bare encoder_checkpoint_sha256/source_state_sha256
                  REJECTED): encoder_checkpoint_path + encoder_state_dict_sha256 + encoder_checkpoint_file_sha256,
                  source_state_path + source_state_artifact_sha256 + source_state_file_sha256,
                  encoder_provenance_path, source_state_provenance_path,
                  dev_input_manifest_path + dev_input_manifest_sha256   ← pins the EXACT eligible DEV universe to re-embed
                  (PD 230 / SCZ 225, ds004000/sub-042 excluded, FROZEN_PIPELINE, cohort-aware keys),
                  dev_feat_dump_paths {cohort: path} + dev_feat_dump_sha256 {cohort: 64-hex}   ← (C3) the DEV feat-dump
                  metadata (subject_id_te/recording_id_te/window_index_te/y_te) = the raw-window↔v3-WindowKey alignment
                  SOURCE OF TRUTH (sha-pinned; the re-embed order/keys cannot drift)
                  scps_cache_path (non-empty) + scps_cache_sha256 (64-hex)   ← (C5) the WINDOW SOURCE: the SHA-PINNED
                  per-condition scps cache {X,subject,cohort} the dump was built from; window_index_te IS its row index,
                  so the replay fetches X[window_index] by EXACT KEYED lookup (no live raw reader, no by-position guess)
dev_cohorts       {PD: DEV_SCOPE[PD], SCZ: DEV_SCOPE[SCZ]}  (exact)
env_lock_path     non-empty (the substrate env lock; re-verified at replay)        env_lock_sha256  64-hex
```
STDLIB preflight: schema + HEAD==compatibility_protocol_commit + clean worktree + output-absent + each PD/SCZ encoder/source
file + the dev-input-manifest file + each per-cohort dev-feat-dump file + each per-disease scps-cache file + the env-lock file
must EXIST and match their sha — fail-closed (FileNotFoundError / ValueError) before any torch import or DEV read.

**(C3) Safety gate uses the EXACT EVAL harm, not a proxy.** `compatibility_replay_pass`'s `L_harm_all_eval ≤ 0.10` is fed the
EXACT all-batch-denominator EVAL harm_indicator loss (`develop.V4CandidateReport.eval_L_harm_all`, additive), NOT the
conditional `harm_rate` (carried descriptively as `harm_among_adapted`, never gating). **(C5)** the raw-window↔v3-WindowKey
alignment (`_load_subject_windows_and_keys`) is REAL + synthetic-tested by EXACT KEYED lookup: the pinned dev-feat-dump metadata
supplies the WindowKeys (recording_id + window_index, verbatim) + labels, and — because `window_index_te` IS the row index of the
SHA-PINNED scps cache the dump was built from — `_load_disease_cache` loads the cohort-filtered {X,subject,cohort} from
`scps_cache_path` (`allow_pickle=False`) and each window is fetched at `cache["X"][window_index]`, VERIFYING
`cache.subject[gi]==subject` AND `cache.cohort[gi]==dataset_id` per row (NO live raw reader, NO by-position guess; NO source-state
refit; NO held-out read). Fail-closed on cache-sha / out-of-range index / subject-or-cohort-at-index mismatch (catches a same-count
REORDER) / shape / dup / finite. `_check_reembed_universe` asserts the re-embedded eligible set == eligible and counts EXACT. NO
`SubstrateReplayNotWiredError` raise-frontier remains; the only DEV read runs at the authorized C-run from the pinned cache (under
acar-v4-regen py3.13).

**C1 authorization gate (executable body).** Without `--compat-authorization` → `SubstrateCompatibilityNotAuthorizedError`
(no torch/cmi, no DEV read, no output). A valid compatibility authorization manifest (`validate_compat_authorization` +
`COMPAT_AUTH_FIELDS` + exact `REQUIRED_COMPAT_STATEMENT`) that BINDS (compatibility_protocol_commit / substrate_protocol_commit
/ substrate_manifest_sha256==this file / env_lock_sha256 / output_path) → `_authorized_replay_and_write`: atomic mkdir →
`_run_compatibility_replay` (runtime==env-lock verify + substrate SEMANTIC-hash verify run for REAL; re-embed old-seven DEV
under the new encoder + derive→FIXED-candidate exploration→`compatibility_replay_pass`) → `compat_manifest.json` then
`compat_RESULT.json` LAST (`allow_nan=False`). Result taxonomy = `SUBSTRATE_COMPAT_STATUSES`
(SUBSTRATE_COMPATIBILITY_PASS / _FAIL / OPERATIONALLY_ABORTED_NO_VERDICT — no selection/external/binding vocabulary). On any
operational abort the claimed output is removed (a partial dir is NEVER read as a FAIL). The two data-heavy stages
(`_reembed_dev_under_substrate`, `_extract_fixed_candidate_stats`) are now FULLY WIRED + synthetic-tested — **no
`SubstrateReplayNotWiredError` raise-frontier remains** (C4/C5): the re-embed reads windows from the sha-pinned scps cache by
EXACT KEYED lookup (window_index_te = cache row index; per-row subject/cohort-at-index verified) and the stats extractor reads
the FIXED-candidate `eval_L_harm_all`. The only DEV read (the pinned-cache `np.load`) still runs ONLY at the authorized C-run,
behind the authorization gate. See notes/ACAR_V4_C1_COMPAT_REPLAY_READINESS.md "C5 RESOLVED".
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

## 7b. Eligible-subject reconciliation (METADATA only; before any raw read)
`run_regen_substrate._verify_eligible_subjects` (preflight, before the B1 gate) lists `sub-*` dirs per cohort (no signal) and
calls `regen_substrate.check_eligible_subjects`: eligible = (all namespaced raw subjects) − `excluded_subjects`; every excluded
subject must exist on disk; eligible count must == `n_eligible_subjects` (== EXACT_ELIGIBLE[disease]); the eligible +
per-cohort hashes must match the manifest (so an extra raw subject not pinned excluded, a missing one, or a wrong-member set
all FAIL before any training). The eligible hashes are computed (manifest-build time) from the DEV substrate's `subject_id_te`
(the authoritative DEV-eligible universe).

## 7c. Executable training path + B1 authorization (gated, hash-bound)
The training body is a **real, reachable implementation** (NOT a placeholder / unconditional raise) behind an explicit,
hash-bound **B1 authorization manifest** (`--b1-authorization`; schema = `regen_substrate.validate_b1_authorization`,
template = `notes/ACAR_V4_B1B_TRAINING_AUTHORIZATION_TEMPLATE.md`). Without it → `SubstrateTrainingNotAuthorizedError` (no
torch/cmi import, no DEV read, no output). With a valid authorization that BINDS (protocol_commit / disease /
dev_input_manifest_sha256 / env_lock_sha256 / output_path / exact statement) → `_authorized_train_and_write` atomically
claims the output and calls `_train_substrate`, which orchestrates:
```
allowlist = _verify_eligible_subjects(spec)                 # eligible cohort-aware ids; excluded NOT included
X,y,subj  = load_eligible_windows(spec, allowlist)          # per-eligible-subject raw open (excluded NEVER opened); finite-checked
            check_training_set(y, subj, allowlist)          # every eligible subject present; labels {0,1}; both classes
bb,dev,encoder_state_dict_sha256 = _train_encoder_and_save(X, y, enc_path)   # cuda-only ERM (RS.TRAINING_SCHEDULE) + canonical sha
ss_art    = _fit_and_serialize_source_state(bb, dev, X, y, subj, spec, ss_path)  # acar.v3 fit→freeze→savez; embedding_dim==16 checked
return {encoder_checkpoint_path, encoder_state_dict_sha256, source_state_path, source_state_artifact_sha256(=ss_art canonical),
        embedding_dim, training_schedule, n_train_windows, n_eligible_subjects}
# _authorized_train_and_write: _verify_runtime_matches_lock (cuda+threads+versions) → mkdir → _train_substrate → add
#   encoder_checkpoint_file_sha256 + source_state_file_sha256 (file bytes) → manifest.json → RESULT.json last. The stdlib
#   preflight (_verify_env_lock) also rejects a non-cuda env lock before the B1 gate.
```
- **Excluded never read (B1):** `load_eligible_windows` iterates the ALLOWLIST only; `_load_subject_signal` calls the shared,
  tested `cmi.data.bids_data.load_cohort` with `subjects={subject}` — a discovery-stage filter that skips every other subject
  BEFORE its signal is opened (the only loader change; backward-compatible `subjects=None` default).
- **Cohort-aware matching (B2):** subjects are the canonical `dsid/sub` key throughout; the same local id in two cohorts is distinct.
- **source_kind (B3):** B1b training requires `source_kind == "raw_bids"` (canonical_features rejected by the validator).
- **Pinned schedule (B4):** `regen_substrate.TRAINING_SCHEDULE` fixes model/shape/optimizer/lr/weight_decay/batch_size/epochs/
  loss/class_weighting/seed/determinism/device in code (and below) — DECLARED as the NEW V4 substrate schedule (the original
  DEV ERM hyperparameters are not reconstructable); it is the substrate's provenance, not a runtime choice.
- **source-state (B5):** delegated to the FIXED acar.v3 fitter (`fit_source_state_artifact` → `freeze_source_state_artifact`
  → `np.savez`), no pickle / no `z_ev,y_ev`.
- Guards (tests monkeypatch `_train_substrate` / `load_eligible_windows` / `_load_subject_signal`; no real raw in tests):
  missing/invalid auth → fail closed; trainer called exactly once after all gates; encoder + source-state file sha256
  recorded; manifest `allow_nan=False`; trainer failure removes the claimed output; `cmi.load_cohort` honors the allowlist.

### Pinned TRAINING_SCHEDULE (declared NEW V4 substrate schedule — provenance, NOT a recovered DEV ERM schedule)
```
model=EEGNet  n_chans=19  n_times=512  embedding_dim=16  n_classes=2
optimizer=adam  lr=1e-3  weight_decay=0.0  batch_size=64  epoch_policy=fixed  max_epochs=100
loss=cross_entropy  class_weighting=balanced  val_split=0.0  device=cuda  seed=0  deterministic=true
```
The encoder embedding (readout input) is the backbone's z (forward → (logits, z)); the source-state f_0 is fit on the
all-DEV eligible embeddings. Recorded in the substrate manifest's `training_schedule`.

### B1b runtime / value / hash fail-closed guards (H4)
The authorized training path is hardened so any abnormal runtime, data, or model value aborts with NO usable artifact:
- **No silent CPU fallback:** `regen_substrate.require_cuda(schedule, cuda_available)` — training runs ONLY on cuda; if the
  schedule isn't `device=cuda` or `torch.cuda.is_available()` is False, it raises (no CPU substitution). Enforced both in
  `_train_encoder_and_save` and earlier by the runtime check below.
- **Runtime == captured env lock:** before any output is claimed or raw is read, `_verify_runtime_matches_lock` snapshots the
  live process with `capture_regen_envlock._probe` (same version-string methods as the lock; it also pins threads to 1) and
  asserts via `regen_substrate.check_runtime_matches_lock`: `device_kind=cuda` on both, the three thread fields = 1, and every
  version (`torch/torchvision/torchaudio/braindecode/moabb/mne/skorch/numpy/scipy/sklearn/python`) equals the lock. `device_name`
  is RECORDED but NOT required to match (only `device_kind=cuda` is hard). Any drift → abort before training.
- **Raw value / label fail-closed:** `_load_subject_signal` → `single_subject_label` (non-empty, all-identical, ∈{0,1}) +
  `assert_finite` (no NaN/Inf windows); `load_eligible_windows` rejects any 0-window subject + `assert_finite` over the whole
  set; `_train_substrate` → `check_training_set` (every eligible subject has ≥1 window, no non-eligible windows, labels ⊆{0,1},
  BOTH classes present).
- **Non-finite training fail-closed:** the ERM loop checks logits + loss finite each step, and parameters/grads finite after
  backward and after step (`_assert_model_params_finite`); embeddings are `assert_finite` before the source-state fit. A NaN/Inf
  model is never saved; the claimed output is removed on abort.
- **Encoder hash schema (no ambiguity):** the manifest records BOTH
  `encoder_state_dict_sha256` = `regen_substrate.canonical_state_dict_sha256` (sorted name|dtype|shape|little-endian bytes —
  the serialization-independent SEMANTIC provenance hash) AND `encoder_checkpoint_file_sha256` = sha256 of the `.pt` file bytes
  (transport/integrity). The source-state likewise records `source_state_artifact_sha256` (the acar.v3 artifact's canonical
  hash) + `source_state_file_sha256` (the `.npz` file bytes). The two meanings never overload one field. **(H5)** these four
  unambiguous names are used pipeline-wide — regen output manifest, the substrate-compatibility manifest
  (`validate_substrate_manifest` / `run_substrate_compatibility`), and the external encoder artifact
  (`prepare_external_dump.ENCODER_ARTIFACT_FIELDS`); the retired bare names `encoder_checkpoint_sha256` /
  `source_state_sha256` are now REJECTED in the substrate-artifact schema. (The only surviving `source_state_sha256` is the
  dump-PROVENANCE v3-recomputable hash in the external dump manifest — a different object, kept aligned with the acar.v3
  `SourceStateArtifact.source_state_sha256` attribute; same value as `source_state_artifact_sha256` by construction, verified
  independently.)

## 7d. This patch CHANGES the runner → re-sequence required before B1b (no reusing earlier artifacts)
The runner changed across H2 (e277f0d, eligible schema + auth gate), H3 (real executable training body + raw_bids-only), and
**H4 (this patch): runtime/value/hash fail-closed guards** (require_cuda, runtime==env-lock, finite/label checks, dual encoder
hash). So the 046507a env lock + manifests are STALE for the current code path. Before any B1b authorization, redo at the
LATEST commit H = H4 (this patch):
```
H = the B1b executable-training-path commit (clean)
recapture the regen env lock for H            (GPU; CAPTURED_AND_VERIFIED, interop=1, versions, protocol_commit=H)
rebuild PD/SCZ input manifests for H           (eligible_subject fields; source_kind=raw_bids; protocol_commit=H; env_lock_sha256=H lock)
rerun the fail-closed preflight at detached H  (expect SubstrateTrainingNotAuthorizedError; eligible check passes on real dirs)
THEN create the B1 authorization manifest(s) and ask for B1b.
```

## 8. B1 SIGN-OFF (the only thing that unlocks real training)
B1 authorizes all-DEV substrate regeneration EXACTLY as specified by: this file + `run_regen_substrate.py` +
`run_substrate_compatibility.py` + the fixed input manifests + the pinned regen env lock. Until B1 is signed:
`_require_b1_authorization` raises and `_train_and_write` is unreachable; no training occurs. Post-B1 sequence: train PD+SCZ
substrates → review artifact hashes → fixed-candidate compatibility replay → if PASS, update `ACAR_FROZEN_v4.md` with the new
substrate hashes → clean run + sign-off + tag `acar-v4-protocol` → THEN held-out preprocessing + the single external Arm-B
run. If B1 is declined or the replay fails → Option C (V4 stays DEV-only exploratory).
