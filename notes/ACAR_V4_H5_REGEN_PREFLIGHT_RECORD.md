# ACAR v4 — H5 regen-manifest fail-closed preflight record **(PASS at the B1 gate; NO training)**

```
RESULT : PD + SCZ regen input manifests rebuilt for H5 (METADATA ONLY) and the fail-closed preflight reached the B1 gate
         (SubstrateTrainingNotAuthorizedError) for BOTH — confirming the B1 authorization gate is the ONLY remaining blocker.
         No EEGNet training, no DEV raw SIGNAL read, no embedding, no source-state fit, no held-out/external read, no output
         written, no tag. B1b (training authorization) remains a SEPARATE decision; NO authorization manifest created.
DATE   : 2026-06-30 (machine UTC)
PROTOCOL COMMIT (H = H5) : b99fa4fcfb83c6ee60996c50dba6828d40561f26
PREFLIGHT RAN AT         : detached clean worktree /home/infres/yinwang/ACAR_V4_REGEN_PREFLIGHT_b99fa4f
                           (HEAD == b99fa4f, git status empty) — so HEAD == manifest protocol_commit.
```
Supersedes the 046507a B1a preflight record (which predates the eligible-subject schema (H2), the executable trainer (H3),
the runtime-safety guards (H4), and the unified dual-hash naming (H5)). The 046507a / b2fbbe8 env lock + manifests are STALE.

## Env lock (repo-EXTERNAL; bound to H5; NOT committed — referenced by path + file sha)
```
path                 : /home/infres/yinwang/acar_v4_regen_capture/acar_v4_regen_env_lock_b99fa4f.json
env_lock_file_sha256 : 61e505b3f0fd4246219dddc8c35778cb365b0edf650286b345bb494c547cab7e   (== manifests' env_lock_sha256)
canonical_hash       : 139d94ea8b80736b1620af0a97dfc6f1380718885be2afb305c966b418bdb769   (hash_regen_env_lock)
status               : CAPTURED_AND_VERIFIED          captured on A40 (node29), job 877074, env acar-v4-regen
protocol_commit      : b99fa4fcfb83c6ee60996c50dba6828d40561f26   (== H)
device_kind/name     : cuda / NVIDIA A40              threads intra/inter/omp = 1/1/1
versions             : torch 2.6.0+cu124 · torchvision 0.21.0+cu124 · torchaudio 2.6.0+cu124 · braindecode 1.5.2 · moabb 1.5.0
                       cuda 12.4 · cudnn + driver recorded non-empty
pipeline_config_sha256 : == canonical FROZEN_PIPELINE hash
```

## Manifests (repo-EXTERNAL; not committed — referenced by path + sha)
```
PD  : /home/infres/yinwang/acar_v4_regen_manifests/acar_v4_regen_PD_inputs_b99fa4f.json
      manifest_file_sha256 = 9475328d84343bf8fc2836e6c7e6f193ddb116d388fb66b49b4a8af679d604ab
      dev_cohorts = ds002778, ds003490, ds004584 ; raw 230 ; n_eligible_subjects 230 ; excluded {} (none)
      eligible_subject_list_sha256 = a8198ea76761b8bb0a48ff011b2b8a8337b787e4d1613afebbd460af28ac4048
      subject_list_sha256          = a8198ea76761b8bb0a48ff011b2b8a8337b787e4d1613afebbd460af28ac4048
      diagnosis_label_sha256       = 038a87d1ea2066fae0cc73e5e5b8b96b02a6dc395d4c707713780670af9d6d29
      source_file_manifest_sha256  = 770e36dcdecd867e30b474339be1944a90683196d7552ebcab7244e04ea60e51
SCZ : /home/infres/yinwang/acar_v4_regen_manifests/acar_v4_regen_SCZ_inputs_b99fa4f.json
      manifest_file_sha256 = 8e9bdff4472b0638f6cd4a7b40ede5e81d95b2c06eb6aa864319036eec15eea8
      dev_cohorts = ds003944, ds003947, ds004000, ds004367 ; raw 226 ; n_eligible_subjects 225 ;
      excluded {ds004000/sub-042: raw sub-* dir not in DEV substrate subject_id_te (dropped by the DEV pipeline)}
      eligible_subject_list_sha256 = 85d1b49f135b16e118c4c1dff287ca153439bf5ab99a61efd9b683be98c89f31
      subject_list_sha256          = 81a5aa3a7616d87cab40e1e7b3eeaae1fad88c2d6cbca94394c045b74d909d14
      diagnosis_label_sha256       = 9b82072b569937bf50f901343d1211d00bc611cac1407a8d89542075111dce26
      source_file_manifest_sha256  = f0f499619b592c4ab9fb4eb16846d09db30f93587ea4fc6a3419d88c5c2d6b42
```
Common: `protocol_commit=b99fa4f`, `repo_clean_required=true`, `source_kind=raw_bids`, `seed=0`,
`pipeline_config_sha256`=canonical FROZEN_PIPELINE hash, `env_lock_path`+`env_lock_sha256`=61e505b3… as above; `source_paths`=
`/projects/EEG-foundation-model/datalake/raw/scps/<cond>/<dsid>`. The raw-data provenance hashes (subject_list / diagnosis_label
/ source_file_manifest) are IDENTICAL to the 046507a record (same DEV raw data, metadata-only) — a consistency cross-check.

## Hash definitions (METADATA ONLY — no signal opened)
```
per_cohort_source_file_manifest_sha256[c] = sha256(sorted [(relpath, size_bytes)] of ALL files under the cohort dir; sizes via
                                            os.path.getsize — NO open()/signal read)
source_file_manifest_sha256               = sha256({cohort: per_cohort_hash})
eligible_subject_list_sha256              = canonical_subject_list_sha256(sorted unique DEV-eligible "{dsid}/{sub}") — the
                                            eligible universe is the DEV substrate's subject_id_te (read from feat_dump_v4 npz:
                                            ONLY the subject_id_te array, allow_pickle=False; never the signal arrays)
per_cohort_eligible_subject_list_sha256   = {cohort: canonical_subject_list_sha256(that cohort's eligible)}
subject_list_sha256                       = sha256(sorted unique raw "{dsid}/{sub-*}")
diagnosis_label_sha256                    = sha256({cohort: sha256(participants.tsv bytes)})
```
Inputs read: feat_dump_v4 `subject_id_te` arrays, participants.tsv bytes, directory listings, file names + sizes. NOT read:
EDF/BDF/BrainVision signal arrays, MNE, any windowing/filter/resample/embed.

## Preflight (both reached the B1 gate; expected fail-closed)
```
cd /home/infres/yinwang/ACAR_V4_REGEN_PREFLIGHT_b99fa4f
PYTHONPATH=$PWD OMP_NUM_THREADS=1 <acar-v4-regen python> -m acar.v4.run_regen_substrate \
    --disease PD  --dev-input-manifest <PD manifest b99fa4f>  --output <absent PD_should_not_train_b99fa4f>
PYTHONPATH=$PWD OMP_NUM_THREADS=1 <acar-v4-regen python> -m acar.v4.run_regen_substrate \
    --disease SCZ --dev-input-manifest <SCZ manifest b99fa4f> --output <absent SCZ_should_not_train_b99fa4f>
```
Both → `acar.v4.regen_substrate.SubstrateTrainingNotAuthorizedError` (exit 1). Message: "manifest + scope + env lock +
eligible subjects + output-absent + clean worktree + HEAD==protocol_commit all pass … No torch/cmi import, no DEV read, no
output written." Output dirs were NOT created (verified). ⇒ every preflight gate passed for H5 (incl. the cuda env-lock check
and the eligible-subject reconciliation on the REAL raw dirs); the ONLY remaining blocker is the B1 authorization.

## Confirmations
no EEGNet/GPU training (only the env-introspection lock capture used a GPU) · no DEV raw SIGNAL read (metadata/file-names/
sizes + participants.tsv + subject_id_te only) · no window/embed/source-state fit · no held-out/external read · no
compatibility replay · no output artifacts · no acar-v4-protocol tag · no external Arm-B. Detached worktree used for the run;
env lock + manifests are repo-external (bound to b99fa4f). v2/v3 untouched; lockbox SEALED.

## Next review point — B1b (separate decision; NO authorization manifest created yet)
B1b = authorize real all-DEV substrate training (the FIRST step that reads DEV raw signal + trains EEGNet + fits the
source-state), exactly per ACAR_V4_SUBSTRATE_REGEN_COMMAND.md + run_regen_substrate.py + these fixed H5 manifests + the H5
operational env lock, run at detached HEAD==b99fa4f. A B1 authorization manifest must explicitly bind: protocol_commit=b99fa4f,
disease, dev_input_manifest_sha256 (PD 9475328d… / SCZ 8e9bdff4…), env_lock_sha256=61e505b3…, output_path, authorized_by,
authorization_time, statement. Not implied by this preflight.
