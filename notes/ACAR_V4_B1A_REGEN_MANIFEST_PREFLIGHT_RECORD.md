# ACAR v4 — B1a regen-manifest fail-closed preflight record **(PASS at the B1 gate; NO training)**

> **SUPERSEDED for B1b by the B1b-readiness patch.** The 046507a manifests below lack the new eligible-subject fields
> (`eligible_subject_list_sha256`, `per_cohort_eligible_subject_list_sha256`, `n_eligible_subjects`, `excluded_subjects`) the
> runner now requires, and bind to commit 046507a. They remain a valid record of the 046507a preflight, but B1b uses the
> H2-rebuilt manifests + H2-recaptured env lock (see ACAR_V4_SUBSTRATE_REGEN_COMMAND.md §7d). The eligible universe is pinned
> PD 230 / SCZ 225, with SCZ ds004000/sub-042 excluded (in raw dirs, not in DEV subject_id_te).

```
DATE   : 2026-06-30 (machine UTC)
RESULT : PD + SCZ regen input manifests built (METADATA ONLY) and the fail-closed preflight reached the B1 gate
         (SubstrateTrainingNotAuthorizedError) for BOTH — confirming the B1 gate is the ONLY remaining blocker. No EEGNet
         training, no DEV raw SIGNAL read, no embedding, no source-state fit, no held-out/external read, no output written,
         no tag. B1a is fully complete; B1b (training authorization) is a separate decision.
PROTOCOL COMMIT (operational H) : 046507ad5a03dc38910a78bac7c29ec0bf8d48c1
PREFLIGHT RAN AT               : detached clean worktree /home/infres/yinwang/ACAR_V4_REGEN_PREFLIGHT_046507a
                                  (HEAD == 046507a, git status empty) — so HEAD == manifest protocol_commit.
ENV LOCK (repo-external)        : /home/infres/yinwang/acar_v4_regen_capture/acar_v4_regen_env_lock_046507a.json
                                  env_lock_sha256 = ceda567c376618739466254f4810e6cba2a76cab525cf2a0d43e82454cdd5b21
```

## Manifests (repo-EXTERNAL; not committed — referenced by path + sha)
```
PD  : /home/infres/yinwang/acar_v4_regen_manifests/acar_v4_regen_PD_inputs_046507a.json
      manifest_file_sha256 = 526ccd6a327677180d79637a5a55585812a7844412cd1028ca10f299501c8ac4
      dev_cohorts = ds002778, ds003490, ds004584 ; 230 subjects (sub-* dirs)
      subject_list_sha256         = a8198ea76761b8bb0a48ff011b2b8a8337b787e4d1613afebbd460af28ac4048
      diagnosis_label_sha256      = 038a87d1ea2066fae0cc73e5e5b8b96b02a6dc395d4c707713780670af9d6d29
      source_file_manifest_sha256 = 770e36dcdecd867e30b474339be1944a90683196d7552ebcab7244e04ea60e51
SCZ : /home/infres/yinwang/acar_v4_regen_manifests/acar_v4_regen_SCZ_inputs_046507a.json
      manifest_file_sha256 = 249ee2208143f97b020c52c030d0a8597140ea6842e48ca3dc5a36e2bece721c
      dev_cohorts = ds003944, ds003947, ds004000, ds004367 ; 226 subjects (sub-* dirs)
      subject_list_sha256         = 81a5aa3a7616d87cab40e1e7b3eeaae1fad88c2d6cbca94394c045b74d909d14
      diagnosis_label_sha256      = 9b82072b569937bf50f901343d1211d00bc611cac1407a8d89542075111dce26
      source_file_manifest_sha256 = f0f499619b592c4ab9fb4eb16846d09db30f93587ea4fc6a3419d88c5c2d6b42
```
Common: `protocol_commit=046507a`, `repo_clean_required=true`, `source_kind=raw_bids`, `seed=0`,
`pipeline_config_sha256=38250f16…`(canonical), `env_lock_path`+`env_lock_sha256` as above; `source_paths` =
`/projects/EEG-foundation-model/datalake/raw/scps/<cond>/<dsid>`.
NOTE: SCZ has 226 `sub-*` dirs vs the 225 DEV SCZ subjects — one extra raw dir; the eligible-subject filter is a
TRAINING-time concern (B1b), not a preflight blocker. Recorded for audit.

## Hash definitions (METADATA ONLY — no signal opened; canonical JSON, sorted keys, compact)
```
per_cohort_source_file_manifest_sha256[c] = sha256(sorted [(relpath, size_bytes)] of ALL files under the cohort dir;
                                            sizes via os.path.getsize — NO open()/signal read)
source_file_manifest_sha256               = sha256({cohort: per_cohort_hash})
subject_list_sha256                       = sha256(sorted unique ["{dsid}/{sub-*}"] across the disease's cohorts)
diagnosis_label_sha256                    = sha256({cohort: sha256(participants.tsv bytes)})   (diagnosis source provenance)
```
Inputs read: participants.tsv (bytes), directory listings, file names + sizes. NOT read: EDF/BDF/BrainVision signal arrays,
MNE, any windowing/filter/resample/embed.

## Preflight (both reached the B1 gate; expected fail-closed)
```
cd /home/infres/yinwang/ACAR_V4_REGEN_PREFLIGHT_046507a
PYTHONPATH=$PWD python -m acar.v4.run_regen_substrate --disease PD  --dev-input-manifest <PD manifest>  --output <absent>
PYTHONPATH=$PWD python -m acar.v4.run_regen_substrate --disease SCZ --dev-input-manifest <SCZ manifest> --output <absent>
```
Both → `acar.v4.regen_substrate.SubstrateTrainingNotAuthorizedError` (exit 1). Message confirms: "manifest + scope + env lock
+ output-absent + clean worktree + HEAD==protocol_commit all pass … No torch/cmi import, no DEV read, no output written."
Output dirs were NOT created (verified). ⇒ every preflight gate passed; the ONLY remaining blocker is the B1 authorization.

## Confirmations
no EEGNet/GPU training · no DEV raw SIGNAL read (metadata/file-names/sizes + participants.tsv only) · no window/embed/
source-state fit · no held-out/external read · no compatibility replay · no output artifacts · no acar-v4-protocol tag.
Detached worktree removed after the run. Manifests + env lock are repo-external (bound to 046507a). v2/v3 untouched;
lockbox SEALED.

## Next review point — B1b (separate decision)
B1b = authorize real all-DEV substrate training (the FIRST step that reads DEV raw signal + trains EEGNet/source-state),
exactly per ACAR_V4_SUBSTRATE_REGEN_COMMAND.md + run_regen_substrate.py + these fixed manifests + the operational env lock,
run at detached HEAD==046507a. Not implied by this preflight.
