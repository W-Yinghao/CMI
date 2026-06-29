# ACAR v4 — external held-out input schema (FROZEN; raw → erm_0 dump)

```
STATUS : FROZEN CONTRACT (pre-tag) — defines how held-out raw EEG becomes an erm_0 feature dump for the Arm-B CLI.
SCOPE  : metadata + pipeline contract only. NO download / NO signal processing happens until acar-v4-protocol is tagged.
CODE   : acar/v4/prepare_external_dump.py (pure selectors/parsers/hashers tested; prepare_dump = executable wiring that
         is FAIL-CLOSED on the missing frozen encoder — FrozenEncoderMissingError, never retrains — see §7).
DATE   : 2026-06-29
```

The held-out sites (ACAR_FROZEN_v4.md §4: `zenodo14808296`/SCZ, `ds007526`/PD) have NO erm_0 dump yet. Before the
external read they must be processed by **exactly the DEV pipeline** that produced `feat_dump_v4`, so the held-out
features live in the same space as the frozen DEV source state. This document freezes that contract; the actual run is a
gated post-tag step.

## 1. Pipeline (must equal the DEV feat_dump_v4 pipeline — pin from the cmi config before tag)
```
raw BIDS EEG → resting-run selection (resting_run_selector; exclude walking for ds007526) → montage/channel harmonize →
resample → bandpass → epoch/window → the FROZEN CITA-no-LPC (erm:0) encoder → tangent features → erm_0 dump.
raw_pipeline_sha256 = raw_pipeline_sha256({resample_fs, bandpass, window_len, window_stride, montage, encoder_ref, ...})
                      and MUST equal the DEV pipeline's (pin the exact numeric params from the cmi preprocessing config).
encoder / source state = the DEV-FROZEN artifact (load_frozen_source_state_artifact) — NEVER refit on held-out labels.
```

## 2. Output dump schema (the v3 loader reads these)
```
z_te            float  [N, d]   tangent-feature embeddings (d == DEV embedding_dim)
subject_id_te   str    [N]      cohort-scoped subject id (cohort_id::subject_id downstream)
recording_id_te str    [N]
window_index_te int    [N]      unique per recording
y_te            int    [N]      diagnosis label ∈ {0,1} (0=HC, 1=patient) — used ONLY for ΔR at CAL λ*/EVAL scoring
feat_hash_te    str             content hash (also used as raw_pipeline_sha256 source in the DEV dumps)
```
`validate_dump_schema(arrays)` is an ALLOW-LIST (rejects source-fit fields like `z_ev`/`y_ev`) enforcing: all fields
present, `z_te` finite 2-D `[N≥1, d≥1]` with `d == embedding_dim`, non-empty string ids, non-negative window_index,
UNIQUE `(subject, recording, window)` rows, `y_te ∈ {0,1}`, and a 64-hex `feat_hash_te` — so a dump that passes here
passes the v3 read.

## 3. Per-site frozen specs (acar/v4/prepare_external_dump.DATASET_SPECS)
```
zenodo14808296 (SCZ): expected 64ch / 1000 Hz; resting eyes-closed; group→{patient:SZ/SCZ, control:HC}.
ds007526       (PD) : channels/Fs CONFIRM at prep (metadata-only); resting only (exclude walking/gait);
                      group→{PD:1, HC:0} from participants.tsv `group`.
```

## 4. Provenance (pinned in the external manifest → ACAR_FROZEN_v4.md §3/§5)
Per stratum: `raw_pipeline_sha256` (== DEV), `full_dump_sha256`, `deployment_input_sha256`, `label_sha256`,
`subject_list_sha256`, `diagnosis_mapping_sha256`, `resting_selection_sha256`, `source_state_ref`, `source_state_sha256`,
`dataset_version`, `expected_n_subjects`, `expected_embedding_dim`. The Arm-B CLI re-hashes the dump and verifies the
DEV-frozen source artifact's `source_state_sha256`/`ref` before any modeling read.

## 5. Firewall (binding)
The held-out diagnosis labels are written ONLY to `y_te` and consumed ONLY for ΔR at CAL λ* selection + EVAL scoring.
They never touch the encoder, the source-state fit, the label-free features, or the action choice before scoring. The
encoder + source state are the DEV-frozen artifacts.

## 6. Gated order
`prepare_dump` (post-tag) emits the dump + provenance; then the unique Arm-B CLI (`acar/v4/run_external_armb.py`) runs
the single confirmatory pass. No raw download / signal processing / encoder run occurs before `acar-v4-protocol` is
tagged.

## 7. Frozen encoder artifact — HARD BLOCKER (the cmi finding)
The DEV `erm_0` dumps saved only the *embeddings*, NOT the trained EEGNet encoder weights. To embed a held-out site into
the DEV feature space, `prepare_dump` REQUIRES a complete, on-disk, hash-verified frozen encoder + source-state artifact
(`acar/v4/prepare_external_dump.require_encoder_artifact`); absent/incomplete → **`FrozenEncoderMissingError`** (it NEVER
retrains/regenerates an encoder). **External Arm B is therefore NOT_YET_EXECUTABLE until this artifact is archived,
hashed, and pinned** — a separate later gated decision (recover the original checkpoint; or regenerate an all-DEV encoder
as an explicitly-declared new substrate; only if neither is possible, V4 stays a DEV-only exploratory candidate). This is
NOT a claim that external validation is infeasible.

Required `encoder_artifact` fields (all pinned + hash-verified before any raw read):
```
encoder_checkpoint_path · encoder_checkpoint_sha256 · encoder_architecture (EEGNet) · encoder_training_command ·
encoder_training_data_scope (which DEV cohorts the encoder was trained on) · encoder_seed · determinism (flags) ·
torch_version · braindecode_version · embedding_dim (== 16) · source_state_path · source_state_sha256 · source_state_ref
```
The frozen pipeline (`prepare_external_dump.FROZEN_PIPELINE`, validated by `validate_pipeline_config`) is pinned at
resample 128 Hz · bandpass 0.5–45 Hz · 4 s windows · 19-ch 10-20 canonical montage · EEGNet · embedding_dim 16, and MUST
equal the DEV pipeline.
