# ACAR v4 — external held-out input schema (FROZEN; raw → erm_0 dump)

```
STATUS : FROZEN CONTRACT (pre-tag) — defines how held-out raw EEG becomes an erm_0 feature dump for the Arm-B CLI.
SCOPE  : metadata + pipeline contract only. NO download / NO signal processing happens until acar-v4-protocol is tagged.
CODE   : acar/v4/prepare_external_dump.py (pure selectors/parsers/hashers/sidecar tested; prepare_dump = FAIL-CLOSED
         scaffold with TWO hard blockers — FrozenEncoderMissingError (no archived encoder; never retrains) and
         ExternalReaderNotWiredError (no held-out BIDS reader; cmi load_crossdataset can't read these sites) — see §7).
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
Per stratum: the 8 hash fields `full_dump_sha256`, `deployment_input_sha256`, `label_sha256`, `subject_list_sha256`,
`diagnosis_mapping_sha256`, `resting_selection_sha256`, `raw_pipeline_sha256`, `source_state_sha256`; plus
`source_state_ref`, `provenance_sidecar_sha256`, `dataset_version`, `expected_n_subjects`, `expected_embedding_dim`.
What the Arm-B CLI VERIFIES at run time (under the atomic `os.mkdir(<out>)` claim, before any modeling read):
- `full_dump_sha256` recomputed from the dump bytes;
- the **provenance sidecar** `<dump>.provenance.json` is sha-pinned (`provenance_sidecar_sha256`) and EVERY manifest hash
  field + `source_state_ref` must equal the sidecar — so the 3 prep-only hashes (`raw_pipeline`/`diagnosis_mapping`/
  `resting_selection`), which can't be recomputed from the .npz, are bound to frozen-prep output, not hand-filled;
- `deployment_input_sha256`/`label_sha256`/`subject_list_sha256` RE-COMPUTED via `acar.v3.loader.{hash_deployment_input,
  hash_labels,hash_subject_list}` (these MUST come from the v3 loaders, NOT a same-named prep helper);
- `source_state_sha256`/`source_state_ref` re-checked against the loaded DEV-frozen artifact (this dump-PROVENANCE
  `source_state_sha256` is the acar.v3 `SourceStateArtifact` canonical/semantic hash — same value as the encoder artifact's
  `source_state_artifact_sha256` by construction, but a v3-bound name verified via the v3 recompute path, NOT a file hash);
- `expected_n_subjects` / `expected_embedding_dim` checked against the built stratum / artifact.

## 5. Firewall (binding)
The held-out diagnosis labels are written ONLY to `y_te` and consumed ONLY for ΔR at CAL λ* selection + EVAL scoring.
They never touch the encoder, the source-state fit, the label-free features, or the action choice before scoring. The
encoder + source state are the DEV-frozen artifacts.

## 6. Gated order
`prepare_dump` (post-tag, post-blockers) emits the dump + its `<dump>.provenance.json` sidecar; then the unique Arm-B CLI
(`acar/v4/run_external_armb.py`) runs the single confirmatory pass. No raw download / signal processing / encoder run
occurs before `acar-v4-protocol` is tagged AND both §7 blockers are resolved (`notes/ACAR_V4_ENCODER_ARTIFACT_DECISION.md`).

## 7. HARD BLOCKERS to executability (two; both fail-closed; decision = ACAR_V4_ENCODER_ARTIFACT_DECISION.md)
**Blocker 1 — frozen encoder + source-state (FrozenEncoderMissingError).** The DEV `erm_0` dumps saved only the
*embeddings*, NOT the trained EEGNet encoder weights. To embed a held-out site into the DEV feature space, `prepare_dump`
REQUIRES a complete, on-disk, hash-verified frozen encoder + source-state artifact
(`acar/v4/prepare_external_dump.require_encoder_artifact`); absent/incomplete → **`FrozenEncoderMissingError`** (it NEVER
retrains/regenerates an encoder).

**Blocker 2 — held-out raw→embedding reader (ExternalReaderNotWiredError).** cmi's `load_crossdataset` only indexes
registered `COHORTS` and raises `KeyError` for the held-out sites (`ds007526`/`zenodo14808296`) — it CANNOT read them. A
dedicated held-out BIDS reader (DATASET_SPECS + `resting_run_selector` + `parse_diagnosis_map` + `validate_channels_fs` →
`X`, `y`, cohort-namespaced `subject_ids`) must be wired at provisioning time; `prepare_dump._embed_heldout_raw` raises
**`ExternalReaderNotWiredError`** until then (audit finding WIRING-1). prepare_dump NEVER mis-routes through
load_crossdataset.

**External Arm B is therefore NOT_YET_EXECUTABLE** until BOTH blockers are resolved by a separate signed-off decision
(`notes/ACAR_V4_ENCODER_ARTIFACT_DECISION.md`: A recover original / B regenerate+declare all-DEV substrate / C suspend →
DEV-only). This is NOT a claim that external validation is infeasible.

**Provenance sidecar.** Once both blockers clear, `prepare_dump` writes `<dump>.provenance.json`
(`provenance_sidecar_dict` → all 8 hash fields + `source_state_ref`, schema `acar_v4_external_provenance/1`). The Arm-B CLI
sha-pins it (`provenance_sidecar_sha256`) and asserts every manifest hash field equals it (§4), binding the 3 prep-only
hashes to frozen-prep output.

Required `encoder_artifact` fields (all pinned + hash-verified before any raw read; **H5** dual-hash naming — canonical
SEMANTIC + file-bytes, never overloaded; the retired bare `encoder_checkpoint_sha256` / `source_state_sha256` are REJECTED here):
```
encoder_checkpoint_path · encoder_state_dict_sha256 (canonical semantic) · encoder_checkpoint_file_sha256 (.pt bytes) ·
encoder_architecture (EEGNet) · encoder_training_command · encoder_training_data_scope (which DEV cohorts) · encoder_seed ·
determinism (flags) · torch_version · braindecode_version · embedding_dim (== 16) ·
source_state_path · source_state_artifact_sha256 (acar.v3 canonical) · source_state_file_sha256 (.npz bytes) · source_state_ref
```
`require_encoder_artifact` verifies the `*_file_sha256` (stdlib file-bytes) at prep; the canonical `encoder_state_dict_sha256`
/ `source_state_artifact_sha256` are required + 64-hex and re-verified by the loader (torch / acar.v3) at run time.
The frozen pipeline (`prepare_external_dump.FROZEN_PIPELINE`, validated by `validate_pipeline_config`) is pinned at
resample 128 Hz · bandpass 0.5–45 Hz · 4 s windows · 19-ch 10-20 canonical montage · EEGNet · embedding_dim 16, and MUST
equal the DEV pipeline.
