# ACAR v4 — external-substrate decision (encoder + held-out reader) **(OPEN; gated; no tag)**

```
STATUS : OPEN — External Arm B is NOT_YET_EXECUTABLE. Two hard, fail-closed blockers (below) must BOTH be resolved by an
         explicit, separately-signed-off decision before acar-v4-protocol can be tagged or any held-out raw is read.
SCOPE  : decision record only. NO retrain, NO download, NO signal processing, NO tag happens from this note.
DATE   : 2026-06-30
UPDATE (2026-06-29..30): Option A read-only search DONE, in-scope + out-of-scope → NOT_FOUND, A FORECLOSED in practice
         (ACAR_V4_ENCODER_CHECKPOINT_SEARCH.md, ACAR_V4_ENCODER_CHECKPOINT_SEARCH_OUT_OF_SCOPE.md). Option B is
         DESIGN-APPROVED FOR CODE SCAFFOLDING (no retrain): the held-out reader (acar/v4/heldout_reader.py — Blocker 2
         selection/validation/windowing/key layer) + substrate trainer SKELETON (acar/v4/regen_substrate.py — dry-run
         validator + SubstrateTrainingNotAuthorizedError + numeric compatibility_replay_pass) are implemented +
         synthetic-tested. Real all-DEV training is GATED behind separate B1 sign-off (ACAR_V4_SUBSTRATE_REGEN_PLAN.md).
         Option C NOT chosen.
```

External Arm B embeds held-out raw EEG into the **DEV feature space** (the frozen `feat_dump_v4` / erm_0 substrate) and
applies the DEV-frozen source state. The DEV erm_0 dumps saved only the **embeddings**, not the trained encoder, and the
held-out sites are not in the cmi cohort registry. So `acar/v4/prepare_external_dump.prepare_dump` is a FAIL-CLOSED
scaffold with two explicit blockers, both raised BEFORE any heavy import / raw read:

## Blocker 1 — frozen DEV encoder + source-state artifact (FrozenEncoderMissingError)
`prepare_dump` requires a complete, on-disk, hash-verified `encoder_artifact`
(`prepare_external_dump.require_encoder_artifact`, fields in `ENCODER_ARTIFACT_FIELDS` / SCHEMA §7). It NEVER trains or
regenerates an encoder. The original DEV EEGNet checkpoint is **not archived**, so this raises today.

## Blocker 2 — held-out raw→embedding reader (ExternalReaderNotWiredError)
`prepare_dump` embeds via `_embed_heldout_raw`, which is **not wired**: cmi's `load_crossdataset` only indexes registered
`COHORTS` and raises `KeyError` for the held-out sites (`ds007526`, `zenodo14808296`) — it CANNOT read them. A dedicated
held-out BIDS reader must be wired from the module's own primitives (`DATASET_SPECS` + `resting_run_selector` +
`parse_diagnosis_map` + `validate_channels_fs` → `X`, `y`, cohort-namespaced `subject_ids`), feeding the frozen encoder
to produce the dump + the provenance sidecar (`provenance_sidecar_dict` / `write_provenance_sidecar`). Until then this
raises (found by the post-commit adversarial audit of 226eb80, finding WIRING-1).

## Options (encoder substrate — pick ONE, with full sign-off; do NOT proceed silently)
```
A  RECOVER the original DEV EEGNet checkpoint + source-state (best: keeps the DEV substrate bit-identical).
   Verify by re-embedding a DEV cohort and checking feat_hash_te bit-matches an archived DEV erm_0 dump.
B  REGENERATE + archive an all-DEV V4 external substrate (a NEW, explicitly-declared frozen encoder/source-state).
   HONESTY REQUIREMENT: if it does not bit-reproduce the DEV erm_0 embeddings, it is a NEW external representation
   substrate, NOT "the original encoder". ACAR_FROZEN_v4.md must then be rewritten: "V4 external substrate = all-DEV
   frozen encoder/source-state produced by command X" — with its own training command, input scope, seed, env lock,
   artifact hashes, and replay/compatibility checks pinned in the protocol.
C  If neither A nor B is feasible: SUSPEND external Arm B; position V4 as a DEV-only exploratory candidate (no external
   confirmation claim).
```
Recommended order: **A → B → C** (search for the original checkpoint first; only then consider B). The held-out reader
(Blocker 2) must be wired and tested as part of whichever of A/B is chosen.

## Hard constraints (binding until a separate decision lands)
- NO retrain / regenerate without the all-DEV substrate's full provenance written into the protocol (Option B above).
- NO `acar-v4-protocol` tag while either blocker is open.
- NO held-out raw download / preprocessing / read before the tag.
- v2 (`9b2f0c1`) and v3 (`817b04f`/`9f4e83f`) frozen results and tags are immutable; the lockbox stays SEALED.

Related: [ACAR_FROZEN_v4.md](ACAR_FROZEN_v4.md) (§4/§5 status) · [ACAR_V4_EXTERNAL_INPUT_SCHEMA.md](ACAR_V4_EXTERNAL_INPUT_SCHEMA.md) (§7 encoder artifact + sidecar).
