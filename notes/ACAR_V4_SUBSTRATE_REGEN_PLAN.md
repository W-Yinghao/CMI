# ACAR v4 — Option B: all-DEV external substrate regeneration **PLAN (DESIGN ONLY — NO RETRAIN YET)**

```
STATUS          : DESIGN-APPROVED FOR CODE SCAFFOLDING / NO RETRAIN YET / NO EXTERNAL READ / NO TAG. The held-out reader
                  (acar/v4/heldout_reader.py) + substrate trainer SKELETON (acar/v4/regen_substrate.py) are implemented +
                  synthetic-tested; real all-DEV training requires the separate B1 sign-off below.
WHY             : the original DEV EEGNet encoder was never archived AND the DEV code path never saved it (no torch.save in
                  cmi/; run_scps_crossdataset np.savez's embeddings only). Confirmed NOT_FOUND in-scope + out-of-scope
                  (ACAR_V4_ENCODER_CHECKPOINT_SEARCH.md, ACAR_V4_ENCODER_CHECKPOINT_SEARCH_OUT_OF_SCOPE.md).
DATE            : 2026-06-29 (machine UTC)
DECISION CONTEXT: Option A foreclosed → B (this plan) or C (DEV-only). C NOT chosen yet. See ACAR_V4_ENCODER_ARTIFACT_DECISION.md.
```

## 0. Binding honesty requirement (this is NOT "recovering the original encoder")
**The DEV `erm_0` dumps are LEAVE-ONE-COHORT-OUT, per-fold** (`audit_{cond}_{coh}_erm_0.npz`: cohort `coh`'s embeddings come
from a model trained on the OTHER cohorts). There is therefore **no single "original DEV encoder"** to recover, and an
encoder trained on ALL DEV cohorts (needed to embed a NEW external site, which is not one of the LOSO folds) is structurally
a **different** model. It **cannot bit-reproduce** the per-fold `feat_hash_te`. ⇒ the regenerated artifact MUST be declared a
**NEW all-DEV V4 external representation substrate**, never "original/recovered". `ACAR_FROZEN_v4.md` must then be rewritten:
"V4 external substrate = all-DEV frozen encoder/source-state produced by command X", with its own command/scope/seed/env
lock/artifact hashes/replay checks pinned (per ACAR_V4_ENCODER_ARTIFACT_DECISION.md Option B).

## 1. Substrate type + scope
```
substrate       : disease-specific all-DEV EEGNet encoder + serialized source-state (one per disease; pooled over that
                  disease's DEV cohorts; NO held-out fold — this is the deliberate difference from the LOSO DEV dumps).
PD train scope  : ds002778, ds003490, ds004584                          (DEV PD cohorts, 230 subjects)
SCZ train scope : ds003944, ds003947, ds004000, ds004367               (DEV SCZ cohorts, 225 subjects)
held-out (NEVER): zenodo14808296 (SCZ), ds007526 (PD) — external; not touched during regeneration.
```

## 2. Preprocessing (MUST equal FROZEN_PIPELINE = the DEV feat_dump_v4 pipeline)
```
19-ch 10-20 canonical montage · resample 128 Hz · bandpass 0.5–45 Hz · 4 s / 512-sample windows · EEGNet · embedding_dim 16
```
(`acar/v4/prepare_external_dump.FROZEN_PIPELINE`; `validate_pipeline_config` enforces equality.)

## 3. Seed / determinism / environment
```
config          : erm:0 (the CITA-no-LPC ERM encoder used by the DEV substrate)
seed            : 0 (deterministic: use_deterministic_algorithms, single-thread, seeded py/np/torch)
env             : eeg2025 conda (torch); pin a regen env lock via acar/v4/regen_envlock.py (schema + validator + canonical
                  hash implemented; status SCHEMA_ONLY_NOT_CAPTURED until a real CAPTURED_AND_VERIFIED capture on the
                  training node at B1). Records torch/braindecode/numpy/scipy/sklearn + CUDA/cuDNN + PINNED device +
                  determinism + seed 0 + threads. run_regen_substrate requires CAPTURED_AND_VERIFIED + file-hash match.
```

## 4. Training command (a NEW, declared, ADD-ONLY step — the original pipeline has no save)
The existing `cmi.run_scps_crossdataset --configs erm:0` does NOT persist weights. Option B requires a NEW dedicated
all-DEV trainer (or an ADD-ONLY `--save-encoder` step) that:
```
- trains ONE erm:0 EEGNet per disease on ALL that disease's DEV cohorts (no held-out fold),
- torch.save's the encoder state_dict (canonical little-endian bytes),
- fits + serializes the matching source-state artifact (reuse acar.v3 SourceStateArtifact discipline),
- emits the provenance fields below.
The EXACT command + input/output/runtime-lock schema + guards are FROZEN in notes/ACAR_V4_SUBSTRATE_REGEN_COMMAND.md +
acar/v4/run_regen_substrate.py (B1-gated, fail-closed: validates fully then raises SubstrateTrainingNotAuthorizedError —
no torch/cmi import, no DEV read — until B1 sign-off). It is run ONLY after B1.
```

## 5. Expected artifacts (must satisfy prepare_external_dump.ENCODER_ARTIFACT_FIELDS / SCHEMA §7)
```
PD  encoder checkpoint  + PD  source-state artifact
SCZ encoder checkpoint  + SCZ source-state artifact
each with: encoder_checkpoint_path/sha256 · encoder_architecture(EEGNet) · encoder_training_command ·
           encoder_training_data_scope(the cohorts in §1) · encoder_seed · determinism · torch/braindecode versions ·
           embedding_dim(==16) · source_state_path/sha256 · source_state_ref
```

## 6. Artifact hash schema
```
encoder_checkpoint_sha256 = sha256(canonical little-endian state_dict bytes)
source_state_sha256       = acar.v3 SourceStateArtifact full-bytes hash (coef/intercept/CLASSES_/moments/priors/schema/...)
sidecar (per external dump, later) = provenance_sidecar_dict(...) sha-pinned by provenance_sidecar_sha256 (already implemented)
```

## 7. Compatibility GATE (PRE-REGISTERED; numeric; prevents post-hoc number chasing)
**This is NOT a new scientific DEV selection run.** It is a substrate-compatibility check for the ALREADY-FIXED V4
candidate; its only purpose is to decide whether external Arm B may run under the new substrate. (This sentence is binding
and also goes in the closeout/result note.)
```
candidate (FIXED) : shift_margin + benefit_ranked + harm_indicator   (registry sha fe5a1f58…)
NO reselection    : no new score family · no new policy family · no new loss · no new λ grid · no comparator switch
where             : the OLD SEVEN DEV cohorts, re-embedded by the NEW substrate (DEV, exploratory — NOT external)
branching:
  1. if the new substrate BIT-REPRODUCES the archived DEV feat_hash_te → recovered-equivalent (will NOT happen for an
     all-DEV encoder vs LOSO per-fold dumps — see §0; documented as the expected branch-2 outcome).
  2. else → DECLARE a NEW V4 external substrate and run the FIXED-candidate DEV compatibility replay below.
```
Numeric pass-line (implemented as the PURE `acar.v4.regen_substrate.compatibility_replay_pass`):
```
per disease (PD, SCZ), ALWAYS required:
    CAL LTT λ* certified  AND  coverage ≥ 0.15  AND  red > 0  AND  EVAL L_harm_all ≤ 0.10
per-disease v2 gate (HARD — NO waiver):
    v2_replay MUST be EVALUABLE for BOTH PD and SCZ, AND red > v2_replay_red for EACH disease.
    if v2_replay is NOT evaluable for either disease → the replay FAILS → external Arm B NOT authorized.
    (rationale: beating v2_replay is the V4 external claim; a non-evaluable v2 must not become an interpretation freedom.)
macro v2 gate:
    require disease-macro red > disease-macro v2_replay.
AUTHORIZE external Arm B iff BOTH diseases pass all absolute + v2 gates AND the macro v2 gate.
on PASS → allowed to draft the final frozen protocol with the NEW artifact hashes (then sign-off → tag).
on FAIL (any disease) → external Arm B NOT authorized → Option C (DEV-only) or a new dated protocol.
```
caveat (record + closeout): re-embedding DEV cohorts with an all-DEV (in-sample) encoder is OPTIMISTIC vs the LOSO
exploration; the replay tests substrate COMPATIBILITY/viability of the fixed candidate, NOT a fresh generalization claim.

## 8. Parallel deliverable — Blocker 2 (held-out raw→embedding reader)
Even with the encoder, external Arm B also needs the held-out BIDS reader wired (currently `ExternalReaderNotWiredError`).
Design + test (synthetic) a held-out reader from the module's own primitives — `DATASET_SPECS` + `resting_run_selector` +
`parse_diagnosis_map` + `validate_channels_fs` → `X`, `y`, cohort-namespaced `subject_ids` → frozen encoder → dump +
provenance sidecar. This is a separate, also-gated work item; both blockers must clear before any tag / external read.

## 9. Stop rules → Option C (DEV-only)
Pick C if ANY of: (a) the DEV raw / GPU / pipeline needed for §4 is unavailable; (b) the fixed-candidate compatibility
replay (§7.4) fails the predeclared minimum; (c) you decline the cost of regenerating the substrate. Until one of these,
External Arm B status stays **NOT_YET_EXECUTABLE** (not infeasible).

## 10. Hard constraints (binding)
NO retrain until this plan is reviewed/approved · NO `acar-v4-protocol` tag · NO held-out raw download/preprocess/read ·
lockbox SEALED · v2 (`9b2f0c1`) and v3 (`817b04f`/`9f4e83f`) frozen results+tags immutable · the V4 DEV candidate stays
EXPLORATORY (DEV exploration #001) — regeneration does not promote it; only a passing §7 gate + frozen protocol does.
