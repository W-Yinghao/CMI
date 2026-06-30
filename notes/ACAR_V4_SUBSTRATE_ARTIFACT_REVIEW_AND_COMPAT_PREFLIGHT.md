# ACAR v4 — substrate artifact-hash REVIEW + compatibility PREFLIGHT **(PASS; NO replay / NO tag)**

```
RESULT : The B1b-trained PD + SCZ all-DEV substrates were INDEPENDENTLY re-hashed (read-only) and PASS every artifact check;
         a substrate-compatibility manifest was built + schema-validated; the fail-closed compatibility PREFLIGHT reached the
         compatibility-authorization gate (SubstrateCompatibilityNotAuthorizedError) with no output written. NO DEV raw read,
         NO re-embedding, NO compatibility replay, NO held-out/external read, NO acar-v4-protocol tag, NO lockbox access.
DATE   : 2026-06-30 (machine UTC)
PROTOCOL COMMIT (H)     : b99fa4fcfb83c6ee60996c50dba6828d40561f26
TRAINING RECORD COMMIT  : 332cd8ac02ea9c5c2e830c54787d4829a9d22f30   (notes/ACAR_V4_B1B_SUBSTRATE_TRAINING_RECORD.md)
RAN AT                  : detached clean worktree /home/infres/yinwang/ACAR_V4_COMPAT_PREFLIGHT_b99fa4f (HEAD==b99fa4f, clean)
ENV                     : acar-v4-regen (read-only artifact load; CPU; torch weights_only=True + acar.v3 load_frozen)
```

## 1. Artifact-hash review (read-only; independent re-computation — PASS both diseases)
For each disease, ALL of the following were INDEPENDENTLY recomputed from the on-disk artifacts and matched the values
recorded in the training manifest/RESULT (which match notes/ACAR_V4_B1B_SUBSTRATE_TRAINING_RECORD.md):
```
sha256(.pt file)                     == encoder_checkpoint_file_sha256
canonical state_dict hash            == encoder_state_dict_sha256   (torch.load weights_only=True → 21 tensors →
                                        regen_substrate.canonical_state_dict_sha256; NO unsafe pickle fallback)
sha256(.npz file)                    == source_state_file_sha256
acar.v3 SourceStateArtifact canonical == source_state_artifact_sha256   (acar.v3.loader.load_frozen_source_state_artifact
                                        self-verifies the blob's stored hash/ref, then == the recorded value)
manifest.json sha256                 == committed record (PD 8576020d… / SCZ 7e0e1e7a…)
RESULT 4 hashes == manifest · retired names (encoder_checkpoint_sha256/source_state_sha256) ABSENT · disease + embedding_dim==16 ·
protocol_commit == b99fa4f
```
Artifacts (repo-EXTERNAL; .pt/.npz remain in the artifact store; NOT committed):
```
PD  /home/infres/yinwang/acar_v4_regen_outputs/PD_all_dev_substrate_b99fa4f
    encoder_state_dict_sha256=10e29b2ffc61e61fffc162cd57ca44386fa5e0f2aa8ec7ccde7982fd2b4bb499
    encoder_checkpoint_file_sha256=573783a62b7f811cdf6282bf1e3a1e6e0a0ca2970ff1ec0d30516bf25ac13275
    source_state_artifact_sha256=efbcbd235b660a98eaaee865fa952fec44eaed5e55f0e73366a1334328d7c4cf
    source_state_file_sha256=57db99ab86865f510991602d6387edf0fbd654cfbc7f973655ee598e3ebd777b   (230 elig, 8523 win)
SCZ /home/infres/yinwang/acar_v4_regen_outputs/SCZ_all_dev_substrate_b99fa4f
    encoder_state_dict_sha256=f9c431632f4e9ca4054d7835f875a6ef671badd4335ffe14f2f0df1d4af865d0
    encoder_checkpoint_file_sha256=29be7be33f076db35575ec6e10874eed3f4176d7376b9fb37ef1a30cdb3c3b38
    source_state_artifact_sha256=084aac66dad68edeaa5f071772407a8fa6f86e53ee8b3022eebff4e48d4e13f3
    source_state_file_sha256=cd7a7ed7b390228c95c80c8184c005dc7bbde0e7e3ea86a699d3e27418438361   (225 elig, ds004000/sub-042 excluded, 9000 win)
```
Safe load: `torch.load(path, map_location="cpu", weights_only=True)` succeeded for both; no unsafe-pickle fallback was used.

## 2. Substrate-compatibility manifest (repo-EXTERNAL; schema-validated; for PREFLIGHT only)
```
path                       : /home/infres/yinwang/acar_v4_compat_manifests/acar_v4_substrate_manifest_b99fa4f.json
substrate_manifest_sha256  : 573cb42b507283aa6c5ffdac5f40b17fc966ad256984f9e3d34894459ba87af1
protocol_commit            : b99fa4f   fixed_candidate {score_family:shift_margin, policy:benefit_ranked, loss:harm_indicator}
operating point            : alpha 0.10 · budget 0.10 · coverage_min 0.15 · v2_replay HARD (no waiver)
env_lock_sha256            : 61e505b3…   training_record_commit : 332cd8a
substrates                 : PD + SCZ — each carries the 4 unambiguous hashes + paths (encoder_state_dict / encoder_checkpoint_file
                             / source_state_artifact / source_state_file); encoder/source provenance paths → the per-disease
                             manifest.json (the unified provenance; the run wrote no separate per-artifact provenance JSONs).
```
`regen_substrate.validate_substrate_manifest` PASSED (fixed candidate, pinned op-point, all 4 hashes 64-hex, retired names rejected).

## 3. Fail-closed compatibility preflight (expected; PASS)
```
cd /home/infres/yinwang/ACAR_V4_COMPAT_PREFLIGHT_b99fa4f   (HEAD==b99fa4f, clean)
PYTHONPATH=$PWD OMP_NUM_THREADS=1 <acar-v4-regen python> -m acar.v4.run_substrate_compatibility \
    --substrate-manifest <substrate manifest b99fa4f> --output <absent compat_should_not_replay_b99fa4f>
```
→ `acar.v4.regen_substrate.SubstrateCompatibilityNotAuthorizedError` (exit 1). Message: "The manifest validated (fixed
candidate, pinned operating point, trained-artifact hashes) + preflight pass … Decision uses
regen_substrate.compatibility_replay_pass (v2_replay HARD). No torch/cmi import, no DEV read, no output written."
report.input_manifest_sha256 = 573cb42b… (== the substrate manifest). Output dir NOT created (verified). ⇒ every preflight gate
passed (manifest validate + git HEAD==protocol_commit + clean worktree + output-absent + per-disease artifact FILE-byte hash
verification against the real .pt/.npz); the ONLY remaining blocker is the compatibility-replay authorization.

## Confirmations
no DEV raw read · no window/embed · no source-state fitting · no compatibility replay · no held-out/external read · no external
preprocessing · no acar-v4-protocol tag · no lockbox access. `.pt`/`.npz`/large binaries NOT committed (recorded by path + hash).
v2/v3 untouched; lockbox SEALED.

## Next gate (NOT started — separate decision): C1 compatibility-replay readiness
`run_substrate_compatibility.py` is still a PREFLIGHT-ONLY fail-closed command (raises before any torch/cmi import or DEV read).
Turning it into an executable replay body is a SEPARATE patch (C1), to be done with the same discipline as B1b: executable body
behind an explicit compatibility-authorization manifest, synthetic-tested, FIXED candidate only, v2-HARD pass-line, NO
reselection, NO score/loss/grid/comparator/threshold changes, atomic output + RESULT sentinel + abort cleanup; the replay is the
old-seven-DEV-substrate compatibility check, NOT a new DEV selection run. Only after C1 + an explicit replay authorization may a
compatibility verdict be produced; only a PASS unlocks ACAR_FROZEN_v4 + the acar-v4-protocol tag.
