# ACAR v4 — B1b all-DEV substrate TRAINING record **(PD + SCZ trained; ACCEPTED; NO replay / NO tag)**

```
RESULT : Authorized B1b all-DEV substrate training COMPLETE for BOTH diseases (PD then SCZ, sequential), at detached
         HEAD==b99fa4f, under the H5 env lock + H5 input manifests + B1 authorization manifests. Both runs exit 0, wrote the
         RESULT.json completion sentinel + manifest.json + encoder .pt + source-state .npz, and PASSED every acceptance
         condition. This is the FIRST authorized DEV raw-signal read + EEGNet training + source-state fit in V4.
         STOP HERE: NO fixed-candidate compatibility replay, NO ACAR_FROZEN_v4 update, NO acar-v4-protocol tag, NO held-out/
         external read. v2/v3 untouched; lockbox SEALED.
DATE   : 2026-06-30 (machine UTC)
PROTOCOL COMMIT (H = H5) : b99fa4fcfb83c6ee60996c50dba6828d40561f26
RAN AT                   : detached clean worktree /home/infres/yinwang/ACAR_V4_REGEN_PREFLIGHT_b99fa4f (HEAD==b99fa4f, clean)
ENV (isolated)           : acar-v4-regen ; GPU A40, node29 (same node as the H5 env-lock capture → driver match)
SLURM JOBS               : env-lock capture 877074 ; PD train 877100 ; SCZ train 877127
```

## Inputs (repo-EXTERNAL; bound to H5; referenced by path + sha)
```
env lock     : /home/infres/yinwang/acar_v4_regen_capture/acar_v4_regen_env_lock_b99fa4f.json
               file_sha256 = 61e505b3f0fd4246219dddc8c35778cb365b0edf650286b345bb494c547cab7e
               CAPTURED_AND_VERIFIED · device_kind=cuda (NVIDIA A40) · threads 1/1/1 · protocol_commit=b99fa4f
PD manifest  : /home/infres/yinwang/acar_v4_regen_manifests/acar_v4_regen_PD_inputs_b99fa4f.json   sha 9475328d84343bf8fc2836e6c7e6f193ddb116d388fb66b49b4a8af679d604ab
SCZ manifest : /home/infres/yinwang/acar_v4_regen_manifests/acar_v4_regen_SCZ_inputs_b99fa4f.json  sha 8e9bdff4472b0638f6cd4a7b40ede5e81d95b2c06eb6aa864319036eec15eea8
PD auth      : /home/infres/yinwang/acar_v4_b1_authorizations/B1b_PD_b99fa4f.json   sha e9735c94c7b0fccac8930662e5918c7fcc208cb836f9a4f2e6d3a1e7f02b6b71
SCZ auth     : /home/infres/yinwang/acar_v4_b1_authorizations/B1b_SCZ_b99fa4f.json  sha 6e71563f3f363e67fc0270ff812b6f7d7c89055f56718c9691f1a37f6d9eb13f
```
Each authorization binds: protocol_commit=b99fa4f · disease · dev_input_manifest_sha256 (matches above) · env_lock_sha256=61e505b3… ·
output_path · authorized_by="user instruction in ChatGPT conversation" · authorization_time · statement==REQUIRED_AUTH_STATEMENT.

## Training schedule (pinned RS.TRAINING_SCHEDULE — recorded verbatim in each substrate manifest; cross-checked == on read)
```
model=EEGNet  n_chans=19  n_times=512  embedding_dim=16  n_classes=2
optimizer=adam  lr=1e-3  weight_decay=0.0  batch_size=64  epoch_policy=fixed  max_epochs=100
loss=cross_entropy  class_weighting=balanced  val_split=0.0  device=cuda  seed=0  deterministic=true
```

## Substrate artifacts (repo-EXTERNAL; the .pt/.npz remain in the artifact store — recorded by path + the 4 unambiguous hashes)
### PD  — output /home/infres/yinwang/acar_v4_regen_outputs/PD_all_dev_substrate_b99fa4f
```
manifest.json sha256          = 8576020d60c6d78f1b26e8c389cb8b3ad5cc9459986c91b0edef12ffb3011d64
n_eligible_subjects           = 230   (no excluded)        n_train_windows = 8523   embedding_dim = 16
encoder  v4_alldev_encoder_PD.pt
  encoder_state_dict_sha256      = 10e29b2ffc61e61fffc162cd57ca44386fa5e0f2aa8ec7ccde7982fd2b4bb499   (canonical semantic)
  encoder_checkpoint_file_sha256 = 573783a62b7f811cdf6282bf1e3a1e6e0a0ca2970ff1ec0d30516bf25ac13275   (.pt bytes)
source-state  v4_alldev_source_state_PD.npz
  source_state_artifact_sha256   = efbcbd235b660a98eaaee865fa952fec44eaed5e55f0e73366a1334328d7c4cf   (acar.v3 canonical)
  source_state_file_sha256       = 57db99ab86865f510991602d6387edf0fbd654cfbc7f973655ee598e3ebd777b   (.npz bytes)
```
### SCZ — output /home/infres/yinwang/acar_v4_regen_outputs/SCZ_all_dev_substrate_b99fa4f
```
manifest.json sha256          = 7e0e1e7a8116a296121dffa3349bd431238dfafbacc7a8d412f295b569e1771f
n_eligible_subjects           = 225   (excluded ds004000/sub-042)   n_train_windows = 9000   embedding_dim = 16
encoder  v4_alldev_encoder_SCZ.pt
  encoder_state_dict_sha256      = f9c431632f4e9ca4054d7835f875a6ef671badd4335ffe14f2f0df1d4af865d0   (canonical semantic)
  encoder_checkpoint_file_sha256 = 29be7be33f076db35575ec6e10874eed3f4176d7376b9fb37ef1a30cdb3c3b38   (.pt bytes)
source-state  v4_alldev_source_state_SCZ.npz
  source_state_artifact_sha256   = 084aac66dad68edeaa5f071772407a8fa6f86e53ee8b3022eebff4e48d4e13f3   (acar.v3 canonical)
  source_state_file_sha256       = cd7a7ed7b390228c95c80c8184c005dc7bbde0e7e3ea86a699d3e27418438361   (.npz bytes)
```

## Acceptance (BOTH diseases — every condition PASS)
```
exit code 0 · output dir exists · RESULT.json (status=SUBSTRATE_TRAINED) + manifest.json sentinels · no .tmp/partial files ·
manifest allow_nan=false-clean · all 4 hashes present + 64-hex (in artifacts AND RESULT) · file-byte hashes == sha256 of the
on-disk .pt/.npz · file-byte hash != canonical semantic hash · training_schedule == RS.TRAINING_SCHEDULE exactly ·
n_eligible_subjects PD=230 / SCZ=225 · SCZ excluded == {ds004000/sub-042} · embedding_dim==16 · retired names
(encoder_checkpoint_sha256/source_state_sha256) ABSENT · live runtime verified == H5 env lock before any output/raw (the run
would have aborted otherwise).
```

## Logs (repo-EXTERNAL)
```
/home/infres/yinwang/acar_v4_regen_outputs/train_PD_b99fa4f_877100.out  (+ .err: only a benign EEGNetv4 deprecation alias warning)
/home/infres/yinwang/acar_v4_regen_outputs/train_SCZ_b99fa4f_877127.out (+ .err: same benign warning)
```

## Confirmations
no held-out/external read · no compatibility replay · no ACAR_FROZEN_v4 update · no acar-v4-protocol tag · no external Arm-B ·
no lockbox access. DEV raw signal was read ONLY for the eligible subjects (excluded subjects never opened — allowlist loader).
The .pt/.npz checkpoints are NOT committed (kept in the artifact store); the repo records their paths + the 4 hashes.

## Next gate (NOT started — separate decisions)
artifact-hash review → fixed-candidate compatibility replay under the v2-HARD rule → if PASS, update ACAR_FROZEN_v4.md with
these substrate hashes → clean tests / sign-off → tag acar-v4-protocol → held-out preprocessing → one external Arm-B run.
If the replay FAILS: V4 falls back to DEV-only exploratory reporting or a new dated protocol; the candidate / score / loss /
grid / comparator / thresholds must NOT be tuned after seeing the new-substrate replay.
