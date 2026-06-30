# ACAR v4 — C5b C-run-readiness record: runtime lock + compat manifest + fail-closed PREFLIGHT **(PASS; NO replay / NO tag)**

```
RESULT : C-run-readiness sequence executed at the C5b compatibility commit. A compatibility RUNTIME LOCK was captured +
         verified on an A40 GPU node; the substrate-compatibility MANIFEST was rebuilt to the current (C1/C3/C5/C5b) schema
         and every sha-pinned file independently re-verified on disk; the fail-closed compatibility PREFLIGHT reached the
         compatibility-authorization gate (SubstrateCompatibilityNotAuthorizedError) with the FULL preflight passing and NO
         output written. This is NOT a compatibility replay. NO DEV cache window read at preflight (file-byte sha only),
         NO re-embedding, NO held-out/external read, NO ACAR_FROZEN_v4 update, NO acar-v4-protocol tag, NO lockbox access.
DATE   : 2026-06-30 (machine UTC)
SUBSTRATE COMMIT (B1b/H5)      : b99fa4fcfb83c6ee60996c50dba6828d40561f26
COMPATIBILITY COMMIT (C5b)     : 5237378a7c8e4cd5c35aff8efb59701f148575c0
RAN AT (preflight)             : detached clean worktree /home/infres/yinwang/ACAR_V4_COMPAT_PREFLIGHT_5237378
                                 (HEAD == 5237378…, git status --porcelain --untracked-files=all EMPTY)
ENV                            : acar-v4-regen (python 3.13.14); preflight is STDLIB-only (no torch/cmi import on this path)
```

## 1. Compatibility runtime lock (GPU introspection only — NO training/replay/data)
```
captured via : python -m acar.v4.capture_regen_envlock --output <lock> --protocol-commit 5237378…   (SLURM A40, job 877655, node27)
lock path    : /home/infres/yinwang/acar_v4_compat_capture/acar_v4_compat_env_lock_5237378.json
env_lock_sha256 (file bytes) = 523b44bc9480551c96b9f6d50bbadc7fccf027ebf2689d04379324f1dc3b767c
status=CAPTURED_AND_VERIFIED · protocol_commit=5237378… · device_kind=cuda (NVIDIA A40, node27) · threads intra/inter/omp = 1/1/1
pipeline_config_sha256 = 38250f16e8a456076b69abcae2336101aabebde51e2f9ee697c8bd354ac2848d   (== B1b H5 lock, canonical)
versions == accepted regen env (== B1b lock): torch 2.6.0+cu124 · torchvision 0.21.0+cu124 · torchaudio 2.6.0+cu124 ·
            braindecode 1.5.2 · moabb 1.5.0 · numpy 2.4.4 · scipy 1.18.0 · sklearn 1.9.0 · python 3.13.14 · cuda 12.4 ·
            cudnn 90100 · driver 610.43.02
```
All lock checks PASS (status / protocol_commit / device_kind / 1/1/1 / pipeline_config_sha256==B1b / every version==B1b / driver==B1b).

## 2. Substrate-compatibility manifest (repo-EXTERNAL; current C1/C3/C5/C5b schema; rebuilt)
```
manifest path              : /home/infres/yinwang/acar_v4_compat_manifests/acar_v4_substrate_compat_manifest_5237378.json
substrate_manifest_sha256  : 6c6ab457aed9f9925380214d154c407037e002cf52c8954976e98367d1b2e605
substrate_protocol_commit  : b99fa4fcfb83c6ee60996c50dba6828d40561f26
compatibility_protocol_commit : 5237378a7c8e4cd5c35aff8efb59701f148575c0
candidate (FIXED, no reselection) : {score_family: shift_margin, policy: benefit_ranked, loss: harm_indicator}
operating point            : alpha 0.10 · budget 0.10 · coverage_min 0.15 · v2_replay HARD (no waiver)
env_lock_path/sha          : <the compat lock above> / 523b44bc…
dev_cohorts                : PD {ds002778, ds003490, ds004584} · SCZ {ds003944, ds003947, ds004000, ds004367}
```
`regen_substrate.validate_substrate_manifest` PASSED; the DEPRECATED bare `protocol_commit` field is REJECTED (two-commit split required).

### Per-disease pins (all independently re-verified == on-disk file bytes BEFORE the preflight)
```
PD  output /home/infres/yinwang/acar_v4_regen_outputs/PD_all_dev_substrate_b99fa4f
    encoder_state_dict_sha256      = 10e29b2ffc61e61fffc162cd57ca44386fa5e0f2aa8ec7ccde7982fd2b4bb499
    encoder_checkpoint_file_sha256 = 573783a62b7f811cdf6282bf1e3a1e6e0a0ca2970ff1ec0d30516bf25ac13275
    source_state_artifact_sha256   = efbcbd235b660a98eaaee865fa952fec44eaed5e55f0e73366a1334328d7c4cf
    source_state_file_sha256       = 57db99ab86865f510991602d6387edf0fbd654cfbc7f973655ee598e3ebd777b
    dev_input_manifest_sha256      = 9475328d84343bf8fc2836e6c7e6f193ddb116d388fb66b49b4a8af679d604ab
    dev_feat_dump_sha256  ds002778 = 9011eab63ae0d27ff73e3906727819c9a7c4a20fc00a186b886a8636bd11b6af
                          ds003490 = 60a4a9a85098027c3491e2f29a401ba9f34d147e51e8ae765e858dd4e3242f30
                          ds004584 = 42a6c548091dd116f601554276a19b1925a77e1259f0a870447d860c0db96662
    scps_cache_sha256 (PD.npz)     = b1115902e549113bf88fc35a6d4f85027d8a3eddce7d0685ead36e4cd40b41c8
SCZ output /home/infres/yinwang/acar_v4_regen_outputs/SCZ_all_dev_substrate_b99fa4f
    encoder_state_dict_sha256      = f9c431632f4e9ca4054d7835f875a6ef671badd4335ffe14f2f0df1d4af865d0
    encoder_checkpoint_file_sha256 = 29be7be33f076db35575ec6e10874eed3f4176d7376b9fb37ef1a30cdb3c3b38
    source_state_artifact_sha256   = 084aac66dad68edeaa5f071772407a8fa6f86e53ee8b3022eebff4e48d4e13f3
    source_state_file_sha256       = cd7a7ed7b390228c95c80c8184c005dc7bbde0e7e3ea86a699d3e27418438361
    dev_input_manifest_sha256      = 8e9bdff4472b0638f6cd4a7b40ede5e81d95b2c06eb6aa864319036eec15eea8
    dev_feat_dump_sha256  ds003944 = 92923fb8c1ba1dd81f39e02457f8b6a0e9140fdba8bb8bd18057c4bbb43e02bb
                          ds003947 = 60ac3e308e75bf4a7f5d8864fc1f252fba2b75dcc393073f768f77e0ccc410ce
                          ds004000 = ddcb8b819d5953bedb2e437df101da56a80f3eddceee000bdca4421b66ca21d1
                          ds004367 = 7a514a63722d24025e3dcb8773606eacd61e5fd8577c1216355013dd4bc44688
    scps_cache_sha256 (SCZ.npz)    = 15b2b59bcd3f80ca39621452786fd092353d8627950324a0df6cd29896155571
```
DEV feat dumps (the alignment SOURCE OF TRUTH; subject_id_te/recording_id_te/window_index_te/y_te): cohort-keyed
`/home/infres/yinwang/CMI_AAAI/archive/lpc-cmi-failed/results/feat_dump_v4/audit_{disease}_{cohort}_erm_0.npz` (acar.config.feat_dump_dir).
scps caches (the WINDOW SOURCE; keyed by the dump's global window_index): `/projects/EEG-foundation-model/datalake/raw/scps/cache/{PD,SCZ}.npz`.

## 3. Fail-closed compatibility preflight (expected; PASS)
```
cd /home/infres/yinwang/ACAR_V4_COMPAT_PREFLIGHT_5237378     # HEAD == 5237378…, clean
PYTHONPATH=$PWD OMP_NUM_THREADS=1 <acar-v4-regen python> -m acar.v4.run_substrate_compatibility \
    --substrate-manifest /home/infres/yinwang/acar_v4_compat_manifests/acar_v4_substrate_compat_manifest_5237378.json \
    --output /home/infres/yinwang/acar_v4_compat_preflight/compat_should_not_replay_5237378
```
→ `acar.v4.regen_substrate.SubstrateCompatibilityNotAuthorizedError` (exit 1). report.input_manifest_sha256 = 6c6ab457… (== the
manifest), compatibility_protocol_commit = 5237378…, substrate_protocol_commit = b99fa4f…, result_taxonomy =
{SUBSTRATE_COMPATIBILITY_PASS, SUBSTRATE_COMPATIBILITY_FAIL, OPERATIONALLY_ABORTED_NO_VERDICT}. The error states the manifest
validated (two-commit, fixed candidate, pinned op-point, trained-artifact + dev-input + env-lock file hashes) and **the full
preflight passed** ⇒ every preflight gate passed: schema validate + git HEAD==compatibility_protocol_commit + clean worktree +
output-absent + per-disease encoder/source-state/dev-input-manifest/dev-feat-dump/scps-cache + env-lock FILE-BYTE hash
verification. The ONLY remaining blocker is the compatibility-replay authorization. **Output dir NOT created** (verified). The
traceback contains NO torch/cmi/numpy frames (runpy → main → run → _require_compat_authorization → raise) ⇒ no torch/cmi import,
**no `np.load` of the scps cache** (file-byte sha only).

## 4. C5b provenance diagnostic disclosure (read-only; pre-build, NOT the preflight)
Before this readiness sequence, the C5b subject-namespacing fix was confirmed by a read-only diagnostic that **read the scps cache
window arrays once** to verify cache↔dump alignment end-to-end (the shipping `_load_subject_windows_and_keys` re-aligned EXACTLY
230 PD / 225 SCZ subjects, 8523 / 9000 windows, against the real sha-pinned caches). It produced **no replay output, no
compatibility verdict, no external result, and no tag**. Future readiness preflight remains **file-SHA / provenance-only** until an
explicit compatibility authorization. This diagnostic is NOT counted as a replay and is NOT scientific evidence.

## Confirmations
no compatibility replay · no DEV cache window read at preflight (file-byte sha only) · no re-embedding · no held-out/external read ·
no external preprocessing · no compatibility-authorization manifest created · no ACAR_FROZEN_v4 update · no acar-v4-protocol tag ·
no lockbox access. v2 (`MEASUREMENT_ONLY`) / v3 (`DEV_STOP / NO_LOCKBOX_CONSUMED`) untouched; lockbox SEALED. The `.pt`/`.npz`/cache
binaries are NOT committed (recorded by path + hash).

## Next gate (NOT started — separate decision): explicit compatibility-replay authorization
A valid compatibility authorization manifest (`validate_compat_authorization` + `COMPAT_AUTH_FIELDS` + exact
`REQUIRED_COMPAT_STATEMENT`) BINDING (compatibility_protocol_commit=5237378 / substrate_protocol_commit=b99fa4f /
substrate_manifest_sha256=6c6ab457… / env_lock_sha256=523b44bc… / output_path / authorized_by / time / statement) unlocks the
old-seven-DEV-substrate compatibility replay under acar-v4-regen 3.13. Outcome ∈ {SUBSTRATE_COMPATIBILITY_PASS,
SUBSTRATE_COMPATIBILITY_FAIL, OPERATIONALLY_ABORTED_NO_VERDICT}. PASS → update ACAR_FROZEN_v4.md substrate hashes → tag
acar-v4-protocol → held-out preprocessing → one external Arm-B. FAIL → DEV-only / new dated protocol; the candidate / score / loss /
grid / comparator / thresholds must NOT be tuned after seeing the new-substrate replay.
