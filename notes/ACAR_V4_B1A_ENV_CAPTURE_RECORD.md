# ACAR v4 — B1a env-capture record **(COMPLETE — operational lock captured, repo-external + commit-consistent)**

```
DATE   : 2026-06-30 (machine UTC)
RESULT : acar-v4-regen built (R1-A/A2), runtime-validated, and OPERATIONAL env lock CAPTURED_AND_VERIFIED on A40 (node33)
         against the clean correction commit H=046507a, with interop/intra/omp pinned to 1 and torchvision/torchaudio/moabb
         in the lock hash. B1a COMPLETE. Still NO training / DEV-raw / held-out / source-state fit / compatibility replay / tag.
STATUS : B1A_COMPLETE — OPERATIONAL_LOCK_CAPTURED (external)
OPERATIONAL LOCK (repo-external; NOT committed):
  path                 = /home/infres/yinwang/acar_v4_regen_capture/acar_v4_regen_env_lock_046507a.json
  env_lock_sha256      = ceda567c376618739466254f4810e6cba2a76cab525cf2a0d43e82454cdd5b21   ← FILE BYTES; use THIS in the
                          manifest (run_regen_substrate._verify_env_lock uses _sha256_file)
  canonical content sha= 324e6020b8a6a5b2446cd29329c3d1b931248baabc9738b621c2a1668756660f   (hash_regen_env_lock)
  protocol_commit      = 046507ad5a03dc38910a78bac7c29ec0bf8d48c1 (H)
  device cuda / NVIDIA A40 / driver 610.43.02 / cuda 12.4 / cudnn 90100 ; torch 2.6.0+cu124 ; torchvision 0.21.0+cu124 ;
  torchaudio 2.6.0+cu124 ; braindecode 1.5.2 ; moabb 1.5.0 ; mne 1.12.1 ; skorch 1.4.0 ; numpy 2.4.4 ; scipy 1.18.0 ;
  sklearn 1.9.0 ; seed 0 ; deterministic=true ; intra=inter=omp=1 ; pipeline_config_sha256=canonical(38250f16…)
SUPERSEDED (diagnostic only): first A40 lock sha 589ceed… (commit 785d963) — interop=20, no import-critical versions,
         commit self-ref. Removed from the repo (was notes/ACAR_V4_REGEN_ENV_LOCK.json).
```

## Proven (still NO training / DEV-raw / held-out / source-state fit / tag)
- Isolated env `acar-v4-regen` built (eeg2025 untouched); exact pins: torch 2.6.0+cu124 / torchvision 0.21.0+cu124 /
  torchaudio 2.6.0+cu124 + braindecode 1.5.2 + moabb 1.5.0 (+ mne 1.12.1, skorch 1.4.0, numpy 2.4.4, scipy 1.18.0,
  sklearn 1.9.0). See `notes/ACAR_V4_REGEN_ENV_INSTALL_LOG.md`.
- Runtime imports PASS (CPU + A40): torch/torchaudio/torchvision; braindecode 1.5.2 + moabb 1.5.0; `EEGNetv4`;
  `cmi build_backbone`; `build_backbone("EEGNet",19,512)`. A40 (node30) acceptance: `cuda.is_available()==True`. ⇒ both
  eeg2025 blockers (torchaudio ABI + braindecode/moabb BNCI name) RESOLVED.

## Correction patch (this commit) — env-lock schema/capture hardened
- `regen_envlock`: lock now carries `torchvision_version`/`torchaudio_version`/`moabb_version`/`mne_version`/
  `skorch_version` (in the hash); a CAPTURED lock MUST have non-empty torchvision/torchaudio/moabb AND
  `torch_intraop==torch_interop==omp==1`.
- `capture_regen_envlock`: pins intra/inter/omp to 1 in the fresh capture process BEFORE any work, and records the new
  version fields. (Guards added; suites green.)
- The in-repo `notes/ACAR_V4_REGEN_ENV_LOCK.json` (the superseded 589ceed lock) is DELETED — the operational lock is
  repo-external (commit-consistency, per REGEN_COMMAND §4).

## Operational recapture — DONE (A40 node33, job 876735, against H=046507a; ~5 s; env introspection only)
Captured + validated (regen_envlock): all accept conditions hold (status CAPTURED_AND_VERIFIED, device cuda, intra=inter=omp=1,
torchvision/torchaudio/moabb non-empty, torch 2.6.0, braindecode 1.5.2, pipeline_config_sha256 canonical, protocol_commit=H).
The lock is repo-EXTERNAL (path + shas above); this record commit references it but does NOT commit the lock content.

## Next (review point — still NO training/tag/external)
```
1. build fixed PD/SCZ regen input manifests:
     protocol_commit = 046507ad5a03dc38910a78bac7c29ec0bf8d48c1 (H)
     env_lock_path   = /home/infres/yinwang/acar_v4_regen_capture/acar_v4_regen_env_lock_046507a.json
     env_lock_sha256 = ceda567c376618739466254f4810e6cba2a76cab525cf2a0d43e82454cdd5b21
     pipeline_config_sha256 = 38250f16e8a456076b69abcae2336101aabebde51e2f9ee697c8bd354ac2848d (canonical)
     + dev_cohorts (exact DEV scope), raw_bids source_paths, subject-list / diagnosis-label / per-cohort raw-file-list hashes
2. run the fail-closed preflight at a CLEAN checkout of H (detached HEAD == 046507a, so HEAD == manifest protocol_commit):
     python -m acar.v4.run_regen_substrate --disease PD|SCZ --dev-input-manifest <abs> --output <abs>
     -> expect SubstrateTrainingNotAuthorizedError (confirms the B1 gate is the ONLY remaining blocker)
3. record commit (references manifest path + sha; preflight result)
4. ask for B1b real all-DEV substrate training authorization
```
(A CPU lock via `--allow-cpu` was NOT taken — separate substrate/runtime decision.) Boundaries unchanged; lockbox SEALED;
v2/v3 untouched.
