# ACAR v4 — B1a env-capture record **(COMPLETE — CAPTURED_AND_VERIFIED)**

```
DATE   : 2026-06-29/30 (machine UTC)
RESULT : acar-v4-regen built (R1-A/A2), RUNTIME-IMPORT-VALIDATED, and GPU env lock CAPTURED_AND_VERIFIED on A40 (node30).
         B1a COMPLETE. Still NO training / DEV-raw / held-out / source-state fit / compatibility replay / tag / external.
STATUS : B1A_COMPLETE — ENV_CAPTURED_AND_VERIFIED
LOCK   : notes/ACAR_V4_REGEN_ENV_LOCK.json · regen_env_lock_sha256 = 589ceedcc4d22b62043674de81902097cd31d8cf14da014a46e8b35863bdb90e
         (cuda · NVIDIA A40 · driver 610.43.02 · cuda 12.4 · cudnn 90100 · torch 2.6.0+cu124 · braindecode 1.5.2 · seed 0 ·
          deterministic=true · pipeline_config_sha256=canonical · protocol_commit=785d963…)
```

## What was authorized + done
- Built isolated `acar-v4-regen` (eeg2025 untouched); exact pins: torch 2.6.0+cu124 / torchvision 0.21.0+cu124 /
  torchaudio 2.6.0+cu124 + braindecode 1.5.2 + moabb 1.5.0 (+ mne 1.12.1, skorch 1.4.0, numpy 2.4.4, scipy 1.18.0,
  sklearn 1.9.0). Details: `notes/ACAR_V4_REGEN_ENV_INSTALL_LOG.md`.
- Runtime import probe (CPU) PASSED: torch/torchaudio/torchvision import (ABI fixed); braindecode 1.5.2 + moabb 1.5.0
  import (BNCI2014_001 path); `from braindecode.models import EEGNetv4` OK; `cmi.models.backbones build_backbone` OK;
  `build_backbone("EEGNet",19,512)` constructs. ⇒ both eeg2025 import blockers resolved.

## GPU capture — DONE (after the quota freed)
- Earlier: 3 sbatch jobs (V100 876435; A40 876445, 876458) PENDING = **QOSMaxGRESPerUser** (my own p30B sweep held the
  per-user GPU quota) → clean stop, no CPU substitution.
- Once the sweep drained, capture was resubmitted. A first GPU run (876692) passed the asserts but the capture tool left
  `driver_version=""` (validator rejects it for a CUDA lock) → fixed in commit 785d963 (`_nvidia_driver_version()` via
  read-only nvidia-smi). Clean capture = job **876698**, A40 **node30** (~15 s; env introspection only, NO training/data) →
  `status=CAPTURED_AND_VERIFIED`. Lock copied to `notes/ACAR_V4_REGEN_ENV_LOCK.json` (validates via regen_envlock).
- NOTE for B1b: the lock records `torch_interop_threads=20` (node default at capture); the B1b training run MUST pin
  interop=1 (deterministic) at run time.

## Next (review point — still NO training/tag/external)
```
1. build fixed PD/SCZ regen input manifests (raw_bids; subject-list / diagnosis-label / per-cohort raw-file-list +
   pipeline_config + env_lock hashes; env_lock_sha256 = 589ceed…; protocol_commit = current clean HEAD)
2. run the fail-closed preflight: python -m acar.v4.run_regen_substrate --disease PD|SCZ ...
   (expect SubstrateTrainingNotAuthorizedError — confirms B1 gate is the ONLY remaining blocker)
3. record commit
4. ask for B1b real all-DEV substrate training authorization
```
(A CPU lock via capture --allow-cpu was NOT taken — it would be a separate substrate/runtime decision.)

Boundary unchanged: no training/GPU-training/DEV-raw/held-out/source-state-fit/compatibility-replay/tag/external. eeg2025
untouched; lockbox SEALED; v2/v3 frozen results+tags untouched.
