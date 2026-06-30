# ACAR v4 — B1a env-capture record **(ENV READY + IMPORT-VALIDATED; GPU capture PENDING on quota)**

```
DATE   : 2026-06-29/30 (machine UTC)
RESULT : acar-v4-regen built (R1-A/A2) and RUNTIME-IMPORT-VALIDATED on CPU — the version fix works. The
         CAPTURED_AND_VERIFIED env lock is NOT yet produced: GPU acceptance is blocked by QOSMaxGRESPerUser (my own
         p30B GPU sweep saturates the per-user GPU quota). Per the B1a stop rule, CPU was NOT substituted.
STATUS : ENV_BUILT_IMPORTS_PASS / GPU_CAPTURE_PENDING_QOSMaxGRESPerUser
```

## What was authorized + done
- Built isolated `acar-v4-regen` (eeg2025 untouched); exact pins: torch 2.6.0+cu124 / torchvision 0.21.0+cu124 /
  torchaudio 2.6.0+cu124 + braindecode 1.5.2 + moabb 1.5.0 (+ mne 1.12.1, skorch 1.4.0, numpy 2.4.4, scipy 1.18.0,
  sklearn 1.9.0). Details: `notes/ACAR_V4_REGEN_ENV_INSTALL_LOG.md`.
- Runtime import probe (CPU) PASSED: torch/torchaudio/torchvision import (ABI fixed); braindecode 1.5.2 + moabb 1.5.0
  import (BNCI2014_001 path); `from braindecode.models import EEGNetv4` OK; `cmi.models.backbones build_backbone` OK;
  `build_backbone("EEGNet",19,512)` constructs. ⇒ both eeg2025 import blockers resolved.

## What is NOT done (and why)
- `torch.cuda.is_available()==True` + `capture_regen_envlock → CAPTURED_AND_VERIFIED` need a GPU node. 3 sbatch jobs
  (V100 876435; A40 876445, 876458) stayed PENDING = **QOSMaxGRESPerUser** (my p30B sweep holds the per-user GPU quota).
  Capture job cancelled (clean stop). **CPU NOT substituted** (stop rule: GPU unavailable → stop, no CPU without a separate
  decision). No CAPTURED lock written; `notes/ACAR_V4_REGEN_ENV_LOCK.json` remains the prior eeg2025 CAPTURE_FAILED record.

## Next (one of; user decision — still NO training/tag/external)
```
(a) wait for a GPU slot under quota (my p30B sweep drains, or I pause/free one), then resubmit the capture sbatch
    -> CAPTURED_AND_VERIFIED lock -> commit lock + this record updated;
(b) raise the QOS GPU cap / use a GPU-capable allocation;
(c) (separate explicit decision only) accept a CPU-captured lock via capture --allow-cpu — NOT taken here.
```
After a CAPTURED_AND_VERIFIED lock exists: build fixed PD/SCZ input manifests → fail-closed preflight
(SubstrateTrainingNotAuthorizedError) → THEN ask for B1b training authorization.

Boundary unchanged: no training/GPU-training/DEV-raw/held-out/source-state-fit/compatibility-replay/tag/external. eeg2025
untouched; lockbox SEALED; v2/v3 frozen results+tags untouched.
