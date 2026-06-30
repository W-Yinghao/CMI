# ACAR v4 — isolated regen env install log (acar-v4-regen; R1-A / Pair A2)

```
STATUS : ENV BUILT + RUNTIME-IMPORT-VALIDATED (CPU). GPU capture (CAPTURED_AND_VERIFIED lock) PENDING — blocked by the
         user's own GPU quota (QOSMaxGRESPerUser), NOT a build/version failure. eeg2025 UNTOUCHED. No training/DEV-raw/
         held-out/tag.
DATE   : 2026-06-29/30 (machine UTC)
NODE   : built on nodecpu05 (CPU; SLURM job 866446, account c2s, qos normal). Network/PyPI + pytorch cu124 index reachable.
ENV    : conda env `acar-v4-regen`, python 3.13.14 ; /home/infres/yinwang/anaconda3/envs/acar-v4-regen/bin/python
```

## Install commands (exactly as run; eeg2025 untouched; no --upgrade)
```
conda create -y -n acar-v4-regen python=3.13
.../acar-v4-regen/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
.../acar-v4-regen/bin/python -m pip install braindecode==1.5.2 moabb==1.5.0
```

## Resolved key pins (pip freeze — ALL required pins match R1-A/A2 exactly; no drift to 2.8 / CPU)
```
torch==2.6.0+cu124        torchvision==0.21.0+cu124      torchaudio==2.6.0+cu124
braindecode==1.5.2        moabb==1.5.0
mne==1.12.1   skorch==1.4.0   numpy==2.4.4   scipy==1.18.0   scikit-learn==1.9.0
```
(Full freeze = 105 packages; captured in scratch `regen_pip_freeze.txt`. canonical FROZEN_PIPELINE sha =
38250f16e8a456076b69abcae2336101aabebde51e2f9ee697c8bd354ac2848d.)

## Runtime import probe — PASS (on nodecpu05, CPU; the version-fix validation)
```
torch 2.6.0+cu124 | torchaudio 2.6.0+cu124 | torchvision 0.21.0+cu124   (torchaudio ABI fixed — imports cleanly)
braindecode 1.5.2 | moabb 1.5.0                                          (BNCI2014_001 path — imports cleanly)
from braindecode.models import EEGNetv4   -> OK  (FutureWarning: EEGNetv4 alias deprecated, removed in braindecode v1.14; present in 1.5.2)
from cmi.models.backbones import build_backbone -> OK
build_backbone("EEGNet", n_chans=19, n_times=512, n_classes=2, device="cpu") -> HookedBackbone OK  (no training, no data)
```
⇒ Both eeg2025 import blockers (torchaudio ABI + braindecode/moabb BNCI name) are RESOLVED in acar-v4-regen.

## GPU acceptance capture — CAPTURED_AND_VERIFIED (A40, after the GPU sweep freed quota)
History: 3 earlier submissions (V100 876435; A40 876445/876458) stayed PENDING = `QOSMaxGRESPerUser` (my own `p30B_f*s*` GPU
sweep held the per-user GPU quota) → clean stop, no CPU substitution. Once the sweep drained, the capture was resubmitted.
A first GPU run (job 876692) succeeded on the asserts but exposed a bug in the capture tool (it left `driver_version=""`,
which the validator rejects for a CUDA lock) — fixed in commit 785d963 (`_nvidia_driver_version()` via read-only nvidia-smi).
The clean capture (job **876698**, A40 **node30**, ~15 s; env introspection only, NO training/data):
```
== acceptance asserts == torch 2.6.0+cu124 | cuda 12.4 | avail True | dev NVIDIA A40 ; braindecode 1.5.2 + moabb 1.5.0 +
   EEGNetv4 + cmi build_backbone import OK ; PROBE_EXIT=0
== capture == status=CAPTURED_AND_VERIFIED  CAPTURE_EXIT=0
```
That first lock (regen_env_lock_sha256 `589ceed…`, protocol_commit `785d963`, A40, driver 610.43.02, cuda 12.4) PROVED the
env works on CUDA, but is **SUPERSEDED — a useful diagnostic, NOT the operational lock** — because it had
`torch_interop_threads=20` (training must pin interop=1), no import-critical version fields (torchvision/torchaudio/moabb),
and a commit self-reference. The env-lock schema + capture tool were corrected (interop/intra/omp pinned to 1 +
torchvision/torchaudio/moabb in the lock hash; see notes/ACAR_V4_B1A_ENV_CAPTURE_RECORD.md), the in-repo lock JSON was
removed, and the OPERATIONAL lock is recaptured EXTERNALLY against the clean correction commit (repo-external path +
env_lock_sha256, per REGEN_COMMAND §4). No binaries / wheels / checkpoints committed. eeg2025 untouched; no
training/DEV-raw/held-out/tag; lockbox SEALED.
