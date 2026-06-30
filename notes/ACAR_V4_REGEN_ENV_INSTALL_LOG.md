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

## GPU acceptance capture — PENDING (blocked by per-user GPU quota, not a failure)
GPU acceptance (`torch.cuda.is_available()==True` + `capture_regen_envlock --device-kind cuda → CAPTURED_AND_VERIFIED`)
requires a GPU node. Three sbatch submissions (V100 job 876435; A40 jobs 876445, 876458) all stayed PENDING; reason
resolved to **`QOSMaxGRESPerUser`** — my own large GPU sweep (`p30B_f*s*`, ~9 running V100 jobs + many pending) saturates my
per-user GPU GRES quota under QOS `normal`. Those are active jobs (not mine to cancel). Per the B1a stop rule, the capture
job was cancelled (clean stop) and CPU was NOT substituted. The env is READY; capture only needs a free GPU slot within quota.

## To complete the capture (when a GPU slot is free under quota)
```
sbatch (A40/V100, --gres=gpu:1, --account=c2s --qos=normal, short --time, --mem=8G) running, from the repo:
  PYTHONPATH=<repo> OMP_NUM_THREADS=1 .../acar-v4-regen/bin/python -m acar.v4.capture_regen_envlock \
      --output <abs>/ACAR_V4_REGEN_ENV_LOCK.json --protocol-commit <HEAD>
expect: status=CAPTURED_AND_VERIFIED, device_kind=cuda. (Sbatch template: scratch/envcap.sbatch.)
```
No binaries / wheels / checkpoints are committed. eeg2025 untouched; no training/DEV-raw/held-out/tag; lockbox SEALED.
