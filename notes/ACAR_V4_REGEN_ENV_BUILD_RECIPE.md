# ACAR v4 — isolated regen env build recipe (R1-A primary) **(DESIGN ONLY — DO NOT EXECUTE)**

```
STATUS : DESIGN ONLY / DO NOT EXECUTE
         NO INSTALLS AUTHORIZED · NO TRAINING · NO DEV RAW READ · NO HELD-OUT READ · NO TAG
DATE   : 2026-06-29/30 (machine UTC)
GOAL   : a reviewable recipe to build an ISOLATED `acar-v4-regen` env (eeg2025 UNTOUCHED) in which the EEGNet training stack
         imports cleanly + CUDA is available, so a CAPTURED_AND_VERIFIED regen env lock can later be produced. Building it
         needs a SEPARATE explicit install approval. This file installs nothing.
WHY R1-A: keep the already-locked torch 2.6.0 / torchvision 0.21.0 direction; only fix the mismatched torchaudio (2.8.0 →
         2.6.0) + align braindecode/moabb. R1-B (whole stack → 2.8.0) is FALLBACK only (§7), to avoid needless runtime drift.
ROOT CAUSE (from notes/ACAR_V4_REGEN_ENV_DIAGNOSIS.md): (A) torchaudio 2.8.0 Requires torch==2.8.0 but torch is 2.6.0 →
         libtorchaudio.so undefined symbol; (B) braindecode 1.2.0 imports moabb.datasets.BNCI2014001, removed by moabb 1.5.0.
```

## 1. Primary torch stack (PINNED; CUDA 12.4 wheels — matches torch 2.6.0)
```
python      : 3.13  (match eeg2025; fall back to 3.11 only if the braindecode/moabb solve forces it)
torch       : 2.6.0   (cu124)
torchvision : 0.21.0  (cu124)
torchaudio  : 2.6.0   (cu124)   ← the fix: same release as torch (was 2.8.0)
```
DRAFT install command (**DO NOT EXECUTE** — for review only; the official previous-versions page lists this exact trio):
```bash
# DO NOT RUN — review only
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
```
Directly fixes §A: torchaudio 2.6.0's binary matches torch 2.6.0 → no `undefined symbol`.

## 2. braindecode / moabb pair (TRIAL MATRIX — resolve by import probe at install time, do NOT guess-pin as final)
Precise constraint discovered (read-only, from the installed source):
- `braindecode/datasets/__init__.py` → `from .moabb import BNCI2014001, HGD, MOABBDataset`; `datasets/moabb.py` →
  `class BNCI2014001(...)` does `from moabb.datasets import BNCI2014001`. This import is reached eagerly via
  `braindecode/__init__ → classifier → eegneuralnet → …datasets`.
- moabb 1.5.0 exposes `BNCI2014_001` (present) but NOT `BNCI2014001` (removed; per MOABB docs the old alias was dropped at
  moabb 1.1). ⇒ the installed braindecode 1.2.0 is incompatible with moabb ≥ 1.1.
```
Pair A1 (keep braindecode 1.2.0):  braindecode 1.2.0 + moabb pinned to the LAST release exposing BNCI2014001 (moabb 1.0.x).
     RISK: moabb 1.0.x may pull older mne/numpy that conflict on py3.13 → may force python 3.11.
Pair A2 (align to modern moabb):   a braindecode whose datasets import path uses BNCI2014_001, paired with moabb 1.5.0
     (the modern name). Identify by import probe (do NOT assume a version number).
DECIDE BY PROBE — accept the FIRST pair for which ALL of these import (no training, no data):
     import braindecode
     from braindecode.models import EEGNetv4
     from cmi.models.backbones import build_backbone
If neither A1 nor A2 yields all three imports → STOP (do not train); escalate to R1-B (§7) or re-review.
```
(skorch + mne + numpy/scipy/sklearn follow whatever the chosen braindecode/moabb pair requires; pin them in the install log.)

## 3. Acceptance tests (ALL must pass on the intended GPU node BEFORE any env-lock capture; NO training)
```bash
# (a) torch + CUDA
python - <<'PY'
import torch
assert torch.__version__.startswith("2.6.0"), torch.__version__
assert torch.cuda.is_available(), "CUDA not available on this node"
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
PY

# (b) matched audio/vision
python - <<'PY'
import torchaudio, torchvision
assert torchaudio.__version__.startswith("2.6.0"), torchaudio.__version__
assert torchvision.__version__.startswith("0.21.0"), torchvision.__version__
print("torchaudio", torchaudio.__version__, "torchvision", torchvision.__version__)
PY

# (c) the actual model + cmi backbone import (the thing that fails today)
PYTHONPATH=/home/infres/yinwang/CMI_AAAI_acar python - <<'PY'
from braindecode.models import EEGNetv4; print("EEGNetv4 OK", EEGNetv4)
from cmi.models.backbones import build_backbone; print("build_backbone OK", build_backbone)
PY

# (d) capture the regen env lock (env introspection only; NO training/data)
PYTHONPATH=/home/infres/yinwang/CMI_AAAI_acar python -m acar.v4.capture_regen_envlock \
    --output /abs/path/ACAR_V4_REGEN_ENV_LOCK.json --protocol-commit <current_commit> --device-kind cuda
```
Required capture result:
```
status = CAPTURED_AND_VERIFIED · device_kind = cuda · seed = 0 · torch_deterministic_algorithms = true
torch 2.6.0 / torchvision 0.21.0 / torchaudio 2.6.0 · braindecode+moabb import path works · pipeline_config_sha256 == canonical
```
Do NOT accept `device_kind=cpu` (capture --allow-cpu) without a SEPARATE explicit decision — GPU partitions are available and
the DEV substrate likely used CUDA; CPU would be another substrate choice.

## 4. Build method (isolated; eeg2025 UNTOUCHED) — for the LATER, separately-approved install step
```
env name      : acar-v4-regen           (conda create -n acar-v4-regen python=3.13 ; or a venv — DO NOT RUN now)
install log   : notes/ACAR_V4_REGEN_ENV_INSTALL_LOG.md   (every command + output captured)
lock output   : notes/ACAR_V4_REGEN_ENV_LOCK.json        (overwrites the CAPTURE_FAILED lock once CAPTURED_AND_VERIFIED)
capture into the install log: `pip freeze` / `conda list`, python exe path, LD_LIBRARY_PATH, CUDA_VISIBLE_DEVICES,
                              SLURM node + GPU type (nvidia-smi -L), and the acceptance-test outputs.
```
eeg2025 is never modified. No package in any existing env is changed.

## 5. Stop rules (explicit)
```
- R1-A cannot import EEGNetv4 AND build_backbone (after trying Pair A1, A2)        → STOP; do not train; review R1-B (§7).
- imports OK but CUDA unavailable on the intended node                              → STOP; do not train; schedule a GPU node
                                                                                       (A100/V100/P100) or review allow_cpu.
- CAPTURED_AND_VERIFIED lock cannot be produced                                     → STOP; do not train.
- success (all §3 pass)                                                             → commit the captured env lock + install
                                                                                       log; THEN ask for B1b training authorization.
```

## 6. NOT authorized now
```
no conda create · no pip install · no environment mutation · no GPU job (except a future APPROVED capture) · no EEGNet
training · no DEV raw read · no held-out read · no source-state fitting · no compatibility replay · no acar-v4-protocol tag.
```

## 7. R1-B fallback (only if R1-A's import matrix cannot be satisfied)
```
python 3.13 ; torch 2.8.0 + torchvision 0.23.0 + torchaudio 2.8.0 (cu124, same release) ; braindecode that imports cleanly
with moabb 1.5.0 (BNCI2014_001 path). DRAFT (DO NOT RUN):
  pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu124
Same §3 acceptance tests but assert torch 2.8.0 / torchvision 0.23.0 / torchaudio 2.8.0. R1-B changes the training runtime
more than R1-A; either way the result is a NEW V4 substrate runtime (plan §0), recorded in the env lock + ACAR_FROZEN_v4.
```

## 8. Next (each separately reviewed)
```
1. (this) draft recipe                         ← DONE (nothing installed)
2. you approve building acar-v4-regen (R1-A)   → installs become authorized (eeg2025 untouched)
3. build env + acceptance tests on a GPU node  → capture CAPTURED_AND_VERIFIED lock + install log
4. build fixed PD/SCZ input manifests          → fail-closed preflight (expect SubstrateTrainingNotAuthorizedError)
5. B1b training authorization (separate sign-off)
```
