# ACAR v4 — regen env diagnosis + repair proposal **(READ-ONLY; NO installs, NO env change, NO training)**

```
DATE   : 2026-06-29/30 (machine UTC)
NODE   : SLURM job 866446 (interactive; NO GPU). eeg2025 env: /home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
SCOPE  : read-only version/import/ldd inspection + GPU availability + a repair PROPOSAL. NOTHING installed/changed/trained.
VERDICT: B1a CAPTURE_FAILED root cause = TWO version-stack mismatches in eeg2025 (NOT ACAR code). DEV raw is available;
         GPU nodes exist via SLURM. Fix = an ISOLATED env (do not mutate eeg2025); B1 still NOT authorized.
```

## 1. Environment as probed (eeg2025)
```
python 3.13.7 ; platform Linux-6.12 glibc2.39
torch        2.6.0+cu124   (importable; torch.cuda.is_available() == False on this node; torch.version.cuda == 12.4)
torchvision  0.21.0+cu124  (importable; correct pair for torch 2.6.0)
torchaudio   2.8.0         (IMPORT FAILS — see §2.1)
braindecode  1.2.0         (IMPORT FAILS — see §2.2)
moabb        1.5.0   mne 1.11.0   skorch 1.2.0   sklearn 1.8.0   numpy 2.4.4   scipy 1.17.0
```

## 2. Root causes (two independent mismatches; both must be fixed)

### 2.1 torchaudio ABI mismatch (torchaudio 2.8.0 vs torch 2.6.0)
`import torchaudio` → `torchaudio/_extension/__init__.py` `_load_lib("libtorchaudio")` →
`OSError: .../libtorchaudio/lib/libtorchaudio.so: undefined symbol: _ZNK5torch8autograd4Node4nameB5cxx11Ev`.
- `importlib.metadata.requires("torchaudio") → torch==2.8.0`. **torchaudio 2.8.0 is built against torch 2.8.0**, but the
  installed torch is **2.6.0** → the C++ symbol it needs is absent → undefined-symbol at load.
- `requirements.txt` confirms a MIXED install: `torch @ .../libtorch_..._1739474892959/work` (conda-forge libtorch 2.6) +
  `torchaudio==2.8.0` (pypi). Classic conda-torch / pypi-torchaudio cross-release break.
- `ldd libtorchaudio.so` shows `libtorch.so / libtorch_cpu.so / libc10.so / libcudart.so.12 => not found` (resolved at
  runtime from the torch package), but the version skew is the real failure.
- Official rule: torchaudio and torch must be the SAME release. ⇒ **torchaudio must be 2.6.0** to match torch 2.6.0 /
  torchvision 0.21.0 (or upgrade the whole trio to 2.8.0 / 2.8.0 / 0.23.0).

### 2.2 braindecode 1.2.0 ↔ moabb 1.5.0 name mismatch
`import braindecode` → `braindecode/classifier.py` → `braindecode/eegneuralnet.py` →
`ImportError: cannot import name 'BNCI2014001' from 'moabb.datasets'`.
- moabb renamed `BNCI2014001` → `BNCI2014_001` (the modern name present in moabb 1.5.0); the deprecated `BNCI2014001` alias
  has been REMOVED by moabb 1.5.0. **braindecode 1.2.0's import chain still references the old `BNCI2014001`** → ImportError.
- `requires("braindecode") → moabb>=1.2.0 (extra), torch<3,>=2.0, torchaudio<3,>=2.0, skorch>=1.2.0, mne>=1.10` — the
  declared floor is satisfied, but the concrete `BNCI2014001` symbol braindecode imports is gone in moabb 1.5.0.
- ⇒ align the **(braindecode, moabb)** pair: either pin moabb to the last release that still exposes `BNCI2014001`, or move
  braindecode to a release whose import chain uses `BNCI2014_001` with moabb 1.5.0. (Cannot be settled here — no installs;
  the operator verifies in the isolated env.)

### 2.3 What ACAR actually needs (so the fix is scoped)
`cmi/models/backbones.py` imports only `from braindecode.models import EEGNetv4, ShallowFBCSPNet, Deep4Net, EEGConformer`
(+ `torch`). It does NOT import torchaudio/moabb directly — both are pulled transitively by braindecode's eager
`__init__ → classifier → eegneuralnet`. So making `import braindecode` succeed (matched torchaudio + aligned moabb) is
sufficient; EEGNet itself uses neither torchaudio nor moabb at runtime.

## 3. GPU availability (the other capture prerequisite)
- This interactive node: **no GPU** (`nvidia-smi` absent; `torch.cuda.is_available()==False`).
- SLURM HAS GPU partitions: `A100` (gpu:3/gpu:8), `V100` / `V100-32GB` / `V100-16GB` (gpu:2-3), `P100` (gpu:3). ⇒ a GPU
  training node is obtainable by submitting to one of these partitions; the env capture + (later, post-B1) training run there.

## 4. Original DEV runtime is NOT recoverable from local files
`~/eeg2025_packages.txt`, `~/eeg2025_sorted.txt`, `~/requirements.txt` all reflect the SAME currently-broken combo
(conda libtorch 2.6 + pypi torchaudio 2.8 + braindecode 1.2.0 + moabb 1.5.0 + torchvision 0.21.0). They do not encode a
known-good legacy pin, and the DEV `feat_dump_v4` substrate was almost certainly produced under an earlier/different state.
⇒ No faithful "legacy" reconstruction is available → a regenerated substrate is a **NEW V4 external representation substrate**
(consistent with ACAR_V4_SUBSTRATE_REGEN_PLAN.md §0), declared as such — not the original DEV runtime.

## 5. Repair proposal — ISOLATED env (do NOT mutate eeg2025). Two candidate specs; verify, then review.
```
Option R1-A (minimal, torch-2.6 family — recommended first attempt):
  python 3.11 (or keep 3.13)
  torch 2.6.0 + torchvision 0.21.0 + torchaudio 2.6.0     # all SAME release (fixes §2.1)
  braindecode + moabb as a MUTUALLY COMPATIBLE pair       # fixes §2.2 — candidates to test in the isolated env:
      (i)  braindecode 1.2.0 + moabb pinned to the last version exposing BNCI2014001 (verify which), OR
      (ii) braindecode upgraded to a release whose import chain uses BNCI2014_001 with moabb 1.5.0
  skorch (matching braindecode), mne 1.x, numpy/scipy/sklearn pinned
  ACCEPTANCE (all must pass, NO training):
      import torch; torch.cuda.is_available() == True       # on a GPU node
      from braindecode.models import EEGNetv4               # OK
      from cmi.models.backbones import build_backbone       # OK
      python -m acar.v4.capture_regen_envlock --output ... --protocol-commit <HEAD> --device-kind cuda  → CAPTURED_AND_VERIFIED

Option R1-B (newest coherent — torchaudio 2.8.0 already installed):
  torch 2.8.0 + torchvision 0.23.0 + torchaudio 2.8.0      # upgrade torch+vision to match the installed torchaudio
  braindecode (BNCI2014_001-era) + moabb 1.5.0
  same ACCEPTANCE checks.
```
Recommendation: **R1-A** (torch 2.6.0/torchvision 0.21.0 are already correct; only torchaudio is wrong + braindecode/moabb
misaligned — the smallest, most controllable change), built as a **separate `acar-v4-regen` env**; eeg2025 untouched. If a
clean (braindecode, moabb) pair on torch 2.6 proves hard, fall back to R1-B. Because the runtime differs from DEV regardless,
record whichever is built as the NEW substrate runtime in the env lock + ACAR_FROZEN_v4 (per plan §0).

## 6. Constraints / status (unchanged)
NO pip/conda install · NO module swap in any active env · NO retrain · NO DEV raw signal loaded · NO held-out/external read ·
NO compatibility replay · NO acar-v4-protocol tag. External Arm B = NOT_YET_EXECUTABLE; lockbox SEALED; v2/v3 untouched.
`run_regen_substrate` still rejects any non-CAPTURED_AND_VERIFIED env lock (correct defence).

## 7. Next steps (each separately reviewed)
```
1. (this) read-only diagnosis + repair proposal      ← DONE
2. review + approve an isolated env spec (R1-A or R1-B)
3. build the isolated acar-v4-regen env (installs — needs explicit approval; eeg2025 untouched)
4. on a GPU node: acceptance checks + capture CAPTURED_AND_VERIFIED env lock
5. build fixed PD/SCZ input manifests (raw_bids; subject/diagnosis/per-cohort-raw-file-list hashes)
6. fail-closed preflight (expect SubstrateTrainingNotAuthorizedError)
7. THEN ask B1b training authorization
```
