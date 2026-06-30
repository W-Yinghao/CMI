# ACAR v4 — regen env metadata/dry-run probe **(READ-ONLY; no installs, no env change, no training)**

```
STATUS : READ-ONLY metadata + dry-run resolution + wheel-source inspection. NOTHING installed/changed/trained; no env
         created; no DEV/held-out read; no tag. Reports written to scratch, NOT into any site-packages.
DATE   : 2026-06-29/30 (machine UTC)
NODE   : SLURM 866446 (no GPU). pip 25.2 via eeg2025 python 3.13.7. Network/PyPI + pytorch cu124 index reachable.
RESULT : R1-A is VIABLE on py3.13. Concrete first install attempt RESOLVED by dry-run:
         torch 2.6.0 + torchvision 0.21.0 + torchaudio 2.6.0 (cu124) + braindecode 1.5.2 + moabb 1.5.0.
         Pair A1 (braindecode 1.2.0 + moabb<1.1) is NOT viable on py3.13 (moabb 0.4.6 sdist build failure).
```

## Commands (read-only; `--dry-run --ignore-installed --no-cache-dir --report`, reports to scratch; `pip download --no-deps` for source inspection)
```
pip index versions braindecode | moabb | torchaudio
pip install --dry-run ... --report r1a_torch.json  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url .../cu124
pip install --dry-run ... --report a1.json         braindecode==1.2.0 "moabb<1.1"
pip install --dry-run ... --report a2.json         braindecode "moabb==1.5.0"
pip install --dry-run ... --report full_a2.json    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 braindecode==1.5.2 moabb==1.5.0 --extra-index-url .../cu124
pip download --no-deps braindecode==1.5.2 / 1.4.0 / 1.3.2   (wheel inspected; NOT installed)
```

## Available versions (PyPI)
```
braindecode : 1.5.2 (latest) … 1.4.0, 1.3.2, 1.3.1, 1.3.0, 1.2.0 (installed/broken), 1.1.0, 1.0.0, 0.8.1, …
moabb       : 1.5.0 (latest), 1.4.3, 1.4.2, 1.4.0, then GAP → 0.4.6, … 0.3.0   (NO 1.0/1.1/1.2/1.3 on PyPI)
torchaudio  : … 2.8.0, 2.7.1, 2.7.0, 2.6.0   (2.6.0 available → matches torch 2.6.0)
```

## R1-A torch trio — RESOLVES (py3.13, cu124)
`r1a_torch.json`: `torch-2.6.0+cu124 torchvision-0.21.0+cu124 torchaudio-2.6.0+cu124` (+ numpy 2.4.4, cu124 nvidia libs).
Confirms the §A fix (torchaudio 2.6.0 matches torch 2.6.0).

## Pair A1 (braindecode 1.2.0 + moabb<1.1) — RESOLUTION FAILED on py3.13
`moabb<1.1` resolves to **moabb 0.4.6** (the PyPI gap leaves nothing between 0.4.6 and 1.4.0). Building moabb 0.4.6's sdist on
python 3.13 fails: `AttributeError: 'build_ext' object has no attribute 'cython_sources'` (old Cython/setuptools vs py3.13).
⇒ **Pair A1 is not viable on py3.13.** (It would require py3.11 + an old toolchain — high-risk, and unnecessary given A2.)

## Pair A2 (modern braindecode + moabb 1.5.0) — RESOLVES, and wheel inspection clears both risks
- `a2.json` (unpinned torch): resolver picks **braindecode 1.5.2 + moabb 1.5.0** (with torch 2.12.1 only because torch was
  unpinned — braindecode has no upper torch bound).
- Wheel source inspection (`pip download --no-deps`, read-only) of braindecode 1.3.2 / 1.4.0 / 1.5.2:
  - all **export `EEGNetv4`** in `braindecode/models/__init__.py` (the deprecation concern does NOT bite ≤1.5.2 — cmi's
    `from braindecode.models import EEGNetv4` will resolve);
  - all import **`from moabb.datasets import BNCI2014_001`** (the MODERN name) in `datasets/moabb.py` → **compatible with
    moabb 1.5.0** (this is exactly what braindecode 1.2.0 got wrong with its old `BNCI2014001` import);
  - `Requires-Dist`: `torch>=2.2` (1.3.2) / `>=2.0` (1.4.0/1.5.2), `torchaudio>=2.0`, `moabb>=1.4.3` (extra) — **all
    satisfied** by torch 2.6.0 / torchaudio 2.6.0 / moabb 1.5.0.
- **FULL pinned dry-run** (`full_a2.json`, torch trio cu124 + braindecode==1.5.2 + moabb==1.5.0) RESOLVES on py3.13 — the
  trio is held (torchaudio stays 2.6.0):
```
torch 2.6.0+cu124 · torchvision 0.21.0+cu124 · torchaudio 2.6.0+cu124
braindecode 1.5.2 · moabb 1.5.0 · mne 1.12.1 · skorch 1.4.0 · numpy 2.5.0 · scipy 1.18.0 · scikit-learn 1.9.0
```

## Decision
```
Python 3.13   : FEASIBLE for A2 (full set resolves). py3.11 fallback NOT needed.
Recommended FIRST install attempt (when separately approved; eeg2025 untouched):
   conda create -n acar-v4-regen python=3.13
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
   pip install braindecode==1.5.2 moabb==1.5.0           # pulls mne 1.12.1 / skorch 1.4.0 / numpy 2.5.0 / scipy 1.18.0 / sklearn 1.9.0
   (NB: braindecode 1.5.2 has no upper torch bound, so install the cu124 trio FIRST, then braindecode/moabb, to keep torch 2.6.0.)
```
This supersedes the build-recipe's Pair-A1-first ordering: **A2 (braindecode 1.5.2 + moabb 1.5.0) is the recommended first
trial; A1 is dropped (not viable on py3.13).**

## Residual unknowns — metadata CANNOT prove these; verify by import probe POST-install on a GPU node (the recipe §3 acceptance tests)
```
- runtime: `import braindecode` + `from braindecode.models import EEGNetv4` + `from cmi.models.backbones import build_backbone`
  all SUCCEED (dependency resolution ≠ runtime import success).
- CUDA available on the chosen node (torch.cuda.is_available() == True).
- capture_regen_envlock → CAPTURED_AND_VERIFIED.
```
If, after a (separately-approved) install, any acceptance probe fails → STOP (do not train); fall back to braindecode 1.4.0
or the R1-B 2.8.0 trio, re-probe.

## Boundary (unchanged)
No pip/conda install · no env create/mutate · no training · no DEV raw read · no held-out read · no source-state fit · no
compatibility replay · no acar-v4-protocol tag. External Arm B = NOT_YET_EXECUTABLE; lockbox SEALED; v2/v3 untouched.
`ACAR_FROZEN_v4.md` status unchanged (NOT_YET_EXECUTABLE); no CAPTURED_AND_VERIFIED lock produced; no env created.
```
METADATA RESULT : R1-A VIABLE (py3.13) — recommended first install = torch2.6.0/vision0.21.0/audio2.6.0 + braindecode1.5.2 + moabb1.5.0
```
