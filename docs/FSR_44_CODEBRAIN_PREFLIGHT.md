# FSR_44 — CodeBrain Preflight (Phase 8A)

**Project FSR — Phase 8 opening.** Preflight for the **Subject-Scaling & Foundation-Model FSR Audit**. PM
direction: turn FSR from a shortcut-verification / recoverability paper into a **scaling-mechanism** paper that
speaks to the EEG foundation-model literature. This document is the **8A preflight only** (load-test + feature-dump
entry points + dataset compatibility); the actual scaling pilot (8B/8C) is held for explicit PM go.

## Scientific question (Phase 8)
> When cross-subject generalization improves with more source subjects or EEG foundation pretraining, does subject
> leakage **decrease**, or does its **functional role** change — from a harmful shortcut to task-useful structure?

CodeBrain is the **primary audit substrate, NOT a SOTA baseline**. We do not claim it is best; we use its
**decoupled** representation to ask *where* subject information lives and how it couples to the task.

## Why CodeBrain (vs LaBraM/EEGPT/BIOT/CBraMod)
Its two-stage design exposes **per-representation** FSR entry points a black-box encoder cannot: a `TFDual`
tokenizer splitting each patch into **temporal** vs **frequency** codebook ids, and an `EEGSSM` (SGConv/S4)
encoder. So FSR can separately audit temporal-code / frequency-code / encoder-hidden / pooled subject leakage —
the deeper "is subject information *spectral* / *temporal* / *encoded*?" question. CBraMod (verified: real 19.8 MB
backbone in `eeg2025`) and LaBraM are the designated **controls**; EEGNet / FBCSP-LGG are the specialist baselines.

## Preflight findings

### 1. Assets — all LOCAL, no download, no exotic deps
- **Base pretrained weights on disk** (byte-match HuggingFace `YjMajy/CodeBrain`): `CodeBrain.pth` 60.9 MB
  (EEGSSM encoder) + `CodeBrain_Tokenizer.pth` 112.3 MB (TFDual), at
  `~/eeg2025/NIPS/CodeBrain/Checkpoints/` (and a second copy under `~/eeg2025/NIPS_finetune/CodeBrain/Checkpoints/`).
- **Code**: fresh upstream clone `~/CodeBrain` (github jingyingma01/CodeBrain, ICLR'26) + an existing lab
  integration with **frozen linear-probe** dirs (`~/eeg2025/NIPS_finetune/CodeBrain/…linearprobe…`) — the
  load/freeze/feature path is already exercised in-house.
- **No `mamba_ssm`/`causal_conv1d` dependency** — `EEGSSM` is **SGConv**-based (long-conv S4), so no CUDA-kernel
  build. Runs in env `eeg2025` (torch 2.6.0+cu124). Standard deps (einops/timm/scikit-learn).

### 2. Load-test (GPU, `codebrain_load_test.json`)
- **EEGSSM encoder: clean load** — 269 checkpoint tensors, **0 missing / 0 unexpected**, 15.17 M params; frozen
  forward OK, finite. `feats` shape `(B, C, n_patch, 200)`; pooled window embedding = `(B, 200)`.
- **TFDual tokenizer: clean load** — 325 checkpoint tensors (unwrapped from `{'model':…}`), **0 missing / 0
  unexpected**; temporal codes `token_t` and frequency codes `token_f`, shape `(B, C·n_patch)` = `(B, 570)` for
  19×30, ids in [0, 4096).
- **GPU required**: `Models/SSSM.py:129` hardcodes `.cuda()` for the sliding-window-attention mask → all CodeBrain
  inference (incl. frozen feature dumps) must run on GPU via SLURM (not CPU). Confirmed on an A40.

### 2b. REAL-EEG forward (SHU-MI, `real_forward_shu.json`) — and an honest token caveat
One SHU-MI motor-imagery session (8 trials, 32→19 ch, 250→200 Hz, 4 patches), fed frozen:
- **Encoder: clean GO on real EEG** — `feats (8,19,4,200)`, pooled `(8,200)`, **finite** and **non-degenerate**
  (mean pairwise-L2 across trials 3.9–5.5, i.e. embeddings genuinely vary per trial). This is the primary substrate.
- **Frequency tokens: diverse** (220–451 unique codes on real EEG) — usable diagnostic.
- **Temporal tokens: DEGENERATE — 1 unique code** on real EEG under **both** `/100` and per-channel z-score
  normalization (and on dummy input). This is **pretrained temporal-codebook collapse** (a known VQ failure mode),
  not an input-scale issue (the frequency path self-normalizes and stays diverse). **Disclosed, not hidden.**
- **Reframe (why this is not a blocker):** in CodeBrain the tokenizer codes are **pretraining targets** — *every*
  downstream model (`model_for_shu/faced/seedv/stress/…`) uses `self.backbone = SSSM(...)`, `feats = backbone(x)`,
  and **none** use `get_codebook_indices`. So the **audit substrate is the EEGSSM encoder** (clean GO); the
  decoupled *token* diagnostic is optional — its **frequency** half works, its **temporal** half is degenerate
  and must be investigated in 8B (window length ≠ 30-patch canonical, the lab's exact usage, or an alternative
  checkpoint) before the "temporal vs frequency token" angle is claimed.

### 3. Input spec (verified)
`x : (B, C, n_patch, 200)` raw EEG @ **200 Hz**, 1 s patches. Canonical pretraining layout **19 ch × 30 patch
(= 570 = `s4_lmax`)**; `input_chans = list(range(C+1))` (index 0 = cls slot, rest = C data channels). Encoder is
channel/length-flexible; the tokenizer's pos/channel embeddings are montage-tied (pos_embed 128 ch, time_embed 60
patch) → feed the canonical layout. MI (~4 s) windows tile/pad to fixed `n_patch`, `C*n_patch ≤ 570`.

### 4. Feature-dump entry points → FSR ladder
See `results/fsr_codebrain_preflight/feature_dump_schema.md`. F0 temporal codes / F1 frequency codes / F2 quantized
code vectors / F3 encoder per-(ch,patch) hidden / F4 pooled 200-d window embedding. FSR L1 (subject probe on each),
L4 (task-head alignment), L5 (subject-subspace reliance), L6 (target consequence) — all frozen, target labels for
final scoring only.

### 5. Dataset compatibility (`dataset_compatibility.csv`)
| dataset | task | subjects | scaling role |
|---|---|---|---|
| **PhysioNetMI** | MI | **109** | **PRIMARY subject-scaling (MI)** — the only ≥30-subject MI set on disk |
| SHU-MI | MI | 25 | MI anchor + CodeBrain-native (has `shu_dataset.py`/`model_for_shu.py`) |
| FACED | emotion | 123 | many-subject **non-MI** scaling (generality check) |
| BNCI2014_001 / 2015_001 | MI | 9 / 12 | FSR-alignment sanity (ties to our 7A–7C frozen results) |
All need resample→200 Hz + channel map to CodeBrain montage. M3CV (~106) excluded (biometric, not MI). ISRUC
(sleep) not on disk.

## Go/No-Go
```
encoder loads clean (0 missing/unexpected) . PASS (269 tensors, 15.17M params)
encoder frozen forward + pooled embedding .. PASS (GPU; real SHU-MI: finite, non-degenerate)
tokenizer loads clean (0 missing) .......... PASS (325 tensors, unwrapped from {'model':...})
  - frequency-token diagnostic ............. PASS (220-451 unique on real EEG)
  - temporal-token diagnostic .............. FAIL (codebook collapsed -> 1 unique; flag for 8B; NOT the substrate)
no download / no mamba dep ................. PASS (local weights; SGConv; env eeg2025; GPU-required)
>=1 MI dataset processable end-to-end ...... PASS (SHU-MI loaded+forwarded end-to-end)
>=30-subject MI dataset for scaling ........ PASS (PhysioNetMI 109)
==> Phase 8B feasibility (ENCODER-based) ... GO. The primary EEGSSM-encoder subject-scaling audit is feasible on
    local assets. The decoupled-TOKEN angle is PARTIAL (frequency yes, temporal degenerate -> 8B investigation).
```

## Mandatory caveats (carried into Phase 8B/8C; PM)
1. **CodeBrain's own scaling is data-hours / model-size, NOT subject-count.** We must run **subject-diversity vs
   sample-size** ourselves (fixed-trials vs growing-trials at each `N_source`); do not cite CodeBrain's scaling law
   as subject scaling.
2. **Start frozen encoder + linear/shallow head**, not full fine-tuning (fine-tuning can amplify subject identity;
   the "Identity Trap" literature). Full-FT is a later, separate arm.
3. **TUEV/TUAB overlap** with CodeBrain's TUH pretraining → not clean cross-dataset generalization evidence; use
   only as pretraining-family stress, never as a main result. (We avoid TUH entirely; our axis is MI subjects.)
4. Firewall unchanged: target labels for **final scoring only**.

## Held for PM go (report-then-go)
Phase 8B = frozen CodeBrain feature dump on one MI dataset (SHU-MI native + BNCI alignment) + the first FSR L1/L4/L5
audit; Phase 8C = the `N_source ∈ {2,4,8,16,all}` subject-scaling curve on PhysioNetMI with the fixed-vs-growing
trials control. **Not started** — awaiting go. Paper 1 (Prior-Decoupled TTA) independent, unaffected.
