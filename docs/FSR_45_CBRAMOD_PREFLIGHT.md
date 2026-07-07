# FSR_45 — CBraMod Preflight (Phase 8A backup / control)

**Project FSR — Phase 8A.** Preflight for **CBraMod** as the designated **control** and **backup substrate** for the
Subject-Scaling & Foundation-Model FSR Audit (PM: "做备选"). Mirrors the CodeBrain preflight (FSR_44). CBraMod is a
**single-stage** EEG foundation encoder (criss-cross channel–time transformer) — no tokenizer, so no VQ-codebook
collapse risk (the issue that degenerated CodeBrain's temporal tokens).

## Findings (`results/fsr_cbramod_preflight/cbramod_preflight.json`)
- **Local weights, no download**: `~/eeg2025/NIPS/Cbramod_pretrained_weights.pth` (19.8 MB, official CBraMod
  pretrained; 2nd copy in `~/eeg2025/ICML_2026/`). Code: `~/eeg2025/CBraMod/models/cbramod.py` (backbone) +
  `model_for_bciciv2a.py` (MI head). Env `eeg2025`.
- **Backbone loads clean**: 211 checkpoint tensors, **0 missing / 0 unexpected**, **4.88 M params**
  (`CBraMod(in_dim=200,d_model=200,seq_len=30,n_layer=12,nhead=8)`; `proj_out → Identity`).
- **Runs on CPU** — CBraMod's forward has **no hardcoded `.cuda()`** (unlike CodeBrain's SSSM), so frozen feature
  dumps can run **CPU-only** (cheaper; no GPU queue). GPU optional.
- **Real SHU-MI forward** (8 trials, 32→19 ch, 250→200 Hz, z-score, 4 patches): `feats (8,19,4,200)`, pooled
  `(8,200)`, **finite**, **non-degenerate** (mean pairwise-L2 = 0.51). End-to-end works.
- **Single-stage, no tokenizer** — the audit feature is the encoder pooled/hidden (same FSR entry points as
  CodeBrain's encoder), and there is **no VQ code diagnostic** and **no code-collapse failure mode**.

## Input spec
`x : (B, C, n_patch, 200)` raw EEG @ **200 Hz**, 1 s patches; criss-cross transformer, `seq_len ≤ 30`,
PatchEmbedding channel conv kernel `(19,7)`. Same input contract as CodeBrain's encoder → the two substrates share
one preprocessing pipeline for the Phase-8 audit.

## Feature entry points → FSR ladder
| id | tensor | shape | FSR use |
|----|--------|-------|---------|
| G3_hidden | `feats` (proj_out=Id) | (B, C, n_patch, 200) | L1/L4/L5 per-channel/per-time reliance |
| G4_pooled | mean over (C,n_patch) | (B, 200) | L1 subject probe, L4 alignment, L5 subspace intervention, L6 target consequence |

(No token-level F0/F1 — CBraMod is single-stage. This is precisely why it is a clean **control**: same
encoder-level FSR audit, without the decoupled-token confound.)

## Go/No-Go
```
backbone loads clean (0 missing/unexpected) . PASS (211 tensors, 4.88M params)
frozen real-EEG forward + pooled embedding .. PASS (SHU-MI: finite, non-degenerate)
runs on CPU ................................. PASS (no hardcoded .cuda(); GPU optional)
single-stage, no VQ-collapse risk ........... PASS
no download / local weights ................. PASS
==> role ................................... GO as CONTROL (encoder-level FSR audit alongside CodeBrain) and as
    BACKUP primary if CodeBrain's GPU/temporal-token issues become blocking. Same 200Hz/patch pipeline; cheaper.
```

## Positioning (Phase 8)
`Primary = CodeBrain` (encoder-based audit; frequency-token diagnostic bonus). `Control/backup = CBraMod` (this
doc). `Specialist baseline = EEGNet / FBCSP-LGG`. The CodeBrain-vs-CBraMod contrast is a **mechanism** comparison
(who leaks more subject info / relies more on it / has better target consequence), **not** a SOTA leaderboard. Held
for PM go with the 8B pilot; CBraMod adds negligible cost (CPU). PC2 still paused; Paper 1 unaffected.
