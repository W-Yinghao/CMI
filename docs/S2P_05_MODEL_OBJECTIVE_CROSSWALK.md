# S2P_05 — Model Objective / Preprocessing Crosswalk (Phase 9A.5)

Consolidated native-vs-S2P crosswalk (machine tables in `results/s2p_source_audit/*.csv`). Reference for the P1
implementation + write-up; ensures S2P runs native pretraining semantics.

## Objective + masking
| | CBraMod (P1 primary) | CodeBrain-Stage2 (P2) |
|---|---|---|
| pretext | masked-patch **reconstruction** | masked-**token** prediction |
| loss | `MSE(x[mask==1], recon[mask==1])` on normalized signal | `CE(y_t,codes_t) + CE(y_f,codes_f)` (both streams, 1:1) |
| mask ratio / unit | 0.5 / per-(B,C,patch) Bernoulli | 0.5 / per-(channel,patch) |
| mask tensor | applied in signal domain (zeros token) | **native `(B,19,30)` long** (flatten to (B,570) is trainer-internal, post-forward) |
| tokenizer | — | FROZEN released `CodeBrain_Tokenizer.pth` (on disk); `get_codebook_indices` under no_grad |
| input | (B,19,30,200) @200Hz | (B,19,30,200) @200Hz, `input_chans=range(20)` (idx0=cls) |
| **normalization** | corpus/per-window **z-score** (our loader) | **µV-scale `/100`** — feed **µV, not z-score** (tokenizer OOD else) |
| checkpoint | best-by-pretrain-val-loss (+optimizer/resume) | best-by-training-loss → **S2P adds subject-disjoint pretrain-val** |
| device | fp32, seed 42, cudnn.deterministic | + hardcoded device_ids/`.cuda()` → `parallel=False` + `CUDA_VISIBLE_DEVICES` remap; **batch_size≥2** |

## Preprocessing (processed TUEG 4704743c vs published)
| axis | S2P (processed TUEG) | CodeBrain paper | disposition |
|---|---|---|---|
| sfreq / patch / segment | 200 Hz / 1 s (200) / 30 s (30 patches) | same | **MATCH** |
| bandpass | 0.5–45 Hz | 0.3–75 Hz + 50 Hz notch | **DOCUMENT** mismatch (esp `token_f`); not claimed as reproduction |
| channels | 19 common 10-20 (`-LE`) | 19 common 10-20 | MATCH (montage-pinned, hashed) |
| scaling for CodeBrain | µV-scale (loader must emit µV for CodeBrain) | µV `/100` | loader normalization **mode per model** |

## S2P intentional deviations (all improvements or documented limits)
- **Subject-disjoint pretrain-val** for both (native CBraMod val is leaky; native CodeBrain has none) → cleaner
  checkpoint selection (pretrain-val loss only, never target).
- **Subject-balanced exposure** via the per-subject hours cap (native samplers are uniform-over-sequences) →
  verify equal sequences/subject per cell.
- **0.5–45 Hz** processed band (not the paper's 0.3–75 Hz+notch) → a controlled-substrate choice, disclosed.
- **From-scratch** encoders per subject subset (not released weights) — the whole point (representation must be
  trainable to test diversity, per Phase-8 close-out).
- **Per-stream logging** for CodeBrain (token_t/token_f CE + unique counts) to catch temporal-code degeneracy.

**None of these change the native pretext**; they harden the firewall and the subject-scaling contrast. Forbidden:
claiming reproduction of CodeBrain/CBraMod published pretraining; SOTA/full-FT framing.
