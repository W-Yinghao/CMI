# S2P_04 — Foundation Source-Code Provenance Audit (Phase 9A.5)

**Project S2P — Phase 9A.5.** Targeted implementation-provenance audit (PM-required before P1): confirm S2P runs the
models' **native** pretraining objective/masking/checkpoint/sampler, fed by our TUEG subject-subset loader via a thin
adapter — **not** a lookalike reimplementation. NOT a code review/rewrite. Two independent source audits (agents).
Detail in `results/s2p_source_audit/*.csv`; decision in `source_audit_go_nogo.json`.

## Verdict — both NEEDS-ADAPTER (thin), NO fundamental blocker
| model | native path identified | objective | masking | verdict |
|---|---|---|---|---|
| **CBraMod** (P1 primary) | `pretrain_main.py → Trainer_valid → CBraMod → generate_mask + MSE on x[mask==1]` | masked-patch **reconstruction** (MSE, target = normalized signal) | ratio 0.5, per-(B,C,patch) Bernoulli, zeros token | **NEEDS-ADAPTER (thin)** |
| **CodeBrain-Stage2** (P2) | `pretrain_EEGSSM.py → EEGSSM_Trainer → SSSM(if_codebook=True) + FROZEN tokenizer` | masked-**token** CE on **both** temporal+frequency (1:1) | ratio 0.5, per-(channel,patch); native mask **(B,19,30)** | **NEEDS-ADAPTER**; tokenizer weight on disk |

## The 9B-0 CodeBrain blocker — resolved
The `IndexError: mask [4,570]` was a **pre-flattened mask**. Native `generate_mask` returns **(B, ch_num, patch_num)
= (B, 19, 30)** (`util.py:31`) which indexes the leading 3 dims of `(B,19,30,200)`. The flatten to **(B,570)** happens
**inside the trainer AFTER the forward** (`Trainer.py:282`), only to index the token-id tensors — it is **never**
passed to the model. **Fix: pass the native 3-D (B,19,30) mask; do not reimplement it.** Verified channel-major
ordering is consistent between mask flatten and token view.

## Critical validity items for P1 (the reason this audit exists)
1. **CBraMod sampler (subject-scaling validity).** The native DataLoader samples **uniformly over sequences** with no
   per-subject weighting → high-hours subjects dominate batches. Our per-subject **hours cap** equalizes this — but P1
   **must verify equal sequences/subject per cell** (else the subject-count axis is diluted). `sampler_audit.csv`.
2. **CodeBrain input scaling (CRITICAL).** Native path does `x/100` assuming **µV-scale** EEG; our loader **z-scores**
   (std≈1) → `/100` gives std≈0.01, **out of the frozen tokenizer's distribution**. CodeBrain must be fed **µV-scale**
   windows (loader normalization mode per model: z-score for CBraMod, µV/raw for CodeBrain).
3. **CBraMod normalizer.** Native `EEGNormalizer` is **hardcoded 129-channel** (HBN) → neutralize it (we pre-z-score;
   do **not** also `/100`).
4. **Val split.** Native CBraMod val is **leaky** (task-based, val⊂train subjects); native CodeBrain has **no val**
   (best-by-training-loss). S2P uses a **subject-disjoint pretrain-val** for both (an intentional, better deviation);
   checkpoint select by **pretrain-val loss only, never target** (firewall).
5. **Preprocessing mismatch to DOCUMENT (not reproduce).** Processed TUEG is **0.5–45 Hz**; CodeBrain paper is
   **0.3–75 Hz + 50 Hz notch**. Spectral mismatch (esp. `token_f`). S2P uses the processed 0.5–45 Hz common pipeline
   for a controlled subject-scaling contrast; we do **not** claim to reproduce CodeBrain pretraining.
6. **Temporal-code collapse (prior FSR finding)** was on non-TUEG downstream EEG; on **TUEG** (the tokenizer's own
   pretraining corpus) `token_t` should stay diverse — but log per-stream CE + unique-token counts to confirm.

## Minimal adapters (no objective/mask reimplementation)
- **CBraMod:** emit bare fp32 `(B,19,30,200)`; neutralize the 129-ch `EEGNormalizer`; hand our subject-disjoint
  train/val loaders to `Trainer_valid`. Mask/MSE/patch-embed/checkpoint all native.
- **CodeBrain:** `SubjectSubsetDataset` wrapper returning `(19,30,200)` **µV-scale**; thin driver mirroring
  `pretrain_EEGSSM.main` (swap dataset, skip LMDB close, `parallel=False`, `CUDA_VISIBLE_DEVICES` remap for hardcoded
  device_ids/`.cuda()`, `batch_size≥2`); `--tokenizer_weight ~/eeg2025/NIPS/CodeBrain/Checkpoints/CodeBrain_Tokenizer.pth`
  (on disk); native `generate_mask` + `get_codebook_indices`. Add a subject-disjoint pretrain-val for checkpoint select.

## Decision
`approve_p1_after_audit`: **CBraMod = recommended True** (thin adapter, native trainer, no blocker). **CodeBrain =
P2** after building its native adapter + a **bounded native-path smoke** confirming the (B,19,30) mask fix runs
end-to-end. Crosswalk detail: S2P_05. P1 protocol to be design-red-teamed before any run.
