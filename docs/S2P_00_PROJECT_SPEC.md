# S2P_00 — Project Spec: Subject-Scaling Pretraining for EEG Foundation Representations

**Project S2P** (new; distinct from FSR/Paper-2 and from Paper-1 Prior-Decoupled TTA). Branch
`project/s2p-subject-scaling`, worktree `CMI_AAAI_s2p`. PM-approved after Phase-8 closed: frozen released
checkpoints + light adapters **cannot** answer the subject-scaling question (8D-0 metric-power gate failed — the
~0.95 subject fingerprint is immovable under a fixed-encoder design). The only clean way is to **control
subject diversity during pretraining**.

## Thesis
> EEG foundation pretraining may improve cross-subject transfer **not by removing** subject information, but by
> **changing its functional role**. To test this, subject diversity must be controlled **during pretraining**, not
> only during downstream probing.

## Primary question
> When pretraining subjects scale, does cross-subject generalization improve by **suppressing subject separability**,
> or by **reorganizing subject information so it is no longer a harmful task lever**?

## Sub-questions
- **Q1 — diversity vs hours:** does *subject diversity* improve downstream cross-subject transfer, separated from a
  pure *pretraining-hours* effect? (fixed-hours vs growing-hours; CodeBrain's own scaling laws are hours/model-size,
  **not** subject-count — this is the gap.)
- **Q2 — separability:** does subject diversity reduce **downstream** pairwise subject separability (now testable —
  the encoder is trained per pretraining subset)?
- **Q3 — reliance:** if separability does not fall, does **functional reliance** change (L4 alignment, L5
  subject-subspace vs variance-null, L6 target consequence)? The strongest FSR result = transfer improves while
  separability stays high but L5 stays null/decreases ⇒ *role change, not erasure*.
- **Q4 — architecture:** is the effect consistent across a masked-reconstruction encoder (CBraMod) and a token-guided
  EEGSSM encoder (CodeBrain Stage-2)?

## Models (primary/secondary)
- **CBraMod (PRIMARY)** — deterministic, CPU-friendly forward, stronger frozen-task sanity, no tokenizer collapse,
  smaller; the lab already pretrained it (HBN). Masked-reconstruction pretraining.
- **CodeBrain Stage-2 EEGSSM (SECONDARY/CONTROL)** — with the **fixed released TFDual tokenizer** (Stage 1 frozen;
  avoids the temporal-code-collapse variable found in FSR). Full CodeBrain Stage-1+2 is deferred (M3, only on signal).

## Scope (PM)
- **Approved now: Phase 9A only** — corpus inventory + parallelization infrastructure + go/no-go.
- **Phase 9B-0** (tiny CBraMod smoke) only after 9A passes. **Full multi-model pretraining not approved.**
- Infrastructure must support **multi-model parallelism from day one** (job-matrix, not hand-runs).
- **Forbidden (permanent):** "subject diversity removes subject leakage"; "foundation models become subject-
  invariant"; "large TUEG pretraining solves cross-subject generalization"; SOTA/leaderboard. **Allowed:** "under
  controlled pretraining subject-count subsets, we estimate whether subject diversity changes downstream subject
  separability, task-head coupling, and target transfer."

## Paper disposition
Paper 1 (Prior-Decoupled TTA) independent, unaffected. Paper 2 (FSR/Observability) frozen; Phase 8 → its
appendix/future-work. S2P is a **separate paper** ("Scaling Subjects, Not Erasing Subjects: Functional Shortcut
Audits of EEG Foundation Pretraining") **only if** the controlled study yields a clean signal.

Docs: S2P_01 (inventory), S2P_02 (parallel pretraining plan), S2P_03 (subject-scaling protocol). Go/no-go:
`results/s2p_inventory/s2p_go_nogo.json`.
