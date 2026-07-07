# FSR_50 — Trained-Representation Subject-Scaling Decision (Phase 8D opening)

**Project FSR — Phase 8D decision.** PM-approved after 8C-1. Records **why** a trained-representation arm is the
clean next step and **what it is bounded to**.

## Why (the 8C-1 boundary)
8C-1 (FSR_49) established, on frozen CBraMod/CodeBrain + full-pool PCA:
- transfer improves with more source **trials** (sample-size, N≥16-driven);
- the **subject subspace is never a task lever** at any `N_source` (extends 8B);
- **but the design CANNOT test whether subject diversity changes the representation** — pairwise-L1 is measured in a
  fixed source-pool PCA space with a frozen encoder, so it is **training/N-independent by construction** (verifier
  MAJOR-1). "Diversity does/does not reduce subject separability" is **not answerable** without a *trained*
  representation.

## Decision
- **Approve Phase 8D:** a **small trained-representation adapter pilot** (A1 = bottleneck adapter on the fixed PCA
  feature, trained on source task labels), so the representation **can** respond to source-subject count. This
  directly targets the original foundation-model / subject-scope question.
- **Do NOT:** run specialist baselines; full-fine-tune the foundation model; add datasets/channels; change the
  target panel. (CodeBrain's own paper full-fine-tunes with a 3-layer MLP head + reports pretraining-hours/model-
  size scaling — a different regime from our frozen/lightly-trained subject-count audit.)
- **Bound:** CBraMod primary, CodeBrain exploratory; pilot grid `N_source ∈ {2,8,all}`, fixed {2,8}, subset_seeds 3,
  train_seeds 2, ≤2 pre-declared bottleneck sizes; source-only training + selection; target labels final-scoring.

## Gate
8D-PASS (real trained-representation effect: target gain w/o reliance increase, or L1 moves with N, or subject-
retained-not-lever) → expand full grid (PM review first). 8D-NULL (A1 ≈ A0 frozen) → **stop Phase 8**, write the
frozen+lightly-trained conclusion as an FSR extension/appendix. Protocol: FSR_51. Results: FSR_52.

## Frozen 8C-1 wording (do not exceed)
> In frozen CBraMod/CodeBrain representations on PhysioNetMI, more source data yields a modest sample-size effect,
> and the measured subject subspace does not become a task lever across `N_source`; the frozen design cannot test
> whether subject diversity changes the representation itself.

Paper 2 remains frozen; PC2 paused; Paper 1 unaffected.
