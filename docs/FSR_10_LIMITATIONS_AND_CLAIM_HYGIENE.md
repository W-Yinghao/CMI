# FSR_10 — Limitations & Claim Hygiene

**Project FSR — Phase 3A.** The limitations the paper must state up front, and the wording rules that keep every claim inside `FSR_05`. Stating these is a strength of an audit paper.

## Limitations (state in the paper)

1. **Provenance limit — leakage pooled correlation is not recomputable.** Per-fold `graph_kl` for seeds 1–2 was pruned from every branch (raw audit `.npz` + r2-gate JSONs uncommitted). `graph_kl→R3` is reproduced only at seed0 (n=42, sign only); the pooled n=126 value (−0.342) is `FROZEN_NOT_RECOMPUTABLE` — carried as support, never as a reproduction. Only `align_k2→R3` (n=126) is fully recomputed.
2. **Dataset heterogeneity / small n.** The alignment→reliance association is significant on 2a but not on 2015 (pooled-positive, not universal); the alignment-vs-leakage contrast regression is at seed0, n=42; within-group partial betas are not individually significant. The TOS capacity factorial is a single dataset (2a).
3. **`align_k` is not a validated estimator.** It is *closer* to functional reliance than raw leakage (correct sign; signed difference excludes 0), but we do not claim it estimates reliance; it is a candidate indicator.
4. **RQ2 negative correlation is not a finding.** The all-cells `corr(subject_removed, target_bAcc)` is negative, but Step-2C sensitivity shows it flips positive (+0.54) on the principled-eraser subset (LEACE/RLACE) and is ns when INLP/random-k are dropped — a confound of over-erasure and the random-k anchor. The RQ2 result is the *absence of a certified benefit* (0/40), not a negative association.
5. **RQ4 is blocked, not answered.** Per-branch leakage (L1) and per-branch reliance (L5) are not measured (no frozen `spatial_z/graph_z/temporal_z`/`node_z` dumps, no per-branch probe). We can state branch load (spatial load-bearing) but not per-branch leakage meaning.
6. **Probes are proxies, not CMI.** L1/L2 use a label-conditional linear-probe posterior-KL surrogate and a linear-probe advantage, not an unbiased mutual information; wording must say "extractable conditional domain information," not "CMI"/"I(Z;D|Y)."
7. **Erasability ≠ full removal.** LEACE drives *linear* subject decode to chance but leaves a nonlinear MLP residual; "erasable" is scoped to the tested operators.
8. **TTA-Control is seed0-only, non-CMI.** The one positive target-unlabeled result is seed0 (no seeds 1/2) and must be walled off from CMI-control; it is support/context, not an FSR headline.
9. **Frozen scope.** All results are re-analyses of frozen artifacts; no new training/tuning. Backbones are specific (DGCNN-style, TSMNet, EEGNet, FBCSP-LGG); generality beyond them is not claimed.

## Claim hygiene — forbidden vs allowed wording

| Forbidden | Allowed |
|---|---|
| "we fully reproduced both pooled correlations" | "align→reliance is recomputed at n=126; leakage→reliance is sign-confirmed at seed0 and frozen-supported at pooled" |
| "high leakage means the model relies on it" | "leakage is decodable (L1); reliance requires L5 evidence" |
| "align_k2 is a validated reliance estimator" | "align_k2 is closer to reliance than raw leakage in this frozen diagnostic" |
| "alignment has the larger effect magnitude" | "alignment is the correctly-signed predictor; leakage has larger |β| but the wrong sign" |
| "erasing subject signal improves DG" | "subject signal is erasable, but no eraser certifies a proven target benefit (0/40)" |
| "more subject removal harms the target" | "the negative correlation is a non-robust confound (over-erasure + random-k); not a finding" |
| "LEACE improves target NLL" | "LEACE's NLL move on 2a-TSMNet is matched by random-k without removing subject → non-specific" |
| "spatial leakage is harmful" / "graph leakage is benign" | "the spatial branch is load-bearing; branch-local leakage/reliance is not yet measured" |
| "CIGL/FCIGL/dCIGL/MetaCMI/CITA-CMI is a positive method" | "CMI control is closed; FSR audits the measurement→control gap" |
| "TTA-Control shows CMI control works" | "TTA-Control is a genuine but non-CMI target-unlabeled positive (seed0)" |
| "FSR is a new DG method / SOTA" | "FSR is an audit framework; a harmful shortcut requires task-coupled reliance evidence" |
| "P6 spatial-CMI result" | "P6 spatial-CMI is a scaffold, never run" |
| "ACAR shows the router works" | "ACAR Stage-2B is a completed DEV_STOP (router refuted on DEV)" |
| "CSC certifies concept shift" | "Z-only concept shift is provably unidentifiable; the CSC certifier abstains" |

## Mandatory caveats (attach wherever the claim appears)
- **C1:** the pooled leakage correlation is `FROZEN_NOT_RECOMPUTABLE`.
- **C2:** dataset-heterogeneous; not a validated estimator.
- **C4:** headline is `benefit_claimable=0/40`; the negative correlation is not robust.
- **C5:** scoped to the flagged 2a-TSMNet cell (1/8).
- **C9:** seed0-only, non-CMI, support only.

## Reviewer-anticipation checklist
- "Is the erasure→target negative correlation an INLP/collapse artifact?" → Yes; Step-2C shows it, and we do not headline it.
- "Is align→reliance just the 2a effect?" → It is dataset-heterogeneous; we say so and report per-dataset strata.
- "Did you fully reproduce the leakage result?" → No; per-fold seeds 1/2 were pruned; we tier the claim.
- "Did any target label leak into fitting?" → No; the firewall and the route inclusion table show only `NO`-tagged routes enter tests.
- "Is this a new DG method?" → No; it is an audit, and CMI-control is a closed premise.
