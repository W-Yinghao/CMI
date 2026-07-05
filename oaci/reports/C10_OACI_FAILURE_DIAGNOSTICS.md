# C10 — OACI failure-mode diagnostics (BNCI2014-001 seeds [0, 1, 2])

> **BNCI2014-001 minimum-seed diagnostics (seeds [0,1,2]). Artifact-only Part 1 on SELECTED checkpoints; epoch-level counterfactual replay is Part 2 (C10b).**

## Q1 — selection → audit optimism (does the selection-time leakage win transfer?)

- Δ selection leakage (OACI−ERM): mean **-0.3261**, reduced **54/54**
- Δ audit leakage (OACI−ERM): mean **+0.0076**, reduced **25/54**
- corr(Δselection, Δaudit): pearson +0.004, spearman +0.091 (n=54) — **near-zero ⇒ selection-optimism / criterion overfit**

## Q2 — audit leakage → target transfer (is held-out leakage predictive of DG?)

- Δ target worst_bacc mean -0.0024 · corr(Δaudit, Δtarget worst_bacc): pearson -0.064, spearman -0.222 (n=54)
- Δ target worst_nll mean -0.0739 · corr(Δaudit, Δtarget worst_nll): pearson -0.133, spearman -0.055 (n=54)
- Δ target worst_ece mean -0.0355 · corr(Δaudit, Δtarget worst_ece): pearson -0.114, spearman -0.009 (n=54)
- if |r| small / wrong sign ⇒ audit leakage reduction is orthogonal to downstream worst-domain DG (points toward case C)

## Risk tradeoff & level effect

- λ mean +2.864; corr(λ, Δtarget worst bAcc): pearson -0.289, spearman -0.261 (n=54) — negative ⇒ heavier penalty costs accuracy
- level 0: Δtarget worst bAcc +0.0011 · Δaudit leakage +0.0096
- level 1: Δtarget worst bAcc -0.0059 · Δaudit leakage +0.0055

## Method comparison (OACI − baseline, target worst bAcc)

- vs ERM: mean -0.0024 (improved 25, harmed 28)
- vs global_lpc: mean -0.0054 (improved 30, harmed 23)
- vs uniform: mean -0.0045 (improved 31, harmed 23)

## Harm localization

- target worst bAcc harmed in **28/54** fold-levels; total loss +1.002; top-5 folds = +0.40 of the loss (diffuse if ≈ 5/28, concentrated if near 1)
- worst fold-levels (seed,target,level,Δbacc): (1,8,1,-0.102), (0,3,0,-0.083), (1,8,0,-0.076), (0,1,0,-0.069), (0,4,1,-0.068)

## Part-1 lean & case A/B/C

- **part1_lean: `case_C_candidate`** (audit⊥target=True, selection_optimism=True, max|audit-target r|=+0.133)
- FINAL case A/B/C is decided in C10b with the epoch-level source-only + oracle replay; Part 1 alone cannot distinguish 'no good checkpoint exists' from 'selector picked badly'.

> **Part 2 (epoch-level counterfactual selector replay) — PENDING C10b GPU replay.** The final case A/B/C call requires the source-only (S1–S4) and source-audit-oracle (S5) replay over OACI's own risk-feasible trajectory.