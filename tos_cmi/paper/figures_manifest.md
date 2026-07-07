# Figures manifest — TOS-CMI measurement-to-control paper

Status legend: **[HAVE]** rendered artifact exists · **[DATA]** data exists, figure to be drawn ·
**[DRAW]** schematic to author. No new compute — figures are rendered from existing result artifacts.

---

## Figure 1 — Pipeline schematic  **[DRAW]**
**Shows:** the measurement→control flow, ending in a branch (delete OR abstain) so reviewers do not read
it as "another projector regularizer."
```
Z → score-Fisher spectrum (G_Y, G_{D|Y}) → candidate nuisance subspace V_D
  → direct-sum projector (RV=0, RT=T) → task-risk gate Δ_Y(k) + domain-gain gate
  → {certified deletion}  OR  {certified refusal / identity}
```
**Source:** hand-drawn (tikz/draw.io). **Message:** TOS-CMI is a localize→certify→(delete|refuse) pipeline.

## Figure 2 — Synthetic certification line  **[DATA]**
**Shows:** the measurement chain is *necessary*, not accuracy-chasing. Four panels:
1. covariance-only leakage → mean-scatter no-op, score-Fisher detects;
2. geometry-safe but conditionally-unsafe synergy → direct-sum alone would delete, task gate refuses;
3. weak nested critic unsafe-accepts → plug-in log-ratio improves;
4. certification at moderate n stays conservative (false-cert rate ~0).
**Source:** `tos_cmi/results/{cert_cells, cert_table_cells, frontier.json, frontier_cells, estimator_diag.json, phase_diagram_powerfloor.json}`; notes `PHASE131_CERTIFICATION.md`, `PHASE13_DIAGNOSIS.md`.
**Draw note:** pull exact false-cert / power numbers from the cert artifacts before captioning. **Message:** geometry alone is insufficient; the gate + plug-in estimator are each load-bearing.

## Figure 3 — TSMNet global-LPC collapse mechanism  **[HAVE]**
**Shows:** raw LPC removes subject leakage only via Z→0 collapse; warm-ramp/scale-invariant prevent it;
collapse-free LPC leaves leakage intact. 5 rows × 4 λ (task CE, λ·LPC penalty, encoder grad-norm,
eff_rank [scale-invariant], feature norm).
**Source artifact:** `tos_cmi/results/tos_cmi_eeg_frozen/lpc_collapse_curves/TSMNet/collapse_curves.png`
(regenerate: `TOS_BB=TSMNet python -m tos_cmi.eeg.collapse_analysis`).
**Companion data:** `.../TSMNet/variant_compare.json` (warm-ramp/scale-invariant keystone).
**Message:** "global CMI works" on TSMNet = "representation collapsed." Optimization, not geometry.

## Figure 4 — TSMNet leakage is high-dimensional / redundant  **[DATA]**
**Shows:** domain decode full Z ≈ 1.00; after low-rank TOS deletion still ≈ 0.95 (≈ random-k); task
preserved. Plus the Fisher-subspace-deletion curve (k=1..7) staying ~0.98 (redundant re-encoding).
**Source:** `tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_TSMNet_LOSO/ablation_report_seed{0,1,2}.json`
(task/domain Z/RZ/PNZ/Rrand, linear+MLP); aggregate3 outputs. Fisher-deletion curve from `verify` workflow re-derivation (recompute via a small script from the ERM npz if a figure is needed).
**Message:** low-rank deletion localizes leakage directions but cannot remove redundant subject identity
from a high-dimensional latent.

## Figure 5 — EEGNet contrast  **[DATA]**
**Shows:** (a) low-rank deletion drops subject leakage substantially (linear 0.82→0.35, MLP 0.88→0.54)
at ~0 task cost, but nonlinear residual remains; (b) global LPC λ-sweep reduces subject leakage
(0.89→0.19) **without collapse** (feat_norm never →0); (c) **target accuracy flat-to-worse** across λ.
**Source:** `.../BNCI2014_001_EEGNet_LOSO/ablation_report_seed{0,1,2}.json` (panel a);
`.../lpc_collapse_curves/EEGNet/raw_lpc_sub*_seed*.json` (panels b,c: subj_dec, feat_norm, tgt vs λ).
**Message:** removability is representation/capacity-dependent, but removability ≠ DG benefit.

## Table 1 — Summary across representations  **[DATA — populated below]**
| property | TSMNet (LogEig/SPD, 210-d) | EEGNet (conv, 16-d) |
|---|---|---|
| Subject decode (ERM) | 0.997 | 0.88 |
| Low-rank deletion effect | dents only (≈ random) | linear ~67% removed; MLP ~45% (≫ random) |
| Informed-vs-random selectivity | 0.04–0.08 | 0.35–0.55 |
| Task cost of deletion | ~0 | ~0 |
| Global LPC at high λ | feature-norm collapse to origin | no collapse, gradual |
| Collapse-free LPC removes leakage? | no | yes (0.89→0.19) |
| Target DG gain from removal? | n/a (collapses) | **no** (flat-to-worse) |
| Gate decision | abstain (not removable) | accept (removable subspace) |
**Source:** PHASE2_REPORT.md, PHASE3_BACKBONE_GENERALITY.md, variant_compare.json, ablation reports.
**Message:** the single clearest contribution summary — measurement→control gap across representations.

---

## Rendering checklist (no compute beyond plotting from existing JSON/NPZ)
- [ ] Fig 2: small plotting script over `tos_cmi/results/cert_*` + `frontier*.json`; pull exact numbers.
- [ ] Fig 3: already rendered (collapse_curves.png); relabel for paper.
- [ ] Fig 4: plotting script over TSMNet `ablation_report_seed*.json` (+ optional Fisher-deletion curve).
- [ ] Fig 5: plotting script over EEGNet `ablation_report_seed*.json` + `lpc_collapse_curves/EEGNet/*.json`.
- [ ] Table 1: numbers locked above; cross-check against claim_evidence_table before camera-ready.
