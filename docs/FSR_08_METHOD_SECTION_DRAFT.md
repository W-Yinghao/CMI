# FSR_08 — Method Section Draft

**Project FSR — Phase 3A.** A draft of the paper's Method section, written from the frozen schema. Notation and definitions are paper-ready; numbers live in the Results section (FSR_04) and are not repeated here.

---

## 3.1 The functional-shortcut-reliance ladder

Let `Z` be an EEG representation, `Y` the task label, `D` the domain (subject/recording). A *harmful functional shortcut* is domain information in `Z` that is simultaneously (i) measurable, (ii) localizable, (iii) interventionable, (iv) functionally relied upon by the task head, and (v) harmful in target consequence. We operationalize this as a six-level ladder and require every shortcut claim to be a relationship *between* levels — never an inference from a low level to a high one.

- **L1 Detectability.** Can `D` be decoded from `Z` given `Y`? Measured by a label-conditional domain probe / posterior-KL proxy `I_w(Z;D|Y)` (a plug-in linear-probe surrogate, **not** an unbiased CMI) and a within-label permutation null.
- **L2 Reducibility.** Does a method reduce the *measured* leakage? Paired ΔKL / Δprobe-accuracy vs a matched baseline.
- **L3 Erasability.** Can a linear/non-linear eraser remove the subject signal? Post-eraser residual subject decodability (linear + MLP) for LEACE / INLP / RLACE / mean-scatter, with a same-rank random-`k` control.
- **L4 Task coupling.** Is the erased/residual subspace aligned with the task head? Fraction of linear task-head row-space energy inside the top-`k` label-conditional subject subspace (`align_k`), and, for multi-branch models, branch-ablation drop + fusion-gate weight.
- **L5 Functional reliance.** Does removing the subspace change the task head's output? Head-replay task-drop `R3` after removing the top-`k` label-conditional subject subspace (source-only fit), with a random-subspace control and an exact-replay firewall.
- **L6 Target consequence.** Is that reliance harmful, benign, or task-useful? Held-out-target bAcc/NLL/ECE delta, worst-subject delta, and a refuse/accept decision — with target labels used only for scoring.

**Direction contract.** Under the naïve "leakage-is-shortcut" hypothesis, `corr(leakage, R3)` and `corr(subject_removed, target_benefit)` should be positive. The FSR question is whether they are.

## 3.2 Claim-strength provenance tiers

Every quantitative statement is tagged:
- **RECOMPUTED** — recomputed from committed per-unit data (e.g. `align_k2→R3` at n=126).
- **RECOMPUTED_SIGN_ONLY** — only a recomputable slice confirms the sign (e.g. `graph_kl→R3` at seed0, n=42, because per-fold leakage for seeds 1–2 was pruned).
- **FROZEN_NOT_RECOMPUTABLE** — value carried from a frozen output whose per-unit inputs are unavailable; support only, never a "reproduction."

A fail-closed validator (`validate_step1_index.py`) enforces the schema; a CIGL reproduction script re-derives the headline and writes a STOP file if it drifts beyond tolerance.

## 3.3 Phase-1 quantitative-inclusion gate

A route enters a cross-level quantitative test only if it has ≥1 predictor-side level {L1,L2,L3,L4} **and** ≥1 endpoint-side level {L5,L6}. Routes lacking this pairing are tagged `SUPPORT_ONLY` / `BOUNDARY_ONLY` / `PROTOCOL_ONLY` / `BACKGROUND_ONLY` and may inform interpretation but not the regression. Of 37 audited routes, 6 are quantitatively included (CIGL for RQ1/RQ3; five TOS erasers for RQ2); RQ4 has zero includable routes because the per-branch L1/L5 metrics do not exist.

## 3.4 Target-label firewall

Target labels `y_T` are used only to *score* the L6 endpoint; they never enter probe/eraser fitting, model selection, hyper-parameter selection, or early stopping. Each route carries a `target_labels_used_for_fit ∈ {NO, YES_FORBIDDEN, AUDIT_ONLY, UNKNOWN}` tag; every quantitative RQ uses only `NO`-tagged routes. One legacy route is `YES_FORBIDDEN` (a retracted, disclosed target-label leak) and is excluded from all tests.

## 3.5 Data and backbones (frozen)

- **CIGL (RQ1/RQ3):** static graph backbone (DGCNN-style) on BCI-IV-2a (BNCI2014_001) + BNCI2015_001, LOSO, seeds {0,1,2}; per-unit `align_k2`, `graph_kl`, `R3_task_drop`.
- **TOS (RQ2):** frozen-feature erasure on BCI-IV-2a + Lee2019_MI + Cho2017 + Schirrmeister2017, backbones TSMNet (SPD, z=210) + EEGNet (conv, z=16); erasers LEACE/INLP/RLACE/mean-scatter(TOS_VD)/random-k; source-only fit, held-out-target deploy.
- **FBCSP-LGG (RQ4):** filter-bank CSP + learnable graph backbone with a 3-way gated fusion over graph/temporal/spatial branches; branch ablation + gate weights only (no per-branch leakage/reliance probe).

## 3.6 Statistics

Spearman correlations with a paired bootstrap CI (`rng(0)`, 2000 resamples, percentile[2.5,97.5]); the alignment-vs-leakage comparison uses a signed-difference bootstrap. Mechanism regressions use OLS with a z-scored outcome and predictors plus one-hot dataset/seed/method controls (fully-standardized β). Robustness: dataset-stratified, per-seed, per-method, leave-one-dataset-out, within-group rank-residualization; for erasure, subset sensitivity across eraser families and a within-dataset×backbone control. All analyses are CPU-only over frozen artifacts.

## 3.7 What the method does not claim

We do not train, tune, or select any model on the analyzed data; we do not propose a CMI regularizer or a DG method; `align_k` is presented as a candidate reliance indicator (closer to reliance than leakage), **not** a validated estimator; and branch-locality is presented as blocked pending a per-branch probe, not as a per-branch leakage result.
