## Active project: CIGL / GraphCMI

CIGL (Conditional Information Graph Learning) studies **label-conditional domain leakage in EEG graph representations at graph, node, and edge levels**: it treats the per-sample learned adjacency `A(X)` as a possible subject fingerprint and regularizes `I(Z_g;D|Y)`, `(1/C)·Σ_v I(Z_v;D|Y)`, and `I(A;D|Y)`. The original Tri-CMI / LPC-CMI work below remains the **theoretical and estimator foundation**. **CITA** is a separate **transductive / test-time alignment** branch and is **not** the main strict-DG project.

Start here: [`docs/CIGL_00_PROJECT_CHARTER.md`](docs/CIGL_00_PROJECT_CHARTER.md) · spec [`docs/CIGL_01_METHOD_SPEC.md`](docs/CIGL_01_METHOD_SPEC.md) · plan [`docs/CIGL_03_IMPLEMENTATION_PLAN.md`](docs/CIGL_03_IMPLEMENTATION_PLAN.md) · acceptance [`docs/CIGL_05_ACCEPTANCE_CRITERIA.md`](docs/CIGL_05_ACCEPTANCE_CRITERIA.md).

Setting labels are never mixed in a main table: CIGL results are `setting = strict_source_only_DG`; CITA/TTA results are `setting = transductive_TTA`. Code/result layout: backbone `cmi/models/gnn.py` (`GraphCMINet`), graph/node/edge leakage heads `cmi/methods/graph_regularizers.py`, trainer branch `method="graphcmi"`, runner config grammar `graphcmi:<lambda_g>:<lambda_node>:<lambda_edge>`, results schema `results/cigl/schema.md`.

---

# Tri-CMI — Label-Prior-Corrected Conditional Mutual Information for Calibration-Free EEG Domain Generalization

**Target:** AAAI-27 Main Technical Track (abstract 2026-07-21, full paper 2026-07-28, supplement/code 2026-07-31, all AoE).
**Working title:** *Tri-CMI: Label-Prior-Corrected Conditional Mutual Information for Calibration-Free EEG Domain Generalization*.

## One-line thesis
Learn a representation `Z = f_θ(X)` that maximizes task predictability while minimizing the **conditional domain leakage** `I(Z; D | Y)` — how much domain (subject/session/device/site) information remains in `Z` *after* the task label `Y` is known. The core estimator (**LPC-CMI**) uses a label-conditioned domain posterior `q_ψ(D|Z,Y)` regularized toward a label-conditional prior `π_y(D)=p(D|Y)`:
`L_CMI = E_i KL( q_ψ(·|z_i,y_i) || π_{y_i}(·) )`. (See `Tri-CMI_EEG_DG_AAAI_research_plan.docx`.)

Claim boundary: **label-conditional domain invariance / conditional domain-leakage minimization** — NOT "causal-invariant". Call the regularizer a *neural plug-in CMI proxy*, not an unbiased estimator.

## Compute environment (verified 2026-06-06)
- **Python env:** conda env `icml` → `/home/infres/yinwang/anaconda3/envs/icml/bin/python`
  - torch 2.8.0+cu128, moabb 1.2.0, **braindecode 0.8**, mne 1.8.0, pyriemann 0.7, skorch 1.2.0, geoopt 0.5.1, einops 0.8.1.
  - (fallback env `eeg2025`: torch 2.6.0+cu124, moabb 1.5.0, pyriemann 0.11, but **no braindecode**.)
- **SLURM GPU partitions:** `A100` (1-day), `V100`/`V100-32GB`/`V100-16GB` (2-day), `H100`, `L40S`, `A40` (4-day), `A30`, `P100`, `3090`. Use V100/A100 per instruction. Login node has **no GPU** — all training via `sbatch`.
- **Data root (read-only datalake):** `/projects/EEG-foundation-model/datalake/raw` — set `MNE_DATA` and `MNE_DATASETS_BNCI_PATH` to this to load MOABB offline (verified: BNCI2014_001 loads with no download).
- **Our download/scratch dir:** `/projects/EEG-foundation-model/yinghao` (check datalake/raw before downloading anything new).

## Datasets available locally (motor-imagery, MCPS — conditional distribution shift)
All present in MOABB cache layout under `datalake/raw/` — **no re-download needed**:
| MOABB id | folder | subj | classes | sessions | role |
|---|---|---|---|---|---|
| BNCI2014_001 (BCI-IV-2a) | MNE-bnci-data/.../001-2014 | 9 | 4 (L/R/feet/tongue) | 2 | LOSO + cross-session |
| BNCI2014_004 (BCI-IV-2b) | MNE-bnci-data/.../004-2014 | 9 | 2 (L/R) | 5 | LOSO + cross-session |
| Lee2019_MI (OpenBMI) | MNE-lee2019-mi-data | 54 | 2 (L/R) | 2 | large-scale LOSO + cross-session + cross-dataset binary |
| Cho2017 (GigaScience) | MNE-gigadb-data | 52 | 2 (L/R) | 1 | cross-dataset binary |
| Schirrmeister2017 (HGD) | MNE-schirrmeister2017-data | 14 | 4 | 1 | strong-encoder robustness |
| Stieger2021 | MNE-Stieger2021-data | 62 | 2 or 4 | 7–11 | cross-session at scale |
| Weibo2014, Zhou2016, PhysionetMI(eegbci), Kaya2018, Jeong2020 | resp. folders | — | — | — | supplement / cross-dataset |

Registry reference: `A Unified EEG Data Registry_v1.0.xlsx` (Sheet1, 834 datasets).

## Evaluation protocols (strictly source-only model selection; no target calibration)
- **A — within-dataset LOSO:** leave one subject out; target excluded from norm/selection/calibration.
- **B — cross-session:** train on sessions, test on unseen session; domain = subject-session pair.
- **C — cross-dataset binary MI:** common labels (left vs right hand); channel intersection / standard montage map.
- **D — synthetic sanity check:** causal feature + spurious domain-label feature; show LPC-CMI fixes label-erasure & spurious reliance (Milestone 1, CPU-only).

## Metrics
Balanced Accuracy (main), Macro-F1, **Worst-Subject Accuracy**, std across targets, ECE/NLL, **conditional domain leakage** (frozen-encoder probe `q_probe(D|Z,Y)` KL-to-prior), label separability (linear probe).

## Two axes (keep distinct — see `notes/harness_objectives.md`)
- **Backbone (encoder, swappable):** EEGNet / ShallowConvNet / Deep4Net / EEGConformer / ATCNet / LogCov. We show *backbone-agnosticism*.
- **Framework (learning paradigm):** ERM · adversarial (DANN/CDANN) · alignment (CORAL/MMD) · invariance-IB (IRM/VREx/IIB) · disentanglement (ManyDG) · contrastive (SupCon/SCLDGN/CLISA) · **our Tri-CMI** (information-regularizer framework). Tri-CMI *is* a framework; the others are competitor baselines + a few alternative-host ablations.

## Status / decisions
- [x] Plan, survey (`domain_generalization_review.pdf` = Li et al., *Cross-Subject Generalization for EEG Decoding: A Survey*), registry, env, data-path reviewed/verified.
- [x] Framework + estimator + backbone selection → `notes/carrier_design.md`, `notes/harness_objectives.md`. **Estimator = LPC-CMI (posterior-KL upper bound).** Tri-CMI = standalone framework; baselines = CDANN/EEG-DG/IIB/DANN/CORAL/MMD/VREx/GroupDRO/SCLDGN/TSMNet.
- [x] **Datasets locked** (baseline-comparability) → `notes/datasets.md`: core **2a + 2b**, scale **Lee2019**, breadth **Cho2017/HGD**. All offline, no download.
- [x] Papers obtained → `papers/` (15/16; only IEEE EEG-Conformer paywalled). Reference repos cloned → `repos/` (SCLDGN, SupContrast).
- [x] **Milestone 1: Synthetic sanity check (Protocol D)** → `notes/synthetic_results.md`, `synthetic/figure2_sanity.png` (8 seeds). lpc_prior uniquely drives leakage→0 while preserving accuracy+label sep; marginal=label-erasure, chain=Y-erasure, uniform=mis-specified.
- [x] **MOABB+braindecode LOSO harness** (`cmi/`) built + verified end-to-end on V100.
- [~] **2a + 2b LOSO main table** (EEGNet × 5 frameworks, 300 ep) running on V100. Then: cross-session, Lee2019 scale, backbones (Conformer/Shallow), baselines (CDANN/IIB/EEG-DG).

See `Tri-CMI_EEG_DG_AAAI_research_plan.docx` §10 for full milestone schedule.
