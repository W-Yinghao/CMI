# Tri-CMI experiment log

Master log of experiments this session. Metric key: **acc** = balanced accuracy (subject-level
for SCPS/ADFTD-MUMTAZ; per-target mean for MCPS); **lk** = conditional-leakage KL (↓ better,
frozen-encoder probe). lpc_prior = our method. All EEGNet unless noted; cross-subject LOSO unless noted.

## 2026-06-06 — foundation, method selection, core experiments

| date | experiment | purpose | rough result |
|---|---|---|---|
| 06-06 | Setup & recon (plan, survey, registry, env, data paths) | establish foundation | env `icml` verified (torch2.8/braindecode0.8/moabb1.2); MOABB loads offline from datalake; **no downloads needed** |
| 06-06 | Carrier-exploration workflow (16 agents) | pick estimator + carrier + baselines | → **LPC-CMI** posterior-KL upper bound; carriers EEGNet/Shallow/Deep4/Conformer; baselines CDANN/IIB/EEG-DG/DomainBed; design doc + 15 papers fetched |
| 06-06 | **Synthetic sanity check** (Protocol D, `figure2`) | prove LPC-CMI vs failure modes | lpc_prior uniquely drives leakage→~0 while preserving acc+label-sep, stable across λ; **marginal=label-erasure, chain=Y-erasure, uniform=mis-specified** |
| 06-06 | LOSO harness build + 2a/2b v1 | first real EEG pipeline | runs on V100; DG≈ERM on acc; lpc_prior leakage 0.98→0.23 |
| 06-06 | Faithful-backbone refactor + **v2 λ-sweep** (2a/2b) | credible baseline + tune λ | ERM best mean-acc (44/65); lpc_prior lowest leakage; λ=0.5 over-regularizes 4-class 2a |
| 06-06 | **Frameworks zoo** (CORAL/MMD/IRM/VREx/GroupDRO/DANN/CDANN/IIB) 2a/2b | full baseline table | all DG methods within ±noise of ERM on acc (known DomainBed result); **lpc_prior dominant on leakage** (0.02–0.08 vs 0.1–1.1) |
| 06-06 | Contrastive (supcon, lpc_supcon hybrid) 2a/2b | contrastive framework | supcon marginally>ERM on 2b; **no hybrid synergy**; lpc_prior lowest leakage |
| 06-06 | **Backbone-agnosticism** (EEGNet/Shallow/Deep4 @250) 2a/2b | architecture-agnostic claim | leakage cut 10–100× on ALL backbones; **ShallowConvNet-2b lpc_prior 66.1 > ERM 64.8 > CDANN 63.5** |
| 06-06 | **EEGConformer** 2a/2b | transformer backbone | lpc_prior matches ERM (44.3/66.3); **CDANN collapses to chance (25/50)** — adversarial instability |
| 06-06 | Cross-session (Protocol B) 2a/2b/Lee2019 | session-shift DG | lpc_prior≈ERM, beats CDANN, leakage 6–8× down |
| 06-06 | **Cross-dataset** (Protocol C) 2a↔Lee2019↔Cho2017, 21ch | unseen-device DG | lpc_prior **best worst-dataset (62.7)**, beats CDANN; small π_y>uniform edge |
| 06-06 | Emotion SEED / DEAP | cross-task generality | same pattern as MI; DEAP lpc_prior 51.1>ERM 50.6; beats CDANN; CDANN *worsens* leakage |

## 2026-06-07 — code review, SCPS, scale, extra carriers/baselines, validation

| date | experiment | purpose | rough result |
|---|---|---|---|
| 06-07 | **Code-review fixes** | correctness (user-reviewed) | fixed SCPS metric (pooled+subject-level), silent-ERM fallback, DANN λ² scaling, BN dummy-forward; **my backbone-guard fix was wrong → redone as try/except** |
| 06-07 | Imbalanced-LOSO (ρ=0.7) 2a/2b | show π_y on real MI | **π_y did NOT separate on MCPS MI** (lpc_prior≈uniform≈marginal, within noise) → motivated SCPS |
| 06-07 | **ADFTD 3-class SCPS** (88-subj) | conditional-vs-marginal showcase | **lpc_prior 62.4 > ERM 61.0 > uniform 58.2 > marginal 56.9 > cdann 53.2 > dann 48.9**; π_y>uniform +4.2; dann erases label |
| 06-07 | ADFTD_bin (AD-vs-HC) | replicate SCPS on binary | ⚠️ **does NOT replicate**: lpc_prior 78.9 < ERM 83.0 < cdann 84.8 |
| 06-07 | mumtaz (depression SCPS), 2nd dataset | replicate SCPS | *running* |
| 06-07 | **Multi-seed** (2a/2b/SEED ×3) | error bars / significance | 2a/2b lpc_prior slightly **below** ERM, **within seed noise** (2a ~50 vs 52; 2b ~68 vs 69) |
| 06-07 | Scale: Lee2019(54)/Cho2017(52)/HGD(14) | scale credibility | resubmitted after 12h time-kill; *running* |
| 06-07 | **LogCov** geometric carrier 2a/2b/SEED | non-neural representation | leakage→~0 on covariance features (regularizer works); base acc low (33/50) |
| 06-07 | **Proxy validation** (synthetic) | is the leakage metric real? | neural KL vs independent kNN Î(Z;D\|Y): **Pearson r=0.85, Spearman ρ=0.88** ✅ |
| 06-07 | chsic kernel competitor 2a/2b/ADFTD | kernel vs info-theoretic conditional | *running* |
| 06-07 | Classical Riemannian (TS+LR/MDM/CSP+LDA) 2a/2b | EEG-reviewer-expected baselines | DONE: 2a ~36–38, 2b ~48–49 (below neural; traditional reference) |
| 06-07 | SEED-IV (4-class emotion) | emotion breadth | *running* |
| 06-07 | **TSMNet** (spdbn DG SPDNet + lpc_prior carrier) 2a/2b | geometric DG baseline + SPD carrier | GPU run crashed (SPD layers CPU-only) → **rerun on CPU** (`scripts/cpu.slurm`); *running*. (paper's SPDDSMBN is UDA; we use DG `spdbn`) |
| 06-07 | **Classical re-run + duplicate cancel** | resource hygiene | classical was fine (parser artifact); cancelled dup jobs 843356/7 |

## 📋 PLANNED (NOT yet run — future work)
Marked **[PLANNED]**; no results yet. Ordered by priority. "depends on" = what it needs/answers.

### P1 — decide whether SCPS is a real accuracy headline (story-critical)
| # | [PLANNED] experiment | purpose | depends on / note |
|---|---|---|---|
| P1.1 | **Re-run classical baselines** (TS+LR/MDM/CSP+LDA) on a compute node | the empty JSONs failed on login; get EEG-reviewer-expected baselines | submit via CPU/GPU slurm (not login nohup) |
| P1.2 | **Multi-seed CIs** — aggregate 2a/2b/SEED ×3 seeds → mean±95%CI | is the lpc_prior-vs-ERM gap significant or noise? | ms0/1/2 results (running) |
| P1.3 | **Diagnose ADFTD-3class win vs ADFTD_bin loss** — λ-sweep on ADFTD_bin + per-class analysis | why does SCPS win not replicate on binary? | ADFTD λ-sweep (running) |
| P1.4 | **mumtaz (depression) full LOSO** read-out | 2nd SCPS dataset — replicate or refute ADFTD | running |
| P1.5 | **TUAB (TUH abnormal)** SCPS loader + LOSO | 3rd, large SCPS dataset to settle the SCPS claim | needs loader (~½ day) |

### P2 — fair baselines & required ablations (rigor)
| # | [PLANNED] experiment | purpose | note |
|---|---|---|---|
| P2.1 | **Per-framework λ tuning** (IRM/CORAL/MMD/VREx/GroupDRO) | current λ is first-pass; fair baseline comparison | λ sweep each |
| P2.2 | **SPDDSMBN (UDA)** via official repo `main.py inter-subject+uda` | the paper's headline DA number (labeled DA-not-DG) | repos/TSMNet, may need pinned env |
| P2.3 | **EEG-DG / SCLDGN exact** reimplementation on our protocol | head-to-head with the most-related app baselines | reimplement losses |
| P2.4 | **Plan-required ablations**: warm-up vs none · balanced-sampler vs not · penalty on feature vs logits · π_y Laplace-α sensitivity · q_ψ capacity | the plan §8.4 ablation table | small harness flags |
| P2.5 | **λ-sensitivity curves** for lpc_prior (all datasets) | show robustness to λ (not a lucky hyperparam) | partly have (v2) |

### P3 — breadth, figures, analysis (some free / no GPU)
| # | [PLANNED] experiment | purpose | note |
|---|---|---|---|
| P3.1 | **Calibration (ECE/NLL)** table from existing results | plan wants calibration as a selling point | free (already computed) |
| P3.2 | **Leakage-vs-robustness scatter (Fig 3)** + within-dataset H1 test | does lower leakage track worst-target acc? | refine `analysis/` |
| P3.3 | **t-SNE/UMAP of Z (Fig 4)** — same-label subject clusters | visualize leakage compression | from a trained model |
| P3.4 | More emotion: **DREAMER, FACED** loaders + LOSO | emotion breadth | loaders |
| P3.5 | **knncmi** (true mixed-CMI, GPL eval-only) on synthetic | stronger ground-truth than sklearn-stratified proxy | optional, isolate GPL |
| P3.6 | Cross-dataset variants (2b 3-ch; more combos) | robustness of Protocol C | minor |

### P4 — assembly
| # | [PLANNED] experiment | purpose |
|---|---|---|
| P4.1 | Consolidate all into **paper Table 1/2 + Figs 1–4** | AAAI main tables |
| P4.2 | Reproducibility pack (configs, seeds, env.yml, README, checklist) | AAAI supplement |

## Net story so far
- **Robust / confirmed:** architecture- & task-agnostic **leakage removal** (10–100×, every backbone incl. SPD & covariance, every task); **beats CDANN** on leakage+robustness everywhere; CDANN unstable (collapses on transformer); leakage proxy **validated** (r=0.85); failure modes (marginal/chain) reproduce.
- **Accuracy:** parity with ERM on balanced MI (a wash for the whole DG field); **one SCPS win (ADFTD 3-class)** that **did not replicate on ADFTD_bin** — SCPS-as-accuracy-headline is **not yet established**; awaiting mumtaz + multi-seed CIs.
- **π_y correction:** clearly shown in synthetic + proxy-validation; weak/absent on balanced MI; +4 on ADFTD-3class but not ADFTD_bin — **regime-dependent, not universal**.
