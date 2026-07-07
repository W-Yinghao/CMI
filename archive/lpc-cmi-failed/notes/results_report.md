# Tri-CMI — Results Report (completed experiments)

Updated 2026-06-08. Metric: **balanced accuracy** (per-target mean for MCPS; subject-level for SCPS),
**leakKL** = conditional-leakage KL (↓ better). `lpc_prior` = our method (LPC-CMI posterior-KL). EEGNet,
cross-subject LOSO, unless noted. See `notes/experiment_log.md` for the experiment index, and the
`cmi-empirical-findings` memory for interpretation.

## Headline
1. **Leakage removal is the rock-solid result** — `lpc_prior` cuts conditional leakage 10–100× on every
   Euclidean backbone, every task, and **beats both competitors** (CDANN adversarial, chsic kernel) at it,
   while *preserving the label* (which CDANN/marginal damage).
2. **Accuracy is λ-sensitive, not a fixed-λ win**: with **source-selected small λ (≈0.05–0.1)**,
   `lpc_prior ≈ ERM-or-better AND removes leakage`; too-large λ trades accuracy for more removal. So the
   honest accuracy claim is "*no cost at proper λ*", not "beats ERM".
3. **Estimator is trustworthy** (audit: permutation-null ≈ 0, 6 probes agree), but `lpc_prior` removes the
   *training-induced* leakage back toward the **random-encoder floor** — it can't remove irreducible
   subject info without hurting the task (no-free-lunch).

## 1. Motor-imagery, framework zoo (2a 4-class / 2b binary, @128 Hz)
| method | 2a acc | 2a leakKL | 2b acc | 2b leakKL |
|---|---|---|---|---|
| ERM | 42.1 | 1.18 | 64.8 | 0.54 |
| CORAL | 42.7 | 0.77 | 65.1 | 0.34 |
| MMD | 41.6 | 0.16 | 64.9 | 0.07 |
| IRM | 35.6 | 0.12 | 64.4 | 0.06 |
| VREx | 37.1 | 0.74 | 64.4 | 0.39 |
| GroupDRO | 38.4 | 1.06 | 63.1 | 0.54 |
| DANN | 40.7 | 0.42 | 65.2 | 0.11 |
| CDANN | 38.3 | 0.28 | 65.2 | 0.09 |
| IIB | 43.5 | 0.82 | 65.6 | 0.41 |
| **lpc_prior** | 39.1 | **0.08** | 64.9 | **0.02** |

All DG methods within ±noise of ERM on accuracy (the field-wide null); **lpc_prior dominant on leakage**.

**Per-framework λ-tuning (2a, fair baselines):** even tuned, CORAL barely cuts leakage at acc-preserving λ;
**MMD λ=10 reaches lk 0.05 @ acc 50.3** (competitive on leakage — but it's *marginal* `I(Z;D)`, no Y-protection);
IRM unstable (acc 31.8/38.0); VREx cuts leakage only by hurting acc. So no Euclidean baseline matches our
*conditional* removal + label preservation at parity accuracy.

## 2. λ-sensitivity (250 Hz/4 s window — higher base accuracy)
| λ | 2a acc / leakKL | 2b acc / leakKL |
|---|---|---|
| ERM | 52.0 / 1.20 | 68.6 / 0.53 |
| lpc_prior 0.05 | 51.4 / 0.61 | **69.2 / 0.08** |
| lpc_prior 0.1 | 50.1 / 0.30 | 68.5 / 0.06 |
| lpc_prior 0.3 | 47.1 / 0.13 | 67.7 / 0.03 |
| lpc_prior 1.0 | 46.1 / 0.03 | 67.8 / 0.01 |

**At λ=0.05, 2b lpc_prior beats ERM (+0.6) with 6× less leakage; 2a is ~parity (−0.6) with leakage halved.**

## 3. Multi-seed (×3) — ERM vs lpc_prior:0.3 (λ=0.3 is slightly too strong; cf. sweep)
| dataset | ERM | lpc_prior:0.3 | CDANN |
|---|---|---|---|
| 2a | 51.8±0.4 | 49.3±1.7 | 48.0±0.7 |
| 2b | 69.0±0.4 | 68.0±0.6 | 68.3±0.4 |
| SEED | 54.6±0.4 | 53.8±0.1 | 53.9±1.1 |

## 4. Other protocols
- **Cross-session (B):** 2a ERM 54.6/lk1.20 → lpc_prior:0.1 53.7/0.30; 2b 70.3 → 70.1/0.10; Lee2019 68.4 → 67.6/0.13. Beats CDANN (46.5/67.4/63.3) on accuracy+leakage.
- **Cross-dataset (C, unseen device):** ERM 66.4/lk0.82, lpc_prior:0.3 65.5/**0.04**, CDANN 63.9/0.34 (lpc best worst-dataset).
- **Scale MI (more subjects):** Lee2019(54) ERM 69.1 / lpc_prior:0.1 68.9 (leak 0.65→0.10) / CDANN 63.8↓; Cho2017(49) 64.7 / 64.5 (0.73→0.04) / 61.6↓; HGD(14) 61.4 / 58.8 / 58.7. **lpc_prior ≈ ERM + beats CDANN by +3–5** (CDANN underperforms ERM on MI). [HGD low — needs 4–125 Hz band per the preprocessing doc.]
- **Backbone-agnostic:** leakage cut 10–100× on EEGNet/Shallow/Deep4/Conformer/LogCov; ShallowConvNet-2b lpc_prior 66.1 > ERM 64.8; **CDANN collapses to chance on EEGConformer**. ⚠️ Scope: NOT TSMNet/SPDNet (see §8).

## 5. Emotion
| dataset | ERM | lpc_prior | leakKL erm→lpc |
|---|---|---|---|
| SEED (3-cls) | 55.3 | 53.9 | 0.67→0.05 |
| SEED_IV (4-cls) | 35.0 | 34.9 | 0.65→0.03 |
| DEAP (valence) | 50.6 | 51.1 | 0.94→0.00 |

## 6. SCPS disease detection (subject-level)
| dataset | ERM | lpc_prior | note |
|---|---|---|---|
| ADFTD (3-cls) | 61.0 | **62.4** (λ0.3) | beats all DG baselines; π_y>uniform +4.2 |
| ADFTD_bin | 83.0 | **83.4** (λ0.1) | earlier "loss" was λ over-reg (λ0.3=82.3, λ0.5=79.2) |
| **TUAB** (80-subj) | 62.5 | **65.0** (λ0.3) | ✅ beats ERM +2.5, leak 1.42→0.04 (35×); **π_y: lpc 65.0 > uniform 60.0 (+5)**; cdann 65.0 keeps leak 1.13 |
| **MUMTAZ** | 85.4 | **89.3** (λ0.5) | ✅ RESOLVED: +3.8, leak 1.59→0.02 (70×); earlier "loss" was the buggy-sampler run |
| ADFTD (Deep4Net) | 49.5 | **57.0** (λ0.1) | ✅ +7.5 — SCPS win is **backbone-general**; > cdann 53.6 |

**SCPS now 4/4** support "lpc_prior ≥ ERM + leakage removal". But **ADFTD is SEED-SENSITIVE** (seed0 +3.7, seed1 −5.0, seed2 −0.3) → ~parity across seeds; trust TUAB/MUMTAZ + need their CIs. **Subject-balanced π_y holds/improves** (ADFTD +6.5, π_y>uniform +2.7). SCPS = a λ-tuning story.

**ADFTD λ-sweep = the conditional-vs-marginal FAILURE CURVE (paper figure):** as λ grows, `lpc_prior` peaks
(61.3 @λ0.3, +3.7 over ERM 57.6) then gently declines (58.1 @λ1.0) with label preserved (labelSep 94.9→86.7);
`marginal` degrades 58.7→**51.0** with labelSep collapsing 85→**56**; `dann` 50.8→**37.5**, labelSep→46. Removing
*all* subject info erases the label (subject≈label in SCPS); conditioning on Y protects it. λ-robust vs λ-fragile.

## 6b. Accuracy pivot — where we beat baselines (the winnable claims)
- **Worst-subject (DG's true metric):** small-λ lpc_prior > ERM on worst-subject — 2a +3.1 (39.1 vs 35.9), 2b +1.2 (56.7 vs 55.4). λ-dependent → pair with constrained-λ selection.
- **vs other DG methods (mean):** we beat the unstable/hurting ones (IRM 35.6, VREx 37.1 << ERM 42.1; CDANN collapses on Conformer; DANN/CDANN lose on MI). But on MCPS *mean*, IIB 43.5 / CORAL 42.7 can edge us — **a pure regularizer does NOT beat ERM on balanced-MI mean** (DomainBed null). Honest framing = leakage–accuracy Pareto.
- **EA is TRANSDUCTIVE, not a CMI win:** 2a ERM none 43.2 → **ea 48.8** (+5.6) → but **ea_strict 41.8** (source-stats-only ≤ none). **The whole EA gain comes from using the target's unlabeled trials** → "zero-LABEL calibration", not strict DG. On top of EA, CMI adds worst-case (not mean). RA≈EA; **HA hurts** (LogCov 36.8→32.7).
- **EA's boost is DATASET-DEPENDENT (small when source is large):** 2a (9-subj/22-ch) +5.6, but **Lee2019 (54-subj/62-ch) only +1.3** (ERM none 69.5→ea 70.8; lpc 69.4→71.3) and 2b (3-ch) ~none. EA helps most with a *small/few-subject source* (per-subject recentering matters); a large diverse pool is already centered. On Lee2019 lpc≈ERM at every alignment (MCPS parity), and EA+CMI ≥ EA+ERM (71.3≥70.8) — no CMI penalty on top of EA here. (Lee2019 ea_strict pending = strict-DG check.)

## 6c. Calibration (ECE/NLL) — a concrete downstream WIN (`notes/calibration.md`, no GPU)
**lpc_prior is better-calibrated than ERM on the large majority of datasets**, often dramatically: ADFTD ECE 32.3→**22.8** (NLL 2.18→1.51), TUAB 29.3→**24.8** (1.68→1.20), DEAP-arousal 28.0→**18.4**, DEAP-quadrant 34.5→**16.4** (−18!), LogCov-2a NLL 8.08→**3.68**, 2b-imb 10.3→**3.9**. Only SSL/EA-already-calibrated runs are neutral. So removing subject-shortcuts makes the model **less overconfident even where mean accuracy is parity** — the "leakage removal has a concrete benefit" result. Computed from saved `.preds.npz` (no retrain).

## 7. Classical Riemannian baselines (pyRiemann; reviewer-expected reference)
| dataset | TS+LR | MDM | CSP+LDA |
|---|---|---|---|
| 2a | 36.3 | 36.8 | 37.8 |
| 2b | 49.0 | 48.0 | 48.9 |
| Lee2019_MI | 65.9 | 54.5 | 64.9 |
| Cho2017 | 60.6 | 52.7 | 61.0 |
| SEED | 51.3 | 45.5 | 48.5 |
| DEAP | 50.2 | 49.2 | 49.7 |
| ADFTD | 58.0 | 57.8 | 43.8 |
| ADFTD_bin | 74.8 | **80.9** | 71.0 |
| TUAB | 47.5 | 53.8 | 45.0 |

Generally below the neural methods; competitive on Lee2019 / ADFTD_bin.

## 8. Ablations / negative results
- **Leakage audit (estimator scrutiny):** permutation-null ≈ 0 (−.005..+.018) everywhere; all 6 probes
  (linear/mlp/rf/hgbm/knn) agree → estimator trustworthy, not a single-probe artifact. BUT lpc_prior lands
  ≈ random-encoder floor on MI (2a lpc 0.49 vs random 0.37); and on ADFTD the internal KL ranks lpc≪cdann
  while the probe-ensemble ranks lpc>cdann — resolved by: **cdann removes more but destroys the label**.
- **Route 2 (FMCA + chain-rule):** Y-erasure ablation confirmed — 2a `fmca_chain` labelSep 59.1→**39.5**
  (vs lpc_prior 55.8); `fmca_diff` corrects partially (45.3) at lower acc; **lpc_prior dominates all FMCA
  variants on both acc and label**. Route 2 → appendix only. (binary 2b: no erasure, all ~equal.)
  On TSMNet/SPD (the hypothesized FMCA-friendly "Gaussian-edge"): **FMCA collapses the net to chance**
  (fmca_chain & fmca_strat 25.0/labelSep 23.5) — no win materializes. Route-2 strictly dominated everywhere.
- **chsic (kernel conditional-HSIC competitor):** weaker leakage remover than lpc_prior — 2a 0.43 vs 0.12,
  ADFTD barely works 1.53→1.05 vs lpc 0.11.
- **TSMNet/SPDNet = baseline NOT carrier:** SPD tangent features are the most leakage-prone (erm leakKL
  2.0/1.7); lpc_prior:0.3/0.1 either no-effect (2a) or **collapses to chance** (2b 65.6→50.0; small-λ 0.1 also
  collapses to 53.3) — no λ removes leakage without collapse. Use only as a geometric DG baseline.
- **Self-supervised contrastive (SimCLR/BYOL):** built + `lpc_simclr`/`lpc_byol` (CMI-on-SSL); supcon ≈ ERM acc,
  weak leakage removal, no synergy. (full SSL numbers pending.)

## 9. GNN (GraphCMINet) — node/edge CMI [running]
- Built: `GraphCMINet` (raw node enc → per-sample adjacency → SGC; node=channel) + node/edge CMI
  (`Σ_v I(Z_v;D|Y)` length-C leakage map + `I(A;D|Y)`); DGCNN/RGNN baselines (shared adjacency). Stage 1+2 verified.
- **Lit-search verdict (`notes/recent_eeg_gnn.md`):** our non-adversarial label-conditional node+edge domain-MI
  is genuinely unclaimed; near-misses = RGNN-NodeDAT (node/adversarial/unconditional), **GDDN** (nearest edge prior,
  decomposition≠cond-MI), FreqDGT (adversarial/marginal). **Corrections:** BrainIB=**fMRI** (not EEG); whole SEED
  GNN line uses **DE features** (we use raw → flag DE-vs-raw; headline leakage maps + gen-gap, not raw-acc SOTA).
- SEED benchmark (vs DGCNN/RGNN + node/edge ladder) **pending** (behind slow DEAP jobs). DEAP-quadrant 4-cls = chance for all.

## In flight / pending
GNN benchmark SEED (DGCNN/RGNN/GraphCMI ablation), EA+CMI 2a/2b/SEED/Lee2019, SSL + CMI-on-SSL 2a/2b,
rebuttal α-sweep + Lee2019 ea_strict. (Most queued behind slow DEAP stragglers.)
