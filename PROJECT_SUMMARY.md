# Tri-CMI — Project Summary (for a new researcher)

**One-line:** a *calibration-free* domain-generalization method for EEG that removes **conditional**
domain leakage `I(Z;D|Y)` via a label-prior-corrected variational estimator, and plugs into *any*
encoder/framework as an added loss term. Target venue: **AAAI**. Status: **preliminary** (results are
exploratory; a unified-preprocessing final re-run is planned).

> Read this top-to-bottom to get the full picture. Numbers here are preliminary and will be re-run on a
> unified pipeline (see §11). Live status is in `notes/results_report.md` + the memory files; this file is
> the narrative.

---

## 1. Motivation
EEG decoders fail to **generalize across subjects/sessions/devices**: a model trained on some subjects
degrades badly on a new one. The usual fix — **unsupervised domain adaptation** — needs 20–30 min of
target calibration, killing "plug-and-play" use. We target **domain generalization (DG)**: train on source
subjects, deploy on an unseen subject with **zero calibration** (strict inductive setting; rigorous protocol
= **leave-one-subject-out, LOSO**).

**Why most DG fails on EEG:** methods align *marginal* feature distributions `P(Z)` (DANN, MMD, CORAL).
But in EEG the nuisance (subject "style", electrode impedance, anatomy) is **entangled with the label**.
Removing *all* domain info also removes label-relevant signal → "negative transfer". The correct target is
**conditional invariance**: a representation `Z` that predicts the task `Y` but is **conditionally
independent of the domain `D` given `Y`** — i.e. `I(Z;D|Y) ≈ 0`.

---

## 2. Core method — LPC-CMI (Label-Prior-Corrected Conditional Mutual Information)
We minimize a **consistent variational plug-in estimator** of the conditional mutual information:

```
I(Z;D|Y)  =  E_{z,y} KL( p(D|z,y) ‖ p(D|y) )
R_CMI(ψ)  =  E_{z,y} KL( q_ψ(D|z,y) ‖ π_y(D) )          # our estimator
```

- `q_ψ(D|Z,Y)` — a small learned variational posterior (MLP head), trained (Step A) to track `p_θ(D|Z,Y)`.
- `π_y(D) = p(D|Y=y)` — the **label-conditional domain prior** (the "LPC" correction): the target is `p(D|Y)`,
  not uniform — what makes the estimator *conditional*.
- **NOT a generic upper bound.** `R_CMI` equals the true `I_θ(Z;D|Y)` **exactly at Step-A convergence**
  (`q_ψ → p_θ(D|Z,Y)`); with an under-trained `q_ψ` it can *under*-estimate. So it is a *consistent plug-in
  surrogate* (consistency theorem to be stated), and the in-loop-KL vs frozen-probe-KL gap we log is exactly
  the `q_ψ`-sufficiency check. (A genuine upper bound is available via the conditional-CLUB log-ratio form if needed.)

**Two-step alternating trainer** (`cmi/train/trainer.py`):
- **Step A:** fit `q_ψ` on **detached** `Z` (so it tracks the current encoder), `n_inner` steps.
- **Step B:** update encoder + task head with `CE(logits,Y) + λ_t · E[KL(q_ψ(D|Z,Y) ‖ π_y(D))]`, with a
  λ warm-up.

**Key property — framework/architecture-agnostic:** the term acts only on the representation `Z`, so it
*adds to any framework's loss*. Implemented hosts: `lpc_prior` (ERM+CMI), `lpc_supcon` (supervised-contrastive+CMI),
`lpc_simclr`/`lpc_byol` (self-supervised+CMI), and `graphcmi` (GNN node/edge CMI, §10).

**Method variants (for ablation):** `marginal` = `I(Z;D)` (no Y), `chain`/FMCA = minimize `I(Z;(D,Y))`
(super-label), `lpc_uniform` = KL to uniform (no π_y), `lpc_prior` = KL to `π_y` (full method).

---

## 3. The regime distinction that drives everything: MCPS vs SCPS
- **MCPS (Multi-Class-Per-Subject):** each subject has all classes (motor imagery, emotion). Label ⟂ subject
  is *weak* → marginal vs conditional barely differ, and `π_y ≈ uniform`.
- **SCPS (Single-Class-Per-Subject):** each subject ≈ one label (disease detection: a patient is
  Alzheimer *or* control). Here **subject ≈ label**, so removing domain marginally *erases the label* —
  this is where **conditional** `I(Z;D|Y)` and the **π_y correction** matter most.

---

## 4. Codebase map (`cmi/`)
- `data/` — loaders, all returning `(X[B,C,T] float32, y, meta{subject,session}, classes)`:
  `moabb_data.py` (motor imagery via MOABB), `emotion_data.py` (SEED/SEED-IV/DEAP[+_valence/_arousal/_quadrant]),
  `diagnosis_data.py` (ADFTD/ADFTD_bin/MUMTAZ/TUAB), `processed_data.py` (datalake epoched MI),
  `geometric.py` (covariance→tangent), `cross_dataset.py` (Protocol C, 21-ch CANON), `alignment.py` (EA/RA/HA + bandpass).
- `models/backbones.py` — `build_backbone(name,...) -> nn.Module` with `forward(x)->(logits,Z)` + `.z_dim`.
  Backbones: EEGNet, ShallowConvNet, Deep4Net, EEGConformer (braindecode, faithful + forward-hook for Z),
  LogCov (covariance→tangent→MLP), TSMNet (SPDNet, GPU-patched), **GraphCMI** (§10). `models/gnn.py` = GraphCMINet.
- `methods/` — `regularizers.py` (DomainPosteriors, empirical/effective priors, the CMI reg),
  `contrastive.py` (SupCon), `augment.py` (SimCLR/BYOL + EEG augs), `dg_penalties.py` (CORAL/MMD/IRM/VREx/cHSIC,
  DANN/CDANN), `fmca.py` (route-2 FMCA), `graph_regularizers.py` (NodePosterior/EdgePosterior).
- `train/trainer.py` — `train_model(...)` the two-step loop; `ALL_METHODS` gates valid methods (fails loud).
- `eval/metrics.py` — classification metrics, `leakage_probe` (frozen-encoder residual `I(Z;D|Y)`),
  `label_separability`; `eval/leakage_audit.py` (multi-probe estimator audit).
- `run_loso.py` (Protocol A/B), `run_cross_dataset.py` (Protocol C), `run_classical.py` (pyRiemann baselines),
  `run_audit.py` (leakage audit). SLURM in `scripts/` (`run.slurm` GPU, `cpu.slurm` CPU).
- `synthetic/` — controlled DGP sanity check + proxy validation. `notes/` — design docs. `repos/TSMNet` — vendored SPDNet.

---

## 5. Backbones tested
| backbone | type | note |
|---|---|---|
| EEGNet | compact CNN | the workhorse; most experiments |
| ShallowConvNet / Deep4Net | CNN (braindecode) | architecture-agnostic check |
| EEGConformer | CNN+Transformer | **CDANN collapses here; lpc_prior stable** |
| LogCov | covariance→tangent + MLP | non-neural geometric carrier |
| TSMNet (SPDNet) | Riemannian | **baseline, NOT a carrier** (lpc collapses it); GPU-patched |
| **GraphCMI** | raw-signal GNN (our) | new (§10); node=channel, per-sample adjacency, node/edge CMI |
| **DGCNN / RGNN** | raw-signal GNN baselines | re-implemented (ChebNet/SGC + *shared* adjacency) for apples-to-apples |

---

## 6. Frameworks / methods (all in one harness)
ERM; **ours** (`lpc_prior`, `lpc_uniform`, `marginal`, `chain`); penalties (CORAL, MMD, IRM, VREx, GroupDRO,
cHSIC); adversarial (DANN, CDANN); IIB; **supervised contrastive** (SupCon) + hybrid `lpc_supcon`;
**self-supervised contrastive** (SimCLR, BYOL) + hybrids `lpc_simclr`/`lpc_byol`; **route-2 FMCA**
(`fmca_chain`/`fmca_diff`/`fmca_strat`); **GNN** (`graphcmi`).

---

## 7. Protocols & datasets
- **Protocol A** = LOSO (cross-subject). **B** = cross-session (leave-one-session). **C** = cross-dataset
  (leave-one-dataset, 21-ch CANON intersection). **D** = synthetic DGP (controlled).
- **Motor imagery (MCPS):** BNCI2014_001 (2a, 4-cls), BNCI2014_004 (2b, binary), Lee2019, Cho2017,
  Schirrmeister2017 (HGD), Stieger2021, PhysionetMI.
- **Emotion (MCPS):** SEED (3-cls), SEED-IV (4-cls), DEAP (valence/arousal/quadrant).
- **Clinical (SCPS):** ADFTD (Alzheimer/FTD/Control), ADFTD_bin, MUMTAZ (depression), TUAB (TUH abnormal).
- **Classical baselines:** pyRiemann TS+LR / MDM / CSP+LDA on all datasets.

---

## 8. Primary results (preliminary — full table in `notes/results_report.md`)
**The rock-solid pillar — diagnostic (leakage):**
- `lpc_prior` cuts conditional leakage **10–100×** on *every* Euclidean backbone, *every* task, while
  **preserving label separability** — and it **beats both competitors at leakage removal**: CDANN
  (adversarial, *unstable* — collapses to chance on EEGConformer) and cHSIC (kernel; *fails* on SCPS).
- Even tuned, marginal baselines can't match: MMD(λ=10) reaches the leakage but does it *marginally*
  (no Y-protection); IRM/VREx are unstable or hurt accuracy.

**Accuracy is λ-sensitive (not a fixed-λ "win"):** with **source-selected small λ (≈0.05–0.1)**,
`lpc_prior ≈ ERM-or-better AND removes leakage` (e.g. 2b lpc_prior:0.05 = 69.2 > ERM 68.6, 6× less leakage;
beats CDANN by +3–5 on Lee2019/Cho2017). Too-large λ trades accuracy for more removal.

**SCPS (the conditional regime) — now 4/4** support "lpc_prior ≥ ERM + leakage removal":
ADFTD-3cls (+up to +7.5 on Deep4Net; +6.5 EEGNet with subject-π_y; **but seed-sensitive — see findings**),
**TUAB (65.0>62.5, +35× leakage cut, clean π_y: lpc 65.0 > uniform 60.0)**, ADFTD_bin (83.4≥83.0 @λ0.1),
**MUMTAZ (89.3>85.4 @λ0.5, 70× cut)** — the earlier MUMTAZ "loss" was the buggy-sampler run, resolved.
The **ADFTD λ-sweep is the headline figure**: as λ grows, `lpc_prior` is λ-robust (peaks, label preserved) while
`marginal`/`dann` **catastrophically erase the label** (labelSep 85→56, acc craters).

**Accuracy pivot — where we actually beat baselines (the honest, winnable claims):**
1. **Worst-subject robustness** (DG's true metric): at small λ, lpc_prior > ERM on worst-subject (2a +3.1, 2b +1.2). λ-dependent → pairs with constrained-λ selection.
2. **Harmful-shortcut regimes (SCPS):** the 4/4 wins above — removing the subject shortcut genuinely helps OOD.
3. **vs other DG methods:** we beat the ones that *hurt* (IRM/VREx unstable; CDANN collapses on transformers; DANN/CDANN lose on MI accuracy). On MCPS *mean*, IIB/CDANN can edge us — a pure regularizer does **not** beat ERM on balanced-MI mean (DomainBed null).
4. **EA is a *transductive*, orthogonal booster, NOT a CMI win:** EA lifts 2a ERM 43.2→48.8, but **strict (source-only) EA gives nothing** (`ea_strict` 41.8 < none 43.2) — the entire EA gain comes from using the target's unlabeled trials → "zero-*label* calibration", not strict DG. And on top of EA, CMI adds **worst-case** (not mean) robustness. So EA results must be labeled transductive; the mean-accuracy win is *not* "EA+CMI > EA+ERM".

**Synthetic (Protocol D):** `lpc_prior` uniquely drives leakage→0 while preserving acc+label; `marginal`=label-erase,
`chain`=Y-erase, `uniform`=mis-specified. Leakage proxy **validated** vs independent kNN `Î(Z;D|Y)` (Pearson r=0.85).

**Leakage-estimator audit (trustworthy):** multi-probe ensemble (linear/MLP/RF/HGBM/HSIC/kNN) agrees;
permutation-null ≈ 0 → not a single-probe artifact. *But*: lpc_prior removes the *training-induced* leakage
back toward the **random-encoder floor** — it cannot remove irreducible subject info without hurting the task
(no-free-lunch); and a *marginal* method (CDANN on ADFTD) can remove *more* leakage but only by destroying the label.

---

## 9. Key findings (honest)
1. **Diagnostic story is strong and robust**; accuracy on balanced MI is a *wash* (field-wide null) — so the
   honest accuracy claim is "**no cost at properly-selected λ + large, label-preserving leakage removal**",
   not "beats ERM everywhere".
2. **Conditional > marginal is demonstrable on real data** (the ADFTD λ failure curve; SCPS).
3. **π_y correction is regime-dependent:** proven in synthetic, clean on TUAB (+5 vs uniform), weak on balanced MI.
4. **Architecture-agnostic claim is scoped** to Euclidean encoders + LogCov; **SPDNet is a baseline not a carrier**
   (it collapses under the penalty).
5. **A bug we found & fixed:** a `(class,domain)`-balanced sampler was *uniformizing* `p(D|Y)`, undermining π_y;
   `classbal` (class-only) is the correct default.
6. **Route-2 FMCA chain-rule is a documented failure** (see §10) — a useful ablation, not a competitor.
7. **The winnable accuracy claim** (goal-pivot 2026-06-08) is **worst-subject + SCPS/hard-shift wins**, not balanced-MI mean. Verified: worst-subject lpc_prior>ERM at small λ; SCPS 4/4; but **ADFTD mean is seed-sensitive** (seed0 +3.7, seed1 −5.0, seed2 −0.3) → ADFTD is ~parity across seeds; trust TUAB/MUMTAZ + worst-case + the multi-seed CIs.
8. **Reviewer rebuttals settled empirically:** (a) **subject-balanced π_y holds/improves** the result (ADFTD +6.5, π_y>uniform +2.7) — not a segment-imbalance artifact; (b) **EA's gain is 100% transductive** (`ea_strict` ≤ none) — strict-DG EA gives nothing, must be labeled zero-label-calibration; (c) α-smoothing already in `empirical_priors` (Laplace α=1); (d) leakage probe is already frozen-encoder + separate-probe + separate-split. Theory fix applied: the estimator is a **consistent plug-in (exact at Step-A convergence), not an upper bound**.

---

## 10. New designs / extensions
- **Route 2 — FMCA + chain-rule** (`cmi/methods/fmca.py`, `notes/route2_fmca*.md`): the old-draft alternative
  (estimate `I(Z;D|Y)=I(Z;(D,Y))−I(Z;Y)` via the FMCA spectral estimator). **Verdict: appendix ablation only.**
  Naive `fmca_chain` reproduces **Y-erasure** (labelSep 59→39 on 2a), collapses SPDNet to chance; strictly
  dominated by `lpc_prior`. Deep reason: FMCA is calibrated MI only in the Gaussian case → chain-rule isn't
  additive in nats. Confirms *why* conditioning-on-Y (route 1) is right.
- **Self-supervised contrastive** (`cmi/methods/augment.py`): SimCLR/BYOL as a distinct framework, + `lpc_simclr`/
  `lpc_byol` hosting our CMI term (answers "does CMI plug into SSL too?" — yes, it's a loss-term add).
- **Alignment** (`cmi/data/alignment.py`, `notes/hyperbolic_alignment.md`): **EA** (Euclidean recentering,
  mainline) / **RA** (Riemannian/AIRM geometric-mean — SPD-AIRM is already hyperbolic-like, SPD(2)≅ℝ×ℍ²) /
  **HA** (Poincaré recentering, exploratory). Accuracy-pipeline only (kept OFF the raw leakage probe).
  The promising hyperbolic angle is *domain-hierarchy* (`dataset⊃subject⊃session`) for `π(D)`/`q(D|Z,Y)`, not trial alignment.
- **GNN — GraphCMINet** (`cmi/models/gnn.py`, `cmi/methods/graph_regularizers.py`, `notes/gnn_design.md`):
  the **most thematically-coherent host** — EEG domain-leakage lives in *connectivity*, which a GNN models
  explicitly. Raw-signal node encoder (LGGNet PowerLayer, no DE) → per-sample **learnable adjacency** → SGC →
  readout, exposing `(logits, graph_Z, node_Z, edge_logits)`. The method becomes
  `CE + λ·I(graph_Z;D|Y) + λ_node·Σ_v I(Z_v;D|Y) + λ_edge·I(A;D|Y)` (config `graphcmi:λ:λ_node:λ_edge`).
  Node = EEG channel (C nodes); DGCNN/RGNN baselines re-implemented for apples-to-apples (shared adjacency).
  **Novelty — verified against a recent EEG-GNN lit-search (`notes/recent_eeg_gnn.md`):** our
  **non-adversarial, label-conditional domain-MI at BOTH node AND the per-sample learned adjacency** is
  genuinely unclaimed. Must-cite near-misses: **RGNN-NodeDAT** (per-node but *adversarial+unconditional*, fixed
  graph), **GDDN** (TAFFC'24 — the nearest *edge* prior: disentangles the adjacency, but by *decomposition not
  conditional-MI*, DE input), **FreqDGT** (closest competitor — adversarial+marginal+graph-level). The **edge
  term `I(A;D|Y)` is the cleanest novelty** (the field uses per-sample adjacency but nobody regularizes it
  against subject leakage). The **node leakage map** (length-C "which electrodes leak subject identity") is a
  figure NodeDAT can't produce. **Two corrections from verification:** BrainIB is **fMRI, not EEG** (scaffold
  only); the whole SEED GNN line (DGCNN/RGNN/FreqDGT/GDDN/MoGE) uses **DE band-power features** — we use raw, so
  accuracy comparisons are *method-level* (flag DE-vs-raw), and we **headline the leakage maps + generalization
  gap**, not raw-accuracy SOTA (raw may trail DE-based RGNN 85.3 / MoGE 88.0 before CMI gains).

---

## 11. Preprocessing policy (decided after a lit search — `notes/preprocessing_decision.md`)
- Current results are **preliminary**; a **unified re-run** (preprocessing + final MI-estimator) is planned.
- **Leakage/diagnostic story stays on RAW signal** — per-subject normalization itself drives a subject-ID
  probe to chance (Fdez 2021), i.e. it would *do the regularizer's job* and confound the metric.
- MI epoched via **MOABB @250 Hz** (the lmdb `_250Hz_6s` store is a fixed-6s pretraining format — 6 s doesn't
  fit 2a's ~4 s MI — usable only for an FM baseline); clinical/emotion 200 Hz from datalake `processed/5e77943a`.
- **Monopolar** everywhere for our method + covariance/SPD arm (bipolar TCP is rank-deficient → breaks SPD);
  bipolar only for a separate CBraMod/BIOT baseline. Task-keyed windows. Dual-band (encoder 4–40 / cov 8–30).
  EA for calibration-free transfer. **No DE features (raw only).** DEAP = both binary and 4-class quadrant.
- **Engineering:** every run now saves per-fold probabilities (`*.preds.npz`) so ECE/NLL/new metrics never
  need a GPU retrain.

---

## 12. Future plan
**Accuracy pivot (user goal 2026-06-08): must beat other methods on accuracy, not just leakage.** Four
battlegrounds being pursued in parallel:
1. **Worst-subject robustness** (already winning at small λ) + **constrained-λ source-selection** (≥ ERM by construction).
2. **Hard-shift regimes** (SCPS 4/4 + cross-dataset) — multi-seed CIs to make the SCPS wins robust (ADFTD is seed-sensitive).
3. **GNN beats DGCNN/RGNN** on SEED/DEAP — benchmark accuracy win (running); node/edge ablation + NodeDAT comparison.
4. **EA+CMI** — *transductive* mean-accuracy variant (clearly labeled zero-label-calibration).

Then: **decide the final MI-estimator** (posterior-KL vs constrained-λ) → **unified re-run** on the locked
preprocessing. **Figures:** ADFTD λ failure-curve, leakage–accuracy Pareto, worst-subject, node leakage map,
ECE/NLL. **Hyperbolic domain-hierarchy** prior as an optional geometric contribution.

---

## 13. How to run
```bash
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python   # env: icml (torch 2.8, braindecode 0.8, PyG 2.6.1)
# LOSO on 2a, our method vs baselines:
$PY -m cmi.run_loso --dataset BNCI2014_001 --backbone EEGNet \
   --configs erm:0 lpc_prior:0.1 cdann:1 --resample 250 --tmin 0.5 --tmax 3.5 --epochs 200 --out results/x.json
# GNN node/edge CMI on SEED:
$PY -m cmi.run_loso --dataset SEED --backbone GraphCMI \
   --configs erm:0 graphcmi:0.3:0.3:0.3 --epochs 200 --out results/seed_gnn.json
# SLURM: sbatch -p V100 scripts/run.slurm <same args>   (A100 24h; V100/A40 4day; CPU partitions for classical)
```
Key flags: `--align {none,ea,ra,ha}`, `--sampler {classbal,raw,domainbal}`, `--protocol {loso,cross_session}`,
`--imbalance ρ`, `--fmin/--fmax`. Methods are validated against `ALL_METHODS` (unknown → hard error).

**Pointers:** narrative = this file; numbers = `notes/results_report.md`; design docs = `notes/*.md`;
live status + conventions = the memory files (`.claude/.../memory/cmi-*.md`).
