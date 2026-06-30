# Claim → evidence table (the wording contract)

Every paper claim maps to one block below. Use **allowed wording**; never the **forbidden wording**.
This file is authoritative for over-claim prevention. No new compute — all evidence is existing artifacts.

Global forbidden phrasings (apply everywhere):
`TOS-CMI improves EEG domain generalization` · `Selective deletion solves LPC collapse` ·
`Domain leakage removal is sufficient for target transfer` · `Task-orthogonal geometry implies safe deletion` ·
`EEG subject leakage is generally low-rank removable` · `global CMI always collapses` ·
`the dim↔type confound is resolved` · `the contrast is capacity-mediated, not architecture-type` (bare, forbidden until 3-seed + EEGNet d_z=210) · `frozen erasure yields no target gain` (forbidden until the Step-3 target-deployment test is run).

---

### C1 — Measurement: score-Fisher localizes conditional domain leakage
- **Claim:** score-Fisher / projection diagnostics localize conditional domain leakage I(Z;D|Y) in EEG latent spaces (a domain-rich, task-light low-rank subspace exists and is identifiable).
- **Phase:** 1 (synthetic) + 2.0/3.0 (EEG). **Fig/Table:** Fig 1, Fig 2, Fig 4/5.
- **Artifacts:** `tos_cmi/results/cert_cells`, `frontier.json`; `.../BNCI2014_001_{TSMNet,EEGNet}_LOSO/ablation_report_seed*.json` (nDcand, candidate selectivity).
- **Limitation:** "localize" = identifies a candidate subspace via a score-Fisher *proxy*, not exact CMI.
- **Allowed:** "score-Fisher localizes candidate conditional-leakage subspaces." **Forbidden:** "measures the exact conditional mutual information."

### C2 — Geometry alone is insufficient (direct-sum necessary, not sufficient)
- **Claim:** task-orthogonal / direct-sum geometry (RV=0, RT=T) is necessary for algebraic task preservation but NOT sufficient for conditional task safety; a task-risk gate is required.
- **Phase:** 1.2–1.3 (synthetic synergy generators). **Fig/Table:** Fig 2 (panel 2).
- **Artifacts:** `tos_cmi/notes/PHASE13_DIAGNOSIS.md`; `tos_cmi/results/cert_cells`.
- **Limitation:** shown on synthetic Gaussian-mixture generators with known Bayes oracle.
- **Allowed:** "direct-sum geometry is necessary but not sufficient for conditional task safety." **Forbidden:** "task-orthogonal geometry implies safe deletion."

### C3 — Certification: weak gate unsafe-accepts; plug-in improves; certified gate abstains (honest negative on default-on)
- **Claim:** a weak nested critic unsafe-accepts conditionally-unsafe deletions; a cross-fitted plug-in log-ratio estimator improves; even so, certified default-on deletion is not achieved at moderate n — the correct behavior is conservative abstention.
- **Phase:** 1.3.x (PHASE131). **Fig/Table:** Fig 2 (panels 3–4).
- **Artifacts:** `tos_cmi/results/{estimator_diag.json, frontier_plugin, frontier_deploy, phase_diagram_powerfloor.json}`; `tos_cmi/notes/PHASE131_CERTIFICATION.md`.
- **Limitation:** synthetic; the negative is about *default-on certified* deletion, not about the measurement chain.
- **Allowed:** "certified deletion often abstains; weak gates can unsafe-accept." **Forbidden:** "our gate certifies safe deletion" / "selective deletion is always safe."

### C4 — TSMNet leakage is high-dimensional / redundant → low-rank deletion insufficient
- **Claim:** in the TSMNet LogEig latent, subject identity is ≈perfectly decodable; deleting the low-rank V_D preserves the task but barely reduces subject decode (≈ random-k) — leakage is redundantly encoded across the latent.
- **Phase:** 2.0. **Fig/Table:** Fig 4, Table 1.
- **Numbers:** subject Z=0.997 → RZ≈0.95 (MLP), random-k≈0.997; task 0.75→0.75; full 7-dim Fisher deletion still ≈0.92–0.98.
- **Artifacts:** `.../BNCI2014_001_TSMNet_LOSO/ablation_report_seed{0,1,2}.json`; PHASE2_REPORT.md.
- **Limitation:** within the LDA cap (8 source subjects → 7 Fisher directions); single dataset.
- **Allowed:** "low-rank selective deletion is insufficient in the high-dimensional TSMNet/2a latent." **Forbidden:** "EEG subject leakage is generally low-rank removable" / "deletion removes the leakage."

### C5 — TSMNet global LPC removes leakage only via representation collapse
- **Claim:** raw global LPC reduces subject leakage at high λ only by a sharp, λ-tied objective-scaling bifurcation to feature-norm collapse at the origin (Z→0); the de-domaining is a collapse artifact.
- **Phase:** 2.1. **Fig/Table:** Fig 3.
- **Numbers:** λ≥1 → src/label→chance, feat_norm 1.09→0.00, top-1 SV→0.001, penalty→~0; NOT a gradient explosion (abs peak grad ~10× smaller than healthy; 0/36 non-finite); eff_rank "stays high" is scale-invariant (non-probative).
- **Artifacts:** `.../lpc_collapse_curves/TSMNet/{collapse_curves.png, summary.json}`; PHASE21_LPC_COLLAPSE_MECHANISM.md.
- **Limitation:** TSMNet/2a, folds {1,5,9}, 3 seeds, 4 λ; grad is a between-epoch diagnostic proxy.
- **Allowed:** "in the tested TSMNet/2a setting, global LPC removes subject leakage only through representation collapse." **Forbidden:** "global CMI always collapses" / "LPC is unstable in general."

### C6 — The collapse is fixable (optimization, not geometry); collapse-free LPC removes nothing on TSMNet
- **Claim:** warm-up scheduling (and scale-normalization at λ=1) prevents the collapse → it is an objective-scaling optimization pathology, not a geometric necessity; but every collapse-free, task-preserving global LPC leaves subject leakage at ERM levels (removes ~0).
- **Phase:** 2.2. **Fig/Table:** Fig 3 (companion), Table 1.
- **Numbers:** warm_ramp avoids collapse at λ=1 & 3 (0/9, 0/9); scale-invariant at λ=1 (0/9) not λ=3 (9/9); in all collapse-free cells subj_dec = 0.997 ≈ ERM 1.00.
- **Artifacts:** `.../lpc_collapse_curves/TSMNet/variant_compare.json`; PHASE22_LPC_OBJECTIVE_SCALING_ABLATION.md.
- **Limitation:** TSMNet/2a; variants are flag-gated ablations, not a proposed method.
- **Allowed:** "preventing the collapse shows global LPC's apparent de-domaining was a collapse artifact (TSMNet/2a)." **Forbidden:** "Selective deletion solves LPC collapse" / "we fixed LPC."

### C7 — Low-rank removability is representation-dependent (EEGNet vs TSMNet)
- **Claim:** on the compact EEGNet latent the same score-Fisher deletion removes a large fraction of subject leakage with negligible task cost — genuine V_D selectivity, not a delete-many-dims artifact — whereas on TSMNet it does not. Removability is representation/capacity-dependent.
- **Phase:** 3.0 (verified wf_cb3b4958). **Fig/Table:** Fig 5, Table 1.
- **Numbers:** EEGNet subject linear 0.82→0.35 (random-k 0.73), MLP 0.88→0.54 (random-k 0.81); selectivity 0.35–0.55 vs TSMNet 0.04–0.08; task 0.64→0.64.
- **Limitation:** **dim↔type confound** — EEGNet differs in architecture AND latent dim (16 vs 210); collinear in the two-point comparison. The Track-C architecture×dimension factorial (**C11**) PROVISIONALLY attributes the contrast to capacity (latent dim) rather than type, but this is seed-0 (multi-seed + EEGNet d_z=210 pending); do NOT write “resolved” yet.
- **Allowed:** "low-rank removability is representation-dependent and capacity-mediated." **Forbidden:** "convolutional (SPD) representations are (un)removable" / asserting type as the causal axis.

### C8 — Removability is partial: nonlinear residual persists
- **Claim:** even on EEGNet, low-rank deletion removes the linear subject leakage (~67%) but a substantial nonlinear residual remains (RZ_mlp ≈ 0.54 ≫ chance 0.125).
- **Phase:** 3.0. **Fig/Table:** Fig 5 (panel a).
- **Artifacts:** `.../BNCI2014_001_EEGNet_LOSO/ablation_report_seed*.json` (domain_RZ_linear vs _mlp).
- **Limitation:** MLP probe is one nonlinear family; residual is probe-dependent.
- **Allowed:** "linearly reducible, nonlinear residual persists." **Forbidden:** unqualified "removable" / "eliminates the leakage."

### C9 — Removable ≠ beneficial: removing leakage gives no DG gain (EEGNet)
- **Claim:** on EEGNet, global LPC reduces subject leakage without collapse (0.89→0.19) but target/LOSO accuracy is flat-to-worse (0.43→0.39, paired-t p≤0.012) and uncorrelated with leakage reduction (corr −0.14, n.s.).
- **Phase:** 3.0 (verified). **Fig/Table:** Fig 5 (panels b,c), Table 1.
- **Artifacts:** `.../lpc_collapse_curves/EEGNet/raw_lpc_sub*_seed*.json` (tgt, subj_dec vs λ).
- **Limitation:** frozen-feature pilot — shows leakage *removal per se* does not buy DG, not that end-to-end training cannot; EEGNet is a weak/high-variance backbone on 2a LOSO.
- **Allowed:** "leakage removal can be real but still not improve target accuracy (EEGNet/2a)." **Forbidden:** "Domain leakage removal is sufficient for target transfer" / "TOS-CMI improves EEG domain generalization."

### C10 — Unified thesis: measurement→control gap
- **Claim:** conditional domain leakage is a measurable property of EEG representations but not a sufficient control target for cross-subject generalization; safe selective invariance should be a certified intervention with refusal, not an always-on regularizer.
- **Phase:** 1+2+3 synthesis. **Fig/Table:** Table 1, all figures.
- **Artifacts:** PHASE2_REPORT.md, PHASE3_BACKBONE_GENERALITY.md, this table.
- **Limitation:** 2a only, 2 backbones; the causal-irrelevance claim is "weak/representation-dependent," not "no relation."
- **Allowed:** "conditional domain leakage is measurable but not a sufficient control target for DG on 2a." **Forbidden:** "domain leakage is irrelevant to generalization" (too strong) / "our method generalizes EEG models."

---

### C11 — Dimension-vs-type (Track C capacity factorial; PROVISIONAL, seed-0)
- **Claim:** an architecture×latent-dimension factorial (TSMNet tangent d_z∈{21,36,55,105,210} via SPD size; EEGNet width F2∈{16,32,64,128}; a matched d_z=210 conv cell in progress) shows the nonlinear erasure residual rises with d_z *within* both architectures and the two nearly coincide at matched d_z — suggesting the TSMNet-vs-EEGNet removability gap is mediated by latent dimension, not SPD-vs-conv type.
- **Phase:** Track C. **Fig/Table:** Fig 6 (§4.5).
- **Numbers (seed-0):** matched d_z 16/21: 0.40/0.40; 32/36: 0.50/0.50; 105/128: 0.65/0.61; TSMNet d_z=210: 0.74.
- **Status:** PROVISIONAL (seed-0). Final verdict gated on 3-seed (running) + EEGNet d_z=210 matched cell + fold-cluster bootstrap CI + paired matched-dim contrast + OLS coefficient CIs.
- **Allowed:** “a seed-0 factorial indicates the contrast is driven largely by latent dimension (provisional)” / “capacity-consistent, pending multi-seed confirmation.” **Forbidden:** “the dim↔type confound is resolved” / “capacity-mediated, not architecture-type” (bare, until 3-seed + EEGNet-210 complete).

---

## Dim↔type confound — exact limitation wording (use verbatim)
> EEGNet differs from TSMNet in both architecture and latent dimensionality; therefore Phase 3 does not
> isolate whether low-rank removability is driven by convolutional inductive bias, latent compression, or
> both. The two-point comparison establishes representation dependence, not the causal factor behind it;
> the Track-C capacity factorial (C11) PROVISIONALLY attributes it to latent dimension (seed-0; multi-seed
> and a matched d_z=210 convolutional cell pending), so it must NOT be written as “resolved” until then.

## Cross-check before camera-ready
- [ ] Every figure caption number traces to a row here.
- [ ] No global-forbidden phrasing in abstract/intro/conclusion (grep).
- [ ] Table 1 numbers == C4/C5/C7/C9 numbers.
- [ ] Every "in general" / "always" claim is scoped to "TSMNet/2a" or "EEGNet/2a".
