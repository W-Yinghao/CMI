# Claim → evidence table (the wording contract)

Every paper claim maps to one block below. Use **allowed wording**; never the **forbidden wording**.
This file is authoritative for over-claim prevention. No new compute — all evidence is existing artifacts.

Global forbidden phrasings (apply everywhere):
`TOS-CMI improves EEG domain generalization` · `Selective deletion solves LPC collapse` ·
`Domain leakage removal is sufficient for target transfer` · `Task-orthogonal geometry implies safe deletion` ·
`EEG subject leakage is generally low-rank removable` · `global CMI always collapses` ·
`the dim↔type confound is resolved` · `the contrast is capacity-mediated, not architecture-type` (empirically REFUTED at high d_z — permanently forbidden) · `a pure dimension effect` (a residual architecture×dimension interaction remains) · `erasure improves target domain generalization` (Step-3 deployment: NO eraser improves target bAcc) · attributing the deployment NLL movement to domain removal (it is matched by same-k random removal).

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
- **Limitation:** **dim↔type confound** — EEGNet differs in architecture AND latent dim (16 vs 210); collinear in the two-point comparison. The Track-C architecture×dimension factorial (**C11**, 3-seed) shows the contrast is LARGELY capacity (latent dim) — coincident at low d_z, ~2/3 of the gap — but with a RESIDUAL architecture effect at high d_z; do NOT write “resolved” or “not architecture-type”.
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
- **Allowed:** "leakage removal can be real but still not improve target accuracy (EEGNet/2a)"; the frozen-erasure target-deployment test (**C12**, Table 3) confirms this directly. **Forbidden:** "Domain leakage removal is sufficient for target transfer" / "TOS-CMI improves EEG domain generalization."

### C10 — Unified thesis: measurement→control gap
- **Claim:** conditional domain leakage is a measurable property of EEG representations but not a sufficient control target for cross-subject generalization; safe selective invariance should be a certified intervention with refusal, not an always-on regularizer.
- **Phase:** 1+2+3 synthesis. **Fig/Table:** Table 1, all figures.
- **Artifacts:** PHASE2_REPORT.md, PHASE3_BACKBONE_GENERALITY.md, this table.
- **Limitation:** 2a only, 2 backbones; the causal-irrelevance claim is "weak/representation-dependent," not "no relation."
- **Allowed:** "conditional domain leakage is measurable but not a sufficient control target for DG on 2a." **Forbidden:** "domain leakage is irrelevant to generalization" (too strong) / "our method generalizes EEG models."

---

### C11 — Dimension-vs-type (Track C capacity factorial; 3-seed, DONE)
- **Claim:** a 3-seed architecture×latent-dimension factorial shows the nonlinear erasure residual rises with d_z within BOTH architectures and the two coincide at low d_z, so latent dimension is the DOMINANT axis of the TSMNet-vs-EEGNet removability contrast (matching dimension removes ≈2/3 of the raw gap) — but NOT the only one: a residual architecture×dimension interaction persists, and at matched d_z=210 the SPD latent retains significantly more recoverable subject identity than conv.
- **Phase:** Track C (SLURM 877939). **Fig/Table:** Fig 6 (§4.5).
- **Numbers (3 seeds × 9 folds, fold-cluster 95% CI):** LEACE residual TSMNet 21/36/55/105/210 = 0.397/0.498/0.559/0.648/0.740; EEGNet 16/32/64/128/210 = 0.393/0.507/0.574/0.609/0.628. Matched-dim (TSMNet−EEGNet): 21v16 +0.004[−0.008,0.014] (overlaps 0); 36v32 −0.008[−0.022,0.004] (overlaps 0); 55v64 −0.015[−0.024,−0.004]; 105v128 +0.039[0.028,0.051]; 210v210 +0.111[0.094,0.125] (excludes 0). OLS log(d_z) +0.089[0.086,0.092]; interaction +0.058[0.051,0.063].
- **Status:** SUPPORTED (refined). Multi-seed + EEGNet d_z=210 complete. Remaining caveat: single dataset (2a), LDA cap from 8 source subjects.
- **Allowed:** “largely capacity-mediated, with a residual architecture effect at high d_z (3 seeds, fold-cluster CI)” / “matching dimension removes ~2/3 of the gap; a residual architecture×dimension interaction remains.” **Forbidden:** “the dim↔type confound is resolved” / “capacity-mediated, not architecture-type” (empirically REFUTED at high d_z) / “a pure dimension effect.”

### C12 — Frozen-erasure target deployment: no accuracy gain (Step 3)
- **Claim:** deployed on the held-out target (source-only fit; NO target selection/calibration/tuning), no eraser (LEACE/RLACE/TOS-V_D/INLP) improves target balanced accuracy on either backbone; INLP collapses the task; the only NLL movement is non-specific (same-k random removal reproduces it).
- **Phase:** Step 3 (SLURM 878002). **Fig/Table:** Table 3.
- **Numbers (3 seeds × 9 folds; paired fold-cluster 95% CI; ΔbAcc vs full Z):** TSMNet LEACE +0.001[−0.004,0.005], RLACE −0.004[−0.006,−0.002], TOS −0.000, INLP −0.062 (src task 0.749→0.533); EEGNet LEACE −0.011[−0.021,−0.002], RLACE −0.012, TOS −0.000, INLP −0.160 (=chance). NLL: TSMNet LEACE ΔNLL −0.031 vs random −0.034 (matched; random subj-decode 0.998 = not erased).
- **Status:** SUPPORTED. **Allowed:** "deployed on the target, no eraser improves target accuracy; the NLL movement is reproduced by random removal (non-specific)." **Forbidden:** "erasure improves target DG" / attributing the NLL blip to domain removal / calling INLP's NLL drop a benefit (it destroys the task).

---

### C13 — Multi-dataset target-deployment validation (branch tos, DONE)
- **Claim:** across four additional real EEG datasets (2b, Lee2019, Cho2017, High-Gamma; 129 subjects beyond 2a), no principled source-fitted eraser (LEACE/TOS/RLACE) yields a practically meaningful target-bAcc gain in any valid dataset-backbone cell; C12 CONFIRM on 9/9 valid cells. Erasure task-safety is heterogeneous: on binary Lee/Cho with EEGNet, LEACE/RLACE drive the source task to chance, so deployment actively HARMS the target.
- **Phase:** branch tos multi-dataset validation. **Fig/Table:** compact `tab:bigN_compact` (main) + `tab:bigN_full` (appendix).
- **Numbers (paired subject-cluster 95% CI):** Cho2017-TSMNet LEACE −0.001[−0.003,+0.000]; Lee2019-TSMNet −0.002[−0.003,+0.000]; High-Gamma-TSMNet −0.001[−0.005,+0.003]; Lee/Cho-EEGNet LEACE/RLACE −0.15..−0.19 (task→chance). All principled upper CIs < +0.01.
- **Status:** SUPPORTED. 2b-TSMNet excluded (degenerate 3-channel SPD metric). **Allowed:** "across additional real EEG datasets, source-fitted erasers do not yield practically meaningful target-bAcc gains in any valid dataset-backbone cell." **Forbidden:** "erasure never helps domain generalization" / "conditional invariance is useless" / "all backbones on all datasets were valid" / "2b confirms TSMNet" / "LEACE is task-safe across datasets".

### C14 — Method-deepening final verdict: real-EEG refusal + source-only acceptance ceiling (branch method-deepen-v1, tag tos-cmi-method-deepen-v2-final @ 0ef7be3)
Consolidates Track B (real-EEG gate), Phase 2 (task-preserving erasure), V2 (semi-synthetic ceiling). See notes/METHOD_DEEPENING_FINAL_VERDICT.md.
- **Track B (real EEG):** a benefit+safety source-only gate has **0 false accepts** and prevents 8/8 harmful erasures on Lee/Cho (sampled pilot); naive domain-gain/safety controllers false-accept harmful/useless erasures. Acceptance power UNTESTED on real EEG (no real positive).
- **Phase 2 (Lee/Cho EEGNet):** tp-LEACE preserves the deployed task decision (task-drop UCB +0.000) AND erases source subject (0.31→0.03), but target transfer stays FLAT (+0.000) → gate ABSTAINS. cc-predicted's exact-zero is a structural tautology (probe re-learns router); tp-LEACE is the clean result.
- **V2 (semi-synthetic ceiling, Stage 2 scoped: 72,000 tasks, 0 fail/degenerate):** EEGNet World A cleanly demonstrates a source-only acceptance ceiling, robust across source_subject_counts {8,16,32,all}×seeds (clean target-beneficial-but-uncertifiable cells at every n_source, 0 principled ACCEPT, oracle-supported, random-k LCB≤+0.006); World B/C robust refusal on BOTH backbones (0 ACCEPT); 0 false accepts across all 72,000 cells; naive controllers false-accept 1807-3368; oracle target-informed selector picks 284 true/0 false (proving the ceiling is source-only, not absence of benefit).
- **Status:** method-deepening evidence (semi-synthetic for V2). **Allowed:** "The source-only gate reliably refuses real EEG erasures that are useless or harmful." / "V2 demonstrates a source-only acceptance ceiling: target-beneficial deployment-shift erasures can exist, but strict source-only evidence cannot certify them when the benefit is not represented in source-domain validation." / "EEGNet World A cleanly demonstrates the ceiling; TSMNet World A is not cleanly demonstrable under the appended-nuisance construction." / "the source-only gate has safe refusal power and exposes an acceptance ceiling." / "crossing the ceiling requires target information or source domains encoding the shift." **Forbidden:** "The source-only gate has acceptance power." / "V2 shows the gate accepts genuine target-beneficial erasure." / "World A succeeds on all backbones." / "TSMNet World A cleanly confirms the ceiling." / "Conditional invariance never helps." (too strong) / "source-only benefit is impossible in general." (it is conditional non-identifiability) / "Target-informed oracle is a deployable method."

### C15 — Source-only acceptance ceiling THEORY (notes/CEILING_THEORY.md; theory spine for the current paper)
- **Claim:** three propositions formalizing the method-deepening result. **P1 (non-identifiability):** two worlds with identical source law but opposite target effect of the same erasure ⇒ any source-only policy that accepts in the beneficial world also accepts in the harmful/neutral world; to control false accepts uniformly it must abstain/reject in the beneficial world (V2 World A = witness). **P2 (source-rich sufficiency):** if the target shift is exchangeable with / in the convex hull of the source-ENVIRONMENT distribution, a calibrated source-LOEO lower bound is a conservative target-gain certificate up to a coverage error ε_coverage (explains Track B accept-when-source-visible + V2 abstain-when-source-invisible). **P3 (target-label sample complexity, NEXT-PAPER bridge):** n=O(ε⁻² log(1/δ)) per class to certify Δ_T(a)>ε with labeled target calibration.
- **Status:** theory note (spine for the manuscript; no new experiments). **Allowed:** "Strict source-only certification cannot license deployment-shift benefit that is not represented in source-domain variation." / "Refusal is the safe action when benefit is source-invisible." / "Source-rich environments or target information can break the ceiling." **Forbidden:** "Source-only acceptance is impossible in general." / "The gate has acceptance power." / "V2 proves no erasure can ever help." / "Target-informed oracle is a deployable method." / stating P3's target-label frontier as a current-paper result (it is the next paper).

### C16 — Source-rich environment discovery: constructive support with a representation-dependent limitation (Fork 2, branch science-source-rich-v1; DONE, PARTIAL @ 8174483)
- **Claim:** the constructive side of P2 (C15) tested by construction. In a semi-synthetic source-rich World A (real EEG latents + a KNOWN injected nuisance D_nuis=z whose target regime is REPRESENTED among source regimes {aligned/reversed/noisy}), a source-only leave-one-ENVIRONMENT-out benefit can safely license accept. **EEGNet (Lee→Cho, frozen params):** E_oracle (leave-one-regime-out) accepts safe target-beneficial erasers AND a source-only DISCOVERED environment (E2 covariance_cluster) recovers the acceptance (Lee oracle6/cov5, Cho oracle6/cov6), while random/margin/augmentation do not — a clean source-only positive witness. **TSMNet (z_dim=210, m=42, same frozen params):** the constructive route does NOT cleanly transfer — Lee E_oracle accepts (Case B) but E2 covariance benefit_lcb is NEGATIVE (max discovered lcb +0.0001 ≪ +0.010); Cho E_oracle is a SAFE target-beneficial threshold near-miss (source-LOEO benefit_lcb +0.0065–+0.0074 < +0.010 → abstain, yet target ΔbAcc +0.033–+0.041, LCB +0.021–+0.028, all 5 folds positive, bootstrap-stable); E2/E4/E5/random 0 accepts. covariance↔regime alignment collapses (subject-level AMI +0.365 EEGNet vs −0.011 TSMNet = chance) — the EEGNet discovery was a low-dim artifact. Gate stayed safe (0 harmful, 0 discovered-env accepts on both TSMNet datasets).
- **Phase:** Fork 2 Phase 1A/1B (EEGNet, 69ec85c) + 1C (TSMNet, 8174483). Adversarially verified (wf_650c9d1f-aec, high confidence, integrity clean). **Artifacts:** `notes/{SOURCE_RICH_FINAL_VERDICT,SOURCE_RICH_PHASE1A_VERDICT,SOURCE_RICH_PHASE1C_VERDICT}.md`; `results/source_rich/smoke/source_rich_{smoke,confirm_cho,tsmnet_lee,tsmnet_cho}_{summary.json,report.md}`.
- **Status:** semi-synthetic, PARTIAL SUCCESS (constructive positive on EEGNet; representation-dependent limitation on TSMNet). **Allowed:** "Source-rich environments can make target-beneficial erasure source-visible in a semi-synthetic EEGNet setting." / "Covariance environments recover the constructed source-visible shift on EEGNet but not on high-dimensional TSMNet." / "Source-rich sufficiency is constructive but environment discovery is representation-dependent." / "The source-only gate did not misfire on high-dimensional latents." **Forbidden:** "Source-rich environments solve source-only acceptance." / "Covariance clustering generally discovers target-beneficial shifts." / "TSMNet confirms the source-rich positive." / "This is a real EEG target-gain result." / "Cho-TSMNet is a negative / world-construction failure." (it is a safe conservative near-miss).

### C17 — Target-information frontier: labels reveal benefit but few-shot safe certification is sample-complexity limited (Fork 1, branch science-target-info-v1; DONE, PARKED, tag tos-cmi-target-info-frontier-v1-final)
- **Claim:** the next-paper bridge (C15 P3) tested by construction. In semi-synthetic Tier-1 worlds (Lee+Cho EEGNet; V2 source-invisible + source-rich source-visible World A), a target-informed gate is given k∈{1..50} labeled target trials/class. **v0 (bootstrap LCB):** accepts rise with k but false-accept ~23% (100% at k=1) — signal exists, estimator unsafe. **v1 (hardened bounded LCB — stratified Maurer-Pontil empirical-Bernstein, Bonferroni over classes×candidate interventions, underpowered→abstain):** 0 false accepts AND 0 deployable true accepts. **Label-budget frontier (k up to the full 50/class calibration budget):** still 0 deployable accepts in both worlds; the bounded cal-LCB rises with k (clipped ~−1.0 at small k → −0.53/−0.59 at k=50) but never nears the +0.01 gate. **B4 oracle (all target labels, diagnostic) confirms the target benefit is REAL** (audit ΔbAcc mean +0.018/+0.021, max +0.080) — so the bottleneck is certification power, not absence of effect.
- **Phase:** Fork 1 (jobs 888028 v0 / 888437 hardened v1 / 888470 frontier). **Artifacts:** `notes/{TARGET_INFORMATION_FRONTIER_FINAL_VERDICT,TARGET_INFO_TIER1_SMOKE_V1_VERDICT}.md`; `results/target_info/{tier1_smoke,tier1_budget_frontier}/*`.
- **Status:** semi-synthetic, supported-negative-for-k≤50-with-a-sound-bound (parked with finding). **Allowed:** "Target labels reveal that source-invisible beneficial interventions can exist, but under a valid finite-sample certificate k≤50 labels/class is insufficient to safely license deployment." **Forbidden:** "Few-shot target labels solve the source-only ceiling." / "The target-informed gate is validated as a deployable method." / "Target labels are useless." / "No target-informed method can work." / "The oracle selector is deployable." / presenting this as a real-EEG target-gain result.

## Dim↔type confound — exact limitation wording (use verbatim)
> The original two-backbone Phase 3 comparison did not isolate architecture from latent dimension. The
> Track-C factorial attributes MOST of the removability contrast to latent dimension, while retaining a
> RESIDUAL high-capacity architecture effect (at matched d_z=210 the SPD latent keeps more nonlinear subject
> residual than conv). So: the two-point comparison established representation dependence, not the causal factor;
> the Track-C capacity factorial (C11; 3 seeds) attributes it LARGELY to latent dimension (≈2/3 of the gap)
> but finds a RESIDUAL architecture effect at high d_z, so it must NOT be written as “resolved” or as
> “not architecture-type” (the latter is empirically refuted at d_z=210).

## Cross-check before camera-ready
- [ ] Every figure caption number traces to a row here.
- [ ] No global-forbidden phrasing in abstract/intro/conclusion (grep).
- [ ] Table 1 numbers == C4/C5/C7/C9 numbers.
- [ ] Every "in general" / "always" claim is scoped to "TSMNet/2a" or "EEGNet/2a".
