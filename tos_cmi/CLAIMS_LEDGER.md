# CLAIMS_LEDGER — TOS-CMI (provenance for every paper claim)

Machine-checkable provenance for each claim in the paper. Authoritative *wording* contract is
`paper/claim_evidence_table.md` (C1–C10, allowed/forbidden); this file adds **script / dataset / seeds /
outputs / status**. Branch `tos`; worktree `/home/infres/yinwang/CMI_AAAI_tos`. Env: synthetic = `icml`;
EEG dumps = `icml` (GPU). `repos/TSMNet` is currently a SYMLINK (untracked) — **must be pinned (submodule/
vendored at a fixed commit) before camera-ready** (see INTEGRATION.md). Seeds {0,1,2} unless noted.

```yaml
# ---------- SYNTHETIC certification (Fig 2; §3) ----------
score_fisher_localizes_covariance_synergy:        # C1
  status: supported
  scope: synthetic (Gaussian-mixture, known Bayes oracle)
  script: tos_cmi/score_fisher.py (+ tests)
  outputs: tos_cmi/results/{cert_cells, frontier.json, frontier_cells}
  figure: Fig 2A
geometry_alone_insufficient_conditional_safety:   # C2
  status: supported
  scope: synthetic synergy ("explaining-away") generator
  outputs: tos_cmi/results/cert_cells ; notes: PHASE13_DIAGNOSIS.md
  figure: Fig 2B
weak_gate_unsafe_accepts_plugin_improves:         # C3
  status: supported
  script: tos_cmi/score_fisher.py (nested vs plug-in log-ratio)
  outputs: tos_cmi/results/{estimator_diag.json, frontier.json}
  figure: Fig 2B,C
certified_default_on_deletion:
  status: NOT_supported (honest negative)
  reason: power floor -> conservative abstention; independent-seed cert borderline; stacking no stable gain
  outputs: tos_cmi/results/phase_diagram_powerfloor.json ; notes: PHASE131_CERTIFICATION.md
  figure: Fig 2D

# ---------- EEG frozen-feature study (BCI-IV-2a / BNCI2014_001, LOSO, domain=subject) ----------
tsmnet_low_rank_deletion_insufficient:            # C4
  status: supported
  backbone: TSMNet (LogEig/SPD, z_dim=210)
  scripts: tos_cmi/run_eeg_frozen_pilot.py -> tos_cmi/eeg/{ablation,adversarial}.py
  seeds: [0,1,2]; folds: 9 (LOSO)
  outputs: results/tos_cmi_eeg_frozen/BNCI2014_001_TSMNet_LOSO/ablation_report_seed{0,1,2}.json
  numbers: subj 0.997 -> RZ 0.96 (MLP) ~ random 0.997; task 0.75->0.75; nDcand ~3/210; full-7-dim 0.92-0.98
  figure: Fig 4
tsmnet_global_lpc_collapse_objective_scaling:     # C5
  status: supported (adversarially verified, wf_c2880caf)
  scripts: tos_cmi/run_lpc_curves.py -> tos_cmi/eeg/collapse_analysis.py
  seeds: [0,1,2]; folds: {1,5,9}; lambda: {0,0.3,1,3}; epochs: 300
  outputs: results/tos_cmi_eeg_frozen/lpc_collapse_curves/TSMNet/{summary.json, collapse_curves.png}
  numbers: feat_norm 1.09->0.00, top-1 SV->0.001, penalty->0, CE->ln4; NOT grad explosion; eff_rank scale-invariant
  figure: Fig 3A-C
tsmnet_collapse_fixable_collapsefree_removes_nothing:  # C6
  status: supported
  variants: raw_lpc / lpc_warm_ramp / lpc_scale_invariant (flag-gated, default-off in cmi/train/trainer.py)
  outputs: results/tos_cmi_eeg_frozen/lpc_collapse_curves/TSMNet/variant_compare.json
  numbers: warm_ramp avoids collapse lambda=1&3 (0/9); collapse-free subj_dec ~0.997 = ERM
  figure: Fig 3D
eegnet_low_rank_removability_representation_dependent:  # C7
  status: supported_with_caveat (adversarially verified, wf_cb3b4958)
  caveat: dim<->type confound provisionally addressed by Track C factorial (seed-0 capacity-mediated TREND; multi-seed + EEGNet-210 hardening in progress; do NOT write 'resolved' yet)
  backbone: EEGNet (conv, z_dim=16)
  scripts: tos_cmi/run_eeg_frozen_pilot.py -> tos_cmi/eeg/ablation.py
  seeds: [0,1,2]; folds: 9
  outputs: results/tos_cmi_eeg_frozen/BNCI2014_001_EEGNet_LOSO/ablation_report_seed{0,1,2}.json
  numbers: subj linear 0.82->0.35 / MLP 0.88->0.54 (>> random 0.73/0.81); task 0.64->0.64; selectivity 0.35-0.55 vs TSMNet 0.04-0.08
  figure: Fig 5A-C
eegnet_removal_partial_nonlinear_residual:        # C8
  status: supported
  numbers: RZ MLP 0.54 >> chance 0.125 (linear ~removed, nonlinear residual persists)
  outputs: same EEGNet ablation_report jsons
  figure: Fig 5A
eegnet_removable_but_no_dg_gain:                  # C9  (THE GAP -> motivates Track B benefit gate)
  status: supported
  scripts: tos_cmi/run_lpc_curves.py (EEGNet raw LPC sweep)
  seeds: [0,1,2]; folds: 9; lambda: {0,0.3,1,3} (n=27 per lambda)
  outputs: results/tos_cmi_eeg_frozen/lpc_collapse_curves/EEGNet/raw_lpc_sub*_lam*_seed*.json
  numbers: subj 0.89->0.17 (no collapse); mean LOSO target 0.43->0.39 paired-t p<=0.001 at lambda>=1; corr(leak,target) -0.14
  figure: Fig 5D
measurement_to_control_gap:                       # C10 (thesis)
  status: supported (synthesis of C4-C9)
  figure: Table 1

# ---------- METHOD-DEEPENING (branch method-deepen-v1, tag tos-cmi-method-deepen-v2-final @ 0ef7be3; DONE) ----------
source_ood_benefit_gate:          # Track B (DONE)
  status: supported
  scope: Lee/Cho sampled pilot (15 of 52/54 folds; labeled pilot, not full LOSO)
  scripts: [tos_cmi/eeg/source_ood_benefit_gate.py, tos_cmi/eeg/run_trackB_benefit_gate.py, tos_cmi/eeg/report_trackB.py]
  finding: benefit+safety source-only gate = 0 false accepts, 8/8 harms prevented, 20/20 correct; naive domain-gain/safety controllers false-accept harmful/useless erasures
  caveat: acceptance power UNTESTED on real EEG (no real positive)
  outputs: notes/METHOD_DEEPENING_FINAL_VERDICT.md ; results/method_deepen/trackB/*
task_preserving_erasure_phase2:   # Phase 2 (DONE, commit f8bd0ef)
  status: supported
  scope: Lee/Cho EEGNet seed0 first-15 folds
  finding: tp-LEACE preserves the deployed task decision (task-drop UCB +0.000) AND erases source subject (0.31->0.03) but target transfer FLAT (+0.000) -> gate ABSTAINS
  caveat: cc-predicted exact-zero is a structural tautology (probe re-learns router); tp-LEACE is the clean result
  outputs: notes/METHOD_DEEPENING_FINAL_VERDICT.md ; results/method_deepen/phase2/*
v2_source_only_acceptance_ceiling:  # V2 (DONE, Stage 1/1B/2; config b8e24e34fc84)
  status: supported_semisynthetic
  scope: real Lee/Cho EEGNet+TSMNet latents + injected ground-truth nuisance; Stage 2 scoped 72,000 tasks, 0 fail/degenerate
  finding: EEGNet World A clean source-only acceptance CEILING robust across source_subject_counts {8,16,32,all}xseeds (0 principled ACCEPT, oracle-supported, random-k LCB<=+0.006); World B/C robust refusal both backbones; 0 false accepts across all cells
  caveat: TSMNet World A NOT cleanly demonstrable under appended-nuisance construction (oracle unsupported at all nuisance_fraction {0.15..0.30}); gate has safe REFUSAL power + acceptance CEILING, NOT acceptance power
  outputs: notes/{V2_STAGE1B_VERDICT,V2_STAGE2_VERDICT,METHOD_DEEPENING_FINAL_VERDICT}.md ; results/method_deepen/v2_stage2/*
source_only_acceptance_ceiling_theory:   # theory spine (notes/CEILING_THEORY.md)
  status: theory_note
  evidence:
    - V2 source-only ceiling
    - World A target-beneficial source-invisible construction
    - Proposition 1 non-identifiability
    - Proposition 2 source-rich sufficiency
  allowed:
    - "Strict source-only certification cannot license deployment-shift benefit that is not represented in source-domain variation."
    - "Refusal is the safe action when benefit is source-invisible."
    - "Source-rich environments or target information can break the ceiling."
  forbidden:
    - "Source-only acceptance is impossible in general."
    - "The gate has acceptance power."
    - "V2 proves no erasure can ever help."
    - "Target-informed oracle is a deployable method."
source_rich_environment_constructive_support:   # Fork 2 (branch science-source-rich-v1; DONE, PARTIAL; commit 8174483)
  status: supported_with_caveats
  scope:
    positive: EEGNet semi-synthetic Lee->Cho source-rich World A (Phase 1A/1B, commit 69ec85c)
    limitation: TSMNet source-rich discovery fails / oracle fragile (Phase 1C, commit 8174483)
  construction: real EEG latents + injected KNOWN nuisance D_nuis=z with source regimes {aligned,reversed,noisy};
    target regime REPRESENTED in source; frozen frac 0.4/0.3/0.3, thresholds 0.02/0.01, k=8, seed0, first5;
    m=max(4,round(0.20*z_dim)) (EEGNet 4, TSMNet 42); target labels AUDIT-ONLY (selection_uses_target=false)
  environments: E0 subject / E_oracle regime DIAGNOSTIC / E2 covariance_cluster / E4 margin_cluster /
    E5 augmentation_shift / random
  finding: EEGNet -- E_oracle accepts safe target-beneficial erasers AND source-only E2 covariance_cluster recovers
    the accept (Lee oracle6/cov5, Cho oracle6/cov6), random 0, margin/aug 0 -> clean source-only positive witness.
    TSMNet -- Lee E_oracle accepts (Case B) but E2 covariance benefit_lcb NEGATIVE (max discovered lcb +0.0001);
    Cho E_oracle safe target-beneficial NEAR-MISS (benefit_lcb ~+0.007<0.010 -> abstain, target +0.033..+0.041
    LCB>0.01 all folds); E2/E4/E5/random 0 both. covariance-vs-regime AMI +0.365 EEGNet vs -0.011 TSMNet (chance)
    -> EEGNet discovery was a LOW-DIM ARTIFACT. Gate SAFE at 210-dim: 0 harmful, 0 discovered-env accepts both.
  verified: adversarial workflow wf_650c9d1f-aec (high confidence, 0 inconsistencies, integrity clean)
  allowed:
    - "Source-rich environments can make target-beneficial erasure source-visible in a semi-synthetic EEGNet setting."
    - "Covariance environments recover the constructed source-visible shift on EEGNet but not on high-dimensional TSMNet."
    - "Source-rich sufficiency is constructive but environment discovery is representation-dependent."
    - "The source-only gate did not misfire on high-dimensional latents (0 harmful, 0 discovered-env accepts)."
  forbidden:
    - "Source-rich environments solve source-only acceptance."
    - "Covariance clustering generally discovers target-beneficial shifts."
    - "TSMNet confirms the source-rich positive."
    - "This is a real EEG target-gain result."
  outputs: notes/{SOURCE_RICH_FINAL_VERDICT,SOURCE_RICH_PHASE1A_VERDICT,SOURCE_RICH_PHASE1C_VERDICT}.md ;
    results/source_rich/smoke/source_rich_{smoke,confirm_cho,tsmnet_lee,tsmnet_cho}_{summary.json,report.md}

# ---------- NOT YET ATTEMPTED (future tracks; must not be claimed) ----------
target_information_frontier:    # Fork 1 (branch science-target-info-v1; DESIGN-LOCK FROZEN @ ceb00f8, PM-approved)
  status: design_locked_frozen
  approved_content_hash: 3ad4ef312e325fa6   # hash of the config BEFORE freeze-metadata insertion
  scope:
    tier1: semi_synthetic target-information budget
    tier2: real EEG false-accept control
  not_yet:
    - driver
    - experiments
    - manuscript claim
  approved_budget_ladder:
    - B0 source-only
    - B1 unlabeled target triage-only
    - B2 k labels/class
    - B3 sequential calibration
    - B4 oracle diagnostic
  design: notes/TARGET_INFORMATION_FRONTIER_DESIGN.md ; config: eeg/configs/target_info_frontier_fixed.yaml
target_information_tier1_smoke_driver:   # Tier-1 smoke DRIVER DESIGN (branch science-target-info-v1; DESIGN ONLY)
  status: design_only
  scope: Lee/Cho x EEGNet x (V2 source-invisible + source-rich source-visible World A) x seed0 x first5 x R=10 ;
    budgets B0/B1-triage/B2 k{1,2,4,8,16}/B3-sequential/B4-oracle-diagnostic
  not_yet:
    - implementation
    - runs
    - manuscript claim
  design: notes/TARGET_INFO_TIER1_SMOKE_DRIVER_DESIGN.md
target_information_tier1_smoke_v1:   # Tier-1 smoke ran (v0 888028 + hardened v1 888437; semi-synthetic)
  status: supported_negative_for_k_le_16
  finding:
    - hardened finite-sample bounded LCB gate has 0 false accepts (v0 bootstrap had ~23%, 100% at k=1)
    - k<=16 labels/class yields 0 deployable true accepts
    - B4 oracle (all target labels) shows the target benefit EXISTS (audit dbacc mean +0.018/+0.021, max +0.080)
  implication:
    - safe certification requires larger k or a sharper valid estimator (instantiates CEILING_THEORY Prop 3)
  outputs: notes/TARGET_INFO_TIER1_SMOKE_V1_VERDICT.md ; results/target_info/tier1_smoke/*
  forbidden:
    - few-shot target labels safely solve the ceiling
    - target-informed gate is validated
    - this is a real-EEG target-gain result
architecture_x_dimension_factorial:               # Track C (DONE, 3-seed; SLURM 877939)
  status: supported_refined  # 3-seed verdict = LARGELY capacity-mediated + RESIDUAL architecture effect at high d_z
  scripts: tos_cmi/run_capacity_factorial.py -> tos_cmi/eeg/factorial_multiseed_analysis.py (file-parallel joblib; fold-cluster + paired + OLS CIs)
  cells: TSMNet tangent d_z{21,36,55,105,210} (SPD m{6,8,10,14,20}) + EEGNet F2{16,32,64,128,210}; 9 LOSO folds; seeds{0,1,2}
  outputs: results/tos_cmi_eeg_frozen/factorial/factorial_multiseed.json ; paper/figures/fig6_capacity_factorial.pdf
  finding: LEACE nonlinear residual rises monotonically with d_z in BOTH archs (OLS log(d_z)=+0.089 [0.086,0.092]).
    per-cell residual [95% fold-cluster CI]: TSMNet 21/36/55/105/210 = .397/.498/.559/.648/.740 ;
    EEGNet 16/32/64/128/210 = .393/.507/.574/.609/.628. matched-dim (TSMNet-EEGNet):
    21v16 +.004[-.008,.014] OVERLAP0 ; 36v32 -.008[-.022,.004] OVERLAP0 ; 55v64 -.015[-.024,-.004] ;
    105v128 +.039[.028,.051] ; 210v210 +.111[.094,.125] EXCLUDES0. Matching dim removes ~68% of the raw
    0.74-vs-0.39 gap => capacity is the DOMINANT axis; BUT interaction +.058[.051,.063] => TSMNet residual grows
    faster with d_z, and at matched d_z=210 SPD retains +0.11 MORE recoverable subject id than conv.
    VERDICT: LARGELY capacity-mediated WITH a residual architecture x dimension interaction at high capacity --
    NOT pure capacity. 'capacity-mediated, not architecture-type' is REFUTED at high d_z. Caveat: single dataset (2a), LDA cap 8 subj.
concept_erasure_baselines_vs_tos:                 # Track G (DONE; cross-seed stable)
  status: supported
  scripts: tos_cmi/eeg/erasure_baselines.py (self-contained LEACE + INLP; LEACE validated: linear subj->chance)
  seeds: [0,1,2]; folds: 9; backbones: TSMNet-210, EEGNet-16 (2a frozen Z)
  outputs: results/tos_cmi_eeg_frozen/BNCI2014_001_{TSMNet,EEGNet}_LOSO/erasure_report.json
  findings:
    - LEACE removes LINEAR subject decode to chance (0.115) on BOTH backbones, task preserved.
    - LEACE DOMINATES TOS V_D deletion on subject removal at equal task cost (TSMNet MLP 0.74 vs TOS 0.96;
      EEGNet MLP 0.39 vs TOS 0.55) -> the score-Fisher PROJECTION is NOT the contribution.
    - nonlinear (MLP) residual after optimal linear erasure is the discriminator: TSMNet 0.74 (subject is
      not eliminated by the tested linear erasure controls; even optimal linear erasure for linear decodability leaves a nonlinear MLP residual) vs EEGNet 0.39.
    - INLP drives subject->chance but DESTROYS task (TSMNet 0.75->0.55, EEGNet 0.64->0.25): over-erasure.
  implication: reframe contribution to measurement + certification/refusal + measurement-to-control gap
    (NOT the eraser); paper needs a concept-erasure baseline table + Related Work SS5.3 update. RLACE/SPLINCE
    still optional (RLACE adversarial; SPLINCE = holstege2025 oblique).
erasure_target_deployment:                        # Step 3 (DONE; SLURM 878002)
  status: supported
  scripts: tos_cmi/eeg/erasure_target_deploy.py (source-only eraser+head fit -> held-out target; file-parallel)
  seeds: [0,1,2]; folds: 9 LOSO; backbones: TSMNet-210, EEGNet-16; methods: full/TOS_VD/LEACE/RLACE/INLP/random_k
  outputs: results/tos_cmi_eeg_frozen/erasure_target_deploy/{*_seed*.csv,*_paired.csv,*_summary.json} ; paper Table 3
  guard: target (Z_t,y_t) used ONLY for final scoring; NO eraser/head/hyperparam/calibration selected on target
  finding: deployed on held-out target, NO eraser improves target bAcc. dbAcc[95% fold-cluster CI] vs full:
    TSMNet LEACE +.001[-.004,.005], RLACE -.004[-.006,-.002], TOS -.000, INLP -.062 (src task .749->.533);
    EEGNet LEACE -.011[-.021,-.002], RLACE -.012, TOS -.000, INLP -.160 (=chance, task destroyed).
    Only movement = small NLL drop, but same-k RANDOM matches it (TSMNet LEACE dNLL -.031 vs random -.034;
    random subj_dec .998 = NOT erased) => NON-SPECIFIC regularization, NOT a domain-removal benefit.
    Closes the measurement-to-control loop: optimal erasure deployed on target != DG gain.
real_eeg_multidataset_erasure_deployment_no_gain:  # C12-real (branch tos; DONE, Case 1 integrated)
  status: supported
  scope: 9 valid dataset-backbone cells (2a both, 2b-EEGNet, Lee/Cho/HGD both); 2b-TSMNet excluded (degenerate 3ch)
  datasets: [BNCI2014_001, BNCI2014_004, Lee2019_MI, Cho2017, Schirrmeister2017]   # 129 subjects beyond 2a
  scripts: [tos_cmi/eeg/erasure_target_deploy.py, tos_cmi/eeg/bigN_report.py, tos_cmi/eeg/dataset_manifest.py]
  acceptance_rule: upper 95% subject-cluster CI < +0.01 for ALL principled erasers (LEACE/TOS_VD/RLACE)
  finding: no principled source-fitted eraser yields a practically meaningful target-bAcc gain in any valid cell
    (e.g. Cho2017-TSMNet LEACE -0.001[-0.003,+0.000]; Lee2019-TSMNet -0.002[-0.003,+0.000]; HGD-TSMNet -0.001[-0.005,+0.003])
  caveat: task-safety heterogeneous -- Lee/Cho EEGNet LEACE/RLACE drive task to chance -> deployment HARMS target -0.15..-0.19
  outputs: notes/REAL_EEG_VALIDATION.md ; results/tos_cmi_eeg_frozen/{validation_manifest.json, erasure_target_deploy/*}
  figure: paper SS4.7 Table (tab:bigN_compact) + App (tab:bigN_full)
end_to_end_tos_training:        {status: not_attempted, plan: Track E (conditional-on-kept critic + PCGrad + anti-collapse)}
```

**Real-EEG multi-dataset validation (branch `tos`, DONE; C12-real supported):** see `notes/REAL_EEG_VALIDATION.md`
for the run manifest + pre-registered acceptance criteria. Key provenance: fold-cap bug fix `ede201a`
(`--target-subjects all`/factorial FOLDS were hardcoded to 9 -> non-9-subject datasets dumped only 9 LOSO
folds; now read from the real MOABB subject_list; existing 9-fold dumps verified full-source-pool),
degenerate-metric guard `10c22e9`, group/array `%4` submission `99b767d`. Datasets = 2b, Lee2019 (54),
Cho2017 (52), High-Gamma (14); Stieger2021 excluded (variable channels). Multi-dataset numbers do NOT
enter the paper claim contract until the CONFIRM/MIXED/OVERTURN readout is in.

**Verification rule for any new claim:** add a block here with `status`, runnable `script`, `dataset`,
`seeds`, `outputs` (committed or regenerable), and `figure`; only then may it enter the paper. Mirrors the
project's hard-won provenance discipline (see top-level `notes/EVIDENCE_LEDGER.md`, `CLOSEOUT.md`).
