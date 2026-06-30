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
  caveat: dim<->type CONFOUND unresolved (EEGNet conv+16d vs TSMNet SPD+210d collinear) -> Track C
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

# ---------- NOT YET ATTEMPTED (future tracks; must not be claimed) ----------
source_ood_benefit_gate:        {status: not_attempted, plan: Track B (inner source-LODO pseudo-target benefit)}
architecture_x_dimension_factorial:               # Track C (PROVISIONAL, seed0; seeds 1,2 + EEGNet-210 dumping)
  status: provisional_seed0  # VERDICT = capacity-mediated (seed-0 factorial; NOT final until 3-seed + EEGNet d_z=210 matched cell)
  hardening_needed: [seeds 1+2 (running), EEGNet F2=210 top-matched conv cell, multiseed analyzer w/ cluster-bootstrap CI + matched-dim contrast]
  scripts: tos_cmi/run_capacity_factorial.py -> tos_cmi/eeg/factorial_analysis.py
  cells: TSMNet m{6,8,10,14,20}=tangent{21,36,55,105,210} + EEGNet F2{16,32,64,128}; 9 folds; seed0
  outputs: results/tos_cmi_eeg_frozen/factorial/* ; factorial_removability_seed0.json
  finding: LEACE nonlinear residual rises monotonically with latent dim WITHIN both archs; at MATCHED dim
    SPD~conv nearly coincide (16/21:0.40/0.40; 32/36:0.50/0.50; 105/128:0.65/0.61; TSMNet-210:0.74) ->
    the TSMNet-0.74-vs-EEGNet-0.39 gap is capacity (210 vs 16), NOT SPD-vs-conv. Resolves the dim<->type
    confound (paper SS4.5, Fig 6; SS6.3 limitation -> resolved). Caveat: 2a, seed0 (multi-seed in progress).
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
      nonlinearly/redundantly encoded; NOT removable by ANY linear method) vs EEGNet 0.39.
    - INLP drives subject->chance but DESTROYS task (TSMNet 0.75->0.55, EEGNet 0.64->0.25): over-erasure.
  implication: reframe contribution to measurement + certification/refusal + measurement-to-control gap
    (NOT the eraser); paper needs a concept-erasure baseline table + Related Work SS5.3 update. RLACE/SPLINCE
    still optional (RLACE adversarial; SPLINCE = holstege2025 oblique).
end_to_end_tos_training:        {status: not_attempted, plan: Track E (conditional-on-kept critic + PCGrad + anti-collapse)}
```

**Verification rule for any new claim:** add a block here with `status`, runnable `script`, `dataset`,
`seeds`, `outputs` (committed or regenerable), and `figure`; only then may it enter the paper. Mirrors the
project's hard-won provenance discipline (see top-level `notes/EVIDENCE_LEDGER.md`, `CLOSEOUT.md`).
