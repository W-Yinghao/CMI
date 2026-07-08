CEDAR_00 — P0 Source-Only Frozen-Latent Scaffold / Red-Team Contract

Active taxonomy:
CE2_target_labels_quarantined_diagnostic_only
CE3_source_only_mask_selection
CE4_grouped_conditional_leakage_probe
CE5_random_subspace_control_required
CE6_red_team_required_before_report

Inactive:
CE1_new_cmi_training_objective
CE7_structured_channel_pruning
CE8_graph_surgery
CE9_tta_preconditioner
CE10_generalization_or_safety_gate_claim

Key result: CEDAR_00 establishes the project scaffold and validates that the
first executable phase is source-only P0 frozen-latent surgery, not another CMI
regularizer.

P0 synthetic witness:
  setup: frozen z / binary y / 3-domain d / grouped cross-fit
  injected nuisance: latent dim 0 carries domain after conditioning on label
  injected task signal: latent dim 1 carries class signal
  selected setup: drop_top_1_of_6
  selected unit: latent dim 0
  baseline leakage advantage: 0.783
  leakage drop fraction: 0.401
  source bAcc before / after: 1.000 / 1.000
  source bAcc drop: 0.000
  random-control drop fraction: 0.040

But this does not become actionability:
target-label decision role:       quarantined / diagnostic-only
deployable selector emitted:      no
checkpoint or mask artifact:      no
P1 channel pruning status:        blocked until reviewed P0 accept
P2 TTA preconditioner status:     blocked until reviewed P1 accept
target generalization claim:      inactive

Red-team review passed before this report:
schema/project/phase checked
claim boundary rejects target-generalization guarantee
permutation null must remain low
target metrics forbidden in source-side decision reasons
highest-utility ACCEPT is the only selected candidate
random-subspace control must be present for every candidate
deployable artifact keys must be empty or absent
grouped cross-fit availability is reported

Validation:
CEDAR focused:      8 passed
red-team smoke:     passed / 8 checks / 0 warnings
compile hygiene:    python -m compileall -q cedar_eeg
P0 smoke:           synthetic /tmp feature dump only
Slurm status:       templates added; no real EEG workload run on login node

Artifact hygiene:
no large CEDAR payload committed
no npz/npy/pt/checkpoint artifacts committed
compact JSON output path supported
row-level atlas emitted inside P0 JSON only

Files added or updated under cedar_eeg/red_team/,
cedar_eeg/runners/run_red_team.py, cedar_eeg/reports/,
cedar_eeg/surgery/selection.py, cedar_eeg/runners/run_p0_frozen_latent.py,
and cedar_eeg/tests/test_p0_contracts.py.
