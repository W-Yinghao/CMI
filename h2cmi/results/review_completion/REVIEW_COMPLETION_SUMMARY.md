# Review Completion Summary

## Completed Analyses

- `four_branch_complete_ci.csv/json`: MI/W1 and Sleep/W2 branch BA, G, P, interaction, and full delta with paired bootstrap CIs from raw rows.
- `sleep_branch_confusion_matrices.json` and `sleep_per_stage_recall.csv`: deterministic W0.1 branch confusion/recall, including deltas vs identity+uniform.
- `v2p_corrected_*`: corrected 9-point q-grid V2P summaries from Wave0 raw/report artifacts; unit key preserves dataset/pair/subject/session/seed/method.
- `geometry_capacity_existing_ci.csv`: W1.geometry CI export for null/reref/gain/dropout and per-operator BA.
- `encoder_backbone_details.*`, `baseline_inventory_and_blockers.md`, and blocker reports: factual implementation and availability audit.

## Numbers Ready To Paste

- Sleep W2 primary: G about -0.020, P about -0.144, interaction about +0.059; deterministic replay re-confirms terminal values to four decimals.
- V2P q-grid: {0.1,...,0.9}; FRSC BA is roughly flat across q while ordinary-accuracy oracle-q can improve at prevalence extremes in the controlled FRSC lens.
- W1.geometry: max(full-cov)-max(diagonal) is not positive/significant on tested perturbations; diagonal adequacy falsification criterion is not met for reref/gain/dropout.

## Claims To Weaken

- Do not claim official SPDIM comparison.
- Do not claim diagonal latent geometry is universally adequate beyond tested perturbations.
- Do not claim an orthogonal-score estimator was evaluated.

## Strengthened Claims

- W2 prior harm is metric-prior mismatch under balanced accuracy, not merely missing confusion or replay failure.
- V2P corrected analysis preserves repeated transitions and shows displacement is distinct from utility.
- Existing geometry falsification is bounded and reproducible from frozen artifacts.
