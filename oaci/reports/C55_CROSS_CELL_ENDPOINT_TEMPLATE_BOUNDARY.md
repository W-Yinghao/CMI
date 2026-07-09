# C55 - Cross-Cell Endpoint-Template Transfer / Information-Boundary Audit (frozen C19 `664007686afb520f`)

## Decision

`C55-G_transfer_requires_unavailable_test_endpoint_scalar`

Secondary: `C55-S1_joint_margin_transfer_dominates;C55-S3_threshold_transfers_but_scalar_unavailable;C55-S5_target_local_transfer_only;C55-S7_no_split_label_budget_available;C55-S8_source_only_escape_hatch_still_closed`

## C54 Replay

- key-only / C52 diagnostic / same-cell endpoint scalar: **0.488 / 0.813 / 0.944**
- binary sign-bit hit / overlap / AUC: **0.944 / 1.000 / 1.000**
- C54 matched cross-cell template hit: **0.704**

## Transfer Boundary

- best template-only transfer: `matched_source_geometry_template_only` hit **0.704**
- best endpoint-scalar transfer: `leave_cell_out_endpoint_scalar_threshold` hit **0.944**
- same-cell minus template-only gap: **0.241**
- same-cell minus endpoint-scalar transfer gap: **0.000**
- requires held-out target endpoint scalar for full close: **True**
- split-label budget available: **False**

## Bottom Line

C55 closes the remaining C54 ambiguity. Cross-cell endpoint templates transfer only partially: the matched candidate-order template reaches the C54 0.704 level, while leave-cell/leave-target/leave-trajectory templates stay below the same-cell endpoint scalar. The 0.944 closure reappears only when the held-out candidate's target endpoint scalar is read and thresholded, so the boundary is an endpoint-scalar availability gap rather than a source/key/template sufficiency result.

## Red-Team Checks

- c54_identity_replay: PASS - C55 replays C54 key-only, C52 diagnostic, endpoint scalar, sign-bit, AUC, overlap, and split-label budget.
- template_vs_test_scalar_separated: PASS - Template-only transfer remains partial while endpoint-scalar transfer closes only after reading held-out endpoint scalars.
- heldout_cell_labels_not_used_for_transfer_template: PASS - Leave-cell, leave-target, and leave-trajectory endpoint transfers fit templates outside the held-out cell.
- availability_ledger_marks_test_endpoint_scalar: PASS - Rows requiring held-out target endpoint scalars are explicitly unavailable under the original source-only DG setting.
- nulls_emitted: PASS - C55 emits field, threshold, block, label, and scalar permutation null controls.
- split_label_boundary_kept_closed: PASS - No split-label construction is available or claimed.
- no_training_or_reinference: PASS - C55 is read-only over committed C54/C53/C52 artifacts.
- no_chosen_checkpoint_artifact: PASS - C55 emits diagnostic summaries, ledgers, nulls, and reports only.
