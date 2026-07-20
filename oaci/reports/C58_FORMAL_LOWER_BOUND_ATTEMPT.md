# C58 - Formal Lower-Bound Attempt / Instrumented Real-EEG Training Gate (frozen C19 `664007686afb520f`)

## Primary Decision

`C58-A_finite_population_lower_bound_established`

Active: `C58-A_finite_population_lower_bound_established;C58-D_empirical_boundary_only_formal_bound_not_yet_supported;C58-F_formalization_requires_new_instrumented_real_eeg_training;C58-H_new_training_not_justified_yet`

Inactive: `C58-B_lecam_style_two_point_bound_established_under_empirical_assumptions;C58-C_fano_assouad_packing_bound_nontrivial;C58-E_source_observable_escape_hatch_found;C58-G_new_training_campaign_scientifically_authorized`

## What C58 Establishes

C58 establishes a finite-population lower-bound style statement for registered information partitions in the frozen C50-C55 audit universe. For a registered partition `G`, `H*_G` is the best empirical hit attainable inside that partition and `M_G = 1 - H*_G` is the corresponding miss-risk floor for that partition family.

The key source-side numbers remain bounded: random/tie is 0.430, best strict source is 0.506, best source scalarization is 0.574, best key-only/source-geometry is 0.488, best template-only transfer is 0.704, and the same-label endpoint scalar reference is 0.944.

## What C58 Does Not Establish

C58 does not claim a formal theorem, does not establish a minimax lower bound, does not convert the same-label endpoint scalar into an available selector, and does not start M1 manuscript drafting. Le Cam and Fano/Assouad rows are empirical proof-attempt ledgers only.

## Training Gate

`TRAINING_NEEDED_BUT_NOT_AUTHORIZED`

C58 itself does not run training or re-inference. Future instrumented real-EEG training is scientifically useful only for split-label cache construction, atom traces, per-trial logits/probabilities, and independent checkpoint-field replication. That future campaign is not authorized here.

## Boundary

The information boundary remains sharp: source-only and key-only classes do not close the residual; label-derived diagnostics and endpoint scalars close it only by crossing into target-label diagnostic content.
