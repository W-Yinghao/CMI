# FSR_19 — PC1-S Subject-Token Positive Control (pre-registration)

**Project FSR — Phase 4C.** Pre-registration of PC1-S: an injected, *known* harmful branch-local shortcut used to show the FSR verification protocol can **detect, localize, and repair** it. This is a **positive control**, not evidence that natural EEG contains this shortcut (Phase 4B verdict: `NO_VERIFIED_HARMFUL_BRANCH_SHORTCUT`). CPU-only on the frozen 4B dumps + checkpoints; **no GPU refit, no CMI/fbdualpc, no architecture search, no target-label fit, α never chosen by target bAcc.** (Doc renumbering: `FSR_18` is the roadmap, so PC1 protocol = FSR_19, results = FSR_20.)

## Primary + controls
- **Primary injected branch: `spatial_z`** (the strongest natural candidate — max leakage, load-bearing, coupled). Not `fused_z` (too close to classifier input).
- **Localization controls:** `graph_z`, `temporal_z` injection at α=1.0 (lower priority) — to confirm FSR localizes the injected shortcut to the injected branch. Not a claim about natural graph/temporal harm.

## Injection (all source-derived; target-label-free)
Per LOSO fold (dataset, held-out subject), operating on the frozen dumps:
1. **Source-only class assignment** `c_d` per source subject `d` = majority source label (tie-break by deterministic subject-id hash). **Target class** `c_T` = deterministic hash of the target subject id mod `n_cls` (never target labels) — an arbitrary, source-derived class so the injected token pushes the target toward a wrong class (the harm).
2. **Class-directed direction** `v_{b,c}` = mean over source of `∂ logit_c / ∂ z_b` through the frozen `head3(_fuse3(...))` (autograd; source-only, label-free). **Subject-unique** `u_d` = deterministic unit vector seeded by `hash(d)` (for L1 decodability).
3. **Token** `token = normalize(u_·) + normalize(v_{b,c_·})` (subject component for L1, class component for L5/L6). Inject `z_b' = z_b + α · scale · token`.
4. **Source-logit-margin normalization:** `scale` chosen (source-only) so that at α=1.0 the mean source class-directed logit shift ≈ the median source correct-vs-runner-up margin. **α ∈ {0, 0.25, 0.5, 1.0, 2.0}**; α selection uses source only.
5. **Sanity (STOP if fail):** α=0 reproduces the original logits/metrics exactly; α>0 produces a positive mean source logit shift toward `c_d` (token actually moves the frozen head).

## Verification (sign convention as Phase 4B: `task_drop = bAcc_orig − bAcc_erased`; `<0` = erasing helps = harmful)
- **L1** source-fit subject probe on injected `z_b'` (vs original `z_b'` baseline; ceiling caveat: spatial_z is already ~0.92 subject-decodable).
- **L5** erase subject subspace from injected `z_b'`, recompose, `task_drop` + logit-SymKL vs random-subspace control.
- **L6** injected target bAcc (vs original) — the induced harm.
- **Localization** — the injected branch is the one flagged (largest injected harm + reliance).

## Repair arms (at α=1.0 and predeclared α=2.0)
- **R0** injected, no repair.
- **R1** oracle token-subspace removal (removes the known injected `{u, v}` span) — upper bound, not deployable.
- **R2** source-estimated subject-subspace erasure (fit on source injected `z_b'` only) — the realistic candidate.
- **R3** random-k / random-subspace control (matched dim, same branch/α).
- **R4** (optional) task-protected source-gated erasure: accept only if source-val task drop ≤ ε=0.01 and source-only subject-reduction gate passes.
`recovery_fraction = (bAcc_repaired − bAcc_injected) / (bAcc_original − bAcc_injected)`.

## Pass/fail
- **Detection pass** (primary `spatial_z`, α=1.0 or predeclared α=2.0): L1 does not fall; **L5 task_drop < 0** (erasing the injected token helps target); L5 SymKL specificity > random; **L6 injected target bAcc < original**; injected branch correctly localized.
- **Repair pass:** **R1** `recovery_fraction ≥ 0.70` and clearly > random-k; **R2** `recovery_fraction >` random-k and no source-val task-safety failure. If R1 passes but R2 fails ⇒ "detect+localize+oracle-repairable; source-estimated repair imperfect." If both ⇒ "detect+localize+repair with source-estimated erasure."

## Outputs (`results/fsr_pc1_subject_token/`)
`pc1_injection_manifest.csv`, `pc1_alpha_grid.csv`, `pc1_token_direction_sanity.csv`, `pc1_l1_subject_decode.csv`, `pc1_l5_l6_harm_curve.csv`, `pc1_repair_results.csv`, `pc1_randomk_specificity.csv`, `pc1_localization_summary.json`, `pc1_target_label_firewall.json`, `pc1_verdict.json` (with `detection_pass`, `oracle_repair_pass`, `source_estimated_repair_pass`, `primary_branch`, `primary_alpha`, `alpha_selection_used_target=false`, `target_labels_used_for_fit=false`, `target_labels_used_for_final_eval_only=true`).

## STOP rules
```text
1 alpha=0 does not reproduce original logits/metrics.
2 token direction fails the source-only logit-shift sanity.
3 target labels used for token assignment / alpha selection / repair fit / repair selection.
4 PC1 requires GPU.
5 PC1 requires retraining FBCSP-LGG.
6 repair reported only on selected positive folds.
7 random-k and oracle repair equally effective but ignored.
8 spatial_z injection produces NO L5/L6 harm across ALL alpha -> STOP, re-audit token design (not a paper failure).
9 results written as natural harmful shortcut rather than injected positive control.
```

## Framing (fixed)
PC1 is a controlled injected shortcut. It shows the protocol can detect/localize/repair a *known* harmful branch-local shortcut; it does **not** claim natural source prevalence creates this shortcut. Manuscript Result 4 = "injected positive control proves the framework can detect and repair a known harmful shortcut," alongside Result 3 (natural verification refuses blind repair).
