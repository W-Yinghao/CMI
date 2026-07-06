# RQ4 probe target-label firewall (spec)

**Not executed.** The firewall the approved RQ4 run MUST enforce and record per artifact, so the run cannot silently leak target labels into fitting. Mirrors the CITA/ACAR firewall discipline.

## Rule
`target_labels_used_for_fit = NO`, `target_labels_used_for_eval = YES`.

- **Per-branch leakage probe (L1)** trains on **domain labels** (subject id `d`), never task `y`, and only on **source subjects**. The held-out target subject is never in the probe training set.
- **Per-branch reliance (L5)** — the subject-subspace projector is fit **source-train-only** (excludes source-val, excludes target); the head-replay uses the frozen task head; target task `y` is used **only** to score the final `task_drop` endpoint.
- No target `y`, target `d`, or target latent participates in: probe fitting, subspace fitting, model selection, hyper-parameter selection, early stopping, or threshold choice.

## Recorded flags (per artifact `.npz`/manifest)
```yaml
firewall:
  probe_train_domains: source_only
  target_subject_in_probe_train: false
  projector_fit_split: source_train_only
  target_y_used_for_probe_fit: false
  target_y_used_for_subspace_fit: false
  target_y_used_for_model_selection: false
  target_y_used_for_hyperparameter_selection: false
  target_y_used_for_early_stopping: false
  target_y_used_for_final_eval_only: true
  target_d_used_for_fit: false
  replay_ok: <bool>            # exact head-replay integrity
  random_control_valid: <bool> # random-subspace / random-projection control ~0
```

## STOP on breach
Any flag above resolving to a target-label use in fitting/selection → **STOP** and write a breach note; do not produce RQ4 numbers. This is stopping-rule (4) of `FSR_11`.

## Why this matters for the claim ledger
RQ4, if run, must remain a `NO`-tagged route to be eligible for a quantitative branch-locality claim. A firewall breach would push it to `YES_FORBIDDEN` (like the retracted LPC batch-rollback) and disqualify it from any RQ table.
