# CMI-Trace P0/P1 revision — execution note

Living log. Updated after every stage with commands, job IDs, outcomes, and failures.
Scientific discipline: no fabricated results; no protocol changes without disclosure; no
target-label selection. A null/contradictory result triggers an honest scope reduction, not tuning.

## 0. Provenance (Stage 1)

- Integration branch: `agent/cmi-trace-p0p1`
- Worktree: `/home/infres/yinwang/CMI_AAAI_cmitrace`
- Base branch (CIGL lineage, latest descendant with code + `results/cigl_r123/final/` + manuscript):
  `project/cigl-functional-cmi` @ `c1d55be6f3d196833e89d50b6e0c5a8e3904e8f7`
  (CIGL_67; descendant of scaffold `project/cigl-r123-scaffold`; verified `git merge-base --is-ancestor`).
  - Diverged sibling `project/cigl-direct-reliance-cmi` @ `e619713` (CIGL_68 direct-reliance NEGATIVE) —
    NOT used as base (later CIGL_67 wall-clock, cleaner descendant); its negative result is noted, code identical.
- TOS lineage (authoritative): branch `tos` @ `1c65d7927d5a4906065fdd21e667eb19f30e2088`.
  TOS files (`tos_cmi/eeg/erasure_target_deploy.py`, etc.) will be brought into the integration branch by
  copy-with-provenance for P0.4/P0.5/P1.1, since CIGL and TOS are divergent histories (no merge).
- Manuscript source **IS in Git** (not PDF-only): `paper/cigl_latex/main.tex` + `sections/*.tex` + `tables/*.tex`.
  - Current `\title{}` = "CIGL: Auditing and Reducing Label-Conditional Domain Leakage in EEG Graph…"
  - The execution prompt refers to the manuscript as **"CMI-Trace: What Domain Invariance Removes in
    Cross-Subject EEG."** This is a target rename, not a separate source. Handled under P0.3 (title decision
    gated on P0.1 completion). New paper package dir: `paper/cmi_trace/` (claim table, figures, tables).

- Compute env: conda `c84c-eeg2025-v3` (torch 2.6.0, numpy, sklearn, scipy). SLURM available;
  idle GPU partitions V100/V100-32GB/P100/A40, idle CPU nodes.
- Harness validated on CPU: `scripts/run_cigl_r2_seed0_gate.py --dry_run_synthetic` runs end-to-end,
  head-replay OK (max_abs_diff ~2e-8), leakage + source/target bAcc produced.

### Key existing infrastructure (reused, not rebuilt)

- Same-backbone contract: `cmi/eval/baseline_registry.py` (`SAME_BACKBONE_CONTRACT`, DGCNN adapter).
- DG penalties already implemented: `cmi/methods/dg_penalties.py` has `coral`, `mmd`, `irm` (IRMv1),
  `vrex` (V-REx), `chsic`, `scldgn`, DANN/CDANN/CDAN. Registry marks coral/label_coral/mmd/irm/vrex as
  DEFERRED (not yet activated as objective rows).
- Trainer dispatch: `cmi/train/trainer.py` (`is_pen` branch applies coral/irm/vrex on flat `z`; for the
  DGCNN adapter flat `z` == graph readout `graph_z`).
- Nested source-only selector **already exists**: `cmi/run_lambda_select.py` (inner leave-one-source-domain
  or random source-domain val; `--select_rule bacc`; retrain-on-all-source; λ=0 always a candidate).
- R2 same-backbone gate runner: `scripts/run_cigl_r2_seed0_gate.py` → `_train_eval` (in
  `scripts/run_cigl_phase3a_dgcnn_gn_regularizer_pilot.py`), firewall metadata, verified `.audit.npz`.
- Reliance (R_rel(k) = `task_drop`): `cmi/eval/head_export.py` + `cmi/eval/gap_diagnostic.py`;
  frozen `results/cigl_r123/final/r3_reliance.csv` (k=1,2,4,8; label_conditional/marginal_domain/random_subspace).
- Audit: `cmi/eval/graph_leakage.audit_graph_node_objects` (posterior-KL CMI + within-label permutation null).

## Stage log

- Stage 1 (provenance/preflight): DONE. Branch + worktree created; base SHAs recorded; env + harness validated.
