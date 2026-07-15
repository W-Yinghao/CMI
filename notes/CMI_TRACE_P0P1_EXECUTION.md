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
- Stage 3 (P0.1 methods): DONE @14bf514. label_coral (conditional-CORAL) + graph-aware coral/label_coral
  (graph readout + mean node) + IRM/VREX activated in OBJECTIVE_METHODS registry; run_loso config parse;
  10/10 tests; CPU smoke all methods train on DGCNN adapter.
- Stage 5 (config freeze): DONE @f8abcdf. configs/cmi_trace_p0p1.yaml config_sha256=002e9241...
- Stage 7 (P0.2 audit + cluster CI): DONE @b19d19d. cmi/eval/objective_effect_report.py (artifact-driven,
  reads .audit.npz; moment gaps / risk var / IRM diag / geometry / reliance k=2 + random control; paired
  fold-cluster bootstrap CI). scripts/run_cmi_trace_objective_comparison.py + aggregate_cmi_trace_objective.py.
  10/10 tests. Validated END-TO-END on REAL BNCI2014_001 fold0 (erm R_rel_k2=+0.038, random ctrl -0.003,
  firewall passed).
- Stage 8 (P0.4 deployment CI): DONE @5279447. tos_cmi/eeg/deployment_ci.py three-state (confirmed via
  lower>+0.01 / ruled-out via UPPER<+0.01 / inconclusive) + deployable_benefit gating; 8/8 tests.

### P0.1 full LOSO fleet — LAUNCHED (Stage 6)

- Env fix: `c84c-eeg2025-v3` torch is CPU-only (cuda_build None); GPU env is **`eeg2025`** (torch 2.6.0+cu124).
  First 6 submissions (896348-896353) correctly FAILED CLOSED ("CUDA not available; refusing CPU fallback") —
  no results written. Resubmitted on `eeg2025`.
- Submitted 6 GPU jobs (2 datasets x 3 seeds), all 8 OBJECTIVE_PRIMARY methods, all LOSO folds, resumable:
  | dataset       | seed | jobid   |
  |---------------|------|---------|
  | BNCI2014_001  | 0    | 896355  |
  | BNCI2014_001  | 1    | 896356  |
  | BNCI2014_001  | 2    | 896357  |
  | BNCI2015_001  | 0    | 896358  |
  | BNCI2015_001  | 1    | 896359  |
  | BNCI2015_001  | 2    | 896360  |
- All 6 RUNNING on A100 nodes (node04/05/06/54/56) as of launch; training real data confirmed (fold0/erm).
- Command per cell: `run_cmi_trace_objective_comparison.py --epochs 80 --probe_epochs 100 --n_perm 50
  --select_epochs 40 --select_inner_folds 3 --primary_k 2`. Writes per-cell JSON + verified `.audit.npz` +
  raw_rows.jsonl under results/cmi_trace_p0p1/objective_comparison/<dataset>/.
- Aggregate when done: `python scripts/aggregate_cmi_trace_objective.py`
  (reads raw_rows.jsonl -> objective_effect_summary.csv / paired_deltas.csv / cluster_intervals.csv /
  selected_hparams.csv / completeness_matrix.csv). These runs need GPU-hours and will NOT complete within
  the implementation session; results are PENDING and must not be pre-empted by fabricated numbers.
- Secondary nested rows (coral_nested/label_coral_nested/irm_nested/vrex_nested) deferred to a follow-up
  launch (`OBJ_METHODS="coral_nested label_coral_nested irm_nested vrex_nested"`) to bound first-launch cost.

### Data-access + infra issues resolved (Stage 6 addendum)

- **NFS file-lock collision**: 3 concurrent per-seed jobs loading the SAME moabb cache hit
  `OSError: [Errno 37] No locks available`. Fix: ONE job per dataset with all 3 seeds (data loaded once).
- **BNCI2015_001 permission-denied**: `--export=ALL` inherited a shell env resolving BNCI to the
  permission-denied shared datalake (`/projects/.../datalake/raw`). Fixed by downloading BNCI2015_001 fresh
  to the writable home cache and pinning `MNE_DATASETS_BNCI_PATH=/home/infres/yinwang/mne_data` in the sbatch.
  BNCI2015_001 now loads (5600×13×384, 2 classes, 12 subjects).
- Superseding job IDs: **896396** (BNCI2014_001 seeds 0/1/2), **896397** (BNCI2015_001 seeds 0/1/2), both
  RUNNING on GPU, resumable. Earlier 896355-360 / 896388 superseded.

### Stage status summary (code + tests)

| Task | Status | Tests | Commit |
|------|--------|-------|--------|
| P0.1 methods + selector + registry | DONE | 10/10 | 14bf514 |
| P0.2 objective→effect audit + cluster CI + runner | DONE (real-data validated) | 10/10 | b19d19d |
| P0.4 deployment CI three-state | DONE | 8/8 | 5279447 |
| P0.5 FMScope bridge (2×2 oracle/source × subj/random) | DONE (synthetic; real dumps absent) | 9/9 | a1df27e |
| P1.1 flat-feature CMI ruler + cross-fitting | DONE (synthetic + real BNCI2014 smoke) | 5/5 | 618b36f |
| P1.2 synthetic ground-truth CMI | DONE (full 39-setting sweep) | 10/10 | a64c17b |
| P1.3 multi-capacity probe + familywise null | DONE | 6/6 (+13 backcompat) | 2f60af7 |
| P1.4 exact-head reliance hardening | DONE | 7/7 | 2f60af7 |
| P0.3/P0.6/P1.5 manuscript | ledger + integration spec DONE; LaTeX edits delegated | gate PASS | 686bd9a |
| **P0.1 full LOSO results** | **PENDING GPU-hours (896396/896397 RUNNING)** | — | — |

P1.2 headline (verified): neural posterior-KL ruler vs MC-truth Pearson **0.999** / MAE **0.046 nats** /
calib slope 0.884; kNN rank-correlated (0.948) but ~2.5× magnitude-biased; capacity h=2 underfits, h=32 best.
