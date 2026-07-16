# CMI-Trace Relaxation Ladder — execution note

Living log. Every command, job ID, failure, retry, protocol deviation, config hash + Git SHA recorded here.

## Scientific question (NOT a tune-to-positive search)
Under which relaxed information regime + readout, if any, does subject-axis erasure become beneficial, and is
the benefit specifically caused by removing subject IDENTITY (vs generic dimensionality reduction / numerical
conditioning)? A protocol ladder (L0 strict/original-head → L1 strict/fresh-head → L2 target-X-unlabeled/fresh
→ L3 oracle-global/fresh) isolates the differences from the concurrent FMScope result one at a time.

## Confirmed P0/P1 result — MUST NOT be overwritten or weakened
1. All tested domain-invariance objectives reduce measured encoder-CMI (both datasets).
2. Lower encoder-CMI does NOT imply lower exact original-head reliance (R_rel(k=2) rises for strong reducers).
3. Target effects modest on BNCI2014_001, null/negative on BNCI2015_001.

## Stage 0 — provenance
- Base branch/SHA: `agent/cmi-trace-p0p1 @ 2a7ce8f` (verified).
- Current branch: `agent/cmi-trace-relaxation-ladder`.
- Env: GPU `eeg2025` (torch 2.6.0+cu124); CPU/tests `c84c-eeg2025-v3` (torch 2.6.0 CPU, sklearn 1.8, scipy).
- SLURM: available; idle V100/V100-32GB/P100/A40/A30 + CPU partitions.
- Do NOT touch/merge/import H2CMI or OACI.

### Key existing infrastructure (reused, not rebuilt)
- **DGCNN audit npzs from P0/P1 ARE ON DISK**: 216 (BNCI2014_001) + 288 (BNCI2015_001) under
  `results/cmi_trace_p0p1/objective_comparison/<ds>/audit/*.audit.npz`. Each carries graph_z, node_z, y,
  d(=subject), source/target indices, and a VERIFIED linear task head (head-replay). → the DGCNN feature
  family ladder (L0–L3 on graph_z) runs on EXISTING artifacts, CPU-only, no regeneration.
- **No TOS EEGNet/TSMNet dumps exist** (pruned) → Stage 7 regenerates them via `tos_cmi/eeg/feature_dump.py`
  (`dump_fold`; dumps Z_source, Z_target, subject_source/target, logits, y — everything L0–L3 need).
- Erasers: `tos_cmi/eeg/erasure_baselines.py` (`leace_eraser` repo-LEACE, `inlp_eraser`, `rlace_eraser`,
  TOS `V_D` via score-Fisher). LW-LEACE + whitening-only implemented fresh in the ladder module.
- CMI ruler: `cmi/eval/conditional_subject_leakage.py` + `cmi/eval/multicapacity_probe.py` (P1.1/P1.3).
- Reliance (L0 anchor): `cmi/eval/reliance_audit.py` / `cmi/eval/leakage_removal.py` (P1.4).
- Deployment CI: `tos_cmi/eeg/deployment_ci.py` (P0.4 three-state).

### Firewall discipline (per level)
| Level | eraser fit sees | head | target Y | source-only DG? | transductive? | oracle? |
|-------|-----------------|------|----------|-----------------|---------------|---------|
| L0 STRICT_SOURCE_ORIGINAL_HEAD | source X + source subj | replay original | scoring only | yes | no | no |
| L1 STRICT_SOURCE_FRESH_HEAD | source X + source subj | fresh on source | scoring only | yes | no | no |
| L2 TARGET_X_UNLABELED_FRESH_HEAD | source X + **target X** + target group | fresh on source | scoring only | **no** | **yes** | no |
| L3 ORACLE_GLOBAL_GEOMETRY_FRESH_HEAD | whole cohort X + subj (LW-LEACE full span) | fresh subject-grouped CV | scoring only | **no** | no | **yes** |

## Stage log
- Stage 0 (provenance): DONE.
