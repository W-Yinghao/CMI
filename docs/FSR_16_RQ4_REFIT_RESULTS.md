# FSR_16 — RQ4 ERM Refit + Reproduction (Phase 4B)

**Project FSR — Phase 4B.** Results of the pre-registered (FSR_15) ERM-only FBCSP-LGG refit that instantiates the branch-local instrument. Runner + raw dumps + scripts live on branch `project/fsr-rq4-refit`; the derived tables are copied here. GPU via SLURM; ERM-only; no target-label fit.

## Instrument (built + verified)
Two default-off additions to the *unchanged* F0 code path (`cmi/run_loso.py`): `--save_ckpt_dir` (`torch.save` best_state per fold) and `--dump_latent_dir` (per-branch latent `.npz`: `graph_z/temporal_z/spatial_z/fused_z` + gate + logits + y + d, source & target). The recompose API `head3(_fuse3(dumped branch latents))` reconstructs the forward logits exactly (identity max-|Δ| = 0.0 on the 4B-0 sanity; 4.3e-6 on strict checkpoint reload), so the L5 subspace-replay is faithful. Dumps are tagged by held-out subject (unique per LOSO fold).

## 4B-0 sanity (1 dataset × 1 fold × 1 seed) — PASS
2a target subject 1, seed 0: checkpoint + latents saved; recompose-identity 0.0; branch ablation runs; firewall clean; refit bAcc **0.476** vs F0 seed0-subj1 **0.488** (|Δ|=0.012 < 0.02); `zero_spatial` the largest drop (−21pp). All plumbing + reproduction gates pass.

## 4B-1 seed0 full LOSO — reproduces F0
21 folds (BNCI2014_001 = 9, BNCI2015_001 = 12), seed 0, exact F0 config + dump.

| dataset | refit mean target bAcc | F0 | \|Δ\| | zero_spatial mean drop |
|---|---|---|---|---|
| BNCI2014_001 (2a) | 0.358 | 0.349 | 0.009 | +0.085 (largest) |
| BNCI2015_001 (2015) | 0.599 | 0.608 | 0.009 | +0.085 (largest) |

**Reproduction gate PASS:** mean target bAcc within ±0.02 of F0 on both datasets; `zero_spatial` remains the largest ablation drop (same sign, +0.085 vs F0 −7.4/−8.8pp direction); graph/temporal branches neutral-to-starved; gate_spatial highest (0.48/0.46). **Firewall clean 21/21** (`rq4_target_label_firewall.json`: target y for final eval only; probes/subspaces source-fit).

## Artifacts
- Runner + scripts + raw dumps/checkpoints: branch `project/fsr-rq4-refit` (`scripts/run_rq4_4b1_seed0.slurm`, `analyze_rq4_branch_local.py`, `aggregate_rq4_verdict.py`; `results/fsr_rq4_refit/{ckpt,latents}`).
- Derived here: `results/fsr_rq4_refit/{branch_leakage_probe,branch_task_coupling,branch_reliance_replay,branch_target_consequence}.csv`, `rq4_branch_local_results.json`, `rq4_target_label_firewall.json`.

The refit is a valid measurement instrument (reproduces F0), so the branch-local L1–L6 audit built on it (FSR_17) is licensed.
