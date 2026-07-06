# FSR_15 — RQ4 ERM-Refit Protocol (Phase 4B pre-registration)

**Project FSR — Phase 4B.** Pre-registered protocol for a strictly-bounded, **ERM-only** FBCSP-LGG re-fit whose sole purpose is to *instantiate the missing branch-local instrument* for RQ4 — trained checkpoints + branch-latent dumps — so per-branch leakage (L1) and per-branch reliance (L5) can be measured on real EEG. This is a **measurement-instrument run**, not method development. Registered before any result is seen; deviations require a dated amendment.

## Scope (hard bounds)

**Train only:** `FBCSPLGGGraph` ERM (`--configs erm:0`), reusing the exact F0 training code path (`cmi.run_loso`) so reproduction is by construction.
**Forbidden:** `fbdualpc`, spatial-CMI, graph-CMI, node-CMI, CITA, TTA-Control; new GNN/Conformer/CSP-init variants; hyper-parameter search; architecture search; any target-label fit/selection; any objective other than plain ERM. This is checkpoint-generation + branch audit, not a method.

## Instrument (minimal, additive, default-off)

The refit adds two default-off options to the F0 runner — `--save_ckpt_dir` (persist `best_state` via `torch.save`) and `--dump_latent_dir` (forward pass → collect `self.last_aux = {graph_z, temporal_z, spatial_z, fused_z}` + `last_gate`, save `.npz`). Everything else (data, model, trainer, ablation) is the unchanged F0 path, guaranteeing the same result as F0. `node_z` is dumped only if it is a stable, interpretable per-channel latent; it is **not** a fusion branch and is never written as one.

## Staged compute (do not run all seeds at once)

- **Stage 4B-0 — sanity:** 1 dataset × 1 target fold × 1 seed. Verify: checkpoint saves; latent dumps save; branch ablation runs; recompose-identity check passes (if L5 replay planned); target labels not used in training/selection/probe.
- **Stage 4B-1 — seed0 full LOSO:** BNCI2014_001 (9 folds) + BNCI2015_001 (12 folds), seed 0. Reproduce F0 branch-load direction; produce branch leakage (L1); attempt branch reliance (L5) if recomposition passes.
- **Stage 4B-2 — seeds 1/2 full LOSO:** only if 4B-1 reproduces F0 within tolerance and the PM approves. One checkpoint is not enough for a general claim; the minimum for a claim is seed0 full LOSO, preferably seeds {0,1,2}.

## Refit reproduction gate (must pass before any branch probe)

The refit ERM must reproduce the frozen F0 summary; otherwise **STOP** (no tuning to rescue). Tolerances:
- mean target bAcc within **±0.02 absolute** of F0;
- `zero_spatial` remains the **largest** ablation drop, same sign, within **±0.03 absolute**;
- graph/temporal branches remain neutral-to-starved; gate weights qualitatively match (not a hard gate);
- no target labels used for training or selection.
If not reproduced: write `refit_reproduction_summary.csv` with `reproduced=false` and record "refit failed to reproduce frozen F0 branch-load result; RQ4 remains blocked."

## Branch-local audit design

**L1 — per-branch leakage probe.** For each branch latent `z_b ∈ {graph_z, temporal_z, spatial_z, fused_z}`: train a **source-only** subject/domain probe `q(D | z_b, Y)` (train on source-train, select on source-val, evaluate the leakage metric on source held-out/source-val); report posterior-KL surrogate, domain-probe accuracy, within-label permutation null, null ratio, bootstrap CI. Compare `L1_spatial` vs `L1_graph` vs `L1_temporal` vs `L1_fused`. The target subject is an **unseen domain** and is never used as a closed-set class for the probe.

**L4 — branch load.** Reuse the existing ablation: `zero_graph`, `zero_temporal`, `zero_spatial` (+ `zero_fused` if applicable, `permute_nodes` if supported) and the gate weights; confirm the spatial branch is load-bearing.

**L5 — branch-local functional reliance.** Minimum: branch-ablation logit change, CE/NLL change, target bAcc drop, logit SymKL (this is branch *load*, not subject-specific reliance). Stronger (subject-subspace replay, only if safe): (1) learn the subject-predictive subspace on source-train `z_b`; (2) remove it with LEACE / a linear projector; (3) **without changing weights**, replay the forward pass substituting the erased `z_b`; (4) recompose the fused representation/logits; (5) score logit SymKL, CE/NLL, bAcc delta on target. **Prerequisite identity check:** original forward logits vs logits recomposed from saved `z_b` must satisfy `max|Δ| < 1e-5`; if it fails, set `branch_reliance_replay_status = NOT_AVAILABLE` (reason: no safe recomposition API / identity check failed) — do not fabricate.

**L6 — target consequence (eval only).** target bAcc delta, target NLL delta, target ECE if available, worst-subject delta, task-collapse flag, harm flag. Target labels used only for final evaluation.

## Output files (results/fsr_rq4_refit/)

```text
refit_run_manifest.csv          checkpoint_manifest.csv          refit_reproduction_summary.csv
branch_latent_manifest.csv      branch_leakage_probe.csv         branch_ablation_reproduction.csv
branch_reliance_replay.csv      rq4_branch_local_results.json    rq4_target_label_firewall.json
```
Docs: `FSR_16_RQ4_REFIT_RESULTS.md` (refit + reproduction), `FSR_17_RQ4_BRANCH_LOCAL_AUDIT.md` (L1/L4/L5/L6 audit).

## Stopping rules (any one → STOP)

```text
1  F0 config cannot be determined.
2  refit target bAcc deviates materially from F0 summary.
3  zero_spatial is no longer the largest / a primary ablation drop.
4  training or probe uses target labels.
5  hyper-parameters changed after seeing a target result.
6  checkpoint saving is incomplete.
7  latent dump cannot be bound to checkpoint/config.
8  recomposed logits do not reproduce original logits (identity check fails) — L5 replay off.
9  probe results reported for a positive subset only.
10 any CMI / fbdualpc / spatial-CMI training is started.
```

## Firewall (recorded per artifact, `rq4_target_label_firewall.json`)
`target_y_used_for_training=false`, `_for_selection=false`, `_for_probe_fit=false`, `_for_subspace_fit=false`, `_for_early_stopping=false`, `_for_final_eval_only=true`; probe trained on source subjects only; target subject held out.

## Manuscript impact (deferred until results exist)
Until 4B produces per-branch L1/L5, no RQ4 claim changes. On success, §6 reframes from "final blocked" to "we ran a pre-registered ERM refit to instantiate the branch-local instrument," and the claim updates to whatever the audit shows (positive or negative). On failure to reproduce/replay, §6 states the refit was attempted and branch-local claims remain unlicensed — still stronger than "no checkpoint, so blocked."
