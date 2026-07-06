# FSR_11 — RQ4 Frozen-Probe Preflight (SPEC ONLY — no probe run)

**Project FSR — Phase 3B.** A preflight specification for the (not-yet-approved) RQ4 branch-local frozen-probe run. **Nothing is executed here.** No GPU, no re-inference, no probe. This document answers the PM's 11 preflight questions from a read-only inspection of the FBCSP branches, and returns a go/no-go recommendation for the PM to decide whether a Phase-4 run is approved.

Companion artifacts (this directory): `checkpoint_inventory.csv`, `representation_dump_plan.md`, `target_label_firewall.md`, `probe_run_manifest_template.yaml`.

## Headline verdict

**RQ4 frozen-probe is BLOCKED at the checkpoint-availability gate.** The per-branch dump *machinery* is ready (the model already exposes detached branch latents), but **there are no weights to run it on**: the FBCSP-LGG F0 pipeline persisted only summary JSONs — it never `torch.save`s a checkpoint (the trainer holds `best_state` in RAM only, `cmi/train/trainer.py:554`). So RQ4 is *double-blocked*: (1) no per-branch leakage/reliance metric (the known gap), and (2) no frozen checkpoint to re-infer from. Even the "allowed" frozen re-inference cannot run until a checkpoint exists.

## The 11 preflight questions

**Q1 — Which FBCSP-LGG checkpoints exist?**
None on disk. 0 committed `.pt/.pth/.ckpt/.safetensors` on either `project/fbcsp-lgg-spatial-cmi-fusion` (39c245a) or `project/fbcsp-lgg-dualcmi-scaffold` (eb47bd0). The trainer clones `best_state = backbone.state_dict()` to CPU in memory for best-epoch restore but does not save it. F0 result JSONs (`results/fbcsp_lgg_f0_full_s012/*.json`) contain `config/classes/summary` only. Cluster `output_root` for weights is **UNKNOWN and probably empty** (the pipeline saves no weights). → checkpoints must be **located or produced** first.

**Q2 — Deterministic re-inference?**
The backbone is pure-torch, CPU-friendly. Re-inference from a fixed checkpoint in `eval()` mode with a pinned seed and the same MOABB preprocessing is deterministic. But with no checkpoint, determinism is moot until Q1 is resolved. A deterministic ERM re-fit (seeds {0,1,2}, same config as F0) that saves `best_state` would give reproducible checkpoints; re-inference from them is then deterministic (verify by a byte/│Δ│ self-replay check, as ACAR/H2CMI did).

**Q3 — Can we dump `spatial_z / graph_z / temporal_z`?**
YES. `FBCSPLGGGraph.forward_graph(x)` sets `self.last_aux = {graph_z, temporal_z, spatial_z, fused_z}` (all detached) and `self.last_gate` ([B,3] softmax) on every call. A dump hook reads `last_aux` after a forward pass. No model surgery needed.

**Q4 — Is there really `node_z`? (if not, do not write a node branch)**
`node_z` **exists** as a per-channel latent `[B, C, node_z_dim]` in the `forward_graph` return tuple (feeding a node-CMI head), but it is **NOT one of the 3 gated fusion branches**. The 3 fusion branches are **graph / temporal / spatial** (`gate3`); `permute_nodes` is a null, not a branch. The FSR ledger's "no separate node branch" refers correctly to the fusion. For RQ4 we probe the **3 fusion branches** (`graph_z/temporal_z/spatial_z`); `node_z` may be included as a clearly-labeled 4th *per-channel latent*, never as a fusion branch.

**Q5 — Is the probe fully source-only?**
Required: the domain/subject probe is trained only on source-subject frozen latents; the held-out target subject is never in the probe training set. Enforced by LOSO grouping + the firewall (`target_label_firewall.md`).

**Q6 — Target labels only for final evaluation?**
Required: the leakage probe uses **domain labels** (subject id), not task `y`. The reliance replay uses target task `y` **only** to score the final endpoint (task-drop), never to fit the probe/eraser or select anything. `target_labels_used_for_fit = NO`; `_for_eval = YES`.

**Q7 — Per-branch leakage L1 definition.**
For each fusion branch `b ∈ {graph, temporal, spatial}` and frozen latent `z_b`: (i) `leakage_auc_b` = subject/domain decode balanced-accuracy from a probe trained on source `z_b`, evaluated on held-out; (ii) `leakage_kl_b` = label-conditional posterior-KL proxy on `z_b`; (iii) `perm_p_b` = within-label permutation null. All source-only; a random-projection control at matched dim.

**Q8 — Per-branch reliance L5 definition.**
Two complementary measures per branch: (i) `ablation_task_drop_b` = bAcc(full) − bAcc(zero_`b`) using the model's existing `zero_graph/zero_temporal/zero_spatial` ablation (already validated in F0); (ii) `r3_subspace_task_drop_b` = head-replay task-drop after removing the top-k label-conditional subject subspace *of `z_b`* (source-only fit), with a random-subspace control and exact-replay firewall — the branch-specific analogue of the CIGL R3.

**Q9 — Output file schema.**
Per `(dataset, fold, seed, branch)` row: `leakage_auc, leakage_kl, perm_p, random_ctrl_auc, ablation_task_drop, r3_subspace_task_drop, random_subspace_drop, n_source_subjects, replay_ok, firewall_ok, checkpoint_sha`. Plus a manifest (`probe_run_manifest_template.yaml`) binding checkpoint SHAs, seeds, config, and the firewall flags.

**Q10 — Compute budget.**
If checkpoints are located: re-inference (forward passes to dump latents) + sklearn probes is **CPU-feasible** (the backbone is CPU-friendly; probes are linear/MLP). If a re-fit is required (Q1): ~F0 training cost — ERM, seeds {0,1,2}, 2 datasets (2a 9 folds + 2015 12 folds), small GPU or CPU. Exact wall-clock/GPU budget is set at approval; the run is bounded to ERM re-fit + dump + probe, nothing else.

**Q11 — STOP conditions.**
- No checkpoint located AND no re-fit approved → **STOP** (cannot re-infer).
- Re-inference non-deterministic (latent dump |Δ| beyond tolerance across replays) → **STOP**.
- `node_z` treated as a fusion branch, or a 4th branch asserted in the gate → **STOP**.
- Target `y` needed for probe/eraser fit or any selection → **STOP** (firewall breach).
- Any `fbdualpc` / CMI-loss training, new architecture, or hyper-parameter search → **STOP** (forbidden).
- P6 spatial-CMI treated as a result → **STOP**.

## Recommendation to the PM (decision required before any Phase-4 run)

The dump/probe/reliance machinery and schema are ready; the blocker is weights. Two options:
- **Option A — locate cluster checkpoints.** Confirm whether any FBCSP-LGG run saved `state_dict`s under `/projects/EEG-foundation-model/...`. The F0 pipeline does not, so this is likely empty; a quick cluster `find` would settle it. If found, Phase 4 is pure frozen re-inference (fully within the allowed list).
- **Option B — approve a minimal deterministic ERM re-fit.** Re-fit the *frozen ERM* FBCSP-LGG config (seeds {0,1,2}, same data/preprocessing as F0) with `torch.save(best_state)` + a latent-dump hook added, then run the source-only per-branch probe + reliance. This is an **ERM training run** — distinct from the forbidden `fbdualpc`/CMI-loss training, but still training, so it needs your explicit approval and a bounded budget.

Until you choose A or B, **RQ4 stays descriptive/blocked** (C7 `READY`: "branch-local leakage/reliance is missing"). This preflight does not run either option.
