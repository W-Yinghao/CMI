# CIGL Phase 3A-J — Fixed-Config Multi-Fold Confirmation Results (BNCI2014_001)

> **EXPLORATORY confirmation — NOT a benchmark / SOTA result.** A replication test of ONE fixed
> source-only candidate (`graph_node_010`) across BNCI2014_001 LOSO folds. No λ grid, no new configs, no
> edge term, no second dataset. A confirmation here means the fold-0 pilot **replicates on this dataset**;
> it is not yet a cross-dataset method claim.

## 1–6. Run provenance

```bash
sbatch scripts/sbatch_cigl_phase3a_dgcnn_gn_multifold_confirmation_bnci001.sh
# -> python scripts/run_cigl_phase3a_dgcnn_gn_multifold_confirmation.py --dataset BNCI2014_001 --device cuda \
#      --folds 0 1 2 3 4 5 6 7 8 --seeds 0 1 2 --epochs 80 --probe_epochs 100 --n_perm 50 --gate_alpha 0.05
```

| field | value |
|---|---|
| SLURM job id | **876990** |
| partition / node | multi-partition `A100,V100,V100-32GB,A40` (default QOS) → **node09** |
| runtime | ~50 min (9 folds × 2 configs × 3 seeds × 80 ep + n_perm=50 graph+node audits; log mtime 15:23) |
| branch / commit_hash | `project/cigl-phase3a-dgcnn-gn-multifold-confirmation` / `d28a3a27e63eedc255db93a64dc682573afd39ed` |
| config_hash | `c3249e1ce814` |
| environment | conda `eeg2025`, torch 2.6.0+cu124 (CUDA) |
| dataset / folds | BNCI2014_001 / LOSO folds 0–8 (held-out subjects 1–9) |

## 7–8. Fixed configs (no λ grid was run)

| config | λ_g | λ_node | λ_edge |
|---|---|---|---|
| `erm_fixed` | 0.000 | 0.000 | 0.000 |
| `graph_node_010` | 0.010 | 0.010 | 0.000 |

Exactly two FIXED configs; **no λ grid, no new configs, no edge term**.

## 9. Fold grouping

- **`fold0_dev`** — fold-0 selected `graph_node_010` in Phase 3A-I, so it is reported **separately** and
  is **not** counted in the primary decision (it passes: graph −48% / node −42%, source drop −0.008).
- **`folds1_8_confirmation`** — the **PRIMARY** decision set (subjects 2–9).
- **`all_folds_descriptive`** — folds 0–8 pooled (descriptive only; decision A).

## 10. Per-fold table (ERM vs graph_node_010; `d` = drop vs ERM; `tgt`/`drop` evaluation-only)

| fold (subj) | ERM src | reg src (d) | ERM tgt | reg tgt (d) | graph KL e→r (red) | node KL e→r (red) | reg clears (g/n) | PASS |
|---|---|---|---|---|---|---|---|---|
| 0 (dev, s1) | 0.458 | 0.466 (−0.008) | 0.431 | 0.447 (−0.016) | 1.26→0.66 (48%) | 0.52→0.30 (42%) | 3/3 | ✓ |
| 1 (s2) | 0.501 | 0.516 (−0.015) | 0.277 | 0.279 (−0.002) | 1.38→0.89 (36%) | 0.60→0.42 (31%) | 3/3 | ✓ |
| 2 (s3) | 0.457 | 0.455 (+0.002) | 0.406 | 0.409 (−0.003) | 1.24→0.52 (58%) | 0.47→0.27 (43%) | 3/3 | ✓ |
| 3 (s4) | 0.488 | 0.495 (−0.007) | 0.291 | 0.288 (+0.003) | 1.35→0.87 (35%) | 0.57→0.38 (32%) | 3/3 | ✓ |
| 4 (s5) | 0.508 | 0.513 (−0.004) | 0.260 | 0.259 (+0.001) | 1.26→0.67 (47%) | 0.50→0.32 (36%) | 3/3 | ✓ |
| 5 (s6) | 0.500 | 0.498 (+0.002) | 0.271 | 0.285 (−0.014) | 1.29→0.75 (42%) | 0.56→0.36 (36%) | 3/3 | ✓ |
| 6 (s7) | 0.505 | 0.510 (−0.006) | 0.236 | 0.231 (+0.005) | 1.39→0.72 (48%) | 0.61→0.36 (41%) | 3/3 | ✓ |
| 7 (s8) | 0.468 | 0.462 (+0.006) | 0.403 | 0.439 (−0.036) | 1.28→0.75 (42%) | 0.52→0.28 (45%) | 3/3 | ✓ |
| 8 (s9) | 0.465 | 0.457 (+0.008) | 0.366 | 0.381 (−0.014) | 1.23→0.68 (44%) | 0.45→0.30 (33%) | 3/3 | ✓ |

(ERM clears the null g3/n3 in every fold too; `target_eval` is evaluation-only.)

## 11. Pass/fail criteria (PRIMARY = folds 1–8)

| # | criterion | threshold | result |
|---|---|---|---|
| 1 | ERM baseline adequacy (src ≥0.45, ≥2/3 seeds) | ≥6/8 folds | **8/8** ✓ |
| 2 | ERM leakage target exists (graph or node clears null, ≥2/3 seeds) | ≥6/8 folds | **8/8** ✓ |
| 3 | regularizer reduces graph or node KL ≥30% (≥2/3 seeds, fold/seed-matched) | ≥5/8 folds | **8/8** ✓ |
| 4 | source retained (reg src ≥0.45 and drop ≤0.02) | ≥5/8 folds | **8/8** ✓ |
| 5 | target guardrail (target drop ≤0.05, eval-only) | reported | 8/8 within guardrail |
| 6 | source-only firewall | required | satisfied (flags below) |
| 7 | edge absent / skipped | required | satisfied |

## 12–13. Primary decision (folds 1–8) and dev fold

**`folds1_8_confirmation` = CONFIRMED (decision A)**: criteria 1–4 hold in **8/8** confirmation folds
(needs 6/8, 6/8, 5/8, 5/8). `fold0_dev` passes too but is **reported separately and excluded** from the
primary aggregate. (`all_folds_descriptive` also A, descriptive only.)

## 14. Edge — explicitly skipped (not faked)

`edge_regularization_used=false`, `edge_logits_dynamic=false`, `edge_audit_skipped=true`,
`edge_skip_reason="static/shared adjacency: edge_logits=None; no per-sample edge object"`.

## 15. Firewall flags

`used_target_labels_for_training=false`, `used_target_labels_for_selection=false`,
`used_target_covariates=false`, `target_eval_is_evaluation_only=true`, `selection_uses_target_eval=false`,
`confirmation_label_selection_uses_target_eval=false`. (Configs are fixed — no selection step at all.)

## 16. Recommended decision — **A: CONFIRMED on BNCI2014_001** *(pending reviewer)*

The fixed candidate `graph_node_010` **replicates across all 8 confirmation folds** (and the dev fold):
it reduces graph KL **35–58%** and node KL **31–45%** while **retaining the source task** (|source drop|
≤ 0.015, every reg fold ≥ 0.45) and staying within the target guardrail. The Phase 3A-I pilot was **not a
fold-0 artifact** — the task-preserving graph/node leakage reduction is reproducible across every LOSO
subject of BNCI2014_001. Per the reviewer's rules → **Decision A: confirmed; a second-dataset confirmation
or method framing may be considered.**

**Honest caveats (keep the claim bounded):**
- **Partial controllability, consistently — not erasure.** Reduced leakage **still clears the null** in
  **every** fold (`reg clears g3/n3` throughout); KL drops ~30–58% but the subject fingerprint is dented,
  not removed. The claim is "significant, reproducible *reduction* at no task cost," **not** "leakage
  eliminated."
- **Modest baseline.** Source bAcc ~0.46–0.51 across folds — consistent and above the 0.45 floor, but a
  weak-decoder regime, not strong-accuracy MI.
- **One dataset, one fixed config.** This is BNCI2014_001 only, with the single pilot-selected
  `graph_node_010` (no λ exploration). A method claim needs a **second dataset**; this is confirmation on
  one dataset, not a cross-dataset/SOTA result.
- **Graph/node only.** Edge-CMI remains out of scope (DGCNN static adjacency).

**Next authorized step would be (reviewer-gated): a second-dataset confirmation** of the same fixed
`graph_node_010`, graph/node-only, source-only, edge-skipped protocol — before any method-paper framing.
No edge-CMI, no λ grid, no SEED/DEAP/SOTA without explicit authorization. Generated per-fold JSON are
gitignored; this doc is the tracked record.
