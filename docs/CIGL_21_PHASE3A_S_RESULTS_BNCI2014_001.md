# CIGL Phase 3A-S — Known-Good MI Decoder Sanity Results (BNCI2014_001, fold-0)

> **EXPLORATORY diagnostic — NOT a benchmark / SOTA result.** One dataset, one LOSO fold, 3 seeds,
> source-only, ERM only (no CMI regularization). Answers the prerequisite question from Gate-3A-R: can
> known-good MI decoders learn the same fold that GraphCMINet-ERM cannot?

## 1–6. Run provenance

```bash
sbatch scripts/sbatch_cigl_phase3a_backbone_sanity_bnci001.sh
# -> python scripts/run_cigl_phase3a_backbone_sanity.py --dataset BNCI2014_001 --device cuda --fold 0 \
#      --seeds 0 1 2 --epochs 80 --probe_epochs 100 --leak_n_perm 10
```

| field | value |
|---|---|
| SLURM job id | **876396** |
| partition / node | multi-partition `A100,V100,V100-32GB,A40` (default QOS) → scheduled on **nodeaudible01** |
| runtime | ~<5 min (queued + ran within one watcher cycle; small decoders on a 3232-trial source) |
| branch / commit_hash | `project/cigl-phase3a-backbone-sanity` / `649ce933bafbfac292534f6f0cce49e7d6210ba0` |
| config_hash | `b89ac551dbac` |
| environment | conda `eeg2025`, torch 2.6.0+cu124 (CUDA) |
| dataset / fold / held-out target | BNCI2014_001 / fold-0 / **subject 1** (never used in training/extraction/probe) |
| source subjects | 2–9 (8 domains); enc-train 3232, probe-pool 1376 |
| seeds / classes / chance / floor | 0,1,2 / 4 / **0.25** / success floor **0.45** |

## 7–8. Candidate table (ERM only; per-candidate means over 3 seeds)

`src` = `source_probe` (held-out **source** trials; the **only** metric used for success). `tgt` =
`target_eval`, **evaluation-only**. bAcc / macro-F1 both shown.

| candidate | family | train bAcc/F1 | **src bAcc/F1** | tgt bAcc/F1 (eval-only) | train−src gap | src per-seed |
|---|---|---|---|---|---|---|
| `graphcmi_current_ref` | GraphCMINet (CIGL ref) | 0.392 / 0.385 | **0.334 / 0.327** | 0.328 / 0.312 | +0.058 | 0.337, 0.320, 0.344 |
| `eegnet` | EEGNet CNN | 0.608 / 0.605 | **0.524 / 0.519** | 0.564 / 0.533 | +0.084 | 0.536, 0.531, 0.504 |
| `shallow_convnet` | ShallowConvNet | 0.848 / 0.848 | **0.561 / 0.560** | 0.539 / 0.490 | +0.287 | 0.555, 0.560, 0.568 |
| `deep_convnet` | DeepConvNet | 0.595 / 0.581 | **0.519 / 0.502** | 0.526 / 0.464 | +0.076 | 0.519, 0.519, 0.518 |
| `dgcnn` | DGCNN (graph baseline) | 0.745 / 0.745 | **0.458 / 0.457** | 0.431 / 0.425 | +0.287 | 0.481, 0.451, 0.442 |

## 9–11. Success decision (source-only)

- **`selected_successful_models` = `['eegnet', 'shallow_convnet', 'deep_convnet', 'dgcnn']`** — all clear
  `source_probe` bAcc ≥ 0.45, judged on **source_probe only**.
- **`known_good_decoders_succeed = true`.**
- **`graphcmi_succeeds = false`** — GraphCMINet stays at `source_probe` 0.334 (≈ 4-class chance 0.25),
  **reproducing Phase 3A-R (0.334) exactly** under a different commit/config.

## 12. GraphCMINet leakage (graph reference only; light audit, n_perm=10)

Per seed `kl_mean` (permutation_p): graph 0.448/0.607/0.531, node 0.442/0.605/0.461, edge
0.863/0.884/0.939; all `permutation_p = 0.0909`. Note 0.0909 = 1/(10+1) is the **resolution floor** of a
10-permutation null, so this light audit cannot certify p<0.05 — but the `kl_mean` magnitudes match the
strong, significant leakage established earlier (Gate-2 / Phase 3A with larger n_perm). Non-graph CNNs
emit **no** leakage fields, by design. Leakage being large while task bAcc is at chance reconfirms
GraphCMINet encodes subject identity but not the label-relevant signal.

## 13. Firewall flags (verbatim from summary `meta`)

`used_target_labels_for_training=false`, `used_target_labels_for_selection=false`,
`used_target_covariates=false`, `target_eval_is_evaluation_only=true`,
`success_selection_uses_target_eval=false`. Success ranks `source_probe` only; `target_eval` is
after-the-fact. (A unit test corrupts only target labels and asserts `source_probe` and
`selected_successful_models` are unchanged.)

## 14. Recommended decision — **A** *(pending reviewer)*

**Decision A: the protocol is usable; GraphCMINet is the bottleneck.** Three known-good CNNs reach
`source_probe` 0.52–0.56 (well above the 0.45 floor and far above GraphCMINet's 0.334) on the **same**
strict source-only fold, so the fold/preprocessing/data are learnable — this is **not** Decision B. The
held-out `target_eval` for those decoders tracks source (0.53–0.56), so the learned task even transfers
to the unseen subject. Crucially, **`dgcnn` — a graph backbone — also clears the floor (0.458)**, so a
graph-based decoder *can* learn this task; the failure is **specific to GraphCMINet**, not to graph
models in general. This argues for A over **D** (graph-compatible task learning is feasible). It is not
**C** (GraphCMINet does **not** succeed).

Two honest caveats on the magnitudes (do not affect the A verdict): `shallow_convnet` and `dgcnn` show
large train−src gaps (+0.287 = some overfitting, though src still clears the floor), while `eegnet` and
`deep_convnet` generalize cleanly (gap ≈ +0.08); and `dgcnn` clears only marginally and exposes no
`forward_graph` objects, so it is not directly CIGL-leakage-compatible.

**Next authorized work would be (reviewer-gated) a graph-backbone redesign** around a known-good temporal
stem (an EEGNet/ShallowConvNet-style temporal+spatial front-end feeding the graph/node/edge structure),
so CIGL's leakage objects sit on a backbone that can actually learn the task. **The CIGL regularizer
remains NOT authorized** until a graph-compatible task backbone with a credible source-only baseline
exists. No Phase 3B, no λ sweep, no full LOSO, no SEED/DEAP pending this review. Generated per-candidate
JSON are gitignored; this doc is the tracked record.
