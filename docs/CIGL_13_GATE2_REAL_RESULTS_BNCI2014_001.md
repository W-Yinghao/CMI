# CIGL Gate-2 — Exploratory Real Evidence (BNCI2014_001, fold-0)

> **EXPLORATORY Gate-2 evidence — NOT a benchmark / accuracy / SOTA result.** This is a source-only
> diagnostic audit of whether a GraphCMINet-ERM encoder's learned graph objects carry
> label-conditional source-domain (subject) leakage. One dataset, one LOSO fold. It informs the
> Gate-2 decision; it is not a results table. No regularizer was trained, no λ swept, no target data used.

## Run provenance

| field | value |
|---|---|
| command | `python scripts/run_cigl_phase2_real_probe.py --dataset BNCI2014_001 --device cuda --seeds 0 1 2 --n_perm 50 --max_folds 1 --epochs 80 --probe_epochs 100 --gate_alpha 0.05` |
| launcher | `sbatch scripts/sbatch_cigl_gate2_bnci001.sh` (multi-partition `A100,V100,V100-32GB,A40`, default QOS) |
| SLURM job id | **875880** |
| selected partition / node | **V100** / node42 (Tesla V100-PCIE-32GB, 32 GB) — pended briefly on `QOSMaxGRESPerUser` then scheduled on V100 (A100 was heavily queued; the multi-partition list paid off) |
| runtime | ~3–4 min wall (~160 s pending + ~200 s running) |
| branch | `project/cigl-gate2-bnci001-evidence` |
| commit_hash | `2b1ba8f979b590eb953532b3518fa77419e01c2a` |
| config_hash | `c342cccf7dac` |
| environment | conda `eeg2025` — torch 2.6.0+cu124 (CUDA available), moabb 1.5.0; GraphCMINet built directly (no braindecode) |
| dataset / fold | BNCI2014_001 (4-class MI, 22 ch) / `BNCI2014_001_fold0` |
| held-out (target) subject | **1** (never used in training, feature extraction, or probe) |
| source subjects (domains) | 2, 3, 4, 5, 6, 7, 8, 9 (**8 domains**) |
| seeds | 0, 1, 2 |
| n_perm | 50 (permutation-null p-floor = 1/51 ≈ 0.0196) |
| gate_alpha | 0.05 |

**Three-way source-only split (per seed):** source (4608 trials) → encoder-train 3232 (GraphCMINet-ERM)
+ held-out probe-pool 1376; the conditional domain probe then splits the probe-pool support-aware by
(Y,D) into 960 train / 416 val.

## Definitions (binding)

- `positive_excess = kl_mean > permutation_mean` — directional only.
- **`clears_null = kl_mean > permutation_mean AND permutation_p <= gate_alpha`** (α=0.05) — the binding
  Gate-2 criterion. The within-train permutation null retrains the probe; the KL proxy is not an
  unbiased CMI.

## Per-seed leakage (held-out KL vs retrained within-train permutation null)

| seed | object | kl_mean | perm_mean | perm_p | positive_excess | clears_null |
|---|---|---|---|---|---|---|
| 0 | graph | 0.4470 | 0.0382 | 0.0196 | ✓ | ✓ |
| 0 | node  | 0.4769 | 0.0371 | 0.0196 | ✓ | ✓ |
| 0 | edge  | 0.6547 | 0.1145 | 0.0196 | ✓ | ✓ |
| 1 | graph | 0.5651 | 0.0369 | 0.0196 | ✓ | ✓ |
| 1 | node  | 0.5658 | 0.0383 | 0.0196 | ✓ | ✓ |
| 1 | edge  | 0.8432 | 0.1601 | 0.0196 | ✓ | ✓ |
| 2 | graph | 0.5097 | 0.0385 | 0.0196 | ✓ | ✓ |
| 2 | node  | 0.5217 | 0.0390 | 0.0196 | ✓ | ✓ |
| 2 | edge  | 0.8667 | 0.1507 | 0.0196 | ✓ | ✓ |

`perm_p = 0.0196` for every cell = the observed KL exceeded **all 50** permutation draws (p hits the
1/(n_perm+1) floor). **All three objects clear null in 3/3 seeds.** Edge leakage is the largest in
absolute KL; graph and node are comparable to each other.

## Map seed-stability (across the 3 seeds, vs random-map null)

| map | mean_corr | min_corr | null_q95 | above_random | degenerate | stability_p |
|---|---|---|---|---|---|---|
| node | **0.826** | 0.758 | 0.230 | **True** | False | 0.005 |
| edge | **0.125** | −0.243 | 0.057 | **True** | False | 0.005 |

The **node** leakage map is robustly seed-stable (channels leaking subject identity reproduce across
seeds). The **edge** map is *above* the random null (so not pure noise) but only **weakly** stable
(mean 0.125, with a negative min pairwise correlation) — the per-edge localization does not yet
reproduce cleanly across seeds, even though the aggregate edge leakage is strong and significant.

## Probe split diagnostics (support-aware, per seed; representative seed-0)

| n_trials (pool) | n_train | n_val | n_classes | n_domains | cells total | cells split | low-support | missing_val_domains | missing_train_domains |
|---|---|---|---|---|---|---|---|---|---|
| 1376 | 960 | 416 | 4 | 8 | 32 | 32 | 0 | [] | [] |

Every (Y,D) cell was large enough to split → all 8 source subjects appear in both probe-train and
probe-val; no domain missing from val; no low-support cells dropped.

## Strict source-only confirmation

- `exploratory = true`, `setting = "strict_source_only_DG"`
- `used_target_labels = false`, `used_target_covariates = false`
- Held-out target subject (1) excluded from encoder training, feature extraction, and the probe.

## Claude's recommended Gate-2 path — **A (proceed to full CIGL)** *(pending reviewer decision)*

Per `docs/CIGL_10` §6, Path A requires ≥2 of 3 objects with `clears_null` in ≥2/3 seeds **and** the
node or edge map above-random stable. This run **exceeds** that: **all three** objects `clears_null`
in **3/3** seeds, and **both** maps are above-random (node strongly, edge weakly).

Two honest caveats the reviewer should weigh before authorizing Phase 3:

1. **Edge-map localization is weak (0.125).** The edge *leakage* is strong and significant, but the
   per-edge "subject-fingerprint" *map* is not yet seed-stable. If CIGL keeps the edge term, the
   edge-map figure/claim should be tempered (or strengthened with more seeds/data) — the node map is
   the robust localization here. This does not block Path A but flags where the edge story is soft.
2. **One fold, one dataset.** This is fold-0 (hold-out subject 1) of BNCI2014_001 only. Before
   committing to full CIGL it is worth confirming on additional LOSO folds and ≥1 more dataset (e.g.
   SEED) that the pattern (all three clear null; node map stable) holds — to rule out a
   subject-1-specific or BNCI-specific artifact.

If the reviewer prefers a narrower, lower-risk start: the strongest, most-reproducible signals are
**graph + node** leakage with a robust node map. Edge-CMI is supported by the leakage magnitude but
not yet by map stability.

## Artifacts

Generated per-seed JSON, per-fold summary JSON, and node/edge map `.npy` live under
`results/cigl/phase2_real/` and are **gitignored** (regenerable; reproduce via the command above).
This doc is the tracked review record.
