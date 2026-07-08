# S2P_08 — P1 Pretraining Results (neutral; pretraining gate only)

**Project S2P — P1 fixed-budget subject-vs-depth frontier.** Neutral results for the **pretraining stage** of the
15-run frontier (CBraMod, T=200 h, N∈{128,256,512,1024,2048}, seeds{0,1,2}). This is the QC/convergence gate; the
**primary endpoint (downstream SHU-MI transfer + L1/L4/L5/L6) is NOT in this doc** — it is the separate downstream
audit (pending). Machine table: `results/s2p_p1_cbramod/p1_pretrain_summary.csv`. Runs @git e58038b (array 888818).

## Pretraining QC gate — 15/15 PASS
| check | result |
|---|---|
| runs completed | **15 / 15** (no missing, no NaN/Inf, no STOP-rule trip) |
| convergence (pretrain-val rel. decrease ≥ 20%) | **15 / 15** — actual **58.9%–64.4%** |
| checkpoint strict-reload param-exact | **15 / 15** |
| firewall: `target_labels_used` | **False** (all) |
| firewall: HBN 129-ch normalizer neutralized | **True** (all) |
| firewall: `div_by_100` (native µV scale) applied | **False** (all — our per-patch z-score, as designed) |
| budget exact (24000 win = 200 h, max−min ≤ 1) | verified pre-launch (loader) + load-time assertion silent |

All runs trained the identical step count (24000 windows × 50 epochs), so optimization is matched across the frontier.

## Pretrain-val reconstruction loss by N (SECONDARY, heavily caveated)
Fixed global val (n=128, shallow/general subjects; identical across all cells). Mean best-val over 3 seeds:

| N | mean best-val | SD | first-val (mean) |
|---|---|---|---|
| 128 | 0.23332 | 0.00698 | ~0.612 |
| 256 | 0.22843 | 0.00621 | |
| 512 | 0.22860 | 0.00864 | |
| 1024 | 0.22636 | 0.00631 | |
| 2048 | 0.22653 | 0.00922 | |

- **Slope = −0.00156 best-val units per log₂(N)** — a slight decrease (better reconstruction at higher N), but the
  step-to-step change is **smaller than the seed-SD (~0.006–0.009)** ⇒ **effectively flat / negligible** at the
  pretrain-val level.
- **CAVEAT (MJ-14, pre-registered):** the fixed val is shallow/general, so higher-N (more general-population)
  training is better-matched to it — this pretrain-val trend is **confounded with train↔val population alignment**
  and is **not** evidence of an allocation effect. It is a secondary diagnostic only.

## Disclosure
- No surprising failures; all cells converged strongly and uniformly (58.9–64.4% val drop), which is the expected
  behavior for a from-scratch masked-recon transformer on 24000 windows × 50 epochs.
- The pretrain-val frontier being ~flat (within seed noise) is a **weak-to-null** signal at the reconstruction level;
  it does **not** pre-empt the primary downstream endpoint (transfer/L1/L4/L5/L6), which is measured separately.

## Not in this doc (pending downstream audit)
Frozen-encoder SHU-MI target-bAcc (primary allocation slope), L1 pairwise subject separability, L4 alignment,
L5 reliance vs variance-null, L6 target consequence, random-init positive-control floor, and the
full/robust/curvature/leave-one-N-out + population-covariate analyses. Framing stays **bundled descriptive
allocation frontier** (never pure-diversity, never population-adjusted).
