# Wave 0 — evidence packet (facts only)

Neutral record of Wave-0 experimental outputs. No interpretation of how a manuscript should present these.
Branch `exp/h2cmi-wave0-mechanism` (pushed, `fefca2b`), off frozen terminal `5bc9bf0`
(tag `h2cmi-review-p0-terminal`). Pre-registrations frozen before their aggregates.
Env: `icml`/`eeg2025`. Source seeds {0,1,2} unless noted. Cluster bootstrap 10k.

## Provenance

| wave | pre-reg | result commit | archived-insert commit | checksums |
|---|---|---|---|---|
| W0.1 | `WAVE0_FROZEN.md` (`e7b7f88`) | `e75a0b0` | `5f17874` | `B1A.sha256` |
| W0.2 | `WAVE0_FROZEN.md` | `658baf8` | `a256e2a` | `w02_w05.sha256` |
| W0.3 | `W0.3_MECH_APPENDUM.md` (`93ac368`) | `aa5031d` | `a1c9d07` | `w03.sha256` |
| W0.4 | `W0.4_MECH_APPENDUM.md` (`3656190`) | `c79a041` | `81696c5` | `w04.sha256` |
| W0.5 | `WAVE0_FROZEN.md` | `658baf8` | `a256e2a` | `w02_w05.sha256` |

Terminal anchor: REVIEW_P0 W2 primary `P = −0.1439`, `G = −0.0200`, computed at `278fc85`/analyzer `9a35cc9`.

## W0.1 — deterministic W2 re-evaluation (n=75 subjects)

- Self-replay: 27/27 (9 branches × 3 seeds) prediction hashes bit-identical. Decomposition residual ≤ 3.7e-17.
- G = −0.0201, P = −0.1439, `fixed_iterative − joint_geometry` = +0.0187 (re-confirm terminal to 4 dp).
- Per-stage recall (identity geometry), primary protocol:

  | prior | W | N1 | N2 | N3 | REM |
  |---|---|---|---|---|---|
  | Unif | 0.924 | 0.288 | 0.885 | 0.534 | 0.639 |
  | π_J | 0.952 | 0.006 | 0.817 | 0.483 | 0.307 |

- Files: `wave0_w2.report.json`.

## W0.2 — fixed-reservoir prevalence intervention (90 (pair,subject) / 72 (dataset,subject))

- Eval-embedding displacement from q=0.5: pooled 0.08, FRSC 0.42, fixed_iterative/joint 0.80, oracle 2.0.
- FRSC BA across q ∈ {0.1,0.5,0.9} = [0.520, 0.524, 0.519].
- Files: `w02_displacement.csv`, `w02_utility.csv`, `wave0_v2p.report.json`.

## W0.3 — same-session decision-prior decomposition (n=75; residual 0; consistency 0)

| term | cross-night | same-session |
|---|---|---|
| P_J | −0.144 [−0.160,−0.128] | −0.134 [−0.150,−0.118] |
| metric-mismatch B_E(ρ_E)−B_E(Unif) | −0.162 [−0.181,−0.144] | −0.161 [−0.179,−0.143] |
| transfer B_E(ρ_A)−B_E(ρ_E) | −0.0055 [−0.016,+0.005] NS | 0.0000 |
| π_J-deviation B_E(π_J)−B_E(ρ_A) | +0.024 [+0.007,+0.041] | +0.028 [+0.012,+0.043] |

- Directionality P_AB−P_BA = −0.007 [−0.017,+0.002] (NS). Per-stage recall table in `W0.3_RESULTS.md`.
- Files: `w03_mechanistic_decomposition.csv`, `w03_stage_recall.csv`, `w03_stage_decomposition.csv`,
  `w03_prior_quality.csv`, `wave0_priordecomp.report.json`.

## W0.4 — batch-size audit (n=75; QC green; residual 5.6e-17)

| n | P_J | metric-mismatch | transfer/noise | π_J-deviation | H(π_J) | min π_J | TV(π_J,ρ_A_full) | missing |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | −0.092 | −0.162 | −0.010 | +0.080 | 1.442 | 0.085 | 0.253 | 0.92 |
| 32 | −0.110 | −0.162 | −0.006 | +0.058 | 1.367 | 0.060 | 0.251 | 0.53 |
| 64 | −0.124 | −0.162 | −0.005 | +0.043 | 1.292 | 0.038 | 0.261 | 0.34 |
| 128 | −0.133 | −0.162 | −0.006 | +0.035 | 1.234 | 0.023 | 0.263 | 0.23 |
| 256 | −0.137 | −0.162 | −0.005 | +0.030 | 1.188 | 0.014 | 0.271 | 0.17 |

- Endpoint Δ_{256−16}: P_J = −0.0445 [−0.054,−0.036]; π_J-deviation = −0.0500 [−0.061,−0.040];
  metric-mismatch = 0.0; TV(π_J,ρ_A_full) = +0.018 [+0.001,+0.034].
- Files: `w04_by_n.csv`, `w04_endpoints.csv`, `wave0_batchsweep.report.json`.

## W0.5 — metric dependence (analyzer-only over W0.2 + prior-decomp)

- V2P/FRSC ordAcc(oracle-q) − ordAcc(unif): q=0.1 → +0.0225 [+0.014,+0.032]; q=0.9 → +0.0333
  [+0.022,+0.046]; q=0.5 → ≈0. V2P/joint → ≈0 (NS).
- Sleep natural ordinary accuracy: argmax prior under BA = Unif; argmax under ordinary accuracy = Unif
  (BA(Unif) 0.66 vs ρ_E 0.49; OrdAcc(Unif) 0.77 vs ρ_E 0.73).
- Files: `w05_v2p_switch.csv`, `w05_sleep_lens.csv`, `wave0_w05.report.json`, `wave0_metricswitch.report.json`.

## QC status

All waves: fan-out addressed by real subject id (regression-gated, `test_wave0_fanout.py` 5/5); weighted
tests 10/10; provenance strict (196 reused seed-0 bundles clean). W0.1/W0.3/W0.4 decomposition residual
≤ 5.6e-17; per-(subject,seed) invariance of `B_E(Unif)`, `metric_mismatch`, `eval_hash` (W0.4).
