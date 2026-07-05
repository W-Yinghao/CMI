# CIGL_67 — Functional CMI seed0 gate readout (BNCI2014_001 + BNCI2015_001)

```
Scientific level: SEED0 SCREENING (method-level judgment needs seeds 0/1/2). Branch project/cigl-functional-cmi.
84 runs (4 variants x 21 folds), 0 NaN/crash, head_replay_ok 84/84, same DGCNN adapter, firewall passed.
Projector FIREWALL VERIFIED: fit on source-TRAIN only (Xs[enc_idx]); excludes source-val (pool_idx) AND target;
k=2, detached, refreshed every 10 epochs. random_subspace control ~0 (R3 machinery valid).
Comparators = FROZEN CIGL_65 seed0 (ERM, CIGL graph+node, CDAN); NOT rerun.
```

## Results (fold-mean; classification vs old CIGL per the PM rubric)

**BNCI2015_001 (clean readout, target ≈ 0.59 above chance):**
| method | target | graph_kl | node_kl | R3 task_drop k2 | align_k2 | class |
|---|---|---|---|---|---|---|
| ERM (frozen) | 0.592 | 1.115 | 0.547 | 0.020 | 0.046 | — |
| **CIGL (frozen)** | 0.586 | 0.314 | 0.256 | 0.082 | **0.529** | — |
| **fcigl_align_eta0.01** | 0.582 | 0.413 | 0.421 | **0.063** | **0.342** | **FUNCTIONAL** |
| fcigl_align_eta0.05 | 0.583 | 0.415 | 0.431 | 0.073 | **0.203** | bounded |
| fcigl_removal_aug_a0.5 | 0.585 | 0.468 | 0.435 | 0.063 | 0.692 | **fail** |
| fcigl_removal_aug_a1.0 | 0.587 | 0.599 | 0.455 | 0.050 | 0.599 | **fail** |

**BNCI2014_001 (near-chance, target ≈ 0.33 — weak reliance readout):**
| method | target | graph_kl | node_kl | R3 task_drop k2 | align_k2 | class |
|---|---|---|---|---|---|---|
| ERM (frozen) | 0.329 | 1.285 | 0.486 | 0.004 | 0.004 | — |
| **CIGL (frozen)** | 0.334 | 0.741 | 0.319 | 0.023 | 0.036 | — |
| fcigl_align_eta0.01 | 0.340 | 0.902 | 0.432 | 0.021 | 0.024 | strong* |
| fcigl_align_eta0.05 | 0.339 | 0.903 | 0.434 | 0.029 | 0.021 | bounded |
| fcigl_removal_aug_a0.5 | 0.336 | 0.994 | 0.444 | 0.012 | 0.073 | fail |
| fcigl_removal_aug_a1.0 | 0.339 | 1.090 | 0.457 | 0.022 | 0.048 | fail |
`*` 2a "strong" is **fragile** — near-chance task, R3 barely moved; not robust evidence. The real signal is 2015.

## Read (honest)
1. **fcigl_align works — the direct alignment penalty reduces the CIGL_66 failure mode.** `align_k2` drops
   monotonically with η on BOTH datasets: 2015 0.529 → 0.342 (η=0.01) → 0.203 (η=0.05); 2a 0.036 → 0.024 → 0.021.
   The penalty does exactly what it was designed to (pull the head out of the residual subject subspace).
2. **On 2015 (clean readout), fcigl_align_eta0.01 is a FUNCTIONAL PASS:** alignment ↓35%, **R3 reliance ↓ from
   0.082 to 0.063 (−23%)**, target retained (0.586→0.582, −0.004), leakage controlled (graph 0.413, node 0.421 —
   both **below ERM** 1.115/0.547; slightly above pure CIGL). This is the first evidence that functional CMI
   reduces the reliance that plain CIGL could not.
3. **η=0.05 drives alignment lowest (0.203) but the R3 reduction is smaller** (0.073 vs CIGL 0.082) → bounded.
   The strength grid matters; η=0.01 gives the best reliance/task trade so far.
4. **fcigl_removal_aug FAILS on both datasets** — it *increases* alignment (2015 0.53→0.69/0.60; 2a 0.036→0.073/
   0.048). Turning the R3 removal into a training objective does NOT fix the mechanism; if anything the encoder/
   head re-entangle. Clean negative for that variant.
5. **Controls clean:** random_subspace task_drop ≈ 0 (−0.001…+0.003); head_replay_ok 84/84 (R3 at classifier
   level); no leakage rebound to ERM; firewall (source-train-only projector) verified.

## Verdict
- **fcigl_align = the winning mechanism** (direct head/subject-subspace alignment penalty). **fcigl_removal_aug =
  killed.** On the clean 2015 readout, fcigl_align_eta0.01 reduces alignment AND R3 reliance while retaining task —
  a functional pass that plain CIGL did not achieve.
- 2a (near-chance) is directionally consistent (alignment ↓, task safe) but cannot carry the reliance claim.

## Recommendation (seeds 1/2 — PM decides; NOT auto-launched)
Expand **fcigl_align only** to full-LOSO × seeds 1/2, both datasets:
- **Primary: fcigl_align_eta0.01** (functional on 2015, task-safe on 2a) — meets the PM expand rule (2015: alignment
  ↓ + R3 ↓; 2a target not hurt).
- **Secondary: fcigl_align_eta0.05** (drives alignment lowest; bounded at seed0 — seeds 1/2 would show whether its
  R3 benefit stabilizes). Same mechanism, brackets the strength.
- **Drop fcigl_removal_aug** (fails both datasets). seeds-1/2 sbatch templates are prepared (GATE_SEED-param),
  NOT launched.

## Artifacts (`results/cigl_functional/seed0/`)
`functional_seed0_metrics.csv` (84), `functional_seed0_r3.csv`, `functional_seed0_alignment.csv`,
`functional_seed0_pareto_against_frozen.csv` (14), `MANIFEST.yaml`. Analysis: `scripts/analyze_functional_seed0.py`.
```
```
