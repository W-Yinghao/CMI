# CIGL_67 — Functional CMI confirmation (seeds 0/1/2): alignment controllable, reliance NOT reduced

```
Method-level judgment (full-LOSO x seeds 0/1/2, BNCI2014_001 + BNCI2015_001). fcigl_align eta {0.01, 0.05} vs
FROZEN CIGL_65 (seeds 0/1/2). 126 fold rows, replay_ok 126/126, random_subspace control ~0, leakage below ERM.
Projector firewall: source-TRAIN only (excludes source-val + target), k=2, detached. Hierarchical bootstrap
(pooled dataset->seed->fold; per-dataset seed->fold), n_boot=4000. Paired (FCIGL - old CIGL) per (dataset,seed,fold).
```

## Headline (honest)
The seed0 screen suggested fcigl_align reduced R3 reliance on 2015. **That did NOT survive seeds 0/1/2.**
- **fcigl_align RELIABLY reduces the task-head alignment scalar** (the CIGL_66 metric) — significant on both
  datasets, both η, with **no task cost**.
- **But it does NOT reduce R3 functional reliance** — the (FCIGL − CIGL) R3 task_drop delta is **not significant**
  (CI includes 0) pooled and on both datasets. The seed0 R3 reduction on 2015 was seed-specific and did not
  replicate.
- Per the PM failure rule ("R3 improvement disappears → downgrade to unstable screen"): **the functional-CMI
  reliance claim is NOT supported at method level.**

## Paired (FCIGL − old CIGL) hierarchical bootstrap 95% CIs (sig = excludes 0)

**fcigl_align_eta0.01:**
| quantity | pooled | 2a | 2015 |
|---|---|---|---|
| target_bacc | +0.003 [−0.003,+0.009] **ns** | +0.001 ns | +0.004 ns |
| graph_kl | +0.166 [+0.129,+0.196] **sig↑** | +0.175 sig | +0.159 sig |
| node_kl | +0.158 [+0.114,+0.192] **sig↑** | +0.123 sig | +0.183 sig |
| **R3_task_drop_k2** | +0.001 [−0.014,+0.019] **ns** | −0.004 ns | +0.005 ns |
| **task_head_alignment_k2** | −0.117 [−0.219,−0.010] **sig↓** | −0.014 sig | **−0.194 [−0.252,−0.143] sig↓** |

**fcigl_align_eta0.05:**
| quantity | pooled | 2a | 2015 |
|---|---|---|---|
| target_bacc | +0.004 [−0.003,+0.011] ns | +0.001 ns | +0.006 ns |
| graph_kl | +0.178 [+0.135,+0.217] sig↑ | +0.177 sig | +0.178 sig |
| node_kl | +0.164 [+0.117,+0.203] sig↑ | +0.124 sig | +0.194 sig |
| **R3_task_drop_k2** | +0.003 [−0.008,+0.017] **ns** | +0.001 ns | +0.005 ns |
| **task_head_alignment_k2** | −0.180 [−0.336,−0.011] **sig↓** | −0.015 sig | **−0.304 [−0.375,−0.229] sig↓** |

## Read
1. **Alignment is controllable, robustly.** fcigl_align drives `align_k2` significantly below CIGL on both
   datasets and both η; η=0.05 pushes it further (2015 −0.30 vs −0.19). The penalty does exactly what CIGL_66
   motivated. **Task is retained** (target CI includes 0 everywhere).
2. **Reliance is NOT reduced.** The R3 task_drop delta is **ns** pooled and per-dataset for both η — controlling
   the alignment scalar does **not** move functional reliance. The seed0 2015 signal (R3 0.082→0.063) was
   unstable; across seeds the delta is ≈0 (pooled +0.001/+0.003).
3. **Measured leakage traded up vs CIGL** (graph/node_kl significantly higher than CIGL by ~0.16), though still
   well below ERM (~0.70 vs ERM 1.12–1.29) — no rebound. fcigl_align spends measured-leakage control to lower
   alignment, and gets neither lower reliance nor higher task.
4. **Controls clean:** random_subspace task_drop ≈ 0 (R3 valid); head_replay_ok 126/126 (classifier-level);
   firewall (source-train-only projector) verified.

## Verdict (method-level)
- **fcigl_align = alignment-controllable, reliance-neutral.** It confirms the alignment penalty works as an
  instrument, but **alignment control alone is NOT sufficient to reduce functional reliance** — for either η.
- **The functional-CMI reliance claim is downgraded to an unstable seed0 screen; it does not enter a method-level
  claim.** fcigl_removal_aug was already killed at seed0.

## Scientific meaning — the gap deepens one level
CIGL_65/66: global CMI reduces measured leakage but not reliance; the residual subject subspace is task-aligned.
CIGL_67 tested the natural fix (penalize that alignment). Result: **you can drive the alignment scalar down
significantly and cheaply, but reliance still does not fall.** So the measurement→control gap is not closed by
controlling the specific task-alignment metric CIGL_66 identified. Honest allowed statement:

> Functional CMI reliably reduces the task-head alignment with the residual label-conditional subject subspace
> while retaining task performance, but this reduction does NOT reliably reduce the classifier's functional
> reliance (R3 task_drop) — controlling the alignment scalar is not sufficient for reliance control.

Do NOT say: FCIGL reduces reliance / is a better decoder / beats CDAN / proves old CIGL wrong. old CIGL remains
the measured-leakage control point; FCIGL is an alignment control point that does not transfer to reliance.

## Recommendation
- **Do not promote FCIGL to a reliance-reduction method.** Stop GPU on this variant family.
- The scientific story is now stronger *as a negative*: two levels of CMI control (global leakage; task-relevant
  alignment) both fail to move functional reliance — a robust measurement→reliance gap. This is the durable
  contribution, jointly with CIGL_65/66.
- Next question (for PM): whether to (a) test a *direct reliance* objective (not an alignment proxy) — e.g.
  penalize a source-only estimate of the R3 quantity itself — or (b) freeze the CMI line here and write up the
  audit + two-level measurement→reliance gap. My lean: (b) or a single carefully-scoped (a) probe, not a sweep.

## Artifacts (`results/cigl_functional/final/`)
`functional_multiseed_metrics.csv` (126), `functional_multiseed_r3.csv`, `functional_multiseed_alignment.csv`,
`functional_vs_frozen_deltas.csv`, `functional_bootstrap_ci.csv` (30), `MANIFEST.yaml`.
Analysis: `scripts/analyze_functional_confirmation.py`.
```
```
