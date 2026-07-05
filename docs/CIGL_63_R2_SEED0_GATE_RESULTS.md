# CIGL_63 — R2 seed0 gate results (BNCI2014_001 + BNCI2015_001, full-LOSO, seed 0)

```
Scientific level: SEED0 SCREENING (not method-level judgment; that needs seeds 0/1/2).
Firewall: strict source-only; target eval-only; head-replay firewall cross-check passed on ALL folds.
Backbone: dgcnn_forward_graph_adapter, IDENTICAL path for all 5 methods (stop-condition #5 satisfied).
Commit ab7af0d. Jobs 883118 (2a) + 883120 (2015). n_perm=50 in-run (all folds p=0.0196, the n_perm=50 floor).
```

Five methods on the same adapter: **erm · cigl_graph_node (λg=λn=0.010) · dann · cond_dann (cdann) · cdan**.
All 5 trained + audited on every LOSO fold (2a: 9 folds; 2015: 12 folds). No NaN, no crashes, no stop-conditions.
Head-replay `replay_ok=True` on every fold (real max_abs_diff ≈ 1e-6, ≤ 1e-5 tol) → R3 ran at the **classifier**
(head-replay) level, firewall cross-check passed on all folds.

## A) Task/leakage Pareto (fold-mean; chance = 0.25 [2a, 4-class] / 0.50 [2015, 2-class])

**BNCI2014_001 (2a)** — target task near chance for ALL methods (hard source-only 4-class LOSO):
| method | target_bacc | source_bacc | graph_kl | node_kl | frontier |
|---|---|---|---|---|---|
| erm | 0.329 | 0.489 | 1.285 | 0.486 | dominated |
| **cigl_graph_node** | 0.334 | **0.489** | 0.741 | **0.319** | ✓ |
| dann | 0.326 | 0.418 | 0.359 | 0.353 | ✓ |
| cond_dann | 0.320 | 0.426 | **0.330** | 0.339 | ✓ |
| cdan | 0.341 | 0.430 | 0.511 | 0.363 | ✓ |

**BNCI2015_001 (2015)**:
| method | target_bacc | source_bacc | graph_kl | node_kl | frontier |
|---|---|---|---|---|---|
| erm | 0.592 | 0.704 | 1.115 | 0.547 | ✓ |
| **cigl_graph_node** | 0.586 | 0.698 | 0.314 | **0.255** | ✓ |
| dann | 0.574 | 0.640 | 0.384 | 0.384 | dominated (by cigl) |
| cond_dann | 0.569 | 0.633 | 0.427 | 0.360 | dominated (by cigl) |
| cdan | 0.592 | 0.662 | 0.313 | 0.405 | ✓ |

**Read (honest):**
- CIGL **reduces the measured label-conditional subject leakage substantially while retaining task**: graph_kl
  −42% (2a) / −72% (2015); node_kl −34% / −53%; task within ±0.006 of ERM on both. A real *control-on-the-Pareto*
  result.
- CIGL is **non-dominated on both** datasets. It **strictly dominates ERM on 2a**, and **dominates DANN and
  cond-DANN on 2015**. CIGL has the **lowest node_kl on both** datasets.
- BUT it is **not a clean win over the adversarial baselines**: on **2a** DANN/cond-DANN reach *lower graph
  leakage* (0.33–0.36) than CIGL (0.741), though at lower source task (0.42 vs 0.49). CIGL's edge is best source
  retention + lowest node leakage, not lowest graph leakage. The PM's "strongest result" (CIGL lower leakage than
  ERM **and** all adversarial baselines) is **not** achieved on 2a; it is nearly achieved on 2015 (CIGL dominates
  2/3 adversarial, ties cdan on graph, wins node).

## B) R3 reliance — is the CMI-measured subject subspace load-bearing for the task? (head-replay, firewall=1.0, n=folds)

Target-task drop when the k-dim **source-fit** subject subspace is projected out (>0 ⇒ removal hurts task ⇒ reliance):

| conditioning · k | 2a ERM | 2a CIGL | 2015 ERM | 2015 CIGL |
|---|---|---|---|---|
| label_conditional k1 | +0.006 | +0.016 | +0.015 | +0.061 |
| label_conditional **k2** | +0.004 | +0.023 | +0.020 | +0.082 |
| label_conditional k4 | +0.009 | +0.038 | +0.062 | +0.090 |
| label_conditional k8 | +0.040 | +0.058 | +0.068 | +0.093 |
| marginal_domain k2 (ctrl) | −0.001 | +0.004 | +0.000 | +0.062 |
| random_subspace k2 (ctrl) | −0.004 | −0.003 | −0.004 | +0.001 |

**Read (honest):**
- The controls behave correctly: **random_subspace removal ≈ 0** on both (the machinery isn't just destroying
  capacity), and **label_conditional > marginal_domain > random** — the label-conditional subject subspace does
  carry above-random task signal.
- **The reviewer-critical screen FAILS on both datasets:** the hypothesis is *ERM relies more than CIGL*
  (ERM task_drop > CIGL). Instead **CIGL's task_drop ≥ ERM's at every k on both datasets** — CIGL's task is, if
  anything, slightly *more* entangled with the residual subject subspace. Removing the subspace also cuts subject
  decodability *more* for CIGL (2a +0.092 vs ERM +0.107 — comparable; 2015 +0.074 vs +0.045).
- Magnitudes are small (≤ 0.09 target-bAcc even at k8). **The measured-leakage reduction (Pareto) does NOT
  translate into reduced functional reliance of the task classifier on the subject subspace.** This is direct
  evidence of the project's **measurement→control gap**: you can move the KL-proxy leakage a lot, without moving
  what the task classifier actually leans on.
- **Caveat:** on 2a the target task is near chance (~0.33), which weakens any reliance readout; 2015 (target ~0.59)
  is a cleaner test and shows the same direction.

## Verdict (seed0 screen — reviewer decides)
- **Positive, bounded:** CIGL is a legitimate, non-dominated Pareto member that reduces measured leakage while
  retaining task, dominating ERM (2a) and the plain/conditional adversarial baselines (2015). This supports the
  *bounded partial control* claim.
- **Negative, clean:** CIGL does **not** reduce the task classifier's *reliance* on the CMI-measured subject
  subspace vs ERM (R3 screen fails on both datasets). The durable contribution remains the **audit + measurement→
  control gap**, not a reliance-reducing decoder.

## Still pending
- **R1 hardened audit (n_perm=1000)** on ERM + CIGL, recomputed from saved features (no retrain) — confirmatory
  tightening of the already-significant (p=0.0196 @ n_perm=50, all folds) leakage; queued as a CPU job.
- **Seeds 1/2** — PM decision. Recommendation below (see report).
- **Infra note:** BNCI2015_001 `.mat` in the datalake `~bci/` path are owner-only (`tmaye`, not group-readable);
  worked around read-only via a world-readable copy at `database/data-sets/001-2015` + `MNE_DATASETS_BNCI_PATH`
  override (data access only — no science/firewall change). The datalake perms should be fixed at source.
```
```
