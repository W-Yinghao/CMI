# CIGL_64 — R2 multi-seed gate (seeds 0/1/2, BNCI2014_001 + BNCI2015_001, full-LOSO)

```
Scientific level: method-level judgment (full-LOSO x seeds 0/1/2).
Commit ee9e616 (= ab7af0d + seed-param sbatch; NO method-semantics change). Jobs 883118/883120 (s0),
883206/883207 (s1), 883203/883204 (s2). 315 (fold,method,seed) runs; 0 NaN, 0 crashes.
Firewall: strict source-only; target eval-only; firewall cross-check passed all folds. Same DGCNN adapter.
```

## A) Task/leakage Pareto (mean ± std over seeds; ✓ = non-dominated)

**BNCI2014_001 (2a, chance 0.25):**
| method | target | graph_kl | node_kl | frontier |
|---|---|---|---|---|
| erm | 0.327±0.003 | 1.298±0.011 | 0.533±0.036 | **dominated by all 4** |
| **cigl_graph_node** | 0.334±0.001 | 0.724±0.042 | **0.332±0.015** | ✓ |
| dann | 0.330±0.003 | 0.369±0.016 | 0.342±0.014 | ✓ |
| cond_dann | 0.329±0.010 | **0.356±0.027** | 0.337±0.014 | ✓ |
| cdan | 0.347±0.008 | 0.489±0.028 | 0.371±0.011 | ✓ |

**BNCI2015_001 (2015, chance 0.50):**
| method | target | graph_kl | node_kl | frontier |
|---|---|---|---|---|
| erm | 0.589±0.004 | 1.166±0.037 | 0.589±0.034 | ✓ |
| **cigl_graph_node** | 0.584±0.001 | 0.399±0.060 | **0.285±0.024** | ✓ |
| dann | 0.566±0.017 | 0.506±0.237 | 0.427±0.034 | **dom by cigl, cond_dann** |
| cond_dann | 0.575±0.007 | 0.452±0.018 | 0.395±0.034 | **dom by cigl** |
| cdan | 0.585±0.005 | 0.373±0.086 | 0.444±0.043 | ✓ |

**Stable reads (seed std small):**
- CIGL reduces measured leakage vs ERM, stable across both datasets × 3 seeds: graph −44% (2a) / −66% (2015),
  node −38% / −52%; **target Δ ≈ 0** (2a +0.007, 2015 −0.005). Meets the PM's **strong-pass** control criteria
  (leakage down, task preserved, non-dominated on both, not carried by one seed/dataset).
- **CIGL is non-dominated on BOTH datasets, all 3 seeds.** Dominates ERM (2a) and DANN + cond-DANN (2015).
  **Lowest node_kl on both.**
- **No adversarial baseline stably dominates CIGL.** On 2a the adversarial methods reach lower *graph* leakage
  (0.36–0.49 vs 0.72) at lower source task; on 2015 CIGL dominates 2/3 of them. **CDAN** is the strongest
  comparator (competitive on both, never dominated by CIGL, never dominates CIGL). DANN's 2a-vs-2015 graph_kl is
  unstable (2015 std 0.237).

## B) R3 reliance (ERM vs CIGL; head-replay 100% for ERM+CIGL, both datasets, both abs-1e-5 and rel-1e-4 tol)

Target task_drop when the k-dim source-fit subject subspace is removed (mean ± std over seeds×folds; n=27 [2a] / 36 [2015]):

| conditioning·k | 2a ERM | 2a CIGL | 2015 ERM | 2015 CIGL |
|---|---|---|---|---|
| label_conditional **k2** | +0.002±0.007 | +0.016±0.021 | +0.022±0.035 | +0.055±0.082 |
| label_conditional k8 | +0.043±0.050 | +0.058±0.055 | +0.069±0.096 | +0.087±0.098 |
| random_subspace k2 (ctrl) | −0.004±0.014 | −0.002±0.013 | +0.000±0.018 | +0.002±0.011 |

- **The reliance-reduction hypothesis (ERM task_drop > CIGL) FAILS on both datasets, all seeds, at k2 and k8** —
  CIGL's task_drop ≥ ERM's throughout. Per the PM's decision rule: **reliance-reduction claim KILLED →
  conclusion is the measurement→control gap.**
- Controls clean (random_subspace ≈ 0). Magnitudes small with large std (esp. 2015 ±0.08) → the subject subspace
  is only weakly load-bearing; the reliable signal is the *direction* (CIGL never relies less than ERM).

## Head-replay availability (honest, corrects the seed0 "every fold" overclaim)
- **ERM + CIGL: 100% head-replay** (27/27 on 2a, 36/36 on 2015) under **both** the absolute 1e-5 and a relative
  1e-4 tol → the ERM-vs-CIGL reliance is entirely classifier-level.
- Adversarial baselines on 2015: some folds fail the *absolute* 1e-5 replay tol (cond_dann 10/12, dann 5/12,
  cdan 2/12 at seed0) because adversarial training yields large, less-conditioned `graph_z` activations →
  float32 accumulation error > 1e-5 in the (genuinely linear) head. Fail-closed → R3 probe-fallback for those.
  A **relative** tol (max|Δ|/(max|logits|+1) ≤ 1e-4) accepts them (recomputable from stored fields, no rerun);
  it does not affect the primary ERM-vs-CIGL result.

## Verdict (method-level, seeds 0/1/2)
- **CIGL is a legitimate bounded CMI-specific control point:** stably non-dominated on both datasets, dominates
  ERM + plain/conditional adversarial DANN, best node leakage, task preserved — not stably dominated by any
  baseline (CDAN is the near-peer).
- **CIGL does NOT reduce the task classifier's reliance on the subject subspace** (R3 negative, stable). This is
  a **central limitation**, not an appendix.
- **Main scientific message:** *CMI can robustly measure and partially control a label-conditional leakage proxy,
  but reducing measured leakage does not automatically reduce the classifier's functional reliance on it —
  the measurement→control gap.*

## Pending
- **R1 hardened n_perm=1000** (CPU 883209, ERM+CIGL+CDAN): running/slow on a contended CPU node (confirmatory
  only; leakage already significant — n_perm=200 → p=0.005, 0/200 exceed; n_perm=50 → p=0.0196 all folds).
- **Positioning** (PM decides): provisional *audit + bounded measured-leakage control + measurement→control gap*
  is confirmed by the multi-seed evidence.
```
```
