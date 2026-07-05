# CIGL_65 — Evidence freeze (real-EEG method-level gate, seeds 0/1/2)

```
Scientific level: method-level judgment (full-LOSO x seeds 0/1/2, BNCI2014_001 + BNCI2015_001).
315 (fold,method,seed) runs; 0 NaN/crash; firewall passed all folds; same DGCNN adapter.
Frozen tables under results/cigl_r123/final/. This is the evidence freeze, NOT a manuscript.
```

## Frozen internal interpretation (authoritative)
1. CIGL is **not** a better EEG decoder.
2. CIGL is **not** a shortcut remover.
3. CIGL is **not** an unbiased CMI estimator.
4. CIGL does **not** eliminate leakage.
5. CIGL **does** stably reduce measured label-conditional graph/node leakage.
6. CIGL retains target task performance approximately.
7. CIGL is a **non-dominated CMI-specific control point** on the measured task/leakage Pareto frontier.
8. CIGL's measured-leakage reduction does **not** reduce classifier reliance on the leakage subspace.
9. **Main message: CMI measurement is useful and controllable as a proxy, but measured-leakage control does not
   guarantee functional-reliance control (the measurement→control / measurement→reliance gap).**

> Wording guard: we do **not** claim "CIGL relies *more* on subject leakage." The correct statement is that the
> reduced/reshaped subject-predictive subspace is **not *less*** functionally load-bearing for CIGL's classifier
> (the CIGL−ERM R3 task_drop CI excludes 0 on the positive side at k2, but the magnitude is small and weakens at k8).

## Hierarchical bootstrap CIs (95%, 4000 boots; hierarchy dataset→seed→fold [pooled] or seed→fold [per-dataset])
`results/cigl_r123/final/bootstrap_ci.csv` — paired per (dataset, seed, fold). "sig" = CI excludes 0.

| comparison | quantity | pooled point [95% CI] | sig | per-dataset |
|---|---|---|---|---|
| CIGL−ERM | **target_bacc** | +0.001 [−0.010, +0.011] | **ns** | 2a ns, 2015 ns |
| CIGL−ERM | **graph_kl** | **−0.684 [−0.791, −0.547]** | **sig↓** | 2a −0.574 sig, 2015 −0.767 sig |
| CIGL−ERM | **node_kl** | **−0.259 [−0.313, −0.181]** | **sig↓** | 2a −0.201 sig, 2015 −0.303 sig |
| CIGL−ERM | R3 task_drop k2 | +0.025 [+0.009, +0.051] | sig↑(small) | 2a +0.014 sig, 2015 +0.033 sig |
| CIGL−ERM | R3 task_drop k8 | +0.017 [+0.001, +0.034] | sig↑(pooled only) | 2a ns, 2015 ns |
| CIGL−CDAN | target_bacc | −0.006 [−0.022, +0.008] | ns | 2a ns, 2015 ns |
| CIGL−CDAN | graph_kl | +0.116 [−0.048, +0.263] | ns | **2a +0.235 sig (CIGL worse)**, 2015 ns |
| CIGL−CDAN | node_kl | **−0.108 [−0.183, −0.028]** | **sig↓ (CIGL better)** | 2a −0.039 sig, 2015 −0.159 sig |

**Reading:**
- **Control confirmed with CIs:** CIGL reduces graph_kl and node_kl vs ERM **significantly on both datasets**
  (CI excludes 0), while **target_bacc is statistically indistinguishable** from ERM (CI includes 0). Not
  accuracy-for-leakage.
- **Bounded vs adversarial:** CIGL has **significantly lower node_kl than CDAN** but **significantly higher
  graph_kl on 2a**; target equal. CIGL and CDAN are distinct, non-dominating Pareto points.
- **Reliance not reduced:** the CIGL−ERM R3 task_drop CI is positive-and-excludes-0 at k2 (small, ~+0.025),
  weakening at k8 — removing the subject subspace does **not** cost CIGL *less* task than ERM. Consistent with
  the measurement→control gap.

## Frozen tables (`results/cigl_r123/final/`)
- `multiseed_pareto.csv` (30) — per (dataset, method, seed) fold-mean target/source/graph_kl/node_kl + frontier.
- `r3_reliance.csv` (1512) — per (dataset, method[ERM,CIGL], seed, fold, conditioning, k) task_drop +
  subject_leak_drop + removal_mode + firewall_passed.
- `head_replay_check.csv` (315) — per fold max_abs_diff, max_abs_logit, replay_ok_abs / replay_ok_rel /
  primary_replay_ok.
- `bootstrap_ci.csv` (24) — the CIs above.
- `r1_hardened_nperm1000.csv` — **PENDING** (CPU job in progress; see below).
- `MANIFEST.yaml`, `data_access_note.md`.

## Head-replay (honest, corrects the earlier seed0 "every fold" wording)
- **ERM + CIGL: 100% absolute-tolerance** head-replay (63/63 each) → the reviewer-critical R3 comparison is fully
  classifier-level.
- Adversarial baselines fail the *absolute* 1e-5 tol on some folds (abs_ok: dann 43/63, cond_dann 36/63,
  cdan 57/63) because adversarial training yields large-scale logits; the head is genuinely linear, so a
  **relative** tol `max|Δ| ≤ max(1e-5, 1e-4·max(1,max|logit|))` recovers **100%** (`primary_replay_ok = 63/63`
  for all methods). This is a measurement-instrument tolerance choice, recomputed from stored fields (no rerun);
  it does not touch the primary ERM-vs-CIGL result.

## R1 hardened (n_perm=1000) — status
Confirmatory only. Two CPU attempts (883209, 883257) at n_perm=1000 are **impractically slow** on the contended
cluster (~40–60 min/fold; 0/63 completed in ~1 h each). **Significance is already established:** in-run n_perm=50
gives p=0.0196 (0/50 exceed) on **all** folds (BH-FDR significant); a 1-fold n_perm=200 spot-check gives p=0.005
(0/200). n_perm=1000 would only tighten p→~0.001 (report as `1/1001≈0.000999`, never 0). `r1_hardened_nperm1000.csv`
will be appended when a run completes (or downshifted to n_perm=200 on PM confirmation).

## Data access
See `data_access_note.md`: BNCI2015_001 datalake `.mat` are owner-only; loaded read-only via a world-readable
copy + `MNE_DATASETS_BNCI_PATH` override. No trial/label/fold/split logic changed.
```
```
