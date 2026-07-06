# CIGL_68 — Direct-reliance CMI confirmation (seeds 0/1/2): NEGATIVE (unstable seed0 signal)

```
STATUS: FROZEN — METHOD-LEVEL NEGATIVE. dcigl_consistency_beta0.1, full-LOSO × seeds 0/1/2, BNCI2014_001 +
BNCI2015_001. 63 dcigl fold rows (+CIGL/FCIGL comparators), replay_ok all, random control ~0, projector
source-train-only. The seed0 R3 reduction did NOT survive multi-seed. Direct optimization of the counterfactual
reliance also fails to robustly move R3. STOP CMI method-development line.
```

## Paired (dcigl β0.1 − old CIGL) hierarchical bootstrap 95% CIs (seeds 0/1/2; sig = excludes 0)
| quantity | pooled | 2a | 2015 |
|---|---|---|---|
| **R3_task_drop_k2** | **−0.007 [−0.024, +0.006] ns** | −0.004 ns | **−0.009 [−0.035, +0.015] ns** |
| target_bacc | +0.004 [−0.004, +0.010] ns | −0.001 ns | +0.007 ns |
| **mean_margin** | **+0.252 [+0.088, +0.416] sig↑** | +0.102 sig | +0.364 sig |
| **prediction_entropy** | **−0.044 [−0.051, −0.037] sig↓** | −0.044 sig | −0.044 sig |
| task_head_alignment_k2 | +0.059 [+0.003, +0.126] sig↑ | +0.014 ns | +0.092 sig |

**dcigl β0.1 − FCIGL-align η0.01, R3_task_drop_k2:** pooled −0.008 [−0.023, +0.005] ns; 2015 −0.014 [−0.031, +0.003] ns.

## Read (honest)
1. **The seed0 R3 signal did NOT replicate.** dcigl β0.1's R3 reduction vs CIGL is **not significant** at seeds
   0/1/2 (pooled −0.007, 2015 −0.009; CIs include 0). The seed0 delta (−0.032 on 2015) was inflated because CIGL's
   *seed0* R3 (0.082) sat far above its multi-seed mean (0.055); dcigl's multi-seed mean R3 (0.046) is only ~0.009
   below CIGL's (0.055). dcigl is also **not** significantly better than FCIGL (−0.008/−0.014 ns).
2. **Directionally favorable but not significant.** Every point estimate points the right way (R3 ≤ CIGL and ≤
   FCIGL; target retained/slightly up; removed_target_bacc 0.545 vs CIGL 0.529 on 2015) — but none reach
   significance across seeds. This is the same fate as FCIGL: a real seed0 screen that does not confirm.
3. **Anti-triviality — the negative is CLEAN.** The PM's concern was that R3 might fall because logits flatten.
   The opposite is true: dcigl is **significantly more confident** (margin +0.25, entropy −0.044) — it did **not**
   reduce R3 by flattening. So there is no trivial confound; the direct objective genuinely made predictions more
   peaked + more removal-consistent in training, but this did **not** translate into a robust reduction of the
   held-out counterfactual reliance (target R3 task_drop).
4. **R3 and alignment remain dissociable** (dcigl align +0.059 sig, i.e. HIGHER than CIGL) — consistent with
   CIGL_66/67: alignment is a diagnostic correlate, not the causal lever, and even directly targeting R3 doesn't
   move the held-out reliance.

## Verdict (method-level)
**CIGL_68 is a method-level NEGATIVE.** Per the PM failure rule (R3 delta collapses to ≈0 / seed0-only), the
direct-reliance objective is **not** promoted. **Stop the CMI method-development line.** dcigl_consistency joins
fcigl_align (alignment proxy) and fcigl_removal_aug (killed) as controllable-but-not-reliance-reducing.

## Final scientific synthesis (the three-level measurement→reliance gap)
```
1. CIGL audit is stable and significant (label-conditional graph/node leakage; n_perm=1000 FDR).
2. Global measured leakage control (CIGL) is stable but does NOT reduce classifier reliance.
3. Task-head alignment is controllable (FCIGL-align, significant) but does NOT reduce reliance.
4. A direct counterfactual-reliance objective (dcigl_consistency) has only an unstable seed0 signal;
   it does NOT robustly reduce reliance across seeds 0/1/2, and does so without a flattening confound.
=> Under this static-adjacency DGCNN EEG setting, CMI proxy/objective control and functional reliance control
   are separated by a deep, three-level measurement→reliance gap.
```
This is the durable contribution: not a winning method, but a systematic demonstration — across a measured-leakage
proxy, a diagnostic-alignment proxy, and a direct reliance objective — that controlling any of them does not yield
functional reliance control on real EEG (method-level, seeds 0/1/2).

## Wording guard
Allowed: "the direct reliance objective produced only an unstable seed0 screen; it does not reduce reliance at
method level; predictions became more confident, not flatter." Do NOT say: dcigl reduces reliance / is a better
decoder / fixes CIGL.

## Next (PM decides)
Method development on this line is exhausted. Options: (a) freeze the three-level gap as the CMI scientific
synthesis and move to write-up; (b) a fundamentally different setting (dynamic adjacency, or a different
reliance estimand) — but that is a new research question, not another β/proxy. My lean: (a).

## Artifacts (`results/cigl_direct_reliance/final/`)
`direct_reliance_metrics.csv` (189), `direct_reliance_r3.csv`, `direct_reliance_logit_diagnostics.csv`,
`direct_reliance_vs_frozen.csv`, `direct_reliance_bootstrap_ci.csv` (42), `MANIFEST.yaml`.
Analysis: `scripts/analyze_dcigl_confirmation.py`.
```
```
