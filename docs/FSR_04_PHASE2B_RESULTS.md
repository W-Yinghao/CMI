# FSR_04 — Phase-2B Results

**Project FSR — Step 2B.** Results of the frozen, CPU-only RQ1–RQ4 analysis pre-registered in `FSR_03`. No GPU, no retraining, no spatial probe, no CMI rescue. All numbers are produced by `scripts/fsr/run_phase2b_*.py` / `build_rq4_branch_report.py` from frozen artifacts; re-run to reproduce. Companion: `FSR_05_CLAIM_LEDGER.md`, `results/fsr_phase2b/`.

Unit (CIGL RQs): `(dataset, method ∈ {erm, cigl_graph_node}, seed, fold)`. Bootstrap: `rng(0)`, `n_boot=2000`, percentile[2.5,97.5]. Claim-strength tiers: **RECOMPUTED** (full recomputation from committed per-unit data), **RECOMPUTED_SIGN_ONLY** (recomputable slice confirms sign), **FROZEN_NOT_RECOMPUTABLE** (value carried from a frozen output whose per-unit inputs were pruned — support only).

---

## RQ1 — Does measured leakage predict functional reliance?

Reported in three claim-strength tiers (never conflated).

### RQ1A — alignment → reliance (full n=126, **RECOMPUTED**)
```text
pooled            align_k2 -> R3_task_drop_k2 : rho = +0.3382 [+0.168, +0.504]  (excludes 0)
BNCI2014_001 (2a)                              : rho = +0.467  (sig)
BNCI2015_001 (2015)                            : rho = +0.103  (ns)
within-dataset rank-residualized               : (dataset-controlled; see rq1 csv)
per-seed / per-method                          : (see rq1_leakage_vs_reliance.csv)
```
**Decision:** *pooled mechanism-positive, dataset-heterogeneous; not universal.* Task-head alignment is positively associated with functional reliance pooled and on 2a, but not on 2015.

### RQ1B — leakage → reliance (seed0 n=42, **RECOMPUTED_SIGN_ONLY**)
```text
seed0 pooled      graph_kl -> R3_task_drop_k2 : rho = -0.4191 [-0.680, -0.100]  (excludes 0, NEGATIVE)
```
**Decision:** *raw graph leakage does not show the expected positive relationship with reliance in the recomputable seed0 slice; its sign is negative.*

### RQ1C — leakage → reliance (pooled n=126, **FROZEN_NOT_RECOMPUTABLE**)
```text
frozen pooled     graph_kl -> R3_task_drop_k2 : rho = -0.342 [-0.507, -0.166]   (from gap_correlations.csv)
status = FROZEN_NOT_RECOMPUTABLE (seeds 1/2 per-fold graph_kl pruned) — support only, NOT a reproduction
```

**RQ1 net:** measured leakage magnitude does not certify reliance — its sign is *negative* where recomputable (seed0) and in the frozen pooled summary, while task-head alignment is the correctly-signed, positively-associated quantity. **Forbidden phrasing:** "we fully reproduced both pooled correlations."

---

## RQ2 — Does erasure strength predict target benefit?

40 cells: {TOS_VD, LEACE, INLP, RLACE, random_k} × {2a, Lee2019_MI, Cho2017, Schirrmeister2017} × {TSMNet, EEGNet}. `E_subject_removed = subj_dec_after(full) − subj_dec_after(eraser)` (L3); `T = dtgt_bAcc / dtgt_NLL` (L6).

```text
corr(E_subject_removed, target_bAcc)  all cells (n=40) : rho = -0.420 [-0.682, -0.113]  (excludes 0, NEGATIVE)
corr(E_subject_removed, target_bAcc)  clean cells (n=30): rho = -0.435 [-0.731, -0.057]  (excludes 0, NEGATIVE)
corr(E_subject_removed, target_NLL)   all cells         : rho = -0.334 [-0.639, +0.039]  (ns)
benefit_claimable (proven bAcc gain)                    : 0 / 40 cells
task_collapse                                           : 10 / 40 cells
binary_harm (Lee/Cho EEGNet LEACE/RLACE, HGD)           : 8 / 40 cells
NLL non-specific (random-k matches LEACE)               : 1 / 8 LEACE-vs-random_k cells (canonical 2a-TSMNet)
```
**Decision:** *erasure strength is NEGATIVELY associated with target benefit (more subject removal → worse target bAcc); the old positive erasure hypothesis is refuted.* **No eraser certifies a proven target benefit** (`benefit_claimable=0`); the one recomputable non-specific NLL cell (2a-TSMNet, LEACE dNLL −0.031 matched by random-k −0.034 at unremoved subject 0.998) confirms that NLL movement there is not a domain-removal benefit. **Forbidden phrasing:** "LEACE improves target NLL" as a DG claim; the raw `improves_target` flag is renamed `raw_improves_target_flag` and never enters a paper claim.

---

## RQ3 — Is task-head alignment a better reliance predictor than leakage magnitude?

Primary test = the signed Spearman *difference* (not |β|); the OLS betas are standardized (z-scored outcome and predictors) and reported honestly.

```text
PRIMARY (seed0, n=42): spearman_diff(align_k2 − graph_kl) = +0.816 [+0.219, +1.333]  (excludes 0)
Model A (align, full n=126): std_beta(align_k2) = +0.157 [-0.144, +0.533]   (CI includes 0 after dataset control)
Model B (paired, seed0 n=42): std_beta(align_k2) = +0.263 (correct +sign)
                              std_beta(graph_kl) = -0.372 (WRONG −sign; larger |β| but wrong direction)
Model C (frozen summary): graph_kl pooled -0.342, FROZEN_NOT_RECOMPUTABLE, sign negative, support only
```
**Decision:** *in the fully recomputable data, task-head alignment is the more positively-associated / correctly-signed reliance predictor (the Spearman difference excludes 0); raw graph leakage carries the wrong (negative) sign in both the seed0 paired analysis and the frozen pooled summary.* Within-group partial betas are not individually significant at this n, reflecting the RQ1A dataset-heterogeneity; `graph_kl` is larger in |β| but negatively signed, so it is **not a valid positive reliance predictor**. **Forbidden phrasing:** "align_k2 is a validated reliance estimator."

---

## RQ4 — Does branch-locality change the meaning of leakage? (DESCRIPTIVE — blocked, not failed)

```text
BNCI2014_001  spatial  ablation_drop +0.0736  gate 0.489  -> load_bearing
BNCI2015_001  spatial  ablation_drop +0.0880  gate 0.572  -> load_bearing
graph / temporal branches: neutral_or_slightly_harmful / weak; starved after fusion
rq4_quantitative_status = BLOCKED_MISSING_METRIC (every branch)
```
**CAN say:** the spatial branch is load-bearing; graph/temporal are neutral/starved; P6 spatial-CMI is a scaffold, not a result. **CANNOT say:** "spatial leakage is harmful", "graph leakage is benign", "per-branch CMI predicts reliance." **Two HIGH missing metrics** (per-branch leakage probe, per-branch R3) block RQ4; producing them needs a PM-approved Phase-3/4 frozen-probe run (`rq4_branch_missing_metric_report.md`). No probe run in Step 2B.

---

## The FSR spine after Phase 2B

```text
L1 leakage is measurable, but raw leakage does NOT certify L5 reliance
   (graph_kl->R3 negative where recomputable; frozen pooled negative).
L4 task-head alignment is closer to L5 reliance (correctly signed; difference excludes 0),
   but it is NOT yet a validated estimator (dataset-heterogeneous; partial betas ns).
L3 erasure can remove subject signal, but erasure strength does NOT certify L6 target benefit
   (corr negative; 0/40 proven-benefit cells; NLL move non-specific where recomputable).
L4 branch load matters (spatial load-bearing), but RQ4 is BLOCKED until per-branch
   leakage + reliance probes exist.
```

## Step 2B stopping-rule status (none triggered)
```text
1 align_k2 full-n unchanged (+0.3382)          5 no CMI-control route as positive method
2 graph_kl seed0 still negative (-0.4191)      6 ACAR still DEV_STOP (BOUNDARY_ONLY)
3 RQ2 no raw-NLL-only benefit (benefit=0)      7 CSC B7.1 still protocol-only
4 no YES_FORBIDDEN/AUDIT_ONLY in RQ tables      8 FBCSP P6 still scaffold (RQ4 blocked)
                                                9 all CPU, no GPU
```
