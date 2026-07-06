# FSR_03 — CPU-only Analysis Pre-Registration

**Project FSR — Step 2A.** Pre-registration of the frozen, CPU-only Phase-2 analysis. This is a plan, not a results document: it fixes the inputs, tests, robustness battery, decision language, and stopping rules **before** the full analysis (Step 2B) is run. No GPU, no new training, no CMI sweep, no `fbdualpc` rescue.

The single confirmatory number permitted in Step 2A — the CIGL headline reproduction (`analyze_cigl_gap.py`) — has been produced and is reported here as the RQ1 **entry gate** (below); it is a reproduction check, not an RQ conclusion. RQ conclusions are Step 2B, gated on the PM's go.

Companion: `docs/FSR_02_METRIC_SCHEMA.md`; results under `results/fsr_phase2/`.

Unit definition (all CIGL RQs): `(dataset, method ∈ {erm, cigl_graph_node}, seed ∈ {0,1,2}, fold)`; `align_k2` = `gap_alignment.csv` (graph_z, k=2); `R3_task_drop_k2` = `r3_reliance.csv` (label_conditional, k=2). Bootstrap: `np.random.default_rng(0)`, `n_boot=2000`, percentile[2.5,97.5] — matching the original CIGL_66 driver.

---

## RQ1 — Does measured leakage predict functional reliance?

**Inputs (frozen):**
```text
results/cigl_r123/final/gap_correlations.csv        (pooled outputs, for comparison only)
results/cigl_r123/final/gap_alignment.csv           (align_k2 per unit)
results/cigl_r123/final/r3_reliance.csv             (R3_task_drop per unit)
results/cigl_r123/final/r1_hardened_nperm1000.csv   (graph_kl per fold — SEED0 only)
results/cigl_r123/final/gap_spectrum.csv, gap_diagnostic_summary.yaml  (support scalars)
```

**Primary test:**
```text
Spearman(graph_kl, R3_task_drop_k2)   + bootstrap CI
Spearman(align_k2, R3_task_drop_k2)   + bootstrap CI
paired bootstrap CI for the difference: rho(align_k2) - rho(graph_kl)
```

**Known headline (to reproduce, not copy):** `graph_kl → R3` ρ = −0.34 [−0.507,−0.166]; `align_k2 → R3` ρ = +0.34 [+0.168,+0.504]; pooled n = 126.

**Entry-gate reproduction status (DONE, `results/fsr_phase2/cigl_gap_reproduction.json`):**
```text
align_k2 -> R3 (pooled, n=126)  : recomputed +0.3382 [+0.167, +0.506]  == frozen +0.3382   REPRODUCED (within tol)
graph_kl -> R3 (seed0,  n=42 )  : recomputed -0.4191 [-0.685, -0.108]  (negative sign confirmed)
graph_kl -> R3 (pooled, n=126)  : NOT recomputable — per-fold graph_kl for seeds 1/2 was pruned
                                   (raw audit .npz + r2-gate JSONs are not committed on any branch);
                                   frozen -0.342 carried, flagged verified-not-recomputable (documented reason,
                                   NOT a mismatch -> no STOP)
difference (align - graph_kl, seed0, n=42): +0.816 [+0.219, +1.333], excludes 0
```
The right-sign result (alignment) reproduces exactly at full n; the wrong-sign result (leakage) reproduces in sign at seed0. The n=126 leakage recomputation is blocked by a raw-artifact prune, recorded in `missing_metric_decisions.csv` as `artifact_missing` (the seed0 recomputation partially covers it).

**Required robustness (Step 2B):**
```text
dataset-stratified Spearman (BNCI2014_001, BNCI2015_001)   [already emitted in cigl_gap_reproduction.csv]
per-seed sensitivity (seed 0/1/2)                          [already emitted]
within-dataset rank residualization (partial out dataset before pooling)
leave-one-dataset-out
cluster bootstrap by target subject/fold (unit id present)
per-method sensitivity (erm vs cigl within the 126 units)
```

**Decision language (fixed):**
```text
If pooled AND >=1 major dataset support the sign:
  "task-head alignment is a better reliance indicator than raw leakage in the available CIGL R3 artifacts."
If pooled holds but one dataset is ns:
  "pooled mechanism-positive, dataset-heterogeneous; not universal."
If the headline does not reproduce (align pooled beyond tolerance):
  STOP and return to Step 1 provenance (write STOP_REPRODUCTION_MISMATCH.md).
```
FORBIDDEN phrasing: "align_k2 is a validated estimator of reliance." ONLY: "align_k2 is closer to functional reliance than graph_kl in this frozen diagnostic."

---

## RQ2 — Does erasure strength predict target benefit?

**Inputs (frozen, branch `tos`):**
```text
tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_{TSMNet,EEGNet}_LOSO/erasure_report.json
tos_cmi/results/tos_cmi_eeg_frozen/erasure_target_deploy/erasure_target_deploy_summary.json
tos_cmi/results/tos_cmi_eeg_frozen/erasure_target_deploy/{Lee2019_MI,Cho2017,Schirrmeister2017}/erasure_target_deploy_summary.json
tos_cmi/results/tos_cmi_eeg_frozen/factorial/factorial_multiseed.json   (capacity, exploratory)
```
(Exact paths authoritative per `FSR_01`. Read via `git show tos:<path>`.)

**Primary derived variables:**
```text
E_subject_removed = ERM_subject_acc - post_eraser_subject_acc   (L3, from erasure_report)
T_target_bacc     = target_bAcc_eraser - target_bAcc_ERM        (L6, dtgt_bacc)
T_target_nll      = target_NLL_eraser  - target_NLL_ERM         (L6, dtgt_nll)
task_safety       = task_bAcc_eraser   - task_bAcc_ERM
```

**Primary tests:**
```text
corr(E_subject_removed, T_target_bacc)      across erasers x (dataset,backbone)
corr(E_subject_removed, T_target_NLL)
paired comparison: LEACE / RLACE / INLP  vs  random_k   (NLL + subject-removal specificity)
```

**Decision rule (fixed):**
```text
If erasers and random_k produce similar NLL movement:
  label NLL as non-specific unless target bAcc AND subject-removal specificity agree.
If erasure improves NLL but not bAcc:  do NOT call it a DG benefit.
If erasure harms binary EEGNet:  record as a task-safety FAILURE, not just a target failure.
```

**Descriptive status (DONE, `results/fsr_phase2/tos_*.csv`):** subject signal is erasable (LEACE linear→chance); no eraser improves target bAcc; the 2a-TSMNet NLL drop is `non_specific=YES` (random-k matches it while leaving subject at 0.998); INLP task-collapse flagged on 6 cells; LEACE/RLACE binary-EEGNet harm flagged on Lee/Cho. These are descriptive tables — the RQ2 correlation and its decision are Step 2B. TOS's job here is to support RQ2; not every TOS subproject becomes a unified regression.

---

## RQ3 — Does task-head alignment beat leakage magnitude?

The modeling version of RQ1, at the finest safe unit.

**Table (finest unit; `unit_id = (dataset, method, seed, fold)`):**
```text
unit_id | dataset | target_subject_or_fold | seed | method | graph_kl | node_kl_if_available |
align_k1 | align_k2 | R3_task_drop | target_bAcc_delta | collapse_guard
```
Coverage caveat (pre-declared): `graph_kl`/`node_kl` at unit level exist only for **seed0** (seeds 1/2 pruned); `align_k2` and `R3_task_drop` exist for all 126 units. The alignment-vs-leakage *contrast* model is therefore fit at seed0 (n=42, both predictors present); the alignment-only relationship uses all 126.

**Primary model:**
```text
R3_task_drop ~ z(graph_kl) + z(align_k2) + dataset + seed          (seed0 units where graph_kl present)
# if route labels available:
R3_task_drop ~ z(graph_kl) + z(align_k2) + dataset + seed + method
```
No overfitting: no random forest, no lasso zoo, no nonlinear search. This is a mechanism test, not a leaderboard. Report standardized coefficients + bootstrap CIs; the pre-registered comparison is `|coef(align_k2)|` vs `|coef(graph_kl)|` and their signs.

**Decision language:** same discipline as RQ1 — "closer to reliance than leakage," never "validated estimator."

---

## RQ4 — Does branch-locality change the meaning of leakage?

**Current status (declared):**
```text
FBCSP:  L4 branch-load STRONG (zero_spatial 2a -7.4pp / 2015 -8.8pp; gate_spatial 0.489/0.572)
        L1 per-branch leakage MISSING  (no frozen spatial_z/graph_z/node_z, no per-branch probe)
        L5 branch-specific reliance MISSING
```
Under the revised Phase-1 gate, no route has the predictor(L1 per-branch)+endpoint(L5 per-branch) pairing RQ4 needs, so **RQ4 has zero quantitative rows**. Step 2 produces only:
```text
results/fsr_phase2/branch_load_table.csv          (descriptive, from FBCSP_F0_AGGREGATE)
results/fsr_phase2/branch_missing_metric_report.md (what is absent and why)
```
Do **not** run frozen probes yet. FBCSP can support: "spatial branch is load-bearing", "graph branch is neutral/slightly harmful", "P6 spatial-CMI is not promoted". It **cannot** support: "spatial leakage is harmful", "graph leakage is benign", "per-branch CMI predicts reliance". The prior-decoupled TTA line (background, §5F of FSR_01) is the conceptual precedent: a single joint number must be decomposed into distinct intervention pathways; FSR applies the same discipline to leakage × erasure × branch-load × target-consequence rather than collapsing them.

*(The `branch_load_table.csv` / `branch_missing_metric_report.md` are the Step-2B RQ4 deliverables; Step 2A registers them here.)*

---

## Stopping rules (return to the PM if any occur)

```text
1. artifact_index cannot be parsed as 37 rows x 18 columns.
2. Any CMI-control route is accidentally labelled a positive method.
3. Any YES_FORBIDDEN row enters an RQ quantitative analysis.
4. ACAR DEV_STOP is overwritten back to pending.
5. CSC B7.1 is treated as run, not protocol-only.
6. FBCSP P6 scaffold is treated as a confirmed spatial-CMI result.
7. CIGL headline correlations fail reproduction beyond tolerance.
8. Any analysis requires GPU.
```
Status at end of Step 2A: (1) validator PASS 15/15; (2) validator check `no_cmi_control_failure_as_positive` PASS (only `TTA_Control_non_CMI` is positive, explicitly non-CMI); (3) `YES_FORBIDDEN` appears once (LPC legacy) and is excluded from all RQ tables (`include_rq*`=NO); (4) ACAR recorded as `ACAR_stage2b_dev_stop`; (5) CSC B7.1 tagged `PROTOCOL` (UNRUN); (6) P6 scaffold `BACKGROUND_ONLY` (DESIGN_ONLY); (7) align reproduced, graph_kl-sign confirmed, no STOP; (8) all four scripts are pure-Python CPU, no GPU.

---

## What is Step 2A (done) vs Step 2B (awaits PM go)

```text
Step 2A (this deliverable):
  - metric schema (FSR_02) + this prereg (FSR_03)
  - fail-closed index validator                -> schema_validation.json  (PASS 15/15)
  - normalized tables                          -> route_metric_table.csv (37), analysis_inclusion_table.csv (37),
                                                  missing_metric_decisions.csv (22)
  - CIGL headline reproduction (entry gate)    -> cigl_gap_reproduction.{json,csv}, cigl_gap_bootstrap.csv (REPRODUCED)
  - TOS erasure descriptive tables             -> tos_erasure_summary.csv (51), tos_randomk_specificity.csv (8),
                                                  tos_task_safety_flags.csv (40)

Step 2B (NOT started — needs PM approval):
  - RQ1 full robustness battery + decision
  - RQ2 corr(E,T) + paired eraser-vs-random_k test + decision
  - RQ3 mechanism regression (seed0 contrast + n=126 alignment)
  - RQ4 branch_load_table.csv + branch_missing_metric_report.md (descriptive only; no probe)
```
