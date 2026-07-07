# A0 — Minimal gate-falsification slice (PRE-REGISTERED, frozen 2026-06-21)

Purpose: **falsify-first**. Decide whether ONE source-free scalar can serve as a single deployable gate, or whether
the evidence forces a TWO-LEVEL controller — *before* unifying sample-level abstention and cohort/batch-level
adaptation-eligibility. Output is an **architecture candidate**, NOT a deployable verdict (Freeze B still needs a
closed-loop pilot). Runs on the hash-bound **`erm:0` = CITA-no-LPC** dumps (the deployment encoder, P1.5-closed),
GPU-free. This file is frozen BEFORE running; thresholds/directions/aggregations are not changed post-hoc.

## 0. Substrate & embedding stage (confirmed)
- 7 held-out cohorts: PD {ds002778, ds003490, ds004584}, SCZ {ds003944, ds003947, ds004000, ds004367}.
- `z_te` = `embed(bb, X_te)` = raw encoder, **PRE-alignment, PRE-gate, PRE-abstention, full unfiltered**. CORAL is
  applied in z-space downstream → no double alignment. `z_se`/`y_se` = source split (for source statistics only).
- `prob_te` = `predict(bb, X_te)` = the backbone head's UN-adapted prediction (a reference, NOT used as base/adapted).

## 1. Frozen readout & adaptation (NO refit-proxy) — VERIFY GATE PASSED BIT-EXACT
The deployed adaptation is a z-space LR probe + matched-CORAL transport. The deployed transduct source is the **`ev`
split** (`z_ev`/`y_ev` in the dump = `embed(bb, Xtr[ei])`), NOT `z_se` (the `pi` split used for the P1.5 leakage audit):
- `probe` = `LogisticRegression(max_iter=2000, C=1.0)` fit on (`z_ev`,`y_ev`) — deterministic, identical to deployment.
- **base** = `probe(z')` (no transport).  **adapted** = `probe(matched_coral(z_ev → z'))` with the ACTUALLY DEPLOYED
  config: `mode=matched_coral`, `shrink=transduct_shrink=0.1` (the run used CLI default 0.1, NOT the manifest
  `aligner.rho=0.2`), `pi_S` = full-train prior (`bincount(y_se ∪ y_ev)`).
- **VERIFY GATE (pre-run, 2026-06-21):** offline `ts_matched_coral_balanced_acc` reproduces the runner-recorded value
  **BIT-EXACTLY (worst |d| = 0.0e+00 across all 7 folds)** → frozen readout/adaptation is the exact deployed
  computation, not a proxy. No serializing redump needed.
- Deployment-facing API: `score_target(state, z_target, batch_meta=None)` — reads ONLY the serialized state
  (frozen probe + source sufficient statistics: μ_pool, Σ_pool, {μ_y, Σ_y}, π_S from `z_ev`) and the unlabeled target.
  No raw source examples and NO target `y` at scoring time. Density/discriminator estimates are **cross-fit**
  (source-only moments; target never used to estimate its own density; no self-inclusion).

## 2. Candidate scores — FROZEN formula, direction, LEVEL, aggregation
Convention: **higher score ⇒ more reason to abstain/withhold-adaptation.**

### Sample-capable (compete for the SINGLE gate). Batch eligibility = `S_B = mean_{i∈B} s_i` (ONLY aggregation).
| key | formula (source-free) | direction |
|---|---|---|
| `g_unc` | binary entropy H(base prob p_i) | high = uncertain |
| `s_support` | squared Mahalanobis to source pool: (z_i−μ_pool)ᵀ Σ_pool⁻¹ (z_i−μ_pool) | high = source-atypical / low support |
| `s_sep` | −|m_0(z_i) − m_1(z_i)|, m_c = Mahalanobis to source class-cond mean μ_y^S | high = near source class boundary |
| `cmi` | per-sample decoder-CMI residual (`concept_shift_study._decoder_cmi_residual`) | high = residual conditional structure |

### Batch-only comparators (may support a TWO-LEVEL controller; NEVER called the same sample→batch scalar)
| key | formula | direction |
|---|---|---|
| `bures_shift` | ‖μ_T−μ_S‖² + Bures²(Σ_S, Σ_T) over the batch | high = more covariate shift |
| `global_sep` | Fisher trace-ratio tr(S_b)/tr(S_w) of the batch under BASE pseudo-labels | low = more class overlap |
| `shift_x_sep` | `bures_shift` × (1/`global_sep`) | high = shifted AND overlapping |

### Oracle-reference-only (NOT a deployment candidate; reported for context)
- `domdisc` = logit P(D=T|z) from a discriminator trained on raw `z_se` ∪ `z_te` (needs source data → not source-free).

## 3. Harm targets (NOT shift detection)
- ℓ_i^base = −log p_i^base[y_i],  ℓ_i^adapted = −log p_i^adapted[y_i]  (same probe; differ only by transport).
- **Sample primary:** continuous `Δℓ_i = ℓ_i^adapted − ℓ_i^base`. Metric: Spearman(score, Δℓ) + C-index.
- **Sample auxiliary:** harmful-flip AUROC of `1[base correct → adapted wrong]`, evaluated **only on the base-correct
  subset**. Record the full 2×2 transition table (w→c, c→w, unchanged-c, unchanged-w).
- **Batch primary:** continuous `mean Δℓ_B`. Metrics: Spearman + C-index + binary harmful-batch AUROC `1[mean Δℓ_B>0]`.

## 4. Generators (each its OWN severity grid; NO Cartesian θ×ρ across generators)
| generator | shift | severity grid | role |
|---|---|---|---|
| `clean` | none | {0} | false-abstention control |
| `lowmargin_rot` | rotate boundary by θ (flips low-margin) | θ ∈ {10,20,30,45}° | MSP-catchable harm |
| `highmargin_cbw` | flip a high-confidence pocket + covariate signature | ρ ∈ {0.05,0.10,0.15} | dangerous (MSP reverses) |
| `covariate_shift_beneficial` | affine covariate shift CORAL fixes | s ∈ {0.5,1.0,2.0}·σ | adaptation HELPS → gate must NOT veto |
| `feature_invisible_conditional` | relabel only, z byte-identical | ρ ∈ {0.10,0.20} | **identifiability + leakage guard; EXCLUDED from gate contest** |
| `local_cond_rot` (optional) | class-cond rotation in a local region | θ ∈ {30,45}° | density-vs-conditional separation |

`feature_invisible_conditional` requirement: every source-free score MUST read numerically identical to `clean`
(z unchanged). If any (incl. `cmi`) separates harm there, HALT and audit label/generator leakage — it is NOT a win.

## 5. Natural-batch regime (one fixed, label-blind condition)
- Batches of **B=32** in natural recording order (no label balancing). Score AND CORAL use the SAME batch.
- Batches with <2 effective classes or B<8 → forced identity fallback (no adaptation, score still recorded).
- Whole-cohort adaptation = sensitivity only, not the main result.
- Sample-level errors are **clustered by (cohort × generator × batch)** — cohort-shared CORAL must not inflate N.

## 6. Held-out protocol & normalization
- **Primary:** within-disease LOCO (leave-one-cohort-out, within PD / within SCZ) × **leave-one-generator-family-out**.
  Disease-stratified summary; report per-cohort spread (7 cohorts is small).
- **Secondary (stronger):** cross-disease PD↔SCZ transfer.
- Any score normalization is fit on NON-held-out cohorts/generators or uses serialized source-state stats ONLY —
  never the pooled target score mean/var.

## 7. Decision rule (THREE outcomes; equivalence margin FROZEN = AUROC/|metric| within 0.03)
A score is **admissible at a level** if its harm-metric is within **0.03** of the best score at that level (absolute
band), with a consistent prespecified direction and NO systematic direction reversal across PD vs SCZ or held-out
generator families, and no significant false veto on `clean` / `covariate_shift_beneficial`.
- **`SINGLE_GATE_CANDIDATE`** — one SAMPLE-CAPABLE score is admissible at BOTH the sample level (Δℓ/flip) and, via the
  frozen mean aggregation, the batch level (mean Δℓ_B), generalizing across held-out cohorts AND generator families.
- **`TWO_LEVEL_CANDIDATE`** — a batch-level score is admissible for adaptation-eligibility AND a (different)
  sample-level score is admissible for abstention, each generalizing as above.
- **`DIAGNOSTIC_ONLY`** — no stable admissible predictor at some level, or direction unstable across cohort/generator.

## 8. Output (immutable; pre-registered schema)
```
results/a0_falsification/<freeze_hash16>/
  a0_results.parquet   # cohort, disease, generator, severity, batch, level, score, target_metric, value
  a0_summary.json      # per (level, score): held-out metric ± spread; admissibility; the 3-way candidate
  run_manifest.json    # deployment-encoder dump hashes, frozen-readout verify residuals, seeds, this file's hash
```
Guards enforced in code: source-free API (no `z_se`/target-`y` at scoring), cross-fit density, generator-leakage
check on `feature_invisible_conditional`, clustered sample-level inference. Synthetic unit tests precede the real run.
