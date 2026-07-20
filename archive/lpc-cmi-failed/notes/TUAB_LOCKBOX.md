> ⛔ **RETRACTED / SUPERSEDED (2026-06-21).** TUAB is **NOT** a clean pre-registered lockbox: 13 TUAB result files exist in the root commit `fb2a878` — a full LOSO method comparison (`erm/lpc_prior/cdann/dann`) — predating this file (`1de7a12`), and the numbers fed calibration + the SCPS scorecard. This frozen spec also pins the now-DROPPED LPC selector + residual-CMI gate. **Do NOT open TUAB on this spec.** See `notes/TUAB_EXPOSURE_AUDIT.md` and `notes/EVIDENCE_LEDGER.md`.

# TUAB LOCKBOX — pre-registered held-out evaluation (DO NOT OPEN until method + baselines are frozen)

**Purpose.** TUAB is reserved as a *lockbox* test set to mitigate the benchmark-overfitting risk of a
multi-round, adaptively-developed method. Everything below is frozen BEFORE any TUAB result is computed. After
the method and baselines are final, TUAB is evaluated **exactly once** with this spec; the result is reported
as-is. **No hyperparameter, selector, gate, or threshold may be tuned on TUAB.**

**Freeze status:** FROZEN. Method development commit = `a7b8966` (R8/R9). This manifest is committed on top of it.
**Do not read TUAB numbers, do not `git log` TUAB result files, do not glance at logs until the unfreeze
condition is met.**

## Unfreeze condition (ALL must hold before running TUAB)
1. The method (CITA = CMI-screened nested-selected transductive matched-CORAL alignment) is final — no further
   changes to selector, aligner, gate, or shrinkage.
2. The baseline suite is implemented and validated on SCZ/PD (matched-CORAL, raw EA, SPDIM, T3A, TENT/SHOT, CAFA).
3. The 5-seed confirmatory + cohort/subject-clustered CIs on SCZ/PD are complete.
4. The semi-synthetic concept-shift study (a) is complete.

## Frozen method spec (the ONLY thing evaluated on TUAB)
- **Backbone:** EEGNet (default), raw signal, same preprocessing as the disease caches.
- **Representation selection:** nested **leave-one-source-recording-group-out** CV over λ ∈ {0, 0.1, 0.3}
  (`erm:0`, `lpc_prior:0.1`, `lpc_prior:0.3`); rule `λ* = argmin Î_w(Z;D|Y) s.t. valBAcc ≥ max−ε`, **ε = 0.02**.
  Selector sees **no target labels** and **no target data**.
- **Predictor:** frozen source readout (the source-fit linear head on the alignment embedding); never updated.
- **Aligner:** **matched-CORAL** (`pmct_transport(ref='pooled', tmap='wc')`), shrink **ρ = 0.2**, **eps = 1e-3**,
  reliability gate `g = g_cov · exp(−κ·Var(π̂_T))`, **κ = 8.0**, α = 1.0, em_iters = 3.
  (PMCT `ref='prior_matched'` reported only as a prior-robustness ablation, not the headline.)
- **Deployment:** strict source-free via `cmi/eval/source_state.pmct_predict_serialized(state, z_tgt)` — state
  hash recorded; no source examples at test time (equivalence to online verified to 0.0).
- **Applicability gate (report, do not tune):** residual decoder CMI `R_dec` vs its source permutation null
  `q^null_0.95` (`--decoder_null_perms 200`, `--dec_domain` = the within-source granularity). Reported as a
  diagnostic; the headline does NOT abstain on TUAB unless `R_dec > q^null_0.95`.

## Frozen TARGET-BATCH structure (critical)
TUAB MUST be evaluated with **class-spanning target batches** — by **recording-group** or **pooled target
batch**, NOT single-class-per-subject LOSO. Single-class LOSO is the known non-identifiable boundary (R6) and is
out of scope for the transductive-alignment headline; if reported at all it goes to the appendix with raw EA as
the EEG-specific baseline. Frozen target unit: **leave-one-recording-group-out** with both classes present in
each held-out group (groups defined by the TUAB cache's recording/site grouping; if unavailable, pooled
balanced target batch of n_T ≥ 64).

## Frozen evaluation
- **Script:** `python -m cmi.run_scps_crossdataset --condition TUAB --domain <group> --dec_domain <subgroup>
  --select nested --transduct all --decoder_null_perms 200 --configs erm:0 lpc_prior:0.1 lpc_prior:0.3 --seed {0..4}`
- **Primary metric:** per-target **balanced accuracy** (macro recall on pooled confusion matrix), CITA_nested +
  matched-CORAL vs ERM. **Secondary:** Î_w(Z;D|Y) leakage; net-correction (w→c, c→w); calibration (ECE/NLL).
- **Reporting:** 5-seed mean ± std + **recording-group/subject-clustered paired bootstrap CI** and worst-group
  performance. Seeds are NOT independent clinical domains — cluster the bootstrap on target group/subject.
- **Success ≠ a gate to publish.** TUAB is confirmatory: report whatever it shows (positive, null, or negative)
  with the frozen spec. A null/negative TUAB result is reported honestly, not retconned.

## Manifest integrity
This file's frozen-spec content is the pre-registration. Compute its hash at freeze time and record it in the
commit message: `sha256(this file)`. Any later edit to the spec invalidates the lockbox.
