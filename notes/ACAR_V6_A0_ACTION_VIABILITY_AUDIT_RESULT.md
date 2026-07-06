# ACAR V6-A0 — Action-Viability Audit (RESULT)

```
status:                    V6_STOP
implementation_base_sha:   ea37fc2c25238c576500dc3fb13755ac4f0890ec
stage1b_run_id:            acar-v5-stage1b-c4412b4-r1
stage1b_registry_sha256:   2bbe55f4cdb4f1a18cee3b2c9e7583dba9fe9e84b9c563fb37781e98ebcbb76d
protocol_tag_target_sha:   4278435975a72b1127803dd2cffab420c083e430
run:                       SLURM 887279, nodecpu04, ~12 min, rc=0
scope:                     exploratory diagnostic only; real DEV labels used ONLY for ΔR_a + f_0 calibration;
                           no policy fitting / no candidate selection / no external / no lockbox
```

**Decision = `V6_STOP`.** `V6_CONTINUE` requires BOTH diseases to pass all four EVAL-primary sub-gates; **PD passes all four, SCZ
fails only `perm_p_subject_block`** (0.208 > 0.05). This is NOT an underpowered-null artifact — the subject-block permutation null
was fully evaluable on both diseases (PD 229/230 permutable, SCZ 221/226; 1000 valid perms each). Per the authorization, `V6_STOP`
means: **at the pre-registered continuation gate there is no dual-disease reliable subject-clustered label-free sign signal** — so
the ACAR router line stops at the diagnostic level. No policy fitting, no alternate metric, no relaxed gate, no rerun.

## Per-disease primary EVAL table (the gate)

```
metric                                PD                 SCZ                gate threshold
oracle_red_upper                      0.4131  PASS       0.5853  PASS       > 0.02
beneficial_coverage_subject_macro     0.7964  PASS       0.7292  PASS       >= 0.15
sign_auroc_subject_balanced           0.7050  PASS       0.6319  PASS       >= 0.60
perm_p_subject_block                  0.0010  PASS       0.2078  FAIL       <= 0.05
--------------------------------------------------------------------------------------
disease gate                          PASS               FAIL
decision                              V6_STOP (both diseases must pass all four)

permutation_null_status               evaluable          evaluable
n_permutable_subjects / n_subjects    229 / 230          221 / 226
n_perm_valid                          1000               1000

descriptive (NOT the gate):
sign_auroc_record_weighted            0.7204             0.6334      (≈ subject-balanced -> not batch-dominated)
beneficial_coverage_batch_weighted    0.7924             0.7231
no_safe_action_rate_subject_macro     0.2036             0.2708
per-action subject-balanced AUROC     mc 0.643           mc 0.554
                                      spdim 0.676        spdim 0.488
                                      t3a 0.902          t3a 0.619
eligible EVAL batches / subjects      655 / 230          1499 / 226
forced tails (excluded)               64                 17
```

## Q1 — oracle envelope + beneficial coverage (subject-macro)

A substantial oracle-selectable benefit EXISTS on both diseases (confirming the V5 report's `red_upper` reading): an oracle that
per-batch picks the best action-or-noop reduces NLL by **0.41 (PD) / 0.59 (SCZ)** nats, and ~**80% (PD) / 73% (SCZ)** of subjects
have, on average, a beneficial action available per batch (`beneficial_coverage_subject_macro`). `oracle_conditional_harm = 0` on
both (sanity: the oracle only adapts when it helps). So the V5 DEV_STOP was NOT because the actions are always harmful — the
benefit exists; the question is whether a label-free rule can find it.

## Q2 — action × provenance strata (descriptive, batch-weighted mean ΔR_a)

```
                    mean ΔR_a  (ΔR>0 = HARMFUL)
stratum (PD)         matched_coral    spdim     t3a       benef_cov
all_eligible          -0.003         +10.94    +1.47       0.792
montage::Pz           -0.002         + 8.26    +1.84       0.798
montage::native       -0.003         +12.14    +1.31       0.790

stratum (SCZ)        matched_coral    spdim     t3a       benef_cov
all_eligible          -0.024         +12.35    +2.43       0.723
fully_native          -0.044         + 9.31    +4.76       0.777
brainvision_repaired  -0.006         +13.86    +1.25       0.739
channel_renamed       -0.007         +13.61    +1.27       0.734
montage::F3F4P3P4     -0.012         +12.92    +1.21       0.708
montage::F7           -0.035         +14.37    +2.19       0.642
```
Mechanism: **`matched_coral` is ~neutral (mean ΔR ≈ 0, slightly beneficial); `spdim` and `t3a` are on-average catastrophic**
(+9…+14 and +1…+5 nats). The oracle benefit comes from *selecting* the beneficial batches (mostly matched_coral), not from any
action being uniformly good. Harm is broadly similar across repair/completion strata (no stratum rescues spdim/t3a).

## Q3 — subject-balanced sign predictability (the binding test)

- **PD: real and significant.** Pooled subject-balanced AUROC **0.705**, permutation **p = 0.001** (well-powered). Driven mostly by
  **t3a (per-action AUROC 0.902)** — label-free features predict *when* t3a helps vs harms on PD strikingly well, even though t3a is
  on-average harmful. The record-weighted AUROC (0.720) ≈ subject-balanced (0.705), so this is not a batch-rich-subject artifact.
- **SCZ: not confirmed.** Pooled AUROC **0.632** clears the ≥0.60 bar but permutation **p = 0.208** — indistinguishable from the
  subject-block-permuted null. Per-action AUROC is near chance (mc 0.554, spdim 0.488, t3a 0.619). The subject-balanced +
  permutation design is what caught this: a naive record-weighted / no-permutation gate might have wrongly passed SCZ's 0.63.

## Q4 — f_0 (source-state LDA) calibration on EVAL

```
                 PD                 SCZ
n_windows        19348              44296
accuracy         0.631              0.575
actual_case_rate 0.664              0.463
mean_confidence  0.721              0.766
ece_10bin        0.090              0.190
mean_nll         0.680              1.025
prior_gap        -0.004             +0.034   (mean predicted case-prob − actual case-rate)
```
`f_0` is moderately overconfident on both (conf > acc). **SCZ `f_0` is markedly worse-calibrated** (ECE 0.19 vs 0.09, NLL 1.02 vs
0.68) — consistent with SCZ having no findable label-free sign signal, while the better-calibrated PD `f_0` does.

## Interpretation (what V6-A0b establishes)

The V5 DEV_STOP is refined, not overturned: an adaptation **benefit envelope exists** on both diseases, and on **PD** there IS a
significant subject-clustered label-free signal for the beneficial sign (especially for t3a). But the pre-registered continuation
gate requires it on **both** diseases, and **SCZ's signal is not significant** against a well-powered subject-block null. So a
single dual-disease label-free router in this action/feature class is not justified: `V6_STOP`. The result also localizes where a
future (separately-authorized) line might look — PD-specific / t3a-focused sign prediction, and SCZ f_0 recalibration — but that is
NOT pursued here.

## Forbidden-stage confirmation

```
policy fitting:            NONE            candidate selection:  NONE
threshold tuning:          NONE            routing:              NONE
Stage-4 / S1 / S2 / S3:    NOT RUN         external / held-out:  NOT READ
ASZED:                     NOT READ        lockbox:              SEALED
substrate rebuild:         NONE            repair/montage/label policy change: NONE
labels used:               ONLY for ΔR_a (batch_action_delta_r) + f_0 calibration diagnostics; EVAL split only
report schema:             diagnostic-only; validated fail-closed on any candidate/route/G1-G6/Stage-4/S1-S3/external/held-out/ASZED/
                           lockbox key (exact-key match). The only threshold-named field is the static pre-registered
                           `gate_thresholds` cut-point block (a gate DEFINITION: coverage_min 0.15 / red_upper_min 0.02 /
                           auroc_min 0.60 / perm_p_max 0.05) — not threshold tuning. No proposed candidate/route/policy/threshold.
decision emitted:          exactly one — V6_STOP
```

## Status

`V6_STOP` is terminal at the diagnostic level under the pre-registered gate. No V6 protocol draft, no policy fitting, no candidate
selection, no relaxed gate, no rerun without a NEW dated authorization. Any continuation is a new hypothesis with a new dated
protocol and a clean confirmation route on a still-sealed substrate.
