# ACAR V6-A0 Closeout — V6_STOP / No Dual-Disease Label-Free Sign Signal

```
status:                    V6_STOP
implementation_base_sha:   ea37fc2c25238c576500dc3fb13755ac4f0890ec
stage1b_run_id:            acar-v5-stage1b-c4412b4-r1
stage1b_registry_sha256:   2bbe55f4cdb4f1a18cee3b2c9e7583dba9fe9e84b9c563fb37781e98ebcbb76d
protocol_tag_target_sha:   4278435975a72b1127803dd2cffab420c083e430
run:                       SLURM 887279, ~12 min, rc=0   (result: notes/ACAR_V6_A0_ACTION_VIABILITY_AUDIT_RESULT.md; commit 28588ac)
decision:                  no V6 protocol draft authorized
external / held-out / ASZED: not read      lockbox: sealed
```

The V6-A0 action-viability audit is **closed at `V6_STOP`**. This is a valid, pre-registered diagnostic stop — NOT an engineering
failure and NOT an underpowered diagnostic (the subject-block permutation null was fully evaluable on both diseases). Under the
pre-registered V6-A0 continuation gate, a **dual-disease label-free ACAR router is not justified**.

## Core result

```
metric (EVAL-primary)                 PD                 SCZ
oracle_red_upper                      0.4131  PASS       0.5853  PASS
beneficial_coverage_subject_macro     0.7964  PASS       0.7292  PASS
sign_auroc_subject_balanced           0.7050  PASS       0.6319  PASS  (>= 0.60, but see perm)
perm_p_subject_block                  0.0010  PASS       0.2078  FAIL  (> 0.05)
--------------------------------------------------------------------------------
disease gate                          PASS               FAIL
permutation_null_status               evaluable          evaluable
n_permutable_subjects / n_subjects    229 / 230          221 / 226   (1000 valid perms each)

Overall: V6_STOP — V6_CONTINUE requires BOTH diseases to pass all four sub-gates.
```

SCZ's `sign_auroc_subject_balanced = 0.632` clears the 0.60 bar but is **not significant** against a well-powered subject-block
permutation null (`p = 0.208`). PD passes all four (sign-AUROC 0.705, `p = 0.001`; per-action t3a AUROC 0.902).

## Interpretation — V5 DEV_STOP refined, not overturned

- **A benefit envelope EXISTS on both diseases.** Policy-independent `oracle_red_upper` = 0.413 (PD) / 0.585 (SCZ);
  `beneficial_coverage_subject_macro` = 0.796 / 0.729. So V5's DEV_STOP was **not** "the actions are always harmful" — some batches
  genuinely benefit; the open question was whether a **label-free** rule can find them.
- **PD has reliable label-free sign signal; SCZ does not.** PD's beneficial-sign is subject-clustered significant (esp. t3a); SCZ's
  is indistinguishable from the permuted null.
- Mechanism: `matched_coral` is ~neutral (mean ΔR ≈ 0), `spdim`/`t3a` are on-average catastrophic; the oracle benefit comes from
  *selecting* the beneficial batches. SCZ `f_0` is also worse-calibrated (ECE 0.19 / NLL 1.02 vs PD 0.09 / 0.68).

Net statement:

```
The action envelope exists, but a dual-disease label-free router is not justified.
PD shows significant label-free sign predictability; SCZ does not under the
subject-balanced, subject-block permutation gate.
Therefore V6 stops before protocol drafting.
```

## What is NOT authorized (binding)

```
No V6 protocol draft (no notes/ACAR_FROZEN_v6.md)   No V6 candidate space
No PD-only router built from the t3a result          No treating SCZ sign_AUROC 0.632 as a pass (it fails the permutation gate)
No relaxing perm_p <= 0.05                            No using record-weighted / per-action / PD-only AUROC as the continuation gate
No rerun of V6-A0b (no new seed / permutation scheme / model / feature set)
No policy fitting / candidate selection / threshold tuning / routing
No Stage-4 / S1 / S2 / S3                             No external / held-out / ASZED    No lockbox
```
Each of these would turn the pre-registered diagnostic gate into a post-hoc rescue.

## Lineage (Direction 2, authoritative)

```
A0/A0'  DIAGNOSTIC_ONLY (closed) — no source-free harm controller reduces deployed loss
v2      MEASUREMENT_ONLY        — label-free harm signal exists, router not deployable
v3      DEV_STOP                — coverage collapse (all-action conformal ~0.6-1.1%)
v4      SUBSTRATE_COMPATIBILITY_FAIL — DEV-only candidate does not transfer to the regenerated substrate
v5      Stage-2B DEV_STOP       — clean substrate; 0/22 candidates pass; harm-control failure; router REFUTED on DEV
v6-A0   V6_STOP                 — benefit envelope EXISTS, but dual-disease label-free sign predictability fails (SCZ not significant)
```

## Future directions (NOT authorized here; new hypothesis only)

The result localizes two possible new lines: **PD-specific / t3a-focused sign prediction**, and **SCZ `f_0` recalibration**
(identity calibration rather than an adaptation router). These may inspire a NEW project but are NOT a reason to continue ACAR-V6.
Any continuation must be a NEW, differently-named, dated protocol with a clean confirmation route on a still-sealed held-out/external
substrate. Binding caveat: **the DEV labels have now been used extensively for diagnosis; any method designed from these findings is
exploratory until confirmed on a clean held-out route.**

## Status

ACAR main line is closed:

```
V5:       DEV_STOP, no safe/useful candidate.
V6-A0:    V6_STOP, no dual-disease reliable label-free sign signal.
External: untouched.    Lockbox: sealed.
```

As rigorous protocol + negative-result research: SUCCESS. As "close the deployment gap with a label-free adaptation router":
NOT SUPPORTED. Next step: documentation closeout only. Authoritative claim status: `notes/EVIDENCE_LEDGER.md` (A10/A11).
