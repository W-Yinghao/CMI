# CSC-realEEG B5.1 — a LEARNED SSL encoder leaves the fitted-null FAIL intact (separation test underpowered)

```
Scope: B5.1 SSL feature canary | development diagnostic only | NOT deployable | NO tag | NOT confirmatory
  B3 certifier + injection BYTE-UNCHANGED (only feature input = the SSL embedding) | SSL trained LABEL-FREE
  labels REGENERATED on Z_ssl | fresh dev seed base 40e6 | n=80/condition (reviewer n>=60)
```

**Safe headline (red-team-corrected — scoped):** *Even a LEARNED, label-free SSL EEGNet representation (masked
time-column reconstruction, MSE 26.82→17.73 — it genuinely learned) leaves the certifier's fitted-null
under-dispersion FAIL FULLY INTACT: NULL_cov false-confirm 18/80 = 0.225 (~9× the 0.025 budget), fixed-margin p
floor-pinned in 92.5% of no-concept cohorts, T_z 6–9 with the no-concept>concept inversion preserved. It does NOT
clear the positive-control gate (POS_concept 24/80 = 0.30 vs NULL_cov 0.225, Fisher one-sided p=0.18, diff-CI
[−0.19,+0.34], ~24% power). The ONLY supported reading: feature richness — random ([[b5_0_random_encoder]]) or THIS
one learned encoder — is not the lever; the study is too underpowered to be leaned on as affirmative proof that learned
features fail.*

## Feature family (cache red-team CLEAN)
`b5_features/b5_1_ssl_encoder/LEE2019_B5_1.npz` (sha256 `04d07143…`). Same EEGNet trunk as B5.0 but SSL-TRAINED
(label-free masked time-column reconstruction, 60 epochs, mask 0.5; weight sha256 `800d7b72…` ≠ B5.0's `a9f2dc94…`).
Cache red-team (leakage PASS + integrity PASS): loss uses ONLY X — no MI/synthetic/session/subject label enters
training, mask, decoder, or loss; batch order is randperm; frozen after training; learned (MSE 26.8→17.7); full-rank
192-dim; preprocessing + trial-inventory parity with SM16/B5.0; distinct manifest. Low note: transductive SSL
(label-free, disclosed).

## Canary (n=80/condition, base 40e6) — positive-control-FIRST
| condition | GT | confirm/80 | rate | T_z_med | ffp_floor | auc |
|---|---|---|---|---|---|---|
| **NULL_cov** | NO_CONC | 18 | 0.225 | 6.35 | 0.93 | 0.521 |
| NULL_cov_plus_label | NO_CONC | 12 | 0.150 | 6.21 | 0.85 | 0.531 |
| NULL_label | NO_CONC | 0 | 0.000 | 8.95 | 0.93 | 0.523 |
| random_label_control | NO_CONC | 0 | 0.000 | 0.01 | 0.01 | 0.519 |
| POS_concept | CONCEPT | 24 | 0.300 | 3.67 | 0.53 | 0.533 |
| POS_concept_plus_cov | CONCEPT | 20 | 0.250 | 2.69 | 0.44 | 0.538 |

## Reading (red-team-corrected; posfirst PASS, overclaim MINOR_ISSUE)
1. **The fitted-null FAIL is fully INTACT under a learned encoder (fully supported).** NULL_cov 0.225 (~9× budget);
   fixed-margin p floor-pinned in 92.5% of no-concept cohorts; T_z 6–9; the no-concept>concept T_z **inversion**
   (NULL_cov 6.35 > POS_concept 3.67) is the same covariate-driven-firing signature as B5.0. A demonstrably-trained
   representation does not fix the certifier's null.
2. **Positive-control gate NOT cleared — but INCONCLUSIVE, not a strong negative.** POS_concept 0.30 vs NULL_cov 0.225:
   Fisher one-sided p=0.18, Newcombe diff-CI [−0.19,+0.34] spans 0; POS_concept_plus_cov (0.25) also fails to separate
   (p=0.43). At n=80 the power to detect the observed 0.30-vs-0.225 gap is only **~24%** (needs true POS ~0.42+ for
   80%). The n≥60 bar was set to CONFIRM a FAIL, not to power a separation test — so "POS doesn't separate" is
   underpowered/inconclusive. n=80 IS well-powered for the ~2× separation a WORKING detector needs, and found none, so
   a **usable** concept detector is ruled out; a modest (non-usable) separation is not.
3. **Do NOT over-read.** POS 0.30 > B5.0's 0.20 and > NULL 0.225 is NOT "learned features help" (both n.s.); do not
   comparator-shop the POS vs NULL_cov_plus_label p=0.018 cell. The B5.0→B5.1 rise in BOTH arms (NULL 0.20→0.225,
   POS 0.20→0.30) with the GAP unchanged is consistent with subject/session covariance overfit inflating both, not
   separation.
4. **Scope:** "feature richness is not the lever" holds for RANDOM (B5.0) + THIS ONE SSL EEGNet encoder (n=1 learned
   family), NOT universal.
5. **Type-I caveat:** NULL_cov CP95u 0.315 / cohort-boot CI anti-conservative (30 of 54 subjects shared); needs a
   subject-cluster bound — moot since POS does not separate (vacuous safety read).

## Relation
Confirms the certifier's fitted-null under-dispersion is not fixed by richer features — random ([[b5_0_random_encoder]])
or this learned SSL encoder. The deployable lever remains the abstention router ([[router_stage1_validation]]), which
itself only works in the soft-covariate regime ([[router_r1_scaleup]]). Builds on [[p3_oracle_diagnostic]].
Package: `csc/results/b5_features/b5_1_ssl_encoder/`.
