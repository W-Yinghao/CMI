# CSC-realEEG B5.0 — frozen random deep features are a NO-OP control (fitted-null FAIL reproduces; canary inconclusive)

```
Scope: B5.0 feature-robustness canary | development diagnostic only | NOT deployable | NO tag | NOT confirmatory
  B3 certifier path BYTE-UNCHANGED (only the feature input is swapped to a frozen deep embedding)
  no method / statistic / n_boot / rank / C / margin change | no oracle generator | encoder never sees any label
  labels REGENERATED on the deep features (not transplanted) | fresh dev seed base 30e6 (disjoint from v2/P3/R1)
```

**Safe headline (red-team-corrected):** *A frozen random-init EEGNet embedding is empirically a statistical REPLICA
of SM16 on every axis — a near-isometry that preserves the covariance geometry the certifier keys on — so it is a
NO-OP control. Under it the fitted-h0 under-dispersion FAIL reproduces unchanged (NULL_cov false-confirm 10/50 = 0.20,
fixed-margin p floor-pinned in 90% of cohorts, T_z ≈ 6) AND the injected concept yields NO decision-level power
(POS_concept 10/50 = 0.20 is statistically IDENTICAL to NULL_cov 0.20; Fisher OR=1.0, p=1.0). With no separated
positive control, B5.0 CANNOT adjudicate whether deep features resolve the FAIL — it is INCONCLUSIVE as a safety test.
The only thing it establishes is that a RANDOM encoder cannot move the operating point, which makes B5.1 (a LEARNED /
covariance-reshaping encoder) the sole informative continuation. No positive claim.*

## Feature family (frozen, cache red-team CLEAN)
`b5_features/b5_0_random_encoder/LEE2019_B5_0.npz` (sha256 `3670f174…`). Frozen random-init EEGNet
(`F1=8,D=2,F2=16,kern=64,pool1=4,pool2=8`, penultimate flatten, **no classifier head, eval mode, no training**,
seed 20260706, weight sha256 `a9f2dc94…`) maps each raw SM16 window `[16×384]` → 192-dim Z_deep. Preprocessing
IDENTICAL to SM16 (SM16_no_FCz 16 ch, 8–30 Hz Butterworth filtfilt, 0.5–3.5 s, resample 384). 10 800 trials.
Cache red-team (leakage PASS + integrity PASS, verified directly): encoder label-free; npz hash==manifest;
full-rank; determinism repro=0; preprocessing + trial-inventory parity with SM16; manifest distinct.

## Canary (n=50/condition, base 30e6, m=30) — statistical replica of SM16
Injection + certifier byte-unchanged; only `IF.CACHE` = Z_deep. 300 cohorts, 0 worker errors, task-id sha256 `491f1481…`.

| condition | GT | confirm | rate | null_sd_T | T_z_med | ffp@floor | session_auc | note |
|---|---|---|---|---|---|---|---|---|
| **NULL_cov** | NO_CONCEPT | 10/50 | 0.200 | 0.000288 | 6.15 | 0.90 | 0.518 | false-confirm |
| **NULL_cov_plus_label** | NO_CONCEPT | 8/50 | 0.160 | 0.000313 | 6.89 | 0.84 | 0.533 | false-confirm |
| NULL_label | NO_CONCEPT | 0/50 | 0.000 | 0.000328 | 8.48 | 0.92 | 0.531 | — |
| random_label_control | NO_CONCEPT | 0/50 | 0.000 | 0.000094 | −0.02 | 0.00 | 0.528 | floor check ✓ |
| POS_concept | CONCEPT | 10/50 | 0.200 | 0.000694 | 2.83 | 0.46 | 0.523 | **NOT power** (= NULL_cov) |
| POS_concept_plus_cov | CONCEPT | 12/**48** | 0.250 | 0.000480 | 3.42 | 0.46 | 0.532 | 2 undecided → denom 48 |

## Reading (red-team-corrected; honest, adversarial)
1. **B5.0 is a NO-OP control.** It is statistically indistinguishable from SM16 on every axis (NULL_cov false-confirm
   0.20 vs 0.19, T_med 2.2e-3 vs 2.4e-3, T_z 6.1 vs 6.0, session_auc 0.52 vs 0.52, null_sd 2.9e-4 vs 3.0e-4;
   between-family Fisher p=0.85 NULL, 0.38 POS). A frozen random encoder preserves the covariance geometry the
   certifier exploits, so it reproduces SM16 **by construction** and carries near-zero information about how a
   *learned* representation would behave. "Deep features preserve the FAIL" over-reads a null — do NOT cite B5.0 as
   evidence that the FAIL is feature-invariant.
2. **The fitted-null under-dispersion FAIL reproduces unchanged.** NULL_cov false-confirm 0.20 (CP95 ≈ [0.11, 0.32],
   ~8× the 0.025 per-decision budget); fixed-margin p pinned at the 1/201≈0.005 floor in 84–92% of NULL cohorts;
   large T_z (6.15/6.89/8.48) vs tiny null_sd_T — the same collapse signature as SM16 B4. The mechanism is not fixed.
3. **NO usable concept power → the canary is INCONCLUSIVE for safety.** POS_concept 10/50 (0.20) is *identical* to
   NULL_cov 10/50 (0.20) (Fisher OR=1.0, p=1.0); T_z is *higher* for the no-concept NULL_cov (6.15) than for POS
   (2.83) — confirmations are covariate/under-dispersion driven, not concept. `any_pos_signal=true` means only "≥1 POS
   confirm" and MUST NOT be read as power. With no separated positive control, the low NULL_label/random rates carry
   zero safety information. (The concept injection *does* perturb the certifier internals — POS floor-frac 0.46 vs
   NULL 0.90, larger POS null_sd_T — so it is not a pure internal no-op; but that perturbation does not separate the
   *decision-level* confirm rate from the covariate false-confirm floor, which is all a verdict rides on.)
4. **Sanity checks pass:** random_label_control (labels destroyed) → null_sd 9.4e-5, T_z≈0, 0/50; NULL_label 0/50.
5. **The n=1 smoke was an unrepresentative low-tail cohort** (T~1e-4); the full n=50 run is the readout.

## CI / dependence caveat (must be recorded; blocks any future SAFE inference from raw counts)
The 50 cohorts/condition each draw 30 of only 54 subjects under one shared **global (transductive) z-score** and one
frozen encoder, so confirm outcomes are positively correlated (effective n < 50) → naive binomial/Wilson CIs are
anti-conservative. The engine's own bound is documented as cohort-level, NOT a subject-cluster bound. The 10 NULL_cov
false-confirms may reflect a few recurrent subjects, not 10 independent failures. seed = base + CIDX·STRIDE + r drives
both the injection RNG and the certifier bootstrap (same code path as SM16, not a B5-specific confound).

## Consequence for B5.1 (learned/SSL features) — now the ONLY informative next step
Because a random encoder cannot move the operating point, the feature-dependence question is answerable ONLY by a
representation that actually re-shapes the covariance (a learned/SSL encoder). B5.1 must clear, IN ORDER:
1. **Positive-control FIRST:** POS_concept true-confirm materially and significantly ABOVE the NULL_cov false-confirm
   rate (real separation, not equality). Until this holds, any NULL reading is uninformative by construction.
2. **Then safety:** NULL_cov false-confirm ≤ α with the under-dispersion signature broken (floor-frac collapsed,
   null_sd_T widened toward the oracle's ~1.66×, T_z off the collapse regime).
3. **n ≥ 60:** n=50 can CONFIRM a fail but structurally CANNOT certify safety (0/50→CP upper 5.8%; 2/50→12%). Do NOT
   read a future "2/50 = 4%" as safe — that is the underpowered trap.
4. **Subject-cluster bound**, not raw counts, for any SAFE claim.

## Relation to the rest of the line
The lever that DID control held-out type-I is the calibrated abstention **router** ([[router_stage1_validation]]),
which works on SM16 log-bandpower where observed_T separates concept from covariate-null — a separation this no-op
random Z_deep does not exhibit. B5.0 does not bear on the router result. Builds on [[p3_oracle_diagnostic]] (the
under-dispersion mechanism) and [[b4_stage1_canary]] (SM16 estimable-null attempts, all FAIL).

## Package (diagnostic-only, no tag)
`csc/results/b5_features/b5_0_random_encoder/`: cache manifest + cache_checks + SHA256SUMS; canary per-condition JSONL
(12 shards) + `.prov.json`/`.sha256`; `b5_canary_tables.json` (with red-team caveat annotation); `b5_redteam_checks.json`;
scripts (`realeeg_b5_canary.py`, `b5_canary_merge.py`, `b5_0_build_cache.py`, sbatch + lifecycle). Red-team: fidelity
PASS, persist FAIL (inconclusive), overclaim MINOR_ISSUE — verdict "commit as diagnostic-only NO-OP control; no
positive claim; B5.1 is the informative continuation."
