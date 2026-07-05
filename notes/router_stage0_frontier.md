# CSC-realEEG Router Stage R0 — a monotone observed-T abstention rule has a VIABLE exposed frontier

```
Scope: Router Stage R0 | development-only exposed frontier | not confirmatory | not deployable
  not a concept certificate | no method/h0/h1/statistic/feature change | no oracle feature use | no tag
```

**Safe headline:** *After the B4 "better estimable null" direction closed (bagged/nested/richer fitted nulls all
fail), the deployable object pivots from a repaired certificate to a SELECTIVE ACTION RULE. A deliberately simple
monotone rule — ALLOW a B3 concept alert only if observed_T ≥ τ, else ABSTAIN — has a viable safety/utility
frontier on the EXPOSED bank: at τ=0.00418 the covariate-drift nulls drop to 7/300 (CP95 upper ≈0.043 < 5%)
while ~75% of true-concept alerts are retained. This is a development-exposed VIABILITY signal, not a validated
result; a fresh-seed R1 (locked τ) is required and not yet authorized.*

Follows the closed B4 line ([b4_stage1_canary.md](b4_stage1_canary.md), [b4_stage1b_canary.md](b4_stage1b_canary.md))
and P3 ([p3_oracle_diagnostic.md](p3_oracle_diagnostic.md)). The router is NOT a concept certificate: it may say
`ALLOW_CONCEPT_ALERT_UNDER_CALIBRATED_RISK` / `ABSTAIN_UNCERTIFIED` / `NO_ACTIONABLE_EVIDENCE`, controlling an
ACTION risk under calibrated null regimes — it does NOT claim concept shift is identified in the strong CSC sense.

## R0 rule (monotone; can only remove B3 confirmations)
`ALLOW if (B3 method_confirm==True AND observed_T ≥ τ) else ABSTAIN`. Primary deployable score = `observed_T`.
No oracle field, no condition label in the decision (condition only buckets the evaluation, never pooled).
Deterministic grid = τ=0 + sorted unique observed_T among all B3 method-confirms. Source = committed exposed
`csc/results/p3_forensics/p3_internal_forensics_merged.jsonl` (1800 cohorts, 6 conditions × 300).

## Frontier (allow counts /300; CP95 upper for nulls)
| τ | NULL_cov | NULL_cov+label | NULL_label | random | POS | POS+cov | NULL_cov CP95u |
|---|---|---|---|---|---|---|---|
| 0.0 (pure B3) | 57 | 32 | 1 | 0 | 81 | 78 | 0.231 |
| 0.00365 | 10 | 11 | 1 | 0 | 63 | 63 | 0.056 |
| **0.00418** | **7** | **7** | 1 | 0 | **61** | 56 | **0.043** |
| 0.00537 | 1 | 2 | 0 | 0 | 56 | 48 | 0.016 |
| 0.00802 | 0 | 0 | 0 | 0 | 45 | 43 | 0.010 |

method_confirm baseline /300: NULL_cov 57, NULL_cov+label 32, NULL_label 1, random 0, POS_concept 81, POS_cov 78.

## Viability screen → VIABLE
Smallest R1-eligible **τ=0.00418**: NULL_cov 7/300 & NULL_cov_plus_label 7/300 (both CP95u ≈0.043 < 0.05),
NULL_label 1, random 0; **POS_concept 61/300 (75% retention)**, POS_cov 56/300 (72%). **98 thresholds** meet BOTH
the safety screen AND the preferred utility (POS_concept≥20, POS_cov≥15). A viable safety/utility frontier exists.

Notably this exceeds the oracle null's retention (40/300 POS_concept on the same bank) — because thresholding
effect size *directly* is more efficient for this discrimination than the oracle's uniform p-recalibration.
(P3 already showed false-confirms and true concepts differ mainly in raw effect size T (~4×), not on the
studentized/subject-consistency axes — so observed-T is the natural score.)

## Honest limits
- **Development-EXPOSED.** τ was read off the exposed P3 bank; this is a viability frontier, not a validated
  operating point. The CP95 upper at τ=0.00418 (0.043) is close to 0.05.
- **Not a certificate.** The router controls action risk; it does not identify concept shift.
- **Utility is modest by design** (the oracle itself retained only ~40–34/300), and R0's higher number is
  in-sample selection — expect regression on fresh data.

## Proposed R1 (NOT authorized — awaiting reviewer go)
Fresh calibration seed block: pick τ on NULL_cov & NULL_cov_plus_label requiring CP95 upper ≤0.05 per null kind
(conservatively, e.g. τ≈0.00537 → NULL 1–2/300). Fresh HELD-OUT seed block: evaluate the LOCKED τ on all 6
conditions. **Allowed future calibration = the abstention threshold τ only; still NOT authorized = recalibrating
B3 p-values or inflating the null.**

## Verification (independent red-team)
See `csc/results/router_stage0_frontier/router_stage0_redteam_checks.json` — re-derivation of the frontier from
the committed P3 rows + guardrails (monotone: allow ⊆ method-confirms, never allows a method-negative; no oracle
field / no condition label in the decision; kind-specific nulls not pooled; deterministic grid).

## Package (diagnostic-only, no tag)
`csc/results/router_stage0_frontier/`: `router_stage0_manifest.json`, `router_stage0_frontier.json`,
`router_stage0_rows.jsonl` (1800, no oracle fields), `router_stage0_tables.md`, `router_stage0_redteam_checks.json`,
`SHA256SUMS`, `scripts/`.

## Red-team verification + exact-τ correction
Independent 2-lens red-team: **0 serious issues; guardrails MATCHES** (monotone allow ⊆ B3-confirms, never allows
a method-negative; no oracle field / no condition label in the decision; kind-specific nulls not pooled;
deterministic 249-point grid regenerable; allow counts non-increasing in τ; oracle-free rows). Re-aggregation
reproduces the baseline + the frontier row EXACTLY from the committed P3 JSONL.
- **Exact τ correction (cosmetic):** the eligible cutpoint is **τ = 0.0041848** (displayed above as 0.00418) →
  NULL_cov 7/300 (CP95u 0.0434), NULL_cov_plus_label 7/300 (0.0434). Using the *rounded* 0.00418 instead gives
  NULL_cov 8/300 (CP95u 0.0476) — **both < 0.05**; safety/viability unaffected. The committed
  `router_stage0_frontier.json` stores the exact τ and NULL_cov=7.
- **Provenance nit:** `router_stage0.py` reads a local copy `realeeg_feas/p3_forensic/p3_internal_forensic_merged.json`
  verified byte-equivalent to the committed `csc/results/p3_forensics/p3_internal_forensics_merged.jsonl` (0/1800
  mismatches; sha256 sidecar matches); the manifest cites the committed jsonl as the source of record.
