# Project B Step-3C Real EEG Benchmark Expansion Report

*Bounded real benchmark expansion, NOT a full MOABB benchmark. Auto-generated.*

## 1. Run status
- runtime_bounded_partial = False
- datasets: BNCI2014_004(ok)

## 2. Aggregate summary

| dataset | eval_unit | mode | n_tgt | mean_strict | mean_rawdTTA | cov | id_rate | off_tta | acc_bAcc | avoided | interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BNCI2014_004 | session | in_source_subject_q95 | 4 | 0.640 | -0.140 | 0.40 | 0.40 | 0.00 | 0.614 | 0.142 | raw TTA harmful; router mostly refuse; TTA blocked (ACAR degenerate/unavailable) |
| BNCI2014_004 | session | nested_source_subject_excess_q95 | 4 | 0.640 | -0.140 | 0.40 | 0.40 | 0.00 | 0.614 | 0.142 | raw TTA harmful; router mostly refuse; TTA blocked (ACAR degenerate/unavailable) |
| BNCI2014_004 | subject | in_source_subject_q95 | 4 | 0.640 | -0.140 | 0.50 | 0.50 | 0.00 | 0.618 | 0.140 | raw TTA harmful; router mostly identity; TTA blocked (ACAR degenerate/unavailable) |
| BNCI2014_004 | subject | nested_source_subject_excess_q95 | 4 | 0.640 | -0.140 | 0.50 | 0.50 | 0.00 | 0.618 | 0.140 | raw TTA harmful; router mostly identity; TTA blocked (ACAR degenerate/unavailable) |

## 3. Subject-level routing
See rows with eval_unit=subject above.
## 4. Session-level routing
See rows with eval_unit=session above.
## 5. TTA harm / benefit
raw_offline_delta_bacc per fold in fold_summary.csv.
## 6. Router action distribution
action_counts per fold in fold_summary.csv; per-domain in per_domain_decisions.csv.
## 7. Reason-code audit
reason_code_audit.csv (top-level vs identity-action vs TTA-blocker).
## 8. Comparison to Step-3A
Step-3A: 2 targets, subject only. Step-3C: more targets + session-level; OFFLINE_TTA remains blocked under degenerate/unavailable ACAR-harm.
## 9. Claim boundary update
See step3c_claim_boundary_update.json. This remains a bounded real benchmark expansion, not a full benchmark; no target-label-tuned thresholds.
