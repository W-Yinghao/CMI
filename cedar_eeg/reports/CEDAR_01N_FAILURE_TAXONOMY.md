CEDAR_01N - Failure Taxonomy

Status: read-only taxonomy over existing CEDAR_01 artifacts. No selector was
rerun. No candidate universe, utility, threshold, or red-team semantic was
changed.

Inputs

```text
run_manifest:    results/cedar/p0_real_shadow/cedar01_bnci2014_001_seed0/run_manifest.json
candidate_table: results/cedar/p0_real_shadow/cedar01_bnci2014_001_seed0/candidate_table.csv
red_team:        results/cedar/p0_real_shadow/cedar01_bnci2014_001_seed0/red_team.json
source_summary:  results/cedar/p0_real_shadow/cedar01_bnci2014_001_seed0/source_decision_summary.json
```

Top-level outcome

```text
total candidates: 54
ACCEPT:           0
ABSTAIN:          0
REPORT_ONLY:      54
```

Red-team outcome:

```text
failures: 0
warnings: 18
warning type: no accepted candidate; report-only or abstention outcome
```

The warnings are scientific non-actionability signals, not protocol failures.

Taxonomy definitions

```text
failed_leakage_vs_random:
  Candidate did not satisfy leakage actionability under the frozen CEDAR gate.
  This includes the frozen min_leakage_drop_frac gate and the PM leakage-vs-random
  criterion.

failed_source_noninferiority:
  Source bAcc drop exceeded 1 percentage point or CE/NLL was materially worse.

failed_permutation_null:
  Permutation null advantage was not low.

failed_grouped_probe:
  Grouped split was invalid or had train/eval group overlap.

failed_collapse_guard:
  Candidate-level effective-rank / entropy collapse guard failed.

failed_r3_or_r3_missing:
  R3 bridge was unavailable in this feature-only shadow audit. This is retained
  as a mandatory caveat for any future method claim, but source actionability
  already failed before R3 could matter.

failed_tie_break_or_completeness:
  Red-team tie-break, candidate completeness, or source-only selection contract failed.
```

Aggregate counts

Primary frozen accept reason counts from the CEDAR_01 result:

```text
leakage_drop_frac below 0.300:          54 / 54
random_control_drop_frac exceeds gate:  18 / 54
source_bacc_drop exceeds 1 point:        3 / 54
```

PM taxonomy counts:

```text
failed_leakage_vs_random:        54 / 54
  min_leakage_drop_frac failed:  54 / 54
  PM random criterion failed:    48 / 54
  random gate exceeded:          18 / 54

failed_source_noninferiority:     3 / 54
failed_permutation_null:          0 / 54
failed_grouped_probe:             0 / 54
failed_collapse_guard:            0 / 54
failed_r3_or_r3_missing:         54 / 54
failed_tie_break_or_completeness: 0 / 54
```

Per-backbone taxonomy

```text
EEGConformerMini:
  candidates:                    27
  accepted:                       0
  report_only:                   27
  failed_leakage_vs_random:      27
  PM random criterion failed:    22
  failed_source_noninferiority:   0
  failed_permutation_null:        0
  failed_grouped_probe:           0
  failed_collapse_guard:          0
  failed_r3_or_r3_missing:       27

EEGNetMini:
  candidates:                    27
  accepted:                       0
  report_only:                   27
  failed_leakage_vs_random:      27
  PM random criterion failed:    26
  failed_source_noninferiority:   3
  failed_permutation_null:        0
  failed_grouped_probe:           0
  failed_collapse_guard:          0
  failed_r3_or_r3_missing:       27
```

Metric ranges

```text
EEGConformerMini:
  leakage_drop_frac:    -0.180 to 0.198
  leakage_drop_abs:     -0.0499 to 0.0426
  random_drop_abs:      -0.0020 to 0.0206
  source_bacc_drop:     -0.0030 to 0.0028
  source_ce_delta:      -0.0123 to 0.0158
  effective_rank_ratio:  0.925 to 0.988

EEGNetMini:
  leakage_drop_frac:    -0.0463 to 0.117
  leakage_drop_abs:     -0.0329 to 0.0783
  random_drop_abs:       0.0091 to 0.0895
  source_bacc_drop:     -0.0037 to 0.0230
  source_ce_delta:      -0.0054 to 0.0227
  effective_rank_ratio:  0.787 to 1.004
```

Best observed candidates by backbone

```text
EEGConformerMini best leakage fraction:
  candidate: drop_top_4_of_32
  leakage_drop_frac: 0.198
  leakage_drop_abs: 0.0426
  random_drop_abs: 0.0151
  source_bacc_drop: 0.0000
  decision: REPORT_ONLY
  reasons: leakage_drop_frac < 0.300; random_control_drop_frac exceeds gate

EEGNetMini best leakage fraction:
  candidate: drop_top_4_of_16
  leakage_drop_frac: 0.117
  leakage_drop_abs: 0.0783
  random_drop_abs: 0.0661
  source_bacc_drop: 0.0106
  decision: REPORT_ONLY
  reasons: leakage_drop_frac < 0.300; source_bacc_drop > 0.010; random_control_drop_frac exceeds gate
```

Interpretation

Leakage evidence was measurable in some folds, but it was not actionable under
the frozen CEDAR contract. The actionability failure is not due to grouped split
failure, permutation-null failure, target leakage into selection, tie-break
failure, or deployable artifact leakage. It is primarily a source-side
acceptance failure: all candidates fail the minimum leakage-drop gate, and most
also fail the leakage-vs-random actionability check.

R3 note

R3 was unavailable in this feature-only shadow audit and remains a required
caveat for any future method claim. Because the CEDAR_01 source-side
actionability gate already failed with 0 ACCEPT candidates, R3 missing is not a
rescue path and does not justify P1.

Target diagnostics

Target diagnostics were computed after source-only selection and did not affect
candidate ranking, acceptance, tie-break, or this failure taxonomy.

Forbidden interpretation

Do not interpret these results as "almost positive". Do not use target
diagnostics to motivate P1. Do not add new k values, rewrite the utility, or
rerun the selector to rescue the result.
