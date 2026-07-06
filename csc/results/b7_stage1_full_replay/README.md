# B7.1 dual-witness full dev replay (diagnostic-only) -- SAFETY-FAIL on the mixed covariate+prior null

primary rule = old_B3_confirm AND B6_plain_confirm  (each witness's own gated state; p_dual=max diagnostic only)
fresh disjoint base 300e6, 8 conditions x 300 = 2400 cohorts; witnesses FROZEN; B6-FM NOT used
development-only / not confirmatory / not deployable / NOT a universal type-I guarantee / no tag / NO retuning on fail
protocol pre-registered + committed BEFORE the run (b7_stage1_protocol.json, sha ed986506).

RESULT (red-team: provenance PASS + science MINOR_ISSUE, 0 serious). B7-primary per condition: NULL_cov_soft 3,
NULL_cov_plus_label 24, strong_0.81 2, strong_0.94 2, NULL_label 0, random 1, POS_concept 44, POS+cov 53.
- Cancels PURE nuisances (strong-cov 2/2, NULL_label 0) + retains STRONG utility (POS 44/53 >= 20/15).
- SAFETY-FAIL: NULL_cov_plus_label 24/300 = 8% (CP95u 0.11) >> 7/300 cap. The MIXED cov+prior null makes BOTH
  witnesses false-confirm (OLD covariate-blind 33, B6 prior-blind 84); their co-occurrence is positively correlated
  (both 24 vs independence 9.24 = 2.6x) -> STRUCTURAL, not tunable. The exposed n=50 (3/50=6%) was screen-tolerated
  above the confirmation bar; the fresh n=300 makes it decisive.

VERDICT: SAFETY-FAIL (kind-specific). Per protocol NO retuning. The observational null-repair line (B3->B4->B5->B6->B7)
is exhausted on real EEG (no single null, fixed-margin variant, or dual-witness AND controls the mixed cov+prior cell
while retaining power). NEXT = B8 information-contract / randomized paired audit design (reviewer decision), NOT another
witness/variant. See notes/b7_stage1_full_replay.md. Related: b7_stage0_dual_witness.md, b6_0_condition_randomization.md.

Note: shards/ each row carries BOTH witnesses (same cohort -> no cross-file join); b7_stage1_rows.jsonl = derived
per-cohort B7 states; b7_stage1_tables.json = per-condition disjoint partition (sum 300) + CP95u + red_team block.
