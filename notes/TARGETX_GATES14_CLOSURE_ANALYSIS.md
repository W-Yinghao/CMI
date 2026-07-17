# Full Gates 1-4 (63/63 EEGNet cells) — verdict + adversarial verification (authoritative)

Full real-EEG G1-primary run, phase primary, no favorable-subject filtering. Branch
`agent/cmi-trace-targetx-observability`. Manuscript FROZEN. Verified by a 3-skeptic + adjudicator Workflow
(`scripts/wf_verify_gates14.js`, run wf_633b3257-08c; high confidence).

## Recorded result
63/63 folds; identity-fallback rate 100% both datasets; `no_contested_candidate=0`; verdict
STOP_ACTIONABILITY_FAILED (Gate 2 false both). Δ_TX = 0 (identity), constrained-hindsight = 0 on all 63 folds.

## Why the RECORDED derivation is artifact-contaminated (right outcome, wrong reason)
The primary selectable basis is `contested = project_basis(B_cond_w, row(W_c))` — the task head's USED
directions. Deleting any such direction removes task-discriminative capacity, so the source-task-safety gate
(drop <= 0.02) rejects ALL of them: 0/225 informed rank>=1 actions pass safety; min recorded source_task_drop
0.079 >> 0.02; median 0.27. B_free (head-null) is non-selectable and there is no full-cond fallback (F2.1c).
=> hard-contested INTERSECT source-safety = EMPTY by construction; identity is forced independent of any
target-X statistic. G1 also structurally starves contested directions: d~ = A_s(mu_s-mu_tcal) lives ~80% in
ker(W_c) (frac_row median 0.20), so contested G1 <= ~1/5 of the discrepancy vs the full-space random Q95.

## Why the CLOSURE is nonetheless VALID (re-grounded on a fair control)
The load-bearing control lifts BOTH constraints: the UNCONSTRAINED hindsight oracle (target labels, full cond
span rank~10 including the free/head-null space, no safety gate; firewall: query_y only for outcome) STILL
fails a fair equal-budget random control:
 - unconstrained-hindsight minus best-of-random (matched budget) 95% CI: BNCI2014 [-0.050,-0.032],
   BNCI2015 [-0.043,-0.012] (informed BELOW random);
 - matched-K P(informed-max > random-max) = 0.315 / 0.141 (both < 0.5, no signal);
 - informed/cond-span deletions are on AVERAGE more harmful than random (mean util -0.081 vs -0.016;
   -0.153 vs -0.004; frac-helpful 21%/14% vs 35%/37%);
 - the nominal +0.032 unconstrained mean is selection-inflation + one seed-unstable subject
   (BNCI2015 subj1 = 40% of positive mass; flagship fold 0.36/0.01/0.28 across seeds; matched by a random
   rank-1 deletion +0.22). Mean excluding the top subject = 0.028.
RESIDUAL (honest): raw unconstrained per-subject cluster CI does exclude 0 (2014 [0.006,0.027],
2015 [0.013,0.081]) -> deletion CAN help some subjects in absolute terms, but that mass is NOT
target-X-attributable (fails vs random) and cannot be selected on.

## Decision implication (PM to confirm)
STOP_ACTIONABILITY_FAILED HOLDS as a valid closure of the target-X selector line, re-grounded on the fair
random control. Do NOT run GPU Gate 5 (per the pre-registered both-datasets-Gate2-fail rule). A SOFT functional
anchor (dR_source<=delta replacing hard U subset row(W_c)) is NOT warranted: it merely opens the free space the
failed unconstrained oracle already searches. This is a full-EEG, cluster-CI, fair-control NEGATIVE, consistent
with the project arc: leakage measurable, but neither source- nor target-X-observables identify a
deployable DG-beneficial deletion. Track B geometry stands as the mechanistic characterization (contested span
tiny; subject/DG geometry mostly head-null; cond head_overlap 0.05-0.23); Track B B2/B3 as a METHOD path are
correspondingly not warranted unless the PM redefines the objective.


## C0 reproducible closeout (fair calibration-selected equal-budget random-basis oracle, 63/63 cells, n_random=100)
`scripts/aggregate_unconstrained_oracle_specificity.py` -> unconstrained_oracle_closure_verdict.json = SUBJECT_SUBSPACE_SELECTOR_CLOSED.
Both informed and random oracles select GREEDILY on T_cal labels (equal rank+budget) and score only on T_query.
dI_specific (informed - mean random): BNCI2014 +0.0028 [-0.0053,+0.0112], BNCI2015 +0.0040 [-0.0094,+0.0177] (CI includes 0).
Informed beats random Q95 in only 3.7% / 5.6% of subjects (~ the 5% chance rate for a Q95 threshold) -> no subspace-specific utility. Machine-readable: unconstrained_oracle_{specificity_full,subject_rows}.csv.
