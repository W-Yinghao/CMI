// HISTORICAL: this workflow tested the hypothesis "NO_CONFIRMED_TICKET on both datasets" and REFUTED it.
// Adjudicated outcome = TARGET_HINDSIGHT_ONLY (existence confirmed under the mechanism-matched greedy oracle;
// the prefix oracle/selector that produced the NO_CONFIRMED hypothesis was too weak). Kept for provenance;
// the CONTEXT string below states the initial (refuted) hypothesis on purpose.
export const meta = {
  name: 'verify-dg-identifiability-verdict',
  description: 'Adversarially test the initial NO_CONFIRMED_TICKET hypothesis -> REFUTED to TARGET_HINDSIGHT_ONLY',
  phases: [
    { title: 'Refute', detail: '5 independent skeptics each attack the verdict from one angle' },
    { title: 'Adjudicate', detail: 'synthesize surviving refutations -> hold / qualify / flip' },
  ],
}

const REPO = '/home/infres/yinwang/CMI_AAAI_cmitrace'
const PY = '/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3/bin/python'

const CONTEXT = `
You are adversarially verifying a research VERDICT in the worktree ${REPO} (branch agent/cmi-trace-dg-oracle).
Always: cd ${REPO}; run python as ${PY} with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; prefer reading existing
artifacts over recompute; if you recompute, keep it CHEAP (<=4 folds, seed 0, small bootstrap) and say so.

THE VERDICT UNDER TEST:
  On real EEGNet features (BNCI2014_001 = 8-src-subject LOSO 4-class; BNCI2015_001 = binary), the "DG-erasure
  ticket" (a subject-subspace whose deletion improves cross-subject generalization) is judged
  NO_CONFIRMED_TICKET on BOTH datasets:
   * cross-fitted TARGET oracle (select deletion subset on T_select target-labels, report on DISJOINT T_query)
     COLLAPSES to ~0: BNCI2014 +0.005 [lo -0.000], BNCI2015 +0.001 [lo -0.003] (LCB<=0). The earlier
     un-cross-fitted numbers were +0.016 / +0.054 -> claimed to be subset-search optimism.
   * the NESTED source-only selector (inner LOSO over source subjects; pseudo-target excluded from basis
     estimation, head fit, and rank selection; refittable rule = family x contested x k x objective) REFUSES
     (k*=0) across ALL 16 pre-declared configs (4 bases {marg,cond,rule,grad} x {full,contested} x
     {mean_1se,cvar25}) -> meta target-gain +0.000 everywhere.
   * directly minimizing conditional subject leakage (CMI-only) was harmful on both (first pass -0.010/-0.063).
  CONCLUSION DRAWN: do NOT build a source-only differentiable subspace supermask (P2); the DG-beneficial
  deletion is either non-existent-once-cross-fitted or not source-identifiable under the tested basis+selector.

KEY FILES:
  tos_cmi/eval/dg_identifiability.py        (all estimators: crossfit_target_oracle[mode=prefix|greedy],
                                             nested_source_meta_multi, select_k_star, the 4 bases, contested_basis)
  scripts/run_dg_identifiability.py / scripts/aggregate_dg_identifiability.py
  results/cmi_trace_dg_identifiability/{dg_identifiability_verdict.json, dg_identifiability_table.csv,
                                        <ds>_EEGNet/raw_rows.jsonl}
  tos_cmi/tests/test_dg_identifiability.py  (synthetic anchor: machinery CAN recover a balanced shortcut,
                                             correctly refuses a majority-sign shortcut)
  results/cmi_trace_dg_oracle/dg_oracle_verdict.json (the FIRST-PASS un-hardened result)

Your job: try HARD to REFUTE the verdict (find a reason the conclusion is wrong or overstated). Default to
"refuted=false" ONLY if you genuinely cannot break it. Return raw structured data, not prose for a human.
`

const FINDING_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['angle', 'refuted', 'severity', 'what_i_checked', 'evidence', 'numbers', 'verdict_impact'],
  properties: {
    angle: { type: 'string' },
    refuted: { type: 'boolean', description: 'true if you found a real hole that undermines the verdict' },
    severity: { type: 'string', enum: ['none', 'minor', 'major', 'fatal'] },
    what_i_checked: { type: 'string', description: 'concrete commands/files, and whether recompute or read-only' },
    evidence: { type: 'string', description: 'the specific finding with file:line or numbers' },
    numbers: { type: 'string', description: 'any CIs / deltas you produced, verbatim' },
    verdict_impact: { type: 'string', enum: ['holds', 'needs_qualification', 'flips_to_hindsight_only', 'flips_to_identifiable', 'inconclusive'] },
  },
}

const SKEPTICS = [
  { angle: 'oracle-too-weak', task: `Is the EXISTENCE oracle too weak, hiding a real ticket? The committed
    crossfit_target_oracle default is mode="prefix" (delete top-k of the ordered basis). Run the STRONGER
    mode="greedy" (arbitrary-coordinate forward selection) cross-fitted, on the cond/full basis, for ~4 folds
    of EACH dataset (seed 0), and also the un-cross-fitted "optimistic" (select+score on same full target) as
    the reference. Does the cross-fitted GREEDY oracle delta_query exceed 0 with LCB>0 on either dataset, or
    does it also collapse like prefix? Also sanity-check: is max_rank=10 leaving out directions (rank of cond
    basis)? Report the three numbers (optimistic / xfit_greedy / xfit_prefix). If xfit_greedy also ~0 -> the
    collapse is real optimism removal, verdict holds.` },
  { angle: 'selector-silently-broken', task: `Is nested source-meta returning k*=0 because of a BUG, not
    honest refusal? (a) Run the synthetic anchor test tos_cmi/tests/test_dg_identifiability.py and confirm
    test_balanced_shortcut_is_source_identifiable PASSES (machinery CAN pick k*>=1 and recover). (b) On real
    BNCI2014 cond/full seed0 sub1, relax the no-harm gate by calling nested_source_meta_multi with eps=0.05
    (very loose) and eps=0.2 -- does a k*>0 appear, and if so is its apply_rule_to_target_full delta_query
    POSITIVE or NEGATIVE on the true target? If relaxing eps only surfaces HARMFUL selections, refusal is
    correct. (c) Inspect select_k_star: is the one-SE rule or the Q0.25 gate pathological (always picking 0)?` },
  { angle: 'coverage-gap', task: `Was a candidate basis / objective / hyperparameter that the DG intuition
    demands NOT tried, such that a real ticket is missed? Review the 16-config grid in
    results/.../dg_identifiability_table.csv. Consider: (a) is the CONTESTED restriction (rowspace of W_c,
    only C-1 dims = 3 for 2014 / 1 for 2015) so aggressive it removes the ticket? Compare full-vs-contested
    oracle deltas in the table. (b) would a UNION of bases, or max_rank>10, or a per-direction instability
    SCORE ordering (vs raw SVD order) plausibly help? You may cheaply test ONE concrete alternative on <=4
    folds. Report whether any real coverage gap exists or the grid is adequate for a go/no-go.` },
  { angle: 'power-not-optimism', task: `Is the cross-fit COLLAPSE actually a loss of statistical POWER (the
    T_select/T_query split halves already-small target data -> noisy oracle that can't detect a real small
    gain), rather than removal of optimism? Quantify: how many target trials per fold per dataset (read a raw
    dump via feat_from_tos_dump)? After a 50/50 stratified split, how many query trials remain, and how large
    is the per-split identity-bAcc standard deviation (read crossfit_target_oracle delta_query_per_split)? If
    the gain the oracle would need to detect (~+0.02) is below the split noise floor, the verdict should be
    QUALIFIED as "under-powered existence test", not "no ticket exists". Give the numbers.` },
  { angle: 'cmi-and-firstpass-consistency', task: `Cross-check consistency with the FIRST-PASS result and the
    CMI story. (a) Confirm results/cmi_trace_dg_oracle/dg_oracle_verdict.json indeed shows CMI-only harmful
    (-0.010/-0.063) and source-meta ~0 -- consistent with the hardened refusal. (b) Since hardened k*=0
    everywhere, there is NO selected subset to certify with the posterior-KL CMI ruler -- confirm that's the
    correct reason CMI certification is skipped (not an omission). (c) Does any raw_rows entry show a config
    with meta_k_star>0 AND meta_delta_query>0 with meta beating its random? If a few exist, are they isolated
    noise or a systematic recoverable family? Grep raw_rows.jsonl and report counts.` },
]

phase('Refute')
const findings = await parallel(SKEPTICS.map(s => () =>
  agent(`${CONTEXT}\n\nYOUR ANGLE: ${s.angle}\n${s.task}`,
        { label: `refute:${s.angle}`, phase: 'Refute', schema: FINDING_SCHEMA, effort: 'high' })
))
const valid = findings.filter(Boolean)

phase('Adjudicate')
const ADJ_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['final_verdict', 'confidence', 'surviving_refutations', 'required_qualifications', 'recommended_next_step', 'reasoning'],
  properties: {
    final_verdict: { type: 'string', enum: ['NO_CONFIRMED_TICKET_HOLDS', 'DOWNGRADE_TO_HINDSIGHT_ONLY', 'DOWNGRADE_TO_UNDERPOWERED', 'FLIP_SOURCE_IDENTIFIABLE', 'HOLDS_WITH_QUALIFICATIONS'] },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
    surviving_refutations: { type: 'array', items: { type: 'string' } },
    required_qualifications: { type: 'array', items: { type: 'string' }, description: 'wording the manuscript MUST include' },
    recommended_next_step: { type: 'string', enum: ['write_up_negative_stop', 'stronger_basis_one_more_round', 'target_x_observability_audit', 'build_p2_supermask'] },
    reasoning: { type: 'string' },
  },
}
const adj = await agent(
  `${CONTEXT}\n\nYou are the ADJUDICATOR. Here are the 5 skeptics' structured findings:\n` +
  JSON.stringify(valid, null, 2) +
  `\n\nWeigh them. A refutation only counts if its evidence is concrete (numbers/code), not speculation. Decide
   the final verdict, the qualifications the manuscript MUST carry, and the single recommended next step. Be
   conservative: if the existence oracle (even greedy, cross-fitted) is ~0 AND the selector's refusal is not a
   bug, NO_CONFIRMED_TICKET_HOLDS. If the collapse is power-limited, prefer DOWNGRADE_TO_UNDERPOWERED.`,
  { label: 'adjudicate', phase: 'Adjudicate', schema: ADJ_SCHEMA, effort: 'high' })

return { findings: valid, adjudication: adj }
