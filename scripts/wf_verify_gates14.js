export const meta = {
  name: 'verify-gates14-artifact',
  description: 'Verify whether Gates1-4 STOP_ACTIONABILITY_FAILED is a design artifact (contested vs safety contradiction) or a real target-X-unobservable negative',
  phases: [
    { title: 'Probe', detail: '3 skeptics test the artifact-vs-real diagnosis independently' },
    { title: 'Adjudicate', detail: 'close the selector line vs redesign (soft anchor)' },
  ],
}

const REPO = '/home/infres/yinwang/CMI_AAAI_cmitrace'
const PY = '/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3/bin/python'
const CONTEXT = `
Worktree ${REPO} (branch agent/cmi-trace-targetx-observability). cd there; run python as ${PY} with
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1. Prefer reading existing artifacts; cheap recompute only, say so.

FULL Gates 1-4 real-EEG run (63/63 EEGNet fold-seed cells) selected IDENTITY on ALL 63 folds ->
STOP_ACTIONABILITY_FAILED both datasets. The pre-registered rule says both-datasets-Gate2-fail => close the
target-X selector line. BUT a diagnostic suggests this is a DESIGN ARTIFACT, not a target-X-observability
negative:
 * The PRIMARY basis is the task-CONTESTED cond span = projection of B_cond onto row(W_c) (the task head's
   used directions). By construction, deleting a direction in row(W_c) removes task-discriminative capacity.
 * The selector's source-task-SAFETY gate requires source-LOSO bAcc drop <= 0.02.
 * Diagnostic over 225 informed rank>=1 actions: 0% pass safety (source_task_drop median 0.27), 11% pass
   specificity, 0% pass BOTH. head_overlap_energy ~0.15 (mean discrepancy d~ lives mostly in the head-NULL
   free space, so contested G1 median 0.73 < random Q95 1.88). hindsight_constrained (contested+safe) = 0
   everywhere; hindsight_UNCONSTRAINED (full cond span) mean +0.032, max +0.36.
Hypothesis: hard-contested anchor (U in row(W_c)) INTERSECT source-safety (drop<=0.02) = EMPTY by construction,
so the selector can NEVER pick a non-identity action -> identity is forced, independent of target-X.

Key files: tos_cmi/eval/targetx_observability.py (audit_fold, g1_select, source_task_drop, build_actions),
tos_cmi/eval/targetx_metric.py (whitened_head_rowspace, project_basis, source_whitener), the config
configs/cmi_trace_targetx_observability.yaml, results/cmi_trace_dg_identifiability/targetx_action_rows_full.jsonl
+ targetx_fold_summary_full.jsonl + targetx_gate_verdict_full.json.

Return raw structured data, not prose for a human.
`
const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['angle', 'is_artifact', 'confidence', 'what_i_checked', 'evidence', 'numbers'],
  properties: {
    angle: { type: 'string' },
    is_artifact: { type: 'boolean', description: 'true if identity-everywhere is a design artifact, not a real target-X negative' },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
    what_i_checked: { type: 'string' },
    evidence: { type: 'string' },
    numbers: { type: 'string' },
  },
}
const SKEPTICS = [
  { angle: 'safety-contradiction', task: `Is hard-contested ∩ source-safety EMPTY by construction? Read the
    full action rows; over ALL informed rank>=1 actions report the fraction with source_task_drop<=0.02, and
    whether ANY contested deletion on ANY fold is safe. Then verify the mechanism on 2-3 real folds: load a
    dump, whiten source, build the contested basis, delete one contested singleton, and measure source-LOSO
    bAcc before/after with a fresh head. Is deleting a row(W_c) direction ALWAYS task-harmful (drop>0.02)? If
    yes -> the gate can never pass -> identity is forced -> ARTIFACT.` },
  { angle: 'specificity-and-free-space', task: `Is G1 structurally biased AGAINST contested directions? G1(U)=
    ||U d~||^2 with d~ = A_s(mu_s-mu_tcal). Check: does d~ live mostly in the head-NULL (free) space (so
    contested directions have low G1 and can't beat the ambient-random Q95)? Compute, on 2-3 folds, the
    fraction of ||d~||^2 captured by row(W_c) vs ker(W_c). Relate to head_overlap_energy and the 11% specificity
    pass rate. Does this compound the artifact (even a hypothetically-safe contested deletion would fail
    specificity)?` },
  { angle: 'is-it-actually-real', task: `Argue the STRONGEST case that identity-everywhere is a GENUINE
    target-X-unobservable negative, NOT an artifact. E.g.: maybe the unconstrained hindsight +0.032 is itself
    within-noise / concentrated in a few subjects and there is genuinely no useful deletion anywhere; maybe the
    safety gate is a legitimate deployment requirement so 'no safe useful deletion exists' is a real (not
    artifactual) conclusion. Quantify the unconstrained hindsight ceiling: cluster CI, per-subject sign,
    concentration. Is there a real beneficial deletion ANYWHERE (full/free span) that survives a fair control,
    or not? Decide if 'close the line' is defensible on its own terms.` },
]

phase('Probe')
const findings = (await parallel(SKEPTICS.map(s => () =>
  agent(`${CONTEXT}\n\nYOUR ANGLE: ${s.angle}\n${s.task}`, { label: `probe:${s.angle}`, phase: 'Probe', schema: SCHEMA, effort: 'high' })
))).filter(Boolean)

phase('Adjudicate')
const ADJ = {
  type: 'object', additionalProperties: false,
  required: ['diagnosis', 'confidence', 'is_valid_closure', 'recommended_next', 'required_wording', 'reasoning'],
  properties: {
    diagnosis: { type: 'string', enum: ['DESIGN_ARTIFACT_contested_vs_safety', 'REAL_TARGETX_NEGATIVE', 'MIXED'] },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
    is_valid_closure: { type: 'boolean', description: 'is STOP_ACTIONABILITY_FAILED a valid closure of the selector line?' },
    recommended_next: { type: 'string', enum: ['close_selector_line', 'redesign_soft_anchor', 'redesign_specificity_or_metric', 'other'] },
    required_wording: { type: 'array', items: { type: 'string' } },
    reasoning: { type: 'string' },
  },
}
const adj = await agent(
  `${CONTEXT}\n\nADJUDICATOR. The 3 skeptics' findings:\n${JSON.stringify(findings, null, 2)}\n\nDecide: is the
   full-run STOP_ACTIONABILITY_FAILED a valid closure of the target-X selector line, or a design artifact of
   contested∩safety being empty? If artifact, is the honest next step a SOFT functional anchor (ΔR_source<=δ
   instead of hard U⊆row(W_c)) — which the PM foreshadowed but has NOT approved — or something else? Be
   conservative: only call it a valid closure if a real beneficial deletion demonstrably fails target-X
   identification independent of the safety/anchor contradiction.`,
  { label: 'adjudicate', phase: 'Adjudicate', schema: ADJ, effort: 'high' })

return { findings, adjudication: adj }
