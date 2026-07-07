# Step 15 — Harm-Control Policy Results

## Question

Given the Step-14 finding that real minimal-label harm-sign calls are **high-precision but
low-coverage**, can a policy use k target labels to reduce offline-TTA adaptation harm — without
claiming R1 target-gain identifiability?

## Regime

- **k = 0** — R1 target-unlabeled: label-based policies must **abstain** (non-identifiable).
- **k > 0** — R2 labeled slice under an iid sampling / coverage contract.

## Policy families (see `06_coverage_aware_harm_control.md`)

`always_identity` (P0) · `always_adapt` (P1) · `plugin_sign` (P2) · `ci_adapt_only_abstain` (P3a) ·
`ci_adapt_only_identity` (P3b) · `ci_three_way` (P4) · `oracle_full_target` (P5, evaluation-only,
never deployable).

## Predeclared best-deployable rule

Among deployable cells with `adaptation_coverage > 0`: keep `harm_rate_among_adapt_decisions <= 0.05`,
maximize `adaptation_coverage`, tie-break minimize `missed_benefit_rate`. Oracle never eligible.

## Reading (numbers in the tracked digest)

Results live in `results_summaries/step15_harm_control_summary.{json,md}` and the Step-15 dashboard.
The expected qualitative picture, consistent with Step 14:

1. Strict CI policies **reduce harm-among-adapt below the constraint** but adapt only a small fraction
   of cells (low coverage; heavy abstention).
2. Identity-default policies **prevent adaptation harm** but **miss benefit**.
3. Plugin (point-estimate) policies **over-adapt** and can harm at small k.
4. The oracle policy is an evaluation upper bound, **not deployable**.

The honest takeaway is a **coverage/control tradeoff**: minimal labels can make adaptation *safe* only
by *abstaining* on most cells — harm is controllable but not freely; it is not a free accuracy gain.

## Claim boundary

Policy decisions use k target labels under an iid sampling contract. They are NOT source-only or
target-unlabeled adaptation claims, NOT R1 target-gain identification, and NOT a SOTA claim. Oracle
full-target labels are used only for evaluation and the forbidden upper-bound policy.
