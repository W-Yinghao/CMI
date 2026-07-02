# C7 — pre-registered K1/K2 decision machinery (acceptance)

Reusable, manifest-driven decision layer built (NOT applied to C6). It becomes native in the C8 multi-seed
run. Machinery commits: `611e5b2` (part A), `97cdead` (part B), `d139f49` (part C). CPU CI: job **879179**,
exit **0**, **708** tests total (28 decision tests: `test_decision_k1` 10 + `test_decision_k2` 6 +
`test_decision_plans` 6 + `test_decision_artifacts` 6).

## K1 — grouped-permutation held-out leakage null

- **statistic** — `grouped_max_probe_extractable_LQ_ov_OACI_minus_ERM`: `T = L_Q^ov,audit(OACI) −
  L_Q^ov,audit(ERM)`, the audit grouped-max-probe extractable-leakage **POINT** estimate (lower ⇒ OACI
  leaks less). Split = held-out **`source_audit`** only. Bootstrap UCLs are companion uncertainty; the null
  targets the point estimate, never the selection-time UCL.
- **permutation scheme** — `paired_swap_within_y_recording_group`: within each `(Y, recording_group)`
  stratum the paired ERM/OACI representations are swapped as a BLOCK (a paired sign-flip). The support
  graph, fold plan and probe config are held FIXED across the null (a permutation only re-labels which
  representation feeds which arm); rows are never shuffled independently, `Y`/`D`/support/folds are never
  changed, and target data is never read.
- **n_permutations** 2000 · **alpha** 0.05 · **seed** 707 (drives ONLY the permutation plan).
- **decision** — `p_lower` = lower-tail permutation p for `Δ < 0`; `p_lower < alpha` ⇒
  `leakage_reduction_detected` (continue to K2), else `stop_no_detectable_heldout_leakage_reduction`.
- **null payload** — the full permutation null is stored in `k1.npz` (`null`, `observed_delta`); `k1.json`
  carries `observed_delta`, `p_lower`, `p_two_sided`, `null_quantiles`, and the identity hashes
  `permutation_plan_hash` (binds seed + n_permutations + strata + bits), `audit_support_hash`,
  `audit_population_hash`, `probe_config_hash`.

**Synthetic controls** (paired features on identical rows; moderate domain signal so `L_Q` is linear in the
leaky-fraction and the observed is the extreme; n_permutations=99, seed=707):
- **positive** (ERM leaky, OACI clean): observed Δ = **−0.5412**, `p_lower` = **0.0100** →
  **`leakage_reduction_detected`**.
- **null** (both clean, nothing to reduce): observed Δ = **+0.0214**, `p_lower` = **0.8800** →
  **`stop_no_detectable_heldout_leakage_reduction`**.

Parallel (`process`, n_jobs>1) permutation null is bit-identical to the sequential loop (tested).

## K2 — reproducible multi-seed worst-domain gain

- **endpoints** `worst_domain_bacc` (↑) + `worst_domain_nll` (↓); per unit `Δ = OACI − ERM`.
- **min_seeds** 3 · **level_policy** `both_levels` (a gain must hold at every (seed, level) unit) ·
  **margins** `bacc_margin` 0.0, `nll_margin` 0.0 · **decision_rule** `stop_if_no_reproducible_gain`.
- **states** `reproducible_gain` (≥1 endpoint reproduces across all seeds/levels) /
  `stop_no_reproducible_gain` / `abstain_insufficient_seeds` / `abstain_missing_endpoint`. All thresholds
  come from the manifest — the decision function hard-codes none.

**Synthetic controls**:
- reproducible bAcc gain (3 seeds × 2 levels, ΔbAcc = +0.02 everywhere, NLL worse) → **`reproducible_gain`**
  (`reproduced_endpoints = [worst_domain_bacc]`).
- reproducible NLL gain (ΔNLL = −0.05 everywhere, bAcc worse) → **`reproducible_gain`**.
- mixed (one (seed,level) unit fails) → **`stop_no_reproducible_gain`**.
- single seed (< min_seeds 3) → **`abstain_insufficient_seeds`**; both endpoints missing →
  **`abstain_missing_endpoint`**. Order-invariant across folds/seeds (tested).

## Artifact integration (additive)

- **paths** — `levels/<level>/decisions/{k1.json, k1.npz, k2.json}`, written THROUGH the writer index
  (`write_artifact_tree_atomic(level_decisions=...)`), so `verify_artifact_tree` verifies them like any
  indexed file — a decision-carrying tree verifies **whole** (tested).
- **verifier** — `verify_decisions(require=False)` tolerates a legacy (decision-less) tree;
  `verify_decisions(require=True)` rejects a decision-less tree and requires a well-formed decision
  (`k1_status`/`k2_status`/`null`) at every level.
- **legacy compatibility** — old artifacts (C6) carry no `decisions/` and still verify legacy-complete;
  `level_decisions` defaults to `None`, so no existing writer caller changes behaviour. `runner.decision`
  `compute_level_decision` is the function the C8 runner will call after audit, before the artifact write;
  the thin call-site wiring lands with C8 (it is not run for existing folds/tests, which would otherwise
  pay a 2000-permutation pass per level).

## C6 handling

**No C6 re-score performed.** The committed C6 descriptive audit null
(`oaci/reports/C6_BNCI001_LOSO_SEED0*`) is unchanged. The K1 permutation null is not stored in the C6
artifacts and, per the C7 decision, was deliberately not backfilled (C6's audit deltas are already an
established descriptive null; the machinery is exercised natively in C8).

## Next

**C8** — BNCI2014-001 MULTI-SEED staged full-bootstrap with native K1/K2 payloads (BNCI2014-001 only; no
BNCI2014-004 until multi-seed aggregation + K1/K2 reporting are validated).
