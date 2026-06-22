# ACAR — AMENDMENT v2 (frozen 2026-06-21, supersedes v1 for execution)

`notes/ACAR_FROZEN.md` (v1) is **kept unchanged** for provenance. This file is an explicit, dated amendment made
**after an expert review found the v1 implementation PROTOCOL_INVALID** (in-sample conformal, a G2 criterion that
passes net-harmful routers, incomplete hard guards, a frozen-vs-code monotonicity deviation, weak determinism). The
v1 estimand / feature set / leakage philosophy stand; what changes is the *calibration unit, the guarantee claim,
the G2 safety criterion, the hard guards, and the reproducibility artifacts*. No run under v1 code is adjudicable.

The smoke numbers seen under v1 are **exploratory only**. The full v1 run was killed before writing any summary
(endpoint not unblinded). The binding go/no-go is the v2 run defined here.

---

## A0. Headline guarantee (re-scoped — this is the only coverage claim ACAR may make on current data)

> **Disease-stratified (PD, SCZ computed separately), for an exchangeable new subject / independent
> recording-cluster drawn from the same calibration population, ACAR's joint one-sided upper bound `U_a` has
> ≥ (1−α) finite-sample *marginal* coverage.** The guarantee does **not** extend to new hospitals, new cohorts, or
> cross-cohort distribution shift. Leave-one-cohort-out (LOCO) results are reported as **cohort-level empirical
> robustness only**, never as covered.

**Meaning of "disease-stratified" (audit assertion 1):** stratification is by **task family — PD vs SCZ** — i.e. q
is computed separately within each disease's pooled subjects. It is **NEVER** stratified by the unknown
patient/control (HC vs Patient) target label `y` — doing so would use the label the router is forbidden to see.
Exchangeability is asserted between a new subject and the **disease-task-stratified CAL subjects** only; it is not
claimed across the y label, across cohorts, or across clinics.

Why this is now valid (and v1 was not): the calibration unit is the **subject/recording cluster**, not the cohort.
The relevant `m` is **the number of CAL subjects in each fold** (≈ 50–55 here, since CAL ≈ 0.30·0.80 of ~225–230
pooled subjects), NOT the pooled total — so the split-conformal rank `k = ⌈(m+1)(1−α)⌉` **varies by fold** and is
recorded per fold. With ~50 CAL subjects, `k ≈ 46 ≤ m` → a non-trivial finite-sample quantile; the pooled 230/225
only guarantees enough subjects exist to populate CAL with `k ≤ m`. v1 used 2–3 *cohorts* as units, where `k > m`
forced a meaningless clip to the max residual. The manifest records `n_fit, n_cal, n_eval, k, q` for every
disease/fold; when any fold has `k > m`, that fold's `q = +∞` (uninformative) and is reported as such.

## A1. Calibration protocol (corrects review §1)

Per disease, on the pooled subjects of that disease's cohorts:

1. **Unit = subject cluster.** All sessions/recordings of one subject form ONE calibration unit and contribute
   exactly **one** nonconformity score (in this dump each subject has a single recording, but the code clusters by
   `subject_id` regardless). Batches inherit their subject id.
2. **Subject-disjoint splits.** FIT / CAL / EVAL partitions are disjoint at the **subject** level. ĝ_a is fit on
   FIT batches; q is calibrated on CAL subjects; coverage and G2 are measured on EVAL subjects. Implemented as
   subject-level K-fold (K=5, fold by stable hash of subject id) so every subject is EVAL out-of-fold exactly once;
   within each fold the non-EVAL subjects are split FIT (70%) / CAL (30%) by subject.
3. **Out-of-fold residuals.** A CAL subject's residual uses ĝ_a fit on FIT only (never on that subject). No
   in-sample residuals.
4. **Joint (simultaneous) nonconformity across actions** — deployment selects an action, so a per-action α bound is
   not simultaneous. One score per CAL subject i:
   `s_i = max_{B ∈ subject i}  max_{a ∈ non-identity}  ( ΔR_a(B) − ĝ_a(φ_a(B)) )`.
   `q_{1−α}` = the `⌈(m+1)(1−α)⌉`-th smallest of `{s_i}` over the **m = number of CAL subjects in this fold** (NOT
   the pooled total; k varies by fold); **if the rank > m, q = +∞** (uninformative — reported as such, never
   clipped). Same `q` is added to every action's ĝ_a at deploy: `U_a(B) = ĝ_a(φ_a(B)) + q`. This yields
   simultaneous one-sided coverage over all actions for a new subject.
5. **Disease-stratified.** PD and SCZ get separate q; calibration sets are never mixed.
6. **Manifest** records `n_fit, n_cal (=m), n_eval, k, q` for every disease/fold (+ split subject lists and CAL
   scores), so coverage validity is auditable per fold.

## A2. Deployment API (corrects review §3 — single end-to-end function the guards exercise)

`acar/deploy.py::route_batch(state, routers, z_batch) -> (chosen_action, U_by_action, phi_by_action)`:
- reads ONLY the serialized source state + frozen routers `{a: ĝ_a}` + shared `q` + δ; **no y argument**.
- `len(z) < MIN_BATCH (8)` → **forced `identity`** (label-blind), and the batch is **retained** in the population
  (v1 wrongly deleted it).
- among non-identity actions with `U_a < −δ`, execute argmin `U_a`; else `identity`.
This is the deployed scoring path; the metamorphic guard is applied to it directly.

### A2.1 Split-isolation invariants (frozen; tested in `acar/tests/test_leakage_guard.py::test_split_isolation`)
- **CAL-label changes** MAY change CAL `ΔR`, the per-subject nonconformity scores, `q`, and hence every `U_a` and
  the abstain/adapt decision — but ONLY through the **shared bound shift**: for every EVAL batch `B` and action `a`,
  `U'_a(B) − U_a(B) = q' − q`. They must NOT change FIT models `ĝ_a`, feature orientation, hyperparameters,
  **best-fixed** selection, the FIT/CAL/EVAL splits, or `φ_a`. Any routing change is attributable solely to the
  `q' − q` shift (which moves the `U_a < −δ` eligibility gate; the argmin over actions is unaffected by a shared
  shift). **Edge case `q = +∞`** (uninformative fold): the shift equality is not evaluated (`∞−∞`); instead every
  calibrated non-identity action has `U_a = +∞` and the batch routes to `identity` (`U_identity ≡ 0`).
- **EVAL-label changes** must leave `q`, `φ`, `ĝ`, `U_a`, and the routed action **bit-identical**, changing only
  `ΔR` and label-derived evaluation metrics.

## A3. G1 — signal exists (corrects review §4 orientation/evaluability)

For ≥1 action `a ∈ {matched_coral, spdim, t3a}`, on BOTH PD and SCZ: out-of-fold (subject-disjoint) AUROC of ≥1
paired feature **or** ĝ_a for `harm = 1[ΔR_a(B) > 0]` is **≥ 0.60**, with:
- **orientation fixed on FIT folds only** (a feature's sign is chosen from training subjects, then applied to EVAL —
  no outcome-dependent orientation on the test set);
- **explicit per-cohort evaluability with the denominator FIXED at `n_total` (= 7 cohorts), not `n_evaluable`**
  (review point 1): a feature/action pair must be **evaluable (non-NaN oriented AUROC) in ≥ n_total − 1 cohorts**
  AND have **oriented AUROC > 0.5 in ≥ n_total − 1 cohorts**. Undefined AUROCs count against the denominator, so a
  feature that is NaN in several cohorts can no longer silently pass. All 7 per-cohort AUROCs are reported.

## A4. G2 — control follows (corrects review §2 — now rejects net-harmful routers)

Per disease, on EVAL subjects, the conformal router's mean NLL reduction `red = −mean ΔR` must satisfy **ALL**:
1. `red_router > 0` (beats identity / no-adaptation — the baseline v1 omitted);
2. `red_router > red_bestfixed`, where best-fixed = the single fixed action (identity/matched_coral/spdim/t3a applied
   always) with the highest `red` **selected on FIT subjects ONLY** (never CAL/EVAL — so CAL/EVAL labels cannot leak
   into the baseline ACAR must beat; review/split-isolation); this implies > every fixed action;
3. `red_router > red_random`, where random = matched-coverage random abstention over best-fixed, using the **exact
   expectation** `red_random = p · red_bestfixed` with **p = adaptation coverage = (1 − abstain_rate)** (the
   abstained fraction contributes 0 reduction; review point 3 — `p` here is coverage, not abstention probability);
4. **cohort-macro paired**: router beats identity (mean ΔR_router < 0) in ≥ (n_evaluable − 1) of the disease's
   cohorts, reported per cohort with harmful-batch counts (always vs router);
5. **benefit retention** (oracle, label-only) `≥ 0.50`:
   `retention = Σ_B max(−ΔR_{a_router(B)}(B), 0) / Σ_B max_a(−ΔR_a(B), 0)` ∈ [0,1];
   denominator zero ⇒ `not_evaluable` ⇒ **fail** (v1 let NaN auto-pass; v1's ratio could exceed 1 by mixing
   counterfactual references — fixed here: both num and denom are over the same per-batch action oracle).

## A5. Coverage event (frozen, explicit) + diagnostic

The guaranteed event is, per disease, for an exchangeable new subject `S`:
```
Pr[ ∀ B ∈ 𝓑(S),  ∀ a,  ΔR_a(B) ≤ ĝ_a(φ_a(B)) + q ]  ≥  1 − α
```
where **𝓑(S) is the set of batches generated by the FIXED finite batching protocol** (recording-grouped,
window-ordered, chunked to B=32). This is finite-sample MARGINAL coverage over subjects; it covers exactly that
fixed batch set, and **does NOT cover an unbounded future stream** of new batches from the same subject.
Diagnostic: on EVAL subjects, empirical coverage of this event must be `≥ 1−α` (a sanity gate on the conformal
implementation, not the scientific endpoint).

## A6. Determinism & artifacts (corrects review §5)

- `canonical_hash` is **record-level** (review point 2), not aggregate-only: it digests every record's `φ_a` and
  `ΔR_a`, the per-fold **split assignments** (FIT/CAL/EVAL subject lists), `q`, the CAL subject scores, and every
  EVAL record's `U_a` and **chosen action** (`ĝ_a` enters via `U_a = ĝ_a + q`), plus PD **and** SCZ
  feature/per-cohort/regressor AUROCs, G2 numbers, coverage and the decision. Two runs that route any single batch
  differently produce different hashes even if aggregates coincide.
- double-run re-collects records **with the same guard list semantics**; a guard failure in either run aborts.
- writes **`run_manifest.json`**: dump file SHA-256 hashes, source-state verify residual (serialized vs deployed
  bit-exact), seed map, fold assignment hash, both git SHAs, double-run hash, `pre_registration` = this file.
- output dir `results/acar_gonogo/<git_sha>_<config_hash>/`; **never overwrites** (errors if the dir exists with a
  different config hash); smoke uses a separate `results/acar_gonogo_smoke/` tree.

## A7. Acknowledged post-freeze deviation (review §4 monotonicity)

v1 froze "monotone gradient-boosting"; the code used an **unconstrained** learned regressor. v2 **adopts the
unconstrained regressor deliberately** and records it here as the amendment: the sign of each paired observable vs
adaptation harm is unknown a priori (A0 showed density/CMI are wrong-signed), so imposing monotonicity would inject
an unverified directional prior. Coverage validity does not depend on ĝ_a being monotone or even correct — only on
exchangeability of calibration/eval subjects.

## A8. Stop rule (unchanged from v1)

G1∧G2 → `PROCEED`; G1∧¬G2 → `MEASUREMENT_ONLY` (report the measurement→control gap honestly); ¬G1 → `TERMINATE`.
A failed go/no-go closes the direction — do **not** swap features and retry. `RUN_QUARANTINED / PROTOCOL_INVALID`
(this amendment's reason) is distinct from `TERMINATE`: it means the protocol, not the hypothesis, was broken.
