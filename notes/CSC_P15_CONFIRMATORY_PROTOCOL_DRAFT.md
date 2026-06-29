# CSC — Frozen identifiable-core CONFIRMATORY protocol (DRAFT, definition only)

**THIS IS A DESIGN DOCUMENT, NOT AN AUTHORIZATION.** Nothing here freezes a manifest, runs a
confirmatory sweep, or touches real EEG. Its purpose is to let a reviewer **approve or reject the
confirmatory design before any unseen clusters are generated.** `FREEZE: not approved · CONFIRMATORY
RUN: not approved · P2: not approved.`

---

## 0. One-page reviewer brief

**What is being proposed.** A single, pre-registered confirmatory test of *one decision*: does the
concept-shift certificate, frozen at an exact commit + manifest, control false certification **and**
retain non-zero visible-concept power on a **development-informed identifiable core**, when evaluated
on a **fresh, previously unseen** set of independent synthetic source–target clusters.

**Honesty up front.** The identifiable core is **development-informed** — it was read off the
CSC-P1.5 operating-region map (`notes/CSC_P15_DEVELOPMENT_OPERATING_REGION.md`, artifact `3e5bcf5`),
**not** declared a priori. That is acceptable *only because* the confirmatory evidence uses a locked
manifest and an unseen cluster set. No claim is made that the core was chosen blind.

**Reviewer decision (BOUND for this tag).** The reviewer has accepted the design as a freeze-candidate and
**bound the headline confirmatory core to `K = 1`, `P_baseline` only**, with: `G = 66` generated clusters,
`N_valid_min = 59`, `source_invalid_cap = 0.10`, `max_forbidden_failures = 0`, **power bar `CP_lower ≥
0.50`**, `base_seed = 900000`, **pointwise** claim. `P_strong` is **secondary descriptive only** (never in
the PASS/FAIL). This document is still **definition-only**: a tag + audit package follows; the unseen-cluster
run needs a *separate* authorization.

**Design choice that closes the obvious loopholes (see §3).** The core is **not** a predicate over a
region (which would admit weak in-region cells and undeclared gap zones); it is a **fixed finite list
of exact operating points** declared in the tag. Confirmatory clusters are *generated at* those exact
points, so every cluster is in-core **by construction** — there is nothing to reclassify post hoc.
The headline claim requires the **CONJUNCTION** (every declared point passes both endpoints); a single
passing point may **never** carry the claim, and **no point may be added after the tag**.

**Endpoints (exact, from Clopper–Pearson; see §5–§6). The headline at each point is the CONJUNCTION of
both** — "controls error" may not be claimed from a point that did not also clear power (otherwise
abstain-on-everything would masquerade as control).
- **Primary — false-certification control.** Cluster-level `any_forbidden_full_suite` over `N` unseen
  independent clusters at the point. **max-failures = 0**: PASS iff `0/N` forbidden, which gives CP
  **upper** bound ≤ α = 0.05 at **N = 59** (0/59 → CP-UB 0.0495). **Any** in-core forbidden event is
  an automatic FAIL. Denominator = independent source–target clusters, never correlated targets.
- **Power — usefulness at the point.** Visible-concept power. PASS iff `n_fired ≥ max(min_fired(N_valid),
  min_fired(G))` for the **one** pre-registered bar **0.50** — i.e. BOTH the conditional (`n_fired/N_valid`)
  AND the unconditional (`n_fired/G`) CP **lower** bound clear 0.50, so the source-invalid exclusion can
  never inflate the headline (§6). With G=66 the effective threshold is **≥ 41 fired**. Conditional AND
  unconditional power + both CP intervals are reported regardless of PASS/FAIL.

**Why it is credible.** P1.5 showed, on DEVELOPMENT seeds, 21/24 cells with 0/12 forbidden,
`false_concept_on_synthetic_null = 0` in all 24 cells, and a core with power 0.75–0.92 (CP-LB
0.47–0.66). But 12 clusters/cell only bounds the forbidden rate at ≤ 0.221 — **not** error control.
This protocol replaces that with an unseen, adequately-powered, pre-registered test.

---

## 1. Frozen method

At tag time, lock and record (no change after the tag):

- **exact code commit, bound to an annotated git TAG** `refs/tags/csc-confirmatory-v1` (recorded in
  the tag as `expected_code_ref`). The confirmatory uses the same audited `execute_protocol` path, the
  parallel harness, and `verify_canary_ref` gating. **Frozen-code provenance guard (pre-run hotfix):**
  `run_confirmatory.py --execute` and the SLURM wrapper BOTH fail closed unless `HEAD ==
  git rev-parse csc-confirmatory-v1^{commit}` AND the working tree is clean — so the run cannot execute
  from the wrong branch/worktree (the branch tip carries the later audit-artifact commit, so the run is
  launched from a DETACHED worktree AT THE TAG). The result payload records `git_head /
  expected_code_ref / expected_code_commit / git_status_clean`.
- **protocol manifest hash** `da2c0f4309847a4e790843b9ece68010a90c33bdb9404097aee72dcbefbb2632`
  (or its successor, recorded verbatim) — this fixes **all** thresholds: `tau_margin`,
  `cov_loading_margin_kappa`, `consensus`, `alpha`, `tau_*` calibration rule, `n_boot`, `n_dir_boot`,
  `target_n_boot`, `tau_n_pseudotargets`, `concept_eigengap_min`, `concept_stability_max_deg`,
  `invalid_null_frac_max`, `label_unit`, `group_aware`, `rng_algorithm`.
- **named stage-seed derivation** (sha256 chain) — unchanged.
- **evaluation knobs not in the manifest, pinned here as tag fields:** the fixed core-point list (§3);
  the unseen `base_seed` (§2); the per-point **generated count `G`** and **minimum valid count
  `N_valid_min = 59`** (§4–§5); the **source-invalid INCONCLUSIVE cap `0.10`** (§4); the **power bar
  `0.50`** (§6). None may change after the tag.
- **lineage cross-references (recorded in the tag too):** `audit_baseline_commit = 4ea423d` (the
  P1.4.5a DEV-AUDIT this builds on) and `dev_artifact_commit = 3e5bcf5` (the P1.5 difficulty-envelope
  artifact the core was read from).
- a **fail-closed audit** (P1.4.5a-style) of the tagged commit recorded alongside.

No knob may be re-tuned for the confirmatory run. If any method change is needed, the tag is void and
the design returns to review.

## 2. Confirmatory target population

- **Synthetic identifiable-core clusters ONLY.** Same `execute_protocol` certifier path as P1.5. No
  real EEG (that is P2, separately gated). No threshold tuning.
- **Fresh, previously UNSEEN seed set**, disjoint from every development set used so far: NOT the
  0–9 audit smoke seeds, NOT the 500000 canary base, NOT the 600000 full-sweep base. Pre-register a
  new range (proposed `base_seed = 900000`), recorded in the tag.
- Each cluster = one fresh source seed + one target per kind from that source's geometry (the
  audited independence convention). Targets are single-condition (the mandatory `tgt_condition_ids`
  contract).
- **Seed derivation (recorded machine-readably in the result artifact, `seed_derivation`).** Source
  seeds are `base_seed + k` for `k ∈ 0..G−1`; each cluster's target seed is `target_seed_base +
  source_seed` (`target_seed_base = 900000`). With `base_seed = 900000, G = 66`: **sources
  `900000..900065`, targets `1800000..1800065`** — disjoint streams (`source_target_seed_streams_
  disjoint = true`). The runner exposes this via `seed_streams(tag)` in both the dry-run and the
  `--execute` payload.

## 3. Development-informed identifiable core — a FIXED, FINITE list of operating points

The core is **NOT a predicate over a region.** A region predicate is gameable two ways the adversarial
review confirmed: (i) it admits weak in-region cells where the forbidden endpoint is *vacuous* because
the certifier abstains on ~everything (silence ≠ control); (ii) its complement leaves *gap zones*
(configs neither in nor out) that invite post-hoc reclassification. Both are closed by making the core
a **fixed finite list of EXACT operating points**, declared in the tag, with clusters **generated at**
those points. Membership is therefore **by construction** (the generative config is fixed at
seed-generation time, never re-derived from an outcome); there are no ranges and no gaps to adjudicate.

**Pre-registered core — REVIEWER-BOUND to `K = 1`, `P_baseline` only (the headline confirmatory core):**

| point | role | config (all other knobs = P1.5 baseline) | DEV power (CP-LB) |
|---|---|---|---|
| `P_baseline` | **PRIMARY (headline, K=1)** | effect 14, subj/dom 22, target_subj 30, prior_alpha 4.0, corr 0.2, leakage 10, concept_domains 3, epochs_max 22, mechanism_family 0, single-condition target | 0.83 (0.56) |
| `P_strong` | **SECONDARY descriptive only — NOT in the PASS/FAIL** | as baseline but `concept_effect_size = 20` | 0.92 (0.66) |

The reviewer has **bound the headline confirmatory core to `K = 1` (`P_baseline` only)**. `P_baseline` is
the more informative favourable operating point; `P_strong` merely confirms an easier higher-effect case
and would raise the conjunction burden without being necessary for the first confirmatory claim. `P_strong`
MAY be run and reported as a **secondary descriptive stress/sanity point**, but it is **NOT** part of the
primary PASS/FAIL. Promoting it to the headline (K = 2) requires a **new protocol version that chooses
K = 2 before tagging** — and then a *simultaneous* familywise claim must use the Bonferroni `N` (§5), not 59.

Every listed knob is pinned to a value the P1.5 map showed retains power **and** had `0/12` forbidden
(strong cells only — `concept_domains` is pinned to **3**, not "≥3": the `=5` cell had power 0.33;
`target_subjects` is pinned to **30**, not "≥20": the `=20` cell had power 0.50, below the bar;
`epochs_max` is pinned to **22**, closing the one-tick gap above the known-bad `=12`;
`mechanism_family` is pinned to the validated `0`). **The named core is development-informed and frozen
before confirmatory evaluation; the confirmatory clusters are unseen and are not pooled with the
DEVELOPMENT seeds.** This is disclosed honestly — no claim is made that the core was chosen without
seeing the P1.5 map; it is acceptable only because the evidence is on a locked manifest + an unseen
cluster set (§0).

**Decision rule over the list (closes the best-of-K loophole):** the headline claim requires the
**CONJUNCTION** — *every* declared point independently passes *both* endpoints (§5 ∧ §6). A single
passing point may **never** carry the claim; a **disjunction** ("whichever point passed") is forbidden;
results are **never pooled** across points; **no point may be added after the tag.** Requiring all
points to pass is strictly more stringent than any one point, so the joint PASS criterion does not
inflate the false-PASS rate. (Whether the per-point CP bounds are reported **pointwise** or as a
**simultaneous familywise** statement is a separate matter, declared per §5: at N = 59 the headline is
a conjunction of *pointwise* checks; a simultaneous core claim needs the Bonferroni `N` in §5.) Each
point gets its own `G` generated unseen clusters with valid `N ≥ 59` (§4).

## 4. Runtime abstention handling (no region complement to adjudicate)

Because clusters are generated *at* the fixed in-core points, there is **no out-of-core region and no
gap zone** — §3's old region complement is deleted. The only runtime exclusion is the certifier's own
fail-closed refusal: a generated cluster whose `source_status != VALID` (INVALID_SUPPORT /
INVALID_RESIDUAL_NULL / INVALID_GEOMETRY_NULL / UNASSESSED_CONCEPT_ATTRIBUTION /
UNSTABLE_CONCEPT_ATTRIBUTION) — the certifier never even rendered a target certificate.

To keep the count rules consistent with "fixed N / no optional stopping", BOTH counts are pinned in the
tag (a generated count `G` and a minimum valid count `N_valid_min = 59`):

- Generate exactly `G` clusters at the point (`G` fixed in the tag; **never** generate-until-K-valid —
  that would be the banned optional stopping). Proposed `G = 66` (headroom: even at the 10% cap the
  valid count stays ≥ 59).
- A cluster is **evaluable** iff `source_status == VALID`. Let `N_valid` = #evaluable. Source-invalid
  clusters are **recorded** (count + reason) and **excluded from the endpoint numerators/denominators**
  (they are neither a power success nor a forbidden failure).
- **INCONCLUSIVE (not PASS)** if EITHER `N_valid < N_valid_min = 59` OR the source-invalid fraction
  `(G − N_valid)/G > 0.10`. This (i) removes the N-contradiction — endpoints are always computed on the
  realized `N_valid ≥ 59`, never on a silently-shrunk N; and (ii) stops "the certifier abstained on
  everything" from masquerading as control.

Membership and evaluability are read from the **generative config / runtime `source_status`**, fixed at
generation time — never re-derived after seeing the certificate outcome. (P1.5 core points had
`source_invalid_rate = 0.000`, so in practice `G = 66` yields `N_valid = 66`.)

## 5. Primary endpoint — false-certification control (per point)

- Statistic: per cluster, `any_forbidden_full_suite` (any target kind certified into its forbidden set,
  per the audited `FORBIDDEN` map). One Bernoulli per cluster.
- Denominator: the `N_valid` evaluable clusters at the point (§4: generate fixed `G`, exclude
  source-invalid, require `N_valid ≥ 59` else INCONCLUSIVE). The CP bound is computed on the realized
  `N_valid`.
- **max-failures = 0 (headline).** PASS iff **`0/N_valid`** forbidden with `N_valid ≥ 59`, giving a
  one-sided 95% CP upper bound ≤ α = 0.05 (0/59 → CP-UB 0.0495). **Any** in-core forbidden event is an
  automatic FAIL. `G` and `N_valid_min = 59` are **pre-committed in the tag**; no optional stopping / no
  generate-until-valid / no increasing `G` after the run starts.
  - For reference: 1/59 → CP-UB ≈ **0.078** (would FAIL); tolerating 1 failure would need N ≈ 93. A
    `≥1-failure` variant is **NOT** the primary claim — if ever reported it is a separate, clearly
    labelled lower-tier secondary, never the headline.
- **Always report; never censor by power (non-vacuity guard, stated safely).** For **every** named
  core point the forbidden-certificate count + CP **upper** bound AND the visible-concept power + CP
  **lower** bound are **always reported**. A point's headline PASS requires **both** endpoints to pass;
  failure of the power endpoint makes the point — and hence the core — **FAIL or INCONCLUSIVE**, but it
  does **NOT** censor, reclassify, or remove the false-certification result from that point's
  count/denominator. (This is the silence-as-control guard expressed correctly: "the certificate
  *controls error*" is asserted for the core only when power also passes, yet the forbidden number
  itself is always on the record, PASS or FAIL.)
- **Pointwise vs simultaneous across the `K` named points (declare in the tag).**
  - **Option A — pointwise (supported at N = 59).** Each point's `0/59` gives a **pointwise** one-sided
    95% CP upper bound ≈ 0.0495. The headline is then the **conjunction of `K` pointwise confirmatory
    checks**, NOT a single simultaneous familywise 95% statement over the whole core.
  - **Option B — simultaneous familywise.** For a simultaneous 95% statement over all `K` points,
    Bonferroni each point at α/K: zero-failure needs `N ≥ ⌈log(0.05/K)/log(0.95)⌉` →
    K=1: 59 · K=2: 72 · K=3: 80 · K=4: 86 · K=5: 90 per point.
  - **This tag is bound to K = 1 (`P_baseline` only, §3), so the headline is the single-point pointwise
    Option A: `0/N_valid` with `N_valid ≥ 59` → CP-UB ≈ 0.0495.** (Option B / Bonferroni applies only to
    a future K ≥ 2 protocol version; `P_strong`, if run, is secondary descriptive and not in the headline.)
- Reported but not gating: `any_false_positive_must_abstain` and `false_concept_on_synthetic_null`
  separately (P1.5 had the latter = 0 in all cells — the asymmetry worth re-confirming).

## 6. Power endpoint — usefulness at the point

- Statistic: per cluster, the boundary_coupled (visible-concept) target fired `CONCEPT_SUSPECT`. One
  Bernoulli per cluster.
- Statistic numerator: `n_fired` = visible-concept fires among the `N_valid` evaluable clusters
  (source-invalid clusters count as non-fires by construction).
- **PASS iff `n_fired ≥ min_fired_for_pass`, where (reviewer fix — source-invalid exclusion cannot
  inflate the headline):**

  ```
  min_fired_for_pass = max( min_fired(N_valid) , min_fired(G) )
  min_fired(n) = smallest k with CP_lower(k, n) >= power_bar (0.50)      # REALIZED n, never hard-coded
  ```

  This requires BOTH the **conditional** CP-lower (`n_fired/N_valid`) AND the **unconditional**
  CP-lower (`n_fired/G`) to clear 0.50. Since `G ≥ N_valid`, the max equals `min_fired(G)`. A high
  conditional power on few valid clusters is therefore **not** enough.

  | realized | `min_fired(N_valid)` (cond) | `min_fired(G)`, G=66 (uncond) | **effective `min_fired_for_pass`** |
  |---|---|---|---|
  | N_valid 66 | 41 | 41 | **41** |
  | N_valid 60 | 37 | 41 | **41** |
  | N_valid 59 | 37 | 41 | **41** |
  | N_valid 72 (G=72) | 44 | 44 | **44** |

  Worked counter-example (the loophole this closes): `37 fired / 60 valid / G=66` → conditional 37/60
  clears 0.50 (CP-LB ≥ 0.50) but unconditional 37/66 does not (needs 41) → **FAIL**.
- **Report power BOTH ways, always:** *conditional* = `n_fired/N_valid` (+ CP-lower) and *unconditional*
  = `n_fired/G` (+ CP-lower). Both are recorded regardless of PASS/FAIL, so the source-invalid exclusion
  is always visible and can never silently lift the headline. (At the proposed points `source_invalid =
  0`, so the two coincide and the effective threshold is `min_fired(66) = 41`.)
- **Bar justification is external, not read off the DEV map.** Proposed bar **0.50** on a *screening*
  rationale: a deployment screen that detects identifiable concept shift in **fewer than half** of
  genuine cases at its own favourable operating point is not actionable. The bar is **one** value (not
  a menu — a menu is a hidden degree of freedom), fixed in the tag, justified before the run on
  deployment need rather than on where the development CP-LBs happen to fall.
- **Acknowledged circularity (minor):** 0.50 sits *below* both core points' DEV power CP-LBs
  (`P_baseline` 0.562, `P_strong` 0.661), so the development evidence cannot *exclude* either point at
  this bar. The bar is chosen on deployment need (a screen firing in < 50% is not actionable), NOT to
  guarantee the development-informed core passes; it is also below the maximally self-serving ~0.55.
- The achieved power **point estimate and full CP interval are reported regardless** of PASS/FAIL, so
  the bar is never the only summary.

## 7. Non-selection rules (binding)

- **Fixed point list.** The exact set of core points (§3) is named in the tag; **no point may be added,
  removed, or moved after the tag.** The headline is the **CONJUNCTION** over the list — **no
  disjunction**, no "report whichever point passed."
- **Fixed `G`, no optional stopping.** The generated count `G` and `N_valid_min = 59` per point are
  pre-committed in the tag; `G` may **not** be increased after the run starts, there is **no
  generate-until-valid**, and **max-failures = 0** is the headline forbidden rule. If `N_valid < 59`
  the point is INCONCLUSIVE (never re-evaluated at a smaller N).
- **No dropping clusters** from the denominator after seeing outcomes. The only exclusion is the §4
  `source_status != VALID` runtime refusal, evaluated on the certifier's own status (not the
  certificate outcome) and **capped at 10%** (else INCONCLUSIVE); the cap value (0.10) is a pinned tag
  field (§1).
- **No grid reshaping** / no off-grid configs (clusters are generated only at the declared points).
- **No changing** `tau_margin`, `cov_loading_margin_kappa`, `consensus`, or any manifest value.
- **No pooling** with DEVELOPMENT seeds (0–9 / 500000 / 600000) or across distinct core points.
- **A failure inside a declared point is a real failure** — there is no out-of-core region to
  reclassify it into (§4 deletes the region complement).
- **Single power bar**, deployment-justified, fixed in the tag; the achieved power point estimate +
  full CP interval are reported regardless.
- The full per-cluster outcome list + per-point `(N, forbidden, fired, source_invalid)` counts are
  recorded verbatim in the confirmatory artifact, regardless of PASS/FAIL.
- **A scientific FAIL is a valid result, not a missing artifact.** The runner exits `1` on a FAIL
  (after writing the JSON); the actual-run wrapper (`csc/run_confirmatory.sbatch`) ALWAYS preserves the
  JSON artifact + sha256 (and the log + sha256) and then exits with the **scientific** RC, and exits
  with an **infrastructure** code (2) only if NO JSON artifact was produced.

## 8. Failure decomposition (reported every run, PASS or FAIL)

For every non-firing in-core visible cluster and every forbidden event, record the binding reason
(same vocabulary as P1.5 / the audited `_concept_failure_reason`):

```
INVALID_SUPPORT  /  support invalid
INVALID_RESIDUAL_NULL  /  INVALID_GEOMETRY_NULL
UNASSESSED_CONCEPT_ATTRIBUTION  /  UNSTABLE_CONCEPT_ATTRIBUTION
geometric_maxstat_not_sig          (no estimable concept atlas)
residual_T_not_sig                 (cross-fitted decoder gate)
not_dominant_or_robust_consensus_abstain
source invalid rate / support invalid rate
```

This makes a FAIL diagnosable (which gate bound, or which boundary leaked) rather than a bare number.

---

## Status

```
CSC-P1.5 DEVELOPMENT report: accepted for descriptive review
This document: confirmatory protocol DRAFT (definition only)
Needs: reviewer approval of §1-§7 BEFORE any tag/run
FREEZE: not approved
CONFIRMATORY RUN: not approved
P2 real EEG: not approved
```

On approval, the next (separately authorized) steps are: tag the frozen commit + record the
P1.4.5a-style audit → generate the unseen core cluster set at the pre-registered base_seed → run the
two endpoints → record the artifact (PASS/FAIL + full decomposition) → only then consider P2.
