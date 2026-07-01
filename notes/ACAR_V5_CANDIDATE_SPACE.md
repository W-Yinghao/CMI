# ACAR V5 — Candidate / Policy Space **(DRAFT — UNTAGGED — NON-BINDING; pinned at sign-off)**

Pre-registers the ENTIRE policy search space for V5 Stage-2 DEV selection. Companion to `ACAR_FROZEN_v5.md`,
`ACAR_V5_ENDPOINTS.md`, `ACAR_V5_SPLITS.md`. **No runs authorized by this draft** (see the hard no-execution clause).

## 0. Principles (the v4 corrections)
- **Bounded, interpretable, pre-registered (PINNED — Step 2b).** v4's 14/90 pass had selection-bias risk. V5's policy space is
  **EXACTLY the five families P1–P5** (no P6–P8 unless a NEW dated amendment is committed BEFORE any run), with a **hard total
  budget of ≤ 24 (family, grid, disease-routing-spec) configurations across BOTH diseases**. No exploratory sweep, no
  "find something pretty then post-hoc justify".
- **No single signed-score sensitivity.** v4 hinged on one signed `d_margin` direction, which flipped under substrate
  regeneration. In V5, `d_margin` (and any signed score) may CONTRIBUTE to a benefit term but may **never alone decide an action**:
  every adapt decision must additionally pass an **adaptation-violence harm veto** built from `flip_rate` + `JS(p0,pa)` (+ optionally
  `entropy`), so a high-disruption adaptation is refused regardless of the benefit score.
- **v4 candidate = NEGATIVE prior.** `shift_margin + benefit_ranked + harm_indicator` is NOT a default and its thresholds are NOT
  reused; it may appear at most as one labeled comparison point, flagged as the v4 negative prior.
- **Actions (z-space, GPU-free), pinned:** `identity` (no-op) · `matched_coral` · `spdim` · `t3a`. (raw-EA / CITA deferred; if ever
  added it is a dated amendment.) `identity` is always available; a router output of `identity` = abstain.
- **Label-free features (paired pre→post), pinned:** `d_entropy`, `d_margin`, `flip_rate`, `JS(p0,pa)`, `Bures`, `post_sep`,
  `n_eff`. Labels are NEVER read by any routing function (enforced by `test_no_label_in_route`).

## 1. Policy families (EXACTLY P1–P5 — pinned; no additions without a dated pre-run amendment)
Each family is an **abstaining router** `π(B) ∈ {identity, matched_coral, spdim, t3a}` with a small grid of pre-registered
thresholds. All thresholds are on FIT-standardized features (subject-balanced standardization; see `ACAR_V5_SPLITS.md`).

Notation: `Qτ[x]` = the FIT-only τ-quantile of the unlabeled feature `x` over FIT batches (τ ∈ {q50,q60,q70,q80,q85,q90}),
recomputed per disease by the frozen algorithm; the **harm veto at level v** = adapt only if `flip_rate ≤ Qv[flip_rate]` AND
`JS ≤ Qv[JS]`. Concrete per-config values are enumerated EXACTLY in §1.6 (the family bullets fix only the RULE FORM).

- **P1 — Benefit score + harm veto.** Rank actions by benefit `= d_margin` of the post-vs-pre adaptation; adapt the best action
  `a*` iff `d_margin(a*) ≥ Qb[d_margin]` AND harm-veto(v). Free params: `(b, v)` (both from the allowed grid).
- **P2 — Low-violence confidence gate.** Adapt the best-confidence action iff `d_margin ≥ Qc[d_margin]` AND `d_entropy ≤ 0`
  (entropy non-increasing) AND harm-veto(v). Free params: `(c, v)`. Emphasizes "improve confidence without disruptive relabeling".
- **P3 — Best-fixed action with abstention.** No per-batch ranking; execute ONE fixed action `a ∈ {matched_coral, spdim, t3a}`
  only when harm-veto(v) clears. Free params: `(a, v)`. The most conservative family; the G5 best-fixed comparator pool.
- **P4 — Action-agreement gate.** Adapt only when ≥ k of three benefit-score variants (`d_margin`-ranked best · `post_sep`-ranked
  best · `JS`-min action) agree on the SAME action AND harm-veto(v). Free params: `(k, v)`. **P4 admissibility (PINNED — Step 2b):**
  the manifest (§1.6) uses ONLY the benefit-score-variant agreement. A "seed-agreement" variant is admissible ONLY as a FROZEN
  3-substrate ensemble (all three seed encoders trained/hashed/registered BEFORE selection; SAME ensemble for DEV selection + G6 +
  external; external runs all three and abstains unless the rule fires) — and, since that ensemble is not in this manifest, adding
  it requires a NEW dated pre-run amendment. Stage-4 stress results (S1–S3) may NOT create or alter the P4 rule.
- **P5 — Conservative direct-selective gate.** A v4-style directly-calibrated selective policy on the selective score
  `s = d_margin(a*)` of the benefit-ranked best action: adapt iff `s ≥ Qλ[s]`, with harm-veto FIXED at v=q90, and the operating
  point λ admissible ONLY where the CAL certification already meets G4 (`UCB[harm_among_adapted] ≤ 0.30`) and G1
  (`LCB[coverage] ≥ 0.15`) — i.e. the conditional-harm cap + coverage floor are baked into calibration, not just `L_harm_all`.
  Free param: `λ` (6 points). Included so the v4 failure mode is tested head-on, NOT as a favored default.

## 1.6 EXACT V5 candidate manifest (PINNED — Step 2c; total = 22 ≤ 24)
The complete Stage-2 candidate set is exactly the rows below. **No implementation may add, remove, reorder, or reinterpret a
candidate ID.** Every candidate has `disease_scope = both` (the SAME rule is evaluated on PD and SCZ separately; the frozen
algorithm recomputes the per-disease FIT-only quantiles), `operating_grid_source = FIT-only unlabeled quantiles {q50…q90}`, and
harm-veto(v) `= flip_rate ≤ Qv[flip_rate] AND JS ≤ Qv[JS]`. `comparator_role = candidate` unless noted.

```
P1 (benefit d_margin + harm veto):  adapt a*=argmax_a d_margin(a) iff d_margin(a*) ≥ Qb[d_margin] AND harm-veto(v)
  V5-P1-001  b=q60  v=q80
  V5-P1-002  b=q60  v=q90
  V5-P1-003  b=q80  v=q80
  V5-P1-004  b=q80  v=q90
P2 (low-violence confidence):  adapt best-confidence a iff d_margin ≥ Qc[d_margin] AND d_entropy ≤ 0 AND harm-veto(v)
  V5-P2-001  c=q60  v=q80
  V5-P2-002  c=q60  v=q90
  V5-P2-003  c=q80  v=q80
  V5-P2-004  c=q80  v=q90
P3 (best-fixed action + abstention):  adapt fixed action a iff harm-veto(v)   [comparator_role = candidate + G5 best-fixed pool]
  V5-P3-001  a=matched_coral  v=q80
  V5-P3-002  a=matched_coral  v=q90
  V5-P3-003  a=spdim          v=q80
  V5-P3-004  a=spdim          v=q90
  V5-P3-005  a=t3a            v=q80
  V5-P3-006  a=t3a            v=q90
P4 (benefit-score-variant agreement):  adapt agreed action iff ≥k of {d_margin-best, post_sep-best, JS-min} agree AND harm-veto(v)
  V5-P4-001  k=2of3  v=q90
  V5-P4-002  k=3of3  v=q90
P5 (conservative direct-selective; G4+G1 baked into CAL):  adapt a* iff d_margin(a*) ≥ Qλ[d_margin], harm-veto v=q90
  V5-P5-001  λ=q50   V5-P5-002  λ=q60   V5-P5-003  λ=q70
  V5-P5-004  λ=q80   V5-P5-005  λ=q85   V5-P5-006  λ=q90
```
Any change to this table (add/remove/retune a row, add the P4 seed-ensemble variant, add a family) is a NEW dated pre-run
amendment committed BEFORE any DEV run — never a post-hoc edit.

## 1.7 Action-record scalarization & quantile universe (PINNED — Step 2e; makes the 22 rows bit-executable)
```
Allowed non-identity actions:  A = {matched_coral, spdim, t3a}
Action tie-break order:        matched_coral ≺ spdim ≺ t3a   (used for EVERY argmax / argmin / agreement tie)

Per disease × fold × substrate × candidate_id, for every FIT batch B and every a ∈ A compute the action-indexed label-free
features: d_entropy_a(B), d_margin_a(B), flip_rate_a(B), JS_a(B), Bures_a(B), post_sep_a(B), n_eff_a(B).

Candidate proposed action a*(B):
  P1: a*(B) = argmax_a d_margin_a(B)                          (tie → action order)
  P2: a*(B) = argmax_a d_margin_a(B)                          (tie → action order)
      "best-confidence" ≡ EXACTLY this benefit-ranked action PLUS the extra veto d_entropy_{a*} ≤ 0
      (NO post-margin / min-entropy alternative selector is allowed)
  P3: a*(B) = the fixed action named by the candidate row
  P4: margin-best = argmax_a d_margin_a(B); post_sep-best = argmax_a post_sep_a(B); JS-min = argmin_a JS_a(B) (ties → action
      order). If the k-of-3 agreement rule yields no agreed action, B has NO proposed-action record.
  P5: a*(B) = argmax_a d_margin_a(B)                          (tie → action order)

Harm veto at level v (all families):  flip_rate_{a*}(B) ≤ Qv[flip_rate]  AND  JS_{a*}(B) ≤ Qv[JS]

FIT-only quantiles Qτ[x] (τ ∈ {q50,q60,q70,q80,q85,q90}):
  computed over THIS candidate_id's FIT proposed-action records ONLY, using the scalar x_{a*}(B) actually used by the rule
  (e.g. Qb[d_margin] over {d_margin_{a*}(B)}; Qv[flip_rate] over {flip_rate_{a*}(B)}; Qv[JS] over {JS_{a*}(B)};
  P5's Qλ[d_margin] over {d_margin_{a*}(B)}). NO CAL/EVAL/external record may contribute.
  If a candidate_id has ZERO FIT proposed-action records, it is NON-EVALUABLE and FAILS.

At evaluation: thresholds are FIXED from FIT; the SAME scalarization + a*(B) rule is applied unchanged to CAL/EVAL/S1/S2/S3/external.
```

## 2. Calibration knob (shared) — PINNED (Step 2b)
Where a family uses a finite λ / threshold grid (P5, and any thresholded family expressed as a sweep), the operating grid is
**≤ 6 operating points per λ-based family**, and every grid point is generated from **FIT-only unlabeled score quantiles**:
`{q50, q60, q70, q80, q85, q90}`. **No CAL/EVAL labels and no external data may define or expand the grid.** λ/threshold selection
is by the Stage-3 constrained criterion + the LTT/Holm certification of H1–H3 (`ACAR_V5_ENDPOINTS.md`), NOT by maximizing utility
alone.

## 3. Selection rule (Stage-2)
Among all pre-registered (family, grid) configs that are G1–G5 **eligible** on DEV-OOF, select by the constrained objective in
`ACAR_V5_ENDPOINTS.md` §Selection (maximize the min-disease utility margin subject to the safety/coverage constraints), with a
deterministic, pre-registered tie-break (prefer lower conditional adapted-harm, then higher coverage, then the more conservative
family P3 ≺ P1/P2/P4 ≺ P5). The selected config is FIXED before Stage-4 robustness and Stage-5 external; **no reselection** after
seeing robustness/external (enforced by `test_fixed_candidate_no_reselection`).

## 4. Explicitly excluded
Open multi-dozen-config sweeps; any policy that decides on a single signed score without the harm veto; reuse of the v4
candidate's thresholds as defaults; per-batch label peeking; adding actions/features/sites after seeing any result (those require a
new dated amendment).
