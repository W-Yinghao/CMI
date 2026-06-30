# ACAR V5 — Candidate / Policy Space **(DRAFT — UNTAGGED — NON-BINDING; pinned at sign-off)**

Pre-registers the ENTIRE policy search space for V5 Stage-2 DEV selection. Companion to `ACAR_FROZEN_v5.md`,
`ACAR_V5_ENDPOINTS.md`, `ACAR_V5_SPLITS.md`. **No runs authorized by this draft** (see the hard no-execution clause).

## 0. Principles (the v4 corrections)
- **Bounded, interpretable, pre-registered.** v4's 14/90 pass had selection-bias risk. V5 caps the space at **≤ 5–8 policy
  families × a small per-family grid**, with a **hard total budget of ≤ 24 (policy, grid) configurations across BOTH diseases**
  (proposed; pinned at sign-off). No exploratory sweep, no "найти что-нибудь красивое" then post-hoc justify.
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

## 1. Policy families (≤ 5–8; the 5 pinned for the draft)
Each family is an **abstaining router** `π(B) ∈ {identity, matched_coral, spdim, t3a}` with a small grid of pre-registered
thresholds. All thresholds are on FIT-standardized features (subject-balanced standardization; see `ACAR_V5_SPLITS.md`).

- **P1 — Benefit score + harm veto.** Adapt with the best-benefit action only if `benefit(a) ≥ τ_b` AND the harm veto passes
  (`flip_rate ≤ τ_f` AND `JS ≤ τ_j`). Grid: `τ_b ∈ {q60, q75}` (per-action benefit quantiles), `τ_f ∈ {q75, q90}`,
  `τ_j ∈ {q75, q90}` — but pinned to ≤ 4 combinations per disease.
- **P2 — Low-violence confidence gate.** Adapt only if confidence improves (`d_entropy ≤ −τ_e` OR `d_margin ≥ τ_m`) AND violence is
  low (`flip_rate ≤ τ_f` AND `JS ≤ τ_j`). Same bounded grid; emphasizes "improve confidence without disruptive relabeling".
- **P3 — Best-fixed action with abstention.** No per-batch action ranking; pick ONE pre-registered action (per disease) and execute
  it only when a single safety gate clears (`flip_rate ≤ τ_f` AND `JS ≤ τ_j`). The most conservative family; a natural lower bound.
- **P4 — Action-agreement gate.** Adapt only when ≥ k of {multiple substrate seeds (Stage-4 S1) OR multiple benefit-score variants}
  agree on the SAME action; abstain otherwise. Directly buys substrate robustness into the policy. Grid: `k ∈ {2-of-3, 3-of-3}`.
- **P5 — Conservative direct-selective gate.** A v4-style directly-LTT-calibrated selective policy, BUT with a **hard conditional
  adapted-harm cap** (G4) and a **coverage floor** (G1) baked into the calibration objective, not just `L_harm_all`. This is the
  "fix v4 in place" family — included so the failure mode is tested head-on, NOT as a favored default.

## 2. Calibration knob (shared)
Where a family uses a finite λ grid (P5, and any thresholded family expressed as a λ sweep), the grid is **pre-registered and
small** (proposed `λ ∈ {≤ 6 values}`, pinned at sign-off). λ selection is by the Stage-3 constrained criterion + LTT/Holm
certification (`ACAR_V5_ENDPOINTS.md`), NOT by maximizing utility alone.

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
