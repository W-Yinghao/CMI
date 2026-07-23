# Panel E — framing decision record (neutral)

**Decision (owner, 2026-07-23): Option 1 — accept as-is.** Frame Panel E as *non-identifiability robust to
calibration labels*. No new run (Option 2, strengthen injected worlds, was declined). Manuscript caption /
discussion prose remains owner-controlled; this file records the pre-committed interpretation + the exact
supporting evidence so the wording is defensible and reproducible.

Data source: `tos_cmi/results/target_info/tier1_smoke/target_info_tier1_smoke_summary.json`
→ consolidated into `paper_evidence.json` (`panel_E_information_regime_ladder`, `panel_E_framing`) +
`panel_E_ladder.csv`.

## Pre-committed interpretation grid — which cell fired
> Panel E hoped to show a **crossing**: "calibration certifies a beneficial erasure at k=X." It does not.
> The observed cell is **flat-0 across the ladder with a safe gate**, which the interpretation grid maps to
> *strong non-identifiability that target labels (up to k=16/class) do not repair*, **not** a crossing.

## Supporting evidence (exact)
Scope: Lee2019_MI + Cho2017 (2 source→target pairs), **EEGNet only**, folds {1..5}, worlds
{v2_source_invisible_world_a, source_rich_source_visible_world_a}, budgets B0..B4, k_grid {1,2,4,8,16},
R=10, world_alpha_grid {0.5,1.0,2.0}, n_boot=100.

| regime | deployable accepts | clears gate |
|---|---|---|
| B0 source-only | 0 | no |
| B1 target-X (unlabeled) | 0 | no |
| B2 calibration k=1 | 0 (true 0 / false 0) | no |
| B2 calibration k=2 | 0 (true 0 / false 0) | no |
| B2 calibration k=4 | 0 (true 0 / false 0) | no |
| B2 calibration k=8 | 0 (true 0 / false 0) | no |
| B2 calibration k=16 | 0 (true 0 / false 0) | no |

- **Deployable false-accept rate = 0.000** (gate is safe: it never certifies a non-beneficial erasure).
- **Any regime crosses ceiling = False.**
- **Non-deployable oracle (B4) upper bound** (diagnostic only, NOT deployable):
  source-invisible world mean ΔbAcc **0.0179**, max **0.080**; source-visible world mean **0.0209**, max **0.080**.

## Honest caveats carried by Option 1 (must survive into the manuscript)
1. **Scope-limited** — 2 source→target pairs, EEGNet only, 5 folds, 2 semi-synthetic worlds. The claim is
   "robust to calibration labels **up to k=16/class in these worlds**," not a universal claim.
2. **The ceiling is itself small.** Even the oracle that *sees* the target tops out at ΔbAcc max 0.080
   (mean ~0.02), including in the source-**visible** world. So the injectable benefit is small relative to the
   +0.01-LCB-beats-random bar. "Injected worlds too weak" is the acknowledged alternative reading; Option 1
   declines it in favour of the non-identifiability reading. The two are not distinguishable from this table
   alone — the manuscript must not claim the *stronger* "no benefit exists," only "no regime **certifies** a
   deployable benefit here, and the gate stays safe."
3. Abstains at small k are expected by design (prior-run note), so the flat low-k rows are not a bug.

## What Option 1 does NOT license
- No claim that a beneficial erasure is *absent* (the oracle finds a small one; it is just not
  source-/calibration-identifiable). Keep the four-object separation: **not certifiable ≠ not present.**
- No crossing narrative, no "k=X suffices" sentence.
