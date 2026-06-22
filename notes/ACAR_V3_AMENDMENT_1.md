# ACAR v3 — Amendment 1 (DESIGN-ONLY, NON-BINDING)

**Date:** 2026-06-22 · **Status:** `NON-BINDING / NO DEV NUMERICAL RUN / NO LOCKBOX ENDPOINT ACCESSED`
**Applies to:** `ACAR_V3_FREEZE_SKELETON.md` (rewritten to fold this in), `ACAR_V3_DESIGN_DRAFT.md` (bannered),
`ACAR_V3_LOCKBOX_AUDIT.md` (verdicts revised). Resolves the six protocol ambiguities + three statistical-detail gaps
+ three metadata-verdict corrections raised in adversarial review of `50adbc1`. Changes the design only; touches no
executable code or result artifact; v2 endpoint `1528a94` / tag `acar-v2-protocol` @ `9b2f0c1` unchanged.

## What changed (and why)

1. **C3 is now a single, well-defined additive one-sided CQR.** The prior skeleton mixed an additive score
   (`ΔR−q̂₀.₉₀`) with a normalized one (`/(q̂₀.₉₀−q̂₀.₅₀)`) — two different algorithms. Fixed:
   `q̂₀.₉₀=q̂₀.₅₀+softplus(d)+ε`, `U_a=q̂₀.₉₀+q`, **`w_min` deleted for C3**. CQR's value is directly learned
   conditional quantiles + finite-sample correction; dividing by a predicted width would re-import C2's scale risk.
   Pinball weights fixed: `L=½ρ₀.₅+½ρ₀.₉`.
2. **S2 calibration gate is candidate-specific** (Gaussian variance≈1 / z₀.₉ tail applies to C2 only; C3 uses
   exceedance-rate/positive-excess/crossing checks), **subject-balanced**, computed **per disease × action** on
   **OOF held-out** DEV subjects (not the external CAL). `max_a`-dominance is subject-level with **fractional ties**
   (action-order invariant). C0 is explicitly comparator-only.
3. **σ_min is per disease × action**, and for the final model derived from **OOF** σ̂ (`Q₀.₀₅`), not in-sample — a
   pooled floor could let one action stay chronically low-scale and re-capture `max_a`.
4. **β-NLL pinned to the exact Seitzer form** with `v=softplus+ε`, **stop-gradient weight `v.detach()**0.5`**, plus
   ε, subject-balanced reduction, per-disease standardized target, Huber-δ units, grad-clip, init.
5. **Two-phase lock added** (`DEV_DESIGN_LOCK` before any DEV run; `EXTERNAL_PROTOCOL_FREEZE` → real
   `ACAR_FROZEN_v3.md` only after DEV gate + audit) to remove the post-DEV forking path. DEV no-pass is
   **`DEV_STOP / NO_LOCKBOX_CONSUMED`**, not `TERMINATE`.
6. **Design draft synced** (banner: V3.4–V3.6 superseded; the C0/C1/C2-only list, "historical/future", and the
   "≥3-cohort" clause are obsolete).
7. **Unique "train once on DEV FIT pool" procedure** specified (outer OOF folds → inner early-stop → select →
   single refit on all S2-admissible DEV subjects → OOF σ_min → serialize+hash); v2 replay = full v2 recipe
   (HGB+Ridge/constant fallback), same pool, identical Arm-B protocol.
8. **Statistical details frozen:** candidate-comparable **width** `W_c = subject-macro mean(U−m_c)` with center
   `m_c=μ̂` (C0–C2) or `q̂₀.₅₀` (C3); **harmful adapted-batch test** = one-sided Wilcoxon signed-rank across subjects,
   α=0.05, Holm across sites, denominator = adapted batches; **coverage** = conditional one-sided exact-binomial
   undercoverage **diagnostic** (Holm across sites), explicitly not an exact test of the marginal theorem.
9. **External deployment substrate frozen from DEV** (§S7): encoder / `f0` / source moments / readout / prototypes /
   action state are DEV-frozen; external labels compute **only** `q`; the raw→feature pipeline is frozen + hashed so
   `f_0,f_a,ΔR_a` are uniquely determined. v2's per-cohort SourceState fit is disallowed for v3 external sites.
10. **Metadata verdicts corrected** (see `ACAR_V3_LOCKBOX_AUDIT.md`): ds007020 rationale changed to "no documented
    usable HC-vs-PD label" (94 subjects, 500 Hz, mortality `living/deceased` labels) — the earlier "UCSD 15+16 /
    overlap / <30 CAL" inference is withdrawn; ds007526 numbers updated (144 subj, 116 PD + 28 HC, 65 ch, 250 Hz,
    CC0, rest+walk); ASZED downgraded to `DATA-INTEGRITY REVIEW REQUIRED` (two Nigerian acquisition units/devices,
    16-ch, 200/256 Hz; a 2026-05 preprint alleges signal reuse — unverified, but blocks upgrade).

## Execution order (this amendment authorizes only design + DEV engineering)

1. **(done here)** amendment + skeleton rewrite + draft banner + audit revision.
2. **Next:** implement `acar/v3/set_features.py` (per-window paired tensor + availability masks + batch context) and
   the synthetic invariance/guard tests. Does **not** wait on the second PD site.
3. Implement C1/C2/C3 + candidate `upper_bound()` + subject joint conformal; commit + tag `acar-v3-dev-design-v1`
   (the `DEV_DESIGN_LOCK`).
4. Run the **DEV gate only** (S2 + S4). Pass ⇒ proceed to audit-completion + `ACAR_FROZEN_v3.md`. Fail ⇒
   `DEV_STOP / NO_LOCKBOX_CONSUMED`.
5. **Parallel, metadata-only:** verify ds007526 primary version/mapping/overlap/CAL counts; clarify ASZED
   acquisition units + integrity flag; source a second independent PD site (HC+PD, resting, raw, ≥30 CAL, no DEV
   overlap) — **do not** add small/overlapping/label-incompatible data to force two sites. If DEV completes with one
   PD site, PD is frozen **site-specific**; the stronger framing is SCZ = replicated external claim, PD = single-site
   confirmatory.

No ACAR inference, adaptation-outcome inspection, or endpoint access occurs before `ACAR_FROZEN_v3.md` is committed
and tagged on a clean `acar` worktree.
