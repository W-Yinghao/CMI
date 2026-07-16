# Target-X observability audit — PRE-REG AMENDMENT 01 (F2.0c; supersedes conflicting parts of the base pre-reg)

Transparent amendment to `notes/TARGETX_OBSERVABILITY_PREREG.md` per PM code review (does NOT silently edit the
frozen base). Branch `agent/cmi-trace-targetx-observability`. All items below are BLOCKING for the smoke gate.

## A1 — target-greedy path is REMOVED from the eligible target-X action set (target-label contamination)
The base pre-reg's candidate actions included "greedy-trajectory prefixes of the **target-greedy** ... paths".
The target-greedy trajectory is constructed with `Y_target`, so admitting it into the set the target-X
selector chooses from violates `Y_cal ∉ selection` even if the final ranking uses only target-X. AMENDMENT:
- **Eligible target-X actions** (deployable candidate set) = { identity; source-fitted-basis singletons {j};
  all rank-≤3 subsets of the basis; the source-greedy trajectory/prefixes; fixed-seed matched-rank random }.
- The **target-greedy** ticket path is used ONLY as: the hindsight oracle, the recovery-ratio denominator,
  and a diagnostic comparison. It is NEVER in the target-X selector's eligible set.

## A2 — UNIQUE primary observable (kills best-of-many multiplicity)
- **PRIMARY = G1 = source–target mean-discrepancy reduction** on `T_cal` (see A4). Rationale: most directly
  tied to the L2 target-geometry positive cell, no pseudo-label, no confidence calibration, most stable def.
- All other 11 observables are **SECONDARY**, reported with **Holm** correction across the secondary family.
- A secondary-only hit CANNOT unlock adaptation; it can at most support an exploratory diagnostic.
- Gate 1/2/3 are evaluated on **G1 only** for the GO decision.

## A3 — identity action + all-negative fallback
- The identity action's target-X score is FIXED = 0 for every observable (reference point).
- If, for a subject, every non-identity deletion's target-X score ≤ 0, the target-X selector RETURNS IDENTITY
  (S_TX = ∅, Δ_TX = 0). No negative-score deletion is ever deployed.

## A4 — frozen observable formulas (no implementation-time freedom)
- **G1 (primary)** = reduction in squared source–target-cal mean discrepancy after deleting S:
  `G1(S) = ‖μ_s − μ_{t,cal}‖² − ‖(I−P_S)(μ_s − μ_{t,cal})‖²`, where `P_S` projects onto span(basis[S]);
  μ_s on source, μ_{t,cal} on `T_cal`. Sign `+` (more reduction ⇒ higher predicted utility).
- **G2** = `‖ P_S (μ_{t,cal} − μ_s) ‖₂²` (offset energy in S, RELATIVE to source origin; not absolute). Sign `+`.
- **G5** = `−[ log κ(Σ_S) − log κ(Σ_identity) ]`, condition-number of the post-deletion source covariance
  (κ = λ_max/λ_min over the retained subspace). Sign `+` (a big condition-number blow-up ⇒ penalized).
  Effective-rank change is a SEPARATE secondary diagnostic, never combined into G5 at runtime.
- **P4** = `−JSD( p̂_t(Ŷ) ‖ p̂_s(Y) )`, prediction class-balance divergence from the FROZEN source empirical
  label prior `p̂_s(Y)` (NOT a uniform prior). Sign as written (closer to source prior ⇒ higher score).
- **C3** = alignment of source task direction with target pseudo-task direction after S: for `n_cls=2` the
  single class-contrast cosine; for `n_cls=4` the **macro-average cosine over all class-pair contrasts**
  (fixed; not one-vs-rest, not a runtime choice). Sign `+`.
- G3, G4, P1, P2, P3, C1, C2 keep the base pre-reg definitions and frozen signs (secondary).

## A5 — new pre-committed outcome verdict
Add `TARGET_X_UTILITY_OBSERVABLE_NOT_CMI_SPECIFIC`: fires when the G1-selected action has LCB95(Δ_TX) > 0 and
beats random/source-greedy/whitening, BUT the posterior-KL specificity gate (Gate 5) FAILS. Interpretation:
"unlabeled target geometry predicts a useful alignment action" — a general transductive-alignment finding
(Discussion/Appendix), NOT support for CMI-guided erasure / TOS improvement / subject-leakage removal helps DG.

## A6 — session split manifest (preflight, before any audit compute)
Emit `results/cmi_trace_dg_identifiability/session_split_manifest.csv`, one row per (dataset, subject):
`dataset, subject, cal_sessions, query_sessions, n_cal, n_query, class_counts_cal, class_counts_query,
fallback_used, exclusion_reason`. A subject with an unusable split (single session and too few trials to block)
is EXCLUDED with a logged reason (never silently dropped).
- BNCI2015 query PRIMARY bAcc = macro-average over query sessions: `(bAcc_1B + bAcc_2C)/2` (equal session
  weight); pooled-trial query bAcc = SENSITIVITY only.

## A7 — firewall reporting (split selection vs outcome)
Every audit report states, verbatim: `query_x_used_for_selection=false`, `query_y_used_for_selection=false`,
`query_x_used_for_outcome=true`, `query_y_used_for_outcome=true` (final scoring necessarily uses query X and Y).
Selection unit = candidate action; inference unit = target subject (cluster).

## A8 — firewall tests (blocking for smoke)
Automated tests assert: (i) the eligible action set contains NO target-greedy-derived action; (ii) target-X
scores are computed from `T_cal` arrays only (perturbing `T_query` X leaves every score unchanged); (iii) the
identity fallback triggers when all scores ≤ 0; (iv) G1 has its frozen sign and is 0 for identity.

## A9 — unchanged from base pre-reg
Session-aware cal/query (A1 of base), hidden-outcome utility, the 5 GO/NO-GO gates (now G1-only for gates
1–3), and the outcome routing table. No adaptation code until all gates pass.
