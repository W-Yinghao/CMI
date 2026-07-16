# Target-X observability audit — FROZEN pre-registration (Fork 2, PM-approved GO for AUDIT ONLY)

Branch `agent/cmi-trace-targetx-observability` (base `agent/cmi-trace-dg-oracle@c861563`). This file is frozen
in the closeout commit BEFORE any observability compute is run. Scope approved: **decide whether an
unlabeled-target statistic can rank beneficial deletion tickets.** NOT approved: adaptation method, CITA v2,
differentiable mask, or any deployment claim. Adaptation stays HOLD until every gate below passes.

The question is strictly: `ticket exists ⇒? target-X makes it observable` — NOT `target-X will make it useful`.

## F2.1 Calibration/query split (deployment-meaningful; frozen)
Per outer target subject, split trials into `T_cal` and `T_query`.
- **PRIMARY = session-aware** (both datasets carry `session_target`): `T_cal` = first session, `T_query` =
  the remaining later session(s). BNCI2014_001: cal=`0train`, query=`1test`. BNCI2015_001: cal=`0A`,
  query=`1B`+`2C`. This is a future-session generalization split.
- Fallback (only if a subject has one session): stratified temporal block (first-half cal / second-half query).
- **Random stratified split = SENSITIVITY ONLY**, never primary.
FIREWALL: all target-X observables use `T_cal` X only. `T_query` X and Y never enter any selector.
`Y_query` is used ONLY to compute the hidden outcome utility. Selection unit = candidate subset; inference
unit = target subject (cluster).

## F2.2 Candidate deletion families + actions (frozen; no 2^r sweep)
- PRIMARY basis: `cond` (label-conditional subject basis). SECONDARY: `marg`, `rule`, `grad`. Full (contested=False).
- Candidate actions per fold (NOT all 2^r subsets — avoids double-search bias):
  identity; each singleton `{j}`; the greedy-trajectory prefixes of the target-greedy and source-greedy paths;
  all rank-≤3 subsets; the source-greedy subset; matched-rank random.

## F2.3 Pre-declared target-X observables (signs FROZEN here; may NOT flip per dataset)
For a candidate deletion S, computed on `T_cal` only (source head = fresh logistic on all source). Sign =
direction hypothesized to predict HIGHER utility U (more positive score ⇒ predicted more beneficial).
### Geometry (source vs target-cal, before/after deleting S)
- G1 `+` reduction in ‖μ_t − μ_s‖² along/after S (mean-discrepancy reduction).
- G2 `+` target offset energy removed in S (‖Proj_S μ_t,cal‖²).
- G3 `+` CORAL/covariance discrepancy reduction after S.
- G4 `+` source–target binary domain-classifier AUC reduction after S (harder to tell apart ⇒ better).
- G5 `−` increase in condition-number / drop in effective rank after S (over-collapse is bad).
### Prediction (source-only head on target-cal)
- P1 `−` target-cal predictive entropy after S (lower entropy ⇒ more confident ⇒ better), signed as stated.
- P2 `+` prediction consistency before↔after S (stable predictions ⇒ safe deletion).
- P3 `+` mean confidence/margin after S.
- P4 `−` class-balance deviation of predictions after S (closer to uniform prior ⇒ better).
### Pseudo-conditional (source-head pseudo-labels on target-cal)
- C1 `+` pseudo-label conditional mean-gap reduction after S.
- C2 `+` pseudo-label conditional covariance-gap reduction after S.
- C3 `+` alignment of source task direction with target pseudo-task direction after S.
All 12 observable signs are frozen from the synthetic DGP / the a-priori DG intuition above; a dataset-specific
sign flip is NOT permitted (a flip ⇒ that observable is declared non-predictive, not re-fit).

## F2.4 Hidden outcome + reported quantities
Hidden ground-truth utility (uses `Y_query` ONLY here):
  U_{t,S} = bAcc_{T_query}(delete S) − bAcc_{T_query}(identity),  source-fitted fresh head.
Report two things, subject as cluster unit (bootstrap 95% CI):
- **Observability** ρ_t = Spearman(target-X score_{t,S}, U_{t,S}) within each subject; aggregate LCB.
- **Actionability** pick S_TX = argmax_S score_{t,S} (target-X only); report Δ_TX = U_{t,S_TX}. Compare to
  identity, source-greedy, matched-rank random, whitening/mean-centering, and the target-hindsight oracle.
The load-bearing number is Δ_TX on the independent `T_query`, NOT ρ.

## F2.5 GO/NO-GO gates (ALL required to advance to an adaptation plan)
1. **Observability**: ≥1 PRE-DECLARED target-X score has LCB95(median ρ_t) > 0.
2. **Actionability**: that score's S_TX has LCB95(Δ_TX) > 0 AND beats matched-rank random, source-greedy, and
   whitening/mean-centering baselines.
3. **Oracle recovery**: Δ_TX / Δ_target-hindsight ≥ 0.25.
4. **Cross-dataset safety**: one dataset clearly positive is allowed with the other inconclusive, but NOT one
   positive while the other is clearly harmful.
5. **Subject-leakage specificity**: S_TX certified by the full posterior-KL ruler (ΔÎ_enc < 0, beats
   matched-rank random).
### Outcome routing (pre-committed)
| result | next |
|--------|------|
| both datasets pass | approve a LIGHT target-X selector (still a plan for PM review, not deployment) |
| only BNCI2014 passes, BNCI2015 no-harm | exploratory only; no main-method claim |
| observables correlate but action gain not > 0 | STOP; write as diagnostic observability |
| only a dataset-specific fitted combination works | overfit verdict; no adapter |
| all fail | close the source/target-X selector line; keep TARGET_HINDSIGHT_ONLY |

## Discipline (per project feedback)
Report format each round: provenance (branch/base/head SHA/commits/config-hash/env/job-IDs); completeness
(expected/completed/failed/excluded cells + reason); firewall (uses cal-X/query-X/cal-Y/query-Y? selection &
inference unit); primary result = pre-declared PRIMARY score only (not the best of many); negative controls
(identity/random/whitening/mean-centering/source-greedy); decision (GO/NO-GO/HOLD); claim changes
(confirmed/weakened/overturned/new-pending/forbidden-wording). No adaptation code until gates pass.
