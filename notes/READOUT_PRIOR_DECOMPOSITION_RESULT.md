# Readout Prior Decomposition — P-A_PARTIAL_DEV_ONLY (verified, high confidence)

**Branch** `agent/cmi-trace-readout-prior-decomposition` (worktree `CMI_AAAI_readout_prior`). **Manuscript FROZEN.**
252 cells, 4 datasets (dev BNCI2014_001/BNCI2015_001 + external Lee2019_MI/BNCI2014_004) × seeds 0/1/2, frozen ERM
EEGNet features. Inference unit = target subject (draw→seed→subject), subject-cluster bootstrap + exact sign-flip,
Holm across budgets. Adversarially verified (workflow wzasaxi6k) — **downgraded my first-pass P-A_PARTIAL framing**;
this is the 2nd catch of the session (R-A→R-D, now P-A→P-A_PARTIAL_DEV_ONLY).

## The decomposition that settles it (verified on all 252 cells to 4 decimals)
`dU_center(k) = U_H2 − U_H1 = dU_MAP_frozen(k) + (U_H0 − U_H1(k))`. The preregistered decisive endpoint dU_center
splits into a genuine adaptation term (dU_MAP_frozen = H2 vs the frozen head) plus the from-scratch **H1 deficit**
(U_H0 − U_H1 = how far a well-regularized ridge falls below the frozen head at low labels = target-DATA-SCARCITY).

## What is REAL, and its exact scope
- **H1 is a FAIR baseline** (not rigged): it recovers the unconstrained full-cal head at Full (|U_H1@Full − fullcal| =
  0.008–0.021), and its τ0 is SMALLER than H2's τs (opposite of an over-shrink strawman). So the baseline is sound.
- **The genuine effect = `dU_MAP_frozen` (source-anchored MAP + target labels beats DEPLOYING the frozen head), and it
  is DEV-ONLY**, growing with data: BNCI2014 +0.111, BNCI2015 +0.131 at Full (LCB>0). This is a legitimate result: on
  datasets with headroom, anchoring the readout to the source head and combining with target labels helps.
- **init-invariance holds** (‖W_H1 − W_H1W‖ ≤ 2.5e-4 all datasets) → NOT a solver artifact (rules out P-C).
- **Subspace PARKED — clean under genuine high power** (matched-random ≈ 50, source-validation metric; only 1/162 Lee
  cells thin): `dGh_specific` never LCB>0 (means mostly negative — B_cond deletion is if anything *less* special than
  random). B_cond erasure stays parked.

## What does NOT survive the high bar (my first-pass over-framing, corrected)
- **dU_center does NOT isolate a prior-CENTER value.** At **Full** (the discriminating regime where H1 is well-
  estimated, not scarcity-crippled) dU_center **vanishes/reverses on BOTH dev sets** (BNCI2014 −0.014, BNCI2015 −0.003;
  on BNCI2015 the fair H1 0.763 actually BEATS H2 0.760). Its positive few-shot value is the tautological "source head
  ≫ a 1–2-shot head" = the H1 deficit, not prior combination.
- **NO external replication.** On both externals, source-only τ maxes the shrinkage (τs Lee = 100, 2b = 70–97), so H2
  **collapses onto the frozen head** — its adaptation vs frozen (`dU_MAP_frozen` few-shot) is statistically ZERO
  (Lee k1 = −0.00004; 2b k1 = +0.0011). 98–100% of each external dU_center is the H1 deficit artifact. The externals
  "win" only because they are NO-HEADROOM regimes (Lee `fullcal_gain` NEGATIVE — even a full target head can't beat
  frozen; corr(dU_center, fullcal_gain) = −0.91). The effect that replicates externally is the weak-baseline artifact;
  the effect that is genuine (dU_MAP_frozen) does not replicate externally.
- **Gate safe but near-VACUOUS externally.** dU_gate_frozen worst-subject LCB > −0.005 on all 4 (honestly "never worse
  than frozen at the −0.005 bar"), gate is code-confirmed source-only. BUT on the externals there is ~no aggregate harm
  to prevent (safety inherited, not earned), the gate fires roughly uncorrelated with per-cell harm, and removes as
  much benefit as harm. Genuine gate utility appears only on dev where the ungated method was already safe.

## Verdict = P-A_PARTIAL_DEV_ONLY_ADAPT_VS_FROZEN_NOT_PRIOR_CENTER_NO_EXTERNAL_REPLICATION
A genuine, growing source-head prior-COMBINING effect over the frozen head, scoped to **dev / headroom-present**,
measured vs the FROZEN head (not the fair ridge), and it does **NOT** externally replicate. The preregistered
prior-center estimand is confounded with target-data scarcity. Answers the PM's P0: the anchoring is **not** a solver
or unfair-baseline artifact, but neither is it a clean externally-replicated prior-center value — it is a dev-only,
headroom-dependent, adapt-vs-frozen improvement.

## Owner options (report-then-wait; manuscript FROZEN; nothing run unilaterally)
1. **ACCEPT the SCOPED claim**: "on development datasets with headroom, source-anchored MAP + target labels beats
   deploying the frozen head (dU_MAP_frozen, growing to +0.11–0.13 at Full)"; DROP dU_center as the headline and DROP
   the external-replication claim. Honest, defensible, narrower.
2. **RE-SCOPE the headline endpoint** to the practical "with 1–2 target labels, anchoring beats fitting from scratch"
   — true and replicates, but frame explicitly as a data-scarcity FALLBACK (not prior-center value), noting it equals
   "deploy frozen head" externally.
3. **LOCKBOX to break the confound at the source**: no natural-session held-out target exists in the frozen dumps
   (Cho2017 / High-Gamma are single-session). Re-dump a genuinely multi-session held-out dataset so a fair target-only
   baseline HAS headroom, then re-test whether `dU_MAP_frozen` (the genuine mechanism) survives externally in a
   headroom-present external regime — the only path to an honest external replication.
4. **Accept Lee-as-external but REINTERPRET**: keep Lee but state plainly it is a no-headroom regime (frozen ≈ oracle),
   so it tests robustness/safety, not adaptation gain.
5. **Keep the subspace PARKED** (clean high-powered negative) and do not develop new erasure methods.
