# C30 — Adversarial Red-Team Verification (ultracode multi-agent)

5 independent skeptic agents each ran a refutation check on the real C22 sidecar (`oaci-c22-scores.json`,
in_regime, 3804 rows, 9 targets) + a synthesis agent. Workflow `wf_995fa8a9-383`. Each verdict below is the
agent's adversarial conclusion; the C30 report/taxonomy were amended to match.

## G1 — two-axis separation (rank ≠ epoch/order trajectory): **SURVIVED (CONFIRMED_REAL)**
- Trajectory confound fails: score within-target AUC 0.659 (strength 0.159) vs epoch 0.519 (0.019) / order 0.484
  (0.016) — trajectory axes 8–10× weaker; residualizing score on epoch *raises* strength to 0.666 (+4.8%), on
  order 0.657 (+1.1%) — nowhere near a ≥50% collapse. within-target corr(score,epoch)=−0.059, (score,order)=−0.137.
- Permutation null (200 within-target label shuffles, RS707): null mean 0.4993, p95 0.5155, max 0.538; observed
  0.6586 sits z=15.14 above, 0/200, p=0.005. Certifies the within-target rank axis ONLY (not transport).

## G2 — source-risk R_src carries the rank: **SURVIVED (CONFIRMED_REAL) — WEAK, caveated**
- R_src within-target AUC 0.376, strength 0.124; permutation null strength mean 0.0075 / p95 0.019 / max 0.025;
  observed ~5× null max, ~20 SD, 0/200, p=0.005. Direction consistent 7/9 targets (higher risk → lower competence).
- Residualization: epoch→0.1245 (unchanged), order→0.1127, **train_surrogate→0.0775 (~38% shrink, same
  source_risk family)** — still ~4× null p95. Keep the "weak/diagnostic" framing; disclose the 38% absorption.

## G4 — leakage not carrier: **NOT RED-TEAMED THIS ROUND** (asserted; leakage strength ~0.04, not re-confirmed).

## G5 — "rank tracks source error only": **PARTIALLY REFUTED → REWORDED**
- Leg (a) "tracks source error" is **TAUTOLOGICAL**: corr(R_src, source_guard_nll)=0.985 within-target / 0.993
  pooled; ratio 0.986±0.017; residualizing R_src on the source NLL → strength 0.016 (chance). R_src *is* the
  source risk — the flag fires only because it compares one source-error quantity to another.
- Leg (b) "weak target carryover" **understates**: the rank does NOT transfer — R_src per-target AUC sign-flips
  (7/9 below 0.5, but t1=0.636, t8=0.565 above; sign-consistency 0.778). The mean strength 0.124 = |mean signed
  AUC − 0.5| *cancels* opposite-sign per-target AUCs, masking a target-LOCAL signal.
- Survives: the negative conclusion — R_src is NOT a calibrated/deployable target-competence score.

## G7 — distributed source residual ("score beats any family"): **PLAUSIBLE → DOWNGRADED**
- Overfit attack REFUTED: the in_regime score is a genuine out-of-target LOTO prediction (AUC>0.5 for all 9
  held-out targets; permutation p=0.00005, strength 0.159 far above null max 0.036). 0.659 is not in-sample overfit.
- But "beats any single family" (gap 0.034) is **WITHIN 9-target cluster-bootstrap noise** (95% CI [−0.059,
  0.136] includes 0; best family ties/beats in 24.4% of resamples). Distributedness kept only in the RESIDUAL
  sense: score|R_src strength 0.106 retained; and the transfer contrast — **the multivariate probe is
  sign-consistent 9/9 (transfers) while single families sign-flip (target-local)** — is the honest evidence that
  the transferable within-target rank is distributed, not any single source family.

## Bottom line
G1 survives cleanly. G2 survives but stays weak/diagnostic (train_surrogate absorbs ~38%; target-local). **G5
reworded** (tautological + non-transferring; only the negative conclusion survives). **G7's "beats any family"
downgraded to noise-level**; distributedness kept in the residual/transfer sense (probe transfers 9/9, families
do not). All permutation certifications are within-target ONLY — none speaks to cross-target transport or
source-observability, which C23–C29 separately found to fail. Nothing here is a deployable selector.
