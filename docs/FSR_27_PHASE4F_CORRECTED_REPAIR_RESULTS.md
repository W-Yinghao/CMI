# FSR_27 — Phase 4F: Corrected Confirmatory Repair Test (results)

**Project FSR — Phase 4F.** Results of the corrected, pre-registered confirmatory repair test (FSR_26) on **8
fresh** token seeds (Phase 4E stays `none`, frozen — **not** re-scored). CPU-only; no GPU/retrain/CMI/fbdualpc/
target-fit. 21 folds × 8 confirm seeds = **168 seed-folds** (+ dev seed 0 for mechanism). Scripts + raw CSVs on
`project/fsr-rq4-refit`. Verdict independently recomputed, clustered-bootstrap-checked, firewall-audited, and
adversarially scrutinized (verification `wrmjni6ky`; GO-WITH-EDITS — all six mandated caveats applied).

## Headline (scoped, verbatim from verification)
> On a **CONTROLLED injected constant-offset (first-moment) spatial shortcut**, a deployable target-X-only
> first-moment mean-aligner (**E4**) — netted of its generic TTA benefit and beating a random-direction control
> (E3) — yields a **shortcut-specific repair of absolute +0.033 bAcc (clustered CI [0.009, 0.058])** on harm of
> CI_lo ≈ 0.005 (8 fresh CONFIRM seeds × 21 folds). **`repair_claim_level = strong`, qualified
> `strong_within_controlled_first_moment_scope`. SCOPE: controlled first-moment constant-offset ONLY; does NOT
> certify natural / general / DG / SOTA shortcut repair (cf. 4B/4D natural = NONE).**

The binding pre-registered level is `strong` (the frozen protocol's gates all passed). **But that word carries a
narrow, qualified meaning** — six mandatory caveats below make the honest scope explicit. This is closer to
**proof-of-plumbing on a construction-matched positive control** than to a general repair discovery.

## Primary + secondary metrics
- **PRIMARY (absolute):** E4 shortcut-specific netted gain over the random control = **+0.0326 bAcc**, clustered
  CI **[0.009, 0.058]** (E3 nets to ≈0, so E4−E3 is the token-specific residual after the generic domain-mean
  TTA benefit is removed). Harm established (pooled +0.035, clustered CI [0.005, 0.068]).
- **SECONDARY (ratio, flagged marginal-denominator):** netted recovery 0.93 — **do not lead with this.** The
  harm denominator is small (mean 0.035, 34% of folds anti-harm, per-fold min netted −6.09 on a near-zero
  denom); the ratio is a division of two smallish numbers.

## Six mandatory caveats (none omittable)
**(a) Primary is the absolute gain, not the 0.93 ratio** — see above.

**(b) 73% of the "recovery" is a MECHANICAL IDENTITY, not a discovery.** A full-space first-moment offset
injection and a full-space first-moment mean-aligner cancel by **algebra**: **123/168 seed-folds (73.2%) have
`E4_inj_bacc == E4_cln_bacc` exactly** → netted = 1.0 by construction. The pooled 0.93 is dominated by these
tautology rows; the **empirical, non-identity subset nets to 0.68** (n=45). The identity rows are a **pipeline
sanity check** ("a first-moment neutralizer inverts a first-moment injection"), not evidence of a discovered
repair.

**(c) Dataset-carried; FAILS leave-one-DATASET-out.** The specificity is not replicated across the two datasets:
| dataset | harm | harm established? | E4−E3 | specificity pass? |
|---|---|---|---|---|
| BNCI2014_001 | +0.0154 | **no** (< 0.02) | +0.0119 | **no** (< 0.02) |
| BNCI2015_001 | +0.0497 | yes | +0.0482 | yes |

Dropping BNCI2015 leaves BNCI2014, which establishes **neither** harm nor specificity. `leave_one_dataset_out_pass
= False`. The effect is **BNCI2015-carried**; N=2 sign-consistency is **descriptive only**, not a powered
replication.

**(d) LOSO-seed is the wrong (near-inert) robustness axis.** The pre-registered leave-one-seed-out passed (all 8
cuts E4−E3 ∈ [0.030, 0.035]) but it varies only the injection token within the **same** 21 folds — it does not
bind generalization. The generalization-relevant axis is **datasets** (caveat c), and that is **not** survived.
Do not present LOSO-seed as the robustness gate.

**(e) E4 is a generic domain-mean TTA operator, not a surgical shortcut remover.** E4 **overshoots** orig
(`E4_inj 0.5146 > orig 0.4956`, raw recovery 1.54) and **helps clean models** (clean drop −0.021, i.e. +0.021 on
un-injected target). The shortcut-specific claim survives only because netting removes this generic TTA component
and E3 (random direction) nets to ≈0. **Never** describe E4 as "surgically removing the shortcut."

**(f) Scope lock.** Certifies only controlled first-moment constant-offset repair. It does **not** speak to the
natural-shortcut question (4B/4D = `NONE`); conflating the two would be a serious over-claim.

## What survives / does not
- **Survives (genuine, small, scoped):** after removing the mechanical-identity tautology and the generic TTA,
  a **shortcut-specific, random-control-beating repair signal of +0.033 bAcc** exists on the controlled
  first-moment injection, with the clustered CI excluding 0 — but it is **BNCI2015-carried** and thin (~1 bAcc
  point). E1's subspace restriction adds nothing (E1−E4 < 0); ERASE is a correctly-non-falsified negative
  control (raw −0.52, clean drop +0.048, regression-to-floor True → `valid_repair = False`); the structural veto
  set + clustered bootstrap + falsification guard all behaved as designed.
- **Does not survive:** an unqualified "strong repair", any general/natural/DG/SOTA claim, cross-dataset
  robustness, or "surgical" language.

## Firewall
Clean: target labels only via `TargetScorer.score`; fit/α/`k,λ` source-only; the comparator **veto set {E1,E3} vs
{ERASE} is structural** (`comparator_veto_set_used_target = false`); only the one-sided negative-control diagnostic
reads final-eval bAcc against frozen thresholds (same category as scoring E4; can only pause, never relax). The
provable no-exclude guarantee holds (clean-safe ⟹ not regression-to-floor).

## PC2 posture
`pc2_gpu_gate = eligible_for_protocol_update`, `pc2_gpu_run_authorized = false`. This warrants recording the
scoped result and drafting a **PC2-E4 protocol update** (successor to FSR_23, which is bound to the dead 4D-D1
primitive) **for PM review only**. **PC2 GPU stays PAUSED**: a construction-matched first-moment positive control
does **not** clear the bar for repairing a *learned* natural reliance (which is not a clean constant offset, so
E4-first-moment is not expected to repair it). The PC2-E4 preflight must state on its face that it does not
authorize a GPU run.

## Manuscript impact (Result 4, current)
*"Erasure and a counterfactual head do not repair the injected shortcut. On a controlled first-moment
constant-offset injection, a deployable first-moment mean-aligner (E4) inverts the offset — largely by
construction (73% mechanical identity; pipeline sanity) — with a small genuine, random-control-beating,
token-specific residual (+0.033 bAcc) that is BNCI2015-carried and does not survive leave-one-dataset-out.
Verification, localization, and attribution succeed; a certified general/natural shortcut repair remains open —
the measurement→control gap persists at the intervention layer."*
