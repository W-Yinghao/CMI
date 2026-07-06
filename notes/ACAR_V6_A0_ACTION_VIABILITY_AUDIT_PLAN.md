# ACAR V6-A0 — Action Viability Audit (PLAN)

```
STATUS:            EXPLORATORY / DIAGNOSTIC-ONLY PLAN. NOT EXECUTED. Execution is a SEPARATE authorization.
purpose:           decide whether ANY adaptation action is worth routing BEFORE writing a V6 router.
NOT in scope:      no candidate selection, no policy fitting, no gate change, no Stage-4, no external/held-out/ASZED, no lockbox.
substrate (reuse): the ADMITTED V5 Stage-1B package — run acar-v5-stage1b-c4412b4-r1, registry_sha256 2bbe55f4…bbcbb76d,
                   10 canonical seed-20260711 DEV refs (PD/SCZ x fold0..4), impl base ba097775… (Stage-2B3 fix) or a
                   result-note-only successor.
prior line:        ACAR v5 CLOSED = DEV_STOP / NO_CANDIDATE_SELECTED (notes/ACAR_V5_CLOSEOUT.md; ledger A8/A9).
```

## Why this audit exists (the reframing)

V5's Stage-2B DEV_STOP was **not** a coverage collapse (v3's failure) and **not** a substrate-compatibility fail (v4's). Coverage
was fine (G1 PD 13/22, SCZ 19/22); the failure was **harm control** — `harm_among_adapted` UCB ∈ [0.61, 0.87] everywhere, EVAL
`red` down to −12.12, and the identity / source-state LDA `f_0` dominated every adaptation policy. So the decisive scientific
question for any successor is not "how do we route better" but:

> **Did V5 fail because the router was bad, or because the available actions are intrinsically harmful?**

**A partial answer is already in the V5 report (reused, no new computation).** The report's per-candidate `red_upper`
(= `−mean_subject mean_batch min(0, min_a ΔR_a)`, the **policy-independent oracle** that per batch takes the best of
{matched_coral, spdim, t3a} or no-op) is **constant across all 21 evaluable candidates** at **PD +0.3725 / SCZ +0.5793** — both
≫ the 0.02 gate. So a beneficial *envelope magnitude* exists: an oracle with per-batch action knowledge could reduce NLL by
~0.37–0.58 nats, while the routed policies *lose* up to 12. The gap `red_upper − red` is enormous. That means the actions are
**bimodal** (some batches benefit, most harm) and the **label-free router cannot separate them** — it is not (yet) established
that the actions are *always* harmful.

This audit closes that "not yet established": it re-derives the envelope at batch level and adds the three things the V5
aggregates do **not** give — **beneficial coverage**, **per-action / per-stratum harm structure**, and **label-free sign
predictability** — which together decide whether a V6 router could ever work.

## The four questions

### Q1 — Can an oracle help at all? (envelope magnitude AND coverage)
Compute, per disease / cohort-stratum / split / action, the per-batch `ΔR_a(B) = R_B(f_a) − R_B(f_0)` (NLL), and the
label-aware **oracle envelope**:
```
oracle_red_upper   = −mean_subject mean_batch min(0, min_a ΔR_a)          (best action or no-op per batch; benefit only)
beneficial_coverage = fraction of eligible batches with min_a ΔR_a < 0    (how OFTEN a beneficial action exists — NOT in the V5 aggregates)
oracle_conditional_harm = among batches the oracle WOULD adapt, the harmful fraction (should be ~0 by construction; sanity)
```
`red_upper` alone (already +0.37/+0.58) can be large from a few huge-benefit batches; `beneficial_coverage` distinguishes
"rare but huge" from "broad". If the oracle envelope is thin (`red_upper ≤ 0.02` OR `beneficial_coverage < 0.15`), **V6 stops** —
the failure is the action family, not the router, and no router can fix it.

### Q2 — Is harm action-specific or universal?
Split by action `{matched_coral, spdim, t3a, stable_matched_coral, identity}` and by Stage-1B provenance stratum
`{native, BrainVision-repaired, channels.tsv-renamed, montage-completed, Pz-completed, F3/F4/P3/P4-completed, F7-completed}`
(the repair/completion policies recorded in the feat-dump provenance). For each (action × stratum) report `harm_among_adapted`,
`beneficial_coverage`, mean `ΔR_a`. If every action harms in every stratum, V6 must **not** continue feature-space adaptation.
If harm concentrates in specific repaired/completed cohorts (ds004584 / ds004000 / ds004367) or specific actions, that localizes
whether the problem is the action or the substrate repair.

### Q3 — Can label-free features predict the SIGN? (the binding gate)
This is the crux: V5's handcrafted scalarizations failed, but that does not prove the features are uninformative. Test, per
disease with **subject-clustered** evaluation (subjects, not batches, are the exchangeable unit — no batch-level leakage):
```
targets:  beneficial(a,B) = 1[ΔR_a(B) < 0] ;  harmful(a,B) = 1[ΔR_a(B) > 0]
features: d_entropy, d_margin, flip_rate, JS, Bures, post_sep, n_eff, source confidence, batch entropy,
          repair/completion provenance one-hots, batch size
metrics:  subject-clustered AUROC + AUPRC (vs prevalence) + a calibration curve, per action; permutation null over subjects
```
If sign predictability is at/near chance (AUROC ≈ 0.5, AUPRC ≈ prevalence, permutation p not significant), the **router line is
dead** regardless of envelope: there is a benefit to capture but no label-free way to find it. Only a **meaningful** margin above
the subject-permutation null (pre-registered threshold, see gate) keeps the router direction alive.

### Q4 — Is identity (source-state `f_0`) already the strongest action?
Characterize `f_0` on the DEV target: NLL, ECE/reliability, mean confidence, and whether errors are **calibration / prior-shift**
(over/under-confident, class-prior mismatch) rather than **representation shift**. V5 strongly suggests `f_0` dominates. If the
residual harm is calibration/prior-shift, that motivates the V6 pivot to **bounded logit calibration** (below); if it is
representation shift that no output-layer fix can touch, that argues to stop the ACAR router line entirely.

## Pre-registered continuation gate (decide BEFORE looking, applied AFTER)

```
V6_CONTINUE  iff ALL hold (per disease, both diseases must pass):
  (a) oracle_red_upper                 > 0.02                          [PD/SCZ already +0.37/+0.58 from V5 — expected pass]
  (b) beneficial_coverage              >= 0.15                          [NEW — the real magnitude test]
  (c) oracle_conditional_harm          low (sanity: ~0 by construction; flags a computation bug if not)
  (d) sign predictability (best action, best feature-set) subject-clustered AUROC
        >= 0.60  AND  permutation-null one-sided p <= 0.05  AND  AUPRC meaningfully > prevalence

Otherwise:
  V6_STOP: no viable action family / no findable benefit — the ACAR label-free routing line ENDS (no V6 router).
```
The AUROC≥0.60 + permutation floor mirrors the pre-registered v2/v3 center-AUROC bar (do not invent a looser one). If (a)/(c)
pass but (b) or (d) fail, the honest conclusion is stronger than V5's: *the benefit exists but is label-free-unfindable* → stop.

## Data source + exploratory firewall (read carefully)

- **Reuse (no new computation):** the V5 report's `red_upper` aggregates (Q1 envelope magnitude) and the Stage-2B closeout.
- **New, label-consuming diagnostic pass (Q1-coverage, Q2, Q3, Q4):** the V5 report persisted only per-candidate/disease
  *aggregates*, NOT per-batch `ΔR_a`. Computing beneficial coverage, per-action/per-stratum harm, and sign predictability needs
  the **per-(disease, stratum, action, batch) `ΔR_a` + label-free features**, which requires re-running the frozen action seam on
  the admitted substrate AND reading DEV labels to form `ΔR`. This is a **label-consuming** pass (over labels already observed in
  V5 selection — no *new* lockbox/external exposure — but a new computation).
- **EXPLORATORY, not confirmatory.** Every output of V6-A0 is diagnostic. The DEV labels have now been used to *observe* that the
  V5 action class is harmful; therefore **any V6 policy designed using these observations is exploratory** and its efficacy is not
  established until a NEW dated protocol evaluates it on a **still-sealed** held-out / external substrate. **Do NOT reverse-engineer
  a policy or threshold from this audit and then claim a confirmatory Stage-2B pass** — that would launder DEV overfitting into a
  false positive. The audit may only *decide whether to write a V6 protocol*, never *select* a V6 policy.
- **Untouched:** external / held-out / ASZED / lockbox are NOT read; no substrate rebuild; no Stage-2B gate change.

## What V6-A0 does NOT do

```
no candidate selection · no policy/threshold fitting · no gate relaxation (G3/G4/G2-epsilon/CAL-EVAL unchanged)
no P3/P4 near-miss rescue · no Stage-4/S1-S3 · no external/held-out/ASZED · no lockbox · no batch-size/MIN_BATCH change
```

## Contingent next step (ONLY if V6_CONTINUE) — the V6 direction, not authorized here

If and only if the audit returns `V6_CONTINUE`, the V6 protocol pivots the **action family away from full feature transport** (V5
proved large 256-D feature movement severely harms NLL) to **identity-dominant, bounded, logit/probability-space calibration**:
```
A1 prior-shift / intercept calibration : logit' = logit0 + clip(δ_B, −δmax, δmax) ; δmax ∈ {0.10,0.25,0.50}, γ ∈ {0.25,0.50,1.0}
A2 temperature / confidence softening  : logit' = logit0 / T ; T ∈ {1.05,1.15,1.30}          (target NLL/overconfidence, not accuracy)
A3 convex probability trust-region     : p' = (1−γ) p0 + γ p_a ; γ ∈ {0.02,0.05,0.10}         (identity-dominant; ONLY if Q3 shows sign signal)
A4 identity-only calibration baseline  : identity + {temperature, prior-shift, softening}
```
matched_coral / spdim / t3a are retained ONLY as **diagnostic / negative comparators**, never as primary policy actions. The V6
candidate universe is **small (≤12 configs; no 90-config sweep)**; the pipeline keeps V5's disciplined FIT(thresholds)/CAL(H1-H3
Holm)/EVAL(G1-G5) structure; the **primary gates are NOT relaxed** (G1 coverage, G2 red−v2 margin ≥ 0.02, G3 L_harm_all,
G4 harm_among_adapted ≤ 0.30, G5 benefit retention). If even conservative identity-dominant calibration cannot pass, the resulting
negative is stronger than V5's. All of this requires a NEW dated protocol (`notes/ACAR_FROZEN_v6.md` + action-space/endpoints/splits
docs), authored only after this audit passes — it is described here for orientation, **not** authorized.

## Authorization status

```
THIS DOCUMENT:   a PLAN only. No code, no data, nothing executed.
NEXT DECISION:   whether to EXECUTE the V6-A0 diagnostic (a separate authorization).
AFTER EXECUTION: V6_CONTINUE -> author notes/ACAR_FROZEN_v6.md (separate authorization) ; V6_STOP -> the ACAR router line ends.
```
