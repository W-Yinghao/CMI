# ACAR v4 — candidate selection (post-v3 EXPLORATORY)

```
STATUS        : EXPLORATORY_CANDIDATE (post-v3 DEV model-selection; NOT a confirmed / external result)
RESULT COMMIT : e4c4e91  (V4_DEV_CANDIDATE_FOUND_FOR_POSSIBLE_FREEZE; results/acar_v4_dev_exploration_001/)
LOCKBOX       : NOT CONSUMED      EXTERNAL ARM : NOT APPROACHED      FREEZE : skeleton only (not tagged)
DATE          : 2026-06-29
```

This records the single candidate selected for a possible future `ACAR_FROZEN_v4.md`, from the 14 (of 90) both-disease
configs that passed the pre-registered G0–G6 in DEV exploration #001. This is **post-v3 exploratory candidate
selection on development data**, not a confirmatory or external claim (selection over 90 configs ⇒ selection bias;
DEV-only; lockbox sealed). See `notes/ACAR_V4_DEV_EXPLORATION_001_RESULT.md` and the manifest
`results/acar_v4_dev_exploration_001/manifest.json`.

## Selection criterion (fixed before reading the per-config harm numbers)
**Minimize the maximum per-disease harmful-adapted-batch rate** among G0–G6 passers, subject to retaining macro
deployed-NLL reduction > the v2-replay comparator (macro 0.0985) and non-vacuous coverage (≥ 0.15 per disease). Rationale:
V4's value is converting the v2/v3 measurement→control gap into a **risk-controlled, externally-confirmable** candidate,
not maximizing DEV utility. The most likely external-Arm-B challenge is the harmful-adapted-batch rate, not whether the
DEV macro red is large.

## Selected candidate
```
score_family   : shift_margin      harm = benefit = +features_v2[:, :, 1]   (= +d_margin; label-free)
policy_family  : benefit_ranked    (CANONICAL representative)
loss           : harm_indicator    (subject-level: mean 1[adapted ∧ ΔR>0]; fallback realizes 0, in denominator)
```
DEV #001 numbers (OOF, subject-macro): **max disease harm 0.2054 (the global minimum among all passers), macro red
0.1585 > v2-replay 0.0985**, both diseases non-vacuous, all 5/5 EVAL folds certified.

| disease | coverage | deployed red | harm rate | per-fold λ* (median) |
|---------|----------|--------------|-----------|----------------------|
| PD  | 0.198 | 0.1161 | 0.1538 | −0.374 |
| SCZ | 0.249 | 0.2009 | 0.2054 | −0.361 |

**Policy representative disambiguation (required — ACAR_FROZEN_v4.md must name ONE policy family).** For
`shift_margin + harm_indicator`, `benefit_ranked` and `direct_selective` are **bit-for-bit numerically identical**
(coverage/red/harm equal to machine precision on both diseases; the only difference is the λ sign, because
`direct_selective`'s gate is `−min(harm)` with decreasing-λ aggressiveness). They are the SAME executed policy. We
therefore freeze **`benefit_ranked`** as the canonical representative and record `direct_selective` as **numerically
equivalent, non-primary**.

## Rejected alternatives (with reasons)
- **High-utility `shift_margin + benefit_ranked + mean`** (macro red 0.419; PD cov 0.86 / SCZ cov 0.54): REJECTED —
  harmful-adapted-batch rate 0.33–0.46. Highest DEV utility but the weakest safety story; hard to defend as a "safe
  router" under external Arm B.
- **Balanced `shift_margin + benefit_ranked + positive`** (macro red 0.228; cov 0.28–0.29): REJECTED — max disease harm
  0.242 buys utility, not a stronger safety story, vs the selected candidate's 0.205.
- **`shift_margin + safe_set + harm_indicator`** (macro red 0.131): REJECTED — strictly dominated by the selected
  candidate (higher max harm 0.230 AND lower macro red 0.131).
- **`n_eff_neg` families**: REJECTED — higher harm (0.27–0.44) at comparable/lower macro red.

## Status / next step (GATED)
This selection is the input to a `ACAR_FROZEN_v4.md` SKELETON (drafted alongside this note; not yet tagged). Freezing
the protocol (tag `acar-v4-protocol`) and any external / held-out Arm B run require explicit sign-off and a re-audited
held-out cohort list. Until then: lockbox sealed, external Arm B unauthorized, no threshold/seed/loss/registry change to
chase a better DEV number. Never edits the frozen v2/v3 commits or tags.
