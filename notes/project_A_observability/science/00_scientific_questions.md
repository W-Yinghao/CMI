# Project A Step 12 — Scientific Questions

Step 12 does **not** write the paper. It turns the Step 8–10 audited behavior into interpretable
scientific findings, under the same claim boundary (oracle target metrics are evaluation-only).

## Q1: Why does offline TTA often harm audited target bAcc?

Candidate mechanisms (see `01_tta_harm_hypotheses.md`):
- target prior estimate is unstable or miscalibrated;
- class-conditional transport contract (C2/C8) fails;
- target support overlap (C1) fails;
- pseudo-label confidence is high but wrong;
- representation is insufficient (C6);
- source leakage diagnostics do not transfer to target gain (TOS-1 ceiling);
- the safety gate learned from source cannot identify target harm.

## Q2: Which diagnostics are source-only, target-unlabeled, or oracle-only?

Separate strictly (enforced in `harm_attribution.py`):
- **R0 source-only**: source conditional leakage `I(Z;D|Y)` per factor, source pseudo-target gain.
- **R1 target-unlabeled**: target prior estimate (entropy, L1 shift), TTA transform norm / condition
  number / density-NLL change / prediction disagreement — all label-free.
- **oracle**: identity/adapt target bAcc and the offline-TTA gain — computed with target labels, used
  ONLY as the retrospective outcome, never as a predictor feature.

## Q3: Does R1 information improve harm prediction over R0?

Studied in `harm_predictor.py` (leave-one-(dataset,target)-out logistic regression). **Important:**
even if R1 diagnostics correlate with oracle harm, this is an *empirical retrospective* harm
predictor — it does NOT make target gain identifiable (TOS-1/TU-2 stand). Balanced accuracy is
judged against the 0.5 majority baseline (harm-rate ≈ 0.83 means "always predict harm" already scores
bAcc 0.5).

## Q4: How much minimal paired information changes the situation?

Studied on a controlled simulator in `minimal_paired.py`:
- k target labels ∈ {0,1,2,4,8,16,32,64};
- phase transition in harm-sign prediction accuracy / risk-CI width / abstention rate;
- **k=0** is the R1 non-identifiability boundary; **k>0** is an R2 labeled slice under an *iid
  sampling contract*, not full-target-risk identification.

## Non-goals

- no SOTA claim;
- no manuscript writing in this step;
- no target metric reported as an R1-identifiable gain;
- no leakage-to-accuracy guarantee;
- the oracle gain is never a predictor feature.
