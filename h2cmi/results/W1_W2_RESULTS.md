# W1 + W2 results (frozen run; W1 690 rows/115 LOSO subjects, W2 140 rows/20 sleep subjects)
# raw sha256: w1_all=03c9d4f4c6ea20c4  w2_all=52e174a44325a69f . W1-B native BTTA-DG = DEFERRED (repo not on disk, no network).

## HEADLINE: the joint's prior-M-step harm is PREVALENCE-CONDITIONAL (real-data, both panels)
W1 balanced-prevalence MI LOSO  -> current_joint is BEST  (overall Δ +0.090, harm 0.17; Cho +0.180).
W2 natural-prevalence sleep      -> current_joint is WORST (Δ -0.043 [-0.071,-0.017], CI EXCLUDES 0,
                                    harm 0.85, ΔmacroF1 -0.080).
This is the first REAL natural-prevalence confirmation of the V1 simulator mechanism and sharpens it:
the joint harms SPECIFICALLY under prevalence shift, not universally. When prevalence is balanced the
prior M-step is harmless (the joint's capacity even helps); under prevalence shift it chases the wrong
prevalence and harms.

## W1-A unseen-subject MI LOSO benchmark (same-backbone; W1-B is a separate deferred external panel)
OVERALL (n=115): identity bAcc 0.671 | EA +0.030 (harm 0.42, wq -0.081) | pooled +0.068 (harm 0.24) |
  canonical_CC +0.053 (harm 0.26) | current_joint +0.090 (harm 0.17, wq -0.023, BEST) | SPDIM +0.052
  (harm 0.27). Per-dataset: Cho2017 current_joint +0.180 (bAcc 0.808); Lee2019 all gains small
  (+0.01..+0.02); BNCI2014_001 pooled +0.051 best there. SPDIM (modern source-free IM baseline) is
  mid-pack and competitive but never best. EA weakest + highest harm.

## W2 Sleep-EDF cross-subject natural-prevalence staging (n=20)
identity (baseline) | always_pooled +0.009 [-0.007,+0.024] harm 0.40 | canonical_CC = metadata_only
  +0.015 [-0.004,+0.033] harm 0.30 | current_joint -0.043 [-0.071,-0.017] harm 0.85 (HARMS) |
  EA -0.001 harm 0.55 | SPDIM -0.002 harm 0.45.
The metadata route (DIAG_COMPATIBLE x DIFFERENT -> CC; first real-data trigger) is the SAFEST adapter:
harm 0.30 vs the joint's 0.85; Δ +0.015 (CI includes 0 -> modest/marginal utility, like V2-B).
KEY contrast (Δ_CC - Δ_pooled) per subject = +0.0058 [-0.0033,+0.0158] -> CC NOT significantly better
than pooled. MECHANISM: slope (Δ_CC-Δ_pooled) vs JS(rho_T,rho_S) = +0.42 [-0.40,+1.56] (CI includes 0);
component slopes Δ_pooled vs JS = -1.11, Δ_CC vs JS = -0.69 -> both degrade as prevalence diverges, CC
LESS steeply (directional support for CC relative robustness, not significant at n=20).

## Conclusion bound (respected)
CC is at most RELATIVELY more prevalence-robust than pooled (directional, n=20 underpowered), NEVER
prevalence-invariant (V2P settled that). The clean confirmatory result is the JOINT'S PREVALENCE-
CONDITIONAL HARM and that the fixed-prior metadata route AVOIDS it. Utility of unlabeled adaptation
remains modest/marginal on real EEG. W1-A and W1-B are NOT cross-ranked (separate panels).
