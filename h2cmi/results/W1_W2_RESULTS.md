# W1 + W2 results (frozen run; W1 690 rows/115 LOSO subjects, W2 140 rows/20 sleep subjects)
# raw sha256: w1_all=03c9d4f4c6ea20c4  w2_all=52e174a44325a69f . W1-B native BTTA-DG = IN PROGRESS
# (GitHub reachable after all; official luo-huan-123/BTTA-DG pinned at commit 5932d026, MIT; bounded
# literature-comparability reproduction, separate panel, cannot change the W1/W2 verdict).

## HEADLINE: joint geometry-prior adaptation is PREVALENCE-DEPENDENT (real-data, both panels)
W1 balanced-prevalence MI LOSO  -> current_joint is BEST  (overall Δ +0.090, harm 0.17; Cho +0.180).
W2 natural-prevalence sleep      -> current_joint is WORST (Δ -0.043 [-0.071,-0.017], CI EXCLUDES 0,
                                    harm 0.85, ΔmacroF1 -0.080).

FRAMING (use this language; do NOT write "causally confirmed by natural prevalence" -- MI and sleep
differ simultaneously in task, class count, signal structure and data-generating process):
> The effect of joint geometry-prior adaptation is prevalence-dependent: estimating the target prior
> can help when class composition is stable, but can drive severe negative transfer when prevalence
> varies.
The strongest evidence is the COMBINATION, supporting a PREVALENCE-CONDITIONAL FAILURE MECHANISM (not a
single causal confirmation): (1) V1 controlled simulator localizes the prior M-step; (2) V2P shows
fixed-prior CC still moves with pool prevalence on real EEG; (3) W1 shows the joint can be effective in
near-balanced MI; (4) W2 shows the joint is significantly harmful under natural prevalence variation.
So the headline is NOT "joint EM is harmful" but the prevalence-dependent statement above.

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

## Conclusion bound (respected) + sleep metadata-route statement (use this language)
> Metadata-selected fixed-prior CC substantially reduces harm relative to joint adaptation, but its
> mean utility and advantage over pooled adaptation are not statistically established.
(NOT a blanket "the metadata route succeeds".) CC is at most RELATIVELY more prevalence-robust than
pooled (directional, n=20 underpowered), NEVER prevalence-invariant (V2P settled that). Utility of
unlabeled adaptation remains modest/marginal on real EEG. W1-A and W1-B are NOT cross-ranked (separate
panels).
