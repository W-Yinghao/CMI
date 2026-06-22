# W1 + W2 frozen pre-registration (starts from the audit-corrected node 9c42fd6 / tag h2cmi-audit-corrected)

Must NOT reference the old 18-route V2-A or 72-unit V2P numbers. Target labels EVAL-ONLY throughout.
After W1+W2 the experiment program stops again (no rescue branches; results only update paper claims).

## W1 — standard unseen-subject MI benchmark + SPDIM + native BTTA-DG (TWO separate panels)
Datasets: BNCI2014_001-LR, Cho2017, Lee2019_MI (the 3 V2 MI datasets).
W1-A  controlled SAME-BACKBONE, disjoint-batch protocol (one frozen H2 source model per fold):
      methods = identity, EA, pooled, canonical fixed-prior CC, current_joint, SPDIM.
      protocol = source: all non-target subjects; target adaptation: earlier target block (unlabeled);
                 target evaluation: later DISJOINT block; unit = target subject (LOSO).
W1-B  native literature protocol (EXTERNAL end-to-end reference, NOT significance-ranked vs W1-A):
      official BTTA-DG + SincAdaptNet, official online-LOSO ordering.
      ** BLOCKED: BTTA-DG is not on disk and this environment has no network. W1-B is DEFERRED until
         the official BTTA-DG repo is placed on disk. W1-A + SPDIM proceed now. **
Do NOT put W1-A and W1-B in one ranking table (different backbone/representation/protocol).

## W2 — Sleep-EDF Sleep-Cassette cross-subject (natural prevalence shift)
Name (honest): "Natural prevalence variation under a shared acquisition protocol" (NOT pure label
shift -- subject-specific p(x|y) may also vary).
Data (frozen): Sleep-EDF Expanded, Sleep-CASSETTE only (no Sleep-Telemetry/temazepam). Channels
Fpz-Cz + Pz-Oz; 100 Hz; 30 s epochs; classes W/N1/N2/N3/REM (S3+S4->N3; movement/unscored discarded);
crop +-30 min around the scored sleep period.
Subject protocol (frozen): for each target subject with 2 valid nights -- source training = ALL nights
of all OTHER subjects; target adaptation = target subject NIGHT 1 (fully unlabeled); target evaluation
= target subject NIGHT 2; statistical unit = target subject. ONLY night1->night2 (no reverse).
Methods: identity, pooled geometry, canonical fixed-prior CC, current_joint, EA, SPDIM, metadata_only.
Frozen metadata route (FIRST real-data trigger of DIAG_COMPATIBLE x DIFFERENT -> CC):
    geometry = DIAG_COMPATIBLE (same dataset/channels/reference/acquisition protocol)
    prevalence_risk = DIFFERENT (sleep-stage composition naturally varies across subjects/nights)
    metadata_only = canonical fixed-prior CC
Conclusion bound: this can ONLY test whether CC is RELATIVELY more robust to prevalence variation than
pooled and yields better subject-level safety/utility. Even if CC wins it is NOT prevalence-invariant
(V2P already confirmed fixed-prior CC moves with pool prevalence).
Primary estimands: paired DbAcc vs identity, macro-F1, subject-level harm rate, worst-quartile DbAcc,
transform displacement, predicted stage occupancy. Frozen MECHANISM analysis (NOT routing): regress
(DbAcc_CC - DbAcc_pooled) against JS(rho_T, rho_S), the post-hoc JS divergence between the target
adaptation-night and source stage compositions.

## Termination boundary (after W1+W2)
NOT included: Sleep-EDF->ISRUC, Sleep-EDF->HMC, additional sleep datasets, additional TTA methods,
new ratios, new eligibility gates, orthogonal-estimator implementation, dataset-specific operator
tuning. Stop after W1+W2; results (positive or negative) only update paper claims.
