# C32 - Claim Boundary Audit (C32R)

Scope: remote-audit reinterpretation patch over the committed C32 artifacts. No experiments rerun; no EEG training;
no score tuning; frozen C19 config `664007686afb520f` unchanged.

## Established

- **Joint-good checkpoints are common at the frozen primary margin.** Joint-good rate is 42.4%, and 94.4% of
  trajectory-regime units contain at least one joint-good candidate.
- **The selected OACI hit rate is essentially trajectory-conditioned random top-1.** Selected OACI hits joint-good
  44.4% of the time; random top-1 is 43.0%.
- **The landscape is locally dense.** Median nearest joint-good order distance from selected OACI is 1, with median
  epoch distance 5.
- **Source scores weakly enrich but do not reliably localize.** Source top-1 is 50.6% vs random 43.0%; source top-3
  and top-5 enrichment are only 1.023x and 1.086x.
- **Target-unlabeled confidence geometry helps pooled/gauge separability only.** Pooled AUC improves from 0.541
  (source score) to 0.583, but top-1 trajectory localization is worse than source (35.2% vs 50.6%).
- **Target-grouped centering is a non-deployable diagnostic ceiling.** It improves pooled AUC to 0.645 and recovers
  the within-target rank signal, but it uses target grouping.

## Weak / Margin-Sensitive

- **Primary C32 taxonomy is frozen-margin-specific.** At robust margin 0.02, joint-good rate drops to 27.8%,
  trajectory-regime coverage to 77.8%, and selected hit to 29.6%; J1/J5 do not fire there.
- **Selected-near-joint-good is not a deployable localization claim.** It reflects local density under the frozen
  primary margin, not selector skill.

## Not Established

- Deployable joint-good localization.
- Source score as a selector.
- Target-unlabeled top-k rescue.
- Target-unlabeled selector.
- Selected OACI far from the joint-good region as a global claim.
- Joint-good scarcity as the primary failure mode.

## Diagnostic-Only

- Target-unlabeled R3 confidence geometry.
- Target-grouped centering.
- Any target endpoint/joint-good oracle quantity.

Canonical C32R conclusion: joint-good checkpoints are common and locally dense, so source-side failure is not an
existence failure. Selected OACI lands near joint-good candidates in trajectory order but hits them at essentially
the trajectory-conditioned random rate. Source-side scores provide only weak enrichment, insufficient for reliable
top-1 or top-k localization. Target-unlabeled confidence geometry improves pooled/gauge separability but does not
rescue trajectory-conditioned localization, while target-grouped centering recovers pooled localization only as a
non-deployable diagnostic ceiling.
