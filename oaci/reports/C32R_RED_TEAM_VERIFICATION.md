# C32R - Red-Team Verification

Scope: verification of the remote-audit reinterpretation patch. This pass intentionally tried to refute the revised
taxonomy using only the committed C32 artifacts.

## Refutation Checks

1. **Should J4 remain primary? No.**
   - Primary median nearest joint-good order distance is 1 and median epoch distance is 5.
   - A global "far from joint-good region" claim conflicts with those numbers.
   - C32R removes J4 from primary taxonomy and reserves it only for future tail/margin-specific evidence.

2. **Should J8 become primary? Yes, but as base-rate/local-density, not selector skill.**
   - Joint-good rate 42.4%, random top-1 43.0%, selected OACI 44.4%, nearest order distance 1.
   - These jointly support a dense/high-base-rate landscape.
   - They do not support deployable localization.

3. **Should J2 be read as localization? No.**
   - Source top-1 enrichment is mild: 50.6% vs 43.0%.
   - Source top-3/top-5 enrichment is weak: 1.023x / 1.086x.
   - C32R labels this weak trajectory enrichment, not reliable localization.

4. **Should target-unlabeled be called a localization improvement? No.**
   - Pooled AUC improves by +0.042, but top-1 trajectory localization falls from 50.6% to 35.2%.
   - C32R renames J6 to pooled/gauge separability without top-k rescue.

5. **Does robust-margin sensitivity stay visible? Yes.**
   - At margin 0.02, J1/J5 do not fire.
   - The main report and claim-boundary audit mark primary taxonomy as frozen-margin-specific.

6. **Artifact hygiene.**
   - No training or re-inference was run for C32R.
   - No selected-checkpoint artifact was emitted.
   - No target labels enter target-unlabeled feature construction.

## Surviving Verdict

Primary C32R cases: `J8 + J1 + J2_weak + J3 + J5_margin_sensitive + J6_pooled_only + J7`.

Robust-margin cases: `J2_weak + J3 + J6_pooled_only + J7`.

The C32R headline is dense-boundary / gauge-broken localization, not endpoint scarcity, not Pareto conflict, and not a
deployable selector.
