# C32 - Red-Team Verification Note

Scope: local adversarial verification of `C32_JOINT_GOOD_LOCALIZATION_AUDIT`. No sub-agent workflow was launched in
this run; the check below is a concrete refutation pass over the generated C32 artifacts and code paths.

## Checks That Changed Or Constrained The Verdict

1. **Unit-definition check caught a selected-OACI duplication bug.**
   - First implementation grouped by `(seed,target,level)`, but the C22 sidecar has three `in_regime` support-regime
     cells per C10 trajectory. That made selected OACI appear three times per unit.
   - Fix: C32 localization units are `(seed,target,level,regime)`.
   - Post-fix gate: `selected_oaci_missing_or_duplicate = 0`; C32 reports 162 trajectory-regime units.

2. **Top-k enrichment wording was downgraded.**
   - Source score top-1 enrichment is mild: 50.6% vs random 43.0% (1.178x).
   - Source top-3 is essentially random: 67.3% vs 65.8% (1.023x).
   - Source top-5 is weakly above random, not absent: 81.5% vs 75.0% (1.086x).
   - Fix: report now says "weakly above" / "weak top-k enrichment", not "not better" or "no top-k advantage".

3. **Target-unlabeled result was bounded.**
   - Target-unlabeled LOTO improves pooled AUC by +0.042 over source score, but top-1 trajectory localization is -0.154
     relative to source.
   - Verdict constrained to `J6_target_unlabeled_pooling_help_no_topk_rescue`.
   - No target-unlabeled selector claim is made.

4. **Robust-margin sensitivity weakens the primary scarcity/near-miss claims.**
   - At the primary C31 margin, joint-good rate is 42.4% and 94.4% of trajectory-regime units have joint-good.
   - At margin 0.02, joint-good rate drops to 27.8% and trajectory-regime coverage to 77.8%; primary J1/J4/J5 do not
     fire there.
   - Report discloses this sensitivity; robust cases are only J2+J3+J6+J7.

5. **No selected-checkpoint artifact / no hash leak.**
   - C32 tables expose aggregate category, distance, rank, and regret fields only.
   - `grep -R "model_hash" oaci/reports/C32_* oaci/reports/c32_tables` returns no candidate hashes.

## Surviving C32 Verdict

Primary-margin cases: `J1 + J2 + J3 + J4 + J5 + J6 + J7`.

Interpretation: joint-good checkpoints are common at the frozen primary margin, and selected OACI is often close to
one, but source-side localization is weak and gauge-broken. Target-unlabeled confidence geometry helps pooled
localization weakly but does not rescue top-k localization. Target-grouped centering repairs pooled localization
more strongly, but this is an oracle diagnostic, not deployable.

Boundaries: diagnostic-only; no selector; no selected-checkpoint artifact; no EEG training; no BNCI2014_004; no
seeds `[3,4]`; target labels never enter target-unlabeled feature construction.
