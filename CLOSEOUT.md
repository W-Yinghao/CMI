# PROJECT CLOSEOUT — LPC-CMI / CITA / CMI-gate line: CLOSED as a FAILURE (2026-06-21)

This research line is closed. After a full pre-submission audit, **no positive method or deployment-safety claim
survived**. The voluminous experiments, records, routes and attempts are archived under
`archive/lpc-cmi-failed/`. The authoritative claim status is `notes/EVIDENCE_LEDGER.md`; the audit machinery and
result summaries are kept at top level (`scripts/`, `results/` audit dirs, `notes/` pre-registrations). The separate
`h2cmi/` project is unrelated and untouched.

## Verdict
The intended contribution — an LPC-CMI regulariser + a CMI-screened transductive method (CITA) + a source-free
CMI/uncertainty **safety gate** — does not hold up. Each pillar was tested rigorously and failed; what remains is a
negative result (a measurement→control gap), not a deployable method.

## Timeline (what was tried, in order)
1. **DualPC / LPC-CMI** — factorised P(z) + JS P(y|z); leakage I(Z;D|Y) regulariser. Multi-round development;
   accuracy parity, "leakage removal" headline.
2. **CITA** — CMI-screened nested-selected transductive matched-CORAL alignment; "+3.0 balanced acc over ERM
   cross-site" headline (later deflated to ~+1.8/+2.3, 2-seed deterministic).
3. **P1.5** — froze the deployment pipeline and audited whether LPC's leakage reduction is genuine suppression or
   representation collapse.
4. **Gate falsification (A0 → adversarial → A0′ → A0′-R → A0-PILOT)** — whether a source-free scalar (uncertainty /
   separability / density / CMI-proxy) can gate adaptation harm.
5. **Survivor audit** — evidence ledger + provenance; TUAB exposure audit; calibration deconfound; frozen survivor
   matrix; decision.

## Why each pillar fell
- **LPC leakage reduction = via representation collapse.** P1.5 (`results/p15_audit*`): at EVERY frozen λ
  (deployment mixture PD→0.1 / SCZ→0.3, fixed 0.3, fixed 0.1) the domain-leakage drop is entangled with collapse
  beyond the pre-registered utility/eff-rank gates → `DROP_LPC_COLLAPSE`. The leakage *measurement* is real
  ("extractable conditional domain information"), but it is not a deployment mechanism.
- **LPC calibration = a temperature/compression side-effect.** Deconfound (`results/calibration_deconfound/`, 130
  datasets, TUAB excluded): a single oracle temperature on ERM beats LPC NLL on **123/130**; LPC beats *raw* ERM
  (115/130) but trivial rescaling does more, with acc ≈ ERM. Not "principled" calibration.
- **CITA accuracy = the generic matched-CORAL transductive lever, not CMI/LPC.** Survivor matrix
  (`results/survivor_matrix/`): **CITA-no-LPC IS plain matched-CORAL** (serialized-equiv |Δ|=0); it equals **SPDIM**
  on accuracy (58.4 vs 58.6, CI [−2.0,+1.2]); the +2.6 over ERM is the standard transductive effect. matched-CORAL
  is better-calibrated than SPDIM's overfit TTA, but that is a matched-CORAL-vs-TTA property, not a contribution.
- **The CMI / uncertainty harm gate does not control harm.** A0 → A0′ → A0′-R → A0-PILOT
  (`results/a0_falsification`, `a0_prime_r`, `a0_pilot`): density/CMI are anti-aligned with adaptation harm
  (they reflect shift/difficulty); the "batch-rollback" positive was a **target-label-leakage** artifact (A0′-R);
  and the surviving sample-abstention candidate (post-alignment `s_sep`, harm-AUROC ~0.7) **does NOT reduce deployed
  loss** in the closed-loop pilot (retained NLL worse than random; net-protection ≈ random vs oracle +6) →
  `DIAGNOSTIC_ONLY`.

## The one surviving finding (the negative result)
**Source-free adaptation diagnostics are not deployment controllers.** Concretely, across this line:
leakage reduction can be representation collapse; shift/density detection inverts relative to adaptation harm;
outcome-conditioned evaluation can fabricate a safety signal; and a 0.7 harm-ranking AUROC need not improve
closed-loop risk. The oracle's large advantage shows harm is *in principle* selectively avoidable — the failure is
the mapping from unlabeled observables to actual loss magnitude. This is a clean **measurement→control gap**. If any
future work is pursued, it is this negative result strengthened by pre-registered cross-adapter / cross-task
replication + identifiability theory — **NOT** a seventh gate score, and NOT a revived CITA method.

## Integrity notes
- **TUAB is NOT a clean lockbox** (`notes/TUAB_EXPOSURE_AUDIT.md`): 13 TUAB result files exist in the ROOT commit
  `fb2a878` (a full LOSO method comparison) predating `TUAB_LOCKBOX.md`; the numbers fed calibration + the scorecard.
  TUAB is an already-examined benchmark; it must not be reported as a preregistered holdout.
- **Naming (binding if anything is ever written up):** "extractable conditional domain information", not "precise
  CMI"; no "CMI-screened" / "safety gate" / "protective abstention"; the deployed branch is plain source-free
  matched-CORAL + reliability gate, CMI report-only.

## Map of what's kept vs archived
- **Kept at top (the conclusion + audit evidence):** `CLOSEOUT.md`, `notes/EVIDENCE_LEDGER.md`,
  `notes/TUAB_EXPOSURE_AUDIT.md`, the frozen pre-registrations (`notes/FREEZE_PROTOCOL.md`, `A0_*_FROZEN.md`,
  `A0_PRIME_R_PROTOCOL_REPAIR.md`), the audit scripts (`scripts/{a0_*,p15_*,survivor_matrix,calibration_deconfound,
  r12_decomp,write_freeze_a1,test_*}.py`), the audit result summaries (`results/{p15_audit*,a0_*,
  calibration_deconfound,survivor_matrix,freeze_a1}/`), the method code (`cmi/`), `README.md`.
- **Archived (`archive/lpc-cmi-failed/`):** all other experiment outputs (`dualpc*`, `r*_dualpc2`, `feat_dump_*`,
  `r12det_*`, `*TUAB*`, all `*.preds.npz`, root result JSONs), the superseded notes/records, the training/runner
  scripts, `analysis/`, `synthetic/`, and the old planning docs.
- **Untouched:** `h2cmi/` (a separate, active project).

Everything is preserved in git history and on `origin/exp/lpc-cmi`; archiving moves files, it deletes nothing.
