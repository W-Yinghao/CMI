# V2 Stage-1 / Stage-1B verdict (FROZEN)

Frozen after the full-lite (Stage 1, job 883340) and the TSMNet nuisance_fraction robustness probe
(Stage 1B, job 885017). This is the authoritative status carried into Stage 2. No World-A redesign; no gate
threshold / alpha / f_align tuning after the negative fraction probe (pre-registered).

## Verdict
```
World A / EEGNet : CLEAN PASS.
    Oracle-supported (injected-nuisance oracle target dbAcc +0.063..+0.065, LCB > +0.01) AND
    random-k does not reproduce; multi-method carriers (leace / rlace / fair_conditional);
    safe target-beneficial cells (24 Lee, 29 Cho) that the source-only gate does NOT accept
    (source-LOSO benefit LCB negative) -> ABSTAIN. 0 principled ACCEPT.

World A / TSMNet : MIXED / NOT CLEAN under the appended-nuisance construction.
    Target-beneficial RLACE cells exist (6-7) and are NOT accepted, but the injected-nuisance
    oracle is NOT target-beneficial at ANY nuisance_fraction in {0.15,0.20,0.25,0.30}
    (oracle best +0.008..+0.009, LCB never > +0.01; flat / non-increasing in m). So the World-A
    ceiling is not cleanly established for high-dimensional latents under this construction.
    Mechanism: on a 210-d high-quality latent the real-Z signal dominates; an appended orthogonal
    deployment-shift nuisance (15-30% of the latent) is too weak a target-harm for its removal
    (oracle) to be clearly beneficial.

World B/C : PASS on BOTH backbones.
    B: 0 unsafe accept (task-entangled leace/inlp/rlace REJECTed by the safety gate).
    C: 0 principled ACCEPT; high-domain-gain-useless cells abundant; domain-gain-only/always-
       domain-gain would false-accept them; our gate does not.

Cross-cutting: our source-only gate has 0 false accepts anywhere (Stage 1 + 1B); non-vacuity
    established (ACCEPT branch is live; it never fires because source-LOSO benefit is never > +0.01).
```

## Decision
No redesign of World A. Stage 2 scopes:
* **World A -> EEGNet only** (clean ceiling; robustness across source_subject_counts + seeds + folds).
* **Worlds B/C -> EEGNet + TSMNet** (refusal / no-false-accept robustness across source_subject_counts).
* **TSMNet World A -> SKIPPED in Stage 2** (Stage-1B verdict carried forward; running all folds would not
  change the mechanism, only waste compute). May appear only as a labeled NOT-CLEAN diagnostic, never in the
  Stage-2 World-A pass criterion.

Honest headline (do NOT overclaim): *"V2 confirms the source-only gate's safety/refusal behavior (B/C, both
backbones, 0 false accepts) and a clean EEGNet source-only acceptance ceiling; the TSMNet World-A ceiling is
not cleanly demonstrable under the appended-nuisance construction (a high-dimensional-latent limitation)."*
