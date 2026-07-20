# C26 — Predicted-Class Mix Mechanism / Counterfactual Audit (frozen C19 `664007686afb520f`)

> C25 localized the weak R3 recovery to the predicted-class-mix family (U2). C26 dissects WHAT that signal is. Q2/Q3/Q4 read-only from the C24 aggregate sidecar; Q1 split-stability + Q5 label diagnostics require a scoped re-persistence re-inference (availability-gated, NOT proxied). DIAGNOSTIC-ONLY.

- **STAGE: FINAL**
- **PRIMARY (provisional): `P1_decision_occupancy_signal`** — predicted-class mix is a split-STABLE (reliability ~0.99), target-specific DECISION-OCCUPANCY pattern that reflects the frozen model's per-class error geometry on the target (deviates from the balanced true prior; tracks per-class recall). That same distinctive pattern IS the target's identity fingerprint (P4, NN 1.0), so in isolation it carries NO standalone permutation-robust OFFSET signal (+0.003); the score-offset recovery emerges ONLY via a confidence-mix INTERACTION (P5), not as a marginal signal. Stable + error-aligned, but identity-entangled and non-standalone -- NOT claimed identity-free or deployable.
- established: **P4_identity_fingerprint_dominant, P5_confidence_mix_interaction, P1_decision_occupancy_signal, P7_label_diagnostic_boundary**

## HARD GATE — identity controls (Q3, reported before any marginal claim)

- target-id acc: predmix **+0.706** / confidence **+0.424** / full **+0.671** (chance +0.111); NN same-target rate **+1.000** (p +0.005).
- predmix recovery survives LOTO permutation: **False** → identity fingerprint dominant: **True**. predmix recovery does NOT survive the permutation control and is identity-separable -> fingerprint dominates.

## Q2 — signed vs symmetric predicted-class mix

| variant | gap closed | perm p | survives |
|---|---:|---:|:--:|
| signed (class vector) | +0.003 | +0.427 | False |
| symmetric (concentration) | -0.472 | +0.735 | False |
| signed + symmetric | +0.159 | +0.230 | False |

- signed carries: **False**; symmetric carries: **False**; signed-specific: **False**.
- **class-rotation counterfactual**: signed gap +0.003; GLOBAL rotation invariant (control) **True**; PER-TARGET scramble gap **-0.627** → class-index alignment matters: **True**. per-target class-index SCRAMBLE destroys the recovery (global rotation inert) -> the cross-target class-index alignment carries the signal (class-index-specific occupancy)

## Q4 — predicted-class mix × confidence/margin interaction

- predmix-only gap **+0.003** / confmargin-only **-0.561** / both **+0.491**; Shapley main(predmix) +0.527, main(confmargin) -0.037, interaction +1.049.
- predmix residualized on confmargin: gap **-0.090** (survives False); predmix needs confidence scaffold: **True**; interaction-dominant: **True**.

## Q1 — split-stability (P1 vs P6; from the P0-gated re-persistence re-inference, label-free)

| split | predprop reliability | split-half gap |
|---|---:|---:|
| half | +0.992 | +0.006 |
| odd_even | +0.994 | -0.001 |
| bootstrap | +0.997 | +0.009 |
- predicted-class mix is split-STABLE: **True** (reliability ~+0.997) → NOT finite-sample noise (P6 ruled out). predicted-class mix is split-stable (target decision-occupancy signal, not finite-sample noise)

## Q5 — label diagnostics (P7; labels QUARANTINED, diagnostic-only, joined post-hoc)

- predmix vs true class prior corr **n/a** (true prior constant/balanced: True); predmix vs per-class recall corr **+0.881**; mix distance from balanced prior **+0.373** → tracks target error geometry: **True**. predmix deviates from the true (balanced) class prior and tracks per-class recall -> reflects the frozen model's target decision-boundary occupancy / error geometry, not the true label prior itself

## Boundary of the claim

> DIAGNOSTIC-ONLY. Families FROZEN (no feature selection). Predicted-class mix is a stable, error-geometry-aligned decision-occupancy pattern that IS the target identity fingerprint (identity-ENTANGLED, disclosed) and contributes to the offset ONLY via the confidence-mix interaction; NOT claimed identity-free, NOT a standalone marginal signal, NOT a selector, NOT deployable. Target labels entered ONLY the quarantined post-hoc label-diagnostics, never the feature path.