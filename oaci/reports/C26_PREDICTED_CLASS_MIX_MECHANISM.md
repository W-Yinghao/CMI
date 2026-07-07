# C26 — Predicted-Class Mix Mechanism / Counterfactual Audit (frozen C19 `664007686afb520f`)

> C25 localized the weak R3 recovery to the predicted-class-mix family (U2). C26 dissects WHAT that signal is. Q2/Q3/Q4 read-only from the C24 aggregate sidecar; Q1 split-stability + Q5 label diagnostics require a scoped re-persistence re-inference (availability-gated, NOT proxied). DIAGNOSTIC-ONLY.

- **STAGE: read-only (Q1 split-stability + Q5 labels pending re-inference)**
- **PRIMARY (provisional): `P4_identity_fingerprint_dominant`** — predicted-class mix IN ISOLATION is a target-identity fingerprint (NN same-target rate high; its +0.003 standalone recovery FAILS the permutation control). It carries no standalone target-marginal offset signal -- C25's Shapley credit was SYNERGY allocation, not a main effect. (The FULL R3 interaction is still permutation-robust per C24; it is the isolated family that is a fingerprint.)
- established: **P4_identity_fingerprint_dominant, P5_confidence_mix_interaction**  ·  unresolved: P1_decision_occupancy_signal, P6_sample_noise_artifact, P7_label_diagnostic_boundary

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

## Q1 / Q5 — split-stability + label diagnostics (require re-persistence re-inference)

- split-stability status: **REQUIRES_REPERSISTENCE_REINFERENCE** — per-sample target logits not persisted; the C24 sidecar stores per-candidate AGGREGATES only. Split-stability needs a scoped re-PERSISTENCE re-inference (P0-validated forward, persist per-split mix summaries). NOT proxied from method-final.
- label-diagnostics status: **REQUIRES_REPERSISTENCE_REINFERENCE** — per-sample target labels not persisted; label diagnostics need the scoped re-persistence re-inference (label diagnostics QUARANTINED, never features).
- next: scoped re-persistence re-inference (P0-validated forward; persist per-split mix summaries + QUARANTINED label diagnostics) → Q1 split-stability + Q5 error-geometry alignment → finalize P1/P6/P7.

## Boundary of the claim

> DIAGNOSTIC-ONLY. Families FROZEN (no feature selection). Predmix is identity-ENTANGLED (disclosed), credited as a transferable marginal relationship only via the permutation control; NOT claimed identity-free. No selector. C26 is NOT finalized until split-stability + label diagnostics complete.