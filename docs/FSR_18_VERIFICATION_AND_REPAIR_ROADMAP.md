# FSR_18 — Verification-and-Repair Roadmap (project reframe)

**Project FSR.** PM decision (this cycle): upgrade FSR from an *audit* framework to a **verification-and-repair** framework. This document is the pre-registration for that upgrade; it changes the project's target, not any existing frozen claim. Repair may only be *claimed* after a shortcut is *verified*; otherwise the correct action is refusal / "no certified repair."

> **Phase-name mapping (to avoid a collision).** "Phase 5A" is already used and committed (submission hardening, FSR_13/14). The PM's new *stress-test* phase is recorded here as **Phase 4C** and the new *repair* phase as **Phase 4D**; the rewrite is **Phase 6**.

## Thesis (upgraded)

Old (audit): *measurable subject leakage is not necessarily functional reliance; erasable subject signal is not necessarily a target benefit.* — retained, both are Real-EEG frozen results.
New (verification + repair): *a subject/domain signal earns the name "harmful shortcut" only after functional verification; when verified, we repair it with branch-local, task-protected, harm-gated interventions; when not verified, the correct action is to refuse blind repair.* The loop is **Detect → Verify → Repair → Certify/Refuse**.

## Definition — five/six conditions for a harmful functional shortcut

A candidate signal is a harmful shortcut only if all hold (probes/KL are surrogates, not unbiased CMI):
1. **Measurable** — subject/domain decodable from the representation (L1).
2. **Task-coupled** — coupled to the task head / decision boundary / a load-bearing branch (L4).
3. **Functionally relied-upon** — intervening on it changes logits/loss/decision (L5).
4. **Target-harmful** — that reliance harms held-out subject/target risk (L6).
5. **Repairable without task collapse** — a repair improves the target consequence while preserving task signal (L6 + task-safety).

`I(Z;D\mid Y)` high is only a *candidate*; it is promoted to "harmful shortcut" only through L4 + L5 + L6 + a repair test.

## Four contributions

- **C1 — Functional-shortcut ladder** (L1–L6): the framework (already built, FSR_00/02).
- **C2 — Verification protocol** (EEG-specific): branch-local leakage probe + task-head alignment + counterfactual branch/subspace replay + target-consequence scoring + random-k / benign-leakage controls. Turns "there is leakage" into "is it functionally used as a shortcut."
- **C3 — Real-EEG evidence that leakage/erasure alone is insufficient** (retained): CIGL (leakage magnitude unreliable; alignment closer to reliance) + TOS (subject erasable, `benefit_claimable=0/40`).
- **C4 — Repair-when-verified, refuse-when-not**: branch-local / task-protected / harm-gated repair with an acceptance gate, plus a refusal path.

## Verification protocol (per branch `b ∈ {graph_z, temporal_z, spatial_z, fused_z}`)
- **L1** per-branch source-only leakage probe (domain-probe bAcc, posterior-KL surrogate, permutation null, null ratio, bootstrap CI); target subject is an unseen domain, never a closed-set probe class.
- **L4** branch load: ablation drop (`zero_*`), gate weight, task-head alignment where applicable.
- **L5** counterfactual replay: subject-subspace erase on `z_b` (source-fit) → recompose `head3(_fuse3(...))` → logit SymKL, CE/NLL, target bAcc delta; random-subspace control; exact-recompose identity check `<1e-5` (verified feasible: `head3(_fuse3(dumped branch z))==forward logits`).
- **L6** target consequence (eval-only): bAcc/NLL/ECE delta, worst-subject, task-collapse/harm flags.

## Repair candidates (Phase 4D) — small, clearly-specified only
```text
R0 identity (no repair)                         R4 counterfactual subject-subspace augmentation
R1 global LEACE                                 R5 reliance-gated branch fusion
R2 branch-local LEACE                           R6 random-k / random-subspace control (falsifier)
R3 task-protected branch erasure
```
Forbidden: CMI-loss rescue, `fbdualpc`, large hyper-parameter sweeps, new architecture search, target-label selection. Repair must be *task-coupled*, not blind erasure (the TOS lesson). R4 (counterfactual consistency: `logits(z) ≈ logits(z with subject component changed)`) is the preferred candidate — it targets L5 reliance directly, unlike a CMI penalty which only asks whether `D` is predictable from `Z`.

## Positive controls (Phase 4C) — a known-harmful shortcut must exist to test the protocol
Natural EEG may have no strongly-verifiable harmful shortcut; the protocol must still be shown to detect and repair one. Two injected controls:
- **PC1 — representation-level subject-token injection:** `z'_b = z_b + α·e_subject`, with `e_subject` spuriously correlated with the label in source but broken/absent in target. Verify L1↑, L4↑, L5↑, L6↑, and that repair recovers.
- **PC2 — subject-class prevalence stress:** induce `subject ↔ class` via source class-composition/sampling, target balanced. Tests that harm attributable to *prevalence* (not geometry) should be repaired at the reliance/sampling level, not by deleting all subject information — the prior-decoupled-TTA discipline applied to shortcuts.

## Repair acceptance criteria (all required; leakage-drop alone is NOT success)
1. verified harmful-shortcut condition met (L1+L4+L5+L6);
2. repair specificity — reduces L5 functional reliance, not merely L1 probe leakage;
3. target consequence — bAcc/worst-subject/NLL/ECE improves or at least does not harm;
4. task safety — source-val and target task do not collapse;
5. specificity control — random-k / random-subspace / task-orthogonal erasure cannot reproduce it;
6. reproducibility — full LOSO seed0, preferably seeds {0,1,2};
7. label firewall — target labels only for final evaluation (never fit/selection/probe/early-stopping).
Outcome grammar: leakage↓ only ⇒ not improvement; L5↓ but target flat ⇒ reliance reduction, not target improvement; target↑ without verification ⇒ generic adaptation, not shortcut repair.

## Phase plan
- **Phase 4B — branch-local real-EEG verification.** FBCSP-LGG ERM refit (FSR_15) → per-branch L1/L4/L5/L6 evidence chain: `branch_leakage_probe.csv`, `branch_task_coupling.csv`, `branch_reliance_replay.csv`, `branch_target_consequence.csv`. Deliverable = whether a *verified* harmful branch shortcut exists in natural EEG. (4B-0 sanity → 4B-1 seed0 full LOSO → 4B-2 seeds 1/2.)
- **Phase 4C — known-harmful stress test / positive controls** (PC1, PC2): `shortcut_stress_manifest.csv`, `shortcut_stress_results.csv`, `repair_recovery_results.csv`.
- **Phase 4D — repair candidates** (R0–R6) under the acceptance gate: repair specificity + target consequence + task safety + controls.
- **Phase 6 — paper rewrite** to verification-and-repair (see titles below).

## Two honest end-states (both publishable)
- **Repair works:** "blind leakage removal fails, but verified branch-local repair reduces functional shortcut reliance and improves target robustness."
- **Natural repair null but positive-control works:** "the framework repairs a known harmful shortcut, but natural EEG leakage in these benchmarks is not automatically harmful; deployment should verify or refuse rather than erase blindly."

## Relation to the repository's other lines
TOS ⇒ repair must be task-safe and benefit-gated, not blind erasure. OACI ⇒ repair needs an explicit verification/support/endpoint reason code (source scalars can decouple from target competence). ACAR ⇒ treat each repair as an *action* with reported harm/benefit/coverage, not an average-leakage-drop. CSC ⇒ if natural EEG can't verify a harmful shortcut from observational data, say a stronger information contract is required rather than force a repair.

## Titles (PM preference: option 1)
1. **Measurable Is Not Reliance: Verifying and Repairing Functional Shortcuts in EEG Representations**
2. From Subject Leakage to Shortcut Repair: Functional Reliance Audits for EEG Representations
3. When Subject Leakage Becomes a Shortcut: Verification and Repair in EEG Representation Learning

## Hard boundary (unchanged)
No repair is claimed before verification. CMI-control stays closed. All runs GPU-via-SLURM, ERM-only for the refit, no target-label fit. Manuscript claims update only when a phase's frozen results exist.
