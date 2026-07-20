# Problem setting and the falsification battery

## Problem setting: support mismatch and conditional leakage

Many domain-generalization (DG) penalties are justified by a **source-side control signal**: reduced
conditional-domain "leakage" (invariance), improved source robustness, or better held-out source endpoints.
The premise is that improving such a signal on the source domains will improve worst-domain performance on an
unseen target domain. We take that premise as a *hypothesis to be falsified*, not assumed.

A conditional-invariance objective aims to make a representation `Z` uninformative about the domain `D` given
the label `Y`. Under **domain–class support mismatch** — where some `(domain, class)` cells have little or no
source support — the population quantity `p(z | y, d)` is not defined for unsupported cells, so a diagnostic
that averages a per-cell alignment term over *all* cells (implicitly smoothing over cells with no support) is
not well-identified. We therefore restrict measurement to **estimable cells** (a support graph separates
estimable from unsupported cells; unsupported cells are flagged, never smoothed), and we measure a
**probe-extractable** quantity: `L_Q^ov`, the extractable conditional-domain information recovered by a
grouped cross-fit domain probe on frozen `Z`, restricted to estimable cells and evaluated against a
recording-grouped permutation null. Crucially, the audit is performed on a **held-out source split**
(`source_audit`); the target is never read.

*Scope note (carried from the adversarial review).* On the balanced 4-class BNCI2014-001 data used here, we do
not separately quantify how many cells are unsupported; "ill-posed under support mismatch" is the *motivation*
for the support-aware construction, and BNCI2014-001 alone does not establish a support-mismatch regime. The
support-aware diagnostic is what we measure with; its distinctive value under genuine mismatch is future work.

## The falsification battery

The battery is a fixed sequence of six gates. Each gate is a pure function of committed, deep-verified
artifacts; the battery reads no target information for any selection decision.

- **G0 — Integrity.** Are the artifacts deep-verified, target-isolated (`target_fit_ids = ∅`), and is the
  checkpoint replay identity-exact? A failure here means the downstream gates would run on untrustworthy
  evidence. *(Outcome: `integrity_ok`.)*
- **G1 — Selection→audit optimism.** Does a selection-time leakage reduction survive at the held-out source
  audit? We compare the per-fold change in selection leakage against the change in audit leakage.
- **G2 — Held-out leakage (K1).** Does the held-out audit leakage reduction survive multiplicity? K1 is a
  recording-grouped 2000-permutation null per fold; the sweep-level line applies Bonferroni and
  Benjamini–Hochberg control. A *weak nominal, non-multiplicity-stable* signal is **not** a success.
- **G3 — Endpoint transfer (K2).** Does any leakage signal convert to a reproducible worst-domain endpoint
  gain over ERM, on both `worst_domain_bacc` and `worst_domain_nll`, at every (seed, level) unit?
- **G4 — Source-audit oracle replay.** Replaying selectors over the method's own risk-feasible trajectory, can
  a **non-deployable source-audit oracle** — which may read the held-out `source_audit` split but never the
  target — identify a gain-reproducing checkpoint? This removes "bad selector" and "bad selection split" as
  explanations. The oracle is a diagnostic upper bound on *source-identifiable* rescue, not a target oracle.
- **G5 — Source→target transfer.** For a source-side *endpoint* objective, does improving the source
  worst-domain endpoint transfer to the target, stay flat, or **anti-transfer** (source improves, target
  worsens)? We report an anti-transfer index (fraction of active cells with source-improvement and
  target-harm) and a source→target instability score.

The battery verdict is `control_hypothesis_supported` only if the endpoint gate certifies a reproducible gain
*and* the oracle can rescue; otherwise it lists the specific falsification reasons
(`falsified_by_no_endpoint_transfer`, `falsified_by_oracle_failure`, `falsified_by_source_target_antitransfer`,
`falsified_by_selection_optimism`). We emphasize that the battery has, to date, only ever returned
`falsified`; a **positive control** — an ERM-beating method certified by the same gates — is not yet available,
so the battery's *discriminative* validity (its ability to certify a genuinely transferring method rather than
only flag failures) remains future work.

## Experimental protocol

- **Data / task.** BNCI2014-001, 9 subjects, 4-class motor imagery, LOSO (each subject held out in turn).
- **Roles.** `source_train` / `source_guard` / `source_audit` / `target_audit`, with `target_audit` used for
  evaluation only. Provenance tracking asserts `target_fit_ids = ∅` (the target never enters any fit).
- **Backbone.** ShallowConvNet; a shared Stage-1 ERM checkpoint per fold, with Stage-2 method objectives
  starting from it under a risk-feasibility constraint `R_src ≤ R_ERM + ε`.
- **Methods.** ERM (baseline), OACI (conditional-domain leakage control), `global_lpc` / `uniform` (posterior
  / uniform alignment baselines), and SRC (source-robust endpoint control). OACI/`global_lpc`/`uniform` were
  evaluated in the confirmatory K1/K2 run (seeds [0,1,2]); SRC in a dedicated one-fold pilot and a 3-target ×
  2-temperature stress replication (seed 0).
- **Reproducibility.** Every fold artifact is deep-verified; the counterfactual selector replay reproduces the
  stored per-checkpoint predictions with **0 argmax flips** (byte-hash where the GPU arch matches the original
  node, numeric to ~10⁻¹⁵ otherwise). All decision numbers are pulled from committed artifacts, not
  transcribed.
- **Pre-registration and scope.** K1/K2 thresholds and the SRC selector guards are fixed in a manifest. Under
  a pre-registered pause, we do **not** run additional seeds or a second dataset (BNCI2014-004) as part of this
  work; the current claims are scoped to BNCI2014-001 / ShallowConvNet accordingly.
