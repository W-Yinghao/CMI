# B2A_FAIL — immutable terminal conclusion (dev seeds 0–2)

    B2A_FAIL
    metadata routing not rejected
    frozen A* multi-action gate lacks power after transfer
    cause hypothesis:
      route-unconditioned max-null calibration (a multi-action selection gate applied to a
        metadata-routed SINGLE-action eligibility problem) + smaller-effect regime

Per pre-registration the gate / threshold / acceptance clauses are NOT re-tuned mid-B2a. Seeds 0–2
are frozen for the B2a terminal result + source-only gate design/diagnostics; they are not replayed
with any modified gate as a verdict.

## Read-only diagnostics (from the 1080 dev rows; do NOT change B2A_FAIL)
1. **metadata UNGATED** (deploy the rule table, NO gate): cov 0.19, false-adapt 0.00, ΔbAcc_diag
   +0.030, harm 0.03, top-1 0.35, regret 0.044. The metadata map is already SAFE without a gate
   (false-adapt 0) -> routing is not the failure. Its all-episode coverage caps at ~0.19 because
   only DIAG-with-known-prevalence episodes are routed to adaptation.
2. **Gate ROC ceiling** (per route): deferred to the B2b source-only null/power banks (the frozen
   per-action evidence was not stored in the 1080 rows). Reported with B2b, diagnostic-only.
3. **Oracle tie audit**: 0 exact identity-ties resolved to adaptation; identity-first == adapt-first
   non-DIAG false-adapt = 0.65 -> the oracle's false-adapt is genuine in-sample overfit, not a
   tie-break artifact.
4. **Denominators**: gate veto rate = 0.91 on {DIAG episodes where metadata proposed adaptation,
   n=34}; metadata-op veto-fail = 0.17 on {all episodes, n=180}.

## STRUCTURAL note for B2b (raised before the new run)
metadata-UNGATED all-episode coverage is already 0.19 < 0.25, and a veto-only gate can only LOWER
coverage. So no gated B2b variant can reach the 0.25 all-episode coverage clause given the frozen
coupler (DIAG ~30% of episodes; ~70% are NONE/UNSUPPORTED where abstention is CORRECT). On the
adaptation-appropriate DIAG stratum, ungated coverage is ~0.79. -> The coverage clause's denominator
(all episodes vs the DIAG/adaptation-appropriate stratum) needs to be fixed before B2b, else B2b
fails coverage structurally regardless of the route-conditioned gate.
