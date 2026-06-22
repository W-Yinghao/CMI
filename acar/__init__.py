"""acar — Action-Conditional Counterfactual Adaptation-Risk Router.

Direction 2 (leak-proof successor to the closed A0 gate-falsification line). Estimand: paired incremental risk
    ΔR_a(B) = R_B(f_a) − R_B(f_0)
predicted from label-free paired pre→post observables, routed via a disease-stratified SUBJECT-clustered
split-conformal upper bound (finite-sample marginal coverage for exchangeable new subjects; LOCO cohort results are
descriptive robustness only). Pre-registration: notes/ACAR_FROZEN.md (v1) amended by notes/ACAR_FROZEN_v2.md (the
binding protocol).

GPU-free: runs on the archived CITA-no-LPC (erm:0) tangent-feature dumps.
"""

__all__ = ["config", "data", "actions", "features", "risk", "regressor", "scoring", "deploy", "conformal"]
