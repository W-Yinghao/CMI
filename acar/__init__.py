"""acar — Action-Conditional Counterfactual Adaptation-Risk Router.

Direction 2 (leak-proof successor to the closed A0 gate-falsification line). Estimand: paired incremental risk
    ΔR_a(B) = R_B(f_a) − R_B(f_0)
predicted from label-free paired pre→post observables, routed via a leave-one-source-cohort-out conformal upper
bound. Pre-registration & frozen go/no-go: notes/ACAR_FROZEN.md.

GPU-free: runs on the archived CITA-no-LPC (erm:0) tangent-feature dumps.
"""

__all__ = ["config", "data", "actions", "features", "risk", "regressor", "conformal"]
