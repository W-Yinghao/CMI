"""C17 — case taxonomy + shared constants for the source-signal identifiability audit."""
from __future__ import annotations

CASE_I = "case_I_source_identifiable_accuracy"
CASE_II = "case_II_calibration_identifiable_only"
CASE_III = "case_III_multivariate_weak_identifiability"
CASE_IV = "case_IV_source_unidentifiable_competence"

CASE_INTERPRETATION = {
    CASE_I: ("source-only signals predict target-accuracy-good checkpoints -> C10's oracle used an incomplete "
             "signal; next science is a pre-registered target-free competence detector"),
    CASE_II: ("source-only signals predict target NLL/ECE/softening but NOT target accuracy -> source "
              "measurements see CALIBRATION, not target discriminative competence"),
    CASE_III: ("no scalar source signal works, but source-only combinations weakly beat permutation -> "
               "competence information exists but is not captured by simple selectors"),
    CASE_IV: ("neither univariate nor multivariate source-only descriptors recover target-good checkpoints -> "
              "OACI enters target-good states but their competence is invisible to the tested source evidence"),
}
DIAGNOSTIC_ONLY = "diagnostic_only_non_deployable"
