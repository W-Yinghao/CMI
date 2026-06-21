"""
csc — Partial-Identification Concept-Shift Certificates with Abstention.

An isolated research package (it does NOT import or mutate the AAAI `cmi/`, `h2cmi/`, or
`oaci/` packages). It asks a deliberately narrow, falsifiable question: *when* can an
unlabeled target reveal that the label-generating rule P(Y|Z) has changed, and when must
it refuse to answer?

The output is a three-state partial-identification certificate -- COVARIATE_COMPATIBLE /
CONCEPT_SUSPECT / UNIDENTIFIABLE -- not a binary detector. See THEORY.md and
PREREGISTRATION.md.
"""

__all__ = ["sim", "certificate", "calibration"]
