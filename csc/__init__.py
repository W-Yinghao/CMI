"""
csc — Falsifiable Concept-Shift Certificates with Abstention.

An isolated research package (it does NOT import or mutate the AAAI `cmi/` or `h2cmi/`
packages). It asks a deliberately narrow, falsifiable question: *when* can an unlabeled
target tell you that the label-generating rule P(Y|Z) has changed, and when must you
refuse to answer?

The output is a three-state certificate -- COVARIATE_ADAPTABLE / CONCEPT_SUSPECT /
UNIDENTIFIABLE -- not a binary detector. See THEORY.md and PREREGISTRATION.md.
"""

__all__ = ["sim", "certificate"]
