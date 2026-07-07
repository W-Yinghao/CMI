"""Task-Orthogonal Selective CMI (TOS-CMI).

Selective conditional invariance: instead of erasing all conditional domain information
(global LPC, which collapses TSMNet), estimate a label Fisher F_Y and a class-conditional
domain Fisher F_{D|Y}, take the generalized spectrum F_{D|Y} v = rho (F_Y + eta I) v, and
apply the leakage penalty I(Z;D|Y) ONLY on the domain-rich / label-light subspace it
selects -- refusing to delete (identity) when no such subspace is risk-feasible.

Public surface:
    config         : FisherConfig, SubspaceConfig, PenaltyConfig, TOSConfig
    fisher         : label_fisher, conditional_domain_fisher, fisher_pair
    subspace       : solve_generalized, select_nuisance, SubspaceSelector, SubspaceReport
    selective_cmi  : SelectivePenalty, ConditionalDomainCritic, label_prior
    data.synthetic : SynthSpec, make
    eval.proposition : bayes_risk_check
    eval.stability   : principal_angles, subspace_overlap, selection_stability
"""
from .config import FisherConfig, SubspaceConfig, PenaltyConfig, TOSConfig
from .fisher import label_fisher, conditional_domain_fisher, fisher_pair
from .subspace import solve_generalized, select_nuisance, SubspaceSelector, SubspaceReport
from .selective_cmi import SelectivePenalty, ConditionalDomainCritic, label_prior

__all__ = [
    "FisherConfig", "SubspaceConfig", "PenaltyConfig", "TOSConfig",
    "label_fisher", "conditional_domain_fisher", "fisher_pair",
    "solve_generalized", "select_nuisance", "SubspaceSelector", "SubspaceReport",
    "SelectivePenalty", "ConditionalDomainCritic", "label_prior",
]
