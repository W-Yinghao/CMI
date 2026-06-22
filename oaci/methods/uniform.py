"""uniform: the full-domain posterior baseline whose target is the UNIFORM domain prior
``π_y(d) = 1/|D0|`` over the fixed level-0 universe. Same posterior architecture, critic loss and
full-domain alignment plan as global-LPC — so the only difference is the target prior, and the two
methods must share an alignment-plan hash.
"""
from __future__ import annotations

import numpy as np

from .posterior import PosteriorObjective


class UniformObjective(PosteriorObjective):
    def __init__(self, level0_domains, class_mass, reference_prior, present_classes,
                 active=True, inactive_reason=None, hidden: int = 0):
        super().__init__("uniform", level0_domains, class_mass, reference_prior, present_classes,
                         active=active, inactive_reason=inactive_reason, hidden=hidden)

    def _prior_vector(self, yy: int) -> np.ndarray:
        n = len(self.level0)
        return np.full(n, 1.0 / n, dtype=np.float64)
