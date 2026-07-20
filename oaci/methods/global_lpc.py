"""global-LPC: full-domain posterior aligned to the cell-mass-smoothed prior

    π_y^α(d) = (M_{d,y} + α) / (Σ_{d'∈D0} M_{d',y} + α|D0|),   α > 0,

over the FIXED level-0 domain universe. A deleted cell has zero observed rows but POSITIVE prior
mass (the +α), so it is represented in π_y without any fabricated row.
"""
from __future__ import annotations

import numpy as np

from .posterior import PosteriorObjective


class GlobalLPCObjective(PosteriorObjective):
    def __init__(self, level0_domains, cell_mass, class_mass, reference_prior, present_classes,
                 alpha: float, active=True, inactive_reason=None, hidden: int = 0):
        if alpha <= 0:
            raise ValueError("global_lpc laplace smoothing alpha must be > 0")
        self.cell_mass = np.asarray(cell_mass, dtype=np.float64)     # [D0, C], level-0 universe
        self.alpha = float(alpha)
        super().__init__("global_lpc", level0_domains, class_mass, reference_prior, present_classes,
                         active=active, inactive_reason=inactive_reason, hidden=hidden)

    def _prior_vector(self, yy: int) -> np.ndarray:
        m = np.array([self.cell_mass[d, yy] for d in self.level0], dtype=np.float64)
        pri = (m + self.alpha) / (m.sum() + self.alpha * len(self.level0))
        return pri
