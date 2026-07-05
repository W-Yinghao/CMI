"""V2 world registry + intervention factories + expected-behaviour metadata (reframed: source-only acceptance
CEILING / non-identifiability -- NO world expects ACCEPT). Thin layer over semi_synthetic_real_latent.inject
and the eraser families. Every deployable intervention is a uniform factory
  factory(Zf, yf, dom_f, n_cls, seed) -> apply(X)
that erases the DOMAIN dom (= injected nuisance z). See notes/V2_SEMI_SYNTHETIC_DESIGN.md.
"""
from __future__ import annotations
import numpy as np

from tos_cmi.score_fisher import ScoreFisherConfig
from tos_cmi.eeg.source_ood_benefit_gate import build_eraser
from tos_cmi.eeg.task_preserving_linear_erasure import tp_leace_factory, alpha_leace_factory
from tos_cmi.eeg.class_conditional_leace import fair_conditional_leace_factory

# --- expected gate behaviour per world (ground truth for scoring; NONE expects ACCEPT -- the ceiling) ---
WORLDS = {
    "A": {"name": "target_beneficial_but_source_uncertifiable", "expect": "REJECT/ABSTAIN (not ACCEPT)",
          "ground_truth": "target_beneficial_source_uncertifiable", "acceptance_expected": False,
          "diagnostic_oracle_expected": "positive_target_gain"},
    "B": {"name": "task_entangled_unsafe", "expect": "REJECT", "ground_truth": "unsafe",
          "acceptance_expected": False},
    "C": {"name": "removable_but_useless_identity", "expect": "REJECT/ABSTAIN", "ground_truth": "neutral",
          "acceptance_expected": False},
}

CONTROLS = ["identity", "random_k"]
PRINCIPLED = ["leace_baseline", "tos_vd", "rlace", "inlp", "tp_leace", "alpha_leace",
              "fair_conditional_leace_disjoint_router"]
DIAGNOSTIC = ["oracle_nuisance_eraser_DIAGNOSTIC_ONLY"]   # uses ground-truth injected dims / labels; NOT deployable

_CFG = ScoreFisherConfig()
# Deployable factories only. The oracle eraser needs m (the injected-block width), so it is built per-cell in
# the driver via oracle_nuisance_eraser_factory(m) -- it is NOT in this deployable dict.
FACTORIES = {
    "identity": lambda Zf, yf, df, nc, sd: (lambda X: X),
    "leace_baseline": lambda Zf, yf, df, nc, sd: build_eraser(Zf, yf, df, nc, "LEACE", _CFG, sd),
    "tos_vd": lambda Zf, yf, df, nc, sd: build_eraser(Zf, yf, df, nc, "TOS_VD", _CFG, sd),
    "rlace": lambda Zf, yf, df, nc, sd: build_eraser(Zf, yf, df, nc, "RLACE", _CFG, sd),
    "inlp": lambda Zf, yf, df, nc, sd: build_eraser(Zf, yf, df, nc, "INLP", _CFG, sd),
    "tp_leace": tp_leace_factory,
    "alpha_leace": alpha_leace_factory(0.5),
    "fair_conditional_leace_disjoint_router": fair_conditional_leace_factory,
    "random_k": lambda Zf, yf, df, nc, sd: build_eraser(Zf, yf, df, nc, "random_k", _CFG, sd),
}
DEPLOYABLE = list(FACTORIES)
INTERVENTIONS = DEPLOYABLE + DIAGNOSTIC


def oracle_nuisance_eraser_factory(m):
    """DIAGNOSTIC (NOT deployable): removes the injected nuisance block exactly by zeroing its m appended dims.
    Uses ground-truth knowledge of WHICH dims are the injected nuisance -- unavailable in real deployment. Shows
    that a target-beneficial erasure EXISTS; its source-LOSO benefit is still ~0 (the ceiling)."""
    def factory(Zf, yf, df, nc, sd):
        def apply(X):
            Y = np.asarray(X, float).copy()
            if m > 0:
                Y[:, -m:] = 0.0
            return Y
        return apply
    return factory
