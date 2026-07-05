"""V2 world registry + intervention factories + expected-behaviour metadata. Thin layer over
semi_synthetic_real_latent.inject and the eraser families. Every intervention is a uniform factory
  factory(Zf, yf, dom_f, n_cls, seed) -> apply(X)
that erases the DOMAIN dom (= injected nuisance z in V2). See notes/V2_SEMI_SYNTHETIC_DESIGN.md.
"""
from __future__ import annotations
import numpy as np

from tos_cmi.score_fisher import ScoreFisherConfig
from tos_cmi.eeg.source_ood_benefit_gate import build_eraser
from tos_cmi.eeg.task_preserving_linear_erasure import tp_leace_factory, alpha_leace_factory
from tos_cmi.eeg.class_conditional_leace import fair_conditional_leace_factory

# --- expected gate behaviour per world (ground truth for scoring) ---
WORLDS = {
    "A": {"name": "beneficial_spurious_nuisance", "expect": "ACCEPT", "ground_truth": "beneficial"},
    "B": {"name": "task_entangled_nuisance", "expect": "REJECT", "ground_truth": "unsafe"},
    "C": {"name": "removable_but_useless_identity", "expect": "ABSTAIN", "ground_truth": "neutral"},
}

# controls (identity, random_k) vs principled erasers (the ones a decision should be judged on)
CONTROLS = ["identity", "random_k"]
PRINCIPLED = ["leace_baseline", "tos_vd", "rlace", "inlp", "tp_leace", "alpha_leace",
              "fair_conditional_leace_disjoint_router"]

_CFG = ScoreFisherConfig()
FACTORIES = {
    "identity": lambda Zf, yf, df, nc, sd: (lambda X: X),
    "leace_baseline": lambda Zf, yf, df, nc, sd: build_eraser(Zf, yf, df, nc, "LEACE", _CFG, sd),
    "tos_vd": lambda Zf, yf, df, nc, sd: build_eraser(Zf, yf, df, nc, "TOS_VD", _CFG, sd),
    "rlace": lambda Zf, yf, df, nc, sd: build_eraser(Zf, yf, df, nc, "RLACE", _CFG, sd),
    "inlp": lambda Zf, yf, df, nc, sd: build_eraser(Zf, yf, df, nc, "INLP", _CFG, sd),
    "tp_leace": tp_leace_factory,
    "alpha_leace": alpha_leace_factory(0.5),   # fixed soft interp for smoke; adaptive selection = round-2
    "fair_conditional_leace_disjoint_router": fair_conditional_leace_factory,
    "random_k": lambda Zf, yf, df, nc, sd: build_eraser(Zf, yf, df, nc, "random_k", _CFG, sd),
}
INTERVENTIONS = list(FACTORIES)
