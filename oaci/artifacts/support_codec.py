"""Support graph + level support-state codec.

The support graph round-trips through ``build_support_graph`` and must reproduce the same
``support_hash``; the level support-state arrays are stored as data with the level's logical hashes.
"""
from __future__ import annotations

import numpy as np

from ..support_graph import build_support_graph
from .deterministic_npz import to_unicode_array

SUPPORT_KIND = "support"


def encode_support(support_state) -> tuple:
    g = support_state.support_graph
    body = {"level": int(support_state.level), "m": int(g.m),
            "domain_names": [str(x) for x in g.domain_names], "class_names": [str(x) for x in g.class_names],
            "deleted_cells": [list(map(str, c)) if isinstance(c, (list, tuple)) else str(c)
                              for c in support_state.deleted_cells],
            "observed_domain_ids": [str(x) for x in support_state.observed_domain_ids],
            "support_hash": g.support_hash(), "level_support_hash": support_state.level_support_hash}
    arrays = {"eligibility_counts": np.ascontiguousarray(g.counts),
              "cell_mass": np.ascontiguousarray(g.cell_mass, dtype=np.float64),
              "reference_prior": np.ascontiguousarray(g.reference_prior, dtype=np.float64),
              "state_eligibility_counts": np.ascontiguousarray(support_state.eligibility_counts),
              "state_cell_mass": np.ascontiguousarray(support_state.cell_mass, dtype=np.float64),
              "source_train_idx": np.ascontiguousarray(np.asarray(support_state.source_train_idx, dtype=np.int64)),
              "source_train_sample_ids": to_unicode_array(support_state.source_train_sample_ids)}
    return support_state.level_support_hash, body, arrays


def decode_support_graph(body, arrays):
    """Rebuild and validate just the support graph (the scientifically load-bearing part)."""
    g = build_support_graph(arrays["eligibility_counts"], int(body["m"]), cell_mass=arrays["cell_mass"],
                            reference_prior=arrays["reference_prior"], domain_names=list(body["domain_names"]),
                            class_names=list(body["class_names"])).validate()
    if g.support_hash() != body["support_hash"]:
        raise ValueError("decoded support graph hash != stored support hash")
    return g
