"""Serialisable K1/K2 decision records for the artifact ``levels/<level>/decisions/`` subtree.

``k1.json`` holds the observed statistic + p-values + identity hashes + decision (the small, canonical-JSON
record); the full permutation null goes to a companion ``.npz`` (``k1_null_arrays``); ``k2.json`` holds the
K2 decision. All are pure data (JSON-native scalars / string-keyed dicts), so the canonical-JSON codec never
sees a non-str key or a NumPy scalar."""
from __future__ import annotations

import numpy as np

_K1_FIELDS = ("statistic", "split_role", "observed_delta", "p_lower", "p_upper", "p_two_sided", "alpha",
              "n_permutations", "permutation_plan_hash", "audit_support_hash", "audit_population_hash",
              "probe_config_hash", "null_quantiles")


def k1_payload(perm_result: dict, decision: dict) -> dict:
    """The canonical ``k1.json`` body: the identity + p-values + the decision, WITHOUT the raw null array."""
    body = {k: perm_result[k] for k in _K1_FIELDS}
    body["k1_status"] = decision["k1_status"]
    body["continue_to_k2"] = bool(decision["continue_to_k2"])
    return body


def k1_null_arrays(perm_result: dict) -> dict:
    """The companion ``.npz`` payload: the permutation null + the observed statistic (float64)."""
    return {"null": np.asarray(perm_result["null"], dtype=np.float64),
            "observed_delta": np.asarray([float(perm_result["observed_delta"])], dtype=np.float64)}


def k2_payload(decision: dict) -> dict:
    """The canonical ``k2.json`` body (the K2 decision dict, already JSON-native)."""
    return dict(decision)
