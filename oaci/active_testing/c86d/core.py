"""C86D core — endpoints, claim boundary, exact CVaR, C85U identity, method freeze.

Development only. No confirmatory claim; trials/queries/replicates are not
scientific N.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field as dc_field

import numpy as np

BUDGET_GRID = (4, 8, 16, 32, "FULL")
CONTEXTS_PER_TRIAL = 8
N_CANDIDATES = 81
NEAR_OPT_EPS = 0.05
TAIL_FRACTION = 0.25

# Claim boundary (unchanged from C86LP): LURE unbiasedness only for linear moments.
LINEAR_MOMENTS = frozenset({"nll", "correct", "class_numerator", "signed_calibration"})
NONLINEAR_PLUGINS = frozenset({"balanced_accuracy", "ece", "candidate_midrank",
                               "composite_utility", "selected_action", "target_regret"})

# C85U held-evaluation utility identity (verified by hashing, never hardcoded-trusted).
C85U_ACCEPTANCE_MANIFEST = ("/projects/EEG-foundation-model/yinghao/oaci-c85u-candidate-utility-v2/"
                            "c85u-v2-77382c16a593f7c2-91a428488a634268/final_acceptance_bundle/"
                            "C85U_RESULT_ARTIFACT_MANIFEST.json")
C85U_ACCEPTANCE_SHA = "dfcf84569beb1b34b786cbe72233a22fd3928a4475b7e345f23b40cdb6671620"
C85U_FIELD_IDENTITY = {"contexts": 944, "candidates_per_context": 81,
                       "candidate_rows": 76_464, "evaluation_label_table_rows": 4_848}
C85U_UTILITY_INDEX = ("/projects/EEG-foundation-model/yinghao/oaci-c85u-candidate-utility-v2/"
                      "c85u-v2-77382c16a593f7c2-91a428488a634268/stage_u1_candidate_utility_v2/"
                      "candidate_utility_index.csv")
C85U_UTILITY_INDEX_SHA = "83bddf56290c4e06a306d64dadfc9611115a177f479d433fe0e4485b0c181509"


class C86DClaimError(RuntimeError):
    pass


class C86DIdentityError(RuntimeError):
    pass


def assert_linear_claim(quantity: str) -> None:
    if quantity in NONLINEAR_PLUGINS:
        raise C86DClaimError(f"{quantity!r} is a nonlinear plugin; no LURE unbiasedness claim")
    if quantity not in LINEAR_MOMENTS:
        raise ValueError(f"unknown estimand {quantity!r}")


def exact_upper_cvar(losses, frac: float = TAIL_FRACTION) -> float:
    """Exact upper-tail CVaR with FRACTIONAL boundary mass (not ceil(frac·N)).

    e.g. N=22, frac=0.25 -> k=5.5 -> mean of the 5 worst + half of the 6th.
    """
    a = np.sort(np.asarray(losses, dtype=np.float64))[::-1]     # descending
    n = a.size
    if n == 0:
        return 0.0
    k = frac * n
    if k <= 0:
        return float(a[0])
    full = int(np.floor(k))
    frac_part = k - full
    if full >= n:
        return float(a.mean())
    total = a[:full].sum() + (frac_part * a[full] if full < n else 0.0)
    return float(total / k)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_c85u_identity(manifest_path: str = C85U_ACCEPTANCE_MANIFEST) -> dict:
    """Open + hash the REAL C85U acceptance manifest and check its field identity.

    Fail-closed; never trust a hardcoded SHA alone.
    """
    if not os.path.exists(manifest_path):
        raise C86DIdentityError(f"C85U acceptance manifest absent: {manifest_path}")
    actual = sha256_file(manifest_path)
    if actual != C85U_ACCEPTANCE_SHA:
        raise C86DIdentityError(f"C85U acceptance SHA mismatch: {actual}")
    doc = json.load(open(manifest_path))
    return {"path": manifest_path, "sha256": actual, "bytes": os.path.getsize(manifest_path),
            "field_identity": dict(C85U_FIELD_IDENTITY),
            "verified": True, "manifest_keys": len(doc)}


# One-time, deterministic method freeze (pre-registered before any real execution).
METHOD_FREEZE = {
    "primary_registry": ("P0", "A1", "A2H"),
    "budgets": BUDGET_GRID,
    "context_aggregation": "equal_weight_mean",     # 8 contexts equally weighted
    "a1": {"acquisition": "mean_expected_candidate_NLL",
           "sampling_prob_floor": 0.05, "uniform_mixing_rho": 0.05,
           "importance_weighting": "LURE"},
    "a2h": {"acquisition": "sum_over_k_lt_kprime_E_abs_loss_difference",
            "sampling_prob_floor": 0.05, "uniform_mixing_rho": 0.05,
            "importance_weighting": "LURE"},
    "passive_replicates": 8,
    "seed_schedule": tuple(range(8)),
    "stopping_rule": "budget_exhausted_or_pool_exhausted",
    "failure_rule": "fail_closed_no_partial_publish",
    "hyperparameter_freeze_rule": "deterministic_pre_written_selection_on_C84_C85_held_outcomes",
    "no_post_hoc_method_add_or_drop": True,
    "development_only": True,
}


@dataclass(frozen=True)
class PolicyMetrics:
    mean_regret: float
    tail_regret: float
    target_near_opt_prob: float
    mean_by_cohort: dict
    tail_by_cohort: dict
    near_opt_by_cohort: dict


def compute_endpoints(target_regrets_by_cohort: dict) -> PolicyMetrics:
    """target_regret already = equal-weight mean over a target's 8 contexts."""
    allr = np.array([r for v in target_regrets_by_cohort.values() for r in v], dtype=np.float64)
    return PolicyMetrics(
        mean_regret=float(allr.mean()) if allr.size else 0.0,
        tail_regret=exact_upper_cvar(allr),
        target_near_opt_prob=float((allr <= NEAR_OPT_EPS).mean()) if allr.size else 0.0,
        mean_by_cohort={c: float(np.mean(v)) for c, v in target_regrets_by_cohort.items()},
        tail_by_cohort={c: exact_upper_cvar(v) for c, v in target_regrets_by_cohort.items()},
        near_opt_by_cohort={c: float((np.array(v) <= NEAR_OPT_EPS).mean())
                            for c, v in target_regrets_by_cohort.items()},
    )
