"""K1/K2 specs read from the strict manifest blocks.

The manifest validates STRUCTURE (``manifest_v2.validate_ranges``); here we validate the EXECUTABLE contract
— the exact ``statistic`` / ``permutation_scheme`` / ``split_role`` the decision code implements — so a
mis-named or unimplemented scheme fails loudly BEFORE any compute, and no paper-level threshold is hard-coded
in code (they all come from the manifest)."""
from __future__ import annotations

from dataclasses import dataclass

_K1_STATISTIC = "grouped_max_probe_extractable_LQ_ov_OACI_minus_ERM"
_K1_SCHEME = "paired_swap_within_y_recording_group"
_K1_SPLIT = "source_audit"
_K2_ENDPOINTS = ("worst_domain_bacc", "worst_domain_nll")


@dataclass(frozen=True)
class K1Spec:
    statistic: str
    split_role: str
    permutation_scheme: str
    n_permutations: int
    alpha: float
    decision_rule: str
    seed: int


@dataclass(frozen=True)
class K2Spec:
    endpoints: tuple
    min_seeds: int
    level_policy: str
    margins: dict            # {endpoint: margin}
    decision_rule: str


def k1_spec_from_manifest(manifest) -> K1Spec:
    k = manifest.k1
    if k is None:
        raise ValueError("manifest has no k1 block")
    if k.statistic != _K1_STATISTIC:
        raise ValueError(f"K1 statistic {k.statistic!r} != implemented {_K1_STATISTIC!r}")
    if k.permutation_scheme != _K1_SCHEME:
        raise ValueError(f"K1 permutation_scheme {k.permutation_scheme!r} != implemented {_K1_SCHEME!r}")
    if k.split_role != _K1_SPLIT:
        raise ValueError(f"K1 split_role {k.split_role!r} != implemented {_K1_SPLIT!r}")
    return K1Spec(statistic=k.statistic, split_role=k.split_role, permutation_scheme=k.permutation_scheme,
                  n_permutations=int(k.n_permutations), alpha=float(k.alpha), decision_rule=str(k.decision_rule),
                  seed=int(k.seed))


def k2_spec_from_manifest(manifest) -> K2Spec:
    k = manifest.k2
    if k is None:
        raise ValueError("manifest has no k2 block")
    eps = tuple(k.endpoints or ())
    if not eps or any(e not in _K2_ENDPOINTS for e in eps):
        raise ValueError(f"K2 endpoints {eps} must be a non-empty subset of {_K2_ENDPOINTS}")
    margins = {"worst_domain_bacc": float(k.bacc_margin), "worst_domain_nll": float(k.nll_margin)}
    return K2Spec(endpoints=eps, min_seeds=int(k.min_seeds), level_policy=str(k.level_policy),
                  margins=margins, decision_rule=str(k.decision_rule))
