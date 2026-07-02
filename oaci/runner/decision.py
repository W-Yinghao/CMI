"""Native K1/K2 decision for one level — the function the C8 runner calls after audit, before the final
artifact write.

K1 runs the paired grouped-permutation null on the ERM and OACI SOURCE-AUDIT frozen features (the same
fixed AuditScope support graph / fold plan / probe config the audit leakage used, so ERM and OACI are
paired by construction). K2 is the reproducible-gain decision over the multi-seed worst-domain units (a
single seed abstains). Returns the artifact decision payloads ``{level, k1_body, k1_null_arrays, k2_body}``
for ``write_artifact_tree_atomic(level_decisions=...)``. K1 is skipped (recorded, not silently) when the
audit is non-estimable or OACI is inactive (no paired representation to test)."""
from __future__ import annotations

import numpy as np

from ..decision.k1_decision import k1_decision
from ..decision.k1_permutation import compute_k1_permutation
from ..decision.k2_decision import k2_decision
from ..decision.payloads import k1_null_arrays, k1_payload, k2_payload

K1_SKIPPED = "skipped_audit_nonestimable_or_oaci_inactive"


def _empty_null() -> dict:
    z = np.zeros(0, dtype=np.float64)
    return {"null": z, "observed_delta": z.copy()}


def compute_level_decision(level, *, feat_by_method, audit_support_graph, audit_fold_plan, cfg,
                           k1_spec, k2_spec, k2_units, parallel_n_jobs: int = 1,
                           parallel_backend: str = "sequential") -> dict:
    """``feat_by_method`` = ``{method_name: FrozenFeatures}`` on the source-audit split; ``k2_units`` =
    the K2 units (see ``k2_decision``). All K1 thresholds come from ``k1_spec`` (manifest-derived)."""
    fe = feat_by_method.get("ERM")
    fo = feat_by_method.get("OACI")
    if fe is None or fo is None or audit_fold_plan is None or audit_support_graph is None:
        k1_body = {"statistic": k1_spec.statistic, "split_role": k1_spec.split_role,
                   "k1_status": K1_SKIPPED, "continue_to_k2": False,
                   "n_permutations": int(k1_spec.n_permutations)}
        k1_null = _empty_null()
    else:
        pr = compute_k1_permutation(fe, fo, audit_support_graph, audit_fold_plan, cfg,
                                    n_permutations=k1_spec.n_permutations, seed=k1_spec.seed,
                                    alpha=k1_spec.alpha, parallel_n_jobs=parallel_n_jobs,
                                    parallel_backend=parallel_backend)
        k1_body = k1_payload(pr, k1_decision(pr))
        k1_null = k1_null_arrays(pr)
    k2 = k2_decision(k2_units, endpoints=list(k2_spec.endpoints), min_seeds=k2_spec.min_seeds,
                     level_policy=k2_spec.level_policy, margins=k2_spec.margins)
    return {"level": int(level), "k1_body": k1_body, "k1_null_arrays": k1_null, "k2_body": k2_payload(k2)}
