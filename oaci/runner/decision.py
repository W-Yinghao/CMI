"""Native K1/K2 decision for one level — called by ``finalize`` after the source-audit leakage (K1) and
after prediction metrics (K2), before the artifact write.

K1 runs the paired grouped-permutation null on the SELECTED ERM and OACI SOURCE-AUDIT features retained by
``run_post_selection_audit`` (same fixed AuditScope support graph / fold plan / probe config, so ERM and OACI
are paired). It reuses those features (no re-forward), never touches target, and cannot run before the
selection lock (finalize is post-AUDIT). When OACI selected the SAME checkpoint as ERM the features are
identical, so K1 is a degenerate zero (recorded, not run). K2 is the single-seed abstain here; the real
multi-seed K2 is decided by the later aggregation. Returns the artifact payloads + the binding hashes that
``level_result_hash`` folds in when decisions are enabled."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecisionContext:
    """Threaded (NOT via execution_config_hash) so enabling decisions never changes the training/audit
    identity. ``enabled`` is set true ONLY for C8 materialized runs; otherwise finalize skips K1/K2 and the
    level hash is byte-identical to a legacy (no-decision) run."""
    enabled: bool
    k1_spec: object = None
    k2_spec: object = None
    parallel_n_jobs: int = 1
    parallel_backend: str = "sequential"

from ..artifacts.canonical_json import canonical_json_hash
from ..decision.k1_decision import k1_decision
from ..decision.k1_permutation import compute_k1_permutation
from ..decision.k2_decision import k2_decision
from ..decision.payloads import k1_null_arrays, k1_payload, k2_payload

K1_SKIPPED = "skipped_audit_nonestimable_or_oaci_inactive"
K1_DEGENERATE = "estimable_degenerate_same_checkpoint"


def _empty_null() -> dict:
    z = np.zeros(0, dtype=np.float64)
    return {"null": z, "observed_delta": z.copy()}


def _null_hash(arrays: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(np.asarray(arrays[k], dtype=np.float64))
        h.update(k.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def decision_binding_hashes(decision: dict) -> dict:
    """The hashes ``level_result_hash`` binds when decisions are enabled (empty ⇒ nothing bound ⇒ legacy
    hash unchanged)."""
    return {"k1": canonical_json_hash(decision["k1_body"]),
            "k1_null": _null_hash(decision["k1_null_arrays"]),
            "k2": canonical_json_hash(decision["k2_body"])}


def compute_level_decision(level, *, feat_by_method, audit_support_graph, audit_fold_plan, cfg,
                           k1_spec, k2_spec, k2_units, model_hash_by_method=None,
                           parallel_n_jobs: int = 1, parallel_backend: str = "sequential") -> dict:
    fe, fo = feat_by_method.get("ERM"), feat_by_method.get("OACI")
    he = (model_hash_by_method or {}).get("ERM")
    ho = (model_hash_by_method or {}).get("OACI")
    base_k1 = {"statistic": k1_spec.statistic, "split_role": k1_spec.split_role,
               "n_permutations": int(k1_spec.n_permutations), "alpha": float(k1_spec.alpha)}
    if fe is None or fo is None or audit_fold_plan is None or audit_support_graph is None:
        k1_body = {**base_k1, "k1_status": K1_SKIPPED, "continue_to_k2": False}
        k1_null = _empty_null()
    elif he is not None and ho is not None and he == ho:
        # OACI selected the ERM checkpoint -> identical audit features -> the null is degenerate (all Δ=0).
        k1_body = {**base_k1, "observed_delta": 0.0, "p_lower": 1.0, "p_two_sided": 1.0,
                   "same_checkpoint": True, "k1_status": K1_DEGENERATE, "continue_to_k2": False}
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
    k2_body = k2_payload(k2)
    k2_body["available_seeds"] = k2.get("n_seeds", 0)
    k2_body["required_min_seeds"] = int(k2_spec.min_seeds)
    return {"level": int(level), "k1_body": k1_body, "k1_null_arrays": k1_null, "k2_body": k2_body}


def build_level_decision(level, audit_intermediate, fold_scope, execution_cfg, *, k1_spec, k2_spec,
                         k2_units, parallel_n_jobs: int = 1, parallel_backend: str = "sequential") -> dict:
    """Extract the paired ERM/OACI source-audit features + the fixed audit scope from the AUDIT intermediate
    and compute the level decision. K1 runs only on the estimable audit split (features present)."""
    af = audit_intermediate.audit_features                       # {method: FeatureArtifact}
    feat_by_method = {n: a.features for n, a in af.items()}
    model_hash_by_method = {n: a.model_hash for n, a in af.items()}
    au = fold_scope.source_audit
    return compute_level_decision(
        level, feat_by_method=feat_by_method, audit_support_graph=(au.support_graph if af else None),
        audit_fold_plan=(au.fold_plan if af else None), cfg=execution_cfg.critic, k1_spec=k1_spec,
        k2_spec=k2_spec, k2_units=k2_units, model_hash_by_method=model_hash_by_method,
        parallel_n_jobs=parallel_n_jobs, parallel_backend=parallel_backend)
