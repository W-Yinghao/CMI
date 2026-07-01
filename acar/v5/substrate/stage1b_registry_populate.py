"""ACAR V5 Stage-1B registry population (pure/stdlib). Registers EXACTLY the 30 built fold substrates into a SubstrateRegistry,
once each, no silent overwrite, using the writer-computed hash set + complete meta. Fail-closed on count/coverage/dup.
"""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5.substrate import stage1b_authorization as SA


class Stage1bRegistryPopulateError(RuntimeError):
    """Raised when registry population is incomplete / duplicated / not exactly the 30 canonical refs."""


def populate_registry(registry, artifacts, *, git_commit, env_lock_sha256,
                      channel_montage, sampling_rate, windowing_config, extra_meta_by_ref=None):
    """Register each built artifact into `registry`. `artifacts` = {ref: artifact_manifest}. Returns the number registered (30).
    meta.cohort_inclusion_list is derived per disease from the frozen DEV cohorts; random_seed from the artifact's seed.
    `extra_meta_by_ref` (optional) merges per-ref non-hash meta (e.g. training_config_sha256) — it can NEVER override the six
    registry hash fields (they live in `hashes`, not `meta`)."""
    if set(artifacts) != set(SA.CANONICAL_FOLD_REFS):
        missing = sorted(set(SA.CANONICAL_FOLD_REFS) - set(artifacts))
        extra = sorted(set(artifacts) - set(SA.CANONICAL_FOLD_REFS))
        raise Stage1bRegistryPopulateError(f"artifacts must be exactly the 30 canonical refs (missing {missing[:3]}, extra {extra[:3]})")
    extra_meta_by_ref = extra_meta_by_ref or {}
    added, n = [], 0
    try:                                                      # ATOMIC: a mid-loop register failure rolls back → all-or-none
        for ref in sorted(artifacts):
            art = artifacts[ref]
            disease, fold, seed = art["disease"], int(art["fold"]), int(art["seed"])
            hashes = {h: art[h] for h in P.REGISTRY_HASH_FIELDS}
            meta = {"channel_montage": channel_montage, "sampling_rate": sampling_rate, "windowing_config": windowing_config,
                    "cohort_inclusion_list": ",".join(sorted(P.DEV_COHORTS[disease])), "random_seed": seed,
                    "git_commit": git_commit, "env_lock_sha256": env_lock_sha256}
            meta.update(extra_meta_by_ref.get(ref, {}))       # additive non-hash meta (never touches `hashes`)
            added.append(registry.register(disease, fold, seed, hashes=hashes, meta=meta))   # raises on dup/bad hash/bad meta
            n += 1
        if n != len(SA.CANONICAL_FOLD_REFS):
            raise Stage1bRegistryPopulateError(f"registered {n} != {len(SA.CANONICAL_FOLD_REFS)}")
    except Exception:
        registry._rollback(added)                             # leave the registry exactly as we found it (no partial population)
        raise
    return n
