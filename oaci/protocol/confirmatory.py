"""Confirmatory runner gate. The runner REFUSES to execute unless the manifest is frozen, the
scientific code tree is clean, hashes match, the target never enters a fit statistic, the cache
fingerprint matches, and there are >= 2 active source domains.
"""
from __future__ import annotations


def confirmatory_refusals(*, manifest_frozen: bool, code_tree_clean: bool, manifest_hash_match: bool,
                          target_in_fit: bool, cache_fingerprint_match: bool,
                          n_active_source_domains: int) -> list:
    refusals = []
    if not manifest_frozen:
        refusals.append("manifest not frozen")
    if not code_tree_clean:
        refusals.append("dirty scientific code tree")
    if not manifest_hash_match:
        refusals.append("manifest/hash mismatch")
    if target_in_fit:
        refusals.append("target appears in a fit statistic")
    if not cache_fingerprint_match:
        refusals.append("cache fingerprint mismatch")
    if n_active_source_domains < 2:
        refusals.append("fewer than two active source domains")
    return refusals


def assert_confirmatory_runnable(**kw) -> None:
    r = confirmatory_refusals(**kw)
    if r:
        raise RuntimeError("confirmatory run refused: " + "; ".join(r))
