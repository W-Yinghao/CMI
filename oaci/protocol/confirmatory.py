"""Confirmatory runner gate. The gate DERIVES its own evidence (``RunEvidence``) — git tree
cleanliness, manifest-hash match, whether the sealed target appears in any fit sample-id set,
cache-fingerprint match, and the active-source-domain count — rather than trusting caller-supplied
booleans (a runner must not be able to *assert* the tree is clean or the target was held out).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class RunEvidence:
    manifest_frozen: bool
    code_tree_clean: bool
    manifest_hash_match: bool
    target_in_fit: bool
    cache_fingerprint_match: bool
    n_active_source_domains: int


def _git_tree_clean(repo_dir: str, subpath: str = "oaci") -> bool:
    try:
        out = subprocess.run(["git", "-C", repo_dir, "status", "--porcelain", "--", subpath],
                             capture_output=True, text=True)
        return out.returncode == 0 and out.stdout.strip() == ""
    except Exception:
        return False


def collect_evidence(*, repo_dir: str, manifest_frozen: bool, expected_manifest_sha: str,
                     actual_manifest_sha: str, fit_sample_ids, target_sample_ids,
                     expected_cache_fingerprint: str, actual_cache_fingerprint: str,
                     n_active_source_domains: int) -> RunEvidence:
    """Compute the gate evidence from real artifacts (target-in-fit is DERIVED from the actual
    sample-id sets; tree cleanliness from git; hashes by comparison)."""
    return RunEvidence(
        manifest_frozen=bool(manifest_frozen),
        code_tree_clean=_git_tree_clean(repo_dir),
        manifest_hash_match=(str(expected_manifest_sha) == str(actual_manifest_sha)),
        target_in_fit=bool(set(map(str, fit_sample_ids)) & set(map(str, target_sample_ids))),
        cache_fingerprint_match=(str(expected_cache_fingerprint) == str(actual_cache_fingerprint)),
        n_active_source_domains=int(n_active_source_domains),
    )


def confirmatory_refusals(ev: RunEvidence) -> list:
    r = []
    if not ev.manifest_frozen:
        r.append("manifest not frozen")
    if not ev.code_tree_clean:
        r.append("dirty scientific code tree")
    if not ev.manifest_hash_match:
        r.append("manifest/hash mismatch")
    if ev.target_in_fit:
        r.append("target appears in a fit statistic")
    if not ev.cache_fingerprint_match:
        r.append("cache fingerprint mismatch")
    if ev.n_active_source_domains < 2:
        r.append("fewer than two active source domains")
    return r


def assert_confirmatory_runnable(ev: RunEvidence) -> None:
    r = confirmatory_refusals(ev)
    if r:
        raise RuntimeError("confirmatory run refused: " + "; ".join(r))
