"""ACAR V5 Stage-2A candidate-manifest BINDING (pure/stdlib; NO I/O, NO data, NO labels).

Stage-2 selection may ONLY consider the frozen 22-row candidate manifest. That manifest is defined once in `protocol.py`
(the SINGLE SOURCE OF TRUTH, self-checked at import). This module deliberately does NOT re-define it — a second definition
could silently drift — it BINDS to the frozen one and RE-VALIDATES, fail-closed, every invariant a Stage-2 runner relies on:

  * exactly 22 rows, family counts {P1:4, P2:4, P3:6, P4:2, P5:6};
  * unique ids, identical to `protocol.CANDIDATE_IDS`, in canonical P1..P5 ascending-ordinal order;
  * every candidate `disease_scope == 'both'` — candidate identity is selected JOINTLY across PD and SCZ (per-disease FIT
    quantiles only); there is NO per-disease manifest and NO disease-specific candidate id.

So a Stage-2 runner cannot select a candidate outside the pinned space, and cannot select per-disease.
"""
from __future__ import annotations
from acar.v5 import protocol as P

EXPECTED_FAMILY_COUNTS = {"P1": 4, "P2": 4, "P3": 6, "P4": 2, "P5": 6}
EXPECTED_TOTAL = 22


class Stage2ManifestError(RuntimeError):
    """Raised when the frozen candidate manifest fails a Stage-2 selection invariant (drift from the pinned 22-row space)."""


def _canonical_order_ok(manifest):
    """Families appear in P1..P5 order, strictly ascending 3-digit ordinal within a family."""
    order = {f: i for i, f in enumerate(P.FAMILIES)}
    prev = (-1, -1)
    for c in manifest:
        fam = c.get("family")
        if fam not in order:
            return False
        try:
            ordinal = int(str(c.get("id", "")).split("-")[-1])
        except ValueError:
            return False
        key = (order[fam], ordinal)
        if key <= prev:
            return False
        prev = key
    return True


def assert_joint_disease_scope(manifest=None):
    """JOINT-selection invariant: every candidate has disease_scope='both' (per-disease FIT quantiles only). A per-disease
    candidate (disease_scope in {'PD','SCZ'}) would mean the candidate identity was chosen per disease — forbidden. Fail-closed."""
    m = manifest if manifest is not None else P.CANDIDATE_MANIFEST
    bad = [c.get("id") for c in m if c.get("disease_scope") != "both"]
    if bad:
        raise Stage2ManifestError(f"candidates are not jointly-scoped (disease_scope != 'both'): {bad}")
    return True


def selection_manifest():
    """Return the frozen 22-row Stage-2 candidate manifest AFTER re-validating every Stage-2 invariant. Fail-closed."""
    m = P.CANDIDATE_MANIFEST
    if len(m) != EXPECTED_TOTAL:
        raise Stage2ManifestError(f"manifest has {len(m)} rows != {EXPECTED_TOTAL}")
    counts = {f: sum(1 for c in m if c.get("family") == f) for f in P.FAMILIES}
    if counts != EXPECTED_FAMILY_COUNTS:
        raise Stage2ManifestError(f"family counts {counts} != {EXPECTED_FAMILY_COUNTS}")
    ids = tuple(c["id"] for c in m)
    if ids != tuple(P.CANDIDATE_IDS):
        raise Stage2ManifestError("manifest ids drifted from protocol.CANDIDATE_IDS")
    if len(set(ids)) != EXPECTED_TOTAL:
        raise Stage2ManifestError("candidate ids are not unique")
    if not _canonical_order_ok(m):
        raise Stage2ManifestError("manifest is not in canonical P1..P5 ascending-ordinal order")
    assert_joint_disease_scope(m)
    return m


def selection_candidate_ids():
    """The 22 canonical candidate id strings (after validation)."""
    return tuple(c["id"] for c in selection_manifest())


def family_counts():
    """The validated per-family candidate counts, e.g. {'P1':4,'P2':4,'P3':6,'P4':2,'P5':6}."""
    m = selection_manifest()
    return {f: sum(1 for c in m if c.get("family") == f) for f in P.FAMILIES}
