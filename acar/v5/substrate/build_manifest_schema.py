"""ACAR V5 Stage-1A substrate-build manifest SCHEMA + ref-type helpers (pure/stdlib; no I/O, no data).

Distinguishes the two substrate ref TYPES so they can never be confused:
  - fold-contained DEV substrate ref:  "<disease>/fold<0..K-1>/seed<seed>"   (Stage-2 selection + S1 robustness)
  - final external-execution ref:      "external_exec/<disease>/all_source_dev"  (Stage-5 only; built after candidate fixed)
Plus a fail-closed structural validator for a Stage-1A plan and forbidden-path/site detection.
"""
from __future__ import annotations
import re
from acar.v5 import protocol as P
from acar.v5.substrate import plan as PLAN

_FOLD_RE = re.compile(r"^(?P<disease>PD|SCZ)/fold(?P<fold>\d+)/seed(?P<seed>\d+)$")
_FINAL_RE = re.compile(r"^external_exec/(?P<disease>PD|SCZ)/all_source_dev$")

# any of these appearing in a source_path is a hard forbidden read target (real DEV / v4 artifacts / caches / external sites)
FORBIDDEN_PATH_MARKERS = ("/projects/", "/scps/cache", "feat_dump_v4",
                          "/home/infres/yinwang/acar_v4_", "/home/infres/yinwang/acar_v3_")
FORBIDDEN_SITE_TOKENS = (tuple(P.EXTERNAL_PRIMARY.values())
                         + tuple(P.EXTERNAL_PROVISIONAL_NOT_ADMITTED) + tuple(P.EXTERNAL_EXCLUDED))


def is_fold_ref(ref):
    m = _FOLD_RE.match(str(ref))
    if not m:
        return False
    return int(m.group("fold")) < P.OUTER_K and int(m.group("seed")) in P.S1_SEEDS


def is_final_external_ref(ref):
    return bool(_FINAL_RE.match(str(ref)))


def path_is_forbidden(path):
    """True if a path targets real DEV/artifact/cache data or any external site token (regardless of authorization)."""
    if not path:
        return False
    s = str(path)
    if any(m in s for m in FORBIDDEN_PATH_MARKERS):
        return True
    return any(tok in s for tok in FORBIDDEN_SITE_TOKENS)


def validate_build_manifest(spec):
    """Fail-closed structural check of a Stage-1A plan (pure; no I/O). Verifies ref types, seed roles, count invariants, and that
    NO external-site token appears in any ref. Returns spec."""
    if not isinstance(spec, dict):
        raise ValueError("plan must be a dict")
    if spec.get("protocol_tag") != "acar-v5-protocol":
        raise ValueError("plan.protocol_tag must be acar-v5-protocol")
    fr = spec.get("fold_contained_refs")
    if not isinstance(fr, list) or not fr:
        raise ValueError("fold_contained_refs must be a non-empty list")
    seen = set()
    for r in fr:
        ref = r.get("ref")
        if not is_fold_ref(ref):
            raise ValueError(f"not a valid fold ref: {ref!r}")
        if ref in seen:
            raise ValueError(f"duplicate fold ref {ref}")
        seen.add(ref)
        roles = r.get("roles", [])
        if not roles or any(role not in (PLAN.SELECTION_ROLE, PLAN.S1_ROLE) for role in roles):
            raise ValueError(f"{ref}: invalid roles {roles}")
        for role in roles:                                    # seed-role consistency (selection only seed 20260711)
            PLAN.assert_seed_role(r["seed"], role)
        if is_final_external_ref(ref):
            raise ValueError(f"{ref}: a final-external ref must NOT appear among fold_contained_refs")
    exp = len(P.DEV_COHORTS) * P.OUTER_K * len(P.S1_SEEDS)
    if len(fr) != exp:
        raise ValueError(f"fold_contained_refs count {len(fr)} != expected {exp}")
    fx = spec.get("final_external_refs", [])
    for r in fx:
        if not is_final_external_ref(r.get("ref")):
            raise ValueError(f"final_external_refs contains a non-final ref: {r.get('ref')!r}")
        if is_fold_ref(r.get("ref")):
            raise ValueError(f"{r.get('ref')}: final-external ref must not match the fold shape")
    # no external-site token anywhere in any ref string (belt-and-suspenders)
    for r in fr + list(fx):
        if any(tok in str(r.get("ref")) for tok in FORBIDDEN_SITE_TOKENS):
            raise ValueError(f"external-site token in ref {r.get('ref')!r}")
    return spec
