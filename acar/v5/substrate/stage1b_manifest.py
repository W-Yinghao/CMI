"""ACAR V5 Stage-1B DEV-source WHITELIST (pure/stdlib; validates path STRINGS only — opens NOTHING). Stage-1B is the first real
DEV read, so 'not a forbidden marker' is not enough: a source path is admissible ONLY if it references disease-matched frozen DEV
source cohorts and NO other-disease cohort / external site / prior-artifact / cache. This is a whitelist, not a blacklist.
"""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5.substrate import build_manifest_schema as SCH

# Stage-1B forbidden markers = prior ACAR artifacts + caches + v4 dumps (NOT a blanket "/projects/" — real DEV lives under the
# datalake, so /projects/ alone is allowed for a disease-matched DEV cohort). External-site tokens are always forbidden.
STAGE1B_FORBIDDEN_MARKERS = ("/scps/cache", "feat_dump_v4", "acar_v4_", "acar_v3_", "feat_dump_v3", "feat_dump_v2")
STAGE1B_FORBIDDEN_SITE_TOKENS = SCH.FORBIDDEN_SITE_TOKENS

_OTHER_DISEASE_COHORTS = {d: tuple(c for dd, cs in P.DEV_COHORTS.items() if dd != d for c in cs) for d in P.DEV_COHORTS}


class Stage1bWhitelistError(RuntimeError):
    """Raised when a Stage-1B source path is not a disease-matched frozen DEV cohort (or hits a forbidden marker/site)."""


def validate_dev_source_path(disease, path):
    """Fail-closed whitelist for ONE Stage-1B real-data source path (string check only; no filesystem access)."""
    if disease not in P.DEV_COHORTS:
        raise Stage1bWhitelistError(f"unknown disease {disease!r}")
    if not path:
        raise Stage1bWhitelistError(f"{disease}: empty source_path")
    s = str(path)
    if any(m in s for m in STAGE1B_FORBIDDEN_MARKERS):
        raise Stage1bWhitelistError(f"{disease}: source_path hits a forbidden artifact/cache marker: {s}")
    if any(tok in s for tok in STAGE1B_FORBIDDEN_SITE_TOKENS):
        raise Stage1bWhitelistError(f"{disease}: source_path references an external/held-out site: {s}")
    own = [c for c in P.DEV_COHORTS[disease] if c in s]
    if not own:
        raise Stage1bWhitelistError(f"{disease}: source_path references no {disease} DEV cohort (whitelist): {s}")
    foreign = [c for c in _OTHER_DISEASE_COHORTS[disease] if c in s]
    if foreign:
        raise Stage1bWhitelistError(f"{disease}: source_path references other-disease cohorts {foreign}: {s}")
    return True


def assert_final_external_schema_only(final_refs):
    """Fail-closed: the Stage-5 final all-source external-execution refs must stay SCHEMA-ONLY in Stage-1A/1B — source_path is
    None (ALWAYS, regardless of any authorization), role == stage5_external_execution, exactly one per disease, diseases == the
    frozen {PD, SCZ}. Raises Stage1bWhitelistError otherwise. (A 'valid' Stage-1B auth may NEVER open a final-external build.)"""
    from acar.v5.substrate import plan as PLAN
    diseases = []
    for e in final_refs:
        if e.get("source_path") is not None:
            raise Stage1bWhitelistError(f"final-external ref {e.get('ref')} must be schema-only (source_path=None), not built")
        if e.get("role") != PLAN.FINAL_EXTERNAL_ROLE:
            raise Stage1bWhitelistError(f"final-external ref {e.get('ref')} role must be {PLAN.FINAL_EXTERNAL_ROLE}")
        if not SCH.is_final_external_ref(e.get("ref")):
            raise Stage1bWhitelistError(f"{e.get('ref')} is not a final-external ref shape")
        diseases.append(e.get("disease"))
    if sorted(diseases) != sorted(P.DEV_COHORTS):
        raise Stage1bWhitelistError(f"final-external refs must be exactly one per disease {sorted(P.DEV_COHORTS)}, got {sorted(diseases)}")
    return True


def validate_stage1b_build_manifest(plan, auth):
    """Fail-closed Stage-1B build manifest check (given a validated auth): every fold ref with a real source_path must be one of
    the authorized 30 fold refs AND its path must pass the disease-matched DEV whitelist; final-external refs must stay
    schema-only (source_path None). Pure — validates strings only; opens nothing. Returns the count of admitted real fold refs."""
    allowed = set(auth["allowed_refs"])
    admitted = 0
    for e in plan.get("fold_contained_refs", []):
        sp = e.get("source_path")
        if not sp:
            continue
        if e["ref"] not in allowed:
            raise Stage1bWhitelistError(f"fold ref {e['ref']} not in the authorized allowed_refs")
        if not SCH.is_fold_ref(e["ref"]):
            raise Stage1bWhitelistError(f"{e['ref']} is not a fold ref")
        validate_dev_source_path(e["disease"], sp)
        admitted += 1
    assert_final_external_schema_only(plan.get("final_external_refs", []))
    return admitted
