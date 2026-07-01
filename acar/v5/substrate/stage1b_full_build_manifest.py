"""ACAR V5 Stage-1B FULL-BUILD manifest validation (pure/stdlib; validates strings only — opens NOTHING). A real Stage-1B build
constructs ALL 30 fold-contained substrates (2 diseases × 5 folds × 3 seeds), each trained on its disease's FULL frozen DEV source
cohort set. So the full-build manifest uses `source_paths_by_cohort` (one path per DEV cohort), NOT a single scalar `source_path`.
"""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5.substrate import build_manifest_schema as SCH
from acar.v5.substrate import stage1b_manifest as MAN
from acar.v5.substrate import stage1b_authorization as SA

EXPECTED_FOLD_BUILDS = len(SA.CANONICAL_FOLD_REFS)               # 30
_ALL_DEV_COHORTS = {c for cs in P.DEV_COHORTS.values() for c in cs}


class Stage1bFullBuildError(RuntimeError):
    """Raised when a Stage-1B FULL build manifest is incomplete / partial / mis-specified."""


def validate_cohort_source_path(disease, cohort, path):
    """One per-cohort real DEV source path (string check only). Must reference EXACTLY this cohort of THIS disease, and no other
    DEV cohort / external site / prior artifact / cache."""
    if disease not in P.DEV_COHORTS:
        raise Stage1bFullBuildError(f"unknown disease {disease!r}")
    if cohort not in P.DEV_COHORTS[disease]:
        raise Stage1bFullBuildError(f"{cohort!r} is not a {disease} DEV cohort")
    MAN.validate_dev_source_path(disease, path)                 # disease-matched; no foreign disease / site / artifact / cache
    parts = set(str(path).replace("\\", "/").split("/"))        # PATH-SEGMENT exactness (Step 1B2): "ds002778_old" != "ds002778"
    if cohort not in parts:
        raise Stage1bFullBuildError(f"{disease}/{cohort}: path has no exact '{cohort}' segment: {path}")
    others = sorted(c for c in _ALL_DEV_COHORTS if c != cohort and c in parts)
    if others:
        raise Stage1bFullBuildError(f"{disease}/{cohort}: path also has other cohort segments {others}: {path}")
    return True


def validate_source_paths_by_cohort(disease, mapping):
    """The per-fold real DEV inputs: keys must be EXACTLY the disease's frozen DEV cohorts, each path cohort-exact."""
    if not isinstance(mapping, dict):
        raise Stage1bFullBuildError(f"{disease}: source_paths_by_cohort must be a dict")
    if set(mapping) != set(P.DEV_COHORTS[disease]):
        raise Stage1bFullBuildError(f"{disease}: source_paths_by_cohort keys must equal {sorted(P.DEV_COHORTS[disease])}, got {sorted(mapping)}")
    for cohort, path in mapping.items():
        validate_cohort_source_path(disease, cohort, path)
    return True


def validate_full_build_manifest(plan, auth):
    """Fail-closed FULL-build manifest check (given a validated auth). Requires: schema valid; final-external schema-only; ALL 30
    canonical fold refs present; each authorized + carrying a complete, cohort-exact source_paths_by_cohort. Returns the count of
    fold substrates that would be built (must be 30). Pure — opens nothing."""
    SCH.validate_build_manifest(plan)
    MAN.assert_final_external_schema_only(plan.get("final_external_refs", []))
    allowed = set(auth["allowed_refs"])
    present = {e["ref"] for e in plan.get("fold_contained_refs", [])}
    if present != set(SA.CANONICAL_FOLD_REFS):
        missing = sorted(set(SA.CANONICAL_FOLD_REFS) - present)
        extra = sorted(present - set(SA.CANONICAL_FOLD_REFS))
        raise Stage1bFullBuildError(f"full build requires ALL 30 canonical fold refs (missing {missing[:5]}, extra {extra[:5]})")
    built = 0
    for e in plan["fold_contained_refs"]:
        if e["ref"] not in allowed:
            raise Stage1bFullBuildError(f"fold ref {e['ref']} not in authorized allowed_refs")
        spb = e.get("source_paths_by_cohort")
        if not spb:
            raise Stage1bFullBuildError(f"{e['ref']}: full build requires source_paths_by_cohort (real DEV inputs), got none")
        if e.get("source_path") is not None:
            raise Stage1bFullBuildError(f"{e['ref']}: full build uses source_paths_by_cohort, not a scalar source_path")
        validate_source_paths_by_cohort(e["disease"], spb)
        built += 1
    if built != EXPECTED_FOLD_BUILDS:
        raise Stage1bFullBuildError(f"full build must construct exactly {EXPECTED_FOLD_BUILDS} fold substrates, got {built}")
    return built
