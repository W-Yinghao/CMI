"""ACAR V5 Stage-1A substrate-build PLAN (deterministic; PLAN-ONLY; reads NO real file, trains NOTHING, dumps NOTHING).

Enumerates the fold-contained DEV substrate refs (disease × fold × seed) with their allowed ROLES, and declares the schema of the
Stage-5 final all-source external-execution substrate ref — kept as a DISTINCT ref type so a final external substrate can never be
slipped into the fold registry. Every `source_path` is None here; real paths are filled ONLY at a later Stage-1B under explicit
authorization (see acar/v5/substrate/stage1_preflight.py).
"""
from __future__ import annotations
from acar.v5 import protocol as P

SELECTION_ROLE = "stage2_selection"
S1_ROLE = "s1_robustness"
FINAL_EXTERNAL_ROLE = "stage5_external_execution"


def assert_seed_role(seed, role):
    """Fail-closed seed-role check: the Stage-2 SELECTION role is allowed ONLY for the canonical selection seed 20260711;
    seeds 20260712/20260713 are S1-robustness ONLY (never selection)."""
    if role == SELECTION_ROLE:
        if seed != P.SELECTION_SEED:
            raise ValueError(f"seed {seed} may NOT carry the selection role (selection seed is {P.SELECTION_SEED})")
    elif role == S1_ROLE:
        if seed not in P.S1_SEEDS:
            raise ValueError(f"seed {seed} not in the pinned S1 seed set {P.S1_SEEDS}")
    else:
        raise ValueError(f"unknown fold-substrate role {role!r}")
    return True


def fold_refs():
    """All fold-contained substrate refs with allowed roles (no file read). seed 20260711 → {selection, s1}; 12/13 → {s1}."""
    out = []
    for disease in sorted(P.DEV_COHORTS):
        for fold in range(P.OUTER_K):
            for seed in P.S1_SEEDS:
                roles = [SELECTION_ROLE, S1_ROLE] if seed == P.SELECTION_SEED else [S1_ROLE]
                out.append({"ref": f"{disease}/fold{fold}/seed{seed}", "disease": disease, "fold": fold,
                            "seed": seed, "roles": roles, "source_path": None})
    return out


def selection_refs():
    return [r for r in fold_refs() if SELECTION_ROLE in r["roles"]]


def final_external_refs():
    """Stage-5 final all-source external-execution substrate — SCHEMA ONLY (NOT built/registered in Stage-1A). Distinct ref
    shape `external_exec/{disease}/all_source_dev`; built ONLY after the candidate is fixed (post Stage-4); NEVER used for
    selection."""
    return [{"ref": f"external_exec/{d}/all_source_dev", "disease": d, "role": FINAL_EXTERNAL_ROLE, "source_path": None,
             "note": "built only AFTER candidate fixed (post Stage-4); NEVER used for candidate selection/reselection"}
            for d in sorted(P.DEV_COHORTS)]


def build_substrate_plan():
    """Return the full deterministic Stage-1A plan (data-free)."""
    fr = fold_refs()
    return {
        "protocol_tag": "acar-v5-protocol",
        "dev_cohorts": {d: list(P.DEV_COHORTS[d]) for d in sorted(P.DEV_COHORTS)},
        "fold_contained_refs": fr,
        "selection_refs": [r["ref"] for r in fr if SELECTION_ROLE in r["roles"]],
        "final_external_refs": final_external_refs(),
        "selection_seed": P.SELECTION_SEED,
        "s1_seeds": list(P.S1_SEEDS),
        "counts": {"diseases": len(P.DEV_COHORTS), "folds": P.OUTER_K, "seeds": len(P.S1_SEEDS),
                   "fold_refs": len(fr), "selection_refs": len(selection_refs()),
                   "final_external_refs": len(P.DEV_COHORTS)},
    }
