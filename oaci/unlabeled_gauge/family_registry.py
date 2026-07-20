"""C25 — FROZEN R3 feature-family registry. The three families partition the 12 label-free target-unlabeled
features the recovering R3 gauge used. Declared before analysis; NOT feature selection (family membership is
fixed, only their pre-declared unions are evaluated). Families named in the spec but not computed by the
recovering gauge are disclosed as out-of-scope (would require target-Z re-inference)."""
from __future__ import annotations

from . import schema


def families() -> dict:
    return {k: list(v) for k, v in schema.FAMILIES.items()}


def family_names() -> list:
    return list(schema.FAMILIES)


def assert_partition(available_features) -> None:
    """The frozen families must exactly partition the R3 features present in the sidecar (fail loud otherwise)."""
    fam_feats = set(schema.ALL_R3_FEATURES)
    avail = set(available_features)
    if fam_feats != avail:
        missing = fam_feats - avail; extra = avail - fam_feats
        raise ValueError(f"C25 family registry does not partition the R3 features: missing={sorted(missing)} "
                         f"extra_in_sidecar={sorted(extra)}")
    # no feature appears in two families
    seen = {}
    for fam, feats in schema.FAMILIES.items():
        for f in feats:
            if f in seen:
                raise ValueError(f"feature {f} in two families: {seen[f]} and {fam}")
            seen[f] = fam


def feature_family_rows() -> list:
    return [{"feature": f, "family": fam, "computed_in_recovering_r3": True}
            for fam, feats in schema.FAMILIES.items() for f in feats] + \
           [{"feature": "(none)", "family": fam, "computed_in_recovering_r3": False} for fam in schema.NOT_COMPUTED_FAMILIES]
