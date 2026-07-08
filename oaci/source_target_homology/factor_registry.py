"""C28 — factor registry. The source-side factor uses the IDENTICAL definition as the C27 target-side factor
(gate #5: source and target factor definitions must be identical). We re-export C27's candidate_features so the
homology comparison is byte-consistent and cannot silently diverge."""
from __future__ import annotations

from ..logit_geometry import factor_registry as c27_fr
from . import schema

candidate_features = c27_fr.candidate_features             # IDENTICAL source/target factor definition
family_feature_names = c27_fr.family_feature_names
feature_family_rows = c27_fr.feature_family_rows
select = c27_fr.select


def assert_identical_definition() -> None:
    """Gate #5: the carrier feature names must match C27's exactly."""
    got = tuple(family_feature_names(schema.CARRIER_FAMILY))
    if got != schema.CARRIER_NAMES:
        raise ValueError(f"C28 carrier names {got} != C27 {schema.CARRIER_NAMES} (definitions must be identical)")
