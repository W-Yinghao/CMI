"""C39 deterministic taxonomy."""
from __future__ import annotations

from . import schema


def _max_share(rows, scope):
    vals = [float(r["positive_selected_advantage_share"]) for r in rows if r["scope"] == scope]
    return max(vals) if vals else 0.0


def classify(identity, concentration, class_domain_rows, audit_stability, support, gauge, bootstrap):
    ident = identity["summary"]
    conc = concentration["summary"]
    aud = audit_stability["summary"]
    sup = support["summary"]
    gau = gauge["summary"]
    boot = bootstrap["summary"]
    class_share = _max_share(class_domain_rows, "class")
    domain_share = _max_share(class_domain_rows, "domain")
    exact = bool(ident["selection_identity_pass"] and ident["source_audit_additive_pass"])
    established = {
        schema.A1: exact,
        schema.A2: exact and conc["concentrated_pair_fraction"] is not None and
        conc["concentrated_pair_fraction"] >= 0.5,
        schema.A3: exact and conc["broad_pair_fraction"] is not None and
        conc["broad_pair_fraction"] >= 0.5,
        schema.A4: exact and class_share >= schema.CLASS_DOMAIN_SHARE_GATE,
        schema.A5: exact and domain_share >= schema.CLASS_DOMAIN_SHARE_GATE,
        schema.A6: exact and sup["support_artifact_pair_fraction"] is not None and
        sup["support_artifact_pair_fraction"] >= schema.SUPPORT_ARTIFACT_SHARE_GATE,
        schema.A7: exact and (
            (aud["mean_atom_sign_preservation_rate"] is not None and
             aud["mean_atom_sign_preservation_rate"] < schema.ATOM_SIGN_STABILITY_GATE) or
            (aud["selection_to_audit_inversion_rate"] is not None and
             aud["selection_to_audit_inversion_rate"] >= schema.INSTABILITY_RATE_GATE)),
        schema.A8: exact and gau["atom_target_gauge_conflict_fraction"] is not None and
        gau["atom_target_gauge_conflict_fraction"] >= schema.GAUGE_CONFLICT_RATE_GATE,
        schema.A9: not exact,
        schema.A10: boot["ucl_quantile_atom_limit"],
    }
    evidence = {
        schema.A1: (
            f"selection_identity={ident['n_selection_identity_pass']}/"
            f"{ident['n_selection_candidates']}, audit_additive="
            f"{ident['n_source_audit_additive_pass']}/{ident['n_source_audit_candidates']}"),
        schema.A2: f"concentrated_pair_fraction={conc['concentrated_pair_fraction']}",
        schema.A3: f"broad_pair_fraction={conc['broad_pair_fraction']}",
        schema.A4: f"max_class_positive_share={class_share}",
        schema.A5: f"max_domain_positive_share={domain_share}",
        schema.A6: f"support_artifact_pair_fraction={sup['support_artifact_pair_fraction']}",
        schema.A7: (
            f"mean_atom_sign_preservation_rate={aud['mean_atom_sign_preservation_rate']}, "
            f"selection_to_audit_inversion_rate={aud['selection_to_audit_inversion_rate']}"),
        schema.A8: f"atom_target_gauge_conflict_fraction={gau['atom_target_gauge_conflict_fraction']}",
        schema.A9: "exact point atom identity failed or audit additive identity failed",
        schema.A10: "UCL is a bootstrap quantile; C39 does not sum per-atom UCLs",
    }
    rows = [{"case": c, "established": int(bool(established[c])), "evidence": evidence[c]}
            for c in schema.ALL_CASES]
    return {
        "cases": [c for c in schema.ALL_CASES if established[c]],
        "case_rows": rows,
        "established": established,
        "evidence": evidence,
    }
