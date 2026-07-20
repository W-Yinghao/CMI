"""C35 deterministic utility-cone taxonomy."""
from __future__ import annotations

from . import schema


def _v(d, key, default=0.0):
    v = d.get(key)
    return default if v is None else v


def classify(pareto, simplex, source, target_unlabeled, scaling):
    ps = pareto["summary"]
    us = simplex["summary"]
    ss = source["summary"]
    ts = target_unlabeled["summary"]
    sc = scaling["summary"]
    cases, evidence = [], {}

    robust = _v(us, "preference_robust_fraction")
    dependent = _v(us, "preference_dependent_fraction")
    narrow = _v(us, "narrow_scalarization_fraction")
    no_regret = _v(us, "no_regret_fraction")
    weak_pareto = _v(ps, "strict_pareto_better_fraction") + _v(ps, "weak_pareto_better_fraction")
    incomparable = _v(ps, "pareto_incomparable_fraction")

    evidence["U1"] = {"preference_robust_fraction": robust}
    if robust >= 0.25:
        cases.append(schema.U1)

    evidence["U2"] = {"preference_dependent_fraction": dependent,
                      "pareto_incomparable_fraction": incomparable}
    if dependent >= 0.30 or incomparable >= 0.30:
        cases.append(schema.U2)

    evidence["U3"] = {"weak_or_strict_pareto_better_fraction": weak_pareto}
    if weak_pareto >= 0.30:
        cases.append(schema.U3)

    evidence["U4"] = {"narrow_scalarization_fraction": narrow, "no_regret_fraction": no_regret}
    if (narrow + no_regret) >= 0.20:
        cases.append(schema.U4)

    robust_src = ss.get("preference_robust_regret", {})
    dep_src = ss.get("preference_dependent_regret", {})
    evidence["U5"] = {"robust_source_misranking_rate": robust_src.get("source_misranking_rate"),
                      "robust_n_pairs": robust_src.get("n_pairs")}
    if _v(robust_src, "n_pairs") >= 10 and _v(robust_src, "source_misranking_rate") >= 0.25:
        cases.append(schema.U5)

    evidence["U6"] = {"dependent_source_misranking_rate": dep_src.get("source_misranking_rate"),
                      "robust_source_misranking_rate": robust_src.get("source_misranking_rate")}
    if _v(dep_src, "source_misranking_rate") >= 0.25 and _v(robust_src, "source_misranking_rate") < 0.25:
        cases.append(schema.U6)

    robust_tu = ts.get("preference_robust_regret", {})
    evidence["U7"] = {"robust_target_unlabeled_agreement_rate": robust_tu.get("target_unlabeled_agreement_rate"),
                      "robust_source_agreement_rate": robust_src.get("source_agreement_rate")}
    if _v(robust_tu, "n_pairs") >= 10 and (
            robust_tu.get("target_unlabeled_agreement_rate") is None or
            _v(robust_tu, "target_unlabeled_agreement_rate") <= _v(robust_src, "source_agreement_rate") + 0.05):
        cases.append(schema.U7)

    evidence["U8"] = {"robust_fraction_range": sc.get("robust_fraction_range"),
                      "dependent_fraction_range": sc.get("dependent_fraction_range"),
                      "narrow_fraction_range": sc.get("narrow_fraction_range")}
    if max(_v(sc, "robust_fraction_range"), _v(sc, "dependent_fraction_range"),
           _v(sc, "narrow_fraction_range")) >= 0.20:
        cases.append(schema.U8)

    if not cases:
        cases.append(schema.U9)
    return {"cases": cases, "evidence": evidence}
