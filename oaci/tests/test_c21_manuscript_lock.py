"""C21 Estimand Boundary + Manuscript Lock. Every claim is evidence-grounded; the within-target ~0.64 is
future-work (never a retroactive estimand swap / success claim); forbidden over-claims are absent from ALL
generated files; no selector/detector claim; the estimand reconciliation + C14->C20 chain are present; and the
lock generates cleanly. No experiments."""
from __future__ import annotations

from oaci.manuscript_lock import ledger, report, schema


def test_every_claim_is_evidence_grounded():
    ledger.validate_claims()
    for c in schema.CLAIMS:
        assert c["commit"] and c["report"] and c["key"] and len(c["text"]) > 20


def test_claim_categories_and_counts():
    counts = ledger.counts()
    assert set(counts) == set(schema.CATEGORIES) and all(counts[c] >= 1 for c in schema.CATEGORIES)
    # the headline established set must include the C19 positive AND the C20 not-established boundary
    est = {c["id"] for c in schema.CLAIMS if c["category"] == "established"}
    assert "E6" in est and "E7" in est


def test_within_target_064_is_future_work_not_established():
    ledger.assert_estimand_not_swapped()
    f1 = next(c for c in schema.CLAIMS if c["id"] == "F1")
    assert f1["category"] == "future_work" and "0.6" in f1["key"]
    # no established/diagnostic claim may assert the probe transports / external success
    for c in schema.CLAIMS:
        if c["category"] in ("established", "diagnostic_only"):
            assert "succeeded" not in c["text"].lower()


def test_not_established_covers_selector_and_external():
    ne = {c["id"]: c["text"].lower() for c in schema.CLAIMS if c["category"] == "not_established"}
    joined = " ".join(ne.values())
    assert "selector" in joined and "external" in joined and "deployable" in joined


def test_forbidden_claims_absent_from_all_generated_files():
    res = report.build()
    for renderer in (report.render_estimand_boundary_md, report.render_claim_ledger_md,
                     report.render_manuscript_lock_md, report.render_paper_outline_md,
                     report.render_forbidden_audit_md):
        text = renderer(res)
        # the forbidden AUDIT file lists the phrases as "FORBIDDEN: ..."; all OTHER files must be clean
        if renderer is report.render_forbidden_audit_md:
            continue
        ledger.assert_no_forbidden(text)
    # canonical conclusion is clean
    ledger.assert_no_forbidden(schema.CANONICAL_CONCLUSION)


def test_estimand_reconciliation_present():
    er = schema.ESTIMAND_RECONCILIATION
    assert er["c19_pooled_loto_auc_in_regime"] == 0.561
    assert er["c20_pooled_loto_auc_cross_regime_range"][1] <= 0.55           # near chance
    assert 0.6 < er["within_target_mean_auc_range"][0] < er["within_target_mean_auc_range"][1] < 0.7
    assert "does not transport" in er["resolution"].lower() or "not calibrated" in er["resolution"].lower()


def test_evidence_chain_c14_to_c20():
    stages = [s for s, _, _ in schema.EVIDENCE_CHAIN]
    assert stages == ["C14", "C16", "C17", "C18", "C19", "C20"]


def test_paper_outline_has_all_stages_and_external_boundary_in_main():
    names = [n for n, _ in schema.PAPER_SECTIONS]
    assert "06_external_boundary" in names and "05_identifiability_and_probe" in names
    md = report.render_paper_outline_md(report.build())
    assert "main" in md.lower() and "not as a limitation" in md.lower()


def test_canonical_conclusion_holds_all_four_positions():
    c = schema.CANONICAL_CONCLUSION.lower()
    assert "do not transfer" in c and "invisible to simple source-audit" in c
    assert "pre-registered low-freedom diagnostic probe" in c and "does not establish stable cross-regime" in c
    assert "not a deployable target-free selector" in c


def test_build_and_lock_ok():
    res = report.build()
    assert res["manuscript_locked"] and res["diagnostic_only_non_deployable"]
    assert res["paper_title"].startswith("When Source-Side Signals Do Not Transfer")


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} c21-manuscript-lock tests")


if __name__ == "__main__":
    _run_all()
