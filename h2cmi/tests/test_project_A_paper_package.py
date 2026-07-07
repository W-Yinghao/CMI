"""Project A — tests for the paper-preparation package (Step 11).

Checks the paper/ package exists, its tables reference the canonical theorem/contract IDs, the results
digest values match the tracked summary JSONs (no drift between prose and evidence), the limitations
carry the no-SOTA / oracle-evaluation-only boundary, and the all-skip dataset is marked
not_applicable (not false) in the tracked combined digest. Run:

    python -m h2cmi.tests.test_project_A_paper_package
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PAPER = _REPO / "notes/project_A_observability/paper"
_SUM = _REPO / "notes/project_A_observability/results_summaries"

_FILES = [
    "README.md", "00_claims_and_contributions.md", "01_outline.md", "02_related_work_map.md",
    "03_theorem_table.md", "04_contract_table.md", "05_experiment_table.md", "06_results_digest.md",
    "07_limitations_and_claim_boundary.md", "figures/figure_plan.md", "tables/table_plan.md",
]


def test_paper_package_files_exist():
    missing = [f for f in _FILES if not (_PAPER / f).exists()]
    assert not missing, f"missing paper-package files: {missing}"


def test_paper_tables_reference_canonical_ids():
    thm = (_PAPER / "03_theorem_table.md").read_text()
    for tid in ("OA-0", "MONO-1", "TOS-1", "TU-1", "TU-2", "MP-1", "PD-1", "ID-1"):
        assert tid in thm, f"theorem table missing {tid}"
    con = (_PAPER / "04_contract_table.md").read_text()
    for i in range(1, 13):
        assert f"C{i}" in con, f"contract table missing C{i}"


def test_results_digest_values_match_tracked_summary_json():
    text = (_PAPER / "06_results_digest.md").read_text()

    def _v(fname, *keys):
        d = json.loads((_SUM / fname).read_text())
        for k in keys:
            d = d[k]
        return d

    step9 = _v("step9_bnci2014_001_expanded_summary.json", "aggregate", "mean_strict_dg_bacc")
    d004 = _v("step10_bnci2014_004_summary.json", "aggregate", "mean_strict_dg_bacc")
    comb_excess = _v("step10_moabb_multidataset_summary.json", "overall_normalized",
                     "mean_strict_dg_bacc_excess_norm")
    comb_harm = _v("step10_moabb_multidataset_summary.json", "overall_normalized",
                   "offline_tta_harm_rate")
    for val, label in ((step9, "step9 strict"), (d004, "step10_004 strict"),
                       (comb_excess, "combined excess-norm"), (comb_harm, "combined harm-rate")):
        assert f"{val}" in text, f"results digest missing {label}={val}"


def test_limitations_include_no_sota_and_oracle_target_boundary():
    text = (_PAPER / "07_limitations_and_claim_boundary.md").read_text().lower()
    assert "no sota claim" in text
    assert "oracle" in text and "evaluation-only" in text
    assert "counterexample" in text and "prove" in text          # counterexamples are the proof layer


def test_all_skip_dataset_marked_not_applicable_not_false():
    c = json.loads((_SUM / "step10_moabb_multidataset_summary.json").read_text())
    assert "BNCI2015_001" in c["datasets_all_skipped"] and c["n_datasets_all_skipped"] >= 1
    pd = c["per_dataset"]["BNCI2015_001"]
    assert pd["claim_boundary_status"] == "not_applicable_all_skipped"
    assert pd["all_target_metrics_identifiable_null"] is None      # null, NOT False
    # the top-level roll-up over the ok-run datasets still holds
    assert c["all_target_metrics_identifiable_null"] is True and c["any_forbidden_violations"] is False


ALL_TESTS = [
    test_paper_package_files_exist,
    test_paper_tables_reference_canonical_ids,
    test_results_digest_values_match_tracked_summary_json,
    test_limitations_include_no_sota_and_oracle_target_boundary,
    test_all_skip_dataset_marked_not_applicable_not_false,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} PAPER-PACKAGE TESTS PASSED")


if __name__ == "__main__":
    run()
