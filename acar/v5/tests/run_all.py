"""Run all ACAR v5 synthetic guard tests. Exit 0 iff every module's ALL-PASS line is reached. No DEV/real/external data."""
from __future__ import annotations
import importlib

MODULES = (
    "test_action_scalarization_quantile_universe",
    "test_exact_manifest",
    "test_quantile_method_pinned",
    "test_low_coverage_degeneracy",
    "test_no_label_in_route",
    "test_subject_disjoint",
    "test_split_ratio_cardinality",
    "test_no_external_before_tag",
    "test_substrate_hash_required",
    "test_fixed_candidate_no_reselection",
    "test_verify_artifacts",
)


def main():
    n = 0
    for m in MODULES:
        mod = importlib.import_module(f"acar.v5.tests.{m}")
        mod.main()
        n += 1
    print(f"\nALL V5 GUARD SUITES PASS ({n} modules)")


if __name__ == "__main__":
    main()
