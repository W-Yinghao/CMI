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
    "test_stage1_plan_counts_and_refs",
    "test_stage1_no_real_data_paths",
    "test_stage1_registry_roundtrip_synthetic",
    "test_stage1_external_sites_forbidden",
    "test_stage1_s1_seed_roles",
    "test_stage1_final_external_ref_separate",
    "test_stage1b_authorization_contract",
    "test_stage1b_final_external_still_forbidden",
    "test_stage1b_dev_source_whitelist",
    "test_stage1b_runtime_lock_required",
    "test_stage1b_fold_refs_only",
    "test_stage1b_full_build_requires_30_refs",
    "test_stage1b_source_paths_by_cohort",
    "test_stage1b_runtime_lock_binds_implementation_sha",
    "test_stage1b_build_gate_validates_schema",
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
