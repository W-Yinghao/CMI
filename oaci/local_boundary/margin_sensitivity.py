"""C33 primary-vs-robust local taxonomy comparison."""
from __future__ import annotations


def margin_sensitivity(primary, robust):
    return {
        "primary_cases": primary["taxonomy"]["cases"],
        "robust_cases": robust["taxonomy"]["cases"],
        "primary_mean_transition_rate": primary["boundary"]["summary"]["mean_transition_rate"],
        "robust_mean_transition_rate": robust["boundary"]["summary"]["mean_transition_rate"],
        "primary_pm1_joint_rate": primary["boundary"]["summary"]["mean_pm1_joint_good_rate"],
        "robust_pm1_joint_rate": robust["boundary"]["summary"]["mean_pm1_joint_good_rate"],
        "primary_selected_hit": primary["pairs"]["summary"]["selected_joint_hit_rate"],
        "robust_selected_hit": robust["pairs"]["summary"]["selected_joint_hit_rate"],
        "case_changes": sorted(set(primary["taxonomy"]["cases"]) ^ set(robust["taxonomy"]["cases"])),
    }
