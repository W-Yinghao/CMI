from pathlib import Path

from star_eeg.red_team.dependency_boundary import (
    classify_protected_paths,
    evaluate_dependency_boundary,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_protected_path_classifier():
    paths = [
        "star_eeg/config.py",
        "results/star/star00a_preflight/preflight_summary.json",
        "docs/S2P_17_ROUTE_B_FINAL_RESULTS.md",
        "h2cmi/train.py",
        "oaci/readme.md",
    ]
    assert classify_protected_paths(paths) == [
        "docs/S2P_17_ROUTE_B_FINAL_RESULTS.md",
        "h2cmi/train.py",
        "oaci/readme.md",
    ]


def test_live_star_worktree_dependency_boundary():
    result = evaluate_dependency_boundary(REPO_ROOT)
    assert result["status"] == "PASS"
    assert result["protected_files_modified"] == []
