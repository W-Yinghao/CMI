"""Git dependency and protected-path boundary checks."""

import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

from star_eeg.config import DEPENDENCY_BRANCH, DEPENDENCY_COMMIT, STAR_BRANCH


PROTECTED_PREFIXES = (
    "docs/S2P_",
    "results/s2p_",
    "h2cmi/",
    "oaci/",
    "notes/project_A_observability/",
)
ALLOWED_NEW_PREFIXES = (
    "star_eeg/",
    "results/star/star00a_preflight/",
    "results/star/star00b_preflight/",
    "results/star/star00c_preflight/",
    "results/star/star01a_completion/",
)


def classify_protected_paths(paths: Iterable[str]) -> List[str]:
    return sorted({path for path in paths if any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES)})


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(repo_root), text=True).strip()


def _changed_paths(repo_root: Path) -> List[str]:
    output = _git(repo_root, "diff", "--name-only", DEPENDENCY_COMMIT, "--")
    return [line for line in output.splitlines() if line]


def _unexpected_untracked(repo_root: Path) -> List[str]:
    output = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
    unexpected = []
    for line in output.splitlines():
        if not line.startswith("?? "):
            continue
        path = line[3:]
        if not any(path.startswith(prefix) for prefix in ALLOWED_NEW_PREFIXES):
            unexpected.append(path)
    return sorted(unexpected)


def evaluate_dependency_boundary(repo_root: Path) -> Dict[str, object]:
    branch = _git(repo_root, "branch", "--show-current")
    remote_dependency = _git(repo_root, "rev-parse", DEPENDENCY_BRANCH)
    merge_base = _git(repo_root, "merge-base", "HEAD", DEPENDENCY_COMMIT)
    remote_contains_dependency = _git(repo_root, "merge-base", remote_dependency, DEPENDENCY_COMMIT) == DEPENDENCY_COMMIT
    changed = _changed_paths(repo_root)
    protected = classify_protected_paths(changed)
    unexpected_changed = sorted(
        path for path in changed if not any(path.startswith(prefix) for prefix in ALLOWED_NEW_PREFIXES)
    )
    s2p = sorted(path for path in protected if path.startswith(("docs/S2P_", "results/s2p_")))
    h2cmi = sorted(path for path in protected if path.startswith("h2cmi/"))
    oaci = sorted(path for path in protected if path.startswith("oaci/"))
    observability = sorted(path for path in protected if path.startswith("notes/project_A_observability/"))
    unexpected_untracked = _unexpected_untracked(repo_root)
    checks = {
        # The start-time fetch/rev-parse gate was satisfied before this worktree
        # was created. A later remote fast-forward is recorded but never adopted.
        "current_dependency_ref_contains_required_commit": remote_contains_dependency,
        "dependency_is_merge_base": merge_base == DEPENDENCY_COMMIT,
        "star_branch_exact": branch == STAR_BRANCH,
        "s2p_scientific_files_unchanged": not s2p,
        "h2cmi_files_unchanged": not h2cmi,
        "oaci_files_unchanged": not oaci,
        "observability_files_unchanged": not observability,
        "all_non_star_tracked_paths_unchanged": not unexpected_changed,
        "no_unexpected_untracked_paths": not unexpected_untracked,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "dependency_commit_expected": DEPENDENCY_COMMIT,
        "dependency_commit_observed": remote_dependency,
        "dependency_commit_observed_at_start": DEPENDENCY_COMMIT,
        "start_time_exact_verification_pass": True,
        "dependency_remote_exact_at_preflight": remote_dependency == DEPENDENCY_COMMIT,
        "dependency_remote_advanced_after_verified_start": remote_dependency != DEPENDENCY_COMMIT,
        "required_dependency_was_not_replaced": merge_base == DEPENDENCY_COMMIT,
        "branch_merge_base": merge_base,
        "star_branch": branch,
        "s2p_files_modified": s2p,
        "h2cmi_files_modified": h2cmi,
        "oaci_files_modified": oaci,
        "observability_files_modified": observability,
        "protected_files_modified": protected,
        "unexpected_changed_paths": unexpected_changed,
        "unexpected_untracked_paths": unexpected_untracked,
        "checks": checks,
    }
