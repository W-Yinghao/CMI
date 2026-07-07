"""Project A — text-hygiene tests: source/doc files use LF and are not single-giant-lines.

These pass on the current tree (the files already use LF); they exist to PREVENT regression
(e.g. a CRLF/CR-only commit, or a tool collapsing a file into one huge line) that would make the
files unreviewable in a diff / on GitHub. Run:

    python -m h2cmi.tests.test_observability_text_hygiene
"""
from __future__ import annotations

import glob
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

_PATTERNS = [
    "h2cmi/observability/*.py",
    "h2cmi/run_real_audited.py",
    "h2cmi/run_real_audited_grid.py",
    "h2cmi/tests/test_observability_*.py",
    "h2cmi/tests/test_eval_bridge_harness_contract.py",
    "notes/project_A_observability/*.md",
    "notes/project_A_observability/results_summaries/*.md",
    "scripts/project_A_*.slurm",
]


def _paths():
    out = []
    for pat in _PATTERNS:
        out += [Path(p) for p in glob.glob(str(_REPO / pat))]
    return sorted(set(out))


# Named, high-value review artifacts get a HARD gate (exact files, min-LF, max-line) so the
# GitHub-raw view can never collapse into giant lines. Thresholds are well below the real LF counts.
_CHECK_FILES = {
    "h2cmi/observability/result_index.py": 80,
    "h2cmi/observability/combine_summaries.py": 60,
    "h2cmi/observability/validate_results.py": 80,
    "h2cmi/run_real_audited.py": 120,
    "h2cmi/run_real_audited_grid.py": 80,
    "h2cmi/tests/test_project_A_paper_package.py": 30,
    "notes/project_A_observability/results_summaries/step10_moabb_multidataset_summary.json": 40,
    "notes/project_A_observability/results_summaries/step10_bnci2015_001_summary.json": 80,
    "notes/project_A_observability/paper/06_results_digest.md": 25,
    "notes/project_A_observability/paper/07_limitations_and_claim_boundary.md": 25,
    # Step-12/13/14 science modules + tracked digests (were NOT gated before — CI gap closed)
    "h2cmi/observability/harm_attribution.py": 120,
    "h2cmi/observability/harm_predictor.py": 150,
    "h2cmi/observability/harm_power.py": 60,
    "h2cmi/observability/real_minimal_labels.py": 120,
    "h2cmi/observability/minimal_paired.py": 90,
    "h2cmi/observability/science_dashboard.py": 150,
    "h2cmi/tests/test_real_minimal_label_curves.py": 60,
    "h2cmi/tests/test_observability_harm_power.py": 30,
    "notes/project_A_observability/results_summaries/step14_science_dashboard.json": 30,
    "notes/project_A_observability/results_summaries/step14_real_minimal_label_curves.json": 120,
    "notes/project_A_observability/results_summaries/step14_harm_predictor_summary.json": 40,
    "notes/project_A_observability/results_summaries/step14_harm_power_summary.json": 15,
    "notes/project_A_observability/results_summaries/step14_science_dashboard.md": 15,
    "notes/project_A_observability/results_summaries/step14_real_minimal_label_curves.md": 18,
    "notes/project_A_observability/results_summaries/step14_harm_predictor_summary.md": 12,
    # Step-15 coverage-aware harm-control policies
    "h2cmi/observability/harm_control.py": 150,
    "h2cmi/tests/test_harm_control_policies.py": 60,
    "notes/project_A_observability/results_summaries/step15_harm_control_summary.json": 40,
    "notes/project_A_observability/results_summaries/step15_science_dashboard.json": 25,
    "notes/project_A_observability/results_summaries/step15_harm_control_summary.md": 15,
    "notes/project_A_observability/results_summaries/step15_science_dashboard.md": 15,
    # Step-16 benefit anatomy + sequential frontier
    "h2cmi/observability/benefit_anatomy.py": 100,
    "h2cmi/observability/sequential_harm_control.py": 150,
    "h2cmi/observability/policy_frontier.py": 90,
    "h2cmi/tests/test_benefit_anatomy.py": 45,
    "h2cmi/tests/test_sequential_harm_control.py": 70,
    "h2cmi/tests/test_policy_frontier.py": 45,
    "notes/project_A_observability/results_summaries/step16_benefit_anatomy.json": 40,
    "notes/project_A_observability/results_summaries/step16_sequential_harm_control.json": 40,
    "notes/project_A_observability/results_summaries/step16_policy_frontier.json": 30,
    "notes/project_A_observability/results_summaries/step16_science_dashboard.json": 25,
    "notes/project_A_observability/results_summaries/step16_benefit_anatomy.md": 12,
    "notes/project_A_observability/results_summaries/step16_sequential_harm_control.md": 15,
    "notes/project_A_observability/results_summaries/step16_policy_frontier.md": 12,
    "notes/project_A_observability/results_summaries/step16_science_dashboard.md": 15,
    # Step-17 estimand consistency + per-estimand frontier
    "h2cmi/observability/estimand_consistency.py": 150,
    "h2cmi/observability/estimand_frontier.py": 100,
    "h2cmi/tests/test_estimand_consistency.py": 60,
    "h2cmi/tests/test_estimand_frontier.py": 40,
    "notes/project_A_observability/science/11_estimand_consistency.md": 25,
    "notes/project_A_observability/science/12_estimand_consistency_results.md": 40,
    "notes/project_A_observability/results_summaries/step17_estimand_consistency.json": 40,
    "notes/project_A_observability/results_summaries/step17_estimand_frontier.json": 30,
    "notes/project_A_observability/results_summaries/step17_science_dashboard.json": 25,
    "notes/project_A_observability/results_summaries/step17_estimand_consistency.md": 12,
    "notes/project_A_observability/results_summaries/step17_estimand_frontier.md": 12,
    "notes/project_A_observability/results_summaries/step17_science_dashboard.md": 15,
    # Step-18 harm mechanisms + deployment-prior stress
    "h2cmi/observability/harm_mechanisms.py": 120,
    "h2cmi/observability/prior_stress.py": 120,
    "h2cmi/tests/test_harm_mechanisms.py": 60,
    "h2cmi/tests/test_prior_stress.py": 60,
    "notes/project_A_observability/science/13_prior_stress_and_harm_channels.md": 30,
    "notes/project_A_observability/science/14_harm_mechanism_results.md": 30,
    "notes/project_A_observability/science/15_prior_stress_results.md": 30,
    "notes/project_A_observability/results_summaries/step18_harm_mechanisms.json": 40,
    "notes/project_A_observability/results_summaries/step18_prior_stress.json": 40,
    "notes/project_A_observability/results_summaries/step18_science_dashboard.json": 25,
    "notes/project_A_observability/results_summaries/step18_harm_mechanisms.md": 12,
    "notes/project_A_observability/results_summaries/step18_prior_stress.md": 8,
    "notes/project_A_observability/results_summaries/step18_science_dashboard.md": 15,
    # Step-19 prior-uncertainty robustness frontier
    "h2cmi/observability/prior_uncertainty.py": 150,
    "h2cmi/observability/prior_robust_policy.py": 100,
    "h2cmi/tests/test_prior_uncertainty.py": 60,
    "h2cmi/tests/test_prior_robust_policy.py": 45,
    "notes/project_A_observability/science/16_prior_uncertainty_frontier.md": 30,
    "notes/project_A_observability/science/17_prior_robustness_results.md": 25,
    "notes/project_A_observability/science/18_prior_uncertainty_results.md": 35,
    "notes/project_A_observability/science/19_prior_robust_policy_results.md": 35,
    "notes/project_A_observability/results_summaries/step19_prior_uncertainty_frontier.json": 40,
    "notes/project_A_observability/results_summaries/step19_prior_robust_policy.json": 40,
    "notes/project_A_observability/results_summaries/step19_science_dashboard.json": 25,
    "notes/project_A_observability/results_summaries/step19_prior_uncertainty_frontier.md": 12,
    "notes/project_A_observability/results_summaries/step19_prior_robust_policy.md": 14,
    "notes/project_A_observability/results_summaries/step19_science_dashboard.md": 15,
    # Step-20 final closeout and claim ledger
    "h2cmi/observability/closeout.py": 120,
    "h2cmi/tests/test_closeout.py": 60,
    "notes/project_A_observability/science/20_closeout_and_claim_ledger.md": 45,
    "notes/project_A_observability/results_summaries/step20_closeout.json": 60,
    "notes/project_A_observability/results_summaries/step20_closeout.md": 30,
}
_MAX_LINE_BYTES = 500


def test_project_a_files_have_no_cr_bytes():
    # HARD: not a single CR byte (CRLF *or* CR-only) — GitHub raw splits on LF, so any CR risks a
    # collapsed / giant-line view. LF-only is the only reviewable state.
    for rel in _CHECK_FILES:
        raw = (_REPO / rel).read_bytes()
        assert b"\r" not in raw, f"{rel} contains CR bytes; GitHub raw view will collapse lines"


def test_project_a_core_files_are_lf_reviewable():
    for rel, min_lf in _CHECK_FILES.items():
        raw = (_REPO / rel).read_bytes()
        assert raw.count(b"\n") >= min_lf, (
            f"{rel} has {raw.count(chr(10).encode())} LF newlines (< {min_lf}); "
            "raw view would be too collapsed to review")
        max_line = max(len(line) for line in raw.split(b"\n"))
        assert max_line < _MAX_LINE_BYTES, (
            f"{rel} has a {max_line}-byte line (>= {_MAX_LINE_BYTES}) — unreviewable in raw view")


def test_project_a_files_use_lf_not_cr_only():
    for p in _paths():
        b = p.read_bytes()
        if not b:
            continue
        assert b.count(b"\n") >= 1, f"{p} has content but no LF newline"
        # no lone CR (CR that is not part of CRLF) -> old-Mac endings that render as one line
        assert b.replace(b"\r\n", b"").count(b"\r") == 0, f"{p} has CR-only line endings"


def test_project_a_review_files_are_not_single_giant_lines():
    for p in _paths():
        if p.suffix not in (".py", ".md"):
            continue
        text = p.read_text()
        if len(text) < 400:
            continue                                  # trivially small file; nothing to check
        lines = text.split("\n")
        nonblank = [ln for ln in lines if ln.strip()]
        assert len(nonblank) >= 5, f"{p} renders as <5 non-blank lines (giant-line file?)"
        longest = max((len(ln) for ln in lines), default=0)
        assert longest <= 2000, f"{p} has a {longest}-char line (>2000) — unreviewable"


ALL_TESTS = [test_project_a_files_use_lf_not_cr_only,
             test_project_a_review_files_are_not_single_giant_lines,
             test_project_a_files_have_no_cr_bytes,
             test_project_a_core_files_are_lf_reviewable]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} TEXT-HYGIENE TESTS PASSED ({len(_paths())} files checked)")


if __name__ == "__main__":
    run()
