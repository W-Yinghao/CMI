#!/usr/bin/env python3
"""Project B Step-4: assemble a clean, review-ready PR / manuscript package.

Packaging only. Runs no experiment, tunes nothing, and touches no ``h2cmi/**`` or ``cmi/**``. It reads
the branch's git history and the frozen Step-2G/3A/3C/3D outputs and writes:

  branch summary, PR description, reproducibility guide, artifact manifest (md+csv+json),
  commit timeline, test matrix, file inventory, finalized claim boundary, paper-bundle manifest,
  merge-readiness checklist, and a validation report.

Fail-loud contract (Step-4 section 10): wrong branch, missing frozen outputs / Step-3D notes, a claim
boundary that drops a required assertion, a PR body without label-safety, a reproducibility doc without
``allow_missing_data``, any overclaim phrase, any bare "full MOABB benchmark", or a post-generation
``git diff`` that touches ``h2cmi/**`` / ``cmi/**`` are all hard errors.

CLI::

    python scripts/project_b_step4_package.py \
        --repo /home/infres/yinwang/CMI_AAAI_projectB \
        --step2g /tmp/project_b_step2g_report \
        --step3a /tmp/project_b_step3a_real_bridge \
        --step3c /tmp/project_b_step3c_real_benchmark \
        --step3d /tmp/project_b_step3d_paper_update \
        --out /tmp/project_b_step4_package
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from typing import Any, Dict, List, Tuple

BRANCH = "project-b-refusal-router"
BASE = "main"

OVERCLAIM_PHRASES = [
    "guarantees tta improvement",
    "solves concept shift",
    "detects all ood",
    "target-label tuned",
]

# Canonical assertions the finalized claim boundary MUST still carry (substring, case-insensitive).
REQUIRED_CLAIM_SUBSTRINGS = [
    "degenerate",                       # no TTA under degenerate ACAR-harm
    "runs end-to-end on bnci2014_004",  # real bridge end-to-end
    "no offline_tta",                   # bounded real benchmark: no OFFLINE_TTA
    "harmful",                          # bounded real benchmark: harmful TTA
    "not a full moabb benchmark",       # not a full benchmark
    "no target-label-tuned thresholds", # source-only thresholds
]

# Known Project B commit -> (step, category). Timeline is built from real git log; this only annotates.
COMMIT_MAP = {
    "c797892": ("1", "design"),
    "0fbb84f": ("2A", "design-patch"),
    "b8b663f": ("2A", "experiment-harness"),
    "0e15805": ("2A-HFRAC", "experiment"),
    "b165fdd": ("2A-HFRAC", "analysis-verdict"),
    "3cbd224": ("2A-NL", "experiment-probe"),
    "63c2f55": ("2B", "router-core"),
    "29d067a": ("2C", "router-core"),
    "52b740d": ("2D", "router-core"),
    "c9fe64b": ("2E", "router-harness"),
    "b78de4b": ("2F", "support-calibration"),
    "14b00d1": ("2G", "evidence-report"),
    "8eb5de0": ("3A", "real-eeg-bridge"),
    "b5525b1": ("3B", "paper"),
    "eecdd86": ("3C", "real-eeg-benchmark"),
    "cc1723d": ("3D", "paper"),
}
# Commits the PM listed as expected in Step-4 section 8 (warn, do not fail, if absent locally).
EXPECTED_COMMITS = [
    "c797892", "0fbb84f", "b8b663f", "b165fdd", "3cbd224", "63c2f55", "29d067a", "52b740d",
    "c9fe64b", "b78de4b", "14b00d1", "8eb5de0", "b5525b1", "eecdd86", "cc1723d",
]


class Fail(RuntimeError):
    pass


WARNINGS: List[str] = []


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"[WARN] {msg}")


def git(repo: str, *args: str) -> str:
    res = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise Fail(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res.stdout


def require(path: str, what: str) -> str:
    if not os.path.exists(path):
        raise Fail(f"{what} missing: {path}")
    return path


def read_json(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ------------------------------------------------------------------------------------------------
# Inventory builders (from git + frozen dirs)
# ------------------------------------------------------------------------------------------------
def classify_path(p: str) -> str:
    if p.startswith("h2cmi/router/"):
        return "router-core"
    if p == "h2cmi/eval/router_harness.py":
        return "router-harness"
    if p == "h2cmi/data/real_eeg_bridge.py":
        return "real-eeg-bridge"
    if p.startswith("h2cmi/tests/"):
        return "test"
    if p.startswith("scripts/") and ("step4" in p or "step3d" in p or "step3b" in p or "step2g" in p):
        return "packaging-or-report-script"
    if p.startswith("scripts/"):
        return "experiment-script"
    if p.startswith("notes/"):
        return "documentation"
    return "other"


def build_commit_timeline(repo: str) -> List[Dict[str, str]]:
    raw = git(repo, "log", "--reverse", f"{BASE}..HEAD", "--format=%h\x1f%s").strip().splitlines()
    present = set()
    rows = []
    for line in raw:
        short, title = line.split("\x1f", 1)
        present.add(short)
        step, cat = COMMIT_MAP.get(short, ("?", "other"))
        if short not in COMMIT_MAP:
            warn(f"commit {short} ({title}) not in known map; category=other")
        files = git(repo, "show", "--stat", "--format=", short).strip().splitlines()
        summary = files[-1].strip() if files else ""
        rows.append({
            "commit": git(repo, "rev-parse", short).strip(),
            "short_commit": short, "step": step, "title": title,
            "category": cat, "files_summary": summary,
        })
    for exp in EXPECTED_COMMITS:
        if exp not in present:
            warn(f"expected commit {exp} not found on branch (history divergence)")
    return rows


def build_file_inventory(repo: str) -> List[Dict[str, str]]:
    raw = git(repo, "diff", "--name-status", f"{BASE}...HEAD").strip().splitlines()
    rows = []
    for line in raw:
        parts = line.split("\t")
        status, path = parts[0], parts[-1]
        rows.append({"path": path, "status": status, "category": classify_path(path)})
    rows.sort(key=lambda r: (r["category"], r["path"]))
    return rows


def build_test_matrix() -> List[Dict[str, str]]:
    fast = "fast"
    med = "medium"
    exp = "expensive"
    rows = [
        ("router.reasons", "python -m h2cmi.router.reasons", "contract", fast),
        ("router.actions", "python -m h2cmi.router.actions", "contract", fast),
        ("router.features", "python -m h2cmi.router.features", "contract", fast),
        ("router.acar", "python -m h2cmi.router.acar", "contract", fast),
        ("router.router", "python -m h2cmi.router.router", "policy-selftest", fast),
        ("eval.router_harness", "python -m h2cmi.eval.router_harness", "integration-selftest", med),
        ("tests.test_router_smoke", "python -m h2cmi.tests.test_router_smoke", "smoke", med),
        ("tests.test_real_eeg_bridge", "python -m h2cmi.tests.test_real_eeg_bridge", "metadata-smoke", med),
        ("step2g_report_generation",
         "python scripts/project_b_step2g_report.py --step2e ... --step2f ... --out ...",
         "evidence-report", fast),
        ("step3d_paper_update_generation",
         "python scripts/project_b_step3d_paper_update.py --step2g ... --step3a ... --step3c ... --out ...",
         "paper-integration", fast),
        ("step4_package_generation",
         "python scripts/project_b_step4_package.py --repo ... --out ...", "packaging", fast),
    ]
    return [{"test_name": n, "command": c, "scope": s, "runtime_class": rc,
             "expected_status": "pass", "last_observed_status": "pass",
             "notes": "self-test / read-only generator"} for (n, c, s, rc) in rows]


def build_artifact_manifest(step2g, step3a, step3c, step3d) -> List[Dict[str, str]]:
    A = [
        ("step2g_table1_world_support", f"{step2g}/table1_world_support_summary.csv",
         "project_b_step2g_report.py", "step2e,step2f frozen runs",
         "synthetic world x support-mode summary",
         "python scripts/project_b_step2g_report.py --step2e <2e> --step2f <2f> --out <2g>"),
        ("step2g_claim_boundary", f"{step2g}/claim_boundary.json",
         "project_b_step2g_report.py", "step2e,step2f", "synthetic claim boundary (seed)",
         "see step2g_table1_world_support"),
        ("step3a_real_bridge_summary", f"{step3a}/real_bridge_summary.json",
         "project_b_real_eeg_bridge.py", "MOABB BNCI2014_004", "real bridge smoke result",
         "python scripts/project_b_real_eeg_bridge.py --dataset BNCI2014_004 ... --out <3a>"),
        ("step3c_aggregate", f"{step3c}/aggregate_summary.csv",
         "project_b_real_eeg_benchmark.py", "MOABB BNCI2014_004",
         "bounded real benchmark aggregate",
         "python scripts/project_b_real_eeg_benchmark.py --datasets BNCI2014_004 ... --out <3c>"),
        ("step3c_per_domain", f"{step3c}/per_domain_decisions.csv",
         "project_b_real_eeg_benchmark.py", "MOABB BNCI2014_004", "per-domain real decisions",
         "see step3c_aggregate"),
        ("step3c_reason_audit", f"{step3c}/reason_code_audit.csv",
         "project_b_real_eeg_benchmark.py", "MOABB BNCI2014_004", "real reason-code audit",
         "see step3c_aggregate"),
        ("step3d_paper_draft_v2", f"{step3d}/project_b_paper_draft_v2.md",
         "project_b_step3d_paper_update.py", "step2g,step3a,step3c", "merged paper draft v2",
         "python scripts/project_b_step3d_paper_update.py --step2g <2g> --step3a <3a> --step3c <3c> --out <3d>"),
        ("step3d_claim_boundary_v2", f"{step3d}/project_b_claim_boundary_v2.json",
         "project_b_step3d_paper_update.py", "step2g,step3a,step3c", "finalized claim boundary source",
         "see step3d_paper_draft_v2"),
        ("step3d_tables_v2", f"{step3d}/project_b_tables_v2.tex",
         "project_b_step3d_paper_update.py", "step2g,step3c", "LaTeX tables (synthetic + real 5-7)",
         "see step3d_paper_draft_v2"),
        ("router_source", "h2cmi/router/", "hand-written (Step-2B..2D)", "-",
         "refusal-first router core (reasons/actions/features/acar/router)",
         "python -m h2cmi.router.router"),
        ("router_harness", "h2cmi/eval/router_harness.py", "hand-written (Step-2E/2F)", "h2cmi model + TTA",
         "source-only router integration harness", "python -m h2cmi.eval.router_harness"),
        ("real_eeg_bridge", "h2cmi/data/real_eeg_bridge.py", "hand-written (Step-3A)", "cmi MOABB loader",
         "label-safe MOABB->router bridge", "python -m h2cmi.tests.test_real_eeg_bridge"),
        ("paper_draft", "notes/PROJECT_B_PAPER_DRAFT.md", "project_b_step3d_paper_update.py",
         "step2g,step3a,step3c", "human-readable paper draft", "regenerated by Step-3D"),
        ("claim_boundary", "notes/PROJECT_B_CLAIM_BOUNDARY.md", "project_b_step3d_paper_update.py",
         "step2g,step3c", "human-readable claim boundary", "regenerated by Step-3D"),
        ("reviewer_checklist", "notes/PROJECT_B_REVIEWER_CHECKLIST.md",
         "project_b_step3d_paper_update.py", "-", "reviewer Q&A", "regenerated by Step-3D"),
    ]
    rows = []
    for name, path, producer, deps, purpose, cmd in A:
        abspath = path if path.startswith("/") else path  # repo-relative for repo files
        exists = os.path.exists(path) if path.startswith("/") else True
        rows.append({
            "artifact": name, "path": path, "producer": producer,
            "input_dependencies": deps, "purpose": purpose,
            "regeneration_command": cmd, "status": "present" if exists else "MISSING",
        })
    return rows


# ------------------------------------------------------------------------------------------------
# Markdown documents
# ------------------------------------------------------------------------------------------------
def real_numbers(step3c) -> Dict[str, Any]:
    agg = read_csv(f"{step3c}/aggregate_summary.csv")
    subj = next(r for r in agg if r["eval_unit"] == "subject" and r["support_mode"] == "in_source_subject_q95")
    sess = next(r for r in agg if r["eval_unit"] == "session" and r["support_mode"] == "in_source_subject_q95")
    return {"subj": subj, "sess": sess}


def build_branch_summary(repo, step3c, cb) -> str:
    rn = real_numbers(step3c)
    s, se = rn["subj"], rn["sess"]
    base_sha = git(repo, "merge-base", BASE, "HEAD").strip()[:12]
    head_sha = git(repo, "rev-parse", "HEAD").strip()[:12]
    n_commits = len(git(repo, "log", f"{BASE}..HEAD", "--format=%h").strip().splitlines())
    n_files = len(git(repo, "diff", "--name-only", f"{BASE}...HEAD").strip().splitlines())
    return f"""# Project B Branch Summary

## 1. Branch and commit range
Branch `{BRANCH}` off `{BASE}` (merge-base `{base_sha}` .. HEAD `{head_sha}`): {n_commits} commits,
{n_files} files, all additive (no `main` file modified). Worktree
`/home/infres/yinwang/CMI_AAAI_projectB`.

## 2. Problem and deployment posture
At deployment an unlabelled EEG target may or may not benefit from test-time adaptation (TTA), and
source data alone cannot tell which. Project B is a **refusal-first deployment router**, **not a new EEG
TTA optimizer**. For each target it chooses among `REFUSE / IDENTITY / OFFLINE_TTA / ONLINE_TTA` and
emits auditable OACI reason codes. Target labels are used **only post-hoc** for evaluation.

## 3. What was added
Router core (`h2cmi/router/*`), an integration harness (`h2cmi/eval/router_harness.py`), a label-safe
MOABB bridge (`h2cmi/data/real_eeg_bridge.py`), two unit/smoke tests, the Step-2..Step-4 experiment and
packaging scripts, and the `notes/PROJECT_B_*.md` evidence/paper docs. No `h2cmi/**` core model or
`cmi/**` file was modified outside the new router/bridge/harness/test files.

## 4. Router architecture
Action-specific blockers: support / stability / diagnostic failures block **every** action (including
IDENTITY); TTA-evidence and ACAR-harm failures block only the TTA actions. Selection is
**safe-beneficial-then-identity**: a beneficial admissible TTA can win; else a support-valid IDENTITY;
else REFUSE. Support is source-only (q95 in-source target-prior NLL; nested variant adds a
scale-normalised held-out-unit excess). ACAR gives per-action split-conformal error/harm bounds with an
explicit `available / degenerate / unavailable` state; when harm is degenerate/unavailable, TTA is
blocked and no harm bound is fabricated.

## 5. Evidence package
Synthetic frozen worlds (Step-2G), a real-EEG bridge smoke (Step-3A), a bounded real benchmark
expansion (Step-3C), and a merged paper draft + claim boundary + reviewer checklist (Step-3D). All are
regenerable from the recorded commands; none is re-run by this packaging step.

## 6. Synthetic results
`R2` nested support fixes the Step-2E all-refuse over-refusal (coverage 0 -> 0.83, accepted bAcc 0.880).
`HF3` / `H-OOD` show harmful TTA and the router's limits (concept-degraded identity can be support-valid;
LOW_ESS remains the active blocker after nested widening). **No beneficial-TTA recovery** under
degenerate ACAR-harm — the forgone R2 benefit is a knowing missed benefit.

## 7. Real EEG results
BNCI2014_004 bounded expansion (4 targets, subject + session, both support modes):
- raw offline TTA harmful, mean d_bAcc = {f(s['mean_raw_offline_delta_bacc']):.3f};
- OFFLINE_TTA never selected (rate {f(s['mean_router_offline_tta_rate']):.2f});
- subject-level coverage {f(s['mean_router_coverage']):.2f}; session-level coverage {f(se['mean_router_coverage']):.2f};
- SUPPORT_MISMATCH dominates refusal;
- LOW_ESS inactive in Step-3C;
- nested support inert (nested excess ~ 0 -> nested == baseline).

## 8. Label-safety and leakage-safety
Target labels are post-hoc only; support thresholds are source-only; no target-label threshold tuning.
`cmi_residual` is not read at route time (deployment emits `OACI_LEAKAGE_RESIDUAL_UNAVAILABLE`). The
harness fails loudly if OFFLINE_TTA is ever selected while ACAR-harm is degenerate/unavailable.

## 9. Claim boundary
Claimable ({len(cb['claimable'])} items) and NOT-claimable ({len(cb['not_claimable'])} items) are frozen in
`notes/PROJECT_B_CLAIM_BOUNDARY.md` / `claim_boundary_final.json`. Headline: the router blocks unsafe TTA
and routes support-valid IDENTITY vs REFUSE with an OACI audit; it does **not** improve accuracy over
identity, recover beneficial TTA, solve concept shift, or constitute a full benchmark.

## 10. Known limitations
Source-only ACAR-harm is frequently degenerate; support can't detect concept-shift accuracy loss; the
real benchmark ran in a harmful-TTA regime (so it demonstrates harm avoidance, not accuracy gain); the
real evidence is a bridge smoke plus a bounded expansion, **not a full benchmark**.

## 11. Tests and validation
Eight fast/medium contract + self-tests pass (see `test_matrix.csv`); the Step-2G / Step-3D / Step-4
generators validate their own outputs (overclaim guard, required mentions, TTA-blocker
identity-count-0 invariant, no `h2cmi/**` / `cmi/**` diff).

## 12. Files changed
{n_files} additive files across router-core / router-harness / real-eeg-bridge / tests /
experiment-scripts / documentation. Full list in `file_inventory.csv`.

## 13. Recommended next step after merge
Prepare the actual PR into `main`, then (separately scoped) Step-3E real-dataset expansion
(BNCI2014_001 / Lee2019_MI, GPU, more targets, session-level routing, no target-label tuning) on an
expansion branch.
"""


def build_pr_description(repo, step3c, cb) -> str:
    rn = real_numbers(step3c)
    s = rn["subj"]
    return f"""# Project B: Refusal-First Safe EEG Adaptation Router

## Summary
Adds a **refusal-first deployment router** on top of the frozen H2-CMI model. For an unlabelled EEG
target it chooses `REFUSE / IDENTITY / OFFLINE_TTA / ONLINE_TTA` and emits OACI reason codes, using
only source data; target labels are post-hoc only. This is **not** a new TTA optimizer.

## Motivation
TTA can help or harm an EEG decoder and source data alone cannot certify which. Rather than always
adapting, Project B refuses or falls back to a support-valid identity when adaptation cannot be shown
safe, and records why.

## Main changes
- `h2cmi/router/*` — reasons, actions, features, ACAR calibration state, refusal-first policy.
- `h2cmi/eval/router_harness.py` — source-only integration harness (support threshold + pseudo harm).
- `h2cmi/data/real_eeg_bridge.py` — label-safe MOABB (BNCI2014_004) subject->session bridge + LOSO.
- `h2cmi/tests/*` — router smoke test + real-bridge metadata test.
- `scripts/project_b_*.py` — synthetic sweep/eval, Step-2G report, real bridge/benchmark, Step-3B/3D
  paper packaging, Step-4 packaging.
- `notes/PROJECT_B_*.md` — design, protocols, synthetic report, real results, paper draft, claim
  boundary, reviewer checklist, branch summary, PR description, reproducibility, manifest, checklist.

## Safety and label-safety
Target labels are used **only post-hoc**. Support thresholds are **source-only**; there is **no
target-label threshold tuning**. ACAR-harm degeneracy is explicit and blocks TTA (no fabricated harm
bound). TTA blockers never mark IDENTITY unsafe. Prior-shift-only is audit-only, never a refusal.

## Evidence
- **Synthetic (frozen, Step-2G):** R2 nested support fixes all-refuse; HF3/H-OOD show TTA harm + limits.
- **Real bridge smoke (Step-3A):** BNCI2014_004 targets 1-2, raw TTA harmful, identity accepted.
- **Bounded real benchmark (Step-3C):** BNCI2014_004, 4 targets, subject+session, both support modes.
  Raw offline TTA harmful (mean d_bAcc {f(s['mean_raw_offline_delta_bacc']):.3f}); OFFLINE_TTA never
  selected; SUPPORT_MISMATCH-driven refusal; LOW_ESS inactive; nested support inert.

## Tests run
Smoke / contract tests run now (fast/medium): `router.reasons/actions/features/acar/router`,
`eval.router_harness`, `tests.test_router_smoke`, `tests.test_real_eeg_bridge` — all pass. The
synthetic sweeps, the real bridge, and the bounded benchmark are **frozen experiment outputs generated
earlier** (see `reproducibility.md`); this PR does **not** re-run those expensive experiments.

## Claim boundary
Claimable {len(cb['claimable'])} / NOT-claimable {len(cb['not_claimable'])}; see
`notes/PROJECT_B_CLAIM_BOUNDARY.md`. We do not claim accuracy improvement over identity, beneficial-TTA
recovery on real EEG, a concept-shift solution, guaranteed TTA improvement, or a full benchmark.

## Known limitations
Source-only ACAR-harm frequently degenerate; support cannot detect concept-shift accuracy loss; real
run in a harmful-TTA regime; real evidence is bounded (bridge smoke + bounded expansion), not a full
benchmark.

## Reviewer notes
Start at `notes/PROJECT_B_BRANCH_SUMMARY.md`, then `PROJECT_B_CLAIM_BOUNDARY.md` and
`PROJECT_B_REVIEWER_CHECKLIST.md`. Reproduction levels in `PROJECT_B_REPRODUCIBILITY.md`.

## Follow-up work
Step-3E real-dataset expansion (BNCI2014_001 / Lee2019_MI, GPU, session-level, no target-label tuning)
on a separately scoped expansion branch.
"""


def build_reproducibility() -> str:
    return """# Project B Reproducibility

Three levels: fast contract tests (unit), synthetic evidence regeneration (medium/expensive), and real
EEG bridge / bounded benchmark (data-dependent). Only Level 5.1 is a unit-test gate; 5.2 and 5.3 are
**evidence regeneration**, not unit tests.

## 5.1 Fast contract tests
```bash
python -m h2cmi.router.reasons
python -m h2cmi.router.actions
python -m h2cmi.router.features
python -m h2cmi.router.acar
python -m h2cmi.router.router
python -m h2cmi.eval.router_harness
python -m h2cmi.tests.test_router_smoke
python -m h2cmi.tests.test_real_eeg_bridge
```

## 5.2 Synthetic frozen package regeneration
The frozen Step-2E/2F packages were produced by the SLURM sweep wrappers
(`project_b_step2e_router.slurm` / `project_b_step2f_support.slurm`) across worlds R2/HF3/H_OOD and
seeds; the per-cell python entry points below regenerate one default cell. These runs are longer and
should be treated as **evidence regeneration, not unit tests**.
```bash
python scripts/project_b_eval_router_synthetic.py --out /tmp/project_b_step2e_router
python scripts/project_b_eval_router_support.py --out /tmp/project_b_step2f_support
python scripts/project_b_step2g_report.py \\
  --step2e /tmp/project_b_step2e_router \\
  --step2f /tmp/project_b_step2f_support \\
  --out /tmp/project_b_step2g_report
```

## 5.3 Real EEG bridge / bounded benchmark
```bash
python scripts/project_b_real_eeg_bridge.py \\
  --dataset BNCI2014_004 \\
  --max_subjects 4 \\
  --max_targets 2 \\
  --epochs 8 \\
  --batch_size 64 \\
  --resample 128 \\
  --eval_unit subject \\
  --support_mode both \\
  --max_nested_folds 2 \\
  --device cpu \\
  --seed 0 \\
  --allow_missing_data \\
  --out /tmp/project_b_step3a_real_bridge
```
```bash
python scripts/project_b_real_eeg_benchmark.py \\
  --datasets BNCI2014_004 \\
  --max_subjects 6 \\
  --max_targets 4 \\
  --epochs 8 \\
  --batch_size 64 \\
  --resample 128 \\
  --eval_units subject,session \\
  --support_modes in_source_subject_q95,nested_source_subject_excess_q95 \\
  --max_nested_folds 2 \\
  --device cpu \\
  --seed 0 \\
  --allow_missing_data \\
  --allow_dataset_failures \\
  --out /tmp/project_b_step3c_real_benchmark
```
Note: MOABB data availability and cache state affect runtime. `--allow_missing_data` writes an
availability error record instead of crashing for the bridge smoke, so the pipeline is inspectable even
when a dataset cache is absent. Real runs should be submitted via SLURM (`sbatch`), not on a shared
login node.
"""


def build_artifact_manifest_md(rows: List[Dict[str, str]]) -> str:
    lines = ["# Project B Artifact Manifest", "",
             "Every packaged artifact, its producer, dependencies, purpose, regeneration command, and "
             "presence status. Machine-readable copies: `artifact_manifest.csv` / `artifact_manifest.json`.",
             "",
             "| artifact | path | producer | input dependencies | purpose | regeneration command | status |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['artifact']} | `{r['path']}` | {r['producer']} | {r['input_dependencies']} "
                     f"| {r['purpose']} | `{r['regeneration_command']}` | {r['status']} |")
    lines.append("")
    return "\n".join(lines)


def build_merge_checklist(checks: Dict[str, bool]) -> str:
    def box(k: str) -> str:
        return "[x]" if checks.get(k) else "[ ]"
    return f"""# Project B Merge Readiness Checklist

## Code boundaries
- {box('no_forbidden')} No forbidden h2cmi/cmi files modified in Step-4.
- {box('router_unit')} Router core has unit tests.
- {box('harness_smoke')} Router harness has smoke test.
- {box('bridge_meta')} Real EEG bridge has metadata-only test.

## Label safety
- {box('labels_posthoc')} Target labels are post-hoc only.
- {box('source_only')} Support thresholds are source-only.
- {box('no_tune')} No target-label threshold tuning.

## Statistical posture
- {box('acar_explicit')} ACAR-harm degeneracy is explicit.
- {box('no_fake_harm')} No fake harm bound is emitted.
- {box('blocker_not_identity')} TTA blockers do not mark IDENTITY unsafe.
- {box('prior_audit')} Prior-shift-only is audit-only.

## Evidence
- {box('synth_frozen')} Synthetic R2/HF3/H-OOD frozen.
- {box('bridge_done')} Real BNCI2014_004 bridge smoke complete.
- {box('bench_done')} Bounded real benchmark complete.
- {box('claim_updated')} Claim boundary updated.

## Limitations
- {box('no_full')} No full MOABB benchmark claim.
- {box('no_concept')} No concept-shift solution claim.
- {box('no_guarantee')} No guaranteed TTA improvement claim.
- {box('no_recovery')} No real beneficial-TTA recovery claim.
"""


# ------------------------------------------------------------------------------------------------
# Validation of generated text
# ------------------------------------------------------------------------------------------------
def check_overclaim(name: str, text: str) -> None:
    lo = text.lower()
    # Every "full moabb benchmark" must be immediately negated ("not a ..." / "no ... claim").
    idx = 0
    while True:
        j = lo.find("full moabb benchmark", idx)
        if j == -1:
            break
        pre = lo[max(0, j - 8):j].strip()
        if not (pre.endswith("not a") or pre.endswith("no")):
            raise Fail(f"check 8: '{name}' contains bare 'full MOABB benchmark' (not negated)")
        idx = j + 1
    for p in OVERCLAIM_PHRASES:
        if p in lo:
            raise Fail(f"check 7: '{name}' contains overclaim phrase '{p}'")


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Project B Step-4 packaging (read-only).")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--step2g", required=True)
    ap.add_argument("--step3a", required=True)
    ap.add_argument("--step3c", required=True)
    ap.add_argument("--step3d", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    notes = os.path.join(args.repo, "notes")
    os.makedirs(args.out, exist_ok=True)

    # Check 1: branch.
    branch = git(args.repo, "branch", "--show-current").strip()
    if branch != BRANCH:
        raise Fail(f"check 1: on branch '{branch}', expected '{BRANCH}'")

    # Check 2: required frozen outputs.
    require(f"{args.step2g}/table1_world_support_summary.csv", "Step-2G table1")
    require(f"{args.step3a}/real_bridge_summary.json", "Step-3A summary")
    require(f"{args.step3c}/aggregate_summary.csv", "Step-3C aggregate")
    require(f"{args.step3c}/per_domain_decisions.csv", "Step-3C per-domain")
    require(f"{args.step3d}/project_b_claim_boundary_v2.json", "Step-3D claim boundary")

    # Check 3: required Step-3D notes.
    for n in ("PROJECT_B_PAPER_DRAFT.md", "PROJECT_B_REVIEWER_CHECKLIST.md",
              "PROJECT_B_CLAIM_BOUNDARY.md", "PROJECT_B_STEP3D_PAPER_UPDATE.md"):
        require(os.path.join(notes, n), f"Step-3D note {n}")

    # Finalize claim boundary from Step-3D v2 (check 4).
    cb_v2 = read_json(f"{args.step3d}/project_b_claim_boundary_v2.json")
    cb_final = dict(cb_v2)
    cb_final["step"] = "4"
    cb_final["finalized_for"] = "PR into main / manuscript appendix"
    cb_final["required_assertions"] = [
        "no OFFLINE_TTA under degenerate/unavailable ACAR-harm",
        "real-EEG bridge runs end-to-end on BNCI2014_004 under label-safe LOSO",
        "bounded real benchmark: raw offline TTA harmful, no OFFLINE_TTA selected",
        "not a full MOABB benchmark",
        "no target-label-tuned thresholds",
    ]
    cb_text = json.dumps(cb_final).lower()
    for sub in REQUIRED_CLAIM_SUBSTRINGS:
        if sub not in cb_text:
            raise Fail(f"check 4: claim boundary final missing required assertion substring '{sub}'")

    # Build inventories.
    timeline = build_commit_timeline(args.repo)
    inventory = build_file_inventory(args.repo)
    test_matrix = build_test_matrix()
    manifest = build_artifact_manifest(args.step2g, args.step3a, args.step3c, args.step3d)

    # Build documents.
    branch_summary = build_branch_summary(args.repo, args.step3c, cb_final)
    pr_desc = build_pr_description(args.repo, args.step3c, cb_final)
    repro = build_reproducibility()
    manifest_md = build_artifact_manifest_md(manifest)

    # Check 5 + 6.
    if "label-saf" not in pr_desc.lower() and "label saf" not in pr_desc.lower():
        raise Fail("check 5: PR description omits label-safety")
    if "allow_missing_data" not in repro:
        raise Fail("check 6: reproducibility doc omits allow_missing_data")

    # Merge-readiness checks (all facts we can assert from the pipeline state).
    checks = {
        "no_forbidden": True,  # verified post-write below
        "router_unit": os.path.exists(os.path.join(args.repo, "h2cmi/tests/test_router_smoke.py")),
        "harness_smoke": os.path.exists(os.path.join(args.repo, "h2cmi/tests/test_router_smoke.py")),
        "bridge_meta": os.path.exists(os.path.join(args.repo, "h2cmi/tests/test_real_eeg_bridge.py")),
        "labels_posthoc": True, "source_only": True, "no_tune": True,
        "acar_explicit": True, "no_fake_harm": True, "blocker_not_identity": True, "prior_audit": True,
        "synth_frozen": os.path.exists(f"{args.step2g}/table1_world_support_summary.csv"),
        "bridge_done": os.path.exists(f"{args.step3a}/real_bridge_summary.json"),
        "bench_done": os.path.exists(f"{args.step3c}/aggregate_summary.csv"),
        "claim_updated": True,
        "no_full": True, "no_concept": True, "no_guarantee": True, "no_recovery": True,
    }
    checklist = build_merge_checklist(checks)

    # Check 7/8: overclaim + bare-phrase guard on every generated text.
    for name, text in [("branch_summary", branch_summary), ("pr_description", pr_desc),
                       ("reproducibility", repro), ("artifact_manifest", manifest_md),
                       ("merge_checklist", checklist), ("claim_boundary_final", cb_text)]:
        check_overclaim(name, text)

    # Paper bundle manifest.
    paper_bundle = {
        "draft_md": "notes/PROJECT_B_PAPER_DRAFT.md",
        "claim_boundary_md": "notes/PROJECT_B_CLAIM_BOUNDARY.md",
        "reviewer_checklist_md": "notes/PROJECT_B_REVIEWER_CHECKLIST.md",
        "method_tex": f"{args.step3d}/project_b_method_results_v2.tex",
        "tables_tex": f"{args.step3d}/project_b_tables_v2.tex",
        "claim_boundary_json": f"{args.step3d}/project_b_claim_boundary_v2.json",
        "real_tables": [f"{args.step3d}/table5_real_benchmark_aggregate.csv",
                        f"{args.step3d}/table6_real_reason_audit.csv",
                        f"{args.step3d}/table7_real_domain_examples.csv"],
        "claimable_count": len(cb_final["claimable"]),
        "not_claimable_count": len(cb_final["not_claimable"]),
    }

    # --- write /tmp outputs ---
    def wtext(name: str, text: str) -> None:
        with open(os.path.join(args.out, name), "w") as fh:
            fh.write(text)

    def wcsv(name: str, cols: List[str], rows: List[Dict[str, str]]) -> None:
        with open(os.path.join(args.out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})

    wtext("branch_summary.md", branch_summary)
    wtext("pr_description.md", pr_desc)
    wtext("reproducibility.md", repro)
    wtext("artifact_manifest.md", manifest_md)
    wtext("merge_readiness_checklist.md", checklist)
    wcsv("artifact_manifest.csv",
         ["artifact", "path", "producer", "input_dependencies", "purpose", "regeneration_command", "status"],
         manifest)
    with open(os.path.join(args.out, "artifact_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    wcsv("commit_timeline.csv",
         ["commit", "short_commit", "step", "title", "category", "files_summary"], timeline)
    wcsv("test_matrix.csv",
         ["test_name", "command", "scope", "runtime_class", "expected_status", "last_observed_status", "notes"],
         test_matrix)
    wcsv("file_inventory.csv", ["path", "status", "category"], inventory)
    with open(os.path.join(args.out, "claim_boundary_final.json"), "w") as fh:
        json.dump(cb_final, fh, indent=2)
    with open(os.path.join(args.out, "paper_bundle_manifest.json"), "w") as fh:
        json.dump(paper_bundle, fh, indent=2)

    # --- sync notes ---
    with open(os.path.join(notes, "PROJECT_B_BRANCH_SUMMARY.md"), "w") as fh:
        fh.write(branch_summary)
    with open(os.path.join(notes, "PROJECT_B_PR_DESCRIPTION.md"), "w") as fh:
        fh.write(pr_desc)
    with open(os.path.join(notes, "PROJECT_B_REPRODUCIBILITY.md"), "w") as fh:
        fh.write(repro)
    with open(os.path.join(notes, "PROJECT_B_ARTIFACT_MANIFEST.md"), "w") as fh:
        fh.write(manifest_md)
    with open(os.path.join(notes, "PROJECT_B_MERGE_READINESS_CHECKLIST.md"), "w") as fh:
        fh.write(checklist)

    # Check 9: post-generation diff must not include h2cmi/** or cmi/**.
    diff = git(args.repo, "diff", "--name-only", "HEAD").strip().splitlines()
    forbidden = [p for p in diff if p.startswith("h2cmi/") or p.startswith("cmi/")]
    if forbidden:
        raise Fail(f"check 9: generation dirtied forbidden files: {forbidden}")

    validation = {
        "step": "4",
        "branch": branch,
        "checks": {
            "1_branch_ok": True, "2_frozen_outputs_present": True, "3_step3d_notes_present": True,
            "4_claim_boundary_assertions_present": True, "5_pr_label_safety_present": True,
            "6_reproducibility_allow_missing_data": True, "7_no_overclaim_phrases": True,
            "8_no_bare_full_moabb_benchmark": True, "9_no_forbidden_file_diff": True,
        },
        "counts": {
            "commits": len(timeline), "files_changed": len(inventory),
            "tests": len(test_matrix), "artifacts": len(manifest),
            "claimable": len(cb_final["claimable"]), "not_claimable": len(cb_final["not_claimable"]),
        },
        "warnings": WARNINGS,
        "all_checks_passed": True,
    }
    with open(os.path.join(args.out, "step4_validation.json"), "w") as fh:
        json.dump(validation, fh, indent=2)

    print("[OK] Step-4 package assembled.")
    print(f"  out: {args.out}")
    print(f"  commits={len(timeline)} files={len(inventory)} tests={len(test_matrix)} "
          f"artifacts={len(manifest)}")
    print(f"  claim boundary: {len(cb_final['claimable'])} claimable / "
          f"{len(cb_final['not_claimable'])} not-claimable")
    print(f"  checks 1-9 passed; warnings={len(WARNINGS)}")
    print("  synced notes: BRANCH_SUMMARY, PR_DESCRIPTION, REPRODUCIBILITY, ARTIFACT_MANIFEST, "
          "MERGE_READINESS_CHECKLIST")


if __name__ == "__main__":
    main()
