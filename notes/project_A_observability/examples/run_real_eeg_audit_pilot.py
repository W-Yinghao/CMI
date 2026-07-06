"""Project A — real-EEG audited pilot wrapper.

Invokes the Tier-2 runner (`h2cmi.run_real_audited`) on one MOABB dataset / one target subject /
one seed, few epochs, CPU. If the MOABB raw cache is unavailable the runner writes a legal SKIP
artifact and exits 0 — so this never fails on a missing local cache. For a real GPU run use
`scripts/project_A_real_eeg_audit_pilot.slurm`. Outputs go under
`notes/project_A_observability/results/real_eeg_audit_pilot/` (gitignored; regenerable).

Run:  conda run -n icml python notes/project_A_observability/examples/run_real_eeg_audit_pilot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from h2cmi.run_real_audited import main  # noqa: E402


def run():
    out = str(_REPO_ROOT / "notes" / "project_A_observability" / "results" / "real_eeg_audit_pilot")
    argv = ["--dataset", "BNCI2014_001", "--subjects", "1", "2", "3",
            "--target-subject", "1", "--epochs", "2", "--n-classes", "4",
            "--fast", "--device", "cpu", "--seed", "0", "--outdir", out]
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(run())
