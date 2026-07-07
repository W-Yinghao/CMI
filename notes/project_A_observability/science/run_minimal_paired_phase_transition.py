"""Project A Step 12 — thin runner for the minimal-paired phase-transition study.

Runs `h2cmi.observability.minimal_paired.run` with the pre-registered grid and writes the tracked
summary. Kept as a script (per the Step-12 plan) so the science pipeline reads as a sequence of
explicit stages; all logic lives in the importable, tested module.

    python notes/project_A_observability/science/run_minimal_paired_phase_transition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from h2cmi.observability.minimal_paired import run, write_md          # noqa: E402
from h2cmi.observability.result_index import write_json_lf            # noqa: E402

_OUT = _REPO / "notes/project_A_observability/results_summaries"


def main():
    summary = run(n_repeats=50, seed=0)
    write_json_lf(_OUT / "step12_minimal_paired_phase_transition.json", summary)
    write_md(summary, _OUT / "step12_minimal_paired_phase_transition.md")
    print(f"minimal_paired -> phase_transition_observed={summary['phase_transition_observed']} "
          f"best_k={summary['best_k_overall']} per_shift={summary['phase_transition_k_per_shift']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
