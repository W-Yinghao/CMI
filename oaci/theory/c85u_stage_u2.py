"""C85U Stage U2: utility field plus immutable actions to endpoint replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from oaci.multidataset.c84s_common import require

from .c85u_historical_decision_replay import run_historical_decision_replay


SELECTION_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/"
    "oaci-c84s-analysis-v5/stage_b_selection_freeze"
)
RESULT_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/"
    "oaci-c84s-analysis-v5/stage_c_scientific_result"
)


def run_stage_u2(
    *, utility_root: str | Path, output_root: str | Path,
) -> dict[str, Any]:
    utility = Path(utility_root).resolve()
    output = Path(output_root).resolve()
    require(utility.name == "stage_u1_candidate_utility",
            "C85U U2 utility-root identity drift")
    require(output.parent == utility.parent and output.name == "stage_u2_historical_replay",
            "C85U U2 output-root identity drift")
    return run_historical_decision_replay(
        utility_root=utility,
        selection_root=SELECTION_ROOT,
        result_manifest_path=RESULT_ROOT / "C84S_RESULT_ARTIFACT_MANIFEST.json",
        historical_table_path=RESULT_ROOT / "method_context_decisions.csv",
        final_root=output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run-real",))
    parser.add_argument("--utility-root", required=True)
    parser.add_argument("--output-root", required=True)
    arguments = parser.parse_args(argv)
    result = run_stage_u2(
        utility_root=arguments.utility_root, output_root=arguments.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_stage_u2"]
