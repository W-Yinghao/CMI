"""C84SR2 full production-path synthetic replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .c84s_common import require, sha256_file, write_json
from .c84sr1_synthetic import run_production_path_calibration


def run_calibration(*, root: str | Path) -> dict[str, Any]:
    result = run_production_path_calibration(root=root, full_chains=2048, branch_chains=8)
    require(result["status"] == "PASS" and result["contexts"] == 944,
            "C84SR2 production-path synthetic status drift")
    require(result["full_scale_Q0_records"] == 9110448 and
            result["full_scale_method_context_rows"] == 18608,
            "C84SR2 full-scale synthetic arithmetic drift")
    root = Path(root)
    summary = {
        "schema_version": "c84sr2_production_path_synthetic_calibration_v1",
        "status": "PASS", "contexts": 944, "candidates_per_context": 81,
        "full_scale_Q0_chains": 2048,
        "full_scale_Q0_records": result["full_scale_Q0_records"],
        "full_scale_method_context_rows": result["full_scale_method_context_rows"],
        "full_scale_primary_gate": result["full_scale_primary_gate"],
        "full_scale_label_frontier_tag": result["full_scale_label_frontier_tag"],
        "underlying_C84SR1_summary_sha256": sha256_file(
            root / "C84SR1_SYNTHETIC_CALIBRATION.json"
        ),
        "selection_freeze_sha256": result["full_selection_manifest_sha256"],
        "scientific_result_sha256": result["full_result_sha256"],
        "branch_results": result["branch_results"],
        "descriptor_compatibility_contract_exercised_by_focused_tests": True,
        "precomputed_method_context_rows_injected": False,
        "real_field_array_access": 0, "real_target_label_access": 0,
        "real_selector_scores": 0, "real_scientific_statistics": 0,
    }
    digest = write_json(root / "C84SR2_SYNTHETIC_CALIBRATION.json", summary)
    return {**summary, "summary_sha256": digest, "root": str(root)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_calibration(root=args.output_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
