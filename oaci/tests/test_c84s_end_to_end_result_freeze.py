from pathlib import Path

import pytest

from oaci.multidataset import c84s_analysis as analysis
from oaci.multidataset import c84s_synthetic_end_to_end as synthetic
from oaci.multidataset.c84s_common import C84SContractError, read_json


def test_s0_s20_are_executable_and_all_pass() -> None:
    rows = synthetic.synthetic_calibration_rows()
    assert len(rows) == 21
    assert [row["scenario"] for row in rows] == [f"S{index}" for index in range(21)]
    assert all(row["pass"] == 1 for row in rows)
    assert all(row["real_label_access"] == row["real_selector_score"] == 0 for row in rows)


def test_production_entrypoint_writes_every_stage_c_table_and_result(tmp_path: Path) -> None:
    result = analysis.run_analysis_and_freeze(
        synthetic.synthetic_method_context_rows("S3"),
        selection_freeze_identity=synthetic.SELECTION_IDENTITY,
        evaluation_view_identity=synthetic.EVALUATION_IDENTITY,
        final_root=tmp_path / "result", draws=64, synthetic=True,
    )
    assert result["primary_gate"].startswith("C84-A_")
    assert result["artifact_manifest_table_count"] == len(analysis.RESULT_TABLE_FIELDS)
    manifest = read_json(tmp_path / "result/C84S_RESULT_ARTIFACT_MANIFEST.json")
    assert manifest["table_count"] == len(analysis.RESULT_TABLE_FIELDS)
    assert {row["path"] for row in manifest["artifacts"]} == set(analysis.RESULT_TABLE_FIELDS)
    assert (tmp_path / "result/C84S_RESULT.json").is_file()


def test_mixed_context_schema_rejects_chain_and_duplicate_rows() -> None:
    rows = synthetic.synthetic_method_context_rows("S0")
    changed = list(rows)
    changed[0] = {**changed[0], "chain": 0}
    with pytest.raises(C84SContractError, match="field-set"):
        analysis.validate_method_context_rows(changed)
    changed = list(rows)
    changed[-1] = dict(changed[0])
    with pytest.raises(C84SContractError, match="duplicate"):
        analysis.validate_method_context_rows(changed)


def test_atomic_stage_c_failure_leaves_no_result_root(tmp_path: Path) -> None:
    final = tmp_path / "result"
    with pytest.raises(C84SContractError, match="injected"):
        analysis.run_analysis_and_freeze(
            synthetic.synthetic_method_context_rows("S0"),
            selection_freeze_identity=synthetic.SELECTION_IDENTITY,
            evaluation_view_identity=synthetic.EVALUATION_IDENTITY,
            final_root=final, draws=64, synthetic=True,
            failure_injection_after="C84S_RESULT_ARTIFACT_MANIFEST.json",
        )
    assert not final.exists()
