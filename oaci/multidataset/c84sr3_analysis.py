"""C84SR3 result derivation with Lee B32 recorded as input-unavailable."""
from __future__ import annotations

from typing import Any

from .c84sr1_analysis import run_analysis_and_freeze_v2 as historical_runner
from .c84sr3_common import METHOD_CONTEXT_ROWS, expected_methods


def run_analysis_and_freeze_v3(*args: Any, **kwargs: Any):
    return historical_runner(
        *args, **kwargs, method_provider=expected_methods,
        expected_row_count=METHOD_CONTEXT_ROWS,
        result_schema="c84sr3_result_v3",
        manifest_schema="c84sr3_result_artifact_manifest_v3",
    )
