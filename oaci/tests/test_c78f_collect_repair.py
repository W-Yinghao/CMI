from __future__ import annotations

import inspect

from oaci.conditioned_ceiling_coverage import c78f_collect_repair as repair


def test_repair_is_additive_and_narrow():
    protocol = repair.build_protocol()
    assert protocol["failure_signature"] == "KeyError: rows"
    assert protocol["repair"].startswith("map descriptor row_count")
    assert protocol["original_collector_unchanged"] is True


def test_repair_has_no_heavy_execution_scope():
    scope = repair.build_protocol()["scope"]
    assert scope["training"] == 0
    assert scope["forward"] == 0
    assert scope["GPU"] == 0
    assert scope["target_label_reads"] == 0
    assert scope["target_metrics"] == 0


def test_fixed_view_uses_frozen_row_count_abi():
    source = inspect.getsource(repair._fixed_view_rows)
    assert 'descriptor["row_count"]' in source
    assert 'descriptor["rows"]' not in source


def test_failed_job_is_retained():
    assert repair.FAILED_COLLECTOR_JOB == "893052"
