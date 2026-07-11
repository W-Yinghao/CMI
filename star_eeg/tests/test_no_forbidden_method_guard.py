from star_eeg.red_team.no_forbidden_method_guard import evaluate_no_forbidden_method_guard


def test_approved_active_registry_is_clear():
    assert evaluate_no_forbidden_method_guard()["status"] == "PASS"


def test_guard_fails_closed_on_forbidden_active_identifier():
    result = evaluate_no_forbidden_method_guard(
        registry={"extra": "low_rank_adapter"},
        import_paths=["star_eeg.objectives.task_anchor"],
    )
    assert result["status"] == "FAIL"
    assert result["registry_violations"]
