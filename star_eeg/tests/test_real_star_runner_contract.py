import inspect
import json

import pytest

from star_eeg.training.real_star_runner import (
    RealStarConfig,
    require_launch_approval,
    run_real_star,
)
from star_eeg.runners.run_star00b_preflight import _blind_chain


def test_real_runner_signature_has_no_evaluation_or_non_source_parameter():
    signature = str(inspect.signature(run_real_star)).lower()
    assert "target" not in signature
    assert "source_val" not in signature
    assert "launch_approval_path" in signature


def test_real_config_freezes_route_b_training_semantics():
    config = RealStarConfig("H200_STAR_TRUE", 0, 10)
    config.validate()
    assert config.batch_size == 64
    assert config.mask_ratio == 0.5
    assert config.gradient_clip_norm == 1.0
    assert config.ssl_loss == "masked_mse_mean"
    assert config.zero_grad_semantics == "set_to_none_true"
    assert config.mixed_precision == "disabled_fp32"
    with pytest.raises(ValueError):
        RealStarConfig("H200_STAR_TRUE", 0, 10, learning_rate=1e-4).validate()
    with pytest.raises(ValueError):
        RealStarConfig("H200_STAR_TRUE", 0, 9).validate()


def test_formal_path_is_blocked_without_new_pm_manifest(tmp_path):
    require_launch_approval(RealStarConfig("H200_STAR_TRUE", 0, 20), None)
    with pytest.raises(PermissionError):
        require_launch_approval(RealStarConfig("H200_STAR_TRUE", 0, 25), None)
    formal = RealStarConfig("H200_STAR_TRUE", 0, 3750)
    with pytest.raises(PermissionError):
        require_launch_approval(formal, None)
    blocked = tmp_path / "blocked.json"
    blocked.write_text(json.dumps({"STAR_01_SCIENTIFIC_TRAINING": "BLOCKED"}))
    with pytest.raises(PermissionError):
        require_launch_approval(formal, blocked)
    approved = tmp_path / "approved.json"
    approved.write_text(json.dumps({"STAR_01_SCIENTIFIC_TRAINING": "APPROVED"}))
    require_launch_approval(formal, approved)


def test_blind_chain_separates_source_val_gate_and_blocks_target():
    chain = _blind_chain()
    source_gate = chain["stages"][2]
    target_scoring = chain["stages"][3]
    assert source_gate["separate_from_training_process"] is True
    assert source_gate["source_val_labels_for_task_gate_only"] is True
    assert source_gate["target_test_samples_or_labels_read"] is False
    assert target_scoring["dependency"] == "afterok_source_only_audit"
    assert target_scoring["currently_blocked"] is True
