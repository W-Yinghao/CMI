import inspect
import json

import numpy as np
import torch

from h2cmi import analyze_fp_gem_prevalence as analyzer
from h2cmi import prepare_fp_gem_prevalence as builder
from h2cmi import run_fp_gem_prevalence as runner


def test_repeat_crop_and_frozen_prevalence_counts():
    original = [f"trial-{index}" for index in range(50)]
    labels = {trial: int(index >= 25) for index, trial in enumerate(original)}
    q01, counts01 = builder.batch_ids("0.1", original, labels)
    q05, counts05 = builder.batch_ids("0.5", original, labels)
    q09, counts09 = builder.batch_ids("0.9", original, labels)
    assert counts01 == [5, 45]
    assert counts05 == [25, 25]
    assert counts09 == [45, 5]
    assert q05 == original
    assert q01 == original[:5] + original[25:] + original[25:45]
    assert q09 == original[:25] + original[:20] + original[25:30]
    assert builder.batch_ids("0.1", original, labels)[0] == q01


class _AliasedBuffers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        shared_mean = torch.zeros(2)
        shared_var = torch.ones(2)
        self.register_buffer("running_mean", shared_mean)
        self.register_buffer("running_mean_test", shared_mean)
        self.register_buffer("running_var", shared_var)
        self.register_buffer("running_var_test", shared_var)


def test_alias_safe_checkpoint_loading_precondition():
    module = _AliasedBuffers()
    assert module.running_mean.data_ptr() == module.running_mean_test.data_ptr()
    assert module.running_var.data_ptr() == module.running_var_test.data_ptr()
    cloned = runner.break_spd_running_buffer_aliases(module)
    assert cloned == 4
    assert module.running_mean.data_ptr() != module.running_mean_test.data_ptr()
    assert module.running_var.data_ptr() != module.running_var_test.data_ptr()


def test_frozen_config_and_manifest_scope():
    config = json.loads(runner.CONFIG_PATH.read_text())
    manifest = json.loads(runner.MANIFEST_PATH.read_text())
    assert config["dataset"] == "Lee2019_MI"
    assert config["source_seeds"] == [0, 1, 2]
    assert config["q_values"] == [0.1, 0.5, 0.9]
    assert config["methods"] == list(runner.METHODS)
    assert config["execution"]["fresh_source_training_permitted"] is False
    assert config["statistics"]["bootstrap_replicates"] == 10000
    assert config["statistics"]["bootstrap_seed"] == 20260710
    assert len(manifest["units"]) == 162
    for unit in manifest["units"]:
        assert unit["batches"]["0.5"]["trial_ids"] == unit["adapt_reservoir_trial_ids"]
        assert set(unit["eval_trial_ids"]).isdisjoint(unit["adapt_reservoir_trial_ids"])


def test_adaptation_api_excludes_q_labels_and_performance():
    signature = inspect.signature(runner.adapt_methods)
    forbidden = {"q", "labels", "y", "y_adapt", "target_performance"}
    assert forbidden.isdisjoint(signature.parameters)
    source = inspect.getsource(runner.adapt_methods)
    assert "ep.y[adapt_idx]" not in source
    assert "train_source_model" not in source
    assert "trainer.fit" not in inspect.getsource(runner)


def test_evaluation_labels_are_read_after_all_adaptation_fits():
    source = inspect.getsource(runner.run_unit)
    label_read = source.index("y_eval = np.asarray")
    assert label_read > source.index('for q in ("0.1", "0.9")')
    assert label_read > source.index("adapt_methods(", source.index('for q in ("0.1", "0.9")'))
    assert "context[\"ep\"].y" not in source[:label_read]


def test_seed_average_sensitivity_and_paired_bootstrap():
    old_replicates = analyzer.BOOTSTRAP_REPLICATES
    analyzer.BOOTSTRAP_REPLICATES = 100
    try:
        subjects = [1, 2]
        per_subject = []
        cube = np.zeros((2, len(runner.METHODS), 3), dtype=np.float64)
        for si, subject in enumerate(subjects):
            for mi, method in enumerate(runner.METHODS):
                sensitivity = 0.01 * mi
                cube[si, mi] = [0.5 - sensitivity, 0.5, 0.5 + sensitivity]
                per_subject.append({
                    "target_subject": subject,
                    "method": method,
                    "prevalence_sensitivity": sensitivity,
                    "endpoint_mean_bacc": 0.5,
                    "worst_prevalence_bacc": 0.5 - sensitivity,
                    "endpoint_mean_prediction_disagreement": sensitivity,
                })
        sensitivity_rows, _, primary = analyzer.bootstrap_endpoints(per_subject, cube)
        fp = next(row for row in sensitivity_rows if row["row_type"] == "method_summary" and row["method"] == "FP-GEM")
        assert abs(fp["estimate"] - 0.05) < 1e-12
        assert primary["comparison"] == "FP-GEM minus Joint-GEM"
        assert abs(primary["estimate"] - 0.01) < 1e-12
        assert primary["support_rule_pass"] is False
    finally:
        analyzer.BOOTSTRAP_REPLICATES = old_replicates
