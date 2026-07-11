import inspect
import json

import numpy as np
import torch

from h2cmi import analyze_fp_gem, run_fp_gem
from h2cmi.config import DensityConfig, TTAConfig
from h2cmi.density.student_t_mixture import ClassConditionalDensity
from h2cmi.tta.class_conditional import B1A_VARIANTS_BY_NAME, ClassConditionalTTA


class _FakeTSMNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Sequential(torch.nn.Linear(3, 2))

    def forward(self, inputs, domains, parameter_t, fm_mean):
        del domains, parameter_t, fm_mean
        return self.classifier(inputs[:, :3])


def test_classifier_pre_hook_replays_exact_logits():
    torch.manual_seed(0)
    model = _FakeTSMNet()
    X = np.random.default_rng(0).normal(size=(7, 3)).astype(np.float32)
    features, logits, error = run_fp_gem.capture_preclassifier(
        model,
        X,
        np.ones(7, dtype=np.int64),
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=4,
    )
    assert features.shape == (7, 3)
    assert logits.shape == (7, 2)
    assert error == 0.0


def test_fp_gem_and_joint_gem_differ_only_in_prior_update():
    torch.manual_seed(1)
    density = ClassConditionalDensity(2, 2, DensityConfig(n_components=1, cov_rank=1, df=8.0))
    with torch.no_grad():
        density.mu[0, 0] = torch.tensor([-2.0, 0.0])
        density.mu[1, 0] = torch.tensor([2.0, 0.0])
        density.L.zero_()
        density.log_s.fill_(-2.0)
    source_prior = np.asarray([0.5, 0.5])
    U = torch.cat((torch.randn(30, 2) * 0.1 - torch.tensor([2.0, 0.0]),
                   torch.randn(4, 2) * 0.1 + torch.tensor([2.0, 0.0])))
    tta = ClassConditionalTTA(
        density,
        source_prior,
        TTAConfig(em_iters=2, em_lr=0.01),
        2,
        "cpu",
    )
    fixed = tta.fit_variant(U, B1A_VARIANTS_BY_NAME["gen_iterative_diag"], tta_seed=4)
    joint = tta.fit_variant(U, B1A_VARIANTS_BY_NAME["joint_iterative_diag"], tta_seed=4)
    np.testing.assert_array_equal(fixed.pi_T.numpy(), source_prior.astype(np.float32))
    assert joint.pi_T[0] > fixed.pi_T[0]
    assert fixed.transform.kind == joint.transform.kind == "diag_affine"


def test_frozen_config_scope_and_names():
    config = json.loads(run_fp_gem.CONFIG_PATH.read_text())
    assert config["datasets"] == ["BNCI2014_001", "Lee2019_MI"]
    assert config["source_seeds"] == [0, 1, 2]
    assert config["method_name"] == "Fixed-Prior Geometry EM (FP-GEM)"
    assert config["ablation_name"] == "Joint-GEM"
    assert config["new_methods_only"] == ["Joint-GEM", "FP-GEM"]
    assert config["aggregation"]["bootstrap_replicates"] == 10000
    assert config["aggregation"]["bootstrap_seed"] == 20260710


def test_target_labels_are_read_only_after_both_gem_fits():
    source = inspect.getsource(run_fp_gem.run_unit)
    assert "ep.y[adapt_idx]" not in source
    assert source.index("y_eval = np.asarray(ep.y[eval_idx]") > source.index("fixed = tta.fit_variant")
    assert source.index("if smoke:") < source.index("y_eval = np.asarray(ep.y[eval_idx]")


def test_p9_seed_precedes_tsmnet_construction_without_reseed():
    source = inspect.getsource(run_fp_gem.train_source_model)
    assert source.index("_set_seed(seed)") < source.index("model = _build_model")
    assert source.count("_set_seed(seed)") == 1


def test_bootstrap_shape_and_pairing():
    rows = []
    for dataset, targets in (("BNCI2014_001", range(1, 3)), ("Lee2019_MI", range(1, 4))):
        for target in targets:
            for index, method in enumerate(analyze_fp_gem.METHODS):
                rows.append({
                    "dataset": dataset,
                    "target_subject": target,
                    "method": method,
                    "acc": 0.5 + 0.01 * index,
                    "bacc": 0.5 + 0.01 * index,
                })
    methods, contrasts = analyze_fp_gem.bootstrap(rows)
    assert len(methods) == 2 * 4 * 6
    assert len(contrasts) == 2 * 4 * 5
    fp_minus_joint = [row for row in contrasts if row["metric"] == "bacc" and
                      row["comparison"] == "FP-GEM minus Joint-GEM"]
    assert len(fp_minus_joint) == 4
    assert all(abs(row["estimate"] - 0.01) < 1e-12 for row in fp_minus_joint)
