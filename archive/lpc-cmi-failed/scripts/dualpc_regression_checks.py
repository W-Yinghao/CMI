"""Fast DualPC regression checks with no EEG data dependency.

These checks guard the semantics that the AAAI DualPC protocol relies on:
method-specific decoder margins, GLS weighting, weighted-loss equivalence,
decoder-domain validity, readiness parsing, and protocol command generation.
They are intentionally small and CPU-only.

Run:
  /home/infres/yinwang/anaconda3/envs/icml/bin/python scripts/dualpc_regression_checks.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from analysis.dualpc_decision_gate import evaluate as evaluate_decision
from analysis.dualpc_paper_summary import summarize_paths
from analysis.dualpc_readiness import parse_file
from cmi.eval.metrics import add_decoder_valid_means, domain_class_span_stats
from cmi.methods.regularizers import DomainPosteriors, empirical_priors
from cmi.train.trainer import _label_shift_weights, resolve_dec_margin, train_model
from scripts.dualpc_protocol import build as protocol_build
from scripts.dualpc_protocol import decision_cmd, paper_summary_cmd, readiness_cmd, selector_cmd
from scripts.dualpc_paper_status import collect_status
from scripts.dualpc_slurm_plan import _is_post, _write_array_script, _write_post_script, _write_task_file
from cmi.run_lambda_select import _label as selector_label


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _assert_close(a, b, tol=1e-6, msg=""):
    if abs(float(a) - float(b)) > tol:
        raise AssertionError(msg or f"{a} != {b}")


class TinyBackbone(nn.Module):
    def __init__(self, n_chans=2, n_times=4, n_cls=2, z_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_chans * n_times, z_dim),
            nn.Tanh(),
        )
        self.task_head = nn.Linear(z_dim, n_cls)

    def forward(self, x):
        z = self.encoder(x.float())
        return self.task_head(z), z


def check_dec_margin_defaults():
    _assert_close(resolve_dec_margin("dualpc", None), 0.0, msg="dualpc tau default changed")
    _assert_close(resolve_dec_margin("dualpc_marginal", None), 0.0,
                  msg="dualpc_marginal tau default changed")
    _assert_close(resolve_dec_margin("dualc", None), 0.02, msg="dualc tau default changed")
    _assert_close(resolve_dec_margin("dual", None), 0.02, msg="dual tau default changed")
    _assert_close(resolve_dec_margin("dualpc", 0.03), 0.03, msg="explicit tau override ignored")
    _assert(selector_label("dualpc", 0.05, 0.05, 0.0, 0.0) == "dualpc:0.05:0.05",
            "default dualpc tau should be omitted from labels")
    _assert(selector_label("dualpc", 0.05, 0.05, 0.02, 0.0) == "dualpc:0.05:0.05:tau=0.02",
            "non-default dualpc tau should be explicit")


def check_label_shift_weights():
    y = np.array([0] * 18 + [1] * 2 + [0] * 2 + [1] * 18, dtype=np.int64)
    d = np.array([0] * 20 + [1] * 20, dtype=np.int64)
    w = _label_shift_weights(y, d, n_dom=2, n_cls=2)
    _assert(np.isfinite(w).all() and (w > 0).all(), "GLS weights must be finite and positive")
    _assert_close(w.mean(), 1.0, tol=1e-6, msg="GLS weights should be mean-normalized")

    def class1_rate(dom, weight=None):
        m = d == dom
        if weight is None:
            return float(y[m].mean())
        return float((weight[m] * y[m]).sum() / weight[m].sum())

    raw_gap = abs(class1_rate(0) - class1_rate(1))
    rw_gap = abs(class1_rate(0, w) - class1_rate(1, w))
    _assert(rw_gap < raw_gap, "GLS weights should reduce per-domain label-prior disparity")


def check_weighted_losses_all_ones_equivalence():
    torch.manual_seed(7)
    n, z_dim, n_dom, n_cls = 12, 5, 3, 2
    z = torch.randn(n, z_dim)
    y = torch.tensor([0, 1] * 6, dtype=torch.long)
    d = torch.tensor([0, 0, 1, 1, 2, 2] * 2, dtype=torch.long)
    priors = empirical_priors(y.numpy(), d.numpy(), n_dom, n_cls)
    post = DomainPosteriors(z_dim, n_dom, n_cls, priors, device="cpu")
    ones = torch.ones(n)

    checks = [
        (post.posterior_loss(z, y, d), post.posterior_loss(z, y, d, weight=ones), "posterior_loss"),
        (post.iib_ce_h(z, y, d), post.iib_ce_h(z, y, d, weight=ones), "iib_ce_h"),
        (post.dec_cmi(z, y, d), post.dec_cmi(z, y, d, weight=ones), "dec_cmi"),
        (post.dec_cmi_residual(z, y, d), post.dec_cmi_residual(z, y, d, weight=ones), "dec_cmi_residual"),
        (post.dec_js_residual(z, d), post.dec_js_residual(z, d, weight=ones), "dec_js_residual"),
        (post.reg("lpc_prior", z, y), post.reg("lpc_prior", z, y, weight=ones), "reg_lpc_prior"),
        (post.reg("marginal", z, y), post.reg("marginal", z, y, weight=ones), "reg_marginal"),
    ]
    for plain, weighted, name in checks:
        _assert_close(plain.item(), weighted.item(), tol=1e-6,
                      msg=f"{name} all-ones weighting changed the value")


def check_decoder_validity_summaries():
    invalid = domain_class_span_stats(
        y=np.array([0, 0, 1, 1]), d=np.array([0, 0, 1, 1]), n_cls=2
    )
    _assert(not invalid["decoder_valid"], "single-class domains must be invalid for decoder concept probes")
    _assert(invalid["decoder_min_domain_classes"] == 1, "invalid min-domain class span not recorded")
    _assert_close(invalid["decoder_single_class_frac"], 1.0, msg="invalid single-class fraction wrong")

    valid = domain_class_span_stats(
        y=np.array([0, 1, 0, 1]), d=np.array([0, 0, 1, 1]), n_cls=2
    )
    _assert(valid["decoder_valid"], "class-spanning domains should be valid")
    _assert(valid["decoder_min_domain_classes"] == 2, "valid min-domain class span wrong")

    summary = {}
    add_decoder_valid_means(summary, [
        {"decoder_valid": False, "decoder_js_res": 9.0},
        {"decoder_valid": True, "decoder_js_res": 0.25, "decoder_cmi_res": 0.5},
    ])
    _assert(summary["decoder_valid_n"] == 1, "valid-only decoder count wrong")
    _assert_close(summary["decoder_js_res_valid_mean"], 0.25,
                  msg="valid-only JS mean should ignore invalid records")
    _assert_close(summary["decoder_cmi_res_valid_mean"], 0.5,
                  msg="valid-only residual mean should ignore invalid records")

    empty_summary = {}
    add_decoder_valid_means(empty_summary, [{"decoder_valid": False, "decoder_js_res": 9.0}])
    _assert(empty_summary["decoder_valid_n"] == 0, "empty valid-only decoder count wrong")
    _assert(empty_summary["decoder_js_res_valid_mean"] is None,
            "valid-only decoder mean should be null when no fold is valid")


def check_readiness_parsing():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        synth = root / "source_select.json"
        synth.write_text(json.dumps({
            "config": {"source_select": True},
            "rows": [],
            "summary": {
                "concept": {
                    "dualpc_select": {
                        "selected_hist": {"dualpc:0.05:0.05": 1},
                        "target_bacc": 0.5,
                        "cond_kl_rw": 0.01,
                        "pz_kl_rw": 0.02,
                        "py_js_rw": 0.003,
                    }
                }
            },
        }))
        rows = parse_file(str(synth))
        _assert(len(rows) == 1, "source-selector synthetic smoke should produce one readiness row")
        _assert(rows[0]["suite"] == "synthetic_selector", "source-selector smoke misclassified")
        _assert(rows[0]["status"] == "PASS", "source-selector smoke should pass path checks")

        selector = root / "selector.json"
        selector.write_text(json.dumps({
            "selection_records": [{
                "target": "1",
                "selected": "dualpc:0.05:0.05",
                "candidates": [{
                    "config": "dualpc:0.05:0.05",
                    "source_val_bacc": 0.5,
                    "selection_probe_valid": True,
                    "selector_penalty": 0.006,
                    "select_cond_kl_rw": 0.001,
                    "select_pz_kl_rw": 0.002,
                    "select_py_js_rw": 0.003,
                }],
                "target_bacc": 0.6,
                "final_selected_probe_valid": True,
                "final_selected_probe_penalty": 0.012,
                "final_selected_cond_kl_rw": 0.004,
                "final_selected_pz_kl_rw": 0.005,
                "final_selected_py_js_rw": 0.006,
                "final_selected_decoder_valid": True,
            }],
        }))
        rows = parse_file(str(selector))
        suites = {r["suite"]: r for r in rows}
        _assert(set(suites) == {"selector", "selector_final"},
                "selector JSON should emit candidate and final readiness rows")
        _assert(suites["selector"]["status"] == "PASS", "selected candidate row should pass")
        _assert(suites["selector_final"]["status"] == "PASS", "final selected row should pass")
        _assert_close(suites["selector_final"]["acc"], 0.6, msg="final selector target bAcc not propagated")

        runner = root / "runner.json"
        runner.write_text(json.dumps({
            "summary": {
                "erm:0": {
                    "subject_balanced_acc": 0.50,
                    "leakage_kl_rw": 0.10,
                    "marginal_leakage_kl_rw": 0.20,
                    "decoder_js_res_rw_valid_mean": 0.010,
                },
                "lpc_prior:0.1": {
                    "subject_balanced_acc": 0.52,
                    "leakage_kl_rw": 0.05,
                    "marginal_leakage_kl_rw": 0.18,
                    "decoder_js_res_rw_valid_mean": 0.012,
                },
                "dualpc:0.1:0.05": {
                    "subject_balanced_acc": 0.53,
                    "leakage_kl_rw": 0.04,
                    "marginal_leakage_kl_rw": 0.17,
                    "decoder_js_res_rw_valid_mean": 0.011,
                    "decoder_valid_n": 3,
                },
            },
        }))
        rows = parse_file(str(runner))
        compare = [r for r in rows if r["suite"] == "runner_compare"]
        _assert(len(compare) == 2, "runner with ERM/LPC baselines should emit two comparison rows")
        _assert(all(r["status"] == "PASS" for r in compare), "safe runner comparisons should pass")
        erm_cmp = next(r for r in compare if r["baseline"] == "erm:0")
        _assert_close(erm_cmp["delta_acc"], 0.03, msg="runner comparison acc delta wrong")
        _assert_close(erm_cmp["delta_cond_kl_rw"], -0.06, msg="runner comparison cond delta wrong")
        _assert_close(erm_cmp["delta_pz_kl_rw"], -0.03, msg="runner comparison P(z) delta wrong")
        _assert_close(erm_cmp["delta_py_js_rw"], 0.001, msg="runner comparison P(y|Z) delta wrong")


def check_paper_summary_parsing():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        runner = root / "runner.json"
        runner.write_text(json.dumps({
            "config": {"dataset": "BNCI2014_001", "seed": 0},
            "summary": {
                "erm:0": {
                    "per_target_balanced_acc_mean": 0.50,
                    "leakage_kl_rw": 0.10,
                    "marginal_leakage_kl_rw": 0.20,
                    "decoder_js_res_rw_valid_mean": 0.010,
                },
                "lpc_prior:0.1": {
                    "per_target_balanced_acc_mean": 0.51,
                    "leakage_kl_rw": 0.05,
                    "marginal_leakage_kl_rw": 0.18,
                    "decoder_js_res_rw_valid_mean": 0.012,
                },
                "dualpc:0.1:0.05": {
                    "per_target_balanced_acc_mean": 0.54,
                    "leakage_kl_rw": 0.04,
                    "marginal_leakage_kl_rw": 0.17,
                    "decoder_js_res_rw_valid_mean": 0.011,
                    "decoder_valid_n": 3,
                },
            },
        }))
        selector = root / "selector.json"
        selector.write_text(json.dumps({
            "config": {"dataset": "BNCI2014_001", "seed": 0},
            "selection_records": [{
                "target": "1",
                "selected": "dualpc:0.05:0.05",
                "target_bacc": 0.60,
                "target_erm_bacc": 0.55,
                "final_selected_probe_valid": True,
                "final_selected_probe_penalty": 0.012,
                "final_selected_cond_kl_rw": 0.004,
                "final_selected_pz_kl_rw": 0.005,
                "final_selected_py_js_rw": 0.006,
            }],
        }))
        out = summarize_paths([str(runner), str(selector)])
        _assert(out["counts"] == {"PASS": 3, "WARN": 0, "FAIL": 0},
                "paper summary aggregate counts wrong")
        compare = out["comparison_summary"]
        _assert(len(compare) == 2, "paper summary should compare DualPC vs ERM and LPC")
        erm_cmp = next(r for r in compare if r["baseline"] == "erm:0")
        _assert_close(erm_cmp["delta_acc_mean"], 0.04, msg="paper summary acc delta wrong")
        _assert_close(erm_cmp["delta_cond_kl_rw_mean"], -0.06,
                      msg="paper summary conditional delta wrong")
        _assert_close(erm_cmp["delta_pz_kl_rw_mean"], -0.03,
                      msg="paper summary P(z) delta wrong")
        _assert_close(erm_cmp["delta_py_js_rw_mean"], 0.001,
                      msg="paper summary P(y|Z) delta wrong")
        selector_rows = out["selector_summary"]
        _assert(len(selector_rows) == 1 and selector_rows[0]["status"] == "PASS",
                "paper summary selector aggregate should pass")
        _assert_close(selector_rows[0]["final_record_frac"], 1.0,
                      msg="paper summary should require final selector records")
        _assert_close(selector_rows[0]["final_probe_valid_frac"], 1.0,
                      msg="paper summary final probe valid fraction wrong")
        _assert_close(selector_rows[0]["delta_target_bacc_vs_erm_mean"], 0.05,
                      msg="paper summary selector target delta wrong")

        legacy_selector = root / "legacy_selector.json"
        legacy_selector.write_text(json.dumps({
            "config": {"dataset": "LegacySet", "seed": 0},
            "selection_records": [{
                "target": "1",
                "selected": "dualpc:0.05:0.05",
                "target_bacc": 0.60,
                "target_erm_bacc": 0.55,
                "candidates": [{
                    "config": "dualpc:0.05:0.05",
                    "selection_probe_valid": True,
                    "selector_penalty": 0.012,
                    "select_cond_kl_rw": 0.004,
                    "select_pz_kl_rw": 0.005,
                    "select_py_js_rw": 0.006,
                }],
            }],
        }))
        legacy = summarize_paths([str(legacy_selector)])
        _assert(legacy["counts"] == {"PASS": 0, "WARN": 1, "FAIL": 0},
                "paper summary should warn when final selector probes are absent")
        legacy_sel = legacy["selector_summary"][0]
        _assert_close(legacy_sel["final_record_frac"], 0.0,
                      msg="legacy selector final record fraction wrong")
        _assert(legacy_sel["target_bacc_mean"] is None,
                "paper summary should not aggregate selection-time target bAcc as final evidence")


def check_decision_gate():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        readiness = root / "readiness.json"
        paper = root / "paper_summary.json"
        readiness.write_text(json.dumps({
            "counts": {"PASS": 2, "WARN": 1, "FAIL": 0},
            "rows": [
                {"status": "WARN", "note": "baseline", "suite": "synthetic", "method": "erm"},
                {"status": "PASS", "note": "synthetic gates pass", "suite": "synthetic",
                 "group": "null_prior", "method": "dualpc"},
                {"status": "PASS", "note": "synthetic gates pass", "suite": "synthetic",
                 "group": "concept", "method": "dualpc"},
                {"status": "PASS", "note": "synthetic gates pass", "suite": "synthetic",
                 "group": "all_three", "method": "dualpc"},
            ],
        }))
        paper.write_text(json.dumps({
            "comparison_summary": [{
                "status": "PASS",
                "task": "loso:BNCI2014_001",
                "method": "dualpc:0.1:0.05",
                "baseline": "erm:0",
                "delta_acc_mean": 0.01,
                "delta_cond_kl_rw_mean": -0.01,
                "delta_pz_kl_rw_mean": -0.02,
                "delta_py_js_rw_mean": 0.001,
            }, {
                "status": "PASS",
                "task": "loso:BNCI2014_001",
                "method": "dualpc:0.1:0.05",
                "baseline": "lpc_prior:0.1",
                "delta_acc_mean": 0.02,
                "delta_cond_kl_rw_mean": -0.005,
                "delta_pz_kl_rw_mean": -0.01,
                "delta_py_js_rw_mean": -0.001,
            }],
            "selector_summary": [{
                "status": "PASS",
                "task": "loso:BNCI2014_001",
                "final_record_frac": 1.0,
                "final_probe_valid_frac": 1.0,
            }],
        }))
        ok = evaluate_decision(str(readiness), str(paper), min_comparison_tasks=1, min_selector_tasks=1,
                               required_synthetic_groups=["null_prior", "concept", "all_three"])
        _assert(ok["decision"] == "HEADLINE_READY", "safe paper evidence should pass decision gate")

        pending = evaluate_decision(str(root / "missing_ready.json"), str(paper), min_comparison_tasks=1)
        _assert(pending["decision"] == "PENDING", "missing readiness should be pending")

        readiness_missing_synth = root / "readiness_missing_synth.json"
        readiness_missing_synth.write_text(json.dumps({
            "counts": {"PASS": 1, "WARN": 0, "FAIL": 0},
            "rows": [{"status": "PASS", "suite": "synthetic", "group": "concept", "method": "dualpc"}],
        }))
        missing_synth = evaluate_decision(
            str(readiness_missing_synth), str(paper), min_comparison_tasks=1, min_selector_tasks=1,
            required_synthetic_groups=["null_prior", "concept", "all_three"]
        )
        _assert(missing_synth["decision"] == "PENDING",
                "missing required synthetic groups should keep decision pending")

        paper_missing_baseline = root / "paper_missing_baseline.json"
        paper_missing_baseline.write_text(json.dumps({
            "comparison_summary": [{
                "status": "PASS",
                "task": "loso:BNCI2014_001",
                "method": "dualpc:0.1:0.05",
                "baseline": "erm:0",
            }],
            "selector_summary": [{
                "status": "PASS",
                "task": "loso:BNCI2014_001",
                "final_record_frac": 1.0,
                "final_probe_valid_frac": 1.0,
            }],
        }))
        missing_base = evaluate_decision(str(readiness), str(paper_missing_baseline),
                                         min_comparison_tasks=1, min_selector_tasks=1)
        _assert(missing_base["decision"] == "PENDING",
                "missing required baseline family should keep decision pending")

        paper_warn = root / "paper_warn.json"
        paper_warn.write_text(json.dumps({
            "comparison_summary": [{
                "status": "PASS",
                "task": "loso:BNCI2014_001",
                "method": "dualpc:0.1:0.05",
                "baseline": "erm:0",
            }, {
                "status": "WARN",
                "task": "loso:BNCI2014_001",
                "method": "dualpc:0.1:0.05",
                "baseline": "lpc_prior:0.1",
            }],
            "selector_summary": [{
                "status": "PASS",
                "task": "loso:BNCI2014_001",
                "final_record_frac": 1.0,
                "final_probe_valid_frac": 1.0,
            }],
        }))
        review = evaluate_decision(str(readiness), str(paper_warn),
                                   min_comparison_tasks=1, min_selector_tasks=1)
        _assert(review["decision"] == "NEEDS_REVIEW", "warning comparisons should require review")


def check_protocol_commands():
    smoke = selector_cmd("/tmp/dualpc_proto", "BNCI2014_001", 0, "smoke", "cpu")
    paper = selector_cmd("/tmp/dualpc_proto", "BNCI2014_001", 0, "paper", "cuda")
    ready = readiness_cmd("/tmp/dualpc_proto")
    paper_summary = paper_summary_cmd("/tmp/dualpc_proto")
    decision = decision_cmd("/tmp/dualpc_proto", 2, 1)
    _assert("--final_probe_epochs 1" in smoke, "smoke selector command missing final probes")
    _assert("--final_probe_epochs 80" in paper, "paper selector command missing final probes")
    _assert("*.json" in ready, "readiness command should keep a glob for runtime expansion")
    _assert("dualpc_paper_summary.py" in paper_summary and "*.json" in paper_summary,
            "protocol should generate a paper aggregate summary command")
    _assert("dualpc_decision_gate.py" in decision and "--min-comparison-tasks 2" in decision
            and "--min-selector-tasks 1" in decision and "--required-baselines erm lpc_prior" in decision
            and "--required-synthetic-groups null_prior concept all_three" in decision,
            "protocol should generate a headline decision gate command")


def check_slurm_plan_generation():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        args = SimpleNamespace(
            profile="smoke",
            device="cpu",
            seeds=[0],
            out_dir=str(root / "out"),
            loso_datasets=["BNCI2014_001"],
            scps_conditions=["PD"],
            partition="V100",
            gpus=1,
            cpus=2,
            mem="8G",
            time="01:00:00",
            max_parallel=1,
        )
        commands = protocol_build(args)
        tasks = [(comment, command) for comment, command in commands if not _is_post(comment)]
        post = [(comment, command) for comment, command in commands if _is_post(comment)]
        task_file = root / "tasks.tsv"
        array_script = root / "array.slurm"
        post_script = root / "post.slurm"
        _write_task_file(task_file, tasks)
        _write_array_script(array_script, task_file, len(tasks), args)
        _write_post_script(post_script, post, args)

        task_lines = task_file.read_text().strip().splitlines()
        _assert(len(task_lines) == 5, "smoke slurm plan should have regression/synthetic/loso/select/scps tasks")
        _assert("dualpc_regression_checks.py" in task_lines[0],
                "first slurm task should be the fast regression check")
        arr = array_script.read_text()
        _assert("#SBATCH --array=0-4%1" in arr, "array script range/concurrency wrong")
        _assert("eval \"$task_cmd\"" in arr, "array script should execute selected task command")
        post_text = post_script.read_text()
        _assert("dualpc_readiness.py" in post_text and "dualpc_paper_summary.py" in post_text
                and "dualpc_decision_gate.py" in post_text,
                "post script should run readiness, paper summary, and decision gate")


def check_paper_status_preflight():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        args = SimpleNamespace(
            profile="smoke",
            device="cpu",
            seeds=[0],
            out_dir=str(root / "out"),
            loso_datasets=["BNCI2014_001"],
            scps_conditions=["PD"],
            partition="V100",
            gpus=1,
            cpus=2,
            mem="8G",
            time="01:00:00",
            max_parallel=1,
        )
        commands = protocol_build(args)
        tasks = [(comment, command) for comment, command in commands if not _is_post(comment)]
        post = [(comment, command) for comment, command in commands if _is_post(comment)]
        task_file = root / "tasks.tsv"
        array_script = root / "array.slurm"
        post_script = root / "post.slurm"
        _write_task_file(task_file, tasks)
        _write_array_script(array_script, task_file, len(tasks), args)
        _write_post_script(post_script, post, args)

        status = collect_status(str(task_file), str(array_script), str(post_script))
        _assert(all(r["status"] == "PASS" for r in status["checks"]),
                "paper status preflight checks should pass for generated files")
        check_names = {r["check"] for r in status["checks"]}
        for name in ("policy_synthetic_groups", "policy_comparison_task_count",
                     "policy_runner_baselines", "policy_selector_task_count"):
            _assert(name in check_names, f"paper status missing {name} policy check")
        task_statuses = [r["status"] for r in status["tasks"]]
        _assert(task_statuses.count("READY") == 1, "regression task should be READY with no JSON output")
        _assert(task_statuses.count("PENDING") == 4, "smoke output tasks should be PENDING before run")
        post_statuses = [r["status"] for r in status["post"]]
        _assert(post_statuses == ["PENDING", "PENDING", "PENDING"],
                "post readiness/paper-summary/decision outputs should be pending before run")


def check_train_model_dualpc_path():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(32, 2, 4)).astype("float32")
    y = np.array([0, 1] * 16, dtype=np.int64)
    d = np.array([0] * 16 + [1] * 16, dtype=np.int64)
    # Add a weak domain/class signal so Step-A has something finite to fit.
    X[d == 1, 0, :] += 0.2
    X[y == 1, 1, :] += 0.2

    _, _, dualpc_diag = train_model(
        TinyBackbone(), X, y, d, n_cls=2, method="dualpc", lam=0.1, gamma=0.1,
        epochs=2, warmup=1, n_inner=1, bs=8, sampler="classbal", device="cpu", seed=3,
    )
    _assert(dualpc_diag["sampler"] == "raw", "dualpc must force raw sampler for explicit GLS semantics")
    _assert_close(dualpc_diag["dec_margin"], 0.0, msg="dualpc train path should default tau to 0")
    _assert("inloop_dec_loss" in dualpc_diag, "dualpc train path must log JS-side inloop_dec_loss")
    _assert(np.isfinite(dualpc_diag["inloop_dec_loss"]), "dualpc inloop_dec_loss must be finite")
    _assert(np.isfinite(dualpc_diag["inloop_reg"]), "dualpc inloop_reg must be finite")
    _assert(0.0 <= dualpc_diag["stepA_dom_acc"] <= 1.0, "dualpc Step-A domain accuracy out of range")

    _, _, lpc_diag = train_model(
        TinyBackbone(), X, y, d, n_cls=2, method="lpc_prior", lam=0.1, gamma=0.0,
        epochs=2, warmup=1, n_inner=1, bs=8, sampler="classbal", device="cpu", seed=4,
    )
    _assert(lpc_diag["sampler"] == "classbal", "non-DualPC sampler should not be forced to raw")
    _assert("inloop_dec_loss" not in lpc_diag, "lpc_prior should not log DualPC decoder loss")


def main():
    checks = [
        check_dec_margin_defaults,
        check_label_shift_weights,
        check_weighted_losses_all_ones_equivalence,
        check_decoder_validity_summaries,
        check_readiness_parsing,
        check_paper_summary_parsing,
        check_decision_gate,
        check_protocol_commands,
        check_slurm_plan_generation,
        check_paper_status_preflight,
        check_train_model_dualpc_path,
    ]
    for fn in checks:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL PASS ({len(checks)} checks)")


if __name__ == "__main__":
    main()
