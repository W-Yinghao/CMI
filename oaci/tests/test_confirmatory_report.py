"""Confirmatory one-fold report (CPU): build the endpoint + k1/k2-style report from the order-invariant
fake closed loop (same FoldRunResult / artifact shape the real driver produces), with a stub fold
exposing the pilot split. No data, no GPU.

Standalone (`python -m oaci.tests.test_confirmatory_report`) and pytest-compatible.
"""
from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import oaci.protocol
from oaci.artifacts.writer import GitEvidence, git_evidence_hash
from oaci.confirmatory.onefold import OneFoldResult, OneFoldSeedRun
from oaci.confirmatory.report import build_onefold_report
from oaci.runner.fake_artifact import run_fake_two_level

_MAN = os.path.join(os.path.dirname(oaci.protocol.__file__), "fake_runner_v1.yaml")
_ORDER = ("ERM", "OACI", "global_lpc", "uniform")


def _ge():
    c, t = "c" * 40, "t" * 40
    return GitEvidence(c, t, ("oaci",), (), True, git_evidence_hash(c, t, ("oaci",), (), True))


def _stub_fold():
    pilot = SimpleNamespace(subjects=[1, 2, 3, 4, 5, 6, 7, 8, 9], target_subjects=[1],
                            source_audit_subjects=[2, 3], source_train_subjects=[4, 5, 6, 7, 8, 9],
                            deleted_cell_level1=SimpleNamespace(domain_id="BNCI2014_001|subject-004",
                                                               class_name="feet"))
    training = SimpleNamespace(stage1_epochs=200, stage2_epochs=200, stage2_steps_per_epoch=20)
    manifest = SimpleNamespace(pilot=pilot, training=training)
    return SimpleNamespace(manifest=manifest, data_evidence_hash="DATAEV", resolved_preprocess_hash="PPH",
                           split_manifest_hash="SPLIT")


def _result():
    runs = []
    for s in (0, 1):
        art = run_fake_two_level(_MAN, tempfile.mkdtemp(), model_seed=s, method_order=_ORDER,
                                 repo_root="/x", git_evidence=_ge())
        runs.append(OneFoldSeedRun(model_seed=s, artifact=art))
    return OneFoldResult(protocol_path="P", manifest_path="M", manifest_hash="MH", dataset="BNCI2014_001",
                         target_subject=1, model_seeds=(0, 1), fold=_stub_fold(), seed_runs=tuple(runs))


def test_report_echoes_split_and_budget_and_notice():
    r = build_onefold_report(_result())
    assert "pipeline validation" in r["notice"] and "NOT confirmatory efficacy" in r["notice"]
    assert r["target_subjects"] == [1] and r["source_audit_subjects"] == [2, 3]
    assert r["source_train_subjects"] == [4, 5, 6, 7, 8, 9]
    assert r["training_budget"]["stage1_epochs"] == 200
    assert r["deleted_cell_level1"]["class_name"] == "feet"


def test_report_has_per_seed_both_hashes_and_verification():
    r = build_onefold_report(_result())
    assert [sb["model_seed"] for sb in r["seeds"]] == [0, 1]
    for sb in r["seeds"]:
        assert sb["artifact_scientific_hash"] and sb["artifact_pure_science_hash"]
        assert sb["deep_verification_ok"] is True
    assert r["all_seeds_deep_verified"] is True and r["all_target_fit_ids_empty"] is True


def test_report_k1_k2_descriptive_structure():
    r = build_onefold_report(_result())
    levels = sorted({k["level"] for k in r["k1_descriptive"]})
    assert levels == [0, 1]                                            # two-level
    for k in r["k1_descriptive"]:
        assert k["statistic"] == "OACI_minus_ERM_audit_leakage_ucl" and "per_seed" in k
    methods = {k["method"] for k in r["k2_descriptive"]}
    assert methods == {"ERM", "OACI", "global_lpc", "uniform"}
    for k in r["k2_descriptive"]:
        assert len(k["worst_domain_bacc_per_seed"]) == 2              # one value per seed


def test_report_per_method_endpoints_present():
    r = build_onefold_report(_result())
    lvl0 = next(l for l in r["seeds"][0]["levels"] if l["level"] == 0)
    erm = next(m for m in lvl0["methods"] if m["method"] == "ERM")
    for key in ("selected_risk", "selected_risk_minus_tau", "selected_epoch", "target_pooled_bacc",
                "target_worst_bacc", "target_pooled_nll", "source_audit_bacc"):
        assert key in erm
    assert "R_ERM_hat" in lvl0 and "tau" in lvl0 and "eligibility_counts" in lvl0


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} confirmatory-report tests")


if __name__ == "__main__":
    _run_all()
