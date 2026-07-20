"""C6a CPU tests: the BNCI2014_001 LOSO plan/driver, the 18-job submitter, and the aggregation logic.
No data, no training, no GPU.

Standalone (`python -m oaci.tests.test_bnci001_loso`) and pytest-compatible.
"""
from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import oaci.protocol
import oaci.confirmatory.staged_demo as staged_demo
from oaci.confirmatory.aggregate import (_protocol_family, aggregate_loso, collect_fold_artifacts,
                                          render_report_md)
from oaci.confirmatory.bnci001_loso import (expected_level_support, materialize_all_loso, validate_loso_plan)
from oaci.confirmatory.loso_plan import loso_fold_spec, loso_plan
from oaci.confirmatory.materialize import materialize_pilot_manifest
from oaci.confirmatory.schema import load_confirmatory
from oaci.confirmatory.submit import build_job_plan, print_plan, validate_launch

_PROTO = os.path.join(os.path.dirname(oaci.protocol.__file__), "confirmatory_v2.yaml")


# ===================== plan / driver =====================
def test_bnci001_loso_plan_has_9_unique_targets():
    plan = loso_plan()
    assert len(plan) == 9 and sorted(f["target"] for f in plan) == list(range(1, 10))
    assert validate_loso_plan()["ok"]


def test_bnci001_loso_cyclic_split_matches_target001_c5():
    f1 = loso_fold_spec(1)
    assert f1["source_audit_subjects"] == [2, 3] and f1["source_train_subjects"] == [4, 5, 6, 7, 8, 9]
    assert f1["deleted_subject"] == 4 and f1["deleted_cell"] == {"domain_id": "BNCI2014_001|subject-004", "class_name": "feet"}
    assert loso_fold_spec(2)["source_audit_subjects"] == [3, 4] and loso_fold_spec(2)["source_train_subjects"] == [5, 6, 7, 8, 9, 1]
    assert loso_fold_spec(9)["source_audit_subjects"] == [1, 2] and loso_fold_spec(9)["source_train_subjects"] == [3, 4, 5, 6, 7, 8]


def test_bnci001_loso_roles_are_disjoint_per_fold():
    for f in loso_plan():
        flat = [f["target"]] + f["source_audit_subjects"] + f["source_train_subjects"]
        assert len(set(flat)) == 9 == len(flat) and set(flat) == set(range(1, 10))
        assert len(f["source_audit_subjects"]) == 2 and len(f["source_train_subjects"]) == 6


def test_bnci001_loso_deleted_cells_are_deterministic():
    a, b = loso_plan(), loso_plan()
    assert [f["deleted_cell"] for f in a] == [f["deleted_cell"] for f in b]
    for f in loso_plan():                                       # first source-train subject x feet, cyclic
        assert f["deleted_subject"] == f["source_train_subjects"][0] and f["deleted_cell"]["class_name"] == "feet"


def test_bnci001_loso_level0_support_tables_are_exact():
    for f in loso_plan():
        sup = expected_level_support(f)
        assert len(sup["level0"]) == 6 and all(len(r) == 4 for r in sup["level0"])
        assert all(c == 144 for r in sup["level0"] for c in r)   # 6 x 4 matrix of 144s
        assert sup["p_ref"] == [0.25, 0.25, 0.25, 0.25]


def test_bnci001_loso_level1_deleted_cells_are_zero():
    for f in loso_plan():
        sup = expected_level_support(f)
        z = [(i, j) for i, r in enumerate(sup["level1"]) for j, c in enumerate(r) if c == 0]
        assert z == [(sup["deleted_row"], sup["deleted_col"])]   # exactly the deleted (subject, feet) cell
        assert sup["level1"][sup["deleted_row"]][sup["deleted_col"]] == 0
        assert sup["classes"][sup["deleted_col"]] == "feet"


def test_bnci001_loso_all_methods_active():
    d = tempfile.mkdtemp()
    out = materialize_all_loso(_PROTO, d, model_seed=0, bootstrap_mode="full")
    assert len(out) == 9
    for spec, mp, m in out:
        assert sorted(m.methods.names) == sorted(["ERM", "OACI", "global_lpc", "uniform"])
        assert m.training.stage1_epochs == 200 and m.probe.audit_bootstrap == 2000   # full budget


def test_bnci001_loso_dry_run_does_not_train():
    d = tempfile.mkdtemp()
    out = materialize_all_loso(_PROTO, d, model_seed=0, bootstrap_mode="full")   # writes manifests, no training
    assert validate_loso_plan()["ok"]
    # no checkpoint artifacts produced by the dry run
    assert not any(f.endswith(".pt") for _root, _dirs, files in os.walk(d) for f in files)


# ===================== staged executor materialization (regression: cyclic split, NOT sorted) =====================
# The C6b GPU sweep materializes via staged_demo._materialize, NOT via the dry-run's materialize_all_loso.
# These lock the executor's own materialization to the validated cyclic plan -- the gap that let the
# sorted-vs-cyclic split bug through (the dry-run validated the cyclic plan; the sweep ran the sorted one).
def _materialize_via_staged(target, out_path, *, dataset="BNCI2014_001", bootstrap_mode="full"):
    args = SimpleNamespace(protocol=_PROTO, dataset=dataset, target_subject=target, manifest_out=out_path,
                           model_seed=0, bootstrap_mode=bootstrap_mode)
    return staged_demo._materialize(args)


def test_staged_executor_materializes_cyclic_split_not_sorted_for_target2():
    import yaml
    p, _m = _materialize_via_staged(2, os.path.join(tempfile.mkdtemp(), "t2.yaml"))
    pilot = yaml.safe_load(open(p))["pilot"]
    assert pilot["source_audit_subjects"] == [3, 4]              # cyclic -- the sorted split would give [1, 3]
    assert pilot["source_train_subjects"] == [5, 6, 7, 8, 9, 1]  # sorted would give [4, 5, 6, 7, 8, 9]
    assert pilot["deleted_cell_level1"]["domain_id"].endswith("subject-005")   # sorted would delete subject-004
    assert pilot["deleted_cell_level1"]["class_name"] == "feet"


def test_staged_executor_split_matches_loso_plan_for_all_targets():
    import yaml
    d = tempfile.mkdtemp()
    by_target = {f["target"]: f for f in loso_plan()}
    for t in range(1, 10):
        p, _m = _materialize_via_staged(t, os.path.join(d, f"t{t}.yaml"))
        pilot = yaml.safe_load(open(p))["pilot"]
        spec = by_target[t]
        assert pilot["target_subjects"] == [t]
        assert pilot["source_audit_subjects"] == spec["source_audit_subjects"]
        assert pilot["source_train_subjects"] == spec["source_train_subjects"]
        assert pilot["deleted_cell_level1"] == spec["deleted_cell"]


def test_staged_target001_manifest_byte_identical_to_default_split():
    # target-001: the cyclic and the default sorted split coincide, so the manifest must be byte-identical
    # to the pre-C6 default-split path -- proving the C5/C4b target-001 staged runs stay valid.
    d = tempfile.mkdtemp()
    p_cyc, _ = _materialize_via_staged(1, os.path.join(d, "t1_cyclic.yaml"))
    proto = load_confirmatory(_PROTO)
    p_def, _ = materialize_pilot_manifest(proto, "BNCI2014_001", target_subject=1,
                                          out_path=os.path.join(d, "t1_default.yaml"), model_seeds=[0])
    assert open(p_cyc, "rb").read() == open(p_def, "rb").read()


def test_staged_executor_rejects_non_bnci_loso_dataset():
    try:
        _materialize_via_staged(2, os.path.join(tempfile.mkdtemp(), "x.yaml"), dataset="OTHER")
    except ValueError:
        return
    raise AssertionError("the LOSO cyclic split is BNCI2014_001-only; another dataset must be rejected")


# ===================== submitter =====================
_REPO = "/home/infres/yinwang/CMI_AAAI_oaci"
_OUT = "/projects/EEG-foundation-model/yinghao/oaci-loso-test"


def test_phase_a_phase_b_job_plan_has_18_jobs():
    jobs = build_job_plan(_OUT, _REPO)
    assert len(jobs) == 18
    assert sum(1 for j in jobs if j["kind"] == "phase_a") == 9
    assert sum(1 for j in jobs if j["kind"] == "phase_b") == 9


def test_phase_b_depends_on_matching_phase_a():
    jobs = build_job_plan(_OUT, _REPO)
    for j in jobs:
        if j["kind"] == "phase_b":
            assert j["depends_on"].startswith(f"{j['fold_id']}:phase_a")
            assert j["partition"] == "CPU" and j["gres"] is None     # no GPU in Phase B
        else:
            assert j["partition"] == "V100" and j["gres"] == "gpu:1"


def test_phase_a_jobs_are_parallel_and_do_not_self_chain_phase_b():
    jobs = build_job_plan(_OUT, _REPO)
    for j in jobs:
        if j["kind"] == "phase_a":
            assert j["depends_on"] is None                          # all Phase-A run in parallel
            assert j["env"]["OACI_CHAIN_PHASE_B"] == "0"            # submitter owns the Phase-B graph
            assert j["env"]["OACI_REPO"] == os.path.abspath(_REPO)


def test_phase_b_rolling_cap_dependency_graph():
    b = [j for j in build_job_plan(_OUT, _REPO, phase_b_cap=3) if j["kind"] == "phase_b"]
    ids = [j["fold_id"] for j in b]
    for i, j in enumerate(b):
        assert j["depends_on_phase_a"] == j["fold_id"]              # afterok on its own Phase A
        assert j["phase_b_cap"] == 3
        assert j["depends_on_prior_phase_b"] == (ids[i - 3] if i >= 3 else None)   # rolling afterany:B_{i-3}
    assert sum(1 for j in b if j["depends_on_prior_phase_b"] is None) == 3          # exactly cap start immediately


def test_phase_b_cap_is_configurable():
    b2 = [j for j in build_job_plan(_OUT, _REPO, phase_b_cap=2) if j["kind"] == "phase_b"]
    ids = [j["fold_id"] for j in b2]
    assert sum(1 for j in b2 if j["depends_on_prior_phase_b"] is None) == 2
    assert b2[2]["depends_on_prior_phase_b"] == ids[0] and b2[8]["depends_on_prior_phase_b"] == ids[6]


def test_submitter_dry_run_prints_all_paths_and_dependencies():
    import io
    jobs = build_job_plan(_OUT, _REPO)
    buf = io.StringIO()
    print_plan(jobs, loso_root=_OUT, file=buf)
    s = buf.getvalue()
    for t in range(1, 10):
        assert f"target-{t:03d}" in s
    assert "staging" in s and "artifact" in s and "depends" in s and "V100" in s and "CPU" in s


def test_submitter_rejects_repo_internal_artifact_root():
    try:
        validate_launch(os.path.join(_REPO, "oaci", "inside"), _REPO, "/projects/EEG-foundation-model/datalake/raw")
    except ValueError:
        return
    raise AssertionError("an in-repo LOSO root must be rejected")


def test_submitter_rejects_missing_datalake():
    try:
        validate_launch(_OUT, _REPO, "/no/such/datalake")
    except ValueError:
        return
    raise AssertionError("a missing datalake must be rejected")


# ===================== aggregation =====================
def _level(L, oaci_audit=0.80, erm_audit=0.75, oaci_bacc=0.45, erm_bacc=0.48):
    base = lambda n, au, ba: {"method": n, "audit_ucl": au, "selection_ucl": 1.3, "target_bacc": ba,
                              "target_nll": 1.2, "target_ece": 0.1, "selected_checkpoint": "c", "selected_risk": 0.85}
    return {"level": L, "R_ERM_hat": 0.85, "tau": 0.88,
            "methods": [base("ERM", erm_audit, erm_bacc), base("OACI", oaci_audit, oaci_bacc),
                        base("global_lpc", 0.8, 0.46), base("uniform", 0.8, 0.46)]}


_FAM = "oaci-confirmatory-v2-pilot-BNCI2014_001"


def _fold(target, *, deep=True, tfit=True, family=_FAM, prov="PROV"):
    # Real folds carry per-fold-DISTINCT context hashes (different manifest per target) but a SHARED
    # protocol_family + provenance_hash; the identity check keys on the latter, not context_hash.
    return {"target": target, "deep_verification_ok": deep, "target_fit_empty": tfit,
            "protocol_family": family, "provenance_hash": prov, "context_hash": f"ctx-{target}",
            "methods_present": ["ERM", "OACI", "global_lpc", "uniform"], "levels": [_level(0), _level(1)]}


def _nine():
    return [_fold(t) for t in range(1, 10)]


def test_protocol_family_strips_target_suffix():
    ids = [f"{_FAM}-target{t:03d}" for t in range(1, 10)]
    assert {_protocol_family(i) for i in ids} == {_FAM}          # all nine collapse to one family
    assert _protocol_family("oaci-confirmatory-v2-pilot-validredbootstrap-BNCI2014_001-target007") \
        == "oaci-confirmatory-v2-pilot-validredbootstrap-BNCI2014_001"


def test_aggregation_accepts_distinct_context_hashes_same_family():
    # The real-world case the old hard-coded "P" stub hid: nine folds, nine DIFFERENT context_hashes, one
    # protocol_family + one provenance -> ACCEPTED (not false-rejected).
    r = aggregate_loso(_nine(), protocol_family=_FAM, provenance_hash="PROV")
    assert r["n_folds"] == 9 and r["protocol_family"] == _FAM and r["provenance_hash"] == "PROV"
    assert len(set(r["per_fold_context_hashes"].values())) == 9   # per-fold hashes genuinely differ


def test_aggregation_rejects_mixed_family_or_provenance():
    bad_fam = _nine(); bad_fam[4] = _fold(5, family="other-protocol")
    bad_prov = _nine(); bad_prov[4] = _fold(5, prov="other-commit")
    for bad in (bad_fam, bad_prov):
        try:
            aggregate_loso(bad)
        except ValueError:
            continue
        raise AssertionError("a mixed protocol family / provenance must be rejected")


def test_aggregation_requires_all_9_targets():
    r = aggregate_loso(_nine())
    assert r["n_folds"] == 9 and r["targets"] == list(range(1, 10))
    try:
        aggregate_loso(_nine()[:8])
    except ValueError:
        return
    raise AssertionError("fewer than 9 targets must be rejected")


def test_aggregation_rejects_duplicate_or_missing_target():
    bad = _nine(); bad[8] = _fold(1)                            # target 1 twice, 9 missing
    try:
        aggregate_loso(bad)
    except ValueError:
        return
    raise AssertionError("a duplicate/missing target must be rejected")


def test_aggregation_rejects_failed_deep_verification():
    bad = _nine(); bad[3] = _fold(4, deep=False)
    try:
        aggregate_loso(bad)
    except ValueError:
        return
    raise AssertionError("a failed deep verification must be rejected")


def test_aggregation_rejects_target_fit_ids():
    bad = _nine(); bad[5] = _fold(6, tfit=False)
    try:
        aggregate_loso(bad)
    except ValueError:
        return
    raise AssertionError("a non-empty target_fit must be rejected")


def test_aggregation_reports_k1_k2_style_endpoints():
    r = aggregate_loso(_nine())
    assert len(r["k1_descriptive"]) == 2 and len(r["k2_descriptive"]) == 2
    k1 = r["k1_descriptive"][0]
    assert abs(k1["delta_leakage_ucl_mean"] - 0.05) < 1e-9      # OACI 0.80 - ERM 0.75
    assert len(k1["per_fold"]) == 9
    k2 = r["k2_descriptive"][0]
    assert abs(k2["delta_target_bacc"]["d_mean"] - (0.45 - 0.48)) < 1e-9
    assert k2["n_bacc_improved"] == 0                            # OACI bAcc below ERM here


def test_aggregation_is_order_invariant():
    import random
    a = aggregate_loso(_nine())
    shuffled = _nine(); random.Random(0).shuffle(shuffled)
    b = aggregate_loso(shuffled)
    assert a["targets"] == b["targets"] and a["k1_descriptive"] == b["k1_descriptive"]
    assert a["k2_descriptive"] == b["k2_descriptive"]


def test_aggregate_output_is_canonical_json_serializable():
    # The runner writes the aggregate via canonical_json (str keys only). CI never serialized it, so an
    # int-keyed dict (per_fold_context_hashes) slipped through and crashed the report writer. Lock it.
    import json
    from oaci.artifacts.canonical_json import canonical_json_bytes
    b = canonical_json_bytes(aggregate_loso(_nine()))
    round_trip = json.loads(b.decode())
    assert sorted(round_trip["per_fold_context_hashes"]) == [f"target-{t:03d}" for t in range(1, 10)]
    assert all(isinstance(k, str) for k in round_trip["per_fold_context_hashes"])


def test_render_report_md_contains_k1_k2_and_identity():
    md = render_report_md(aggregate_loso(_nine()))
    assert "# C6" in md and "## k1" in md and "## k2" in md
    assert _FAM in md and "Δ audit_ucl" in md
    assert "level 0" in md and "level 1" in md and "not the final multi-seed" in md


def test_collect_fold_artifacts_requires_all_nine():
    d = tempfile.mkdtemp()                                       # empty -> target-001 artifact missing
    try:
        collect_fold_artifacts(d)
    except ValueError:
        return
    raise AssertionError("a missing per-target artifact must be rejected")


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} bnci001-loso tests")


if __name__ == "__main__":
    _run_all()
