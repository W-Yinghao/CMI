"""C7 part C: the decision subtree is written through the writer index (verify stays whole), old artifacts
verify legacy-complete, the require-check enforces presence, and the runner integration produces valid
payloads. Standalone + pytest-compatible."""
from __future__ import annotations

import os
import tempfile

import numpy as np

import oaci.protocol
from oaci.artifacts.decision_codec import read_level_decisions, verify_decisions
from oaci.artifacts.verify import verify_artifact_tree
from oaci.artifacts.writer import GitEvidence, git_evidence_hash, write_artifact_tree_atomic
from oaci.decision.payloads import k1_null_arrays, k1_payload, k2_payload
from oaci.decision.plans import K1Spec, K2Spec
from oaci.runner.decision import K1_SKIPPED, compute_level_decision
from oaci.runner.fake import DEFAULT_METHOD_ORDER
from oaci.runner.fake_artifact import run_fake_two_level

_MAN = os.path.join(os.path.dirname(oaci.protocol.__file__), "fake_runner_v1.yaml")


def _ge():
    c, t = "c" * 40, "t" * 40
    return GitEvidence(c, t, ("oaci",), (), True, git_evidence_hash(c, t, ("oaci",), (), True))


def _fake():
    return run_fake_two_level(_MAN, tempfile.mkdtemp(prefix="oaci-dec-"), model_seed=0,
                              method_order=DEFAULT_METHOD_ORDER, repo_root=os.path.join(tempfile.gettempdir(),
                              "oaci-fake-repo-marker"), git_evidence=_ge())


def _synth_perm(delta=-0.42):
    null = np.array([0.1, -0.2, 0.05, -0.4, 0.3], dtype=np.float64)
    return {"statistic": "grouped_max_probe_extractable_LQ_ov_OACI_minus_ERM", "split_role": "source_audit",
            "observed_delta": delta, "null": null, "null_quantiles": {"0.5": 0.05}, "p_lower": 0.33,
            "p_upper": 0.67, "p_two_sided": 0.66, "alpha": 0.05, "n_permutations": 5,
            "permutation_plan_hash": "plan_" + "a" * 8, "audit_support_hash": "sup", "audit_population_hash": "pop",
            "probe_config_hash": "cfg"}


def _synth_decisions(levels):
    dec = {}
    for L in levels:
        pr = _synth_perm()
        dec[int(L)] = {
            "k1_body": k1_payload(pr, {"k1_status": "stop_no_detectable_heldout_leakage_reduction",
                                       "continue_to_k2": False}),
            "k1_null_arrays": k1_null_arrays(pr),
            "k2_body": k2_payload({"k2_status": "abstain_insufficient_seeds", "continue": False,
                                   "reason": "1 seed(s) < min_seeds 3"})}
    return dec


def _levels(res):
    return [int(l) for l, _ in res.fold_result.level_items]


def test_decision_payload_roundtrip_preserves_hashes():
    res = _fake()
    dec = _synth_decisions(_levels(res))
    out = write_artifact_tree_atomic(res.fold_result, res.context, tempfile.mkdtemp(prefix="oaci-decw-"),
                                     level_decisions=dec)
    rep = verify_artifact_tree(out.artifact_dir, deep=True)                 # decisions are INDEXED -> tree whole
    assert rep.ok, rep.errors[:3]
    for L in _levels(res):
        rd = read_level_decisions(out.artifact_dir, L)
        assert rd["k1"]["permutation_plan_hash"] == dec[L]["k1_body"]["permutation_plan_hash"]
        assert rd["k1"]["k1_status"] == "stop_no_detectable_heldout_leakage_reduction"
        assert rd["k2"]["k2_status"] == "abstain_insufficient_seeds"
        assert np.array_equal(rd["k1_null"]["null"], dec[L]["k1_null_arrays"]["null"])


def test_artifact_verifier_accepts_legacy_no_decision_artifact():
    res = _fake()                                                           # written WITHOUT decisions
    assert verify_artifact_tree(res.write_result.artifact_dir, deep=True).ok
    rep = verify_decisions(res.write_result.artifact_dir, require=False)    # legacy tolerated
    assert rep["with_decisions"] == [] and rep["levels"] == _levels(res)


def test_artifact_verifier_requires_decision_when_manifest_requests_it():
    res = _fake()
    try:
        verify_decisions(res.write_result.artifact_dir, require=True)       # legacy tree, decisions required
    except ValueError:
        pass
    else:
        raise AssertionError("require=True must reject a decision-less tree")
    out = write_artifact_tree_atomic(res.fold_result, res.context, tempfile.mkdtemp(prefix="oaci-decw2-"),
                                     level_decisions=_synth_decisions(_levels(res)))
    ok = verify_decisions(out.artifact_dir, require=True)                   # now present + well-formed
    assert ok["with_decisions"] == _levels(res)


def test_compute_level_decision_produces_valid_payloads():
    from oaci.tests.test_decision_k1 import FAST, _paired, _plan
    fe, fo, sg = _paired(erm_leaky=True, oaci_leaky=False)                  # OACI clean -> K1 detects
    k1_spec = K1Spec("grouped_max_probe_extractable_LQ_ov_OACI_minus_ERM", "source_audit",
                     "paired_swap_within_y_recording_group", n_permutations=49, alpha=0.05,
                     decision_rule="stop_if_within_null_band", seed=707)
    k2_spec = K2Spec(("worst_domain_bacc", "worst_domain_nll"), min_seeds=3, level_policy="both_levels",
                     margins={"worst_domain_bacc": 0.0, "worst_domain_nll": 0.0},
                     decision_rule="stop_if_no_reproducible_gain")
    out = compute_level_decision(0, feat_by_method={"ERM": fe, "OACI": fo}, audit_support_graph=sg,
                                 audit_fold_plan=_plan(fe, sg), cfg=FAST, k1_spec=k1_spec, k2_spec=k2_spec,
                                 k2_units=[])
    assert out["level"] == 0
    assert out["k1_body"]["k1_status"] == "leakage_reduction_detected" and out["k1_body"]["p_lower"] < 0.05
    assert out["k2_body"]["k2_status"] == "abstain_insufficient_seeds"     # zero seeds -> abstain
    assert out["k1_null_arrays"]["null"].shape[0] == 49


def test_compute_level_decision_skips_k1_when_oaci_absent():
    from oaci.tests.test_decision_k1 import FAST, _paired, _plan
    fe, _fo, sg = _paired()
    k1_spec = K1Spec("grouped_max_probe_extractable_LQ_ov_OACI_minus_ERM", "source_audit",
                     "paired_swap_within_y_recording_group", 10, 0.05, "stop_if_within_null_band", 0)
    k2_spec = K2Spec(("worst_domain_bacc",), 3, "both_levels", {"worst_domain_bacc": 0.0, "worst_domain_nll": 0.0},
                     "stop_if_no_reproducible_gain")
    out = compute_level_decision(1, feat_by_method={"ERM": fe}, audit_support_graph=sg,   # no OACI
                                 audit_fold_plan=_plan(fe, sg), cfg=FAST, k1_spec=k1_spec, k2_spec=k2_spec,
                                 k2_units=[])
    assert out["k1_body"]["k1_status"] == K1_SKIPPED and out["k1_null_arrays"]["null"].shape[0] == 0


def test_no_oaci_runtime_import_from_cmi_or_h2cmi():
    import oaci.decision
    roots = [os.path.dirname(oaci.decision.__file__)]
    extra = ["runner/decision.py", "artifacts/decision_codec.py", "leakage/permutation.py"]
    pkg = os.path.dirname(os.path.dirname(oaci.decision.__file__))
    files = [os.path.join(r, f) for r in roots for f in os.listdir(r) if f.endswith(".py")]
    files += [os.path.join(pkg, e) for e in extra]
    for fp in files:
        src = open(fp).read()
        for bad in ("import cmi", "from cmi", "import h2cmi", "from h2cmi"):
            assert bad not in src, f"{fp} must not `{bad}`"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} decision-artifacts tests")


if __name__ == "__main__":
    _run_all()
