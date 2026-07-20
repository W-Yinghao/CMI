"""C7 part B: manifest K1/K2 wiring (strict + executable) + payload roundtrip + isolation. Standalone."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np

import oaci.protocol
from oaci.decision.payloads import k1_null_arrays, k1_payload, k2_payload
from oaci.decision.plans import k1_spec_from_manifest, k2_spec_from_manifest
from oaci.decision.report import render_decision_summary
from oaci.protocol.manifest_v2 import load_v2

_PROTO = os.path.join(os.path.dirname(oaci.protocol.__file__), "confirmatory_v2.yaml")


def _manifest():
    return load_v2(_PROTO)


def test_k1_k2_specs_from_confirmatory_manifest():
    m = _manifest()
    k1 = k1_spec_from_manifest(m)
    assert k1.statistic == "grouped_max_probe_extractable_LQ_ov_OACI_minus_ERM"
    assert k1.split_role == "source_audit" and k1.permutation_scheme == "paired_swap_within_y_recording_group"
    assert k1.n_permutations == 2000 and k1.alpha == 0.05 and k1.seed == 707
    k2 = k2_spec_from_manifest(m)
    assert tuple(k2.endpoints) == ("worst_domain_bacc", "worst_domain_nll")
    assert k2.min_seeds == 3 and k2.level_policy == "both_levels"
    assert k2.margins == {"worst_domain_bacc": 0.0, "worst_domain_nll": 0.0}


def test_k1_spec_rejects_unimplemented_scheme_or_split():
    for field, bad in (("permutation_scheme", "row_shuffle"), ("split_role", "target_audit"),
                       ("statistic", "something_else")):
        m = _manifest(); setattr(m.k1, field, bad)
        try:
            k1_spec_from_manifest(m)
        except ValueError:
            continue
        raise AssertionError(f"k1 spec must reject {field}={bad!r}")


def test_manifest_validate_ranges_enforces_k1_k2():
    m = _manifest(); m.k1.n_permutations = 0
    try:
        m.validate_ranges()
    except ValueError:
        pass
    else:
        raise AssertionError("n_permutations=0 must fail validate_ranges")
    m2 = _manifest(); m2.k2.level_policy = "any_level"
    try:
        m2.validate_ranges()
    except ValueError:
        return
    raise AssertionError("level_policy=any_level must fail validate_ranges")


def _perm_result():
    null = np.array([0.1, -0.2, 0.05, -0.4, 0.3], dtype=np.float64)
    return {"statistic": "grouped_max_probe_extractable_LQ_ov_OACI_minus_ERM", "split_role": "source_audit",
            "observed_delta": -0.42, "null": null, "null_quantiles": {"0.05": -0.4, "0.5": 0.05},
            "p_lower": 0.16, "p_upper": 0.84, "p_two_sided": 0.5, "alpha": 0.05, "n_permutations": 5,
            "permutation_plan_hash": "abc123def456", "audit_support_hash": "sup", "audit_population_hash": "pop",
            "probe_config_hash": "cfg"}


def test_decision_payload_roundtrip_preserves_hashes():
    from oaci.artifacts.canonical_json import canonical_json_bytes
    pr = _perm_result()
    dec = {"k1_status": "stop_no_detectable_heldout_leakage_reduction", "continue_to_k2": False}
    body = k1_payload(pr, dec)
    rt = json.loads(canonical_json_bytes(body).decode())
    for k in ("permutation_plan_hash", "audit_support_hash", "audit_population_hash", "probe_config_hash",
              "p_lower", "observed_delta", "k1_status"):
        assert rt[k] == body[k]
    assert "null" not in rt and rt["null_quantiles"]["0.05"] == -0.4      # heavy null lives in the npz
    # npz roundtrip
    d = tempfile.mkdtemp(); p = os.path.join(d, "k1_null.npz")
    np.savez(p, **k1_null_arrays(pr))
    z = np.load(p)
    assert np.array_equal(z["null"], pr["null"]) and z["observed_delta"][0] == pr["observed_delta"]


def test_render_decision_summary():
    per_level = [{"level": 0, "k1": {"k1_status": "stop_no_detectable_heldout_leakage_reduction",
                                     "observed_delta": 0.01, "p_lower": 0.33, "alpha": 0.05,
                                     "n_permutations": 2000, "permutation_plan_hash": "deadbeefcafe00"},
                  "k2": {"k2_status": "abstain_insufficient_seeds", "reason": "1 seed(s) < min_seeds 3"}}]
    md = render_decision_summary(per_level)
    assert "level 0" in md and "K1" in md and "K2" in md and "abstain_insufficient_seeds" in md


def test_no_oaci_decision_runtime_import_from_cmi_or_h2cmi():
    import oaci.decision
    root = os.path.dirname(oaci.decision.__file__)
    files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".py")]
    files.append(os.path.join(os.path.dirname(oaci.protocol.__file__), "..", "leakage", "permutation.py"))
    for fp in files:
        src = open(os.path.abspath(fp)).read()
        for bad in ("import cmi", "from cmi", "import h2cmi", "from h2cmi"):
            assert bad not in src, f"{fp} must not {bad}"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} decision-plans tests")


if __name__ == "__main__":
    _run_all()
