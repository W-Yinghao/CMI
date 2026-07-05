"""C14c method-closure table + report serialization + no cross-package import."""
from __future__ import annotations

from oaci.artifacts.canonical_json import canonical_json_bytes
from oaci.falsification.payloads import build_method_closure_table, forbids_oaci_v2_selector
from oaci.falsification.report import render_md
from oaci.tests.test_falsification_battery import c8, c10, c12, _ANTI
from oaci.falsification.battery import run_battery


def _closed_gate_map():
    return {"G3_endpoint_transfer": {"status": "stop_no_reproducible_gain"},
            "G4_oracle_rescue": {"status": "oracle_fails_to_rescue"},
            "G5_source_target_transfer": {"status": "source_target_antitransfer_detected"}}


def test_method_closure_table_contains_oaci_and_src():
    tbl = build_method_closure_table(_closed_gate_map())
    hyps = [r["method_hypothesis"] for r in tbl]
    assert any(h.startswith("OACI") for h in hyps) and any(h.startswith("SRC") for h in hyps)
    assert any("support-aware extractable leakage" in h for h in hyps)   # retained-as-measurement entry
    oaci = next(r for r in tbl if r["method_hypothesis"].startswith("OACI"))
    src = next(r for r in tbl if r["method_hypothesis"].startswith("SRC"))
    assert oaci["status"] == "closed_as_control_objective" and src["status"] == "closed_as_control_objective"


def test_method_closure_prevents_oaci_v2_selector_recommendation():
    tbl = build_method_closure_table(_closed_gate_map())
    assert forbids_oaci_v2_selector(tbl)
    oaci = next(r for r in tbl if r["method_hypothesis"].startswith("OACI"))
    assert "NO OACI-v2" in oaci["next_allowed_action"]


def test_c14_report_json_is_canonical_serializable():
    bat = run_battery(c8(), c10(), c12(_ANTI))
    blob = canonical_json_bytes(bat)                                    # raises on any int mapping key
    assert blob and b'"verdict"' in blob and b'"method_closure_table"' in blob
    md = render_md(bat)
    assert "FALSIFIED" in md and "CONTROL-HYPOTHESIS" in md and "Do not over-claim" in md
    # the report must NOT contain the forbidden over-claims as assertions
    assert "All DG is impossible" in md and "~~All DG is impossible.~~" in md   # only struck-through


def test_battery_verdict_matches_expected_falsification_set():
    bat = run_battery(c8(), c10(), c12(_ANTI))
    assert bat["verdict"]["control_hypothesis_status"] == "falsified"
    assert set(bat["verdict"]["falsification_reasons"]) == {
        "falsified_by_no_endpoint_transfer", "falsified_by_oracle_failure", "falsified_by_source_target_antitransfer"}


def test_no_oaci_runtime_import_from_cmi_or_h2cmi():
    import sys
    import oaci.falsification.battery, oaci.falsification.report, oaci.falsification.gates  # noqa: F401
    import oaci.falsification.transfer, oaci.falsification.oracle, oaci.falsification.antitransfer  # noqa: F401
    leaked = [m for m in sys.modules if m == "cmi" or m.startswith("cmi.") or m == "h2cmi" or m.startswith("h2cmi.")]
    assert leaked == [], leaked


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} method-closure tests")


if __name__ == "__main__":
    _run_all()
