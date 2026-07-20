"""C16-C battery discriminative validity: the battery certifies a synthetic transferring regime and falsifies
decoupled / anti-transfer regimes (so it is not merely a negative-result wrapper). Deterministic."""
from __future__ import annotations

from oaci.artifacts.canonical_json import canonical_json_bytes
from oaci.mechanism.discriminative_controls import _evidence, run_controls
from oaci.falsification.battery import run_battery


def test_discriminative_validity_true():
    r = run_controls()
    assert r["discriminative_validity"] is True
    assert r["positive_pass"] == r["positive_total"] and r["negative_pass"] == r["negative_total"]


def test_positive_transfer_certified():
    c8, c10, c12 = _evidence(0.9, leakage_detectable=True)
    v = run_battery(c8, c10, c12)["verdict"]
    assert v["control_hypothesis_status"] == "control_hypothesis_supported" and not v["falsification_reasons"]


def test_decoupled_falsified():
    c8, c10, c12 = _evidence(0.0, leakage_detectable=False)
    v = run_battery(c8, c10, c12)["verdict"]
    assert v["control_hypothesis_status"] == "falsified"
    assert "falsified_by_no_endpoint_transfer" in v["falsification_reasons"]


def test_anti_transfer_falsified_by_antitransfer():
    c8, c10, c12 = _evidence(-0.9, leakage_detectable=False)
    v = run_battery(c8, c10, c12)["verdict"]
    assert v["control_hypothesis_status"] == "falsified"
    assert "falsified_by_source_target_antitransfer" in v["falsification_reasons"]


def test_confusion_matrix_is_diagonal_and_serializable():
    r = run_controls()
    for true_label, row in r["confusion_matrix"].items():
        assert list(row.keys()) == [true_label] and sum(row.values()) >= 1     # only correct predictions
    assert canonical_json_bytes(r)


def test_regime_is_deterministic():
    a = run_controls(seed=0)["regimes"]; b = run_controls(seed=0)["regimes"]
    assert {n: v["battery_status"] for n, v in a.items()} == {n: v["battery_status"] for n, v in b.items()}


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} discriminative-control tests")


if __name__ == "__main__":
    _run_all()
