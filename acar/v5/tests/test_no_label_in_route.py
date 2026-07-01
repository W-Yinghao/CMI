"""Guard: routing is LABEL-FREE — no label/y parameter, no label token in the routing source, and injecting a label into the
batch cannot change the decision (metamorphic). Synthetic only."""
from __future__ import annotations
import inspect
from acar.v5 import protocol as P
from acar.v5 import deploy, scalarization
from acar.v5.tests._util import ok, batch


def test_route_signature_has_no_label():
    params = list(inspect.signature(deploy.route).parameters)
    assert params == ["candidate", "batch", "thresholds"], params
    forbidden = ("y", "label", "labels", "target", "y_te", "y_true")
    assert not any(p in forbidden for p in params)
    ok("route() signature = (candidate, batch, thresholds); no label/y/target parameter")


def test_routing_source_has_no_label_access():
    # inspect ACCESS patterns (not the bare word "label", which legitimately appears in the "label-free" docstrings)
    src = inspect.getsource(deploy.route) + inspect.getsource(scalarization)
    forbidden = ('["label"]', "['label']", ".label",
                 '["y"]', "['y']", '["y_te"]', "['y_te']", ".y_te", "y_true",
                 '["target"]', "['target']", ".target",
                 '["harmful"]', "['harmful']", ".harmful")
    for tok in forbidden:
        assert tok not in src, f"routing source must not access {tok!r}"
    ok("routing source (route + scalarization) performs no label/harm ACCESS")


def test_injected_label_does_not_change_decision():
    c = next(x for x in P.CANDIDATE_MANIFEST if x["id"] == "V5-P1-001")
    fit = [batch(f"b{i}", matched_coral={"d_margin": float(i) / 10, "flip_rate": 0.1, "JS": 0.1}) for i in range(11)]
    th = scalarization.fit_quantiles(c, fit)
    b = batch("x", matched_coral={"d_margin": 1.0, "flip_rate": 0.05, "JS": 0.05})
    d0 = deploy.route(c, b, th)
    # poison: attach a label to the batch and to every action's feature dict, both classes
    for y in (0, 1):
        b_poison = {"batch_id": b["batch_id"], "label": y, "y_te": y,
                    "features": {a: dict(b["features"][a], label=y, y_te=y) for a in P.ACTIONS}}
        assert deploy.route(c, b_poison, th) == d0, "decision changed under an injected label — LABEL LEAK"
    ok("metamorphic: injecting a label (both classes) leaves the routing decision byte-identical")


def main():
    print("ACAR v5 guard: no label in route")
    test_route_signature_has_no_label()
    test_routing_source_has_no_label_access()
    test_injected_label_does_not_change_decision()
    print("ALL V5 NO-LABEL-IN-ROUTE GUARDS PASS")


if __name__ == "__main__":
    main()
