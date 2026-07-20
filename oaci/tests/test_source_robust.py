"""C11b SRC (source-robust) method: objective (smooth worst-domain balanced CE over SOURCE domains, no
critic) + source-train-only endpoint selector (role-gated, ERM fallback, K1 measurement-only). Objective
math is checked against a manual log-sum-exp of the repo's own balanced_ce; selector tests use synthetic
per-candidate endpoint tables. No GPU."""
from __future__ import annotations

import torch

from oaci.methods.source_robust import SRCObjective
from oaci.train.objective import BatchView
from oaci.train.risk import balanced_ce


def _obj(nc=2, nd=2, temp=0.1):
    o = SRCObjective(n_classes=nc, n_source_domains=nd, smooth_temperature=temp)
    torch.manual_seed(0)
    o._classifier = torch.nn.Linear(4, nc)              # stand in for model.classifier (stashed in full_surrogate)
    return o


def test_source_robust_objective_uses_source_domains_not_target():
    o = _obj()
    z = torch.randn(12, 4)
    y = torch.tensor([0, 1] * 6)
    d = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])   # two SOURCE domains — no target concept exists
    w = torch.ones(12)
    pen = o.encoder_penalty(None, z, BatchView(y, d, w))
    assert pen.requires_grad and pen.ndim == 0
    # it MUST use domain ids: a domain-less batch is rejected (target is never part of the objective)
    try:
        o.encoder_penalty(None, z, BatchView(y, None, w))
    except ValueError:
        pass
    else:
        raise AssertionError("SRC objective must require source domain ids")


def test_source_robust_smooth_worst_matches_manual_lse():
    for temp in (0.05, 0.1, 0.5):
        o = _obj(nc=2, nd=3, temp=temp)
        z = torch.randn(30, 4)
        y = torch.randint(0, 2, (30,))
        d = torch.randint(0, 3, (30,))
        w = torch.rand(30) + 0.1
        logits = o._classifier(z)
        # manual: per-domain balanced_ce (the repo primitive) then temperature-smoothed max
        per = [balanced_ce(logits[d == dd], y[d == dd], n_classes=2, weight=w[d == dd]) for dd in torch.unique(d)]
        manual = temp * torch.logsumexp(torch.stack(per) / temp, dim=0)
        got = o.encoder_penalty(None, z, BatchView(y, d, w))
        assert torch.allclose(got, manual, atol=1e-6), (temp, float(got), float(manual))


def test_smooth_max_upper_bounds_true_worst_and_shrinks_with_temp():
    o = _obj(nc=2, nd=3, temp=0.1)
    z = torch.randn(30, 4); y = torch.randint(0, 2, (30,)); d = torch.randint(0, 3, (30,)); w = torch.ones(30)
    logits = o._classifier(z)
    true_worst = max(float(balanced_ce(logits[d == dd], y[d == dd], n_classes=2)) for dd in torch.unique(d))
    lse01 = float(o.encoder_penalty(None, z, BatchView(y, d, w)))
    o2 = _obj(nc=2, nd=3, temp=0.5); o2._classifier = o._classifier
    lse05 = float(o2.encoder_penalty(None, z, BatchView(y, d, w)))
    assert lse01 >= true_worst - 1e-6 and lse05 >= true_worst - 1e-6      # LSE upper-bounds the true max
    assert lse01 <= lse05 + 1e-6                                          # smaller temp -> tighter to the max


def test_source_robust_inactive_below_min_domains_and_no_critic():
    assert SRCObjective(n_classes=2, n_source_domains=1).active_status().active is False
    o = SRCObjective(n_classes=2, n_source_domains=2)
    assert o.active_status().active is True
    assert o.build_critic(40, "cpu") is None                             # non-adversarial
    try:
        o.critic_loss(None, None, None)
    except RuntimeError:
        pass
    else:
        raise AssertionError("SRC has no critic; critic_loss must raise")


# ---- source-train-only endpoint selector ----
from oaci.runner.source_endpoint_selector import _AccessLog, _Gated, select_source_endpoint


def _row(h, *, is_erm=False, feasible=True, R_src=0.8, gb=0.50, gn=1.20, ge=0.10,
         sa_bacc=0.5, tb=0.5, k1p=0.3, leak=0.8):
    return {"model_hash": h, "is_erm": is_erm, "feasible": feasible, "epoch": 100, "R_src": R_src,
            "source_guard_worst_bacc": gb, "source_guard_worst_nll": gn, "source_guard_worst_ece": ge,
            # forbidden fields present in the table (the pilot carries them) but must NOT be read:
            "source_audit_worst_bacc": sa_bacc, "target_worst_bacc": tb, "k1_p_lower": k1p,
            "audit_leakage_point": leak, "selection_leakage_point": leak}


def test_source_endpoint_selector_uses_source_train_only():
    tbl = [_row("erm", is_erm=True, gb=0.50, gn=1.20), _row("s1", R_src=0.79, gb=0.52, gn=1.05),
           _row("s2", R_src=0.81, gb=0.51, gn=1.15)]
    r = select_source_endpoint(tbl, tau=0.85)
    assert set(r["access"]["roles_actually_read"]) <= {"meta", "source_risk", "source_guard"}
    assert not r["access"]["target_read"] and not r["access"]["read_source_audit"] and not r["access"]["read_leakage"]
    assert r["access"]["forbidden_fields"] == []
    assert r["chosen_model_hash"] == "s1"                          # feasible + passes guard + min source NLL


def test_source_endpoint_selector_rejects_source_audit_or_target_access():
    g = _Gated(_row("x"), _AccessLog())
    assert g["source_guard_worst_nll"] == 1.20 and g["R_src"] == 0.8      # allowed
    for forbidden in ("target_worst_bacc", "source_audit_worst_bacc"):
        try:
            _ = g[forbidden]
        except PermissionError:
            continue
        raise AssertionError(f"reading {forbidden} must raise")


def test_src_k1_is_measurement_only_not_selection_score():
    # reading K1 / leakage inside the selector must raise -> K1 can never drive SRC selection
    g = _Gated(_row("x"), _AccessLog())
    for meas in ("k1_p_lower", "audit_leakage_point", "selection_leakage_point"):
        try:
            _ = g[meas]
        except PermissionError:
            continue
        raise AssertionError(f"reading {meas} (a measurement) must raise in the selector")
    # and a full selection never reports having read leakage
    r = select_source_endpoint([_row("erm", is_erm=True), _row("s1", R_src=0.79, gn=1.0)], tau=0.85)
    assert not r["access"]["read_leakage"]


def test_source_endpoint_selector_falls_back_to_erm():
    # every SRC candidate harms source_guard worst bAcc beyond the margin -> guards fail -> ERM
    tbl = [_row("erm", is_erm=True, gb=0.55, gn=1.10, ge=0.10),
           _row("s1", R_src=0.79, gb=0.40, gn=1.05, ge=0.10),   # bAcc drop 0.15 >> 0.02 margin
           _row("s2", R_src=0.80, gb=0.42, gn=1.02, ge=0.10)]
    r = select_source_endpoint(tbl, tau=0.85)
    assert r["fallback_erm"] and r["chosen_model_hash"] == "erm" and r["selection_reason"] == "erm_fallback"
    assert r["n_guard_pass"] == 0
    # infeasible-risk candidate is also excluded
    tbl2 = [_row("erm", is_erm=True), _row("bad", R_src=1.0, gb=0.60, gn=0.9)]   # R_src 1.0 > tau 0.85
    assert select_source_endpoint(tbl2, tau=0.85)["chosen_model_hash"] == "erm"


def test_source_endpoint_selector_is_order_invariant():
    tbl = [_row("erm", is_erm=True, gn=1.20), _row("s1", R_src=0.79, gn=1.05), _row("s2", R_src=0.80, gn=1.05)]
    a = select_source_endpoint(tbl, tau=0.85)["chosen_model_hash"]
    b = select_source_endpoint(list(reversed(tbl)), tau=0.85)["chosen_model_hash"]
    assert a == b                                                 # tie on NLL -> deterministic hash tie-break


# ---- pilot report (deep verify + target isolation) on a synthetic pilot body ----
from oaci.confirmatory.src_onefold import _assemble, _signal, deep_verify_pilot, render_md


def _method_block(*, tb, tn, erm_tb, erm_tn, leak, erm_leak, feasible=True, fallback=False, target_read=False):
    return {"model_hash": "m", "target_worst_bacc": tb, "target_worst_nll": tn, "target_worst_ece": 0.1,
            "source_audit_worst_bacc": tb, "source_audit_worst_nll": tn, "source_audit_worst_ece": 0.1,
            "source_guard_worst_bacc": tb, "source_guard_worst_nll": tn, "source_guard_worst_ece": 0.1,
            "audit_leakage_point": leak, "selected_R_src": 0.8, "risk_feasible": feasible, "fallback_erm": fallback,
            "selection_reason": ("erm_fallback" if fallback else "source_endpoint_best"), "n_feasible": 5,
            "n_guard_pass": 0 if fallback else 3,
            "access": {"allowed_roles": ["source_guard", "source_risk"], "roles_actually_read": ["source_guard", "source_risk"],
                       "target_read": target_read, "read_source_audit": False, "read_leakage": False, "forbidden_fields": []},
            "K2_delta_target_worst_bacc": tb - erm_tb, "K2_delta_target_worst_nll": tn - erm_tn,
            "K2_delta_target_worst_ece": 0.0, "K2_delta_source_audit_worst_bacc": tb - erm_tb,
            "K1_delta_audit_leakage_MEASUREMENT_ONLY": (None if (leak is None or erm_leak is None) else leak - erm_leak)}


def _pilot_body(*, src_tb, src_tn, target_read=False, fallback=False):
    erm_tb, erm_tn, erm_leak = 0.50, 1.20, 0.80
    levels = {}
    for L in (0, 1):
        erm = {"model_hash": "e", "target_worst_bacc": erm_tb, "target_worst_nll": erm_tn, "target_worst_ece": 0.1,
               "source_audit_worst_bacc": erm_tb, "source_audit_worst_nll": erm_tn, "source_audit_worst_ece": 0.1,
               "source_guard_worst_bacc": erm_tb, "source_guard_worst_nll": erm_tn, "source_guard_worst_ece": 0.1,
               "audit_leakage_point": erm_leak}
        levels[str(L)] = {"n_source_domains": 8, "tau": 0.85, "smooth_temperature": 0.1, "ERM": erm,
                          "OACI": _method_block(tb=0.48, tn=1.25, erm_tb=erm_tb, erm_tn=erm_tn, leak=0.82, erm_leak=erm_leak),
                          "SRC": _method_block(tb=src_tb, tn=src_tn, erm_tb=erm_tb, erm_tn=erm_tn, leak=0.79,
                                               erm_leak=erm_leak, fallback=fallback, target_read=target_read),
                          "target_fit_ids_empty": True, "src_selector_target_read": target_read}
    return _assemble("BNCI2014_001", 1, 0, 0.1, levels)


def test_src_artifact_deep_verifies():
    body = _pilot_body(src_tb=0.55, src_tn=1.10)
    assert deep_verify_pilot(body)                               # pilot_hash round-trips
    tampered = {**body, "levels": {**body["levels"], "0": {**body["levels"]["0"], "tau": 999.0}}}
    assert not deep_verify_pilot(tampered)                       # any edit breaks the hash


def test_src_target_fit_ids_empty():
    body = _pilot_body(src_tb=0.55, src_tn=1.10)
    assert body["all_target_fit_ids_empty"] and body["no_selector_read_target"]
    leaky = _pilot_body(src_tb=0.55, src_tn=1.10, target_read=True)
    assert leaky["no_selector_read_target"] is False            # a selector target read is detected


def test_src_signal_and_report_canonical_serializable():
    from oaci.artifacts.canonical_json import canonical_json_bytes
    win = _pilot_body(src_tb=0.55, src_tn=1.10)                  # SRC beats ERM on target -> signal
    lose = _pilot_body(src_tb=0.48, src_tn=1.30, fallback=True)  # SRC worse / falls back -> no signal
    assert _signal(win)["SRC_shows_signal"] is True and _signal(lose)["SRC_shows_signal"] is False
    assert isinstance(render_md(win), str) and "## C11c signal" in render_md(win)
    assert canonical_json_bytes({**win, "signal": _signal(win)})   # serializes (no int keys)


def test_no_oaci_runtime_import_from_cmi_or_h2cmi():
    import sys
    import oaci.methods.source_robust  # noqa: F401
    import oaci.runner.source_endpoint_selector  # noqa: F401
    import oaci.confirmatory.src_onefold  # noqa: F401
    leaked = [m for m in sys.modules if m == "cmi" or m.startswith("cmi.") or m == "h2cmi" or m.startswith("h2cmi.")]
    assert leaked == [], leaked


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} src objective tests")


if __name__ == "__main__":
    _run_all()
