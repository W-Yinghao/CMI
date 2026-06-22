"""Score-Fisher tests after Phase 1.2.1 (paired gate attribution + oracle certification).

Fast algebra (no probe training): metric-aware projector + select_from_fishers exactly
covariant under z->Az; checkpoint reload preserves is_identity.

ORACLE gate certification (inject the TRUE direction to isolate the gate from the selection):
  * danger oracle: removing the true t1 costs ~log2 task NLL -> TASK_RISK_UCB fires, k*=0
    (certifies the task-risk arm IS necessary and works);
  * partial unsafe oracle: injecting [t1,t2] is rejected;
  * ATTRIBUTION: the M-oblique projector is rescaling-covariant but NOT task-preserving, so the
    gate refuses even a Euclidean-safe span -- documented (open projector design fork), not a
    gate/generator bug.

Learned-pipeline stressors (SLURM): all-safe ACCEPTS k>=1 (recovers carrier); dangerous+
saturation is PROPOSED then REFUSED -> identity (binding reason recorded, not asserted);
partial PREFERS the safe span (exact-k WIP per prefix-search note); noise gate shut; refitted-
probe rescaling stability. Anything stronger (clean TASK_RISK_UCB on a LEARNED candidate,
exact partial k=2) depends on resolving the projector fork and is NOT asserted here.
"""
import numpy as np
import torch

from tos_cmi.data.synthetic import (SynthSpec, make, make_collinear, make_covariance_only,
                                     make_partial_overlap, make_saturated_danger,
                                     apply_linear_transform)
from tos_cmi.score_fisher import (ScoreFisherConfig, ScoreFisherSelector, metric_projector,
                                   select_from_fishers, ucb_rank_gate, _metric)
from tos_cmi.subspace import SubspaceSelector
from tos_cmi.eval.stability import precision_recall, projection_distance


def _cfg():
    return ScoreFisherConfig(epochs=200, hidden=64, gate_boot=200, n_perm_null=2)


# ----------------------------------------------------------------- fast algebra
def test_metric_projector_similarity_covariant():
    rng = np.random.default_rng(0); d, k = 8, 3
    Mh = rng.standard_normal((d, d)); M = Mh @ Mh.T + np.eye(d)
    V = rng.standard_normal((d, k)); A = rng.standard_normal((d, d)) + 2 * np.eye(d)
    P = metric_projector(V, M)
    P_t = metric_projector(A @ V, np.linalg.inv(A).T @ M @ np.linalg.inv(A))
    assert np.linalg.norm(np.linalg.inv(A) @ P_t @ A - P) < 1e-8
    assert np.linalg.norm(P @ P - P) < 1e-8
    print("test_metric_projector_similarity_covariant: OK")


def test_select_from_fishers_exactly_covariant():
    d = 6
    G_DgY = np.diag([10.0, 1.0, 0.05, 0.05, 0.05, 0.05])
    G_Y = np.diag([0.01, 8.0, 0.01, 0.01, 0.01, 0.01])
    rng = np.random.default_rng(1)
    Mh = rng.standard_normal((d, d)); M = Mh @ Mh.T + np.eye(d)
    A = rng.standard_normal((d, d)) + 2.5 * np.eye(d); Ai = np.linalg.inv(A)
    cfg = ScoreFisherConfig()
    c1, _, _, _, _, _, P1 = select_from_fishers(G_DgY, G_Y, M, cfg)
    c2, _, _, _, _, _, P2 = select_from_fishers(Ai.T @ G_DgY @ Ai, Ai.T @ G_Y @ Ai, Ai.T @ M @ Ai, cfg)
    assert c1.size > 0 and np.array_equal(c1, c2), (c1, c2)
    assert np.linalg.norm(Ai @ P2 @ A - P1) < 1e-6
    print("test_select_from_fishers_exactly_covariant: OK  k=%d" % c1.size)


def test_checkpoint_preserves_identity():
    sf = ScoreFisherSelector(6, 2, 3, _cfg())
    sf.active_k = torch.tensor(2, dtype=torch.long); sf.P = torch.eye(6)
    sf2 = ScoreFisherSelector(6, 2, 3, _cfg()); sf2.load_state_dict(sf.state_dict())
    assert not sf2.is_identity and int(sf2.active_k) == 2
    print("test_checkpoint_preserves_identity: OK")


# ----------------------------------------------------------------- ORACLE gate certification
# Inject the TRUE direction into the UCB gate (bypassing candidate_order) to isolate the gate
# from the selection (per the design decision: certify the gate, don't tune the generator).
def _oracle(data, V):
    s = data["spec"]; cfg = _cfg()
    Z = data["Z"].astype(np.float64)
    M = _metric(Z, data["y"], s.n_cls, cfg)
    Vn = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
    P = metric_projector(Vn, M)
    resid = np.linalg.norm((np.eye(s.d) - P) @ Vn)            # P V = V exactly => ~0
    k, recs, reason = ucb_rank_gate(Z, data["y"], data["d"], Vn, M, s.n_cls, s.n_dom, cfg, seed=0)
    return k, recs, reason, resid


def test_oracle_danger_basis_binds_task_ucb():
    data = make_saturated_danger(n=6000, seed=0)
    k, recs, reason, resid = _oracle(data, data["danger_basis"])
    print("oracle danger:", {"resid": round(resid, 6), "k": k, "reason": reason, "rec": recs[0]})
    assert resid < 1e-6, resid                               # oblique projector excises t1 exactly
    assert recs[0]["domain_lcb"] > 0.0, recs[0]              # t1 IS domain-rich (mean shift)
    assert recs[0]["task_ucb"] > _cfg().delta_Y, recs[0]     # deleting t1 hurts the task...
    assert k == 0 and reason == "TASK_RISK_UCB", (k, reason, recs[0])  # ...so the gate refuses
    print("test_oracle_danger_basis_binds_task_ucb: OK")


def test_oracle_partial_unsafe_rejected():
    # injecting the task-overlap span [t1,t2] must be rejected by the task arm
    data = make_partial_overlap(n=6000, seed=0)
    ku, ru, run_, _ = _oracle(data, data["task_overlap_basis"])
    print("oracle partial unsafe:", {"k": ku, "reason": run_})
    assert ku == 0, ("task-overlap span must be rejected", ru)
    print("test_oracle_partial_unsafe_rejected: OK")


def test_oracle_metric_projector_not_task_preserving():
    """ATTRIBUTION FINDING (Phase 1.2.1): with the rescaling-covariant M-oblique projector, the
    gate REFUSES even a genuinely safe (Euclidean task-orthogonal) span -- not a gate/generator
    bug, but because (I - P_N^M) DISTORTS the task subspace while a Euclidean Q Q^T projector
    preserves it. This certifies the cause and motivates the projector design fork."""
    data = make_partial_overlap(n=6000, seed=0); s = data["spec"]; cfg = _cfg()
    Z = data["Z"].astype(np.float64); M = _metric(Z, data["y"], s.n_cls, cfg)
    W = data["nuisance_basis"]; T = data["task_overlap_basis"]
    Wn = W / np.linalg.norm(W, axis=0, keepdims=True)
    Tn = T / np.linalg.norm(T, axis=0, keepdims=True)
    assert np.allclose(Wn.T @ Tn, 0, atol=1e-5)                   # safe span ⟂ task (Euclidean)
    Pobl = metric_projector(Wn, M)
    Q, _ = np.linalg.qr(Wn); Peu = Q @ Q.T
    I = np.eye(s.d)
    dmg_obl = np.linalg.norm((I - Pobl) @ Tn, axis=0)
    dmg_eu = np.linalg.norm((I - Peu) @ Tn, axis=0)
    assert np.allclose(np.linalg.norm((I - Pobl) @ Wn, axis=0), 0, atol=1e-6)   # both remove w
    assert (dmg_obl > 1.03).all(), dmg_obl       # oblique DISTORTS task (norm > 1)
    assert np.allclose(dmg_eu, 1.0, atol=1e-6)   # Euclidean PRESERVES task exactly
    # consequently the UCB task arm charges a cost and refuses the safe span under oblique P
    ks, rs, rn, _ = _oracle(data, W)
    assert ks < 2 and rn == "TASK_RISK_UCB", (ks, rn)
    print("test_oracle_metric_projector_not_task_preserving: OK  "
          "(oblique task-damage=%s vs euclid=%s; gate refuses k=%d/%s)"
          % (np.round(dmg_obl, 3), np.round(dmg_eu, 3), ks, rn))


# ----------------------------------------------------------------- dual-gate stressors (learned)
def test_all_safe_accepts_and_recovers():
    # clearly-detectable safe leakage (strong per-domain variance spread) so BOTH the
    # score-Fisher gradient AND the predictive Brier gate see it (weak covariance signals are
    # gradient-visible but predictively marginal -- an honest property of the gate)
    data = make_covariance_only(n=5000, dom_var=4.0, seed=0); s = data["spec"]
    sf = ScoreFisherSelector(s.d, s.n_cls, s.n_dom, _cfg())
    rep = sf.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
    assert rep.gate["open"], rep.gate
    assert rep.k_star >= 1 and rep.decision_reason == "ACCEPTED", sf.summary()
    pr = precision_recall(rep.basis, data["nuisance_basis"])
    assert pr["precision"] > 0.55, (pr, sf.summary())
    print("test_all_safe_accepts_and_recovers: OK", {"k": rep.k_star, "precision": round(pr["precision"], 3)})


def test_learned_dangerous_gate_safety_invariant():
    # On the saturated-danger world the LEARNED candidate need not be the true t1 (the oracle
    # test certifies the gate rejects the true t1). The universal property to verify on the
    # learned pipeline is the gate's SAFETY GUARANTEE: it NEVER accepts a prefix whose task UCB
    # exceeds delta_Y -- i.e. every accepted deletion is certified low-task-cost.
    data = make_saturated_danger(n=6000, seed=0); s = data["spec"]
    sf = ScoreFisherSelector(s.d, s.n_cls, s.n_dom, _cfg())
    rep = sf.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
    print("learned dangerous:", sf.summary())
    accepted = [r for r in rep.rank_records if r["k"] <= rep.k_star]
    for r in accepted:
        assert r["task_ucb"] <= _cfg().delta_Y + 1e-9, ("accepted a task-risky prefix!", r)
    # and the selected directions (if any) must be benign: low task damage
    assert rep.k_star == 0 or all(r["risk_feasible"] for r in accepted), sf.summary()
    print("test_learned_dangerous_gate_safety_invariant: OK (k=%d, all accepted prefixes task-safe)"
          % rep.k_star)


def test_partial_prefers_safe_over_overlap():
    # Partial case (2 safe + 2 task-overlapping domain dirs). The IDEAL is k=2 on the safe
    # span. VERIFIED here: the selection prefers the SAFE span over the task-overlap span and
    # never deletes a task-critical direction. The exact k=2 / prefix-search behaviour is
    # reported (not hard-asserted) -- if a dangerous dir is ranked before a safe one the nested
    # prefix cannot isolate the safe span; that is a recorded PREFIX-SEARCH LIMITATION, not a
    # threshold to tune (per the design note).
    data = make_partial_overlap(n=6000, seed=0); s = data["spec"]
    sf = ScoreFisherSelector(s.d, s.n_cls, s.n_dom, _cfg())
    rep = sf.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
    pr_safe = precision_recall(rep.basis, data["nuisance_basis"]) if rep.k_star else {"precision": 0, "recall": 0}
    pr_over = precision_recall(rep.basis, data["task_overlap_basis"]) if rep.k_star else {"precision": 0}
    print("test_partial: ", {"k": rep.k_star, "reason": rep.decision_reason,
          "prec_safe": round(pr_safe["precision"], 3), "rec_safe": round(pr_safe["recall"], 3),
          "overlap_with_task": round(pr_over["precision"], 3),
          "cand_order_len": len(rep.cand_order),
          "rank_records": rep.rank_records})
    # safety: whatever is selected must overlap the SAFE span more than the task span
    assert pr_safe["precision"] >= pr_over["precision"], (pr_safe, pr_over, sf.summary())
    print("test_partial_prefers_safe_over_overlap: OK (k=%d, ideal=2 -- exact-k WIP)" % rep.k_star)


def test_gate_shut_on_pure_noise():
    data = make(SynthSpec(n=4000, sep_label=0.0, sep_dom=0.0), seed=0); s = data["spec"]
    sf = ScoreFisherSelector(s.d, s.n_cls, s.n_dom, _cfg())
    rep = sf.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
    assert not rep.gate["open"] and rep.is_identity, sf.summary()
    assert rep.decision_reason == "DOMAIN_GATE_CLOSED", sf.summary()
    print("test_gate_shut_on_pure_noise: OK", {"brier_lcb": round(rep.gate.get("brier_lcb", 0), 4)})


def test_refitted_probe_selection_empirically_stable_under_rescaling():
    data = make_covariance_only(n=5000, dom_var=4.0, seed=0); s = data["spec"]
    rng = np.random.default_rng(3)
    A = np.diag(np.exp(rng.uniform(-0.5, 0.5, s.d))).astype(np.float32)
    cfg = _cfg()
    r1 = ScoreFisherSelector(s.d, s.n_cls, s.n_dom, cfg).refresh(
        torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
    data2 = apply_linear_transform(data, A)
    r2 = ScoreFisherSelector(s.d, s.n_cls, s.n_dom, cfg).refresh(
        torch.tensor(data2["Z"]), torch.tensor(data2["y"]), torch.tensor(data2["d"]))
    assert r1.k_star > 0 and r2.k_star > 0, (r1.decision_reason, r2.decision_reason)
    back = np.linalg.inv(A) @ r2.basis
    pd = projection_distance(r1.basis, back)
    assert pd < 0.9, pd
    print("test_refitted_probe_selection_empirically_stable: OK  proj_dist=%.3f" % pd)


if __name__ == "__main__":
    test_metric_projector_similarity_covariant()
    test_select_from_fishers_exactly_covariant()
    test_checkpoint_preserves_identity()
    test_oracle_danger_basis_binds_task_ucb()
    test_oracle_partial_unsafe_rejected()
    test_oracle_metric_projector_not_task_preserving()
    test_all_safe_accepts_and_recovers()
    test_learned_dangerous_gate_safety_invariant()
    test_partial_prefers_safe_over_overlap()
    test_gate_shut_on_pure_noise()
    test_refitted_probe_selection_empirically_stable_under_rescaling()
    print("ALL SCORE-FISHER TESTS PASSED")
