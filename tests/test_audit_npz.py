"""CIGL R2->R3 — .audit.npz sidecar: save/load round-trip + schema validation (feeds R3 reliance tests)."""
import numpy as np

from cmi.eval.audit_npz import save_audit_npz, load_audit_npz, validate_audit_npz


def _payload(N=30, C=8, Zg=64, Zn=16, n_cls=2, n_dom=3):
    rng = np.random.default_rng(0)
    return dict(graph_z=rng.standard_normal((N, Zg)), node_z=rng.standard_normal((N, C, Zn)),
                y=rng.integers(0, n_cls, N), d=rng.integers(0, n_dom, N),
                model_logits=rng.standard_normal((N, n_cls)), probe_logits=rng.standard_normal((N, n_dom)),
                probe_predictions=rng.integers(0, n_dom, N), node_leakage_map=rng.random(C))


def test_save_load_roundtrip(tmp_path):
    pl = _payload()
    p = save_audit_npz(tmp_path / "f0", fold=0, seed=1, target_subject="3", method="cigl_graph_node",
                       dataset="2a", **pl)
    assert p.endswith(".audit.npz")
    d = load_audit_npz(p)
    assert d["fold"] == 0 and d["seed"] == 1 and d["target_subject"] == "3" and d["method"] == "cigl_graph_node"
    assert np.allclose(d["graph_z"], pl["graph_z"]) and d["node_z"].shape == pl["node_z"].shape
    assert "node_leakage_map" in d and d["node_leakage_map"].shape == (8,)


def test_validate_catches_problems(tmp_path):
    pl = _payload()
    p = save_audit_npz(tmp_path / "ok", fold=0, seed=0, target_subject="1", **pl)
    assert validate_audit_npz(load_audit_npz(p)) == []               # valid
    bad = load_audit_npz(p); del bad["node_z"]
    assert any("node_z" in s for s in validate_audit_npz(bad))       # missing required
    bad2 = load_audit_npz(p); bad2["d"] = bad2["d"][:5]
    assert any("!= N" in s for s in validate_audit_npz(bad2))        # N mismatch
