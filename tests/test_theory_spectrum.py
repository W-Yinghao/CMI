"""Unit tests for the CMI-Trace theory-spectrum modules (E1 subject_spectrum, E2 rank_threshold, E3 kstar).

Synthetic, fast, no GPU. E3 asserts the EXACT squared-loss identity (the pre-registered QC gate). E1/E2 use a
controlled DGP with a KNOWN linear head + injected subject/task structure and assert structural predictions.
"""
import numpy as np
import pytest

from cmi.eval import subject_spectrum as ss
from cmi.eval import rank_threshold as rt
from cmi.eval import kstar_worlds as kw


# ------------------------------------------------------------------ synthetic audit sidecar (E1/E2)
def make_synth_audit(n_sub=6, per=120, dz=12, n_cls=3, seed=0, subject_task_coupling=0.0):
    """Z = task(y) + subject(d) + noise, with a KNOWN linear head reading task dirs. Subject dirs are (mostly)
    orthogonal to the head. `subject_task_coupling` rotates one subject dir toward a head dir (raises its tau)."""
    rng = np.random.default_rng(seed)
    task_dirs = np.linalg.qr(rng.standard_normal((dz, n_cls)))[0][:, :n_cls]      # [dz, n_cls]
    subj_dirs = np.linalg.qr(rng.standard_normal((dz, dz)))[0][:, n_cls:n_cls + 4]  # [dz,4] orthogonal-ish
    if subject_task_coupling > 0:
        subj_dirs[:, 0] = ((1 - subject_task_coupling) * subj_dirs[:, 0]
                           + subject_task_coupling * task_dirs[:, 0])
        subj_dirs[:, 0] /= np.linalg.norm(subj_dirs[:, 0])
    Z, y, d = [], [], []
    subj_off = rng.standard_normal((n_sub, subj_dirs.shape[1])) * 2.0
    for s in range(n_sub):
        yy = rng.integers(0, n_cls, per)
        z = 0.4 * rng.standard_normal((per, dz))
        for c in range(n_cls):
            z[yy == c] += 1.5 * task_dirs[:, c]
        z += subj_off[s] @ subj_dirs.T                        # subject-within-label offset
        Z.append(z); y.append(yy); d.append(np.full(per, s))
    Z = np.vstack(Z).astype(np.float32); y = np.concatenate(y); d = np.concatenate(d)
    W = task_dirs.T.astype(np.float64); b = np.zeros(n_cls)   # head reads task dirs: logits = Z W^T
    logits = (Z @ W.T + b).astype(np.float32)
    target = n_sub - 1
    src_idx = np.where(d != target)[0]; tgt_idx = np.where(d == target)[0]
    return {
        "graph_z": Z, "node_z": Z[:, None, :], "y": y.astype(np.int64), "d": d.astype(np.int64),
        "model_logits": logits, "task_head_weight": W, "task_head_bias": b,
        "task_head_kind": "linear", "task_head_input": "graph_z", "task_head_replay_ok": True,
        "task_head_replay_max_abs_diff": 0.0, "dataset": "SYNTH", "method": "erm", "seed": seed, "fold": 0,
        "target_subject": target, "source_indices": src_idx, "target_indices": tgt_idx,
    }, W, logits


# ------------------------------------------------------------------ E3: the exact identity (QC gate)
def test_e3_identity_exact_and_worlds_separate():
    out = kw.run_worlds(spur_strength=3.0, seed=0)
    assert out["identity_ok"], f"squared-loss identity violated: {out}"
    assert out["beneficial"]["identity_residual"] < 1e-10
    assert out["legitimate"]["identity_residual"] < 1e-10
    # Prop-2 prediction on this DGP
    assert out["beneficial"]["K_star"] < 1.0, out["beneficial"]
    assert out["legitimate"]["K_star"] > 1.0, out["legitimate"]
    assert out["beneficial"]["gain_direct"] > 0 > out["legitimate"]["gain_direct"]
    assert out["worlds_separate"]


def test_e3_sign_of_1_minus_kstar_matches_gain():
    for seed in range(4):
        out = kw.run_worlds(spur_strength=2.5, seed=seed)
        for world in ("beneficial", "legitimate"):
            w = out[world]
            assert np.sign(1.0 - w["K_star"]) == np.sign(w["gain_star"]) or abs(w["gain_star"]) < 1e-12


def test_e3_two_worlds_share_source():
    W = kw.make_two_worlds(spur_strength=3.0, seed=1)
    # source identical across worlds by construction; target differs only on spur columns
    assert W["beneficial"]["Z"].shape == W["legitimate"]["Z"].shape
    diff = np.abs(W["beneficial"]["Z"] - W["legitimate"]["Z"])
    nonspur = [j for j in range(W["beneficial"]["Z"].shape[1]) if j not in W["spur"]]
    assert diff[:, nonspur].max() < 1e-9                      # target-X (non-spur) shared
    assert diff[:, W["spur"]].max() > 1e-6                    # spur block flipped


# ------------------------------------------------------------------ E1: whitened subject spectrum
def test_e1_runs_and_firewall_and_shapes():
    data, W, _ = make_synth_audit(seed=1)
    spec = ss.subject_spectrum(data, k_spec=4, n_perm=6, n_random=8, seed=0)
    assert spec["firewall_passed"] and spec["head_replay_verified"]
    assert spec["d_z"] == 12 and len(spec["directions"]) == 4
    assert spec["effective_rank"] > 0 and 0 <= spec["top2_energy_concentration"] <= 1.0001
    for r in spec["directions"]:
        assert np.isfinite(r["lambda_excess_over_null"]) and np.isfinite(r["tau_ce_reliance"])


def test_e1_task_coupled_direction_has_higher_reliance():
    """A subject direction rotated toward the head (coupled) should have higher exact-head tau than the
    orthogonal case — the mechanism E1 measures."""
    d_ortho, *_ = make_synth_audit(seed=2, subject_task_coupling=0.0)
    d_coup, *_ = make_synth_audit(seed=2, subject_task_coupling=0.8)
    s_o = ss.subject_spectrum(d_ortho, k_spec=4, n_perm=4, n_random=6, seed=0)
    s_c = ss.subject_spectrum(d_coup, k_spec=4, n_perm=4, n_random=6, seed=0)
    tau_o = max(r["tau_ce_reliance"] for r in s_o["directions"])
    tau_c = max(r["tau_ce_reliance"] for r in s_c["directions"])
    assert tau_c > tau_o, f"coupled tau {tau_c} should exceed orthogonal {tau_o}"


def test_e1_matching_and_delta_lambda():
    d1, *_ = make_synth_audit(seed=3)
    d2, *_ = make_synth_audit(seed=4)
    s1 = ss.subject_spectrum(d1, k_spec=4, n_perm=4, n_random=4, seed=0)
    s2 = ss.subject_spectrum(d2, k_spec=4, n_perm=4, n_random=4, seed=0)
    pairs = ss.paired_delta_lambda(s1, s2)
    assert len(pairs) == 4
    for p in pairs:
        assert 0.0 <= p["abs_cos"] <= 1.0001 and np.isfinite(p["delta_lambda"])


# ------------------------------------------------------------------ E2: rank threshold + geometry
def test_e2_rD_and_structural():
    data, W, logits = make_synth_audit(seed=5)
    src = data["d"] != data["target_subject"]
    rec = rt.rank_threshold_fold(data["graph_z"][src], data["y"][src], data["d"][src], logits[src],
                                 seed=0, n_perm=6)
    assert rec["r_D"] >= 1 and rec["head_exact"]              # synthetic head is exactly linear
    # geometry consistency: S_D splits between the head kernel and row space, no more than r_D total
    assert rec["dim_SD_in_ker"] + rec["dim_SD_in_row"] <= rec["r_D"]
    assert 0.0 <= rec["logit_change_remove_SD_relative"]      # finite, non-negative
    # informed eraser removes subject info at least as fast as a same-rank random one at k=1
    k1 = rec["sweep"][0]
    assert k1["resid_subject_bacc_linear"] <= k1["resid_subject_bacc_random_linear"] + 0.15
    # residual subject decodability is (weakly) non-increasing in eraser rank for the informed projector
    lin = [s["resid_subject_bacc_linear"] for s in rec["sweep"]]
    assert lin[-1] <= lin[0] + 0.05


def test_e2_geometry_primitives_controlled_kernel():
    """Directly validate kernel_rowspace + subspace_intersection_dim + logit-change with a KNOWN whitened
    head whose kernel contains the subject span -> exact-head-safe (logit change ~ 0)."""
    rng = np.random.default_rng(0)
    dz = 6
    Zt = rng.standard_normal((400, dz))
    W_tilde = np.zeros((2, dz)); W_tilde[0, 0] = 1.3; W_tilde[1, 1] = -0.8   # head uses only e0,e1
    ker, row = rt.kernel_rowspace(W_tilde)
    assert len(row) == 2 and len(ker) == dz - 2
    S_D = np.eye(dz)[[3, 4]]                                   # subject span e3,e4 -> inside ker
    assert rt.subspace_intersection_dim(S_D, ker) == 2
    assert rt.subspace_intersection_dim(S_D, row) == 0
    P_SD = S_D.T @ S_D
    logit_change = float(np.linalg.norm(Zt @ P_SD @ W_tilde.T, axis=1).mean())
    assert logit_change < 1e-9                                 # removing a kernel subspace does not move logits
    # a subject span that overlaps the row space DOES move logits
    S_bad = np.eye(dz)[[0, 3]]
    assert rt.subspace_intersection_dim(S_bad, row) == 1
    assert float(np.linalg.norm(Zt @ (S_bad.T @ S_bad) @ W_tilde.T, axis=1).mean()) > 1e-3


def test_e2_head_recovery_failclosed():
    data, W, logits = make_synth_audit(seed=6)
    Wc, bc, ok, mx = rt.recover_linear_head(data["graph_z"], logits, tol=1e-6)
    assert ok and mx < 1e-6                                   # exactly-linear synthetic head recovers
    # break linearity -> fail closed
    nl = logits + 0.5 * (data["graph_z"][:, :logits.shape[1]] ** 2)
    _, _, ok2, mx2 = rt.recover_linear_head(data["graph_z"], nl, tol=1e-6)
    assert not ok2 and mx2 > 1e-6


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-x"]))
