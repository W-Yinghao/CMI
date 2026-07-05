"""R3 secondary — node masking. Structure, determinism, and source-fit firewall (supportive evidence only)."""
import numpy as np
from cmi.eval.node_masking import node_masking_curve

N_DOM, C, ZN, N_PER, TARGET = 4, 6, 3, 40, 3
LEAK_NODES = [0, 1]                                             # channels carrying subject info


def make_node_data(seed=0):
    rng = np.random.default_rng(seed)
    nz, ys, ds = [], [], []
    for dd in range(N_DOM):
        for cc in range(2):
            b = rng.standard_normal((N_PER, C, ZN)) * 0.2
            b[:, 2, 0] += 3.0 * cc                             # task signal on a low-leakage channel
            for ch in LEAK_NODES:
                b[:, ch, 0] += (dd - 1.5) * 2.0                # subject signal on leak channels
            nz.append(b); ys += [cc] * N_PER; ds += [dd] * N_PER
    node_z = np.concatenate(nz); y = np.array(ys); d = np.array(ds)
    lm = np.zeros(C); lm[LEAK_NODES] = 1.0                     # node leakage map ranks leak channels high
    return {"node_z": node_z, "y": y, "d": d, "node_leakage_map": lm}


def test_node_masking_schema_and_finite():
    r = node_masking_curve(make_node_data(), TARGET, mask_k=2, n_random=15, seed=0)
    for k in ("mask_k", "base_task_bacc", "top_leak_mask_drop", "bottom_leak_mask_drop",
              "random_mask_drop_mean", "random_mask_drop_ci", "top_exceeds_random"):
        assert k in r
    assert np.isfinite(r["base_task_bacc"]) and len(r["random_mask_drop_ci"]) == 2
    assert np.isfinite(r["random_mask_drop_ci"]).all()


def test_node_masking_deterministic():
    a = node_masking_curve(make_node_data(), TARGET, mask_k=2, n_random=15, seed=0)
    b = node_masking_curve(make_node_data(), TARGET, mask_k=2, n_random=15, seed=0)
    assert a["random_mask_drop_mean"] == b["random_mask_drop_mean"] and a["random_mask_drop_ci"] == b["random_mask_drop_ci"]


def test_mask_k_clamped_to_channel_count():
    r = node_masking_curve(make_node_data(), TARGET, mask_k=99, n_random=5, seed=0)
    assert r["mask_k"] == C
