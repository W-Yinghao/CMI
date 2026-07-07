#!/usr/bin/env python
"""FSR Phase 8B — CodeBrain batch-size determinism invariance (STOP-1 disposition per PM). Compares the bs=64 and
bs=32 CodeBrain dumps: raw embedding diffs (F0, F1) + claim-level metric deltas (task, L1, L4, L5, L6). PASS iff
L1 delta <= 0.005, task delta <= 0.005, and L5/L6 signs + gate decision unchanged. Writes codebrain_batch_invariance.csv."""
import csv, json
from pathlib import Path
import numpy as np

OUT = Path("results/fsr_codebrain_cbramod_8b")


def emb_diff(a, b):
    A = np.load(OUT / "embeddings" / a)["X"]; B = np.load(OUT / "embeddings" / b)["X"]
    return dict(max_abs=round(float(np.max(np.abs(A - B))), 6),
                median_trial_l2=round(float(np.median(np.linalg.norm(A - B, axis=1))), 6),
                rel_median=round(float(np.median(np.linalg.norm(A - B, axis=1)) / (np.median(np.linalg.norm(A, axis=1)) + 1e-9)), 6))


def g(d, *ks):
    for k in ks:
        d = d.get(k) if isinstance(d, dict) else None
        if d is None:
            return None
    return d


def main():
    f0 = emb_diff("codebrain_shu_F0.npz", "codebrain_shu_F0_bs32.npz")
    f1 = emb_diff("codebrain_shu_F1.npz", "codebrain_shu_F1_bs32.npz")
    a = json.load(open(OUT / "f1_audit_summary_codebrain_shu.json"))
    b = json.load(open(OUT / "f1_audit_summary_codebrain_shu_bs32.json"))
    rows = []

    def cmp(name, va, vb, tol):
        dv = None if (va is None or vb is None) else round(abs(va - vb), 5)
        rows.append(dict(metric=name, bs64=va, bs32=vb, abs_delta=dv, tol=tol,
                         within_tol=bool(dv is not None and dv <= tol)))
        return dv

    def sv(s):
        x = s["source_val_bacc"]
        return x[0] if isinstance(x, list) else x
    d_sv = cmp("F1_source_val_bacc", sv(a), sv(b), 0.005)
    d_tb = cmp("F1_target_bacc", a["target_bacc"][0], b["target_bacc"][0], 0.005)
    d_l1 = cmp("F1_L1_marginal_bacc", g(a, "L1_marginal", "bacc"), g(b, "L1_marginal", "bacc"), 0.005)
    d_l4 = cmp("F1_L4_alignment_k2", a.get("L4_alignment_k2"), b.get("L4_alignment_k2"), 0.02)
    d_l5 = cmp("F1_L5_drop_subject_k2", a["L5_drop_subject_k2"][0], b["L5_drop_subject_k2"][0], 0.01)
    d_l6 = cmp("F1_L6_delta_subject_k2", a["L6_delta_subject_k2"][0], b["L6_delta_subject_k2"][0], 0.01)
    gate_same = bool(a.get("task_gate_pass") == b.get("task_gate_pass"))
    l5_sign_same = bool(a.get("L5_subject_beats_variance") == b.get("L5_subject_beats_variance"))

    passed = bool((d_l1 is not None and d_l1 <= 0.005) and (d_tb is not None and d_tb <= 0.005)
                  and gate_same and l5_sign_same and all(r["within_tol"] for r in rows))
    verdict = dict(feature_emb_diff=dict(F0=f0, F1=f1), gate_decision_same=gate_same,
                   L5_beats_variance_sign_same=l5_sign_same, all_metrics_within_tol=all(r["within_tol"] for r in rows),
                   invariance_pass=passed,
                   note=("CodeBrain bs64 vs bs32: raw embeddings differ by a tiny SGConv FFT batch-size numerical "
                         "path; PASS means every claim-level metric (task, L1, L4, L5, L6) and the gate + L5-sign "
                         "decisions are unchanged -> the audit conclusions are batch-size-invariant."))
    with open(OUT / "codebrain_batch_invariance.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["metric", "bs64", "bs32", "abs_delta", "tol", "within_tol"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    (OUT / "codebrain_batch_invariance_verdict.json").write_text(json.dumps(verdict, indent=2, default=str) + "\n")
    print("F0 emb diff:", f0); print("F1 emb diff:", f1)
    for r in rows:
        print(f"  {r['metric']}: bs64={r['bs64']} bs32={r['bs32']} |delta|={r['abs_delta']} within_tol={r['within_tol']}")
    print(f"gate_same={gate_same} L5_sign_same={l5_sign_same} ==> INVARIANCE_PASS={passed}")


if __name__ == "__main__":
    main()
