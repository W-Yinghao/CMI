"""Phase 2.0 -- offline DIAGNOSTIC on the frozen-feature dumps (CPU; no GPU, no training of the
encoder). Per dump it reports (a) collapse metrics and (b) the score-Fisher diagnostic. The
certified gate runs in EXPLORATORY mode (task_power_floor OFF) -- on real EEG with no matched power
table the expected, correct behaviour is IDENTITY / abstain; the decision_reason is the diagnostic.

  python -m tos_cmi.eeg.report tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_TSMNet_LOSO
"""
from __future__ import annotations
import glob
import json
import sys
import numpy as np

from tos_cmi.score_fisher import ScoreFisherConfig, select_score_fisher


def _bacc(logits, y, n_cls):
    pred = logits.argmax(1)
    recalls = [(pred[y == c] == c).mean() for c in range(n_cls) if (y == c).any()]
    return float(np.mean(recalls)) if recalls else float("nan")


def _nll(logits, y):
    z = logits - logits.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def _eff_rank(Z):
    Zc = Z - Z.mean(0, keepdims=True)
    s = np.linalg.svd(Zc, compute_uv=False)
    p = s / max(s.sum(), 1e-12); p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))            # participation/entropy effective rank


def _linear_probe_acc(Z, lab, n_lab, seed=0, frac=0.5):
    """Quick held-out linear (multinomial logistic) probe accuracy; chance = 1/n_lab."""
    try:
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(seed); idx = rng.permutation(len(lab))
        cut = int(frac * len(lab)); tr, te = idx[:cut], idx[cut:]
        if len(np.unique(lab[tr])) < 2:
            return float("nan")
        clf = LogisticRegression(max_iter=300, multi_class="auto")
        clf.fit(Z[tr], lab[tr])
        return float((clf.predict(Z[te]) == lab[te]).mean())
    except Exception as e:
        return float("nan")


def analyze(npz_path, cfg=None):
    d = np.load(npz_path, allow_pickle=True)
    Zs = d["Z_source"].astype(np.float64); ys = d["y_source"]; doms = d["domain_source"]
    subj = d["subject_source"]; n_cls = int(d["n_cls"]); n_dom = int(d["n_dom_source"])
    out = {"tag": npz_path.split("/")[-1], "method": str(d["method"]), "lam": float(d["lam"]),
           "target_subject": int(d["target_subject"]), "z_dim": int(d["z_dim"]),
           "n_source": int(len(ys)), "n_dom_source": n_dom,
           "target_bacc": _bacc(d["logits_target"], d["y_target"], n_cls),
           "source_bacc_fit": _bacc(d["logits_source"], ys, n_cls),   # backbone-fit (not held-out)
           "target_nll": _nll(d["logits_target"], d["y_target"]),
           "source_nll_fit": _nll(d["logits_source"], ys),
           "eff_rank_source": _eff_rank(Zs),
           "label_probe_acc": _linear_probe_acc(Zs, ys, n_cls),
           "label_chance": 1.0 / n_cls,
           "domain_probe_acc": _linear_probe_acc(Zs, doms, n_dom),
           "domain_chance": 1.0 / n_dom}
    out["domain_probe_adv"] = (out["domain_probe_acc"] - out["domain_chance"]
                               if out["domain_probe_acc"] == out["domain_probe_acc"] else float("nan"))
    cfg = cfg or ScoreFisherConfig()                        # exploratory: task_power_floor OFF
    def _sf(cluster_id):
        try:
            rep = select_score_fisher(Zs, ys, doms, n_cls, n_dom, cfg, seed=0, cluster_id=cluster_id)
            recs = rep.rank_records or []
            return {"decision_reason": rep.decision_reason, "k_star": int(rep.k_star),
                    "is_identity": bool(rep.is_identity),
                    "gate_open": bool(rep.gate.get("open")) if rep.gate else None,
                    "domain_brier_lcb": (rep.gate.get("brier_lcb") if rep.gate else None),
                    "eigengap_k": int(getattr(rep, "eigengap_k", 0) or 0),
                    "rho_top": [round(float(x), 4) for x in np.atleast_1d(rep.rho)[:6]],
                    "k1_task_ucb": (recs[0].get("probe_task_gain_ucb") if recs else None),
                    "k1_domain_lcb": (recs[0].get("domain_lcb") if recs else None)}
        except Exception as e:
            return {"error": repr(e)[:300]}
    # PRIMARY (exploratory): trial-level folds so the selector actually runs (domain==subject means
    # group-aware folds can't cover all subjects per fold). SECONDARY (group-aware): records the
    # certification caveat -- when domain==cluster==subject, certified deletion is infeasible
    # (FOLD_COVERAGE_FAILURE), consistent with the honest-negative certification line.
    out["scorefisher"] = _sf(cluster_id=None)
    out["scorefisher_groupaware"] = {"decision_reason": _sf(cluster_id=subj).get("decision_reason"),
                                     "note": "domain==subject -> group-aware coverage caveat"}
    return out


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "tos_cmi/results/tos_cmi_eeg_frozen"
    paths = sorted(glob.glob("%s/*.npz" % base)) or sorted(glob.glob("%s/**/*.npz" % base, recursive=True))
    rows = []
    for p in paths:
        r = analyze(p)
        rows.append(r); sf = r.get("scorefisher", {})
        print("[%s] tgt%d %s lam=%g | tgt_bAcc=%.3f src_bAcc_fit=%.3f effrank=%.1f "
              "labelP=%.2f domP=%.2f(adv %.2f) | SF: %s k=%s"
              % (r["method"], r["target_subject"], "", r["lam"], r["target_bacc"],
                 r["source_bacc_fit"], r["eff_rank_source"], r["label_probe_acc"],
                 r["domain_probe_acc"], r.get("domain_probe_adv", float("nan")),
                 sf.get("decision_reason"), sf.get("k_star")), flush=True)
    out = "%s/diagnostic_report.json" % base
    json.dump(rows, open(out, "w"), indent=1)
    print("wrote", out, "(%d dumps)" % len(rows))
    print("EEG_REPORT_DONE")


if __name__ == "__main__":
    main()
