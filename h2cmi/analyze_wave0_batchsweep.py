"""WAVE 0 / W0.4 analyzer (frozen W0.4_MECH_APPENDUM). By-n curves + primary endpoint contrast
Delta_{256-16}. Average draws AND seeds within subject before cluster bootstrap (draws are NOT
pseudo-subjects). Emits QC sentinels."""
from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np

NB = 10000
NS = [16, 32, 64, 128, 256]
FIELDS = ["P_J", "transfer", "prior_estimate_deviation", "TV_piJ_rhoA_draw", "TV_piJ_rhoA_full",
          "min_piJ", "H_piJ", "missing_classes"]


def _load():
    rows = []
    for f in glob.glob("results/h2cmi/wave0_batchsweep/w0p4_*.jsonl"):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r.get("marker") == "BATCHSWEEP" and "P_J" in r:
                    rows.append(r)
    return rows


def _subject_n(rows, field, n):
    """subject -> mean over (seed, draw) at this n."""
    by = defaultdict(list)
    for r in rows:
        if int(r["n"]) == n:
            by[int(r["target_subject"])].append(float(r[field]))
    return {s: float(np.mean(v)) for s, v in by.items()}


def _boot(subj_vals, seed=0):
    v = np.array([subj_vals[s] for s in subj_vals], float)
    if len(v) < 2:
        return dict(mean=float(v.mean()) if len(v) else float("nan"), ci=[float("nan")] * 2, n=len(v))
    rng = np.random.default_rng(seed)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(NB)]
    return dict(mean=round(float(v.mean()), 4), ci=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                excludes_0=bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0), n=len(v))


def _paired_delta(rows, field, n_hi, n_lo):
    hi = _subject_n(rows, field, n_hi); lo = _subject_n(rows, field, n_lo)
    common = set(hi) & set(lo)
    return _boot({s: hi[s] - lo[s] for s in common})


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/h2cmi/wave0_batchsweep.report.json")
    args = ap.parse_args()
    rows = _load()
    subj = sorted({int(r["target_subject"]) for r in rows})
    # QC sentinels
    mm = defaultdict(set); bu = defaultdict(set); eh = defaultdict(set)
    for r in rows:
        k = (r["target_subject"], r["seed"])
        mm[k].add(round(float(r["metric_mismatch"]), 6)); bu[k].add(round(float(r["B_unif"]), 6)); eh[k].add(r["eval_hash"])
    qc = dict(n_subjects=len(subj),
              metric_mismatch_invariant=all(len(v) == 1 for v in mm.values()),
              B_unif_invariant=all(len(v) == 1 for v in bu.values()),
              eval_hash_invariant=all(len(v) == 1 for v in eh.values()),
              max_abs_residual=max(abs(float(r["residual"])) for r in rows) if rows else 0.0,
              prior_floor=rows[0].get("prior_floor") if rows else None)
    by_n = {}
    for n in NS:
        by_n[str(n)] = {f: _boot(_subject_n(rows, f, n)) for f in FIELDS}
        by_n[str(n)]["metric_mismatch"] = _boot(_subject_n(rows, "metric_mismatch", n))
    endpoint = dict(
        Delta_256_16_P_J=_paired_delta(rows, "P_J", 256, 16),
        Delta_256_16_prior_estimate_deviation=_paired_delta(rows, "prior_estimate_deviation", 256, 16),
        Delta_256_16_TV_piJ_rhoA_full=_paired_delta(rows, "TV_piJ_rhoA_full", 256, 16),
        Delta_256_16_metric_mismatch=_paired_delta(rows, "metric_mismatch", 256, 16))
    rep = dict(marker="WAVE0_BATCHSWEEP", qc=qc, by_n=by_n, primary_endpoint_contrast=endpoint)
    json.dump(rep, open(args.out, "w"), indent=2, default=str)
    print(f"[W0.4] subjects={len(subj)} | QC: mm_inv={qc['metric_mismatch_invariant']} Bunif_inv={qc['B_unif_invariant']} evalhash_inv={qc['eval_hash_invariant']} resid={qc['max_abs_residual']:.1e}")
    print(f"  {'n':>4} {'P_J':>18} {'mm':>10} {'transfer':>10} {'deviation':>12} {'TV(piJ,rhoA_full)':>18} {'min_piJ':>9} {'H(piJ)':>7} {'miss':>5}")
    for n in NS:
        b = by_n[str(n)]
        print(f"  {n:>4} {b['P_J']['mean']:>+9.4f}{str(b['P_J']['ci']):>9} {b['metric_mismatch']['mean']:>+10.4f} {b['transfer']['mean']:>+10.4f} {b['prior_estimate_deviation']['mean']:>+12.4f} {b['TV_piJ_rhoA_full']['mean']:>18.4f} {b['min_piJ']['mean']:>9.4f} {b['H_piJ']['mean']:>7.3f} {b['missing_classes']['mean']:>5.2f}")
    e = endpoint["Delta_256_16_P_J"]; d = endpoint["Delta_256_16_prior_estimate_deviation"]; t = endpoint["Delta_256_16_TV_piJ_rhoA_full"]
    print(f"  ENDPOINT Delta_256-16: P_J={e['mean']:+.4f} {e['ci']} {'SIG' if e.get('excludes_0') else 'NS'} | deviation={d['mean']:+.4f} {d['ci']} | TV_full={t['mean']:+.4f} {t['ci']}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
