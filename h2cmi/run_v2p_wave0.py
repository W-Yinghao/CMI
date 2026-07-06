"""WAVE 0 / W0.2 — fixed-reservoir prevalence UTILITY curves (movement -> harm/benefit).

Extends the P0 V2P_WEIGHTED design to a 9-point q-grid {0.1..0.9} (anchor 0.5) on the SAME fixed
reservoir (same trials reweighted; trial identity + temporal position invariant). Per (unit, seed, est, q)
records, under a FIXED evaluation set:
  displacement from q=0.5: eval-embedding, translation, log-scale;
  utility (pi_dec = Unif): BA, ordinary accuracy, macro-F1, NLL, ECE, per-class recall;
  matched-prevalence ordinary accuracy under pi_dec in {Unif, pi_J, oracle q(diagnostic)}.
Addressing is by real (pair, subject) -- no bench index. Old V2P + W0.1 artifacts untouched.

  python -m h2cmi.run_v2p_wave0 --pairs BNCI2014_001:0>1 --seeds 0,1,2 \
      --out results/h2cmi/wave0_v2p/w0p2_BNCI2014_001_0_1.jsonl
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from h2cmi.eval.harness import _embed, _predict_generative, _predict_transform
from h2cmi.eval.p0_eval import _record
from h2cmi.tta.weighted_tta import fit_weighted_pooled, fit_weighted_em, effective_weights, canonical_weights
from h2cmi.p0_source import get_source_p0, ProvenanceError
from h2cmi.run_w2_wave0 import _gpu_manifest
from h2cmi.run_v2 import build_cfg, PAIRS_B
from h2cmi.data.real_eeg import load_dataset, contiguous_split
from h2cmi.data.real_metadata import MOABB_CLASS
from h2cmi.grid_io import require_clean_git, source_code_signature, append_row, sha256_file, stable_hash_int

K = 2
QGRID = [round(0.1 * i, 2) for i in range(1, 10)]          # 0.1 .. 0.9
ANCHOR = "q0.50"
EST = ["pooled", "fixed_reference_oneshot", "fixed_iterative", "joint", "oracle_label_conditional"]
OUT_DIR = "results/h2cmi/wave0_v2p"


def _fit(est, density, Ua, w, pi_star, cfg, device, oracle_y, pooled_ref, ts):
    if est == "pooled":
        return fit_weighted_pooled(Ua, w, pooled_ref, device), np.asarray(pi_star)
    kind = {"fixed_reference_oneshot": "oneshot", "fixed_iterative": "iterative",
            "joint": "joint", "oracle_label_conditional": "oracle"}[est]
    T, piT = fit_weighted_em(density, Ua, w, pi_star, cfg.tta, K, device, kind,
                             oracle_labels=(oracle_y if kind == "oracle" else None), tta_seed=ts)
    return T, np.asarray(piT.cpu().numpy() if torch.is_tensor(piT) else piT)


def _recalls(ye, preds):
    return [float((preds[ye == c] == c).mean()) if (ye == c).sum() else float("nan") for c in (0, 1)]


def _ord_acc(ye, preds, q):
    r0, r1 = _recalls(ye, preds)
    return float(q * r0 + (1 - q) * r1)


def _pair_units(pairs, seeds, args, code_sig, commit, only_subj=None):
    for ds, s_src, s_tgt in pairs:
        # load ONLY the requested subjects (avoids the 3x-redundant full 54-subject Lee load per chunk)
        sub_list = sorted(only_subj) if only_subj else MOABB_CLASS(ds)().subject_list
        ep = load_dataset(ds, sub_list)
        for subj in np.unique(ep.subject):
            m_src = (ep.subject == subj) & (ep.session == s_src)
            cfg0 = build_cfg(ep.X.shape[1], args.epochs, args.device, seed=0)
            if m_src.sum() < cfg0.tta.min_target * 2:
                continue
            ai, ei = contiguous_split(ep, subj, s_tgt)
            if len(ai) < cfg0.tta.min_target or len(ei) < 8:
                continue
            Xa, ya = ep.X[ai], ep.y[ai]; Xe, ye = ep.X[ei], ep.y[ei]
            if (ya == 0).sum() < 4 or (ya == 1).sum() < 4:
                continue
            yield ds, s_src, s_tgt, subj, ai, ei, Xa, ya, Xe, ye, ep, m_src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--seed0-root", default="results/h2cmi/v2_bundles")
    ap.add_argument("--new-root", default="results/h2cmi/p0_v2pw_bundles")   # REUSE frozen V2P bundles
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", default="", help="optional comma-separated REAL subject ids to restrict to")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    only_subj = set(int(s) for s in args.subjects.split(",") if s != "") if args.subjects else None
    os.makedirs(OUT_DIR, exist_ok=True)
    commit = require_clean_git(allow_dirty=args.allow_dirty,
                               ignore_prefixes=["results/h2cmi", args.new_root, args.seed0_root])
    code_sig = source_code_signature()
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    pairs = []
    for p in (args.pairs.split(",") if args.pairs else []):
        d, ss = p.split(":"); a, b = ss.split(">"); pairs.append((d, int(a), int(b)))
    pairs = pairs or PAIRS_B
    # append-only + skip-if-done: (subject,seed) already carrying a __disp__ row is never recomputed
    done = set()
    if os.path.exists(args.out):
        import json as _json
        for l in open(args.out):
            if l.strip():
                r = _json.loads(l)
                if r.get("ratio") == "__disp__" and r.get("estimator") == EST[-1]:
                    done.add((int(r["subject"]), int(r["seed"])))
    uni = np.full(K, 1.0 / K)
    for ds, s_src, s_tgt, subj, ai, ei, Xa, ya, Xe, ye, ep, m_src in _pair_units(pairs, seeds, args, code_sig, commit, only_subj):
        if only_subj is not None and int(subj) not in only_subj:
            continue
        for seed in seeds:
            if (int(subj), int(seed)) in done:
                print(f"[V2PW0] subj {int(subj)} seed {seed} already recorded -> skip", flush=True); continue
            # NB: no use_deterministic_algorithms here -- W0.2 is aggregate utility curves (seed-averaged +
            # bootstrapped), like the non-deterministic terminal V2P; forcing it crashed on the frozen
            # encoder's adaptive_pool BACKWARD (reached only because embeddings weren't detached; now fixed).
            cfg = build_cfg(ep.X.shape[1], args.epochs, args.device, seed=seed)
            tag = f"B:{ds}:s{int(subj)}:sess{s_src}"
            try:
                model, pooled_ref, R_src, pi_star, val = get_source_p0(
                    args.seed0_root, args.new_root, tag, cfg, code_sig, K,
                    lambda ms=m_src: (ep.X[ms], ep.y[ms], ep.subject[ms]))
            except ProvenanceError as pe:
                append_row(args.out, dict(panel="V2PW0", pair=f"{ds}:{s_src}>{s_tgt}", subject=int(subj),
                                          seed=int(seed), provenance_fail=str(pe))); print(f"PROV FAIL: {pe}"); continue
            Ua = _embed(model, Xa, args.device).detach(); Ue = _embed(model, Xe, args.device).detach()  # frozen encoder: no backward into it
            density = model.head.density
            base = dict(panel="V2PW0", commit=commit, code_sig=code_sig, pair=f"{ds}:{s_src}>{s_tgt}",
                        dataset=ds, subject=int(subj), src_sess=s_src, tgt_sess=s_tgt, seed=int(seed),
                        seed0_validated=bool(val), n_adapt=int(len(ai)), n_eval=int(len(ei)), gpu=_gpu_manifest(),
                        adapt_ids_hash=stable_hash_int(*[int(i) for i in ai]),
                        eval_ids_hash=stable_hash_int(*[int(i) for i in ei]))
            store = {e: {} for e in EST}
            for q in QGRID:
                rname = f"q{q:.2f}"
                w = effective_weights(ya, q)
                cmass = [float(canonical_weights(ya, q)[ya == c].sum()) for c in (0, 1)]
                ts = stable_hash_int("V2PW0", ds, int(subj), int(seed), rname)
                for est in EST:
                    T, piJ = _fit(est, density, Ua, w, pi_star, cfg, args.device, ya, pooled_ref, ts)
                    a = np.asarray(T.a.detach().cpu().numpy(), float); b = np.asarray(T.b.detach().cpu().numpy(), float)
                    p_unif = _predict_transform(model, Ue, T, uni)
                    rec = _record(p_unif, ye, K, T)
                    r0, r1 = _recalls(ye, p_unif)
                    # matched-prevalence ordinary accuracy under three decision priors
                    ord_unif = _ord_acc(ye, p_unif, q)
                    ord_piJ = _ord_acc(ye, _predict_transform(model, Ue, T, piJ), q)
                    ord_oracleq = _ord_acc(ye, _predict_transform(model, Ue, T, np.array([q, 1 - q])), q)
                    row = dict(base, estimator=est, ratio=rname, q=q, class0_mass=cmass[0], class1_mass=cmass[1],
                               weight_sum=float(w.sum()), pi_J=[float(x) for x in piJ],
                               recall_class0=r0, recall_class1=r1,
                               ord_acc_matched_unif=ord_unif, ord_acc_matched_piJ=ord_piJ,
                               ord_acc_matched_oracleq=ord_oracleq, **rec)
                    append_row(args.out, row)
                    store[est][rname] = (a, b, T.apply(Ue.detach()).detach().cpu().numpy())
            # displacement from q=0.5 for every q (per estimator)
            for est in EST:
                a0, b0, z0 = store[est][ANCHOR]
                for q in QGRID:
                    rname = f"q{q:.2f}"
                    a, b, z = store[est][rname]
                    append_row(args.out, dict(base, estimator=est, ratio="__disp__", q=q,
                        logscale_disp=float(np.linalg.norm(a - a0)),
                        translation_disp=float(np.linalg.norm(b - b0)),
                        embed_disp=float(np.linalg.norm(z - z0, axis=1).mean())))
        print(f"[V2PW0 {ds}:{s_src}>{s_tgt}] subj={subj} done", flush=True)
    if os.path.exists(args.out):
        print(f"[V2PW0] -> {args.out} sha={sha256_file(args.out)[:12]}", flush=True)


if __name__ == "__main__":
    main()
