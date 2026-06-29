"""REVIEW_P0 provenance audit (finalizer #2B). Audits every REUSED seed-0 bundle (W1 from w1_bundles,
V2P from v2_bundles) for STRICT provenance: code_sig / data_hash / epochs / n_chans non-null, checkpoint
SHA-256 present, source row count recorded. Flags any bundle that the (formerly permissive) validator
would have accepted with a missing field. W2 used pre-registered NO-reuse (all seeds trained into
p0_w2_bundles) -- recorded as freshly-trained, not audited for reuse."""
from __future__ import annotations

import glob
import hashlib
import json
import os


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def audit_root(root, tag_prefix, label):
    recs, flags = [], []
    for js in sorted(glob.glob(os.path.join(root, "*.json"))):
        try:
            m = json.load(open(js))
        except Exception:
            continue
        if not str(m.get("tag", "")).startswith(tag_prefix):
            continue
        sig = m.get("sig"); pt = os.path.join(root, f"{sig}.pt") if sig else ""
        rec = dict(tag=m.get("tag"), sig=sig, code_sig=m.get("code_sig"), data_hash=m.get("data_hash"),
                   epochs=m.get("epochs"), n_chans=m.get("n_chans"), n_train=m.get("n_train"),
                   ckpt_sha256=(_sha(pt)[:16] if os.path.exists(pt) else None))
        for f in ("code_sig", "data_hash", "epochs", "n_chans", "n_train"):
            if rec[f] is None:
                flags.append(f"{label} {rec['tag']}: {f} is None")
        if rec["ckpt_sha256"] is None:
            flags.append(f"{label} {rec['tag']}: checkpoint .pt missing")
        recs.append(rec)
    return recs, flags


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h2cmi/review_p0_provenance.json")
    args = ap.parse_args()
    w1, f1 = audit_root("results/h2cmi/w1_bundles", "W1:", "W1-seed0")
    v2, f2 = audit_root("results/h2cmi/v2_bundles", "B:", "V2P-seed0")
    code_sigs = set(r["code_sig"] for r in (w1 + v2) if r["code_sig"])
    rep = dict(marker="REVIEW_P0_PROVENANCE_AUDIT",
               note="W1 & V2P seed-0 bundles REUSED after validation; seeds 1/2 trained; W2 all seeds "
                    "trained (pre-registered no-reuse). Strict: every reused bundle has non-null code_sig/"
                    "data_hash/epochs/n_chans/n_train + present checkpoint.",
               n_reused_W1=len(w1), n_reused_V2P=len(v2), reused_code_sigs=sorted(code_sigs),
               flags=f1 + f2, strict_ok=(len(f1 + f2) == 0),
               W1_seed0=w1, V2P_seed0=v2)
    json.dump(rep, open(args.out, "w"), indent=2)
    print(f"reused seed-0 bundles: W1={len(w1)} V2P={len(v2)} | code_sigs={sorted(code_sigs)}")
    print(f"strict provenance OK (no None fields, all checkpoints present): {rep['strict_ok']}")
    if rep["flags"]:
        print("FLAGS:"); [print("  " + x) for x in rep["flags"][:20]]
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
