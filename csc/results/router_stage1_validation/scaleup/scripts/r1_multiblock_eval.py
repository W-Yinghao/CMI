"""R1 scale-up: evaluate the LOCKED tau_R1 on K fresh INDEPENDENT held-out blocks to bound the block-to-block
drift the single-block R1 could not estimate. tau is NOT re-derived (loaded from the lock). Fail-closed on any
worker error / missing shard / wrong count. Reports per-block allow counts + CP95u + the drift across all blocks
(including the original validation block 90e6 for reference). DEVELOPMENT-diagnostic, NO tag, NO retune."""
import os, sys, json, glob, math, hashlib
import numpy as np
try:
    from scipy.stats import beta
    def cp_upper(k, n, a=0.05): return 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
except Exception:
    def cp_upper(k, n, a=0.05): return min(1.0, (k + 1.645 * math.sqrt(k + 1)) / n)

FEAS = "/home/infres/yinwang/realeeg_feas"
LOCK = json.load(open(f"{FEAS}/router_stage1/router_stage1_tau_lock.json"))
TAU = LOCK["tau_R1"]
MBDIR = f"{FEAS}/router_stage1/multiblock"
NEW_BASES = [100_000_000, 110_000_000, 120_000_000]
CONDS = ["NULL_cov", "NULL_cov_plus_label", "POS_concept"]
NULLS = {"NULL_cov", "NULL_cov_plus_label"}
STARTS = (0,)   # consolidated: one 300-cohort file per (block, condition)


def _read(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def T(r):
    t = r.get("observed_T")
    return t if isinstance(t, (int, float)) and t == t else None


def load_block(base):
    out = {}
    for c in CONDS:
        recs = []
        for s in STARTS:
            p = f"{MBDIR}/r1_{base}_{c}_{s}.jsonl"
            if not os.path.exists(p):
                print(f"FAIL-CLOSED: missing {p}"); sys.exit(2)
            pr = json.load(open(p + ".prov.json")) if os.path.exists(p + ".prov.json") else {}
            if pr.get("n_worker_errors", 1) != 0:
                print(f"FAIL-CLOSED: worker errors in {p}"); sys.exit(2)
            rs = _read(p)
            if any("__worker_error__" in r for r in rs):
                print(f"FAIL-CLOSED: worker-error rows in {p}"); sys.exit(2)
            recs.extend(rs)
        if len(recs) != 300:
            print(f"FAIL-CLOSED: {base}:{c} has {len(recs)} != 300"); sys.exit(2)
        out[c] = recs
    return out


def allow_count(recs):
    return sum(1 for r in recs if r.get("b3_state") == "CONCEPT_CONFIRMED" and T(r) is not None and T(r) >= TAU)


def main():
    print(f"=== R1 multi-block drift @ LOCKED tau_R1={TAU} (>=) ===")
    # original validation block (90e6) for reference
    ref = {}
    vmerged = f"{FEAS}/router_stage1/validation/r1_validation_merged.json"
    if os.path.exists(vmerged):
        V = json.load(open(vmerged))["per_cohort"]
        for c in CONDS:
            ref[c] = allow_count([r for r in V if r["condition"] == c])
    blocks = {90_000_000: ref} if ref else {}
    for base in NEW_BASES:
        blk = load_block(base)
        blocks[base] = {c: allow_count(blk[c]) for c in CONDS}

    print(f"\n  {'block':>12} " + " ".join(f"{c:>20}" for c in CONDS))
    for base, row in blocks.items():
        tag = " (orig valid)" if base == 90_000_000 else ""
        print(f"  {base:>12} " + " ".join(f"{row[c]:>20}" for c in CONDS) + tag)

    # drift summary on the primary nulls
    print("\n  === primary-null drift (allow/300, CP95u) ===")
    summary = {}
    for c in CONDS:
        counts = [blocks[b][c] for b in blocks]
        cps = [cp_upper(k, 300) for k in counts]
        kind = "NULL" if c in NULLS else "POS"
        summary[c] = dict(kind=kind, per_block_allow=counts, per_block_cp95u=[round(x, 4) for x in cps],
                          min=min(counts), max=max(counts), mean=round(float(np.mean(counts)), 2),
                          max_cp95u=round(max(cps), 4))
        flag = ""
        if c in NULLS:
            flag = "  <-- SAFE across blocks" if max(counts) <= 7 else "  <== BREACH (>7 cap) in some block"
        print(f"  {c:22} allow={counts} (mean {np.mean(counts):.1f}) max_CP95u={max(cps):.4f}{flag}")

    n_blocks = len(blocks)
    null_max = {c: summary[c]["max"] for c in CONDS if c in NULLS}
    all_safe = all(v <= 7 for v in null_max.values())
    print(f"\n  >>> {n_blocks} blocks; primary-null max allow {null_max}; "
          f"ALL blocks within 7-cap: {'YES' if all_safe else 'NO'}")
    tables = dict(diagnostic_only=True, tau_R1=TAU, tau_source="locked; not re-derived",
                  n_blocks=n_blocks, bases=list(blocks.keys()), per_block=blocks, drift_summary=summary,
                  all_primary_nulls_within_cap=bool(all_safe))
    outp = f"{MBDIR}/r1_multiblock_tables.json"
    json.dump(tables, open(outp, "w"), indent=1, default=str)
    print(f"\n  saved {outp}")


if __name__ == "__main__":
    main()
