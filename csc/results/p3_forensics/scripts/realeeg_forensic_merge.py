"""P3.0b/c merge: gather the 18 forensic shards, verify each shard's sha256 + provenance (n=100, 0 worker
errors, consistent cache/engine/b3 hashes), canonical-sort, and fail-closed check for exactly 1800 unique
records (no missing / no duplicate / no worker error). Writes the merged development-forensic JSON + sha256.
DIAGNOSTIC ONLY -- must NOT be merged with the csc-realeeg-v2 scientific result."""
import os, sys, json, hashlib, glob

OUTDIR = "/home/infres/yinwang/realeeg_feas/p3_forensic"
BASES = [50000000, 60000000, 70000000]
CONDS = ["NULL_cov", "NULL_label", "NULL_cov_plus_label", "random_label_control", "POS_concept", "POS_concept_plus_cov"]
EXPECT = len(BASES) * len(CONDS) * 100  # 1800

def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

def fail(msg): print(f"INFRA-FAIL: {msg}"); sys.exit(2)

records, provs, hashset = [], [], {"cache": set(), "engine": set(), "b3": set()}
for base in BASES:
    for cond in CONDS:
        stem = f"{OUTDIR}/shard_{base}_{cond}.jsonl"
        if not os.path.exists(stem): fail(f"missing shard {base}:{cond}")
        if not os.path.exists(stem + ".prov.json"): fail(f"missing prov for {base}:{cond}")
        if not os.path.exists(stem + ".sha256"): fail(f"missing sha256 for {base}:{cond}")
        want = open(stem + ".sha256").read().split()[0]
        if sha(stem) != want: fail(f"sha256 mismatch for {base}:{cond}")
        prov = json.load(open(stem + ".prov.json")); provs.append(prov)
        if prov.get("n_worker_errors", 1) != 0: fail(f"shard {base}:{cond} had {prov['n_worker_errors']} worker errors")
        if prov.get("n_records") != 100: fail(f"shard {base}:{cond} has {prov.get('n_records')} != 100 records")
        hashset["cache"].add(prov.get("cache_sha256")); hashset["engine"].add(prov.get("engine_sha256"))
        hashset["b3"].add(prov.get("b3_manifest_sha256"))
        recs = [json.loads(l) for l in open(stem) if l.strip()]
        errs = [r for r in recs if "__worker_error__" in r]
        if errs: fail(f"shard {base}:{cond} contains {len(errs)} worker-error records")
        records.extend(recs)

for k, s in hashset.items():
    if len(s) != 1: fail(f"inconsistent {k}_sha256 across shards: {s}")
ids = [r["task_id"] for r in records]
if len(records) != EXPECT: fail(f"{len(records)} records != {EXPECT}")
if len(set(ids)) != EXPECT: fail(f"{len(set(ids))} unique task_ids != {EXPECT} (dup/missing)")
records.sort(key=lambda r: (r["seed_block"], r["condition_index"], r["cohort"]))  # canonical

merged = dict(diagnostic_only=True, not_confirmatory=True, not_merged_with_v2=True,
              n_records=len(records), n_seed_blocks=len(BASES), seed_blocks=BASES, conditions=CONDS,
              cache_sha256=list(hashset["cache"])[0], engine_sha256=list(hashset["engine"])[0],
              b3_manifest_sha256=list(hashset["b3"])[0], per_cohort=records)
out = f"{OUTDIR}/p3_internal_forensic_merged.json"
json.dump(merged, open(out, "w"), indent=1, default=str)
with open(out + ".sha256", "w") as f:
    f.write(sha(out) + "  " + os.path.basename(out) + "\n")
# quick sanity: per (seed_block,condition) false-confirm counts
print(f"MERGE OK: {len(records)} records, all hashes consistent, canonical-sorted -> {out}")
print(f"  sha256 {sha(out)}")
print("\n  false-confirm counts by seed block x condition (NO_CONCEPT conditions):")
for base in BASES:
    row = []
    for cond in CONDS:
        fc = sum(1 for r in records if r["seed_block"] == base and r["condition"] == cond and r.get("false_confirm"))
        tc = sum(1 for r in records if r["seed_block"] == base and r["condition"] == cond and r.get("true_confirm"))
        row.append(f"{cond}={fc}FC/{tc}TC")
    print(f"    base {base}: " + "  ".join(row))
