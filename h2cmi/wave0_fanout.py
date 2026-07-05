"""Canonical, bench-index-FREE fan-out addressing for Wave 0 (regression-gated preflight).

The W0.1 stall was caused by using a bench *index* as an identity (job key + output filename) while the
runner mapped index->real_subject_id -- for non-contiguous benches these diverge. Rule enforced here:
`bench_index` is NEVER an identity. It may appear only as a non-identifying log field. Units are addressed
by (wave, dataset, real_subject_id, protocol[/pair], [q|batch_n]). This module is the single source of
truth for expected-unit manifests, output paths, doneness, and the preflight coverage assertion that
gates every wave that fans out.

  python -m h2cmi.wave0_fanout --wave W0.1 --coverage      # expected-vs-done report before submission
"""
from __future__ import annotations

import json
import os

SLEEP_DATASET = "sleep-edf-cassette"
OUT_DIR_DEFAULT = "results/h2cmi/wave0_w2det"
_IDENTITY_KEYS = ("wave", "dataset", "real_subject_id", "protocol", "pair_id", "q", "batch_n")


def load_bench(cache="results/h2cmi/p0_sleep_cache"):
    return json.load(open(os.path.join(cache, "p0_benchmark.json")))["subject_ids"]


def w2_units(wave, bench, protocols=("primary", "secondary"), q=None, batch_n=None):
    """Expected job-level units for a sleep-W2 wave, addressed by REAL subject id (order-independent)."""
    units = []
    for proto in protocols:
        for sid in bench:
            u = dict(wave=wave, dataset=SLEEP_DATASET, real_subject_id=int(sid), protocol=proto)
            if q is not None:
                u["q"] = q
            if batch_n is not None:
                u["batch_n"] = batch_n
            units.append(u)
    return units


def unit_id(u):
    """Stable canonical id. Contains NO bench index. Order of construction is fixed + explicit."""
    parts = [str(u["wave"]), str(u["dataset"]), f"s{u['real_subject_id']}", str(u["protocol"])]
    if u.get("pair_id") is not None:
        parts.append(f"pair{u['pair_id']}")
    if u.get("q") is not None:
        parts.append(f"q{u['q']}")
    if u.get("batch_n") is not None:
        parts.append(f"n{u['batch_n']}")
    return "|".join(parts)


def output_path(u, out_dir=OUT_DIR_DEFAULT, prefix="p0w2det"):
    """Path is keyed by REAL subject id (matches the runner's own output naming, which was always
    real-id-based); q/batch_n disambiguate W0.2/W0.4 variants. Never contains a bench index."""
    name = f"{prefix}_{u['protocol']}_{u['real_subject_id']}"
    if u.get("q") is not None:
        name += f"_q{u['q']}"
    if u.get("batch_n") is not None:
        name += f"_n{u['batch_n']}"
    return os.path.join(out_dir, name + ".jsonl")


def is_done(u, out_dir=OUT_DIR_DEFAULT, min_decomp=3):
    f = output_path(u, out_dir)
    return os.path.exists(f) and sum(1 for l in open(f) if '"__decomposition__"' in l) >= min_decomp


def preflight_assert(units):
    """Manifest-level invariants (raise on violation) -- run BEFORE any submission."""
    ids = [unit_id(u) for u in units]
    dup = [i for i in set(ids) if ids.count(i) > 1]
    assert not dup, f"duplicate canonical_unit_id: {dup[:5]}"
    for u in units:
        assert "real_subject_id" in u, "unit missing real_subject_id"
        assert "bench_index" not in u, "bench_index must never be a unit identity field"
    subs = [u["real_subject_id"] for u in units]
    # per (protocol,q,batch) group, each real_subject covered exactly once
    from collections import Counter
    grp = Counter((u["protocol"], u.get("q"), u.get("batch_n"), u["real_subject_id"]) for u in units)
    over = [k for k, c in grp.items() if c > 1]
    assert not over, f"real_subject covered >1x within a group: {over[:5]}"
    return dict(n_units=len(units), n_subjects=len(set(subs)), ok=True)


def coverage_report(units, out_dir=OUT_DIR_DEFAULT):
    done = [u for u in units if is_done(u, out_dir)]
    missing = [u for u in units if not is_done(u, out_dir)]
    return dict(n_expected=len(units), n_done=len(done), n_missing=len(missing),
                missing_unit_ids=[unit_id(u) for u in missing],
                missing_addressing=[dict(real_subject_id=u["real_subject_id"], protocol=u["protocol"])
                                    for u in missing])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", default="W0.1")
    ap.add_argument("--cache", default="results/h2cmi/p0_sleep_cache")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--coverage", action="store_true")
    args = ap.parse_args()
    bench = load_bench(args.cache)
    units = w2_units(args.wave, bench)
    pf = preflight_assert(units)
    print(f"preflight OK: {pf['n_units']} units, {pf['n_subjects']} unique subjects, no dup/no bench_index")
    if args.coverage:
        cov = coverage_report(units, args.out_dir)
        print(f"coverage: {cov['n_done']}/{cov['n_expected']} done, {cov['n_missing']} missing")
        if cov["n_missing"]:
            print("  missing:", ", ".join(u for u in cov["missing_unit_ids"][:40]))


if __name__ == "__main__":
    main()
