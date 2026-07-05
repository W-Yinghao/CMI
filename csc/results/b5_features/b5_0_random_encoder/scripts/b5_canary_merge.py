"""Merge + analyze the B5.0 6-condition canary (development-only). Reads the 6 per-condition JSONL shards, fails
CLOSED on any worker error / missing shard / duplicate, and reports the canary question directly:
  (1) does the SM16 NULL_cov false-confirmation PERSIST under the deep Z_deep feature family? (type-I on the nulls)
  (2) is there any POS signal (true-confirm rate on POS_concept / POS_concept_plus_cov)?
  (3) null-calibration signature (null_sd_T, T_z, fixed_margin_p floor) -- is the plug-in under-dispersion the
      same mechanism the SM16 v2 / P3 forensics exposed, now under deep features?
NOT confirmatory, NOT deployable, NO tag. Prints a table + writes b5_canary_tables.json + merged sha256."""
import os, sys, json, glob, hashlib
import numpy as np

CDIR = "/home/infres/yinwang/realeeg_feas/b5_features/b5_0_random_encoder/canary"
CONDS = ["NULL_cov", "NULL_cov_plus_label", "NULL_label", "random_label_control", "POS_concept", "POS_concept_plus_cov"]
NOCONCEPT = {"NULL_cov", "NULL_label", "NULL_cov_plus_label", "random_label_control"}
DECIDED = {"CONCEPT_CONFIRMED", "NO_CONCEPT_EVIDENCE_AFTER_PAIR_AUDIT"}


def _read(path):
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def med(xs):
    xs = [x for x in xs if x is not None and x == x]
    return float(np.median(xs)) if xs else float("nan")


def main():
    all_recs, per = [], {}
    STARTS = (0, 25)   # exact cohort-range halves; avoids the NULL_cov / NULL_cov_plus_label glob-prefix collision
    for c in CONDS:
        paths = [os.path.join(CDIR, f"b5_canary_{c}_{s}.jsonl") for s in STARTS]
        for p in paths:
            if not os.path.exists(p):
                print(f"FAIL-CLOSED: missing shard {p}"); sys.exit(2)
        recs = []
        for p in paths:
            recs.extend(_read(p))
        errs = [r for r in recs if "__worker_error__" in r]
        if errs:
            print(f"FAIL-CLOSED: {len(errs)} worker error(s) in {c}; first={errs[0]}"); sys.exit(2)
        ids = [r["task_id"] for r in recs]
        if len(ids) != len(set(ids)):
            print(f"FAIL-CLOSED: duplicate task_id in {c} ({len(ids)} recs, {len(set(ids))} unique)"); sys.exit(2)
        if len(recs) != 50:
            print(f"FAIL-CLOSED: condition {c} has {len(recs)} cohorts, expected 50 (shards: {[os.path.basename(x) for x in paths]})"); sys.exit(2)
        per[c] = recs; all_recs.extend(recs)

    rows = {}
    print(f"\n{'condition':24s} {'n':>4} {'decided':>8} {'confirm':>8} {'rate':>7} "
          f"{'null_sd':>10} {'T_z_med':>8} {'ffp_floor':>10} {'ovlp_auc':>9}")
    print("-" * 100)
    for c in CONDS:
        recs = per[c]
        gt_noconcept = c in NOCONCEPT
        states = [str(r.get("b3_state")) for r in recs]
        decided = [r for r, s in zip(recs, states) if s in DECIDED]
        confirm = [r for r in recs if r.get("false_confirm") or r.get("true_confirm")]
        n_conf = len(confirm)
        rate = n_conf / len(recs) if recs else float("nan")
        null_sd = med([r.get("null_sd_T") for r in recs])
        tz = med([r.get("T_z") for r in recs])
        ffp = [r.get("fixed_margin_p") for r in recs if r.get("fixed_margin_p") is not None]
        ffp_floor = float(np.mean([1.0 if (p is not None and p <= 0.005 + 1e-9) else 0.0 for p in ffp])) if ffp else float("nan")
        ovlp = med([r.get("session_auc") for r in recs])
        kind = "FALSE" if gt_noconcept else "TRUE"
        rows[c] = dict(n=len(recs), n_decided=len(decided), n_confirm=n_conf, confirm_kind=kind,
                       confirm_rate=rate, null_sd_T_med=null_sd, T_z_med=tz,
                       fixed_margin_p_floor_frac=ffp_floor, session_auc_med=ovlp,
                       ground_truth_noconcept=gt_noconcept)
        print(f"{c:24s} {len(recs):>4} {len(decided):>8} {n_conf:>8} {rate:>7.3f} "
              f"{null_sd:>10.6f} {tz:>8.2f} {ffp_floor:>10.2f} {ovlp:>9.3f}  [{kind}-confirm GT]")

    # --- canary verdict (development screen) ---
    null_ff = {c: rows[c]["n_confirm"] for c in CONDS if c in NOCONCEPT}
    pos_tc = {c: rows[c]["n_confirm"] for c in CONDS if c not in NOCONCEPT}
    persists = any(v > 0 for c, v in null_ff.items() if c.startswith("NULL_cov"))
    pos_signal = any(v > 0 for v in pos_tc.values())
    print("\n=== B5.0 canary readout (development-only) ===")
    print(f"  NULL false-confirm counts (type-I): {null_ff}")
    print(f"  POS true-confirm counts (power):    {pos_tc}")
    print(f"  NULL_cov FAIL persists under Z_deep? {'YES' if persists else 'NO'}")
    print(f"  Any POS signal under Z_deep?         {'YES' if pos_signal else 'NO'}")

    tables = dict(scope="B5.0 canary development-only; Z_deep frozen-random feature family; NOT confirmatory; NO tag",
                  base_seed=30_000_000, n_per_condition={c: rows[c]["n"] for c in CONDS},
                  per_condition=rows, null_false_confirm=null_ff, pos_true_confirm=pos_tc,
                  null_cov_fail_persists=bool(persists), any_pos_signal=bool(pos_signal),
                  b5_cache_sha256=(all_recs[0].get("b5_cache_sha256") if all_recs else None))
    outp = os.path.join(CDIR, "b5_canary_tables.json")
    json.dump(tables, open(outp, "w"), indent=1, default=str)
    # merged sha256 over canonically-sorted task_ids for provenance
    canon = sorted(all_recs, key=lambda r: r.get("task_id", ""))
    mh = hashlib.sha256(json.dumps([r.get("task_id") for r in canon], separators=(",", ":")).encode()).hexdigest()
    print(f"\n  merged {len(all_recs)} records; task-id sha256 {mh[:12]}; tables -> {outp}")
    return tables


if __name__ == "__main__":
    main()
