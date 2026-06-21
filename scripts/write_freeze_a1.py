"""Write the Freeze A1 manifest — ONLY after the deterministic decomposition's layered pred-hash equivalence
passes. Self-validating (re-checks the gate before writing); atomic; refuses to overwrite. Freezes the PROCESS
(config), not the headline magnitude. Freezes BOTH branches (CITA+LPC, CITA-no-LPC); P1.5 decides which enters
Freeze B. Per notes/FREEZE_PROTOCOL.md §3.
Exit codes: 0 written | 1 gate failed (not eligible, nothing written) | 2 manifest already exists (refused).
"""
import os, sys, json, glob, hashlib, subprocess

OUT = "results/freeze_a1/manifest.json"

# ---- the FROZEN process (mirror of FREEZE_PROTOCOL.md §3; the thing pinned by the hash) ----
FROZEN = dict(
    deterministic=True, select="nested", select_pipeline="full", domain="cohort", transduct="all",
    leakage_split="grouped", configs=["erm:0", "lpc_prior:0.1", "lpc_prior:0.3"],
    branches=["CITA+LPC", "CITA-no-LPC"],
    aligner=dict(mode="matched_coral", tmap="wc", rho=0.2, eps=1e-3, gate="reliability", kappa=8.0,
                 alpha=1.0, em_iters=3),
    selector=dict(rule="argmin leakage s.t. valBAcc>=max-eps", eps=0.02),
    eval=dict(metric="per_target_balanced_acc + ts_matched_coral", uncertainty_unit="cohort"),
)

def gate_pred_hash_equivalence():
    """Re-validate the layered gate: encoder vs full per-config prediction-hash must match for EVERY available
    (cohort,seed,config). Returns (ok, summary)."""
    def load(pat):
        d = {}
        for f in sorted(glob.glob(pat)):
            s = f.split("_s")[-1].split(".")[0]; o = json.load(open(f)); fo = o.get("folds", {})
            for lbl in ("erm:0", "lpc_prior:0.1", "lpc_prior:0.3"):
                for r in fo.get(lbl, []):
                    d[(r["held_out"], s, lbl)] = r.get("pred_hash")
        return d
    summary = {}
    total = mism = 0
    for cond in ("PD", "SCZ"):
        enc = load(f"results/r12det_dualpc2/r12det_encoder_{cond}_s*.json")
        full = load(f"results/r12det_dualpc2/r12det_full_{cond}_s*.json")
        keys = set(enc) & set(full)
        m = sum(1 for k in keys if enc[k] is None or enc[k] != full[k])
        summary[cond] = dict(n=len(keys), mismatches=m)
        total += len(keys); mism += m
        if not keys:
            return False, {"error": f"{cond}: no r12det data"}
    return (mism == 0 and total > 0), dict(per_cond=summary, total=total, mismatches=mism)

def main():
    ok, summ = gate_pred_hash_equivalence()
    if not ok:
        print(f"FREEZE-A1 GATE FAILED — pred-hash equivalence not satisfied: {summ}; NOTHING written."); sys.exit(1)
    if os.path.exists(OUT):
        print(f"Freeze A1 manifest already exists ({OUT}) — refusing to overwrite (immutable)."); sys.exit(2)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "nogit"
    proto_hash = hashlib.sha256(open("notes/FREEZE_PROTOCOL.md", "rb").read()).hexdigest()[:16] if os.path.exists("notes/FREEZE_PROTOCOL.md") else None
    h = hashlib.sha256(json.dumps(dict(FROZEN=FROZEN, commit=commit, proto=proto_hash), sort_keys=True).encode()).hexdigest()
    manifest = dict(status="FROZEN", freeze="A1", hash=h, code_commit=commit, protocol_hash=proto_hash,
                    frozen=FROZEN, evidence=dict(decomposition_gate="pred-hash equivalence PASSED", **summ),
                    note="Process frozen (magnitude accepted as-is). Both branches frozen; P1.5 decides Freeze B. "
                         "feat_dump_v2 is NOT co-source (pre-determinism, encoder-selection) -> P1.5 domain-leakage "
                         "needs a metadata-enriched frozen redump (REQUIRES_FROZEN_REDUMP).")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    json.dump(manifest, open(tmp, "w"), indent=2)
    os.rename(tmp, OUT)                                    # atomic publish
    print(f"FREEZE A1 WRITTEN: {OUT}  hash={h[:16]}  commit={commit}")
    print(f"  gate: {summ['total']} configs, {summ['mismatches']} pred-hash mismatches (0 required)")

if __name__ == "__main__":
    main()
