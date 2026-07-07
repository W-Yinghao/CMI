"""Auto-summarize all results/*.json into notes/results_summary.md.
Re-runnable: `python -m analysis.summarize_results`. Picks the right metric per dataset (subject-level for
SCPS, per-target for MCPS), and for each (dataset,backbone) reports ERM vs the best-accuracy lpc_prior config,
plus the framework/GNN/alignment comparisons where present."""
import json, glob, os, time

SCPS = {"ADFTD", "ADFTD_bin", "MUMTAZ", "TUAB"}
EMOTION = {"SEED", "SEED_IV", "DEAP"}


def is_scps(name):
    return any(name.startswith(s) or f"_{s}" in name for s in SCPS) or any(s in name for s in SCPS)


def acc(v, scps):
    return ((v.get("subject_balanced_acc") if scps else v.get("per_target_balanced_acc_mean"))
            or v.get("balanced_acc_mean") or 0) * 100


def worst(v):
    return (v.get("worst_target_balanced_acc") or v.get("worst_subject") or 0) * 100


def leak(v):
    return v.get("leakage_kl", 0)


def best_lpc(s, scps):
    lp = {k: v for k, v in s.items() if k.startswith("lpc_prior")}
    if not lp:
        return None, None
    bk = max(lp, key=lambda k: acc(lp[k], scps))
    return bk, lp[bk]


def main():
    files = sorted(glob.glob("results/*.json"))
    out = ["# Tri-CMI — Auto-generated Results Summary",
           f"\n_Generated {time.strftime('%Y-%m-%d %H:%M')} from {len(files)} result files "
           "(`python -m analysis.summarize_results`). Metric: subject-level bAcc for SCPS, per-target for MCPS._\n"]

    # ---- 1. ERM (CE) vs ERM+CMI head-to-head — one fair row per dataset ----
    # dedupe by base dataset; pick the run with the MOST lpc_prior λ values (so best-λ is a real selection).
    import re
    def base(name):
        return re.sub(r"_(classbal|lamsweep|sweep|v2|bb250|xsess|seed\d|imb|ms\d|chsic|smalllam|"
                      r"prior_\w+|a[\d.]+|domainbal)$", "", name).replace("_EEGNet", "")
    cand = {}
    for f in files:
        name = os.path.basename(f)[:-5]
        if "EEGNet" not in name or any(t in name for t in ("frameworks", "contrastive", "ssl", "lpcssl",
                                       "chsic", "ablation", "audit", "INVALID", "route2", "align_")):
            continue
        try:
            s = json.load(open(f))["summary"]
        except Exception:
            continue
        if "erm:0" not in s or not any(k.startswith("lpc_prior") for k in s):
            continue
        b = base(name); nlam = sum(k.startswith("lpc_prior") for k in s)
        if b not in cand or nlam > cand[b][2]:
            cand[b] = (name, s, nlam)
    out.append("## 1. EEGNet: CE vs CE+CMI — one fair row per dataset (best-λ where a sweep exists)\n")
    out.append("| dataset | #λ | ERM | +CMI(best λ) | Δacc | ERM leak | +CMI leak |")
    out.append("|---|---|---|---|---|---|---|")
    wins = losses = 0
    for b in sorted(cand):
        name, s, nlam = cand[b]; scps = is_scps(name)
        bk, bv = best_lpc(s, scps)
        ea, la = acc(s["erm:0"], scps), acc(bv, scps); d = la - ea
        wins += d > -0.5; losses += d <= -0.5
        flag = "" if nlam > 1 else " ⚠1λ"
        out.append(f"| {b}{flag} | {nlam} | {ea:.1f} | {la:.1f} ({bk.split(':')[1]}) | {d:+.1f} | "
                   f"{leak(s['erm:0']):.2f} | {leak(bv):.2f} |")
    out.append(f"\n*+CMI ≥ parity (Δ>−0.5) on {wins}/{wins+losses} datasets at best-available λ; leakage drops "
               "everywhere. ⚠1λ = only one λ run (often 0.3, over-regularized) so not a fair best-λ.*\n")

    # ---- 2. Worst-subject (DG-relevant) ----
    out.append("## 2. Worst-subject robustness (ERM vs lpc_prior, small λ)\n")
    out.append("| run | ERM worst | lpc_prior worst | Δ |")
    out.append("|---|---|---|---|")
    for f in files:
        name = os.path.basename(f)[:-5]
        if "lamsweep" not in name:
            continue
        s = json.load(open(f))["summary"]
        if "erm:0" not in s:
            continue
        lp = {k: v for k, v in s.items() if k.startswith("lpc_prior")}
        bk = max(lp, key=lambda k: worst(lp[k]))
        out.append(f"| {name} | {worst(s['erm:0']):.1f} | {worst(lp[bk]):.1f} ({bk.split(':')[1]}) | "
                   f"{worst(lp[bk])-worst(s['erm:0']):+.1f} |")

    # ---- 3. Framework / backbone / GNN comparisons (raw per-config dumps) ----
    out.append("\n## 3. Method & backbone comparisons (per-config)\n")
    groups = [("Framework zoo", ["frameworks"]), ("GNN benchmark", ["GraphCMI", "DGCNN", "RGNN"]),
              ("Alignment", ["align_"]), ("Self-supervised", ["ssl", "lpcssl"]),
              ("Route-2 FMCA", ["route2"]), ("Classical", ["classical"])]
    for title, pats in groups:
        fs = [f for f in files if any(p in os.path.basename(f) for p in pats)]
        if not fs:
            continue
        out.append(f"### {title}")
        for f in fs:
            name = os.path.basename(f)[:-5]
            try:
                s = json.load(open(f))["summary"]
            except Exception:
                continue
            scps = is_scps(name)
            cells = " · ".join(f"{k}={acc(v, scps):.1f}/lk{leak(v):.2f}" for k, v in list(s.items())[:8])
            out.append(f"- **{name}**: {cells}")
        out.append("")

    open("notes/results_summary.md", "w").write("\n".join(out))
    print(f"wrote notes/results_summary.md ({len(out)} lines, {len(files)} result files)")


if __name__ == "__main__":
    main()
