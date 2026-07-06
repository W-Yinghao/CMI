#!/usr/bin/env python
"""FSR Step 2A — descriptive TOS erasure summary supporting RQ2 (erasure strength vs target benefit).

CPU-only, read-only. Reads the frozen TOS artifacts from git branch `tos` (@1c65d79) via `git show`
(they are not checked out in this worktree). Produces descriptive tables only — it does NOT mint a
new headline. Decision language is fixed by FSR_03: subject signal is erasable; erasure is not
reliable target control; random-k falsifies non-specific NLL movement; refusal-first is the safe
behavior.

Outputs (results/fsr_phase2/):
    tos_erasure_summary.csv        per (dataset, backbone, eraser): subject removal + target deltas
    tos_randomk_specificity.csv    LEACE vs random_k NLL + subject-removal specificity
    tos_task_safety_flags.csv      task-collapse (INLP) and binary-harm (LEACE/RLACE) flags

    python scripts/fsr/analyze_tos_erasure.py
"""
from __future__ import annotations
import csv, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "fsr_phase2"
REF = "tos"
BASE = "tos_cmi/results/tos_cmi_eeg_frozen"
ERASERS = ["full", "TOS_VD", "LEACE", "RLACE", "INLP", "random_k"]
PRINCIPLED = {"LEACE", "RLACE", "TOS_VD"}

# dataset label -> (deploy summary path, erasure_report paths by backbone [2a only])
DEPLOY = {
    "BNCI2014_001": f"{BASE}/erasure_target_deploy/erasure_target_deploy_summary.json",
    "Lee2019_MI": f"{BASE}/erasure_target_deploy/Lee2019_MI/erasure_target_deploy_summary.json",
    "Cho2017": f"{BASE}/erasure_target_deploy/Cho2017/erasure_target_deploy_summary.json",
    "Schirrmeister2017": f"{BASE}/erasure_target_deploy/Schirrmeister2017/erasure_target_deploy_summary.json",
}
REPORTS_2A = {
    "TSMNet": f"{BASE}/BNCI2014_001_TSMNet_LOSO/erasure_report.json",
    "EEGNet": f"{BASE}/BNCI2014_001_EEGNet_LOSO/erasure_report.json",
}


def git_json(path):
    try:
        out = subprocess.run(["git", "-C", str(REPO), "show", f"{REF}:{path}"],
                             capture_output=True, text=True, check=True).stdout
        return json.loads(out)
    except Exception:  # noqa: BLE001
        return None


def _num(x):
    try:
        return round(float(x), 5)
    except (TypeError, ValueError):
        return ""


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    reports = {bb: git_json(p) for bb, p in REPORTS_2A.items()}  # 2a subject-removal specificity
    summ_rows, spec_rows, safety_rows = [], [], []

    for ds, dp in DEPLOY.items():
        dep = git_json(dp)
        if dep is None:
            summ_rows.append(dict(dataset=ds, backbone="", eraser="", note="deploy summary NOT FOUND on tos"))
            continue
        s = dep.get("summary", dep)
        backbones = sorted({k.split("|")[0] for k in s})
        for bb in backbones:
            full = s.get(f"{bb}|full", {})
            chance = _num(full.get("chance_task"))
            rep = reports.get(bb) if ds == "BNCI2014_001" else None
            agg = rep.get("aggregate", {}) if rep else {}
            for er in ERASERS:
                cell = s.get(f"{bb}|{er}")
                if cell is None:
                    continue
                # subject removal (2a erasure_report; deploy has subj_dec_after for all datasets)
                subj_lin = _num(agg.get(f"subj_{er}_lin")) if agg else ""
                subj_mlp = _num(agg.get(f"subj_{er}_mlp")) if agg else ""
                subj_dec_after = _num(cell.get("subj_dec_after_mean"))
                tgt_bacc = _num(cell.get("tgt_bacc_mean"))
                dtgt_bacc = _num(cell.get("dtgt_bacc"))
                tgt_nll = _num(cell.get("tgt_nll_mean"))
                dtgt_nll = _num(cell.get("dtgt_nll"))
                summ_rows.append(dict(
                    dataset=ds, backbone=bb, eraser=er, chance_task=chance,
                    subj_lin_2a=subj_lin, subj_mlp_2a=subj_mlp, subj_dec_after=subj_dec_after,
                    tgt_bacc=tgt_bacc, dtgt_bacc=dtgt_bacc, tgt_nll=tgt_nll, dtgt_nll=dtgt_nll,
                    worst_subject_tgt_bacc=_num(cell.get("worst_subject_tgt_bacc")),
                    improves_target=cell.get("improves_target", ""),
                    verdict=_verdict(er, dtgt_bacc, tgt_bacc, chance)))

                # task-safety flags
                if er != "full":
                    task_collapse = (isinstance(tgt_bacc, float) and isinstance(chance, float)
                                     and tgt_bacc <= chance + 0.02)
                    binary_harm = (isinstance(chance, float) and chance >= 0.5
                                   and isinstance(dtgt_bacc, float) and dtgt_bacc <= -0.05)
                    safety_rows.append(dict(
                        dataset=ds, backbone=bb, eraser=er, chance_task=chance, tgt_bacc=tgt_bacc,
                        dtgt_bacc=dtgt_bacc, task_collapse_flag=("YES" if task_collapse else "NO"),
                        binary_harm_flag=("YES" if binary_harm else "NO"),
                        improves_target=cell.get("improves_target", "")))

            # random-k specificity vs LEACE (the falsifier)
            leace, rk = s.get(f"{bb}|LEACE"), s.get(f"{bb}|random_k")
            if leace and rk:
                spec_rows.append(dict(
                    dataset=ds, backbone=bb,
                    leace_dtgt_nll=_num(leace.get("dtgt_nll")), randomk_dtgt_nll=_num(rk.get("dtgt_nll")),
                    leace_subj_dec_after=_num(leace.get("subj_dec_after_mean")),
                    randomk_subj_dec_after=_num(rk.get("subj_dec_after_mean")),
                    leace_dtgt_bacc=_num(leace.get("dtgt_bacc")), randomk_dtgt_bacc=_num(rk.get("dtgt_bacc")),
                    nll_move_nonspecific=_nonspecific(leace, rk)))

    # documented boundary rows the RQ2 tables must acknowledge (numeric JSON not committed on tos)
    summ_rows.append(dict(dataset="2a", backbone="TSMNet(in-loss)", eraser="LPC_global_collapse",
                          note="TSMNet feat_norm 1.09->0.00, CE->ln4 collapse (source: notes/PHASE21 + CLAIMS_LEDGER C5; not a deploy cell)"))
    summ_rows.append(dict(dataset="2a", backbone="TSMNet(in-loss)", eraser="collapse_free_removes_nothing",
                          note="warm_ramp collapse 0/9; collapse-free subj_dec ~0.997 = ERM (source: CLAIMS_LEDGER C6; variant_compare.json not committed on tos)"))
    summ_rows.append(dict(dataset="2a", backbone="TSMNet", eraser="refusal_gate",
                          note="source-only gate: 0 unsafe-accepts w/ power floor vs 6; EEG accept 5/9 VACUOUS (certified_accept=False); default-on NOT certified (source: PHASE131 + PHASE2_EEG_FROZEN_PILOT)"))

    _wcsv(OUT / "tos_erasure_summary.csv", summ_rows, _summ_cols())
    _wcsv(OUT / "tos_randomk_specificity.csv", spec_rows, None)
    _wcsv(OUT / "tos_task_safety_flags.csv", safety_rows, None)

    # console readout
    print(f"tos_erasure_summary: {len(summ_rows)} rows across {len(DEPLOY)} datasets")
    print("random-k specificity (NLL move non-specific where random_k matches LEACE NLL w/o removing subject):")
    for r in spec_rows:
        print(f"  {r['dataset']:16s} {r['backbone']:7s} LEACE dNLL={r['leace_dtgt_nll']} "
              f"randomk dNLL={r['randomk_dtgt_nll']} (randomk subj_dec_after={r['randomk_subj_dec_after']}) "
              f"-> non_specific={r['nll_move_nonspecific']}")
    inlp = [r for r in safety_rows if r["eraser"] == "INLP" and r["task_collapse_flag"] == "YES"]
    harm = [r for r in safety_rows if r["binary_harm_flag"] == "YES"]
    print(f"INLP task-collapse cells: {[(r['dataset'], r['backbone']) for r in inlp]}")
    print(f"binary-EEGNet harm cells (dtgt_bacc<=-0.05): {[(r['dataset'], r['backbone'], r['eraser']) for r in harm]}")


def _verdict(er, dtgt_bacc, tgt_bacc, chance):
    if er == "full":
        return "baseline"
    if isinstance(tgt_bacc, float) and isinstance(chance, float) and tgt_bacc <= chance + 0.02:
        return "task_collapse"
    if isinstance(dtgt_bacc, float) and dtgt_bacc <= -0.05:
        return "harms_target"
    if isinstance(dtgt_bacc, float) and dtgt_bacc < 0.01:
        return "no_target_gain"
    return "inconclusive"


def _nonspecific(leace, rk):
    ln, rn = leace.get("dtgt_nll"), rk.get("dtgt_nll")
    rk_subj = rk.get("subj_dec_after_mean")
    try:
        # random_k did NOT remove subject (high subj_dec_after) yet moves NLL comparably to LEACE
        similar = abs(float(ln) - float(rn)) <= 0.02
        unremoved = float(rk_subj) >= 0.9
        return "YES" if (similar and unremoved) else "NO"
    except (TypeError, ValueError):
        return "unknown"


def _summ_cols():
    return ["dataset", "backbone", "eraser", "chance_task", "subj_lin_2a", "subj_mlp_2a",
            "subj_dec_after", "tgt_bacc", "dtgt_bacc", "tgt_nll", "dtgt_nll",
            "worst_subject_tgt_bacc", "improves_target", "verdict", "note"]


def _wcsv(path, rows, cols):
    if not rows:
        Path(path).write_text("")
        return
    if cols is None:
        cols = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


if __name__ == "__main__":
    sys.exit(main())
