"""C16 report — renders the mechanism deep dive (currently the C16-A target-oracle ceiling; harm decomposition
and discriminative-validity sections are appended by their modules). Markdown + canonical JSON + CSV tables.
The target oracle is labeled NON-DEPLOYABLE everywhere."""
from __future__ import annotations

import argparse
import csv
import os

from ..artifacts.canonical_json import canonical_json_bytes
from .target_oracle import load_replay, run_target_oracle


def _f(x, nd=4):
    return "n/a" if x is None else (f"{x:+.{nd}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x))


def _wcsv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow(r)


def write_target_oracle_tables(res, outdir) -> list:
    w = []
    sel = res["selectors"]
    # 1. ceiling by target/seed/level (target oracle bAcc)
    p = os.path.join(outdir, "target_oracle_ceiling_by_target_seed_level.csv")
    _wcsv(p, ["seed", "target", "level", "chosen_epoch", "chosen_is_erm", "target_bacc_delta", "target_nll_delta", "target_ece_delta"],
          [[r["seed"], r["target"], r["level"], r["chosen_epoch"], r["chosen_is_erm"], r["target_bacc_delta"],
            r["target_nll_delta"], r["target_ece_delta"]] for r in sel["target_oracle_bacc"]["per_fold"]]); w.append(p)
    # 2. target vs source oracle (worst-held-out-target K2 units)
    p = os.path.join(outdir, "target_oracle_vs_source_oracle.csv")
    rows = []
    for u_t, u_s in zip(sel["target_oracle_bacc"]["per_unit"], sel["source_audit_oracle"]["per_unit"]):
        rows.append([u_t["seed"], u_t["level"], u_t["delta_worst_bacc"], u_t["delta_worst_nll"],
                     u_s["delta_worst_bacc"], u_s["delta_worst_nll"]])
    _wcsv(p, ["seed", "level", "target_oracle_dbacc", "target_oracle_dnll", "source_oracle_dbacc", "source_oracle_dnll"], rows)
    w.append(p)
    # 3. endpoint tradeoffs (does the bAcc-optimal choice help or hurt NLL?)
    p = os.path.join(outdir, "target_oracle_endpoint_tradeoffs.csv")
    _wcsv(p, ["seed", "level", "target_oracle_bacc_dnll", "target_oracle_joint_dnll"],
          [[ub["seed"], ub["level"], ub["delta_worst_nll"], uj["delta_worst_nll"]]
           for ub, uj in zip(sel["target_oracle_bacc"]["per_unit"], sel["target_oracle_joint"]["per_unit"])]); w.append(p)
    # 4. rescue taxonomy
    p = os.path.join(outdir, "target_oracle_rescue_taxonomy.csv")
    _wcsv(p, ["quantity", "value"], [["case_label", res["case_label"]],
                                     ["source_audit_oracle_k2", res["source_audit_oracle_k2"]],
                                     ["target_oracle_bacc_k2", res["target_oracle_bacc_k2"]],
                                     ["target_oracle_joint_k2", res["target_oracle_joint_k2"]],
                                     ["target_oracle_rescues_bacc", res["target_oracle_rescues_bacc"]],
                                     ["target_oracle_rescues_joint", res["target_oracle_rescues_joint"]],
                                     ["source_oracle_rescues", res["source_oracle_rescues"]]]); w.append(p)
    # 5. checkpoint existence summary (per selector: K2 + how often it picks a non-ERM checkpoint)
    p = os.path.join(outdir, "checkpoint_existence_summary.csv")
    rows = []
    for name, sv in sel.items():
        n_nonerm = sum(1 for r in sv["per_fold"] if not r["chosen_is_erm"])
        rows.append([name, sv["k2_status"], sv.get("reproduced_endpoints"), n_nonerm, len(sv["per_fold"]),
                     sv["non_deployable"]])
    _wcsv(p, ["selector", "k2_status", "reproduced_endpoints", "n_chose_nonERM", "n_fold_levels", "non_deployable"], rows)
    w.append(p)
    return w


def render_target_oracle_md(res) -> str:
    sel = res["selectors"]
    L = ["# C16-A — Target-oracle ceiling (does OACI's trajectory contain a target-good checkpoint?)", "",
         "> The target oracle is a **NON-DEPLOYABLE, post-hoc diagnostic**: it reads target_audit only to test "
         "checkpoint EXISTENCE, never as a selection method.", "",
         f"- **CASE: `{res['case_label']}`** — {res['interpretation']}", "",
         "| selector | K2 (worst-held-out-target) | reproduced | deployable |", "|---|---|---|---|"]
    for name, sv in sel.items():
        L.append(f"| {name} | `{sv['k2_status']}` | {sv.get('reproduced_endpoints') or '—'} | "
                 f"{'NO (diagnostic)' if sv['non_deployable'] else 'yes'} |")
    L += ["", "## What this splits from C10's case C", "",
          f"- source-audit oracle rescues K2: **{res['source_oracle_rescues']}** (reproduces C10 case C)",
          f"- target oracle rescues worst-domain **bAcc**: **{res['target_oracle_rescues_bacc']}** "
          "→ target-accuracy-good checkpoints **DO exist** in the trajectory but are **not source-observable**",
          f"- target oracle rescues **joint** (bAcc+NLL): **{res['target_oracle_rescues_joint']}** "
          "→ a separate **calibration barrier**: even the accuracy-optimal checkpoints do not jointly improve NLL", "",
          "## Target-oracle bAcc ceiling — worst-held-out-target Δ vs ERM (per seed, level)", "",
          "| seed | level | Δ worst bAcc | Δ worst NLL |", "|---:|---:|---:|---:|"]
    for u in sel["target_oracle_bacc"]["per_unit"]:
        L.append(f"| {u['seed']} | {u['level']} | {_f(u['delta_worst_bacc'])} | {_f(u['delta_worst_nll'])} |")
    L += ["", "> **Mechanism.** The measurement→control decoupling for *accuracy* is a **source-side "
          "observability failure** (the good checkpoints exist; source signal cannot identify them), while "
          "*calibration* harm persists even at the target-accuracy ceiling. This is a diagnostic result; the "
          "target oracle is never deployable and makes no selection claim."]
    return "\n".join(L)


def build_target_oracle(replay_dir):
    return run_target_oracle(load_replay(replay_dir))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.mechanism.report")
    ap.add_argument("--replay-dir", default="/projects/EEG-foundation-model/yinghao/oaci-c10-replay")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--tables-dir", required=True)
    args = ap.parse_args(argv)
    res = build_target_oracle(args.replay_dir)
    for p in (args.out_md, args.out_json):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    tables = write_target_oracle_tables(res, args.tables_dir)
    with open(args.out_json, "wb") as f:
        f.write(canonical_json_bytes(res))
    with open(args.out_md, "w") as f:
        f.write(render_target_oracle_md(res))
    print(f"wrote {args.out_md} + {args.out_json} + {len(tables)} tables; case={res['case_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
