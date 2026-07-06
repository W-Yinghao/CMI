"""C21 — generate the Estimand Boundary + Manuscript Lock artifacts from the claim ledger. Emits the estimand-
boundary report, the final claim ledger (4 categories), the manuscript lock statement, the paper outline v2,
and a forbidden-claim audit, plus 9 machine-checkable tables. Every generated MD is guarded against the
forbidden over-claims; every claim is evidence-grounded. NO experiments are run."""
from __future__ import annotations

import argparse
import csv
import json
import os

from . import ledger, schema


def _writecsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def build() -> dict:
    ledger.validate_claims(); ledger.assert_estimand_not_swapped()
    return {"paper_title": schema.PAPER_TITLE, "estimand_reconciliation": schema.ESTIMAND_RECONCILIATION,
            "claims": list(schema.CLAIMS), "claim_counts": ledger.counts(),
            "evidence_chain": [dict(stage=s, finding=f, key=k) for s, f, k in schema.EVIDENCE_CHAIN],
            "reviewer_objections": [dict(objection=o, response=r) for o, r in schema.REVIEWER_OBJECTIONS],
            "canonical_conclusion": schema.CANONICAL_CONCLUSION, "paper_sections": list(schema.PAPER_SECTIONS),
            "diagnostic_only_non_deployable": True, "manuscript_locked": True}


# ---------- tables ----------
def write_tables(res, tdir) -> None:
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "final_claim_ledger.csv"),
              [{"id": c["id"], "category": c["category"], "claim": c["text"], "report": c["report"],
                "commit": c["commit"], "key_evidence": c["key"]} for c in schema.CLAIMS],
              ["id", "category", "claim", "report", "commit", "key_evidence"])
    _writecsv(os.path.join(tdir, "c14_c20_evidence_chain.csv"), res["evidence_chain"], ["stage", "finding", "key"])
    er = res["estimand_reconciliation"]
    _writecsv(os.path.join(tdir, "c19_c20_estimand_reconciliation.csv"),
              [{"quantity": k, "value": v} for k, v in er.items()], ["quantity", "value"])
    _writecsv(os.path.join(tdir, "within_target_vs_pooled_estimand.csv"),
              [{"estimand": "pooled_loto_auc", "c19_in_regime": er["c19_pooled_loto_auc_in_regime"],
                "c20_cross_regime": f"{er['c20_pooled_loto_auc_cross_regime_range']} (near chance)",
                "status": "C19 passes in-regime; C20 does NOT transport (external validation not established)"},
               {"estimand": "within_target_mean_auc", "c19_in_regime": f"~{er['within_target_mean_auc_range']}",
                "c20_cross_regime": f"~{er['within_target_mean_auc_range']} (stable)",
                "status": "within-target RANKING persists across regimes; NOT the pre-registered pooled estimand (future-work)"}],
              ["estimand", "c19_in_regime", "c20_cross_regime", "status"])
    bc = ledger.by_category()
    _writecsv(os.path.join(tdir, "diagnostic_vs_deployable_claims.csv"),
              [{"id": c["id"], "claim": c["text"], "diagnostic_only": c["category"] == "diagnostic_only",
                "deployable": False} for c in schema.CLAIMS if c["category"] in ("diagnostic_only", "not_established")],
              ["id", "claim", "diagnostic_only", "deployable"])
    _writecsv(os.path.join(tdir, "method_closure_final.csv"),
              [{"objective": "OACI-control (DG control objective)", "status": "CLOSED / failed under protocol (C14)"},
               {"objective": "SRC-control (endpoint-aligned)", "status": "CLOSED / anti-transfer = memorization (C12/C16)"},
               {"objective": "deployable target-free selector", "status": "NOT established (no-selector gate)"},
               {"objective": "measurement / falsification / observability framework", "status": "RETAINED (contribution)"},
               {"objective": "diagnostic source-only probe", "status": "in-regime weak positive (C19); regime-local (C20)"}],
              ["objective", "status"])
    _writecsv(os.path.join(tdir, "external_validation_status.csv"),
              [{"item": "cross-regime (held-out support regimes)", "status": "NOT established (C20; Holm 0/4; Simpson)"},
               {"item": "external dataset (new cohort)", "status": "PROTOCOL only (C20-B); no execution"},
               {"item": "BNCI2014_004", "status": "BARRED_pending_explicit_approval"},
               {"item": "real support-mismatched clinical EEG", "status": "future work (cohorts not provisioned)"}],
              ["item", "status"])
    _writecsv(os.path.join(tdir, "reviewer_objection_matrix_v2.csv"), res["reviewer_objections"],
              ["objection", "response"])
    _writecsv(os.path.join(tdir, "future_work_boundary.csv"),
              [{"id": c["id"], "future_work": c["text"], "allowed_now": False,
                "reason": "manuscript locked; no experiment drift; needs approval / new protocol"}
               for c in bc["future_work"]], ["id", "future_work", "allowed_now", "reason"])


# ---------- MD renderers ----------
def render_estimand_boundary_md(res) -> str:
    er = res["estimand_reconciliation"]
    return ("# C21 — Estimand Boundary\n\n"
            "> The crux lesson of C19->C20. C19's weak robust-core competence signal is REAL in-regime; C20 "
            "shows it does not transport as a stable cross-regime POOLED estimand.\n\n"
            f"- C19 pooled LOTO AUC (in-regime): **{er['c19_pooled_loto_auc_in_regime']}** — passes the "
            "pre-registered in-regime criteria.\n"
            f"- C20 pooled LOTO AUC (cross-regime): **{er['c20_pooled_loto_auc_cross_regime_range']}** — near "
            "chance; external validation NOT established.\n"
            f"- within-target mean AUC: **~{er['within_target_mean_auc_range']}** — persists across ALL held-out "
            "regimes.\n\n"
            f"## Resolution\n> {er['resolution']}\n\n"
            f"## Why C19 stands\n> {er['why_c19_stands']}\n\n"
            f"## Forbidden reading (guarded)\n> {er['forbidden_reading']}\n")


def render_claim_ledger_md(res) -> str:
    bc = ledger.by_category()
    titles = {"established": "Established", "diagnostic_only": "Diagnostic-only", "not_established": "Not established",
              "future_work": "Future work"}
    L = ["# C21 — Final Claim Ledger", "", f"> {res['claim_counts']}", ""]
    for cat in schema.CATEGORIES:
        L.append(f"## {titles[cat]}")
        for c in bc[cat]:
            L.append(f"- **{c['id']}** {c['text']}  \n  *evidence:* {c['report']} ({c['commit']}) — {c['key']}")
        L.append("")
    return "\n".join(L)


def render_manuscript_lock_md(res) -> str:
    ch = "\n".join(f"- **{s}**: {f}  ({k})" for s, f, k in schema.EVIDENCE_CHAIN)
    return (f"# C21 — Manuscript Lock\n\n**Title:** {res['paper_title']}\n\n"
            "> STORY FROZEN. No further empirical branches (no new experiments / no probe tuning / no estimand "
            "swap / no BNCI2014_004 / no seeds [3,4] / no retrain / no selector / no new DG penalty). The chain "
            "C14->C20 is the final paper object.\n\n"
            f"## Evidence chain\n{ch}\n\n## Canonical conclusion\n> {res['canonical_conclusion']}\n")


def render_paper_outline_md(res) -> str:
    secs = "\n".join(f"{i+1}. **{name}** — {desc}" for i, (name, desc) in enumerate(schema.PAPER_SECTIONS))
    return (f"# C21 — Paper Outline v2\n\n**Title:** {res['paper_title']}\n\n{secs}\n\n"
            "> C19/C20 go in the MAIN text as a paired result (Result 5: in-regime weak positive; Result 6: "
            "external boundary / largely regime-local), NOT as a limitation appendix.\n")


def render_forbidden_audit_md(res) -> str:
    return ("# C21 — Forbidden-claim audit\n\nEvery generated C21 file is guarded against these over-claims; the "
            "claim ledger is evidence-grounded and the within-target ~0.64 is future-work only.\n\n"
            + "\n".join(f"- FORBIDDEN: {s}" for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS)
            + "\n\n- diagnostic_only + no-selector across C19/C20: enforced.\n"
            + "- estimand not swapped: within-target ~0.64 is future_work (F1), not a success claim.\n")


_SECTION_CLAIMS = {"03_falsification_battery": ("E1",), "04_mechanism_results": ("E2", "E3"),
                   "05_identifiability_and_probe": ("E4", "E5", "E6"),
                   "06_external_boundary": ("E7",),
                   "07_discussion_limitations": ("D1", "D2", "N1", "N2", "N3", "N4", "F1", "F2", "F3")}


def write_manuscript_scaffold(res, mdir) -> None:
    """A grounded scaffold (NOT finished prose): draft v0.2 + one stub per section carrying its claims/evidence
    and a TODO for prose. Regenerable from the ledger; keeps the paper anchored to committed evidence."""
    sdir = os.path.join(mdir, "sections"); os.makedirs(sdir, exist_ok=True)
    by_id = {c["id"]: c for c in schema.CLAIMS}
    for name, desc in schema.PAPER_SECTIONS:
        L = [f"# {name.replace('_', ' ').title()}", "", f"> Scope: {desc}", ""]
        for cid in _SECTION_CLAIMS.get(name, ()):
            c = by_id[cid]
            L.append(f"- **[{cid} · {c['category']}]** {c['text']}  \n  *evidence:* {c['report']} ({c['commit']}) — {c['key']}")
        if name == "06_external_boundary":
            er = res["estimand_reconciliation"]
            L += ["", f"- Estimand boundary: C19 pooled {er['c19_pooled_loto_auc_in_regime']} (in-regime, passes) "
                  f"vs C20 pooled {er['c20_pooled_loto_auc_cross_regime_range']} (cross-regime, near chance); "
                  f"within-target ~{er['within_target_mean_auc_range']} persists but is NOT the pooled estimand.",
                  f"- {er['resolution']}"]
        L += ["", "TODO: prose (this is a locked-evidence scaffold, not finished text)."]
        text = "\n".join(L); ledger.assert_no_forbidden(text)
        open(os.path.join(sdir, f"{name}.md"), "w").write(text)
    draft = [f"# {res['paper_title']}", "", "## Abstract (seed = canonical conclusion)", "",
             res["canonical_conclusion"], "", "## Sections", ""]
    draft += [f"{i+1}. [{n}](sections/{n}.md) — {d}" for i, (n, d) in enumerate(schema.PAPER_SECTIONS)]
    draft += ["", "> STORY LOCKED (C21). No new experiments; every result maps to a committed C14->C20 artifact."]
    dtext = "\n".join(draft); ledger.assert_no_forbidden(dtext)
    open(os.path.join(mdir, "C21_MANUSCRIPT_DRAFT_V0_2.md"), "w").write(dtext)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.manuscript_lock.report")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--manuscript-dir", default="oaci/manuscript")
    args = ap.parse_args(argv)
    res = build()
    files = {"C21_ESTIMAND_BOUNDARY.md": render_estimand_boundary_md(res),
             "C21_FINAL_CLAIM_LEDGER.md": render_claim_ledger_md(res),
             "C21_MANUSCRIPT_LOCK.md": render_manuscript_lock_md(res),
             "C21_PAPER_OUTLINE_V2.md": render_paper_outline_md(res),
             "C21_FORBIDDEN_CLAIM_AUDIT.md": render_forbidden_audit_md(res)}
    os.makedirs(args.out_dir, exist_ok=True)
    for name, text in files.items():
        if name != "C21_FORBIDDEN_CLAIM_AUDIT.md":   # the audit file DEFINES the phrases (a list); others guarded
            ledger.assert_no_forbidden(text)
        open(os.path.join(args.out_dir, name), "w").write(text)
    json.dump(res, open(os.path.join(args.out_dir, "C21_ESTIMAND_BOUNDARY.json"), "w"), indent=2, sort_keys=True)
    write_tables(res, os.path.join(args.out_dir, "c21_tables"))
    write_manuscript_scaffold(res, args.manuscript_dir)
    print(f"[C21] manuscript_locked={res['manuscript_locked']} claims={res['claim_counts']} "
          f"title={res['paper_title'][:50]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
