# Figure and table plan

Figures are *specified*, not generated: no new plots are produced for C16. Each figure/table names the
committed artifact(s) that already contain the data, so C18 (paper-grade figures) can render them without any
new experiment.

## Figures

- **Figure 1 — Battery schematic.** The gate pipeline G0 integrity → G1 selection/audit optimism → G2 K1
  (held-out leakage) → G3 K2 (endpoint transfer) → G4 source-audit oracle → G5 source→target anti-transfer,
  with each gate labeled by what it falsifies. *Source:* `oaci/falsification/` gate definitions; C14 gate list.
- **Figure 2 — Evidence chain.** A timeline/flow across C8 → C10 → C12 → C14 showing OACI leakage-control
  closed, SRC endpoint-control closed, measurement retained. *Source:* `c15_tables/evidence_chain_c8_c10_c12_c14.csv`.
- **Figure 3 — Selection-vs-audit leakage transfer.** Per-fold Δ selection leakage (−0.326, 54/54) contrasted
  with Δ audit leakage (+0.008, 25/54); near-zero correlation. *Source:* C10a `selection_to_audit_optimism`;
  `c14_tables/selection_audit_optimism.csv`.
- **Figure 4 — Source→target endpoint instability.** Scatter of Δ source-guard NLL (x) vs Δ target
  worst-domain NLL (y) over the 6 SRC active cells; anti-transfer quadrant highlighted (Pearson −0.947).
  *Source:* C12 cells; `c14_tables/source_target_transfer.csv` and `antitransfer_flags.csv`.
- **Figure 5 — Method-closure map.** OACI + SRC closed as control objectives (under this protocol);
  support-aware leakage / K1-K2 / oracle replay / anti-transfer retained as instruments. *Source:*
  `c14_tables/method_closure_table.csv`.

## Tables

- **Table 1 — Battery gates.** Each gate, its question, and what it falsifies (G0–G5) with the BNCI2014-001
  outcome. *Source:* `c15_tables/falsification_protocol_steps.csv`.
- **Table 2 — Claim–evidence map.** The four scoped claims, their status, and the committed number backing
  each. *Source:* `C15_CLAIM_EVIDENCE_MAP.json`, `c15_tables/claim_evidence_map.csv`.
- **Table 3 — K1/K2 aggregate.** K1 sweep (11 nominal / 0 Bonferroni / 0 BH of 54) and K2
  (`stop_no_reproducible_gain`; worst-domain bAcc 2/6 improved, worst-domain NLL improved-not-reproducibly).
  *Source:* `C8_BNCI001_LOSO_SEEDS012_K1K2.json`.
- **Table 4 — Oracle replay selectors.** S0–S5, each selector's K2 verdict, whether it reads source_audit, and
  the identity-check summary (216/216, 0 flips). *Source:* `c14_tables/oracle_rescue.csv`; C10b report.
- **Table 5 — SRC anti-transfer summary.** Per (target, temperature, level): Δ source-guard NLL, Δ target NLL,
  blowup flag, fallback; with the seed-0 / n=6 / no-λ-sweep caveats attached. *Source:* C12 report;
  `c14_tables/antitransfer_flags.csv`.
- **Table 6 — Limitations / objection summary.** A condensed reviewer-objection matrix for the Limitations
  section (objection, answered-by-committed-evidence, response). *Source:* `C15_REVIEWER_OBJECTION_MATRIX.md`;
  `c15_tables/limitation_boundary_table.csv`.

## Provenance note
Every quantity in every figure/table above is already committed to branch `oaci` and is regenerable from the
committed reports via `oaci/confirmatory/c15_spine.py` and `oaci/falsification/report.py`. No number in the
manuscript is transcribed by hand.
