# FSR_13 — Submission Hardening Checklist

**Project FSR — Phase 5A.** Submission-readiness status for the `paper/fsr/` manuscript. Automated checks are enforced by `scripts/fsr/check_bib_hygiene.py` and `scripts/fsr/check_paper_claims.py`; their reports are in `results/fsr_phase5a/`.

## Status

| item | status | evidence |
|---|---|---|
| BibTeX hygiene (no `verify`/`others`/`ICCV/TPAMI`/external-`Anonymous`; no unused/undefined keys) | **PASS** | `check_bib_hygiene.py` 0 issues, 13 entries all cited |
| `identitytrap2026` → real authors (`lin2026identitytrap`: Lin, Wu, Jung; arXiv 2606.06647) | **DONE** | refs.bib |
| Scatter Component Analysis split from MTAE (`ghifary2017sca`, TPAMI 2017, DOI 10.1109/TPAMI.2016.2599532) | **DONE** | refs.bib |
| LEACE / INLP / RLACE / DomainBed / EEGNet / FBCSP / TSMNet / Amnesic-Probing full entries | **DONE** | refs.bib |
| Internal repo notes moved out of Related Work → Provenance appendix | **DONE** | `07_related_work.tex` cites `prior_decoupled_tta_internal`; `cigl_internal`/`tos_cmi_internal` cited in `appendix/provenance_appendix.tex` |
| Claim hygiene (no forbidden claim asserted; safe phrases allowed) | **PASS** | `check_paper_claims.py` 0 violations, 7 negated/safe |
| Related work at submission level (6 subsections, external cites) | **DONE** | `07_related_work.tex` |
| Provenance appendix (artifact sources, tiers, firewall, RQ4 search) | **DONE** | `appendix/provenance_appendix.tex` |
| Route-ledger appendix (inclusion table + tags) | **DONE** | `appendix/route_ledger_appendix.tex` |
| Abstract compressed (2 problem / 2 method / 3 results / 1 boundary) | **DONE** | `main.tex` |
| "audit framework, not DG method" repetition reduced | **DONE** | kept as bookends (abstract, intro, conclusion); trimmed elsewhere |
| `main.pdf` compiles, 0 undefined cites/refs | **PASS** | pdflatex+bibtex clean |

## Still open (not blocking this commit)
- **Venue-specific formatting** (template, page limit, anonymization macros) — pending venue choice.
- **Two bib entries to double-check at camera-ready** — LEACE NeurIPS 2023 volume/pages, RLACE PMLR 162 pages (authors/venue/year set; exact pages best-effort).
- **RQ4 status will change** — see below.

## RQ4 status update (PM decision, this cycle)
The earlier "RQ4 blocked, Option B not approved" is **revised**: a strictly pre-registered, ERM-only FBCSP-LGG re-fit (Phase 4B) is now **approved** to instantiate the missing branch-local instrument (checkpoints + branch-latent dumps). Phase 5A does **not** change any RQ4 claim; the manuscript's RQ4 section stays as-is until Phase-4B results exist. Once 4B produces per-branch L1/L5, §6 will be reframed from "final blocked" to "blocked in frozen artifacts; we ran a pre-registered ERM refit to measure it," and the claim will be updated to whatever the refit shows (positive or negative). Protocol: `docs/FSR_15_RQ4_ERM_REFIT_PROTOCOL.md`.

## Enforcement gates (run before every submission build)
```bash
python scripts/fsr/check_bib_hygiene.py    # must exit 0
python scripts/fsr/check_paper_claims.py   # must exit 0
cd paper/fsr && pdflatex main && bibtex main && pdflatex main && pdflatex main  # 0 undefined
```
