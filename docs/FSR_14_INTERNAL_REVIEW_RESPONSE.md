# FSR_14 — Internal Review Response (Phase 5A)

**Project FSR — Phase 5A.** Point-by-point response to the PM's submission-hardening review, plus the mandatory list of **every changed claim sentence** (before → after) with a confirmation that no claim was strengthened.

## Responses to the review points

1. **`identitytrap2026` was `Anonymous`.** Fixed to `lin2026identitytrap` — Jun-You Lin, Ying Choon Wu, Tzyy-Ping Jung, arXiv:2606.06647. `check_bib_hygiene.py` now fails on `Anonymous` in any external entry.
2. **SCA conflated with MTAE.** Removed the conflated entry; added `ghifary2017sca` (Ghifary, Balduzzi, Kleijn, Zhang; TPAMI 39(7):1414–1430, 2017; DOI 10.1109/TPAMI.2016.2599532). The checker fails on `ICCV/TPAMI`.
3. **Internal citations in Related Work.** `07_related_work.tex` now cites only external work plus `prior_decoupled_tta_internal` (the decomposition-discipline manuscript). `cigl_internal` and `tos_cmi_internal` moved to `appendix/provenance_appendix.tex` as the internal evidence sources.
4. **Bib checker.** `check_bib_hygiene.py` fails on `verify`, external `Anonymous`, `others`, `ICCV/TPAMI`, `preprint`-without-eprint, and unused/undefined keys. Current: **PASS** (13 entries, all cited).
5. **Claim checker.** `check_paper_claims.py` fails on HARD phrases (`SOTA`, `unbiased CMI`, `state-of-the-art`) always, and on ASSERTED phrases only when not negated (markup-tolerant negation lookbehind). Current: **PASS** (0 violations; 7 legitimate negated disclaimers recognized).
6. **Provenance appendix.** Added — artifact-source table, claim-strength tier definitions, target-label firewall table, and the RQ4 checkpoint-search summary framed as a provenance limitation (not an experimental result).
7. **Abstract compression + repetition.** Abstract now 2/2/3/1 sentences; "audit framework, not DG method" kept only as bookends.

## Every changed claim sentence (before → after)

The only claim-bearing prose edited in Phase 5A is the **abstract** (all other edits are references, related-work text, appendix additions, or table relocation — none touch an FSR claim). Each pair below preserves claim strength.

- **(F1)** before: *"measured leakage magnitude does not certify reliance --- in a frozen graph-CMI diagnostic, task-head alignment is positively associated with functional reliance while recomputable leakage carries the wrong sign."*
  after: *"measured leakage magnitude does not certify functional reliance: task-head alignment is positively associated with reliance while recomputable leakage carries the wrong sign."*
  — same claim, reworded; no strengthening. (Provenance tiers remain in §4.)

- **(F2)** before: *"subject signal is erasable, but erasure strength does not certify a target benefit (0/40 cells achieve a proven gain)."*
  after: *"subject signal is erasable, yet no eraser achieves a proven target benefit across 40 held-out-target cells, and aggressive erasure can harm the target."*
  — same claim (0/40 = "no eraser achieves a proven benefit across 40 cells"); the added "aggressive erasure can harm the target" is factual (task-collapse / binary-harm cells) and weaker than a general negative-correlation claim, which remains explicitly withheld (§5, Step-2C). No strengthening.

- **(F3)** before: *"branch load matters --- the spatial branch of a fusion backbone is load-bearing --- but branch-local leakage/reliance is not measured, so branch-local shortcut claims require an instrument that does not yet exist."*
  after: *"a fusion backbone's spatial branch is load-bearing, but branch-local leakage and reliance are unmeasured and cannot be measured without a new run."*
  — same claim; no strengthening. (Now consistent with the approved Phase-4B refit.)

- **(F4)** before: *"FSR is an audit framework, not a new domain-generalization method."*
  after: *"FSR is an audit framework, not a domain-generalization method: it makes explicit what measurement is required to support what control conclusion."*
  — same boundary; the added clause is descriptive, not a new claim.

## Claims explicitly NOT changed
- RQ1/RQ3 provenance tiers (RECOMPUTED / SIGN_ONLY / FROZEN_NOT_RECOMPUTABLE) — unchanged (§4).
- RQ2 headline = `benefit_claimable=0/40`; the negative correlation remains `NOT_ROBUST_DO_NOT_HEADLINE` (§5) — unchanged.
- RQ4 claim (spatial load-bearing; branch-local unmeasured) — unchanged pending Phase-4B results.
- The 10-row claim ledger statuses — unchanged (still derived from the frozen artifacts; all match expectation).

## Integrity note
Any future prose edit that touches a claim must be logged here with a before/after pair and must pass `check_paper_claims.py`. The scientific boundary is fixed; wording may be polished but claims may not be silently strengthened.
