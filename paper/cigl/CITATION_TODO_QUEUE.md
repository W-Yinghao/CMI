# CIGL Citation TODO Queue (Phase 4F / v0.4)

> Per-citation work queue. Pairs with `REFERENCES_VERIFIED.md` (verification ledger). **No fields are
> fabricated**; "missing field" = not yet confirmed from a reliable source. "Required before submission?"
> flags whether the manuscript can be submitted with the field still open (TODO) or not.

| # | item | current status | exact field(s) missing | suggested source to verify | required before submission? |
|---|---|---|---|---|---|
| 1 | **BNCI2014_001** primary dataset (BCI Competition IV-2a) | **RESOLVED** (reviewer-verified) — `tangermann2012review` (*Front. Neurosci.* 6:55, DOI 10.3389/fnins.2012.00055) + `brunner2008graz2a` data description | data-description URL/form (minor) | reviewer; MOABB docs page | resolved (data-desc URL = minor TODO) |
| 2 | **BNCI2015_001** primary dataset (`001-2015`) | **RESOLVED** (reviewer-verified) — `faller2012autocalibration`, IEEE TNSRE 20(3):313–319, DOI 10.1109/TNSRE.2012.2189584 | — | reviewer; IEEE Xplore | resolved |
| 3 | **MOABB** (benchmark/protocol) | **RESOLVED** (reviewer-verified) — `jayaram2018moabb`, *J. Neural Eng.* 15(6):066011, DOI 10.1088/1741-2552/aadea0 | — | reviewer; MOABB docs | resolved |
| 4 | **EEGNet** (Lawhern et al.) | **RESOLVED** (reviewer-verified) — `lawhern2018eegnet`, *J. Neural Eng.* 15(5):056013, DOI 10.1088/1741-2552/aace8c | — | reviewer; PubMed | resolved |
| 5 | **ShallowConvNet / DeepConvNet** (Schirrmeister et al.) | verified title/venue/year | volume, pages, DOI | *Human Brain Mapping* 2017 article page | YES (vol/pages/DOI) |
| 6 | **DGCNN** (Song et al., EEG emotion, dynamical graph CNN) | title/method verified | exact venue (likely *IEEE Trans. Affective Computing*), year, DOI | IEEE Xplore record for the DGCNN emotion paper | YES (venue/year/DOI) |
| 7 | **RGNN** (Zhong, Wang, Miao) | verified arXiv id 1907.07835 + title | journal version (likely *IEEE TAC*), year, DOI | arXiv:1907.07835; published TAC version | NO (arXiv citable) — prefer journal if available |
| 8 | **LGGNet** (Ding et al.) | verified arXiv id 2105.02786 + title | journal/conf version, year, DOI | arXiv:2105.02786; published version | NO (arXiv citable) — prefer published |
| 9 | **Conditional invariant representations** (Li et al.) | verified venue AAAI 2018 | exact author list, title, DOI/page | AAAI 2018 proceedings record | YES (author list/DOI) |
| 10 | **CCMI** (Mukherjee, Asnani, Kannan) | verified title/venue/year | PMLR volume (v115?), pages | PMLR UAI 2020 (v115) proceedings page | YES (vol/pages) |

## Status (Phase 4F)
- **Resolved (4/10):** items 1–4 (BNCI2014_001, BNCI2015_001, MOABB, EEGNet) — reviewer-verified, now in
  `REFERENCES_DRAFT.bib` with full DOIs. The dataset-primary blocker is **cleared**.
- **Remaining (6/10):** items 5 (Schirrmeister vol/pages/DOI), 6 (DGCNN exact venue/year/DOI), 9 (Li author
  list/DOI), 10 (CCMI vol/pages) are required-before-submission; items 7–8 (RGNN/LGGNet) are arXiv-citable
  now (`@misc` with eprint) and may be upgraded to published versions.
- This queue does not invent any field; remaining entries stay `% TODO` in `REFERENCES_DRAFT.bib` and
  `TODO: verify` in `REFERENCES_VERIFIED.md` / `RELATED_WORK_MATRIX.md` until confirmed from the listed source.
