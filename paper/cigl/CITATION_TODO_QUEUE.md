# CIGL Citation TODO Queue (Phase 4E / v0.3)

> Per-citation work queue. Pairs with `REFERENCES_VERIFIED.md` (verification ledger). **No fields are
> fabricated**; "missing field" = not yet confirmed from a reliable source. "Required before submission?"
> flags whether the manuscript can be submitted with the field still open (TODO) or not.

| # | item | current status | exact field(s) missing | suggested source to verify | required before submission? |
|---|---|---|---|---|---|
| 1 | **BNCI2014_001** primary dataset (BCI Competition IV-2a) | TODO (no primary cite) | author list, title, year, venue/DOI | Brunner/Tangermann et al., BCI Competition IV 2a description; MOABB dataset docs page for `BNCI2014_001` | **YES** (a used dataset must be cited) |
| 2 | **BNCI2015_001** primary dataset (`001-2015`) | TODO (no primary cite) | author list, title, year, venue/DOI | Faller et al. 2012, IEEE TNSRE (auto-calibration); MOABB dataset docs page for `BNCI2015_001` | **YES** |
| 3 | **MOABB** (benchmark/protocol) | verified title/venue/year | volume, DOI | MOABB docs "cite us" page (Jayaram & Barachant, *J. Neural Eng.* 2018) | YES (vol/DOI) — title ok |
| 4 | **EEGNet** (Lawhern et al.) | verified title/venue/year + PMID 29932424 | DOI, volume/article no. | *J. Neural Eng.* 2018 record / PubMed | YES (DOI) — identity ok |
| 5 | **ShallowConvNet / DeepConvNet** (Schirrmeister et al.) | verified title/venue/year | volume, pages, DOI | *Human Brain Mapping* 2017 article page | YES (vol/pages/DOI) |
| 6 | **DGCNN** (Song et al., EEG emotion, dynamical graph CNN) | title/method verified | exact venue (likely *IEEE Trans. Affective Computing*), year, DOI | IEEE Xplore record for the DGCNN emotion paper | YES (venue/year/DOI) |
| 7 | **RGNN** (Zhong, Wang, Miao) | verified arXiv id 1907.07835 + title | journal version (likely *IEEE TAC*), year, DOI | arXiv:1907.07835; published TAC version | NO (arXiv citable) — prefer journal if available |
| 8 | **LGGNet** (Ding et al.) | verified arXiv id 2105.02786 + title | journal/conf version, year, DOI | arXiv:2105.02786; published version | NO (arXiv citable) — prefer published |
| 9 | **Conditional invariant representations** (Li et al.) | verified venue AAAI 2018 | exact author list, title, DOI/page | AAAI 2018 proceedings record | YES (author list/DOI) |
| 10 | **CCMI** (Mukherjee, Asnani, Kannan) | verified title/venue/year | PMLR volume (v115?), pages | PMLR UAI 2020 (v115) proceedings page | YES (vol/pages) |

## Notes
- Items 1–2 (dataset primaries) are the **highest-priority** open citations — a used dataset must be cited;
  do not submit with these as TODO.
- Items 7–8 are arXiv-citable now; upgrade to the published version if confirmed.
- This queue does not invent any field; entries stay `TODO: verify` in `REFERENCES_VERIFIED.md` /
  `RELATED_WORK_MATRIX.md` until confirmed from the listed source.
