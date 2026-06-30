# CIGL Citation TODO Queue (Phase 4G / v0.5)

> Per-citation work queue. Pairs with `REFERENCES_VERIFIED.md` (verification ledger). **No fields are
> fabricated**; "missing field" = not yet confirmed from a reliable source. "Required before submission?"
> flags whether the manuscript can be submitted with the field still open (TODO) or not.

| # | item | current status | exact field(s) missing | suggested source to verify | required before submission? |
|---|---|---|---|---|---|
| 1 | **BNCI2014_001** primary dataset (BCI Competition IV-2a) | **RESOLVED** (reviewer-verified) — `tangermann2012review` (*Front. Neurosci.* 6:55, DOI 10.3389/fnins.2012.00055) + `brunner2008graz2a` data description | data-description URL/form (minor) | reviewer; MOABB docs page | resolved (data-desc URL = minor TODO) |
| 2 | **BNCI2015_001** primary dataset (`001-2015`) | **RESOLVED** (reviewer-verified) — `faller2012autocalibration`, IEEE TNSRE 20(3):313–319, DOI 10.1109/TNSRE.2012.2189584 | — | reviewer; IEEE Xplore | resolved |
| 3 | **MOABB** (benchmark/protocol) | **RESOLVED** (reviewer-verified) — `jayaram2018moabb`, *J. Neural Eng.* 15(6):066011, DOI 10.1088/1741-2552/aadea0 | — | reviewer; MOABB docs | resolved |
| 4 | **EEGNet** (Lawhern et al.) | **RESOLVED** (reviewer-verified) — `lawhern2018eegnet`, *J. Neural Eng.* 15(5):056013, DOI 10.1088/1741-2552/aace8c | — | reviewer; PubMed | resolved |
| 5 | **ShallowConvNet / DeepConvNet** (Schirrmeister et al.) | **RESOLVED** — `schirrmeister2017deep`, *HBM* 38(11):5391–5420, 2017, DOI 10.1002/hbm.23730 | — | reviewer; Wiley | resolved |
| 6 | **DGCNN** (Song et al.) | **RESOLVED** — `song2018dgcnn`, *IEEE TAC* 11(3):532–541, 2020 (early access 2018), DOI 10.1109/TAFFC.2018.2817622 | — | reviewer; IEEE Xplore | resolved |
| 7 | **RGNN** (Zhong, Wang, Miao) | **RESOLVED** — `zhong2019rgnn`, *IEEE TAC* 13(3):1290–1301, 2022, DOI 10.1109/TAFFC.2020.2994159 | — | **Crossref-verified** | resolved |
| 8 | **LGGNet** (Ding, Robinson, Tong, Zeng, Guan) | **RESOLVED** — `ding2021lggnet`, *IEEE TNNLS* 35(7):9773–9786, 2024, DOI 10.1109/TNNLS.2023.3236635 | — | **Crossref-verified** (3rd author = Tong, not Zhang) | resolved |
| 9 | **Conditional invariant representations** (Li, Gong, Tian, Liu, Tao) | **RESOLVED** — `li2018conditional`, *Proc. AAAI* 32(1), 2018, DOI 10.1609/aaai.v32i1.11682 | — | reviewer; AAAI | resolved |
| 10 | **CCMI** (Mukherjee, Asnani, Kannan) | **RESOLVED** — `mukherjee2020ccmi`, *PMLR* 115:1083–1093 (UAI), 2020 | — (PMLR has no DOI) | reviewer; PMLR | resolved |

## Status (Phase 4G)
- **Resolved (10/10):** all citations now have verified core fields in `REFERENCES_DRAFT.bib`. Items 1–6, 9
  reviewer-verified; items 7–8 (RGNN/LGGNet) **independently Crossref-verified** (DOIs resolve; LGGNet 3rd
  author corrected Zhang→Tong; both upgraded from arXiv to the published IEEE versions). **All citation
  blockers are cleared.**
- **Only remaining (minor, not a citation):** the Brunner Graz-2a *data-description URL* (`brunner2008graz2a`
  `note`/`% TODO`); not required for the citation to be valid.
- This queue invents nothing; the lone remaining `% TODO` in `REFERENCES_DRAFT.bib` is that data-desc URL.
