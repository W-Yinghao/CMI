# CIGL References (Phase 4E / v0.3 — verification status)

> Verified from reviewer-provided sources / standard records. **"verified" here means the title/venue/year
> baseline** (plus author names and the PMID/arXiv id where independently known); it does **not** assert the
> exact volume/pages/DOI, which stay **TODO: verify** until checked against the publisher record. **No BibTeX
> is finalized here** — this is a verification ledger, not a `.bib`. See `CITATION_TODO_QUEUE.md` for the
> per-item work queue and required-before-submission flags.

| key | authors | title | venue / year | status |
|---|---|---|---|---|
| moabb2018 | Jayaram, Barachant | MOABB: trustworthy algorithm benchmarking for BCIs | *J. Neural Engineering* **15(6):066011**, 2018 | **RESOLVED** — DOI 10.1088/1741-2552/aadea0 (reviewer-verified) |
| eegnet2018 | Lawhern, Solon, Waytowich, Gordon, Hung, Lance | EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces | *J. Neural Engineering* **15(5):056013**, 2018 (PMID 29932424) | **RESOLVED** — DOI 10.1088/1741-2552/aace8c (reviewer-verified) |
| schirrmeister2017deep | Schirrmeister, Springenberg, Fiederer, Glasstetter, Eggensperger, Tangermann, Hutter, Burgard, Ball | Deep learning with convolutional neural networks for EEG decoding and visualization | *Human Brain Mapping* **38(11):5391–5420**, 2017 | **RESOLVED** — DOI 10.1002/hbm.23730 (reviewer-verified) |
| song2018dgcnn | Song, Zheng, Song, Cui | EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks | *IEEE Trans. Affective Computing* **11(3):532–541**, 2020 (early access 2018) | **RESOLVED** — DOI 10.1109/TAFFC.2018.2817622 (reviewer-verified) |
| zhong2019rgnn | Zhong, Wang, Miao | EEG-Based Emotion Recognition Using Regularized Graph Neural Networks | *IEEE Trans. Affective Computing* **13(3):1290–1301**, 2022 | **RESOLVED** — DOI 10.1109/TAFFC.2020.2994159 (Crossref-verified; supersedes arXiv:1907.07835) |
| ding2021lggnet | Ding, Robinson, Tong, Zeng, Guan | LGGNet: Learning from Local-Global-Graph Representations for Brain–Computer Interface | *IEEE Trans. Neural Netw. Learn. Syst.* **35(7):9773–9786**, 2024 | **RESOLVED** — DOI 10.1109/TNNLS.2023.3236635 (Crossref-verified; 3rd author = Tong, not Zhang; supersedes arXiv:2105.02786) |
| li2018conditional | Li, Gong, Tian, Liu, Tao | Domain Generalization via Conditional Invariant Representations | *Proc. AAAI* **32(1)**, 2018 | **RESOLVED** — DOI 10.1609/aaai.v32i1.11682 (reviewer-verified) |
| mukherjee2020ccmi | Mukherjee, Asnani, Kannan | CCMI: Classifier based Conditional Mutual Information Estimation | *PMLR* **115:1083–1093** (UAI), 2020 | **RESOLVED** — vol/pages verified (reviewer); PMLR has no DOI (normal) |
| bnci2014_001 | Brunner, Leeb, Müller-Putz, Schlögl, Pfurtscheller | BCI Competition 2008 – Graz data set A (BNCI2014_001 / IV-2a) | data desc. 2008; MOABB cites Tangermann et al. *Front. Neurosci.* **6:55**, 2012 | **RESOLVED** — Tangermann DOI 10.3389/fnins.2012.00055 (reviewer-verified); data-desc URL = TODO |
| bnci2015_001 | Faller, Vidaurre, Solis-Escalante, Neuper, Scherer | Autocalibration and recurrent adaptation … plug-and-play online ERD-BCI (BNCI2015_001) | *IEEE TNSRE* **20(3):313–319**, 2012 | **RESOLVED** — DOI 10.1109/TNSRE.2012.2189584 (reviewer-verified) |

## Notes
- These verified-where-known entries replace the corresponding `TODO: verify citation` placeholders in
  `RELATED_WORK_MATRIX.md` only where the row above is "verified"; rows still bearing `TODO: verify` keep
  their placeholder in the matrix.
- Dataset primary citations (2a, 2015) are now **RESOLVED** (reviewer-verified): BNCI2014_001 → Tangermann
  et al. 2012 (DOI 10.3389/fnins.2012.00055) + Brunner et al. 2008 Graz-2a description; BNCI2015_001 →
  Faller et al. 2012 (DOI 10.1109/TNSRE.2012.2189584). MOABB + EEGNet DOIs also resolved.
- **All 10 citations RESOLVED** (Phase 4G): Schirrmeister, DGCNN, RGNN, LGGNet, Li 2018, CCMI added with full
  fields/DOIs (RGNN & LGGNet DOIs/pages/authors Crossref-verified — LGGNet 3rd author corrected to Tong).
  Only a **minor** sub-field remains TODO: the Brunner Graz-2a data-description URL (a note, not a citation).
- A draft `.bib` now exists at `REFERENCES_DRAFT.bib` (verified-identity entries only; unverified fields are
  `% TODO` comments, never fabricated values).
