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
| schirrmeister2017 | Schirrmeister, Springenberg, Fiederer, et al. | Deep learning with convolutional neural networks for EEG decoding and visualization | *Human Brain Mapping*, 2017 | verified (title/venue/year); **TODO: verify** vol/pages/DOI (ShallowConvNet & DeepConvNet) |
| dgcnn_song | Song, Zheng, et al. | EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks | IEEE (venue TBD) | title/method verified; **TODO: verify** exact venue (IEEE Trans. Affective Computing?)/year/DOI |
| rgnn_zhong2019 | Zhong, Wang, Miao | EEG-Based Emotion Recognition Using Regularized Graph Neural Networks | arXiv:1907.07835 (RGNN; NodeDAT + EmotionDL) | verified (arXiv id/title); **TODO: verify** journal version (IEEE Trans. Affective Computing?) |
| lggnet_ding2021 | Ding, et al. | LGGNet: Learning from Local-Global-Graph Representations for Brain–Computer Interface | arXiv:2105.02786 | verified (arXiv id/title); **TODO: verify** journal version/year |
| condinv_li2018 | Li, et al. | Domain Generalization via Conditional Invariant Representations | *AAAI*, 2018 | verified (venue/year); **TODO: verify** exact author list/DOI |
| ccmi_mukherjee2020 | Mukherjee, Asnani, Kannan | CCMI: Classifier based Conditional Mutual Information Estimation | *PMLR* (UAI), 2020 | verified (title/venue/year); **TODO: verify** volume (v115)/pages |
| bnci2014_001 | Brunner, Leeb, Müller-Putz, Schlögl, Pfurtscheller | BCI Competition 2008 – Graz data set A (BNCI2014_001 / IV-2a) | data desc. 2008; MOABB cites Tangermann et al. *Front. Neurosci.* **6:55**, 2012 | **RESOLVED** — Tangermann DOI 10.3389/fnins.2012.00055 (reviewer-verified); data-desc URL = TODO |
| bnci2015_001 | Faller, Vidaurre, Solis-Escalante, Neuper, Scherer | Autocalibration and recurrent adaptation … plug-and-play online ERD-BCI (BNCI2015_001) | *IEEE TNSRE* **20(3):313–319**, 2012 | **RESOLVED** — DOI 10.1109/TNSRE.2012.2189584 (reviewer-verified) |

## Notes
- These verified-where-known entries replace the corresponding `TODO: verify citation` placeholders in
  `RELATED_WORK_MATRIX.md` only where the row above is "verified"; rows still bearing `TODO: verify` keep
  their placeholder in the matrix.
- Dataset primary citations (2a, 2015) are now **RESOLVED** (reviewer-verified): BNCI2014_001 → Tangermann
  et al. 2012 (DOI 10.3389/fnins.2012.00055) + Brunner et al. 2008 Graz-2a description; BNCI2015_001 →
  Faller et al. 2012 (DOI 10.1109/TNSRE.2012.2189584). MOABB + EEGNet DOIs also resolved.
- **Still TODO** (do not invent): Schirrmeister vol/pages/DOI; DGCNN exact venue/year/DOI; RGNN/LGGNet
  published-version DOI; Li et al. author list/DOI; CCMI volume/pages. See `CITATION_TODO_QUEUE.md`.
- A draft `.bib` now exists at `REFERENCES_DRAFT.bib` (verified-identity entries only; unverified fields are
  `% TODO` comments, never fabricated values).
