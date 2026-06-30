# CIGL References (Phase 4E / v0.3 — verification status)

> Verified from reviewer-provided sources / standard records. **"verified" here means the title/venue/year
> baseline** (plus author names and the PMID/arXiv id where independently known); it does **not** assert the
> exact volume/pages/DOI, which stay **TODO: verify** until checked against the publisher record. **No BibTeX
> is finalized here** — this is a verification ledger, not a `.bib`. See `CITATION_TODO_QUEUE.md` for the
> per-item work queue and required-before-submission flags.

| key | authors | title | venue / year | status |
|---|---|---|---|---|
| moabb2018 | Jayaram, Barachant | MOABB: trustworthy algorithm benchmarking for BCIs | *J. Neural Engineering*, 2018 | verified (title/venue/year); **TODO: verify** vol/DOI |
| eegnet2018 | Lawhern, Solon, Waytowich, Gordon, Hung, Lance | EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces | *J. Neural Engineering*, 2018 (PMID 29932424) | verified (title/venue/year/PMID); **TODO: verify** DOI |
| schirrmeister2017 | Schirrmeister, Springenberg, Fiederer, et al. | Deep learning with convolutional neural networks for EEG decoding and visualization | *Human Brain Mapping*, 2017 | verified (title/venue/year); **TODO: verify** vol/pages/DOI (ShallowConvNet & DeepConvNet) |
| dgcnn_song | Song, Zheng, et al. | EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks | IEEE (venue TBD) | title/method verified; **TODO: verify** exact venue (IEEE Trans. Affective Computing?)/year/DOI |
| rgnn_zhong2019 | Zhong, Wang, Miao | EEG-Based Emotion Recognition Using Regularized Graph Neural Networks | arXiv:1907.07835 (RGNN; NodeDAT + EmotionDL) | verified (arXiv id/title); **TODO: verify** journal version (IEEE Trans. Affective Computing?) |
| lggnet_ding2021 | Ding, et al. | LGGNet: Learning from Local-Global-Graph Representations for Brain–Computer Interface | arXiv:2105.02786 | verified (arXiv id/title); **TODO: verify** journal version/year |
| condinv_li2018 | Li, et al. | Domain Generalization via Conditional Invariant Representations | *AAAI*, 2018 | verified (venue/year); **TODO: verify** exact author list/DOI |
| ccmi_mukherjee2020 | Mukherjee, Asnani, Kannan | CCMI: Classifier based Conditional Mutual Information Estimation | *PMLR* (UAI), 2020 | verified (title/venue/year); **TODO: verify** volume (v115)/pages |
| bnci2014_001 | (BCI Comp. IV-2a) | BNCI2014_001 / BCI Competition IV dataset 2a | MOABB dataset | **TODO: verify** primary citation (Brunner/Tangermann et al.) |
| bnci2015_001 | Faller, et al. | BNCI2015_001 (001-2015), right_hand vs feet MI | MOABB dataset | **TODO: verify** primary citation (Faller et al. 2012, IEEE TNSRE) |

## Notes
- These verified-where-known entries replace the corresponding `TODO: verify citation` placeholders in
  `RELATED_WORK_MATRIX.md` only where the row above is "verified"; rows still bearing `TODO: verify` keep
  their placeholder in the matrix.
- Dataset primary citations (2a, 2015) are still **TODO** — do not invent DOIs.
- Final `.bib` is deferred to the polish/submission step; nothing here fabricates pages/DOIs.
