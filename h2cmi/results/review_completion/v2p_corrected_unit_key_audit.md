# V2P Corrected Unit-Key Audit

Raw rows are `results/h2cmi/wave0_v2p/*.jsonl`. Schema fields map the requested unit key as `(dataset, pair, subject, target_session, source_seed, method)` = `(dataset, pair, subject, tgt_sess, seed, estimator)`. Repeated BNCI2014_004 transitions remain distinct through `pair`/`tgt_sess`; bootstrap cluster is `(dataset, subject)`. Executed q-grid is `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`; no missing q values are invented. Confirmatory status: corrected reanalysis from frozen Wave0 raw artifacts.
