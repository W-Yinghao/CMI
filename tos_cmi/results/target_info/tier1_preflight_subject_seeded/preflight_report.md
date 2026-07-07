# Fork 1 Tier-1 --- REAL split preflight (split/hash/unavailable-k ONLY; NO metrics)

Purpose: split_hash_unavailable_k_only. No metrics emitted: True.
split_rng_scheme: **subject_seeded_v1** (global_split_seed 20240707, calib_fraction 0.5).

## Datasets / folds checked
- datasets: Lee2019_MI, Cho2017 ; backbones: EEGNet ; seed: 0 ; folds: [1, 2, 3, 4, 5]
- dumps checked: 10 ; R = 10 ; k-grid = [1, 2, 4, 8, 16]

## Target trial counts per class (per held-out target subject)
- Lee2019_MI fold 1 (subj 1) class 0: n_target_total = 100
- Lee2019_MI fold 1 (subj 1) class 1: n_target_total = 100
- Lee2019_MI fold 2 (subj 2) class 0: n_target_total = 100
- Lee2019_MI fold 2 (subj 2) class 1: n_target_total = 100
- Lee2019_MI fold 3 (subj 3) class 0: n_target_total = 100
- Lee2019_MI fold 3 (subj 3) class 1: n_target_total = 100
- Lee2019_MI fold 4 (subj 4) class 0: n_target_total = 100
- Lee2019_MI fold 4 (subj 4) class 1: n_target_total = 100
- Lee2019_MI fold 5 (subj 5) class 0: n_target_total = 100
- Lee2019_MI fold 5 (subj 5) class 1: n_target_total = 100
- Cho2017 fold 1 (subj 1) class 0: n_target_total = 100
- Cho2017 fold 1 (subj 1) class 1: n_target_total = 100
- Cho2017 fold 2 (subj 2) class 0: n_target_total = 100
- Cho2017 fold 2 (subj 2) class 1: n_target_total = 100
- Cho2017 fold 3 (subj 3) class 0: n_target_total = 100
- Cho2017 fold 3 (subj 3) class 1: n_target_total = 100
- Cho2017 fold 4 (subj 4) class 0: n_target_total = 100
- Cho2017 fold 4 (subj 4) class 1: n_target_total = 100
- Cho2017 fold 5 (subj 5) class 0: n_target_total = 100
- Cho2017 fold 5 (subj 5) class 1: n_target_total = 100

## Calibration / audit split summary
- calibration+audit index disjoint on ALL 100 splits: True
- total calibration∩audit overlap across all splits: 0
- n(calibration,audit) per (dataset,fold): {'Cho2017/f1/s1': (100, 100), 'Cho2017/f2/s2': (100, 100), 'Cho2017/f3/s3': (100, 100), 'Cho2017/f4/s4': (100, 100), 'Cho2017/f5/s5': (100, 100), 'Lee2019_MI/f1/s1': (100, 100), 'Lee2019_MI/f2/s2': (100, 100), 'Lee2019_MI/f3/s3': (100, 100), 'Lee2019_MI/f4/s4': (100, 100), 'Lee2019_MI/f5/s5': (100, 100)}

## Per-subject split diversity
- distinct calibration splits per target subject (want = R = 10): values seen = [10] over 10 subjects
- min 10 / max 10 distinct splits per subject

## k availability + nested-k check
- schema rows (dataset x fold x split x k x class): 1000
- UNAVAILABLE (dataset,fold,split,k) entries: 0
- nested-k subset checks passed (k=1 subset of k=2 ... subset of k=max, all within calibration pool): 500/500
- all requested k available on every split (no UNAVAILABLE entries)

## Hash summaries
- distinct calibration_idx_hash: 100 ; distinct audit_idx_hash: 100
- distinct calibration_label_hash: 51 ; distinct audit_label_hash: 51

## Confirmation
- NO estimator and NO intervention was run: this preflight produced only calibration/audit splits, per-class trial counts, k-availability, and index/label hashes. No predictive quality, no benefit, no likelihood, and no decision of any kind was computed. Split, count, availability, and hash provenance only.

