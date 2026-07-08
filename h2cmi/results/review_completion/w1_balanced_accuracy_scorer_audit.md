# W1 Balanced-Accuracy Scorer Audit

- status: PASS
- sklearn_version: `1.5.2`
- scorer: `sklearn.metrics.balanced_accuracy_score`
- absent_class_handling: `ignored_present_labels_only; when y_true has one class, the score is the recall of that present class and equals ordinary accuracy for that one-class evaluation set`

## Code Paths

- `h2cmi/run_w1_mi.py: balanced_accuracy_score(ye, pred)`
- `h2cmi/eval/p0_eval.py: _record(... balanced_accuracy_score(y, pred) ...)`
- `h2cmi/run_spdim_probe.py: _metrics(... balanced_accuracy_score(y_true, y_pred) ...)`

## Deterministic Cases

| case | y_true counts | accuracy | balanced accuracy | acc == bAcc | warnings |
|---|---:|---:|---:|---|---|
| single_class_all_correct | `[0, 4]` | 1.000000 | 1.000000 | `True` | A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels. |
| single_class_half_correct | `[0, 4]` | 0.500000 | 0.500000 | `True` | y_pred contains classes not in y_true |
| single_class_all_wrong | `[0, 4]` | 0.000000 | 0.000000 | `True` | y_pred contains classes not in y_true |
| balanced_two_class_mixed | `[2, 2]` | 0.750000 | 0.750000 | `True` | none |
| imbalanced_two_class_mixed | `[1, 3]` | 0.750000 | 0.833333 | `False` | none |

## Conclusion

For one-class `y_true`, the project scorer is numerically defined but degenerates to the present-class recall, which equals ordinary accuracy on that one-class evaluation set. This is project-consistent but not a fair two-class balanced-accuracy interpretation.
