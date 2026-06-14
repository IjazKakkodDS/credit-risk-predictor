# Model Validation Evidence

## Run summary

- Run type: sample-scale temporal validation run.
- Source sample: first 100,000 rows from the ignored local LendingClub CSV.
- Rows used after resolved-outcome filtering: 87,892.
- Training rows: 52,735.
- Calibration rows: 17,578.
- Test rows: 17,579.
- Positive-class rate: 20.03%.
- Model: balanced logistic regression with Platt calibration.
- Untouched-test ROC-AUC: 0.7322.
- Untouched-test PR-AUC: 0.4437.
- Calibrated Brier score: 0.1470.
- Precision at calibration-selected threshold 0.20: 0.3432.
- Recall at threshold 0.20: 0.6843.
- F1 at threshold 0.20: 0.4572.

## Artifacts

- [Metrics](metrics.json)
- [Classification report](classification_report.json)
- [Confusion matrix](confusion_matrix.png)
- [ROC curve](roc_curve.png)
- [Precision-recall curve](pr_curve.png)
- [Calibration curve](calibration_curve.png)
- [Threshold analysis](threshold_analysis.json)
- [Training run](training_run.json)
- [Feature columns](feature_columns.json)
- [Split summary](split_summary.json)
- [Calibrated metrics](calibrated_metrics.json)
- [Provenance](provenance.json)
- [Artifact checksums](artifact_checksums.json)
- [Segment stability](segment_stability.md)

Threshold selection occurs on calibration data. The final test split remains
untouched until evaluation. The ordered 100K prefix spans only October through
December 2015, so this is not broad multi-vintage validation.

This is reproducible validation evidence for the committed pipeline. It is not
a production deployment claim.
