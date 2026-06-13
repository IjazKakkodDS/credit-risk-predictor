# Model Validation Evidence

## Run summary

- Run type: sample-scale reproducible run.
- Source sample: first 100,000 rows from the ignored local LendingClub CSV.
- Rows used after resolved-outcome filtering: 87,892.
- Training rows: 70,313.
- Test rows: 17,579.
- Positive-class rate: 20.03%.
- Model: logistic regression with balanced class weights.
- ROC-AUC: 0.7359.
- PR-AUC: 0.4209.
- Precision at threshold 0.50: 0.3340.
- Recall at threshold 0.50: 0.6705.
- F1 at threshold 0.50: 0.4459.

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

The highest observed F1 on the threshold grid occurred at 0.55. This is
analytical evidence, not a business-optimal threshold.

The calibration curve shows predicted probabilities above observed event rates
across the evaluated bins. Balanced class weighting improves adverse-class
attention but the current probabilities should be recalibrated before any
probability-based decision policy is considered.

This is reproducible validation evidence for the committed pipeline. It is not
a production deployment claim.
