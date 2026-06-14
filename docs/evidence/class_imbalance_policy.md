# Class Imbalance Policy

## Observed distribution

The CR-3 canonical validation run loaded a deterministic 100,000-row prefix
sample from the ignored local LendingClub CSV. After retaining only resolved
outcomes, 87,892 rows remained:

- 70,288 Fully Paid rows mapped to class 0.
- 17,604 Charged Off or Default rows mapped to class 1.
- Observed positive-class rate: 20.03%.

These values come from `reports/model_validation/training_run.json` and are
sample-scale evidence, not a claim about the full source distribution.

## Current handling

Credit default datasets are typically imbalanced because resolved defaults are
less common than successful repayment. Accuracy alone can therefore obscure
poor detection of the adverse class.

The logistic regression baseline uses `class_weight="balanced"`. This
weights classes inversely to their training frequency without creating
synthetic observations. The holdout report includes:

- ROC-AUC to measure ranking across both classes.
- PR-AUC to emphasize performance on the less frequent default-risk class.
- Precision, recall, and F1 at threshold 0.50.
- Threshold-grid results for alternative operating points.

CR-3 applies Platt calibration on an independent calibration split. Calibration
reduced calibration-split Brier score from 0.2096 to 0.1470. Calibration
stability across broader vintages is still unproven.

## Decision trade-off

A false approval can expose a lender to loss, while a false rejection can deny
credit to a borrower who may repay and can reduce legitimate lending activity.
Higher recall for default risk usually increases the number of false positives.
Higher precision usually misses more adverse cases.

The current class weighting and threshold analysis support technical review,
but they do not encode an approved business cost policy. Before operational use,
the project still needs loss-given-default assumptions, false-rejection costs,
manual-review capacity, segment analysis, and governance approval.
