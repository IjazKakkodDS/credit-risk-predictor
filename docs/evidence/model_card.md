# Credit Risk Predictor Model Card

## Purpose

This model estimates the probability that a resolved LendingClub-style loan
will be charged off or default. It is a portfolio validation system, not a live
lending decision service.

## Data context

The local ignored source contains approximately 2.26 million accepted-loan
records. The canonical CR-3 validation run uses a deterministic 100,000-row
prefix, of which 87,892 rows have resolved eligible outcomes.

## Prediction moment and features

The intended prediction moment is pre-decision application time. The model uses
27 application and credit-file fields. LendingClub `grade`, `sub_grade`, and
`int_rate` are excluded from model inputs because they are underwriting or
pricing outputs. Payment, recovery, hardship, settlement, and other post-outcome
fields are also excluded.

See [Feature Availability Policy](feature_availability_policy.md).

## Target

- Positive class: `Charged Off` or `Default`
- Negative class: `Fully Paid`
- Active, late, grace-period, policy-exception, and unresolved statuses are
  excluded.

## Validation design

The CR-3 canonical run sorts resolved rows by `issue_d` and allocates 60% to
training, 20% to calibration and threshold selection, and 20% to untouched
testing. The 100K prefix spans only October through December 2015, and both the
calibration and test partitions are within December 2015. This is ordered
validation but not strong multi-vintage validation.

## Model and calibration

- Logistic regression with balanced class weights
- Median imputation and standardization for numeric fields
- Frequent imputation and one-hot encoding for categorical fields
- Platt sigmoid calibration fitted on the calibration split

Calibration reduced calibration-split Brier score from 0.2096 to 0.1470.

## Untouched test metrics

At the calibration-selected analytical threshold of 0.20:

- ROC-AUC: 0.7322
- PR-AUC: 0.4437
- Brier score: 0.1470
- Precision: 0.3432
- Recall: 0.6843
- F1: 0.4572

The threshold maximizes F1 on the calibration split. It is not an operating
threshold selected from approved business costs.

## Segment analysis

Diagnostic segment metrics were generated for grade, purpose, home ownership,
state, application type, and term where groups had at least 100 rows and both
classes. Small or one-class groups are explicitly skipped. These results are
not fairness certification.

## Scale evidence

Temporal runs completed for 100K, 500K, and 1M requested source rows. The 1M
stage retained 571,494 resolved outcomes and peaked at approximately 3.24 GB
process-tree RSS. The full 2.26M source was not executed.

## Limitations

- Narrow temporal range in the canonical 100K run
- Performance declines at larger source prefixes
- No approved lending cost matrix
- No protected-class fairness assessment
- No external validation dataset
- No drift monitoring or serving layer
- No full-source execution

## Non-goals

This repository does not claim a deployed API, automated lending decisions,
formal regulatory clearance, or external operational adoption.

## Tier A future work

- Multi-vintage out-of-time validation
- Business cost and review-capacity threshold policy
- Protected and operational segment governance
- External validation
- Full-source resource evidence if safe
- Monitoring and change-control design
