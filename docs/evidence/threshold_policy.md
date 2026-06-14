# Threshold Policy

A default probability threshold of 0.50 is usually insufficient for credit risk
decisions. It reflects a mathematical convention, not the cost of an incorrect
decision, the prevalence of default, operational review capacity, or the risk
appetite of a specific lending context.

## Decision trade-off

Lowering the threshold generally identifies more potentially adverse cases and
raises recall, but it can also increase false positives and reduce precision.
Raising the threshold generally reduces false positives, but it can miss more
adverse cases.

In business terms:

- A false approval can expose the lender to credit loss, servicing cost, and
  concentration risk.
- A false rejection can turn away a creditworthy applicant, reduce revenue, and
  create customer and fairness concerns.

Threshold selection should therefore be cost-sensitive. A candidate threshold
should be evaluated against estimated error costs, expected class prevalence,
calibration quality, review capacity, and segment-level behavior. The selected
threshold should be approved for a defined use case rather than treated as a
universal model property.

## CR-3 methodology

CR-3 selects thresholds on the calibration split rather than the final test
split. The latest temporal run generated
[`reports/model_validation/threshold_analysis.json`](../../reports/model_validation/threshold_analysis.json).

The calibration-selected best-F1 threshold is 0.20. On the untouched test split
it produced precision 0.3432, recall 0.6843, and F1 0.4572. Probability
calibration uses Platt sigmoid fitting on the calibration split.

These are analytical thresholds for reviewer discussion. They are not a
deployment policy or an operating choice because no approved loss or
review-cost matrix is available. The 100K temporal window is also narrow, so
the selected value is not assumed stable across vintages.

## Further evidence required

Threshold readiness requires:

- Holdout probability predictions from a traceable model artifact.
- Precision, recall, specificity, and confusion matrices across candidate
  thresholds.
- Expected-cost or expected-value analysis with documented assumptions.
- Calibration evidence.
- Stability checks across relevant time periods and segments.
- A written decision rule, owner, review schedule, and rollback criteria.

The current artifact establishes reproducible threshold analysis. A validated
operating threshold still requires business loss assumptions, review-capacity
constraints, segment stability checks, ownership, monitoring, and approval.
