# Feature Availability Policy

## Prediction moment

The intended prediction moment is application-time, before a lending decision
and before repayment outcomes exist. Features must be available from the
application, credit-file snapshot, or policy context at that moment.

## Current model features

The CR-3 model uses requested loan terms, employment and housing attributes,
income verification, stated purpose, geography, debt-to-income measures,
credit-file counts, balances, utilization, FICO ranges, and application type.

The following fields are excluded from the model:

- `grade` and `sub_grade`, because they are LendingClub underwriting outputs.
- `int_rate`, because it is a pricing decision rather than a pure application
  input.
- `issue_d`, which is used only for temporal splitting.
- Recoveries, collection recovery fees, total payments, principal and interest
  received, last payment amount, outstanding principal, settlement fields,
  hardship status, debt settlement flags, and similar post-outcome fields.

`grade`, `sub_grade`, and `int_rate` may be retained for diagnostic segment
analysis, but they are not model inputs. A future ablation can compare
application-only performance with a clearly labeled post-underwriting scoring
scenario.

Feature availability must be reviewed against the actual lending workflow,
data contracts, legal obligations, and policy ownership before operational use.
