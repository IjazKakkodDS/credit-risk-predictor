# Resource Profile

The repository does not yet contain reproducible resource measurements. No
runtime, memory, inference throughput, or deployment cost figures are asserted.

## Planned measurements

Future benchmark runs should capture:

- Preprocessing runtime by dataset size.
- Model training runtime by algorithm and tuning configuration.
- Batch inference latency, including p50 and tail behavior where repeated runs
  support it.
- Peak and steady-state memory usage.
- Cost per 1,000 predictions after a deployment target and pricing basis are
  selected.

Measurements should include row count, feature count, hardware, operating
system, Python and dependency versions, process configuration, warm-up policy,
and the exact command used.

## Likely bottlenecks to investigate

- Categorical encoding, especially high-cardinality fields.
- Class imbalance handling and resampling strategies.
- Model training on large accepted-loan datasets.
- Probability calibration and threshold search.
- SHAP or other explainability workloads if they are added later.

These are investigation priorities, not measured findings. Resource claims
should be added only after repeatable benchmark evidence is committed.
