# Resource Profile

## Latest measured run

The latest reproducible run used a deterministic 100,000-row prefix sample from
the ignored local source. Target filtering retained 87,892 resolved loans,
split into 70,313 training rows and 17,579 test rows.

Measured by `scripts/train_credit_risk_model.py`:

- Preprocessing plus logistic regression fitting: 2.3392 seconds.
- Test preprocessing and probability inference: 0.0446 seconds.
- Input features: 30.
- Transformed features: 146.
- Sample mode: true.

Measured by `scripts/benchmark_inference.py` using 10,000 synthetic rows and
five end-to-end preprocessing and prediction runs:

- Median batch time: 0.0363 seconds.
- Throughput at the median batch time: approximately 275,724 rows per second.
- Total measured time across five runs: 0.1812 seconds.

These timings are local sample-scale evidence from one execution environment.
They are not full-source or deployed-service benchmarks.

## Not measured

- Peak or steady-state memory.
- Cold process startup.
- Concurrent request behavior or tail latency under load.
- Training or inference infrastructure cost.
- Full-source preprocessing and training time.

## Scaling caveats and next targets

The deterministic prefix sample may not represent later source periods. Runtime
will also vary with hardware, dependency versions, process contention, category
cardinality, and model configuration.

The next evidence targets are 100K resolved rows, 500K rows, 1M rows, and then
2M+ only with runtime, memory, environment, and cost records.
