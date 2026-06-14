# Batch Benchmark Report

| Rows | Status | p50 seconds | p95 seconds | Rows/sec | Peak RSS MB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10,000 | success | 0.028259 | 0.031786 | 353864.55 | 156.97 |
| 100,000 | success | 0.188801 | 0.191193 | 529658.21 | 178.43 |
| 500,000 | success | 1.285682 | 1.807288 | 388898.65 | 264.12 |

Benchmarks include preprocessing and prediction with generated values drawn from fitted categorical vocabularies. They are local batch measurements, not deployed-service claims.
