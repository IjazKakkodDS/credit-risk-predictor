# Scale Benchmark Report

| Requested rows | Status | Usable rows | Training sec | Total sec | Peak RSS MB | ROC-AUC | PR-AUC | Calibrated Brier |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100000 | success | 87,892 | 1.0432 | 2.9906 | 505.54 | 0.7322 | 0.4437 | 0.1470 |
| 500000 | success | 391,168 | 12.0387 | 20.6102 | 1790.68 | 0.7195 | 0.3980 | 0.1453 |
| 1000000 | success | 571,494 | 17.2885 | 32.3325 | 3241.73 | 0.6955 | 0.3727 | 0.1588 |

Only successful rows are scale evidence. Full-source, 2M, 5M, and 10M execution remain unclaimed unless explicitly present above.
