# Data Scale Context

The original LendingClub accepted-loan source is described in this repository
as containing approximately 2.26 million records. That figure is source-data
context, not evidence that the current public pipeline has been executed or
benchmarked at that scale.

Raw LendingClub data is excluded from the public repository. There is no
committed full-scale run log, runtime profile, memory profile, or cost record.
Public evidence should therefore treat 2.26 million records only as the scale of
the original source.

## Planned benchmark ladder

| Stage | Intended purpose |
| --- | --- |
| 100K rows | Completed temporal correctness and calibration run |
| 500K rows | Completed medium stress run with memory capture |
| 1M rows | Completed senior scale run with memory capture |
| 2M+ rows | Run only with runtime, memory, environment, and cost evidence |

The completed 1M source-prefix stage retained 571,494 resolved outcomes. It does
not establish full-source execution. The full local source, 2M, 5M, and 10M
targets remain unclaimed.

Completed stages record row counts, split strategy, runtime, peak process-tree
memory, artifact size, ROC-AUC, PR-AUC, and Brier score in
`reports/scale_benchmark_results.json`.
