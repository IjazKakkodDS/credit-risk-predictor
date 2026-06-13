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
| 10K rows | Smoke test for data contracts and pipeline correctness |
| 100K rows | Local end-to-end workflow check |
| 500K rows | Medium stress test with runtime and memory capture |
| 1M rows | Senior scale proof with reproducible evidence |
| 2M+ rows | Run only with runtime, memory, environment, and cost evidence |

Each completed stage should record the dataset version, row and feature counts,
hardware or execution environment, dependency versions, command used, runtime,
peak memory, output artifacts, and any failures or caveats.
