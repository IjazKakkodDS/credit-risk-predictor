# Resource Profile

## CR-3 measured training ladder

All CR-3 scale stages used temporal ordering, Platt calibration, and an
untouched test partition.

| Requested source rows | Usable rows | Training seconds | Total pipeline seconds | Peak process-tree RSS |
| ---: | ---: | ---: | ---: | ---: |
| 100,000 | 87,892 | 1.0432 | 2.9906 | 505.54 MB |
| 500,000 | 391,168 | 12.0387 | 20.6102 | 1,790.68 MB |
| 1,000,000 | 571,494 | 17.2885 | 32.3325 | 3,241.73 MB |

The full 2.26M source was not requested. Peak memory is measured from the
training process tree and includes Python plus native library allocations.

## CR-3 batch inference ladder

| Rows | p50 | p95 | Rows per second | Peak process RSS |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.0283 s | 0.0318 s | 353,865 | 156.97 MB |
| 100,000 | 0.1888 s | 0.1912 s | 529,658 | 178.43 MB |
| 500,000 | 1.2857 s | 1.8073 s | 388,899 | 264.12 MB |

Categorical values are drawn from fitted transformer vocabularies. These are
local batch measurements, not service-concurrency or deployment benchmarks.

## Not measured

- Cold process startup.
- Concurrent request behavior or tail latency under load.
- Training or inference infrastructure cost.
- Full-source preprocessing and training time.

## Scaling caveats and next targets

The deterministic prefix sample may not represent later source periods. Runtime
will also vary with hardware, dependency versions, process contention, category
cardinality, and model configuration.

The next scale target is the full local source only if 3.24 GB peak memory at
the 1M stage leaves sufficient safety margin. Future 2M, 5M, and 10M targets
remain planning context rather than completed evidence.

Cost per 1,000 predictions is not estimated because no deployment target,
instance class, utilization profile, or pricing basis has been selected.
