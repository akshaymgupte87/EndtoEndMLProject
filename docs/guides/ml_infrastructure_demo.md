# ML infrastructure demonstration

Run from the repository root with project dependencies installed:

```powershell
.\venv\Scripts\python.exe infrastructure_demo.py
.\venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///artifacts/ml_infrastructure/mlflow.db
```

Unset `REDIS_URL` for this isolated benchmark. No Java process, Amazon download,
cloud credentials, or running API server is required. Use `--output` to isolate
another experiment and `--requests` to change the measured requests per mode.

The command writes `artifacts/ml_infrastructure/evidence.json`, two serving
bundles, validation reports, environment versions and Git state, and a local
SQLite MLflow database. MLflow records training configuration, split and weight
hashes, checkpoint, manifest, run report, and validation metrics. Open the
`ml-infrastructure-synthetic` experiment to inspect both repeat runs; inspect
`two-tower-recommender` for the gated registry run. The registry name is
`synthetic-infrastructure-demo`, with alias `candidate`.

## Selection and registration criteria

Checkpoint selection minimizes sampled validation BPR loss. Release selection
then compares full-catalog ranking quality against popularity calculated from
training interactions. Both rankers use the same users, validation targets,
catalog and training-seen exclusions. The gate requires at least 5% relative
NDCG@10 lift, no Recall@10 regression, finite metrics and a positive baseline
NDCG. A test report cannot authorize registration. Failure leaves the candidate
unregistered; the batch workflow separately supports popularity fallback.

The demo exercises actual registration and negative cases for test reports and
zero lift. This is a quality gate, not a complete release approval system:
report-to-checkpoint cryptographic binding, fairness, drift, security and latency
release gates are not implemented here. The existing registration helper trusts
the supplied validation report. Registration does not deploy the model.

## Reproducibility and latency evidence

Two CPU runs use seed 42, deterministic Torch algorithms and one Torch thread.
The command verifies equal split hashes, model tensor hashes and validation
reports, and reads persisted metrics back from MLflow. Tensor hashes avoid
depending on byte identity of serialized checkpoint containers. This verifies
repeatability in the recorded environment, not across arbitrary hardware or
package versions. The environment artifact records installed package versions;
the repository lockfile supplies the dependency resolution.

The dataset is generated: 96 users, 40 items, 576 training interactions, and
separate validation/test interactions. The shared demo helper also computes
test metrics for its run report; the infrastructure registration decision uses
only validation targets. Synthetic results demonstrate plumbing and are not
evidence of real recommendation quality. For the real-data multi-seed
comparison, see [the evaluation report](../research/two_tower_test_report.md)
and `src/models/summarize_comparison.py`.

Latency covers the full in-process FastAPI ASGI request through response
serialization, with concurrency one and 20 warmup requests. Each mode measures
200 successful requests by default and retains every sample. Uncached mode
clears the memory cache before each timed request; cached mode preloads all
users. It excludes model startup, sockets, TLS and proxy costs. The existing
Prometheus `recommendation_request_seconds` measures a narrower handler scope.

Observed locally on 2026-11-16 (Python 3.14.7, Windows CPU):

| Evidence | Result |
|---|---|
| Repeated dataset, weights and validation metrics | Identical; best epoch 23 |
| Validation NDCG@10, model / popularity | 0.650912 / 0.113771 |
| Validation Recall@10, model / popularity | 1.000000 / 0.270833 |
| Registration | Version 1, `candidate` alias |
| Test-split and zero-lift gate cases | Both rejected |
| Uncached p50 / p95 / p99 | 0.89 / 1.27 / 1.35 ms |
| Memory-cached p50 / p95 / p99 | 0.83 / 1.67 / 2.23 ms |

These small sequential measurements do not establish a cache speedup or a
production SLO. Measure the deployed HTTP endpoint under representative
concurrency, catalog size and cache conditions before setting a latency budget.
The run completed successfully; the existing API, registration and tracking
test selection passed 30 tests.
