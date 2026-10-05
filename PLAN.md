# Project status

The implementation is present for the local recommender and the small
deployment-learning path. Remaining work is runtime verification in services
that are not available locally; it is not a list of features to build.

## Implemented

- PySpark review cleanup, sparse-entity filtering, chronological train/
  validation/test splits, and training-only user/item mappings.
- Popularity and Spark implicit ALS baselines; PyTorch two-tower training,
  evaluation, cohort comparison, experiment tracking, and gated local model
  registration. Pandas builds bounded candidate tables with training-only
  history affinity; an XGBoost ranker compares with two-tower scoring on
  sampled validation candidates.
- A bounded batch command that trains, evaluates, and exports a serving bundle.
- FastAPI recommendations with health/readiness/model-info/metrics endpoints,
  optional Redis caching and fallback, and S3 bundle loading.
- Docker/Compose configuration, Prometheus and OpenTelemetry configuration,
  a Kubernetes manifest, one Redis outage exercise, a manual Airflow DAG, and
  Kafka event producer/consumer examples.
- A minimal Terraform ECS/Fargate deployment using ECR, S3, IAM, CloudWatch,
  and the existing default VPC. No secrets are required by the API.

## Verified

- The full local test suite last passed **100 tests**. Test commands, per-epoch
  observations, ranking metrics, and limitations are in the
  [test report](docs/two_tower_test_report.md).
- A local Docker Compose end-to-end run passed from batch artifact export
  through API serving, Redis cache/fallback/recovery, Prometheus scraping, and
  OpenTelemetry trace export. After adding 200 ms Redis timeouts, a rebuilt
  API returned recommendations in 131 ms with Redis down. Five subsequent
  uncached requests with Redis healthy had wall-clock latency 7.77–21.78 ms
  (median 9.22 ms); these are sequential local smoke requests, not an SLO or
  load test. The smoke model did not beat
  popularity and the serving image is 1.57 GB;
  these limits are recorded in the
  [deployment walkthrough](docs/deployment_walkthrough.md#end-to-end-run-record-2026-10-05).
- XGBoost ranked 10,000 validation users' sampled 100-item candidate lists;
  Recall@10 was 52.59% versus 42.25% for the two-tower on those same lists.
  This sampled-candidate result is separate from full-catalog metrics and
  does not establish that the API should serve XGBoost.
- The batch workflow passed a 500-user, two-epoch smoke run. This checks the
  workflow, not model quality or 17.4M-row scaling.
- A fresh 500-user bundle contains a SHA-256 model version, pipeline run ID,
  training/evaluation MLflow run IDs, timestamp, and seed; the live
  `/model-info` response returned those values.
- Terraform formatting and validation passed locally. AWS plan/apply has not
  run; no AWS resources have been created.
- Kafka producer/consumer passed a one-event local Compose check; the JSONL
  evidence is in `docs/run_evidence/kafka_event_demo_2026-10-05.jsonl`.
- Airflow started locally, discovered the batch DAG, and reported no import
  errors. Triggering its training task was blocked by Windows denying WSL
  access (`Wsl/Service/E_ACCESSDENIED`), so no Airflow training run is claimed.

## Remaining runtime checks

Run only when the relevant local service or account is available:

1. Run the Airflow DAG task and apply the Kubernetes manifest to a local
  cluster, then exercise Redis failure in that deployment.
2. Review the exact Terraform plan and cost estimate, then deploy to AWS only
  after account access and CIDR are configured; verify CloudWatch logs,
  dashboard, alarms, and teardown.
3. Measure larger data only when suitable compute and storage are available.

Detailed commands and pass criteria are in the
[deployment walkthrough](docs/deployment_walkthrough.md) and
[AWS runbook](docs/aws_deployment_runbook.md). The project keeps Kubernetes,
chaos engineering, Prometheus, and OpenTelemetry as small learning examples;
it does not add autoscaling, a microservice fleet, or automated production
promotion.

The evidence-based improvement sequence—serving reliability, stronger
validation, then one measured AWS deployment or scale run—is explained in the
[interview story and improvement plan](docs/project_interview_story.md#practical-improvement-plan).
