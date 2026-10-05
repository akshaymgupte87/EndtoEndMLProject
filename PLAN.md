# Project status

The implementation is present for the local recommender and the small
deployment-learning path. Remaining work is runtime verification in services
that are not available locally; it is not a list of features to build.

## Implemented

- PySpark review cleanup, sparse-entity filtering, chronological train/
  validation/test splits, and training-only user/item mappings.
- Popularity and Spark implicit ALS baselines; PyTorch two-tower training,
  evaluation, cohort comparison, experiment tracking, and gated local model
  registration.
- A bounded batch command that trains, evaluates, and exports a serving bundle.
- FastAPI recommendations with health/readiness/model-info/metrics endpoints,
  optional Redis caching and fallback, and S3 bundle loading.
- Docker/Compose configuration, Prometheus and OpenTelemetry configuration,
  a Kubernetes manifest, one Redis outage exercise, a manual Airflow DAG, and
  Kafka event producer/consumer examples.
- A minimal Terraform ECS/Fargate deployment using ECR, S3, IAM, CloudWatch,
  and the existing default VPC. No secrets are required by the API.

## Verified

- The full local test suite last passed **92 tests**. Test commands, per-epoch
  observations, ranking metrics, and limitations are in the
  [test report](docs/two_tower_test_report.md).
- The batch workflow passed a 500-user, two-epoch smoke run. This checks the
  workflow, not model quality or 17.4M-row scaling.
- Terraform formatting and validation passed locally. AWS plan/apply has not
  run; no AWS resources have been created.

## Remaining runtime checks

Run only when the relevant local service or account is available:

1. Build and run the Docker/Compose stack; verify API requests and telemetry.
2. Apply the manifest to a local Kubernetes cluster and run the Redis outage
   exercise against the demo.
3. Run the Airflow DAG and Kafka producer/consumer with their services active.
4. Review the exact Terraform plan and cost estimate, then deploy to AWS only
   after account access and CIDR are configured; verify CloudWatch logs,
   dashboard, alarms, and teardown.
5. Measure larger data only when suitable compute and storage are available.

Detailed commands and pass criteria are in the
[deployment walkthrough](docs/deployment_walkthrough.md) and
[AWS runbook](docs/aws_deployment_runbook.md). The project keeps Kubernetes,
chaos engineering, Prometheus, and OpenTelemetry as small learning examples;
it does not add autoscaling, a microservice fleet, or automated production
promotion.
