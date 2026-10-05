# Amazon Reviews Recommender

A learning-sized end-to-end recommender built with PySpark, implicit ALS, a
PyTorch two-tower model, MLflow, a FastAPI recommendation service, and a small
AWS deployment path.

## Start here

1. [Study guide](docs/two_tower_study_guide.md): setup, run order, code flow,
   concepts, and links to implementations and tests.
2. [Test report](docs/two_tower_test_report.md): test commands, training logs,
   offline metrics, and comparison results.
3. [PySpark notes](docs/pyspark_recommender_notes.md): data preparation and
   Spark execution concepts.
4. [Deployment walkthrough](docs/deployment_walkthrough.md): batch job, API,
   local observability, events, and Kubernetes.
5. [AWS runbook](docs/aws_deployment_runbook.md): Docker/ECR, ECS/Fargate,
   CloudWatch, optional SageMaker MLflow experiment, and cleanup.
6. [Project status](PLAN.md): implemented features and remaining verification.

## Quick checks

From the project root in the configured WSL environment:

```bash
uv sync --dev
uv run pytest
```

Install the optional Airflow package only when running its DAG:
`uv sync --extra airflow --dev`.

The full local test suite last passed 92 tests. Docker, Kubernetes, Airflow,
Kafka, AWS, and the 17.4M-row workload have not been verified end to end; see
[PLAN.md](PLAN.md) for the remaining checks. Model results and their limits are
recorded in the [test report](docs/two_tower_test_report.md).

The AWS demo uses ECS/Fargate, ECR, S3, IAM, and CloudWatch, reuses the default
VPC, and serves non-sensitive demo data over HTTP. It does not use Lambda,
Route 53, an ALB, a custom VPC, Secrets Manager, or Certificate Manager.
