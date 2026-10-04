# Project Plan

This roadmap records planned work. Implement it incrementally and update the
plan as milestones are completed.

## Data and modeling

- Dataset: Amazon Reviews 2023, starting with a manageable sample and scaling
  to millions of interactions.
- Data engineering: PySpark ingestion, validation, cleaning, filtering,
  feature and interaction preparation, and train/validation/test splits.
- Baselines and models: popularity baseline, ALS/collaborative filtering, and
  a PyTorch two-tower model with embeddings and negative sampling.
- Evaluation: Recall@K, Precision@K, and NDCG@K overall and across active,
  medium-history, sparse, cold-start, popularity-heavy, and long-history users.
- Experiment lifecycle: MLflow parameters, metrics, artifacts, model
  registration/versioning, and a quality gate that can reject a worse model.
- Findings: `experiment_findings.md` with 3–5 actual measured findings.
- Learning lab: four notebooks covering PySpark, general ML, recommender
  experiments, and two-tower experiments.

## Serving and operations

- Orchestration: Airflow pipeline for ingest → prepare → train → evaluate →
  quality gate → register → publish.
- API and packaging: FastAPI, Redis caching, and Docker.
- Operational endpoints: `/health`, `/ready`, `/metrics`, and `/model-info`.
- Metrics and observability: Prometheus application metrics, structured logs
  with request/trace ID, model version, latency, cache hit, and candidate
  count; OpenTelemetry tracing, preferably exported to AWS X-Ray.
- Streaming: lightweight Kafka click/view producer and consumer, with events
  feeding a future Airflow retraining run.

## AWS deployment

- AWS is the only cloud target.
- Primary deployment: Terraform-managed ECR and infrastructure, with
  ALB → ECS Fargate → FastAPI and CloudWatch monitoring/logging.
- Kubernetes manifests are secondary deployment evidence; ECS remains primary.

## Chaos and resilience lab

- Stop Redis and verify cache dependency failure handling.
- Inject slow inference and measure timeout behavior.
- Simulate model and other dependency failures.
- Verify retries, fallbacks, and graceful service degradation.

## Security hardening

- Apply least-privilege IAM.
- Store sensitive configuration in AWS Secrets Manager or Systems Manager
  Parameter Store; keep secrets out of source and container images.
- Use private networking and narrowly scoped security groups.
- Scan dependencies and container images.
- Add API authentication where appropriate.
- Document threat boundaries and security assumptions.
