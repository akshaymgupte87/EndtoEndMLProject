# Production portfolio project plan

## Product goal

Build a credible end-to-end recommendation service that can be reproduced,
deployed, monitored, and safely updated. “Production grade” is the target to
demonstrate with evidence; this personal portfolio project is not a claim of
operating a commercial production service.

The current project is a strong prototype with useful model experiments,
batch packaging, an API, local observability, and deployment examples. It is
not production-ready yet. Its two-tower quality is weak and inconsistent
across runs, the training path is bounded/local, user identity is an internal
index, and the API has no authentication. The AWS foundation is now deployed,
but it has no validated model bundle or immutable API image and the ECS service
is still configured for zero running tasks.

## Current evidence

- Spark data preparation, temporal splits, training-only ID mappings,
  popularity/ALS/two-tower models, bounded XGBoost experiments, MLflow
  tracking, and a validation-gated local registry exist in code.
- A batch command exports checksummed serving bundles with run lineage.
- FastAPI serves recommendations with optional Redis, metrics, traces, and
  S3 bundle loading. Compose, a local Kubernetes deployment, and an Airflow
  batch run have been exercised locally.
- A matched offline comparison reported two-tower Recall@10 of 2.21% versus
  popularity at 2.09% across three seeds. A later bounded Airflow smoke run
  scored zero two-tower Recall@10/20 while popularity scored 0.014/0.018.
  These results make model validation and quality the first priority.
- The reported XGBoost lift is on sampled candidate lists; it is not directly
  comparable with full-catalog metrics and is not the API's serving model.
- The base AWS stack has been applied in `us-west-2`: ECR, private S3, IAM,
  ECS cluster/task/service definitions, and CloudWatch dashboard/alarms. The
  ECS service remains at zero tasks. There is no deployed API, load test,
  online experiment, or evidence at the advertised large-data scale.
- Terraform and notebook code now provide opt-in SageMaker managed MLflow
  tracking/Model Registry synchronization and Bedrock explanations. Neither
  integration has been enabled or exercised in AWS yet.

See [the test report](docs/research/two_tower_test_report.md), [deployment walkthrough](docs/guides/deployment_walkthrough.md), and [AWS runbook](docs/guides/aws_deployment_runbook.md) for dated details. Some older sections still describe Airflow and Kubernetes as unverified; treat the dated October 14, 2026 entries in this file as newer evidence and reconcile those historical notes as a documentation task.

## Focused delivery scope

Keep this portfolio centered on one end-to-end recommendation path. Kafka,
Bedrock, SageMaker, Kubernetes, and multiple orchestration systems remain
optional demos; they are not prerequisites for calling the core project
production-oriented.

### 1. Trust the model decision

- The October 14 smoke result has 500/500 validation targets evaluated across
  the full 137,070-item catalog. Popularity retrieved 7 targets at 10 and 9 at
  20; two-tower retrieved none. The run trained only 5,713 pairs for two
  epochs, so this indicates an undertrained smoke candidate, not an evaluator
  defect or a final model-quality result.
- Keep the existing fixed-cohort validation protocol and popularity baseline.
  Diagnose sample/training limits, then require the existing registration gate
  before packaging a two-tower candidate. Keep the temporal test split for one
  final report after choices are frozen.
- Use popularity for unknown users and whenever a learned candidate fails its
  validation gate. The batch artifact records its serving policy so a failed
  learned candidate can produce a baseline-only serving bundle without being
  represented as a passing model.

**Exit evidence:** a documented validation report, a passing quality gate for
any learned candidate, and a working baseline fallback.

### 2. Make the existing path easy to reproduce

- Record dataset/run inputs, seed, code revision, metrics, and model checksum
  in the existing manifests. Preserve dated runs and avoid replacing prior
  artifacts on failure.
- Add CI for the existing unit suite and one small API/bundle smoke check.
- Keep the local Spark setup documented; do not claim large-data scale until
  it has been measured.

**Exit evidence:** a clean checkout can run the documented checks and
reproduce a bundle and validation report.

### 3. Make serving safe enough for a demo deployment

- Keep this as a restricted portfolio API for now. Do not expose it publicly
  without authentication and TLS. Replace internal user indices with an
  external identity only if the project becomes a real client-facing service.
- Verify complete bundle checksums, expose a clear unknown-user fallback,
  retain Redis outage behavior, and test those request paths.

**Exit evidence:** passing tests for bundle corruption, known/unknown users,
input validation, and cache outage.

### 4. Deploy and measure once

- Use the existing small AWS ECS/Fargate + S3 path after the evaluation gate
  and API checks pass. Review the exact plan, CIDR, IAM, and estimated cost.
- Record basic request success, latency under a declared small workload,
  resource use, logs, and cost. Exercise rollback to the prior image/model,
  then tear down disposable resources.

**Exit evidence:** a dated deployment report with endpoint checks, basic
measurements, rollback, cost, and teardown.

## Immediate priority order

1. Use the existing quality gate in `src.pipeline.batch_job`; confirm the
   Airflow smoke candidate is marked `popularity` and never served as a
   passing two-tower model.
2. Add the smallest CI path for the current tests and API smoke behavior.
3. Refresh stale status docs and record a reproducible validation/bundle run.
4. Keep the applied AWS service at zero tasks until a gated candidate bundle
   and immutable API image are uploaded. Enable optional SageMaker/Bedrock
   features only after their access and cost checks.

## Definition of done for the portfolio

The portfolio is ready to present when a reviewer can follow one documented
command or CI workflow from a versioned input through data validation,
training, evaluation, gated artifact creation, deployment, and monitored API
request; reproduce the evidence; inspect tradeoffs and limitations; and see a
tested rollback. Until all of that evidence exists, describe the repository
as a production-oriented portfolio project in progress, not a production
system.
