# From batch model to deployment

This is the implementation guide for the deployment half of the project.
It uses one small serving model, one API image, and one AWS path. Read
[PLAN.md](../PLAN.md) for the full scope and
[the concept map](two_tower_study_guide.md#concept-map-what-this-project-teaches-so-far)
for what each technology teaches.
For detailed AWS deployment, SageMaker MLflow, CloudWatch, and cleanup
instructions, use the [AWS deployment runbook](aws_deployment_runbook.md).

## What is implemented

- `src.pipeline.batch_job` runs existing two-tower training and validation,
  exports item vectors, source item IDs, the model user vocabulary, seen-item
  histories, checksums, and a ZIP bundle.
- `src.models.register_model` enforces the validation gate and registers an
  approved checkpoint in the local MLflow registry. The frozen checkpoint is
  registered as version 2 with alias `candidate`.
- `src.api.app` serves top-N recommendations for a mapped `user_idx`, hides
  that user's training items, and exposes `/health`, `/ready`, `/model-info`,
  and Prometheus `/metrics`. Redis is optional; API requests fall back to
  model scoring when Redis is down. Model bundles can be local or downloaded
  from S3 at startup.
- `Dockerfile` and `compose.yaml` define the API, Redis, Prometheus, and an
  OpenTelemetry Collector stack. The stack was built and exercised locally on
  2026-10-05; the full test record and limits appear below.
- `deploy/kubernetes/recommender.yaml` provides a small Deployment and
  Service. `deploy/aws/` contains a Terraform ECS Fargate path.
- `deploy/airflow/dags/recommender_batch.py` is a manually triggered DAG for
  the batch job. `src.events` sends and consumes validated Kafka click/view
  events as JSON Lines.
- `deploy/chaos/cache_outage.ps1` exercises API fallback with Redis stopped.

## Where experiment tracking should live

Keep the current local MLflow setup as the default while learning and testing.
It uses `sqlite:///mlflow.db`, which is easy to inspect and costs nothing to
provision, but the database and local artifact files are machine-local. It is
not a shared or durable production tracking service.

When we deliberately run the AWS deployment experiment, use **Amazon
SageMaker AI managed MLflow** as the remote tracking server, not Amazon
Bedrock. SageMaker's tracking server accepts the MLflow client, stores run
metadata in a managed backend, and uses an S3 bucket in our account for
artifacts. A local client connects with the tracking server ARN and
`sagemaker-mlflow` plugin, authenticated using AWS IAM/SigV4. SageMaker Model
Registry integration is a possible later registry step; it is not needed to
serve this project from ECS.

Amazon Bedrock is not a training experiment tracker for this ID-embedding
recommender. Its evaluation jobs are aimed at foundation models and knowledge
bases, while its invocation logging and CloudWatch metrics apply when an
application calls Bedrock Runtime. Consider Bedrock only if a later project
version adds a foundation-model feature such as generating explanations; then
Bedrock's evaluation and invocation observability would cover that feature,
while MLflow would still record recommender training and retrieval metrics.

Use a staged tracking experiment:

1. Continue local experiments in SQLite; keep recording params, per-epoch
   BPR losses, offline ranking metrics, reports, and checkpoints as today.
2. Before creating a managed server, check account access, IAM permissions,
   supported region, server version/client compatibility, S3 artifact bucket,
   and current SageMaker pricing. Managed tracking requires AWS resources and
   artifacts in S3. AWS documents a 200 MB MLflow model-download limit, so
   measure the project model/artifact sizes and test upload/download before
   moving large artifacts; our smoke bundle was 16.9 MB, but this does not
   establish all future artifacts fit.
3. Create a small SageMaker MLflow Tracking Server and S3 artifact location
   only for the experiment. Install a compatible `sagemaker-mlflow` client in
   the WSL environment, connect by tracking-server ARN, and run one bounded
   training/evaluation pair. Record the same seed, data/cohort, parameters,
   metrics, and artifact checksums as the local run.
4. Compare run metadata, epoch metrics, artifact upload/download, registry
   behavior if enabled, and cleanup. Then stop/delete the tracking resources
   and remove disposable S3 artifacts if the account is charged for them.
5. Keep local tracking available as a fallback. Do not switch the default
   tracking URI for all future runs until the IAM, artifact, and cleanup flow
   has passed and its ongoing cost is understood.

The decision is therefore **local MLflow now; SageMaker managed MLflow as a
small, optional AWS experiment later; Bedrock only if we add a Bedrock-backed
feature**. This preserves the project’s simple model-learning path while
teaching the managed AWS alternative in the deployment phase.

## Run the bounded batch job

Use the WSL Spark/Python/Java environment described in the study guide. This
example trains a small smoke model. It checks plumbing and artifact generation;
it is not a quality run. The current prepared mapped dataset must already
exist. The recorded smoke run succeeded; see its results at the end of this
document.

```bash
python -m src.pipeline.batch_job \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --output artifacts/batch_smoke \
  --max-users 500 --max-train-pairs 30000 --epochs 2 --seed 42
```

For a comparison-style run, pass the shared cohort and candidate files and
use the intended user/interaction/epoch limits:

```bash
python -m src.pipeline.batch_job \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --output artifacts/batch_run \
  --user-ids-file artifacts/comparisons/movies_tv_5m/cohort_v1/cohort.json \
  --candidate-items artifacts/comparisons/movies_tv_5m/cohort_v1/candidate_items.json \
  --max-users 10000 --max-train-pairs 200000 --epochs 10 --seed 42
```

The output contains `best_model.pt`, `history.json`, `validation_metrics.json`,
`batch_manifest.json`, and `model_bundle.zip`. Reuse a new/empty output
directory for each run. Store the ZIP in an access-controlled model bucket for
AWS; the ECS task reads it using its task role.

### Register the gated candidate

Registration reads the validation JSON and refuses a test report or a
checkpoint that misses the 5% NDCG@10 lift / non-regressing Recall@10 gate.
After batch training and validation:

```bash
python -m src.models.register_model \
  --checkpoint artifacts/batch_run/best_model.pt \
  --validation-report artifacts/batch_run/validation_metrics.json \
  --registered-name amazon-reviews-two-tower
```

MLflow records a version and moves the `candidate` alias to it. The project
does not automatically mark a model as production. The PyTorch flavor stores
the project-generated model using Python serialization; only use checkpoints
and registry entries created from trusted project artifacts.

## Run the API and local observability

Keep two kinds of records distinct. MLflow tracks offline training and
evaluation (configuration, epoch BPR losses, ranking metrics, and artifacts)
in local SQLite `mlflow.db`; model registration adds a validation gate result
and registry version. The API does not send online requests or traces to
MLflow. Runtime observability is separate: Prometheus scrapes API
counters/histograms, OpenTelemetry sends FastAPI request traces to the
Collector, and container output is visible in Docker logs. The Compose
Collector's debug exporter writes traces to its log; it does not persist or
send them to a hosted backend. Terraform sends ECS container logs to CloudWatch
Logs, but does not deploy Prometheus or an OpenTelemetry Collector in AWS.

Use this sequence for one reviewable deployment experiment:

1. **Identify the model.** Record the SHA-256 model version, pipeline run ID,
   training/evaluation MLflow run IDs, training timestamp, seed, report path,
   dataset/cohort, code revision, and serving-bundle checksums. The batch
   manifest and `/model-info` now include the first five lineage fields.
2. **Start the local stack.** Set `MODEL_DIR` to the bundle's `serving`
   directory, run `docker compose up --build`, wait for `/ready`, and check
   `/model-info` against the expected dimensions. Save `docker compose ps` and
   startup logs as evidence.
3. **Send a small diagnostic workload.** Request a known user; repeat the
   same user/limit to exercise Redis; request an unknown user (expected 404);
   then request a different limit to force scoring. Record statuses and
   elapsed times. In Prometheus confirm target UP and inspect
   `recommendation_requests_total`, `recommendation_request_seconds`,
   `recommendation_cache_hits_total`, and
   `recommendation_cache_errors_total`.
4. **Inspect the trace.** Find the FastAPI server span in
   `docker compose logs otel-collector`; confirm a request span arrives and
   inspect its duration/status. Instrumentation is automatic FastAPI HTTP
   tracing only: there are no child spans around model scoring or Redis, so
   traces cannot yet separate those internal timings.
5. **Test one cache outage.** Stop Redis and use an uncached request. Confirm
   recommendations still return and the cache-error counter increases.
   Restart Redis and repeat; verify Redis cache hits resume. Save before/after
   metric snapshots and API/Collector logs.
6. **Write a dated experiment record.** Include model/version/run IDs, image
   tag, configuration, workload, HTTP outcomes, counts/latency, cache
   observations, trace result, outage/recovery result, and limitations. Link
   it to the MLflow run and validation report. Treat this as a wiring and
   recovery check, not a load or availability benchmark.

**Local pass criteria:** the API is ready and serves the identified artifact;
Prometheus scrapes it; success and unknown-user requests affect the expected
counters; an HTTP trace arrives at the Collector; Redis outage does not break
an uncached recommendation; the error counter rises during outage and Redis
cache hits return after recovery. Fix any missing signal before attempting
AWS.

## Kubernetes learning deployment

The manifest expects the API image to be built locally and the lineage-bearing
serving bundle at `/models` inside the node. For kind, load the image and mount
the model bundle directory to the node before applying the manifest. A plain
`kubectl apply` on a fresh cluster will not work until the hostPath model
directory exists on that node. Kubernetes has not been run in this session;
the runtime test still needs to verify readiness, a recommendation, and
Redis failure behavior.

Build the same image and start a local cluster (kind must be installed):

```powershell
docker build -t endtoendmlproject:local .
kind create cluster
kind load docker-image endtoendmlproject:local
docker cp .\artifacts\e2e_lineage_2026-10-05\serving kind-control-plane:/models
kubectl apply -f deploy/kubernetes/recommender.yaml
kubectl port-forward service/recommender-api 8000:8000
```

The local kind node must have the serving files at `/models`, because the
manifest uses a node `hostPath`. For a different local cluster, copy the
serving directory into the node path or replace it with a PersistentVolume.
Delete the Deployment, Service, and kind cluster when finished.

## AWS deployment with Terraform

The AWS configuration is the minimal learning deployment: ECS Fargate, ECR,
S3, IAM, and CloudWatch. It uses the account's existing default VPC without
creating VPC resources, runs one task with a public IP, and restricts ingress
to the supplied client CIDR. It uses direct HTTP on port 8000, so it is for
non-sensitive demo data. It has no Route 53, ALB, ACM, Secrets Manager,
application authentication, NAT Gateway, autoscaling, or remote Terraform
state. The task's public IPv4 and Fargate runtime still incur charges.

Follow the step-by-step [AWS deployment runbook](aws_deployment_runbook.md) for
AWS SSO setup, default-VPC check, client CIDR, ECR/S3 bootstrap, image push,
cost estimate, plan review, task-IP discovery, CloudWatch checks, and cleanup.
Terraform has not been applied; the AWS plan requires Docker, an authenticated
profile, and an account default VPC with public subnets. Fargate is selected
because the account's Lambda memory limit is 512 MB.

### End-to-end run record (2026-10-05)

This run tested the actual prepared data through batch training, validation,
serving-artifact export, Docker image build, FastAPI, Redis, Prometheus, and the
OpenTelemetry Collector. It used the WSL Python 3.11 / Java 21 / Spark 4.2
environment and the local Docker engine. It did not apply Terraform or deploy
to AWS.

The batch command used 500 users, at most 30,000 training pairs, two epochs,
seed 42, and the prepared `movies_tv_5m` dataset. It completed successfully:
epoch 1 training/validation BPR loss was 0.693139 / 0.693128; epoch 2 was
0.692850 / 0.693125, so epoch 2 was selected. It evaluated 500 validation
users against all 137,070 catalog items and wrote a checkpoint, validation
report, serving directory, and `model_bundle.zip` under
`artifacts/e2e_run_2026-10-05/`. The two-tower model got 0% Recall@10 and
Recall@20; popularity on the same users got 1.4% and 1.8%. This passed the
batch/artifact plumbing check, not a model-quality gate.

The API image initially measured 3.49 GB because the API imported the training
module to load a checkpoint, pulling Spark and MLflow into serving. Checkpoint
loading now lives in `src.models.two_tower`, which lets the API avoid those
training-only dependencies. The Dockerfile installs CPU-only PyTorch; the
rebuilt image measured 1.57 GB. The image built and Compose started healthy
API, Redis, Prometheus, and Collector containers.

Serving checks passed: `/ready` returned `ready`; `/model-info` reported 500
users, 137,070 items, and 16-dimensional embeddings; a known user received
five recommendations; a user absent from the model vocabulary received 404.
The five recommendations had no overlap with that user's five training-seen
items. Prometheus reported the `api:8000` target `up`, and its counters showed
successful/unknown-user requests and Redis cache hits. Collector logs showed
OTLP request spans.

For the first cache-outage check, Redis was stopped and a fresh user/limit
request still returned HTTP 200. `recommendation_cache_errors_total` rose to 2
(read and write failures), but that uncached request took about 7.94 seconds.
Redis was restarted; a fresh key and repeat request succeeded and Redis cache
hits resumed. In a follow-up after configuring 200 ms connect/read
timeouts and rebuilding the API, an uncached request with Redis stopped
returned HTTP 200 and seven items in 131 ms. This is one local smoke
measurement, not an SLO or load test. Five subsequent uncached requests with
limits 9–13 also returned HTTP 200; wall-clock latencies were 21.78, 8.81,
10.12, 9.22, and 7.77 ms (median 9.22 ms). Prometheus recorded six
recommendation requests after restart and no Redis errors in that healthy-cache
sample. These local results have no concurrency and exclude network/load
variation. The stack was left running for
inspection at `http://localhost:8000` and `http://localhost:9090`; stop it
with `docker compose down` from WSL in the repository directory.

After separating checkpoint loading, the updated full WSL pytest suite passed:
**100 passed in 25.72 seconds**, with a non-failing PySpark/Pandas
compatibility warning and a pytest cache permission warning on the mounted
Windows workspace. These synthetic behavior tests complement, but do not
replace, the real-data batch and serving checks above.

The fresh serving bundle's `/model-info` response included model version
`sha256:a0ab430675c393096fb6f7b35369c218dbc33997cc666c5c46c32d92c97d9e17`,
pipeline run `06300aa2-5c67-4b3d-85bc-38aa31c6545d`, training MLflow run
`4bc4c84a194242e1a382d95fdd002eae`, and evaluation MLflow run
`50a5461dd7444f5980583cce048abe3f`. Seed was 42; the cohort had 500 users and
137,070 items. This was a repeated 500-user, two-epoch smoke train; recall
remained zero at K=10 and K=20, while popularity had Recall@10 1.4% and
Recall@20 1.8%. This validates lineage and serving, not model quality or scale.

## Airflow and Kafka demos

Airflow is an optional dependency, kept out of the everyday project install:

```bash
uv sync --extra airflow
```

The optional `airflow` Compose profile starts Airflow 2.10.5 with the DAG
mounted from `deploy/airflow/dags/recommender_batch.py`. On 2026-10-05, the
container started, `airflow dags list` found
`recommender_batch_training`, and `airflow dags list-import-errors` returned
no errors. The training task could not be triggered in this environment:
Windows denied WSL access (`Wsl/Service/E_ACCESSDENIED`) while invoking Docker
Compose. Therefore the orchestration shell command and data/runtime mounts
still need an actual task run. The DAG uses container defaults under
`/opt/venv`, `/opt/recommender`, and `/usr/lib/jvm/java-21-openjdk-amd64`.
Override them through Airflow Variables named `recommender_project_dir`,
`recommender_data_dir`, `recommender_output_root`, `recommender_python`,
`recommender_python_dir`, `recommender_spark_home`, `recommender_java_home`,
and `recommender_spark_driver_memory` when the worker layout differs. The
base Airflow image still needs a compatible Python/Spark/Java runtime installed
or mounted; DAG discovery alone does not prove the task runtime is ready.

Start the optional single-node Kafka broker in Compose, then send one event
and consume a bounded batch:

```bash
docker compose --profile events up -d kafka
python -m src.events.produce_event --user-idx 12 --item-idx 31 --event-type click
python -m src.events.consume_events --max-events 1 --output data/events/demo.jsonl
docker compose --profile events stop kafka
```

On 2026-10-05, this producer/consumer check succeeded: one click event
(`user_idx=12`, `item_idx=31`) was sent and consumed. The exact JSONL record is
saved in [run evidence](run_evidence/kafka_event_demo_2026-10-05.jsonl). The
event file is a future training-data input only. The running recommender does
not consume Kafka synchronously.

## Tests and current verification

Run all tests in WSL using the full-suite command in the study guide. Focused
deployment tests cover the API, event schema, and model-registration gate:
`python -m pytest tests/test_api.py tests/test_event_schema.py tests/test_model_registration.py -q`.
The [test report](two_tower_test_report.md#complete-pytest-inventory-and-pass-criteria)
lists every one of the 100 collected pytest cases and its pass condition. It
also gives separate pass criteria for configuration checks, batch smoke,
offline evaluation, and external-runtime exercises; those are not included in
the pytest count.
Terraform syntax was formatted and `terraform validate` passed after provider
initialization. `docker compose config --quiet` passed. The batch smoke job
ran end-to-end on 500 users, exported a model bundle, and wrote validation
metrics; its two-epoch model scored zero hits, which is expected to be treated
only as a plumbing check. Kafka's one-event producer/consumer path and Airflow
DAG discovery were checked as recorded above; the Airflow training task,
Kubernetes deployment, and AWS deployment remain unverified.

### Batch smoke run log

On 2026-10-05, the batch command ran on the prepared dataset with seed 42,
500 users, 5,713 train pairs, 500 validation pairs, and two epochs. It selected
epoch 2 (sampled validation BPR 0.693125), wrote the validation report and a
16.9 MB `model_bundle.zip`, and exited successfully. The two-tower smoke
checkpoint had Recall@10 and Recall@20 of 0%; popularity had 1.4% and 1.8%.
This poor tiny-run ranking is expected: the run verifies training → evaluation
→ artifact export, not recommendation quality. Artifacts are ignored local
files under `artifacts/batch_smoke/`. MLflow training/evaluation run IDs were
`d3b073cf954a4d3fb1d2a5578e7c6fc2` and
`e7f09dcfe71649e69ace37d62dca607e`.
