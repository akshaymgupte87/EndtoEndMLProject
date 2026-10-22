# Product Recommendation Service

> **Portfolio summary:** I built a learning-sized recommendation system for
> ranking products from historical interaction data. My work spans Spark data
> preparation and temporal splits, popularity/ALS/two-tower experiments,
> validation-gated batch packaging, and a FastAPI serving path with optional
> Redis and local observability. Core technologies: Python, PySpark, PyTorch,
> FastAPI, MLflow, Docker Compose, Prometheus, and OpenTelemetry. The hardest
> engineering decision was keeping evaluation comparable and leakage-aware:
> training-only ID maps, shared candidates, and an explicit baseline gate.
> [Interview story](docs/guides/project_interview_story.md) ·
> [Study guide](docs/guides/two_tower_study_guide.md)

## What I built

I implemented the data and model workflow in [`src/models`](src/models),
including deterministic training-only ID mappings, negative sampling,
two-tower training/evaluation, baseline comparisons, and serving-bundle
creation. I also built the API and batch path in [`src/api`](src/api) and
[`src/pipeline`](src/pipeline), plus tests and local observability. Course-
derived and inherited material is identified in the relevant guides; this
summary describes my engineering contributions and does not claim ownership of
the underlying Amazon dataset or third-party frameworks.

## Evaluation

The source is the public Amazon Reviews 2023 Movies and TV category; it is not
included in Git. The matched experiment uses a fixed 10,000-user cohort, 110,923
training interactions, 10,000 validation targets, and a 50,507-item candidate
catalog (three seeds). On validation, two-tower averaged Recall@10 2.21% and
NDCG@10 0.011180; popularity averaged 2.09% and 0.010294. These are
independent validation labels, not a final unbiased test estimate. A later
frozen-candidate test evaluation had 9,934 eligible targets, with Recall@10
1.490% / NDCG@10 0.007616 for two-tower and 1.399% / 0.007134 for popularity;
however, earlier validation experiments informed the iteration, so this test
is a project checkpoint rather than an untouched estimate. Absolute recall is
low, tail-item results are weak, and no online business impact has been
measured. Details and limitations: [evaluation report](docs/research/two_tower_test_report.md).

## First run: synthetic demo

The interactive demo is an inspectable offline ranking experiment, not a list of
hard-coded recommendations. With Python 3.11+ and the project dependencies
installed, run:

```bash
python -m pip install -e .
python demo.py
```

The **Project overview** tab explains the service and summarizes the saved
real-data evaluation in plain language and a proportional comparison chart. The current project checkpoint records
two-tower Recall@10 1.490% / NDCG@10 0.007616 versus popularity Recall@10
1.399% / NDCG@10 0.007134, measured over 9,934 users and a 50,507-item shared
candidate catalog. This is a small offline gain: the overview says so plainly
and notes that no online business impact has been measured. Expand the metric
details to see exact definitions and the training curve. The page also gives
concrete next steps: inspect weak performance on less popular movies, tune a
stronger baseline, and evaluate with a time-based split. The report and
checkpoint are stored project artifacts, not recomputed when the page opens.
If artifacts are absent, that panel says the saved report is unavailable.

The seeded synthetic cohort has 96 users and 40 movie IDs. Each user's six
training interactions fit an ID-embedding two-tower model with pairwise BPR
loss; a separate validation interaction selects the checkpoint. The page
compares full-catalog Recall@10 and NDCG@10 against a popularity ranker computed
only from training interactions, using one held-out test item per user. It also
shows the selected user's training history, live API recommendations, dot
product scores, and descriptive same-cohort interaction support. Test targets
are excluded from training and checkpoint selection.

The generator intentionally creates four balanced genre cohorts, each with a
ten-title catalog, so users sharing a cohort provide a learnable collaborative
signal. High scores on this controlled dataset demonstrate the training and
evaluation path; they do not establish performance on real audiences.

The page separates the project overview, an example user trace, and technical
details into tabs. The example user trace shows its training history, live
recommendations, and the held-out item's rank under both approaches. In
**Technical details**, the **Run experiment** button performs a new server-side
training and evaluation run. Change the seed to change the generated
interaction split, then inspect the measured runtime, epoch loss chart and log,
run report, downloadable split CSV, model checkpoint, and artifact manifest.
The tab clearly distinguishes this live synthetic retraining from the saved
real-data evaluation, and groups the downloads under **Download run artifacts**.
The report includes the run ID, training configuration, best validation
epoch/loss, split fingerprint, and checkpoint SHA-256. The live recommendation
API switches to the new checkpoint only after the serving bundle passes its
checksum validation. It is expected to finish quickly because the example has
only 576 training pairs; the page reports the measured run time and does not
add an artificial delay.

All generated model files live in a temporary directory and are removed when
the server exits. The experiment is synthetic and demonstrates the workflow,
evaluation protocol, and serving path; its metrics are not evidence of
real-world recommendation quality. The demo does not require the Amazon
dataset, Java, a running Spark service, AWS credentials, or Docker. If the
browser does not open automatically, use `http://127.0.0.1:8766`; port 8766
must be available. The API also exposes `/health`, `/ready`, `/model-info`,
`/recommendations/{user_idx}`, and `/metrics`. Press Ctrl+C to stop. For the
real-data workflow, use the setup and dataset instructions below.

This repository is a production-oriented portfolio project for an end-to-end
product recommender. It currently contains Spark data preparation, offline
model experiments, batch artifact packaging, a FastAPI service, local
observability, and deployment examples. The next goal is to demonstrate the
controls and operational evidence that make an ML service safe to reproduce,
release, monitor, and roll back.

It is **not production-ready today**. Model quality is weak and inconsistent
across runs, training is bounded/local, the public API contract still uses an
internal user index, authentication is not implemented, and AWS has not been
deployed. Treat Compose, Kubernetes, Airflow, Kafka, and Terraform as portfolio
components with the level of verification recorded in [the project plan](PLAN.md),
not as proof of a production service.

The first priority is a trustworthy evaluation result. A matched offline
comparison showed a small two-tower gain over popularity, while a later smoke
run scored zero recall for two-tower. The sampled XGBoost result uses a
different candidate protocol and should not be compared directly with
full-catalog metrics. Results and caveats are in the
[test report](docs/research/two_tower_test_report.md).

## Project status and roadmap

[PLAN.md](PLAN.md) defines the production portfolio goal, current evidence,
phase-by-phase exit criteria, and immediate priority order. The work is
organized around reproducible ML evidence, safe serving, CI and release
rollback, measurable service objectives, and one bounded cloud deployment.
The project is complete as a portfolio when a reviewer can trace and reproduce
the full path from versioned data through a gated model release to monitored
serving and tested rollback.

## Validation results

On a shared offline validation cohort of 10,000 users and a 50,507-item
candidate catalog, the mean results across three seeds were:

| Model | Recall@10 | NDCG@10 |
|---|---:|---:|
| Popularity baseline | 2.09% | 0.010294 |
| Spark implicit ALS | 1.82% | 0.009612 |
| PyTorch two-tower | 2.21% | 0.011180 |

The two-tower model showed a small offline ranking gain over popularity in
this experiment. This is not evidence of improved conversion or retention:
there has been no online A/B test. The model also has weak rare-item coverage.

## Architecture

```text
Historical interactions → PySpark preparation and time-based splits
  → popularity / implicit ALS / two-tower experiments
  → validation and cohort evaluation → batch-exported serving bundle
  → S3 serving bundle → FastAPI recommendation endpoint → optional Redis cache
                                      ↘ optional Bedrock explanation
```

The API serves two-tower recommendations only when its validation report
passes the quality gate; otherwise the batch bundle selects the popularity
baseline. Unknown users also receive popularity results when that fallback is
packaged. The API filters training-seen items for known users and exposes
health, readiness, model information, and Prometheus metrics endpoints.
Prometheus and OpenTelemetry are configured for local Compose
observability. On AWS, ECS loads the model bundle from S3 using its task role.
The optional Bedrock explanation route uses the Bedrock Runtime Converse API.
XGBoost is an offline ranking experiment and is not used by the serving API.

## Setup

The Spark pipeline is tested in Linux/WSL with Python 3.11 and Java 21. From
the repository root:

```bash
sudo apt-get update
sudo apt-get install -y openjdk-21-jdk python3.11 python3.11-venv gzip curl
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . pytest
```

Download the Amazon Reviews 2023 Movies and TV review data. It is not stored
in this repository:

```bash
mkdir -p data/raw
curl --fail --location \
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Movies_and_TV.jsonl.gz \
  --output data/raw/Movies_and_TV.jsonl.gz
gzip --decompress --keep data/raw/Movies_and_TV.jsonl.gz
```

Configure Spark in the same shell used to run the project:

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export SPARK_HOME="$(python -c 'import pathlib,pyspark; print(pathlib.Path(pyspark.__file__).parent)')"
export PATH="$SPARK_HOME/bin:$JAVA_HOME/bin:$PATH"
export PYSPARK_PYTHON="$(command -v python)"
export PYTHONPATH="$PWD"
export SPARK_LOCAL_IP=127.0.0.1
export PYSPARK_SUBMIT_ARGS='--driver-memory 8g pyspark-shell'
```

## Run

Prepare the cleaned and time-split dataset, then create the integer mappings
used by the models:

```bash
python -m src.data.prepare \
  --input data/raw/Movies_and_TV.jsonl \
  --output data/processed/movies_tv_5m
python -m src.models.prepare_tower_data \
  --data data/processed/movies_tv_5m \
  --output data/processed/movies_tv_5m/two_tower/v1
```

Run a small batch training, evaluation, and artifact-export smoke test:

```bash
python -m src.pipeline.batch_job \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --output artifacts/batch_smoke \
  --max-users 500 --max-train-pairs 30000 --epochs 2 --seed 42
```

The smoke run validates the pipeline and serving artifact format; it is not a
model-quality or large-scale benchmark. To serve its exported model locally:

```bash
export MODEL_DIR=./artifacts/batch_smoke/serving
docker compose up --build -d
curl http://localhost:8000/ready
curl http://localhost:8000/model-info
```

The recommendation endpoint is `/recommendations/{user_idx}?limit=10`; use a
user index present in `artifacts/batch_smoke/serving/user_ids.json`. Open
Prometheus at `http://localhost:9090` to inspect API metrics. Stop the local
services with `docker compose down`.

## AWS deployment (ECS, S3, optional Bedrock)

For the current guided deployment checkpoint and copy-ready commands, see the
[AWS deployment quick start](docs/guides/aws_deployment_quickstart.md). The longer
[AWS deployment runbook](docs/guides/aws_deployment_runbook.md) covers validation,
monitoring, costs, and cleanup. Optional SageMaker MLflow experiment tracking,
SageMaker Model Registry synchronization, and Bedrock explanations are covered
in the [AWS MLflow and Bedrock guide](docs/guides/aws_mlflow_bedrock.md).
The SageMaker training-to-ECS deployment workflow is described in the
[AWS cloud ML pipeline guide](docs/guides/aws_cloud_ml_pipeline.md).

The Terraform in `deploy/aws` provisions an ECR repository, encrypted private
S3 model bucket, ECS/Fargate API, CloudWatch logs/alarms/dashboard, and a
task role scoped to the model bundle. It expects an existing default VPC with
public subnets and an immutable image already pushed to ECR. Choose a narrow
client CIDR such as your public IP with `/32`; the API currently has no
authentication, so do not expose it broadly.

The AWS foundation is already applied with `deploy_service = false`; it has no
running API task. Run the end-to-end notebook in its documented Spark
environment, use its printed bundle path only if the validation gate passes,
push the API image to ECR, and then follow the [AWS deployment quick start](docs/guides/aws_deployment_quickstart.md)
to upload the bundle and start ECS. Do not use the undertrained Airflow smoke
bundle as a production candidate.

For Bedrock explanations, confirm model availability and account permissions,
set `bedrock_model_id` to a supported regional model ID, and apply. The optional
`POST /explain/{user_idx}?limit=5` route adds cautious text to the selected
recommendations; Bedrock does not choose or reorder items. The ECS task role
gets invocation access only for the configured model. See the
[AWS MLflow and Bedrock guide](docs/guides/aws_mlflow_bedrock.md) for setup and request
examples.

The SageMaker pipeline runs the batch training and validation workflow as a
managed Training Job after encoded data is staged in S3. It promotes the
resulting gated bundle to the API's S3 key and refreshes the ECS/Fargate
FastAPI service. The API continues to serve recommendations from that bundle;
a SageMaker inference endpoint is not required. The pipeline infrastructure
and images are created only after applying the Terraform setup in the
[cloud ML pipeline guide](docs/guides/aws_cloud_ml_pipeline.md). An opt-in managed
MLflow server can separately track experiments and synchronize model registry
entries. Review the guide and current AWS costs before running cloud jobs.

On Windows, run the Spark tests inside the configured WSL Ubuntu environment.
This avoids starting Spark with the Windows Python/Java installation. The
launcher selects the Python 3.11, PySpark, and Java 21 paths documented below,
then runs the complete suite without skipping tests:

```powershell
.\scripts\test-wsl.ps1 -q
```

Pass normal pytest options or test paths after the script name, for example
`.\scripts\test-wsl.ps1 tests/test_prepare.py -q`. The launcher expects
`/home/akshay/miniconda3/bin/python`, PySpark beneath that environment, and
Java 21 at `/usr/lib/jvm/java-21-openjdk-amd64`. If your WSL paths differ,
adjust the constants near the top of `scripts/test-wsl.ps1`.

From a Linux or WSL shell, activate the environment described above and use
the portable launcher instead:

```bash
./scripts/test-linux.sh -q
```

The Linux launcher uses `PYSPARK_PYTHON` when set, otherwise `python3` (then
`python`). It checks that the selected interpreter is Python 3.11+ with
PySpark 4.x, locates Java from `JAVA_HOME` or `PATH`, requires Java 17+
(Java 21 is the tested version), and verifies that `SPARK_HOME` contains a
Spark distribution. By default, `SPARK_HOME` comes from that interpreter's
PySpark package, so PySpark, its Spark jars, and Python workers stay aligned.
Set `SPARK_HOME` only when intentionally using another compatible Spark
distribution. `SPARK_LOCAL_IP`, `PYSPARK_SUBMIT_ARGS`, and
`PYTEST_CACHE_DIR` can also be overridden. Missing or incompatible setup
fails with a preflight message; pytest failures and exit codes are preserved.
