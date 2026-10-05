# Product Recommendation Service

This project builds a recommendation system from historical product
interactions and serves recommendations through an HTTP API. The pipeline
uses PySpark for data preparation and baselines, PyTorch for two-tower
retrieval, and FastAPI for serving. Redis can cache responses; Docker Compose
includes local Prometheus metrics and OpenTelemetry request tracing.

The repository also contains a bounded Pandas/XGBoost ranking experiment, a
Kafka event example, an Airflow batch DAG, a Kubernetes manifest, and a small
Terraform path for deploying the API to AWS ECS/Fargate. These are learning
and demonstration components; the system has not been deployed to AWS or
validated at production scale.

## Results

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
  → FastAPI recommendation endpoint → optional Redis cache
```

The API serves two-tower recommendations, filters items already seen by the
user, and exposes health, readiness, model information, and Prometheus metrics
endpoints. Prometheus and OpenTelemetry are configured for local Compose
observability. XGBoost is an offline ranking experiment and is not used by the
serving API.

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

Run the test suite in the configured Linux/WSL environment:

```bash
python -m pytest -q
```
