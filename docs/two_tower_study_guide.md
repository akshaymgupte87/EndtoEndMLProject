# Two-Tower Recommender: Study Guide

This guide follows the project from data preparation through training,
evaluation, batch packaging, and deployment code. Each step explains its data
contract, purpose, run command, and verification. Implementation code now
exists through the planned deployment demonstrations; the status map separates
that code from external runtimes that have not yet been verified.

## Goal and current position

A two-tower recommender learns one vector for a user and one vector for an
item. It scores a user/item pair by comparing those vectors. At serving time,
item vectors can be computed ahead of time; the system then ranks items by
their dot-product score with the requested user's vector.

The current model workflow is:

```text
Prepared temporal splits
    ↓
Integer user/item mappings
    ↓
Popularity baseline and implicit ALS
    ↓
Positive training pairs and negative examples
    ↓
PyTorch user and item towers
    ↓
Validation-based training and model selection
    ↓
Validation ranking comparison and cohort analysis
    ↓
Validation quality gate and one test checkpoint (test now consumed)
    ↓
Saved model and item vectors for recommendation serving
```

The mapping, sampling, model structure, training loop, validation ranking,
cohort analysis, local MLflow tracking, and implicit ALS baseline now exist.
A controlled comparison has also been completed: popularity, ALS, and
two-tower used one fixed 10,000-user cohort, the same 110,923 train rows,
validation targets, and 50,507-item candidate catalog. ALS and two-tower
were each repeated at three seeds. A fresh candidate passed the project
validation gate and received one test evaluation; that test score is a
checkpoint rather than an untouched estimate because earlier experiments
informed this iteration. The current test split is consumed. The checkpoint
is registered locally; production promotion and external serving remain
later steps. The single ordered remaining-work list is in [PLAN.md](../PLAN.md).
Results and the evaluation audit are in the
[test report](two_tower_test_report.md).

For a concise recruiter-facing project summary, architecture decisions, and
honest discussion of impact and limitations, see the
[interview story](project_interview_story.md).

## Concept map: what this project teaches so far

Use this map to distinguish three states: **implemented** means runnable
project code and/or a measured experiment exists; **partial** means the code
shows one useful piece but not a full production treatment; **planned** means
the concept is in the roadmap but there is no implementation to run yet.
Dependencies listed in `pyproject.toml` do not by themselves count as an
implemented feature.

### Recommenders and ML

| Concept | Status and plain-language explanation | Code/evidence |
|---|---|---|
| Popularity baseline | **Implemented.** Recommend items with the most training interactions. It is a simple reference point that a learned model must beat. | [Popularity model](../src/models/popularity.py), [shared-catalog ranking](../src/models/evaluate_two_tower.py), [tests](../tests/test_popularity.py) |
| Collaborative filtering / matrix factorization / ALS | **Implemented.** ALS factorizes the user-item interaction table into user and item vectors. Similarity is measured by their dot product. Collaborative filtering uses other users' interaction patterns rather than item text. | [ALS training/evaluation](../src/models/collaborative_filtering.py), [tests](../tests/test_collaborative_filtering.py), [matched experiment results](two_tower_test_report.md#matched-three-model-validation-comparison-2026-10-05) |
| Implicit vs explicit feedback | **Implemented as implicit only.** A review event is treated as evidence of interaction, not as a numeric star-rating preference. ALS counts repeated events as confidence; the two-tower learns observed pairs versus sampled unobserved pairs. An explicit-rating model is not implemented. | [ALS `implicitPrefs`](../src/models/collaborative_filtering.py), [pairwise model](../src/models/two_tower.py) |
| Embeddings and two-tower retrieval | **Implemented in a simple ID-only form.** User IDs and item IDs index learned vectors; dot products score candidates. “Retrieval” means finding a short list from a large catalog. There are no user/item content features and no ANN index yet. | [Model](../src/models/two_tower.py), [training loop](../src/models/train_two_tower.py), [top-K retrieval](../src/models/evaluate_two_tower.py) |
| Negative sampling | **Implemented.** For each observed user-item pair, sample an item absent from that user's training history. The sampled item teaches the model what to rank below a known interaction; absence is not guaranteed dislike. | [Sampler](../src/models/sampling.py), [pairwise dataset](../src/models/two_tower.py), [tests](../tests/test_sampling.py) |
| Cold start | **Partially measured, not solved.** Training-only mappings exclude unknown IDs from encoded holdouts; the ID-only model cannot create a personalized vector for a new user or item. The report counts candidate-catalog targets that cannot be ranked. Content-based fallback is not implemented. | [Training-only mappings](../src/models/prepare_tower_data.py), [coverage report](two_tower_test_report.md#matched-three-model-validation-comparison-2026-10-05) |
| Popularity bias | **Partially demonstrated.** User-history and target-popularity slices show recommendations work much better for popular targets; the rarest target group has almost no hits. Fairness or debiasing methods are not implemented. | [Cohort metrics](../src/models/evaluate_two_tower.py), [measured slices](two_tower_test_report.md#matched-three-model-validation-comparison-2026-10-05) |
| Temporal splitting and leakage | **Implemented.** Per-user events are ordered by time; earlier events train the model and later events are held out. IDs are mapped using train only, and evaluation masks items already seen in training. | [Temporal split](../src/data/prepare.py), [training-only mappings](../src/models/prepare_tower_data.py), [split-isolation test](../tests/test_evaluate_two_tower.py) |
| Recall@K, Precision@K, NDCG@K | **Implemented and tested.** Recall asks whether a held-out item appeared in the top K; Precision measures the relevant share of K slots; NDCG rewards placing a hit higher. One target per user means Precision equals Recall/K in these experiments. | [Metric implementation](../src/models/evaluate_two_tower.py), [hand-calculated test](../tests/test_evaluate_two_tower.py) |
| Offline evaluation | **Implemented.** Models are compared on historical held-out interactions, not live user experiments. We have fixed-cohort, shared-catalog validation comparisons and one test checkpoint; the report explains limits on interpreting that test as unbiased. | [Evaluator](../src/models/evaluate_two_tower.py), [test and training report](two_tower_test_report.md) |
| Gradient-boosted ranking | **Implemented as an offline comparison.** XGBoost learns pairwise ordering from observed training items and sampled unseen items, using item popularity and co-occurrence with a user's training history. It is evaluated on the same sampled validation candidates as the two-tower. Item text and metadata are not features. | [XGBoost ranker](../src/models/xgboost_ranker.py), [tests](../tests/test_xgboost_ranker.py), [measured result](two_tower_test_report.md#xgboost-pairwise-ranking-experiment-2026-10-05) |
| Pandas candidate tables | **Implemented for bounded model data.** Spark selects the cohort rows; Pandas builds numeric candidate-feature tables for XGBoost, including training-only history affinity. It does not collect or replace the full Spark dataset. | [Feature table builder](../src/models/xgboost_ranker.py), [PySpark notes](pyspark_recommender_notes.md) |
| Retrieval vs ranking | **Partially implemented.** Two-tower scores the catalog; XGBoost learns an independent ordering over sampled user/item candidates. The experiment compares their scores on the same validation lists. XGBoost is not yet a cascade stage in the API. | [Two-tower evaluation](../src/models/evaluate_two_tower.py), [XGBoost ranker](../src/models/xgboost_ranker.py) |

### PySpark

| Concept | Status and plain-language explanation | Code/evidence |
|---|---|---|
| Lazy execution; transformations/actions | **Implemented and demonstrated in preparation.** `select`, `where`, `join`, and `withColumn` describe work; actions such as `count`, `collect`, and writes make Spark execute it. | [Data preparation](../src/data/prepare.py), [PySpark notes](pyspark_recommender_notes.md) |
| DAGs, stages, and tasks | **Explained, not deeply instrumented.** Spark turns transformations into an execution graph and runs task groups by stage. Current notes explain the idea, but there is no recorded Spark UI walkthrough yet. | [PySpark notes](pyspark_recommender_notes.md) |
| Partitions and shuffles | **Partially implemented.** Local jobs expose `--shuffle-partitions`; grouping, sorting, joins, and windows can shuffle data. We have not done a tuning experiment showing task sizes or shuffle cost. | [Preparation configuration](../src/data/prepare.py), [cohort builder](../src/models/prepare_comparison_cohort.py) |
| Narrow/wide transformations | **Explained, not explicitly demonstrated.** Row filters/selects are usually narrow; groupings and many joins require exchange/shuffle. No focused before/after exercise exists yet. | [PySpark notes](pyspark_recommender_notes.md) |
| Joins and join strategies | **Implemented joins; strategy tuning is partial.** User/item mappings and cohort selection use joins. We have not forced or compared sort-merge, shuffled-hash, and broadcast join plans. | [Mapping joins](../src/models/prepare_tower_data.py), [cohort joins](../src/models/prepare_comparison_cohort.py) |
| Broadcast joins | **Not implemented as an explicit technique.** Some cohort lists are small in this bounded run, but code does not call `broadcast()` or teach the physical choice yet. | Future PySpark lab |
| Caching/persistence | **Implemented.** Preparation persists reused review/ranked data to memory and disk, then unpersists the ranked frame. This avoids recomputing expensive lineage for multiple split writes. | [Persistence lifecycle](../src/data/prepare.py) |
| Parquet | **Implemented.** Prepared train/validation/test splits and encoded model data are written and read as Parquet. | [Dataset writer](../src/data/prepare.py), [mapping/encoding](../src/models/prepare_tower_data.py) |
| Partition pruning and predicate pushdown | **Not intentionally demonstrated.** Parquet is used, but the data is not laid out by partition columns and no explain-plan/storage-scan lesson checks pruning or pushdown. | Future PySpark lab |
| Physical plans | **Not implemented as a lesson.** There are no saved `explain("formatted")` outputs or comparison of logical and physical plans. | Future PySpark lab |
| Skew | **Not measured.** No skewed-key diagnostic, salting, or task-duration comparison exists. | Future PySpark lab |
| UDF tradeoffs | **Not implemented.** Current transformation logic uses Spark SQL expressions and window functions; there is no Python UDF to compare. | [Preparation uses built-in expressions](../src/data/prepare.py) |
| Window functions | **Implemented.** `row_number` and per-user `count` assign interaction chronology and identify the latest validation/test events. | [Temporal split](../src/data/prepare.py), [split tests](../tests/test_prepare.py) |
| Safe large-data handling | **Partial/local development only.** The pipeline streams through Spark DataFrames and writes Parquet, but several bounded model steps collect rows to the driver. Production-scale distributed training and cloud object-store publication are not implemented. | [Preparation](../src/data/prepare.py), [bounded pair loader](../src/models/train_two_tower.py) |

### MLOps and platform

| Concept | Status and plain-language explanation | Code/evidence |
|---|---|---|
| Experiment tracking | **Implemented locally.** MLflow records parameters, metrics, epoch history, and artifacts in a local SQLite-backed tracking store. SageMaker managed MLflow is a later AWS option; Bedrock is not the tracker for this PyTorch recommender. | [Tracking helper](../src/models/experiment_tracking.py), [tracking test](../tests/test_experiment_tracking.py), [AWS tracking decision](deployment_walkthrough.md#where-experiment-tracking-should-live) |
| Model artifacts | **Implemented locally and S3-readable.** Two-tower checkpoints include weights and ID metadata; the batch job adds item vectors, ID maps, seen history, checksums, and a ZIP. API startup can download the ZIP from S3. Current files are local ignored artifacts unless uploaded. | [Checkpoint save/load](../src/models/train_two_tower.py), [batch export](../src/pipeline/batch_job.py), [API artifact loader](../src/api/app.py) |
| Model registration | **Implemented locally behind the validation gate.** The current checkpoint is MLflow registry version 2 under alias `candidate`. There is no automated production promotion. | [Registration command](../src/models/register_model.py), [gate tests](../tests/test_model_registration.py), MLflow run `05edeb1157c248ddb288bef43b08ef17` |
| API serving | **Implemented and exercised through Docker Compose.** FastAPI loads a batch bundle and serves top-N items by mapped user index; `/model-info` includes model and run lineage. | [API](../src/api/app.py), [API tests](../tests/test_api.py), [run record](deployment_walkthrough.md#end-to-end-run-record-2026-10-05) |
| Caching | **Implemented as optional Redis plus process-memory cache.** If Redis is unavailable, the API counts the error and still scores recommendations. | [API cache behavior](../src/api/app.py), [fallback test](../tests/test_api.py), [chaos script](../deploy/chaos/cache_outage.ps1) |
| Metrics / Prometheus | **Implemented and scraped in the Compose run.** API exposes request counts, latency, errors, cache hits, and `/metrics`; Prometheus reported the API target UP. | [API metrics](../src/api/app.py), [Prometheus scrape config](../deploy/prometheus/prometheus.yml), [run record](deployment_walkthrough.md#end-to-end-run-record-2026-10-05) |
| OpenTelemetry | **Request tracing exported to the local Collector.** FastAPI request spans appeared in Collector logs during the Compose run. | [API tracing setup](../src/api/app.py), [Collector config](../deploy/otel/collector.yml), [run record](deployment_walkthrough.md#end-to-end-run-record-2026-10-05) |
| Orchestration | **Airflow container and DAG discovery verified; training task not run.** DAG imports cleanly, but Windows denied WSL access during the task trigger. | [Airflow DAG](../deploy/airflow/dags/recommender_batch.py), [deployment record](deployment_walkthrough.md#airflow-and-kafka-demos) |
| Streaming-event concepts | **One Kafka producer/consumer flow verified.** A click event was validated and consumed into retained JSON Lines for future offline training. | [Event schema](../src/events/schema.py), [producer](../src/events/produce_event.py), [consumer](../src/events/consume_events.py), [evidence](run_evidence/kafka_event_demo_2026-10-05.jsonl) |
| Containerization / Kubernetes | **Docker/Compose verified locally; Kubernetes manifest not run.** The batch-exported bundle served through Compose; API health, requests, Redis cache/fallback/recovery, Prometheus scrape, and OpenTelemetry traces passed. The API image is about 1.57 GB. A local Kubernetes cluster has not been launched. | [End-to-end run](deployment_walkthrough.md#end-to-end-run-record-2026-10-05), [Dockerfile](../Dockerfile), [Compose](../compose.yaml), [Kubernetes manifest](../deploy/kubernetes/recommender.yaml) |
| AWS / Terraform | **Terraform validates; AWS not applied.** ECS Fargate task, ECR, S3 model bucket, IAM, existing default-VPC networking, and CloudWatch are configured. | [Terraform](../deploy/aws/main.tf), [AWS runbook](aws_deployment_runbook.md) |

The code path now reaches a batch-exported model bundle and a locally tested
recommendation API. Compose observability and Kafka were exercised; Airflow's
training task, Kubernetes, and AWS still need runtime verification. See the
[deployment walkthrough](deployment_walkthrough.md) and the verification
statuses in [PLAN.md](../PLAN.md).

## Setup and first run

This walkthrough assumes Linux or WSL because Spark's local Windows
filesystem can fail with Hadoop `NativeIO` errors. It uses Python 3.11, Java
21, and the PySpark version in `pyproject.toml`. Start with a fresh checkout:

```bash
git clone https://github.com/akshaymgupte87/EndtoEndMLProject.git
cd EndtoEndMLProject
sudo apt-get update
sudo apt-get install -y openjdk-21-jdk python3.11 python3.11-venv gzip curl
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . pytest
```

Set Spark to use the same Python environment and the PySpark jars installed in
it. Run these exports in the same WSL shell before Spark jobs or tests:

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export SPARK_HOME="$(python -c 'import pathlib,pyspark; print(pathlib.Path(pyspark.__file__).parent)')"
export PATH="$SPARK_HOME/bin:$JAVA_HOME/bin:$PATH"
export PYSPARK_PYTHON="$(command -v python)"
export PYTHONPATH="$PWD"
export SPARK_LOCAL_IP=127.0.0.1
export PYSPARK_SUBMIT_ARGS='--driver-memory 8g pyspark-shell'
```

Get the Movies and TV review file from the [official Amazon Reviews 2023
page](https://amazon-reviews-2023.github.io/). The raw data is not in Git. The
project's preparation code expects uncompressed JSON Lines at
`data/raw/Movies_and_TV.jsonl`:

```bash
mkdir -p data/raw
curl --fail --location \
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Movies_and_TV.jsonl.gz \
  --output data/raw/Movies_and_TV.jsonl.gz
gzip --decompress --keep data/raw/Movies_and_TV.jsonl.gz
```

The complete first-run order is data preparation, ID mapping, model training,
validation evaluation, and tests. These commands reproduce the five-million
input-row development sample. The model is trained on a smaller bounded user
sample for development; it does not consume all five million rows during
training.

```bash
python -m src.data.prepare \
  --input data/raw/Movies_and_TV.jsonl \
  --output data/processed/movies_tv_5m \
  --limit 5000000 \
  --shuffle-partitions 32

python -m src.models.prepare_tower_data \
  --data data/processed/movies_tv_5m \
  --output data/processed/movies_tv_5m/two_tower/v1 \
  --shuffle-partitions 32

python -m src.models.train_two_tower \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --output artifacts/two_tower/movies_tv_5m/v1 \
  --max-users 2000 --max-train-pairs 100000 \
  --epochs 10 --seed 42

python -m src.models.evaluate_two_tower \
  --checkpoint artifacts/two_tower/movies_tv_5m/v1/best_model.pt \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --output artifacts/two_tower/movies_tv_5m/v1/validation_metrics.json \
  --k 10 20

python -m pytest -q -o cache_dir=/tmp/endtoendml-pytest-cache
```

Training and evaluation create local MLflow runs in `mlflow.db`. To compare
them, run `mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1`
and open `http://127.0.0.1:5000`. The `data/`, `artifacts/`, and MLflow files
are ignored by Git, so each checkout creates its own copies. Use new versioned
output folders when deliberately keeping multiple runs.

The project has a known benchmark limitation: `src/data/prepare.py` calculates
user/item eligibility before temporal splitting, while the two-tower mapping
is built from training only. The notes identify this as possible future-data
leakage in preprocessing eligibility; do not interpret the current experiment
as a production-grade benchmark.

## Code map: which function does what

Use this index to jump between an explanation below and the implementation.
File paths are relative to the project root. The order follows the data:
prepare IDs, make training examples, score them with the model, then train and
save that model.

### `src/models/prepare_tower_data.py`

| Function | What it does |
|---|---|
| `parse_args()` | Reads input split path, output path, and Spark shuffle setting from the command line. |
| `build_id_mapping(frame, source_column, index_column)` | Makes a sorted, distinct ID table and assigns zero-based integer indices. Called for users and items using training data only. |
| `encode_split(frame, user_mapping, item_mapping)` | Replaces known string IDs with integer indices; returns encoded rows and a count of rows dropped for unknown IDs. |
| `main()` | Starts Spark, reads the three splits, builds the two mappings, encodes and writes all outputs, and stops Spark. |

### `src/models/sampling.py`

| Function or method | What it does |
|---|---|
| `_validate_index(value, name)` | Rejects invalid, fractional, boolean, or negative IDs before they become tensor indices. |
| `build_seen_items(positive_pairs)` | Builds each user's set of observed training items, removing duplicate IDs within that set. |
| `UniformNegativeSampler.__init__(...)` | Copies and validates the training histories, catalog size, and random seed. |
| `UniformNegativeSampler.seen_items_by_user` | Exposes a read-only view of the observed-item sets for checks by the dataset. |
| `UniformNegativeSampler.sample(user_idx)` | Draws a catalog item the user did not see in training; uses a fallback for dense histories. |

### `src/models/two_tower.py`

| Function or method | What it does |
|---|---|
| `PairwiseTrainingDataset.__init__(...)` | Validates positive user/item pairs against the training history and stores them as integer tensors. |
| `PairwiseTrainingDataset.__len__()` | Reports the number of positive training pairs. |
| `PairwiseTrainingDataset.__getitem__(index)` | Returns one `(user, positive item, sampled negative item)` triple. |
| `TwoTowerRecommender.__init__(...)` | Creates the trainable user and item embedding tables. |
| `TwoTowerRecommender.encode_users(indices)` | Looks up user vectors for the supplied user indices. |
| `TwoTowerRecommender.encode_items(indices)` | Looks up item vectors for supplied item indices. |
| `TwoTowerRecommender.score(user_indices, item_indices)` | Computes one dot-product score for each aligned user/item pair. |
| `TwoTowerRecommender.forward(users, positives, negatives)` | Scores each user's positive and sampled negative item for training. |
| `bpr_loss(positive_scores, negative_scores)` | Gives a larger penalty when a negative item scores above its positive; training minimizes this penalty. |

### `src/models/train_two_tower.py`

| Function or method | What it does |
|---|---|
| `TrainingConfig.validate()` | Checks training settings such as vector size, learning rate, epoch count, and requested device. |
| `_fixed_validation_dataset(...)` | Samples one repeatable negative per validation pair while excluding known training and validation items. |
| `_evaluate_pairwise_loss(model, loader, device)` | Measures average validation BPR loss without changing model weights. |
| `_save_checkpoint(...)` | Writes model weights and metadata to a temporary file, then replaces the target checkpoint. |
| `load_two_tower_checkpoint(path, device)` | Restores a saved model and returns its metadata. |
| `train_two_tower(...)` | Runs mini-batch training, records epoch losses, selects the best validation-loss weights, and optionally saves them. |
| `_read_sampled_pairs(...)` | Uses Spark to select bounded users, keep each selected user's complete history, and collect their encoded train/validation pairs. |
| `main()` | Parses run settings, starts Spark, loads the bounded sample, calls training, writes `history.json`, and stops Spark. |

### `src/models/evaluate_two_tower.py`

| Function | What it does |
|---|---|
| `_validate_k(k)` | Checks that a requested top-K cutoff is a positive integer. |
| `recommend_top_k(model, users, seen_items, k, batch_size)` | Scores the complete item catalog in batches and removes each user's training items before selecting the top K. |
| `recommend_popular_top_k(users, popularity_order, seen_items, k)` | Makes a global popularity ranking for comparison, also removing each user's training items. |
| `ranking_metrics_at_k(ranked, relevant, k)` | Computes macro-averaged Recall@K, Precision@K, and NDCG@K over users with validation targets. |
| `_load_validation_examples(spark, data_path, model_user_ids)` | Reads only selected users' training histories and validation targets; filters targets that were already seen in training. |
| `evaluate_validation(checkpoint, data, ks, ...)` | Loads the checkpoint, ranks the catalog for validation users, compares with same-sample popularity, and returns report data. |
| `main()` | Reads CLI settings, runs validation evaluation, writes and prints the JSON report. |

### `src/models/demo_two_tower.py`

| Function | What it does |
|---|---|
| `main()` | Creates a tiny made-up dataset, trains for demonstration, and prints sample users' top-scored items. It does not read the Amazon data. |

### `src/models/experiment_tracking.py`

| Function | What it does |
|---|---|
| `log_experiment_run(...)` | Creates an MLflow run and records parameters, final metrics, epoch-step metrics, tags, and local artifacts in the selected tracking store. |

### `src/models/collaborative_filtering.py`

| Function | What it does |
|---|---|
| `fit_implicit_als(train, ...)` | Counts each user/item pair and fits Spark ML ALS in implicit-feedback mode. Repeated interactions increase the confidence weight. |
| `_factor_map(frame)` | Collects Spark's learned factor rows into an index-to-vector lookup for ranking. |
| `recommend_top_k_als(...)` | Computes user/item factor dot products in batches and removes each user's training items from candidates. |
| `evaluate_als_validation(...)` | Ranks validation targets for a supplied user cohort and compares ALS with a popularity list built from the full training split. It never reads test. |
| `main()` | Loads configuration and user IDs, fits/saves ALS, writes validation metrics, archives the model, and logs params/metrics/model/report to MLflow. |

### `src/models/xgboost_ranker.py`

| Function | What it does |
|---|---|
| `sample_unseen_items(...)` | Draws distinct negative candidate IDs that are absent from the selected user's training history. |
| `build_labeled_pairs(...)` | Creates each user's grouped training rows: observed training items have label 1; sampled unseen items have label 0. |
| `build_validation_candidates(...)` | Keeps validation targets and adds reproducible unseen negatives without reading the test split. |
| `add_training_only_features(...)` | Uses Pandas to form numeric item-popularity and history-affinity features from training data only. Positive training rows use leave-one-out affinity to remove their own contribution. |
| `_load_cohort_rows(...)` / `_feature_maps(...)` / `_cooccurrence_counts(...)` | Uses Spark to collect only checkpoint users' train/validation rows, then computes training histories, item counts, and item-pair co-occurrence counts. |
| `run_experiment(...)` | Fits `XGBRanker` on grouped training candidates, scores sampled validation candidates with both XGBoost and two-tower, and saves a JSON model and metrics report. |
| `main()` | Validates command-line options and runs the offline experiment. |

## Mini code-flow walkthrough

Follow this call chain when stepping through the project in a debugger:

1. `src.data.prepare.main()` calls `load_reviews()` to read and validate JSONL,
   `filter_interactions()` to remove sparse users/items, and
   `write_dataset()` to save cleaned interactions and temporal splits. The
   split helpers put each user's latest row in test and the previous row in
   validation.
2. `src.models.prepare_tower_data.main()` reads those three splits.
   `build_id_mapping()` creates user/item integer indices from training only;
   `encode_split()` joins each split to those mappings and counts held-out
   rows with unknown IDs.
3. `src.models.train_two_tower.main()` calls `_read_sampled_pairs()` to select
   bounded users while preserving each full training history. Then
   `train_two_tower()` creates `PairwiseTrainingDataset`. Its
   `__getitem__()` asks `UniformNegativeSampler.sample()` for an unseen
   training item. `TwoTowerRecommender.forward()` scores the positive and
   negative pairs; `bpr_loss()` produces the penalty; `loss.backward()` and
   `optimizer.step()` update embedding rows. Each epoch computes validation
   loss, and the lowest-loss checkpoint is saved.
4. `src.models.evaluate_two_tower.main()` loads that checkpoint and calls
   `evaluate_validation()`. `_load_validation_examples()` reads only train and
   validation; `recommend_top_k()` scores all catalog items and masks each
   user's training items; `recommend_popular_top_k()` makes the paired
   baseline. `ranking_metrics_at_k()` computes Recall, Precision, and NDCG;
   `_metrics_by_cohort()` repeats them for user-history and target-popularity
   groups. The report is written to JSON.
5. `src.models.collaborative_filtering.main()` fits implicit ALS from the
   complete train split, turns factor tables into user/item vectors, and
   ranks the same fixed validation cohort against a full-train popularity
   baseline. It does not read test.
6. The training and evaluation `main()` functions call
   `log_experiment_run()` to store their parameters, metrics, and artifacts in
   MLflow. This records the experiments; it does not register or deploy a
   model.
7. `src.models.xgboost_ranker.main()` uses Spark to load the bounded cohort,
   Pandas to form grouped candidate features, and XGBoost to learn pairwise
   ordering. It compares XGBoost and the saved two-tower checkpoint on the
   same sampled validation candidates. It does not yet add a second stage to
   the API.

## XGBoost: pairwise ranking experiment

Two-tower learns user and item vectors and scores their dot product. XGBoost
offers a different model family: a gradient-boosted tree ranker learns rules
for ordering candidate pairs. Here its features are the item's training
interaction count and how often that candidate co-occurs with the user's other
training items. Counts are `log1p` transformed. User IDs identify ranking
groups; raw IDs are not predictive features.

Spark reads the selected users' encoded train and validation rows. Only those
bounded rows are collected to the driver. Pandas builds numeric feature
tables; it does not replace Spark for the five-million-row input or
the 17.4M-row deployment target. XGBoost trains on unique positive train
pairs and uniformly sampled items absent from each user's training history.
For validation, each user's held-out targets are combined with 99 sampled
unseen items. The same candidates are scored by XGBoost and the existing
two-tower checkpoint. The test split is never opened.

Run from the WSL project root after preparation, ID mapping, and the fixed
comparison cohort exist:

```bash
python -m src.models.xgboost_ranker \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --checkpoint artifacts/comparisons/movies_tv_5m/two_tower/seed42/best_model.pt \
  --output artifacts/xgboost_ranker/cohort_v4 \
  --estimators 150 --seed 42 --validation-negatives 99 --k 10 20
```

The output directory must be new or empty; choose a versioned folder such as
`artifacts/xgboost_ranker/cohort_v4` for a rerun. It contains
`xgboost_ranker.json` and `validation_metrics.json`. The report records the
candidate protocol, feature importance, and both models' ranking metrics.
This is an offline comparison,
not yet an API cascade or a full-catalog estimate. Because the negatives are
sampled uniformly, the score is easier than ranking against every catalog
item; compare it only with the two-tower score on these same candidates.
The held-out test split for the current cohort has already been consumed, so
this experiment uses validation only.

Measured outcomes are recorded in the [test report](two_tower_test_report.md#xgboost-pairwise-ranking-experiment-2026-10-05).

## Running the project's tests

The project uses `pytest`. Pytest discovers every `test_*.py` file under
`tests/` (configured by `[tool.pytest.ini_options]` in `pyproject.toml`). The
shared Spark fixture in `tests/conftest.py` starts one small `local[2]` Spark
session for tests that need Spark, then stops it after the suite. The suite
checks application behavior on small synthetic examples; it does not retrain
the model on the full Amazon dataset.

### Which test file covers which part

| Test file | What it checks |
|---|---|
| `tests/test_collaborative_filtering.py` | ALS ranking, seen-item removal, archive contents, source-ID factor alignment, and comparison aggregation. |
| `tests/test_evaluate_two_tower.py` | Ranking metrics, candidate filtering, exclusion of training-seen items, and isolation between selected validation/test directories. |
| `tests/test_experiment_tracking.py` | SQLite-backed MLflow parameter, metric, epoch-step, and artifact logging. |
| `tests/test_api.py` | API recommendations, user handling, seen-item filtering, metrics, and Redis fallback. |
| `tests/test_event_schema.py` | Event schema validation, JSON round trips, and invalid payload rejection. |
| `tests/test_model_registration.py` | Validation-only model gate and rejection of invalid or regressing reports. |
| `tests/test_prepare.py` | Filters sparse users/items and assigns the latest two interactions to validation and test. |
| `tests/test_prepare_tower_data.py` | Deterministic integer mappings and encoding behavior for known and unknown IDs. |
| `tests/test_popularity.py` | Popularity ordering, deterministic tie-breaking, K validation, and per-user hit-rate behavior. |
| `tests/test_sampling.py` | Seen-item histories and valid, reproducible uniform negative sampling, including dense histories and invalid inputs. |
| `tests/test_two_tower.py` | Training-example triples, embedding lookup/scoring, and pairwise loss behavior. |
| `tests/test_train_two_tower.py` | Training loop, validation-loss checkpoint selection, checkpoint round-trip, and input validation. |
| `tests/test_xgboost_ranker.py` | Deterministic candidates, history exclusion, training-only Pandas features, and a small grouped XGBoost fit/predict check. |

The complete inventory of all 100 collected cases, including the expanded
parameterized inputs and a concrete pass condition for each case, is in the
[test results report](two_tower_test_report.md#complete-pytest-inventory-and-pass-criteria).
It also distinguishes the pytest suite from offline evaluation, batch smoke,
Terraform, Compose, and runtime checks. Passing a behavioral test means its
contract held on the fixture; it does not imply that a recommender beats the
baseline or that AWS/container infrastructure is live.

### Run the full suite in WSL

On this Windows setup, run Spark tests in WSL Ubuntu with the Miniconda Python,
PySpark package, and Java 21 that work together. From the project root in
PowerShell, this command runs all test files quietly and sends pytest's cache
to WSL's `/tmp` directory so it does not try to write a cache to the
Windows-mounted project folder:

```powershell
wsl.exe -d Ubuntu --cd /mnt/c/Users/aksha/PycharmProjects/EndtoEndMLProject --exec /usr/bin/env JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 SPARK_HOME=/home/akshay/miniconda3/lib/python3.11/site-packages/pyspark PATH=/home/akshay/miniconda3/lib/python3.11/site-packages/pyspark/bin:/home/akshay/miniconda3/bin:/usr/lib/jvm/java-21-openjdk-amd64/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin PYSPARK_PYTHON=/home/akshay/miniconda3/bin/python PYTHONPATH=/mnt/c/Users/aksha/PycharmProjects/EndtoEndMLProject SPARK_LOCAL_IP=127.0.0.1 'PYSPARK_SUBMIT_ARGS=--driver-memory 8g pyspark-shell' /home/akshay/miniconda3/bin/python -m pytest -q -o cache_dir=/tmp/endtoendml-pytest-cache
```

For visible per-test progress, replace `-q` with `-v`. To see a short summary
at the end without one line per test, omit `-q` and `-v`. A successful run
ends with a summary such as `100 passed`; the count can grow as tests are
added. The latest verified count and warnings are recorded in [the test
report](two_tower_test_report.md). Warnings are listed separately and do not
mean a test failed unless pytest reports a nonzero exit status.

The command sets a few environment values for the child process: `JAVA_HOME`
selects Java 21; `SPARK_HOME` and the first `PATH` entries select Spark from the
same Miniconda PySpark installation; `PYSPARK_PYTHON` selects that WSL Python
for Spark workers; `PYTHONPATH` lets Python import this project from the
mounted folder; `SPARK_LOCAL_IP` avoids Spark host-address ambiguity; and
`PYSPARK_SUBMIT_ARGS` gives the test Spark driver an 8 GB heap. If your WSL
username, project path, or Miniconda location differs, update the corresponding
paths before running the command.

### Run one file or one test

Keep the same WSL environment prefix and replace the final pytest arguments.
For example, to run all evaluator tests:

```text
-m pytest -v -o cache_dir=/tmp/endtoendml-pytest-cache tests/test_evaluate_two_tower.py
```

To run one named test, append `-k` and a distinctive substring from its
function name, for example:

```text
-m pytest -v -o cache_dir=/tmp/endtoendml-pytest-cache tests/test_evaluate_two_tower.py -k validation_loader
```

For a Windows PowerShell test command that is copy-and-pasteable, keep the
same WSL command above and replace everything after the WSL environment setup
with `/home/akshay/miniconda3/bin/python -m pytest ...`. In WSL itself, first
open the project directory and establish the environment once in that shell:

```bash
cd /mnt/c/Users/aksha/PycharmProjects/EndtoEndMLProject
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export SPARK_HOME=/home/akshay/miniconda3/lib/python3.11/site-packages/pyspark
export PATH="$SPARK_HOME/bin:/home/akshay/miniconda3/bin:$JAVA_HOME/bin:$PATH"
export PYSPARK_PYTHON=/home/akshay/miniconda3/bin/python
export PYTHONPATH="$PWD"
export SPARK_LOCAL_IP=127.0.0.1
export PYSPARK_SUBMIT_ARGS='--driver-memory 8g pyspark-shell'
python -m pytest -v -o cache_dir=/tmp/endtoendml-pytest-cache
```

The WSL pytest cache option is optional. The recent run without it still
passed all tests but emitted a permission warning because `/mnt/c` does not
allow the Linux user to create the configured `.cache/pytest` directory. The
other current warning is PySpark's pandas-version compatibility notice.

## Step 1: create model-ready integer IDs

### Why string IDs need mappings

The prepared split files identify users and products with original string IDs.
Those IDs are useful for understanding the data, but a PyTorch embedding layer
uses an integer row index. For example, an embedding table with 20,000 users
has rows numbered from 0 to 19,999. Each interaction must therefore refer to
`user_idx` and `item_idx` values in those ranges.

The preprocessing command builds two lookup tables:

```text
user_id  → user_idx
item_id  → item_idx
```

It takes distinct IDs from the training split, sorts them, and assigns
zero-based indices. Sorting makes the mapping deterministic: the same training
IDs produce the same mapping regardless of the source row order. The mapping
tables are written beside the encoded splits so future training and serving can
translate indices back to original user and item IDs. We generated this
artifact for `movies_tv_5m` in the first real-data run described below.

### Why mappings are fitted on training only

Validation and test represent future interactions. If IDs from those splits
were used to construct the embedding tables, the model's vocabulary would
depend on held-out information. Fitting mappings on training only keeps the
model's learned ID space tied to data available at training time.

If a validation or test row contains a user or item absent from training, the
current implementation excludes that pair from the encoded split and reports
the number excluded. An embedding model cannot score a missing ID without an
explicit cold-start strategy. We will use those exclusion counts when
interpreting evaluation coverage; they must not be hidden from the reported
metrics.

### Output layout and run command

The input is the directory containing the existing `train`, `validation`, and
`test` Parquet folders. The output is a new directory containing the mapping
tables and encoded pairs:

```text
<output>/
    user_mapping/   user_id, user_idx
    item_mapping/   item_id, item_idx
    train/          user_idx, item_idx
    validation/     user_idx, item_idx
    test/           user_idx, item_idx
```

Run it after the Spark-prepared dataset is available:

```bash
python -m src.models.prepare_tower_data \
  --data data/processed/movies_tv_5m \
  --output data/processed/movies_tv_5m/two_tower/v1 \
  --shuffle-partitions 32
```

The same command is available as `prepare-tower-data` after installing the
project package. The current implementation writes the requested output
folders in overwrite mode, so use a dedicated versioned output directory and
do not point it at the source split directory.

### What this step does not do

Step 1 only defines the transformation that creates stable integer IDs and
removes held-out pairs whose IDs are unknown to the training vocabulary. It
does not train a network or compute recommendation metrics. Step 2 defines
positive and negative examples; Steps 3 and 4 implement the model and training
loop.

### Verification checklist

When we run this step in a working Spark/Java environment, check that:

1. Both mappings contain distinct original IDs and contiguous indices from
   zero through `number_of_ids - 1`.
2. Mapping inputs come only from the training split.
3. Every encoded row has valid indices within the saved mapping sizes.
4. The training encoded row count equals the input training row count.
5. Validation and test exclusion counts are shown and their encoded counts
   match input counts minus excluded rows.
6. Re-running from the same source data produces identical mappings.

Spark could not initialize reliably in the Windows Python environment during
earlier checks. This mapping run used WSL Ubuntu with Miniconda Python 3.11,
PySpark 4.2.0, and Java 21. The full test suite has also now passed in that
WSL environment; details are recorded below.

### First run on the prepared 5M sample (2026-10-04)

We ran `prepare_tower_data.py` from WSL against
`data/processed/movies_tv_5m` and wrote the results to
`data/processed/movies_tv_5m/two_tower/v1`. This folder did not exist before
the run, so the job did not replace an earlier mapping output. The directory
name says `5m` because it came from the five-million-row preparation sample;
after filtering and splitting, the three prepared splits contained 3,167,935
interaction rows.

The run created mappings for **238,645 training users** and **137,070 training
items**. Spark verified that their indices are unique and continuous: users
run from 0 to 238,644, and items from 0 to 137,069. Here are the interaction
counts before and after encoding:

| Split | Prepared rows | Encoded rows | Excluded rows |
|---|---:|---:|---:|
| Train | 2,687,407 | 2,687,407 | 0 |
| Validation | 240,034 | 237,857 | 2,177 |
| Test | 240,494 | 237,067 | 3,427 |
| **Total** | **3,167,935** | **3,162,331** | **5,604** |

Why are some validation and test rows excluded? The model has an embedding
row (a learned vector) only for a user and item that appeared in training. If
a held-out interaction contains a user or item first seen later, the current
ID-only model has no vector to look up, so the encoding step drops that whole
interaction row. A row is counted once as excluded even if both IDs are
unknown. This does not mean those are bad interactions or negative examples;
it means this model version cannot score them. Later evaluation will measure
recommendation quality on the encoded held-out rows, so these counts tell us
that the evaluation covers fewer rows than the prepared validation and test
splits.

### Mapping questions and answers

**Q: Why does the two-tower model need mappings?**

The source data identifies people and products with text IDs. PyTorch's
embedding tables work like numbered rows: for example, user index `12` selects
the thirteenth user vector. The mappings translate each original ID to the
row number the model can look up. The two tables are `user_id → user_idx` and
`item_id → item_idx`; the item mapping also lets a later recommendation
system translate a scored item index back into the original product ID.

**Q: Why not give the model the original text IDs directly?**

The model needs a compact integer address for each embedding row. The text ID
is a label, not a meaningful numeric measurement: treating it as a number
would invent an order and distance between users or products. A lookup table
preserves identity without suggesting that one ID is numerically closer to
another.

**Q: Why are mappings made from training data only?**

Training data is the history the model is allowed to learn from. Validation
and test stand in for later events. If their IDs were allowed to create
embedding rows, the model's vocabulary would learn that future users or items
exist before it is evaluated. Building mappings from training only keeps the
evaluation honest and makes unseen-ID coverage visible.

**Q: What happens when a later interaction has an ID missing from the
mapping?**

The current ID-only model cannot look up a vector for that user or item, so
the encoding step drops that interaction and counts it. As the results above
show, 2,177 validation rows and 3,427 test rows were dropped. The model does
not learn that these users dislike those items; it simply cannot represent
those rows yet. Supporting them will require a cold-start plan, such as
feature-based user/item representations or a fallback recommendation.

**Q: Where are the mappings saved, and why keep them?**

They are saved in `data/processed/movies_tv_5m/two_tower/v1/user_mapping`
and `.../item_mapping`, alongside the encoded splits. Training needs the same
index assignments used for its interaction rows. Recommendation serving also
needs the item mapping to turn ranked item indices back into product IDs.
Rebuilding mappings independently later could assign different row numbers
and make a saved model point at the wrong users or products.

The mapping outputs are under `data/`, which `.gitignore` excludes. They are
present on this machine for the next modeling step, but they were not included
in the GitHub source-code commit.

Historical mapping-stage run (before evaluator and MLflow tests were added):

```text
64 passed in 23.56s
```

The current full-suite result is 77 passed in 20.75 seconds; use the current
[test report](two_tower_test_report.md) for the complete breakdown. The older
count is kept here only as a record of the mapping milestone.

The tests included the Spark mapping checks and synthetic model-training
checks. Pytest reported two warnings: PySpark's current pandas-version
compatibility notice and an inability to write its test cache under the
Windows-mounted project directory. Neither failed a test. To avoid the cache
warning on later WSL runs, use a Linux temporary folder:

```bash
python -m pytest -v -o cache_dir=/tmp/endtoendml-pytest-cache
```

### Reading `prepare_tower_data.py` from top to bottom

The command-line parser collects `--data`, the folder containing prepared
train/validation/test Parquet splits; `--output`, a separate destination for
encoded files; and `--shuffle-partitions`, a Spark execution setting. The
script starts local Spark, opens the three split folders, and later stops
Spark in a `finally` block so it is shut down even when work fails.

`build_id_mapping` selects one ID column, drops nulls, removes duplicates,
sorts the string IDs, and assigns zero-based row numbers with `zipWithIndex`.
Its explicit schema stores the original ID as a string and the index as a
64-bit integer. It is called only with the training DataFrame, once for users
and once for items. A user or product that appears only in validation or test
therefore does not get an embedding row.

`encode_split` selects the source user and item IDs, joins them to both
mappings using inner joins, and returns only `user_idx` and `item_idx`. Rows
with an unknown user or item disappear. It counts source rows and encoded
rows, and returns their difference as the exclusion count. The main function
writes the mappings and each encoded split as Parquet and prints those
exclusion counts. Keep this output in a dedicated versioned directory because
the writes use overwrite mode.

The distinction between preparation modules matters: `tests/test_prepare.py`
tests `src/data/prepare.py` (filtering sparse entities and temporal splits).
The new `tests/test_prepare_tower_data.py` checks the two-tower mapping
functions directly with tiny Spark DataFrames. One test verifies sorted,
distinct, zero-based indices and that null IDs are omitted. The other builds
mappings from a small training frame, then checks that encoding preserves
duplicate known pairs, drops rows with an unknown user or item, and reports
the exact number dropped. These are function-level tests; they do not yet run
the command-line entry point or verify its complete saved directory layout.

Run the focused checks in the WSL environment where Spark works:

```bash
python -m pytest -v tests/test_prepare_tower_data.py
```

## Step 2: define positive and negative training examples

### Reading `sampling.py`

`build_seen_items` turns interaction pairs into a dictionary of sets. For
example, repeated rows `(user 0, item 2)` collapse to one entry in user 0's
set. This is the lookup used to avoid calling an already-observed training
item a negative. It validates that IDs are non-negative integers; booleans,
fractions, and strings are rejected instead of being silently converted.

`UniformNegativeSampler` receives the catalog size and that dictionary. It
copies each user's history into immutable sets, checks that users and item IDs
are valid, and rejects users with no history or with every catalog item
already seen. The copy prevents later edits by a caller from changing the
sampler's behavior unexpectedly. A seeded `random.Random` generator makes
sampling reproducible.

When `sample(user_idx)` is called, it first makes up to 64 random catalog
draws, returning the first item absent from that user's seen set. This is
quick when histories occupy only a small share of the catalog. For a dense
history, repeated draws might keep hitting seen items; the fallback instead
draws a rank among the unseen items and walks the sorted seen IDs to translate
that rank back to an actual catalog index. That guarantees a valid draw
without building a potentially large list of all unseen catalog items.
Sampling for an unknown user raises an error because no history is available
to define valid negatives.

The sampler only knows the history passed to it, which must be built from
training pairs. It deliberately does not inspect validation or test data.
Therefore a later positive can be sampled as a training negative; this is the
chosen tradeoff for the first version, not a claim that the user dislikes that
item.

### Positive examples

The model is an **implicit-feedback** recommender. We interpret each observed
user/item interaction in the training split as a positive example: the user
engaged with that item. We do not use the optional rating as a target in this
step. This matches the popularity baseline's definition of positive feedback
and keeps comparisons consistent.

An interaction row is represented by the integer pair `(user_idx, item_idx)`.
If the training history contains the same user/item more than once, it is one
distinct item in the user's “already seen” set, but repeated rows can still
appear as repeated positive examples. We may decide to deduplicate those pairs
for training later if counts show that repeat reviews are accidental or
overweight frequent users.

### Sampled negative examples

Implicit feedback does not give us explicit dislikes. To teach the model to
rank an observed item above alternatives, we pair each positive training pair
with an item sampled from the known training item catalog that this user did
not interact with in training. The current sampler draws uniformly from those
eligible item indices. For example, if a user has training positives `{2, 5}`
in a catalog with indices `0` through `7`, valid negatives are `{0, 1, 3, 4,
6, 7}`.

The sampler consults only the training-positive history. Validation and test
are kept out of this process to preserve the time split. This has a known
tradeoff: an item the user interacts with later may be sampled as a training
negative. We accept that possibility for this first prototype rather than
using held-out behavior to alter training examples. Later, measured false
negative rates or a changed evaluation protocol may justify a different
sampling strategy.

Uniform sampling is simple and gives broad catalog coverage, but it often
produces easy negatives (items the user would never consider). Hard-negative
sampling can improve ranking but adds complexity and may amplify noisy or
false negatives. We will begin with uniform sampling, then use validation
results to decide whether a more complex strategy is warranted.

### Pairwise learning objective

The intended model input for one training example is a triple:

```text
(user_idx, positive_item_idx, negative_item_idx)
```

The two-tower model will produce a user vector `u` and item vectors
`v_positive` and `v_negative`. We will use their dot products as scores and a
pairwise Bayesian Personalized Ranking (BPR) loss:

```text
positive_score = dot(u, v_positive)
negative_score = dot(u, v_negative)
loss = -log(sigmoid(positive_score - negative_score))
```

Minimizing this loss encourages the positive item score to exceed the negative
item score. This objective is for implicit ranking; it is not a rating
prediction loss. The sampler is implemented in `src/models/sampling.py`; the
model and training loop are described in Steps 3 and 4 below.

### Reproducibility and edge cases

The sampler accepts a seed so its sequence can be repeated in tests. It checks
that item indices fit the catalog, rejects users who have observed every item
(there is no valid negative for them), and errors if asked to sample for a user
without training history. Rejection sampling is efficient for sparse users; a
uniform fallback handles unusually dense users without looping forever.

The behavior tests are in `tests/test_sampling.py`: they check de-duplicated
seen-item sets, exclusion of training positives, repeatability, saturated
catalogs, and unknown users. These tests do not require Spark or a GPU. To run
them in the configured project environment:

```bash
pytest tests/test_sampling.py -q
```

In more detail, `tests/test_sampling.py` also checks malformed, fractional,
boolean, and negative IDs; invalid histories and out-of-catalog items; and
that the sampler holds an unmodifiable snapshot when the original dictionary
changes. The dense-catalog test forces the fallback path and checks that each
possible unseen rank maps to the expected catalog item. These tests check
validity and repeatability. They do not prove statistically uniform
distribution over a huge number of random draws.

This step established the example and negative-sampling contract. The next
section implements the PyTorch dataset and model structure; it still does not
run an optimizer or train model weights.

## Step 3: build the PyTorch dataset and two towers

### Reading `two_tower.py`

The file contains three pieces: the dataset that makes triples, the neural
network that scores them, and the loss that teaches the network which score
should be larger.

`PairwiseTrainingDataset` is given positive `(user_idx, item_idx)` pairs and
the negative sampler. At construction it rejects empty data, malformed pairs,
invalid indices, and positives that are missing from the sampler's recorded
training history. It stores positives as PyTorch `long` tensors because
embedding-table indices must be integer tensors. When `__getitem__` fetches a
row, it samples one eligible negative and returns three scalar tensors:
user, positive item, negative item. A DataLoader batches many such triples.
With multiple DataLoader worker processes, it reseeds each worker's copied
sampler from PyTorch's worker seed, avoiding identical random streams while
letting a seeded DataLoader reproduce them.

`TwoTowerRecommender` creates two trainable embedding tables. One table has a
row for every user; the other has a row for every item. Each row is a vector
of `embedding_dim` numbers. Looking up an ID selects its vector, and learning
changes these vectors so observed items tend to align with the user's vector
more than sampled alternatives. This first implementation is ID-only: it does
not read product titles, categories, or other side information. The
constructor checks that table sizes and vector width are positive integers.

`encode_users` and `encode_items` are the lookup operations. `score` looks up
aligned user and item batches, multiplies the vectors element by element, then
sums each row to make one dot-product score per pair. It checks the input
shapes match to catch accidental broadcasting. `forward` calls `score` twice
for each batch, once for positive items and once for negative items, and
returns both score vectors.

`bpr_loss` checks that those score vectors have matching shapes, are
non-empty, floating point, and finite. It calculates
`softplus(negative_score - positive_score)` and averages across the batch.
This is a numerically stable form of the pairwise objective: if the positive
score is already above the negative score, its penalty is small; if the
negative score is higher, the penalty is large. Backpropagation uses that
penalty to update the selected embedding rows. The loss is a training signal,
not a direct statement of recommendation quality.

### Dataset: turn pairs into training triples

`PairwiseTrainingDataset` stores the positive `(user_idx, item_idx)` rows as
integer tensors. When PyTorch requests one row, the dataset asks the negative
sampler for an item the user did not interact with in training and returns:

```text
user_idx, positive_item_idx, negative_item_idx
```

Sampling when an example is fetched means the same positive pair can receive a
different negative on a later epoch. PyTorch's `DataLoader` can group these
scalar values into batches. For now, use `num_workers=0`: the sampler owns a
seeded random-number generator, and worker processes would need separately
seeded sampler copies to avoid repeated random streams. The eventual training
step will set and explain those loader details.

The current dataset holds its positive pair list in memory. It is intended for
the small development experiment, not automatically for the full 17.4M-row
production job. We will measure its memory use and choose a streaming or
partitioned input path before scaling training.

### ID-only user and item towers

`TwoTowerRecommender` has two embedding tables:

```text
user_idx → user embedding vector
item_idx → item embedding vector
```

An embedding table is a trainable matrix. Looking up an ID selects one row of
that matrix, and gradient updates change the selected rows. Both tables use
the same configurable vector size so their outputs can be compared directly.
The initial values are drawn from a small normal distribution to start scores
near zero.

For this first model, each tower is a direct ID embedding lookup. There are no
side features such as title text, categories, or review content. Dot product
combines the two vectors into a relevance score:

```text
score(user, item) = sum(user_vector * item_vector)
```

The same user vector is scored against the positive and sampled negative item
vectors. `forward` returns those two score tensors, one score per example in
the batch. Later we can add feature-processing layers if the ID-only model
provides a useful baseline.

### Numerically stable BPR loss

The model uses the pairwise BPR objective defined in Step 2. In code it is
computed as `softplus(negative_score - positive_score)`, which is equivalent
to `-log(sigmoid(positive_score - negative_score))` but behaves more reliably
for large score differences. The loss is averaged over a batch. A lower loss
means the positive scores tend to exceed their paired negative scores; it is a
training objective, not a ranking metric like Recall@10 or NDCG@10.

### Scope and files

The dataset, model, and loss are in `src/models/two_tower.py`. Step 3 defines
the model and its inputs; Step 4 adds the optimizer, epochs, and checkpoint
selection. This first model uses only ID embeddings, without text or other
item features.

Focused tests in `tests/test_two_tower.py` check the returned triple, tensor
score shapes, finite scores, and that BPR loss favors positive-over-negative
scores. Run them in the configured environment with:

```bash
pytest tests/test_two_tower.py -q
```

The test file goes further: it verifies exact dot products using manually
assigned vectors; rejects malformed pairs, bad dimensions, mismatched score
shapes, empty or non-finite scores, and integer scores; checks loss remains
finite for very large score gaps; and performs one inspected gradient step to
confirm the positive/negative margin improves. A two-worker test checks
negative streams differ between workers but repeat when the same DataLoader
seed is reused. These are correctness checks on small examples. They do not
show that the model has learned useful patterns from Amazon data.

Step 4 below connects the encoded Parquet data, sampler, dataset, and model in
an explicit training loop with batches and optimizer updates.

## Step 4: train with mini-batches and choose a checkpoint

### Bounded local data sample

`src/models/train_two_tower.py` reads the encoded training and validation
Parquet directories. It first finds users present in both splits, shuffles
those candidates with the configured seed, and selects up to `--max-users`.
For every selected user, it loads the user's complete training history. It
never truncates a selected user's history: the negative sampler must know all
training items seen by that user or it could label a known positive as a
negative.

`--max-train-pairs` bounds the total in-memory training examples. The loader
adds complete user histories until the budget is reached, skipping a user if
their whole history would exceed it. It also remaps selected users to
contiguous model indices `0..n-1`. Item indices stay in the complete training
catalog so the checkpoint's item embeddings line up with `item_mapping`.
Defaults of 2,000 users and 100,000 training pairs make this a development
experiment, not a full-data training run. Use less when memory is constrained.

### Mini-batch optimizer loop

For each epoch, PyTorch's `DataLoader` shuffles positive training examples.
When a positive pair is fetched, `PairwiseTrainingDataset` draws a fresh
training-only negative. The model scores positive and negative pairs, BPR loss
computes their ranking error, and gradient updates affect the selected user
and item rows. AdamW's weight decay also shrinks embedding rows with no gradient
in that batch. The script reports mean training loss and validation loss after
each epoch. The default device is CPU; `--device cuda` can be used
when a compatible CUDA PyTorch build and GPU are available.

The script does not open the test split. It chooses a checkpoint by the
lowest validation BPR loss. Validation negatives are generated once from a
fixed seed, excluding each selected user's training positives and validation
targets, so epoch-to-epoch loss changes mostly reflect model changes rather
than new negative draws. Validation loss remains a sampled pairwise training
objective; Step 5 evaluates the saved checkpoint with full-catalog ranking
metrics.

### Checkpoint and history

The saved `best_model.pt` contains the best model state, architecture size,
training settings, selected source user IDs, item mapping location, selected
epoch, and validation loss. Keeping the selected-user map is necessary because
the checkpoint's user indices refer to the sampled subset, while the source
Parquet has the full training vocabulary. `history.json` stores the per-epoch
losses and sample sizes for review. The test split remains untouched for a
later final evaluation step.

Run the full input mapping step first, then train using its output directory:

```bash
python -m src.models.prepare_tower_data \
  --data data/processed/movies_tv_5m \
  --output data/processed/movies_tv_5m/two_tower/v1

python -m src.models.train_two_tower \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --output artifacts/two_tower/movies_tv_5m/v1 \
  --max-users 2000 \
  --max-train-pairs 100000 \
  --epochs 10 \
  --seed 42
```

The training command requires Spark to read the encoded Parquet files. The
Windows Spark runtime needs Hadoop's Windows `winutils.exe` for some local
filesystem operations and may fail to list Parquet files with `NativeIO` even
when the Java process starts. If Spark reports either Windows error, use the
WSL Python 3.11+ and Java 21 environment
documented in the preprocessing notes. PySpark itself currently documents
support for Python 3.10+ and Java 17+; the project metadata now allows Python
3.11 to match that WSL setup.

### Checks for this step

`tests/test_train_two_tower.py` trains for a few epochs on a tiny synthetic
dataset. It checks that epoch metrics are produced, the best checkpoint is
saved, and reloading that checkpoint returns the same scores as the in-memory
best model. Run it with:

```bash
pytest tests/test_train_two_tower.py -q
```

The synthetic test verifies the training mechanics, not recommendation quality
on Amazon data. We have also completed one bounded run on real interactions;
its configuration and results are recorded next. Do not use the reported
validation BPR loss as a substitute for Recall@10 or NDCG@10.

### First real-data training run (2026-10-04)

We trained from the mapped `movies_tv_5m` data in WSL and wrote artifacts to
`artifacts/two_tower/movies_tv_5m/v1`. This was a bounded development run, not
a full-data or production run. It used seed 42, CPU, 16 numbers per user/item
vector, batches of 512, 10 epochs, and a limit of at most 2,000 users and
100,000 training interactions.

The actual sample contained 2,000 users, 21,938 complete training
interactions, and 2,000 validation interactions, with the full 137,070-item
catalog. We used fewer than the 100,000 interaction limit because that number
is a ceiling, not a target: the 2,000 selected users had 21,938 training rows
in total, and the job keeps their full histories rather than cutting a user's
history in half. The model therefore learned user vectors only for those
2,000 selected users. It has vector slots for all 137,070 catalog items,
though this sample may not have updated every item's vector. This checkpoint
cannot yet provide a personalized score for every user in the full dataset; a
later larger training run or a fallback is needed for users outside this
subset.

The mean training loss fell from 0.693145 in epoch 1 to 0.564963 in epoch 10.
The sampled validation loss fell from 0.693146 to 0.688978; epoch 10 had the
lowest validation loss and was saved as the best checkpoint. This is a small
validation improvement, so the run confirms that training and checkpointing
work, but does not yet show that the model produces useful recommendations.
The validation loss compares each held-out item with one sampled alternative;
it does not measure whether the held-out item appears near the top of a ranked
catalog list.

The run saved `best_model.pt` (about 8.9 MB) and `history.json` with the full
loss curve, pair counts, user count, and catalog size. We reloaded the
checkpoint and verified that its user embedding table is 2,000 by 16, its
item table is 137,070 by 16, and all weights are finite. The checkpoint also
stores the selected source user indices and the item-mapping path so later
scoring can translate between source IDs and the sampled model indices.

The validation ranking evaluation is described in Step 5 below. The final
test split remains unused until model choices are finished.

## Step 5: measure validation ranking quality

### What the evaluator does

`src/models/evaluate_two_tower.py` loads the saved checkpoint and reads only
the encoded `train` and `validation` folders. It uses the checkpoint's saved
list of 2,000 source user indices to find those users' complete training
histories and held-out validation items. It never opens `test`.

For each selected user, it scores all 137,070 catalog items, but hides items
the user already interacted with during training. This reflects the
recommendation task: rank new candidates, rather than suggest an item already
in the user's history. The evaluator also removes a validation target if that
item was in that user's training history. There were no such overlapping
targets in this run.

It makes two rankings over the same users, candidates, and validation targets:

1. **Two-tower:** scores each item with the saved user and item vectors.
2. **Popularity baseline:** sorts items by how often they appeared in the same
   21,938 training interactions used for the sampled model, breaking ties by
   item index. It is a simple check of whether learned user-specific vectors
   improve on recommending generally frequent items.

The output metrics are macro-averaged across the 2,000 evaluated users:

- **Recall@K** is the share of each user's relevant validation items found in
  their first K recommendations, then averaged across users. Each user in this
  run had one validation target, so Recall@10 of 1% means 20 of 2,000 users had
  that target somewhere in their top 10.
- **Precision@K** is the share of the first K recommendations that are
  relevant, averaged across users. At K=10, 0.001 means 20 relevant
  recommendations out of 20,000 displayed candidate slots.
- **NDCG@K** also rewards a relevant item more when it appears nearer the top.
  It is useful when order matters, not only whether the item appears in the
  list.

### Run command and observed results

Run this from the project root, in the WSL Python/Spark/Java environment used
for mapping and training:

```bash
python -m src.models.evaluate_two_tower \
  --checkpoint artifacts/two_tower/movies_tv_5m/v1/best_model.pt \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --output artifacts/two_tower/movies_tv_5m/v1/validation_metrics.json \
  --k 10 20 \
  --batch-size 128
```

The validation report contains 2,000 users, 2,000 eligible validation
targets, and the full 137,070-item catalog. Results from this run:

| Method | K | Recall@K | Precision@K | NDCG@K | Validation hits |
|---|---:|---:|---:|---:|---:|
| Two-tower | 10 | 1.00% | 0.100% | 0.00496 | 20 / 2,000 |
| Popularity, same sample | 10 | 1.55% | 0.155% | 0.00754 | 31 / 2,000 |
| Two-tower | 20 | 1.45% | 0.0725% | 0.00610 | 29 / 2,000 |
| Popularity, same sample | 20 | 2.40% | 0.120% | 0.00975 | 48 / 2,000 |

The simple popularity baseline performed better than this first two-tower
model on all three reported metrics at both K values. That means the present
model has not yet shown a recommendation-quality improvement over the simpler
baseline. This is useful evidence: training loss went down, but lower training
loss did not translate into better top-of-list recommendations in this
experiment. We should tune or improve the model using validation and compare
again before considering it a candidate. These are results for a sampled set
of 2,000 users, not for every user in the dataset.

The full JSON report is saved locally at
`artifacts/two_tower/movies_tv_5m/v1/validation_metrics.json`; model artifacts
under `artifacts/` are ignored by Git. Ranking behavior and the split-loading
rules have tests in `tests/test_evaluate_two_tower.py`. The full WSL suite at
that Step 5 checkpoint reported **72 passed**. The later suite with MLflow and
ALS ranking tests reports **77 passed** in the linked test report. Its only
warning is PySpark's pandas compatibility notice; it did not fail a test.

At this historical step, test had not yet been evaluated. The larger
validation-only experiments below tested whether a broader user sample
changed the comparison. The final test result and its limitation are recorded
at the end of this guide; the current sample's test split is now consumed.

### Larger training sample: 10,000 users

To test the small-sample limitation, we trained a separate checkpoint in
`artifacts/two_tower/movies_tv_5m/v2` with the same architecture, embedding
size, optimizer, seed, and 10 epochs, but allowed up to 10,000 users and
200,000 training pairs. The sampler selected 10,000 users, 110,923 complete
training interactions, and 10,000 validation targets. The pair limit is a
maximum; complete user histories were kept. The catalog remained 137,070
items. Test data was not read.

Training BPR loss fell from 0.693145 at epoch 1 to 0.345362 at epoch 10;
sampled validation BPR loss fell from 0.693118 to 0.593487. The best
validation-loss checkpoint was epoch 10.

| Method | K | Recall@K | Precision@K | NDCG@K | Validation hits |
|---|---:|---:|---:|---:|---:|
| Two-tower | 10 | 2.25% | 0.225% | 0.01163 | 225 / 10,000 |
| Popularity, same sample | 10 | 2.09% | 0.209% | 0.01029 | 209 / 10,000 |
| Two-tower | 20 | 3.34% | 0.167% | 0.01436 | 334 / 10,000 |
| Popularity, same sample | 20 | 2.99% | 0.150% | 0.01252 | 299 / 10,000 |

On this 10,000-user validation sample, the two-tower model beats popularity
on all three metrics at K=10 and K=20. This is a positive result for the
larger run, but the absolute Recall@10 is still 2.25%, the cohort differs
from v1, and it is only one seeded experiment. We then ran a second
10,000-user experiment with seed 7 and added validation slices by training
history and target popularity. Those results and their limitations are in
the linked test report. The machine-readable report is saved locally at
`artifacts/two_tower/movies_tv_5m/v2/validation_metrics.json`.

These validation checks are complete for this iteration: overall and cohort
Recall/Precision/NDCG at K=10 and K=20 were compared with popularity on two
seeds. The result is mixed across seeds and user/item groups, so this model is
not yet a stable quality-gate winner. Cold-start pairs with unknown IDs were
excluded during mapping and are counted in Step 1; they cannot receive a
ranking score from this ID-only model. Do not evaluate test until a model
configuration has been selected. Step 6 below records training and evaluation
runs in MLflow so these comparisons are reproducible.

## Step 6: track runs with MLflow

The training and validation commands now log an MLflow run automatically.
By default they use experiment `two-tower-recommender` and a local SQLite
tracking database at `mlflow.db` in the project root. `.gitignore` excludes
the database and local MLflow artifacts. Training runs record model settings,
seed, sample sizes, per-epoch training and validation BPR losses, the best
epoch, `history.json`, and the checkpoint. Evaluation runs record overall
and cohort Recall/Precision/NDCG at each requested K and attach the complete
validation JSON report. The evaluator reads train and validation only.

Override the experiment name or store location with
`--mlflow-experiment NAME` and `--mlflow-tracking-uri URI`. For example, use
`sqlite:////home/akshay/recommender-tracking.db` to store the database in the
WSL home directory instead of the project folder. Run the MLflow UI from the
project root in WSL to compare logged runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000` in a browser. The current SQLite store has
the seed-42 evaluation and seed-7 training/evaluation runs. The first seed-42
training run occurred before logging was added, so its epoch history remains
in the local JSON report rather than MLflow. New CLI runs log both training
and evaluation by default.

The test `tests/test_experiment_tracking.py` uses a temporary SQLite database
to verify parameters, final metrics, epoch-step metrics, and artifact upload.
The latest complete WSL suite passes **100 tests** (25.72 seconds; one
non-failing PySpark/Pandas compatibility warning and one mounted-filesystem
pytest cache warning). MLflow records are local
experiment evidence; this step does not register or deploy a production model.

## Step 7: compare implicit ALS with popularity

The v3 plan calls for a standard collaborative-filtering model between
popularity and the two-tower model. This project uses Spark ML's implicit ALS:
it factorizes user/item interaction counts, producing one vector per user and
item. A user's score for a product is the dot product of those vectors.
Repeated interactions increase the confidence weight; they are not treated as
explicit star ratings.

### Run the comparison

The evaluation uses the same 10,000 validation users as the seed-42 two-tower
checkpoint so the held-out users and validation targets align across those
experiments. If you are starting from a fresh checkout, prepare the data and
mappings as described above, then create that checkpoint with:

```bash
python -m src.models.train_two_tower \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --output artifacts/two_tower/movies_tv_5m/v2 \
  --max-users 10000 --max-train-pairs 200000 \
  --epochs 10 --seed 42
```

In WSL, export the cohort IDs from the checkpoint:

```bash
python -c 'import json; from pathlib import Path; from src.models.train_two_tower import load_two_tower_checkpoint; _, metadata = load_two_tower_checkpoint(Path("artifacts/two_tower/movies_tv_5m/v2/best_model.pt")); Path("artifacts/two_tower/movies_tv_5m/als_validation_users.json").write_text(json.dumps(metadata["model_user_ids"]), encoding="utf-8")'
```

Then run ALS from the project root in the WSL Spark environment:

```bash
python -m src.models.collaborative_filtering \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --output artifacts/two_tower/movies_tv_5m/als_v2 \
  --user-ids artifacts/two_tower/movies_tv_5m/als_validation_users.json \
  --rank 16 --max-iter 10 --reg-param 0.1 --alpha 1.0 --seed 42 \
  --k 10 20 --shuffle-partitions 32
```

The job trains on all **2,687,407 encoded training interactions** and
evaluates on 10,000 validation users. The popularity comparator is calculated
from that same full training split. The saved Spark model and JSON report are
written under the new `als_v2` output directory. MLflow stores the
hyperparameters, ranking metrics, report, and a ZIP archive of the Spark
model. The user-ID JSON file
is generated from the two-tower checkpoint solely to reuse its evaluation
cohort; it contains integer indices, not personal user data.

### Results and interpretation (2026-10-04)

The model learned factors for 238,645 users and all 137,070 items. It had a
factor for every one of the 10,000 validation targets, with no seen-target
rows removed. Full-catalog ranking results were:

| Method | K | Recall@K | Precision@K | NDCG@K |
|---|---:|---:|---:|---:|
| Implicit ALS | 10 | 0.56% | 0.056% | 0.00285 |
| Full-train popularity | 10 | 2.08% | 0.208% | 0.01068 |
| Implicit ALS | 20 | 0.91% | 0.0455% | 0.00372 |
| Full-train popularity | 20 | 3.29% | 0.1645% | 0.01368 |

ALS did not beat popularity in this run. This is validation evidence, not a
reason to tune against test. Rank and regularization are one initial
configuration; any later changes should be deliberate validation experiments
with each run logged. Spark ML ALS does not expose the same per-epoch
validation-loss history as the two-tower loop; this run records its fixed
10-iteration setting and final validation ranking metrics. The two-tower run was trained on the selected
10,000-user sample, while ALS used the full training split, so compare each
method against its recorded training scope before drawing a direct
model-family conclusion. The held-out test split remains untouched.

The full command and report are also recorded in
[`two_tower_test_report.md`](two_tower_test_report.md). Run the focused checks
with `pytest tests/test_collaborative_filtering.py -q`; run the full suite
using the WSL instructions above.

## Completed step: fair comparison across the three recommenders

The earlier results did not answer which of the three methods is best under
the same conditions. We therefore ran the controlled validation experiment
below before choosing further model changes.

### 1. Use one fixed user cohort and one data split

The command `src.models.prepare_comparison_cohort` selected 10,000 users who
have training history and a validation target. It saved their mapped
`user_idx` values in one JSON file and retained each selected user's complete
training history. The selection seed is 42. The manifest records the counts
and SHA-256 hashes, so later runs can verify that cohort and candidate inputs
have not changed.

The generated files are
`artifacts/comparisons/movies_tv_5m/cohort_v1/cohort.json`,
`candidate_items.json`, and `manifest.json`. They are ignored local outputs.

### 2. Match training data and candidate items across models

All three methods were trained/scored from the same selected users' training
interactions. The candidate catalog is the unique item IDs in those rows.
Validation targets outside that catalog are counted as misses in the same
10,000-target denominator: 7,387 targets are rankable and 2,613 are outside
the sample catalog. This prevents a model from looking better by silently
discarding difficult targets.

### 3. Repeat learned models across seeds

The cohort, training rows, validation targets, candidate catalog, and model
settings were fixed. ALS and two-tower ran with seeds 42, 7, and 123. The
seed changes learned parameters but not the cohort or split. Popularity is
deterministic for the fixed rows. Results are in the linked report; three
seeds show run-to-run variation and do not establish statistical
significance.

Evaluation records Recall@K, Precision@K, and NDCG@K overall and by user
history and target popularity. The cohort manifest and reports record the
user/target/catalog/coverage counts and seeds. New tests cover fixed candidate
filters and ALS source-user to local-factor index mapping. The test split was
not loaded in the experiment.

### 4. Keep test data sealed during selection

All comparisons, seed repeats, cohort analysis, and parameter changes use
validation only. Training and evaluation commands for this phase must read
the train and validation folders and must not open `test`. After the quality
gate and model settings are chosen from validation, freeze the candidate and
run the test evaluation once as a final estimate. If the test result leads
to another model change, that test result is no longer an unbiased final
estimate; a new untouched test set would be needed.

The completed outputs are the versioned comparison JSON/CSV and six learned
model runs in the ignored `artifacts/comparisons/movies_tv_5m/` directory.
The full commands for regenerating the cohort, running each model, and
summarizing results are in the test report's “Reproduction and tracking”
section. The fresh candidate passed the written validation gate and received
one test evaluation. Its result is in the report; the current sample's test
split is consumed and must not be used for tuning.

## Next step: freeze a candidate and apply the quality gate

The three-seed comparison was used to understand variation. A fresh,
fixed-seed candidate was then trained on the same cohort and evaluated on
validation. The project gate was **NDCG@10 at least 5% above popularity's
validation NDCG@10, and Recall@10 no lower than popularity**. The candidate
passed both. This is a practical project gate, not a statistical significance
test. Since earlier experiments had already inspected the same cohort's
validation results, the later test score is a project checkpoint rather than
an untouched unbiased estimate.

To create the candidate from the existing training split and cohort:

```bash
python -m src.models.train_two_tower \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --output artifacts/model_selection/two_tower_seed42 \
  --user-ids-file artifacts/comparisons/movies_tv_5m/cohort_v1/cohort.json \
  --max-users 10000 --max-train-pairs 200000 --epochs 10 --seed 42

python -m src.models.evaluate_two_tower \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --checkpoint artifacts/model_selection/two_tower_seed42/best_model.pt \
  --candidate-items artifacts/comparisons/movies_tv_5m/cohort_v1/candidate_items.json \
  --output artifacts/model_selection/two_tower_seed42/validation_metrics.json \
  --split validation --k 10 20
```

The fresh checkpoint passed both conditions, so it was frozen and evaluated
once with `--split test`. Results: test Recall@10 was 1.490% for two-tower
versus 1.399% for popularity; Recall@20 was 2.335% versus 2.184%. There were
9,934 eligible test users, and 2,947 targets were outside the sampled catalog.
The test report is in the linked report's “Frozen-candidate test evaluation”
section.

The test score has a limitation: the cohort and seed-42 validation scores had
already been inspected in earlier experiments, and the numeric gate was set
after that history. Treat this as a final project checkpoint, not a clean
unbiased estimate. Do not tune against it. A future unbiased estimate needs
a new holdout set that has not informed model experiments.

### Run a visible synthetic demo

To watch the model train and print top recommendations without Spark or the
Amazon dataset, run:

```bash
python -m src.models.demo_two_tower
```

The example creates two groups of users with overlapping item patterns. It
prints training and sampled validation loss for each epoch, then shows each
user's top three item scores. This demonstrates the data flow, embedding
updates, and retrieval step. Its tiny repeated preferences and sampled
validation pairs are for illustrating model mechanics; the output is not an
estimate of Amazon recommendation quality and is not a held-out benchmark.
