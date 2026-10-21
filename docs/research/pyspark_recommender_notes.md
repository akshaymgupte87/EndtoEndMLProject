# PySpark Recommender Study Notes

These notes explain the decisions and PySpark concepts used in the recommender notebook.

## Reading order

- Start with [README.md](../README.md) for the current project state.
- Use this file for Spark concepts, data preparation, time splits, and WSL
  troubleshooting.
- Continue with the [two-tower study guide](two_tower_study_guide.md) for
  model setup, run order, method flow, and test commands.
- Use the [test and training report](two_tower_test_report.md) for current
  test outcomes, epoch traces, and validation findings.
- Use [PLAN.md](../PLAN.md) to see what remains.

The data and model artifacts are generated locally and are ignored by Git. A
fresh checkout must download the review data and recreate the prepared splits,
ID mappings, checkpoints, and MLflow database before those commands can run.

## Understanding the quality aggregation

```python
quality = reviews.agg(
    F.count("*").alias("rows"),
    F.sum(F.col("user_id").isNull().cast("int")).alias("null_users"),
    F.sum(F.col("item_id").isNull().cast("int")).alias("null_items"),
    F.sum(F.col("timestamp").isNull().cast("int")).alias("null_timestamps"),
    F.sum(F.col("rating").isNull().cast("int")).alias("null_ratings"),
    F.countDistinct("user_id").alias("distinct_users"),
    F.countDistinct("item_id").alias("distinct_items"),
    F.min("rating").alias("minimum_rating"),
    F.max("rating").alias("maximum_rating"),
).collect()[0]
```

`reviews.agg(...)` applies aggregate functions across the entire Spark DataFrame and returns a one-row Spark DataFrame.

`F.count("*")` counts all rows. The null checks use three steps:

```python
F.col("user_id").isNull()       # True or False for each row
F.col("user_id").isNull().cast("int")  # True becomes 1, False becomes 0
F.sum(...)                       # adds the 1 values
```

For example, `null_items` is the number of rows where `item_id` is missing.

`F.countDistinct("user_id")` and `F.countDistinct("item_id")` count unique users and products. `F.min("rating")` and `F.max("rating")` show the rating range.

The aggregate result is still a Spark DataFrame until `collect()` is called. Because the aggregation returns only one row, collecting it is safe. `[0]` extracts that first row as a Spark `Row` object, which can be accessed like this:

```python
quality["rows"]
quality["distinct_users"]
quality["minimum_rating"]
```

## PySpark performance principles

PySpark follows many SQL performance principles, but its execution is distributed and lazy.

### Spark is lazy

Transformations such as `select`, `where`, `join`, and `dropDuplicates` describe a computation but usually do not execute it immediately. Actions such as `count`, `show`, `collect`, and `write` trigger execution.

### Avoid repeated actions

This can recompute the same pipeline several times:

```python
interactions.count()
interactions.select("user_id").distinct().count()
interactions.select("item_id").distinct().count()
```

If the DataFrame will be reused, persist it once:

```python
interactions = interactions.persist()
```

For a machine with limited memory, disk persistence is safer:

```python
from pyspark import StorageLevel

interactions = interactions.persist(StorageLevel.DISK_ONLY)
```

Caching is useful only when the same DataFrame is reused. It should not be added automatically to every DataFrame.

### Select and filter early

Load only the columns needed by the recommender and remove invalid rows early. This reduces memory use, file scanning, and shuffle traffic.

### Avoid large `collect()` calls

Collecting one summary row is safe:

```python
quality = reviews.agg(...).collect()[0]
```

Collecting millions of review rows is unsafe because it moves all data into the notebook process:

```python
reviews.collect()  # avoid for large DataFrames
```

### Understand shuffles

Operations such as `groupBy`, `dropDuplicates`, `distinct`, joins, sorting, and window functions can move data between partitions. This network-like movement is called a shuffle and is often the most expensive part of a Spark job.

Inspect the plan with:

```python
interactions.explain("formatted")
```

`Exchange` usually indicates a shuffle. `Sort`, `SortMergeJoin`, and large scans can also be important cost centers.

### Tune local shuffle partitions

For this local 5M-row experiment, the notebook uses:

```python
.config("spark.sql.shuffle.partitions", "32")
```

Too many partitions create scheduling overhead. Too few create oversized tasks. The right value depends on data size, CPU cores, and memory.

### Prefer Parquet after preparation

JSONL is expensive to parse repeatedly. After preparation, read the Parquet output for later notebook work:

```python
interactions = spark.read.parquet(
    "data/processed/movies_tv_5m/interactions"
)
```

Parquet is columnar and supports column and predicate pruning, so later operations can read less data.

### Practical rule for this project

For the 5M interaction experiment:

1. Read only required columns.
2. Filter invalid rows early.
3. Expect `groupBy`, `distinct`, joins, and windows to shuffle.
4. Avoid repeated actions unless the DataFrame is persisted.
5. Write cleaned data to Parquet and use it for subsequent experiments.

## Why preprocessing steps are ordered

Preprocessing turns raw events into examples a recommender can learn from.
The order follows data dependencies: each operation should have the valid,
well-defined inputs it needs, and evaluation information must not leak into
training.

### 1. Load interactions

Read the source and select the fields needed for the task, such as user ID,
item ID, timestamp, and rating. Confirm the schema before relying on field
names or types.

### 2. Remove invalid and duplicate rows

An event without a user, item, or usable timestamp cannot reliably represent a
user-item interaction or be assigned to a time split. Remove invalid rows
before counting activity or selecting each user's latest interactions.
Deduplicate before splitting so repeated copies of an event do not distort
counts or determine which event appears latest.

### 3. Choose the interaction signal and eligible population

Decide what an event means to the model. For explicit rating prediction,
ratings are the target. For implicit recommendation, define which events count
as positive feedback, such as any review or ratings above a chosen threshold.
This is a modeling decision, not a universal rule, and should be documented.

Filtering users and items with very few interactions can provide more
collaborative evidence and more stable evaluation. The threshold affects the
size and representativeness of the data, especially cold-start coverage. The
current notebook keeps users and items with at least five interactions.

### 4. Create a temporal evaluation split

For each user, use earlier events for training, a later event for validation,
and the most recent event for testing. This simulates using past behavior to
recommend what the user may choose later. Split after invalid-row removal and
deduplication so malformed or repeated events do not affect chronology.

### 5. Fit ID mappings and features using training history

ALS and embedding models commonly need numeric user and item indices. Build
the mappings from the training population, preserve them for serving, and
map original IDs back for API responses. Compute data-dependent aggregates
and features using training history only; validation/test behavior must not
influence training inputs.

### Leakage caveat in the current notebook

The current exploratory notebook filters users and items by interaction
counts across the full 5M-row sample before making the temporal split. This
is convenient for learning and initial model development, but validation and
test activity influences which users/items remain. A stricter benchmark
splits first, learns frequency thresholds and ID mappings from training data,
then applies those mappings to validation and test. Document which approach
was used when reporting metrics.

The ordering is a dependency chain rather than an unchanging recipe:

```text
raw records
→ valid, unique events
→ define feedback and eligible population
→ time-respecting evaluation split
→ fit mappings and features on training history
→ train and evaluate
```

For each preprocessing step, ask:

1. What later operation depends on this result?
2. Could this calculation use information from the future?

Those questions help determine both a sensible order and where leakage could
enter the experiment.

## Save prepared splits as partitioned Parquet

After the split is created, save it so later experiments can start from
prepared data instead of reparsing the large JSONL and repeating cleaning,
filtering, and window operations. Parquet stores columns efficiently and
Spark can read only the columns a later step needs.

If the batch preparation job has already written `interactions`, `train`,
`validation`, and `test` Parquet directories, load those outputs in the
notebook instead of rebuilding and writing the same lineage again:

```python
interactions = spark.read.parquet(str(OUTPUT_PATH / "interactions"))
train = spark.read.parquet(str(OUTPUT_PATH / "train"))
validation = spark.read.parquet(str(OUTPUT_PATH / "validation"))
test = spark.read.parquet(str(OUTPUT_PATH / "test"))
```

This avoids repeating JSON parsing, filtering, and temporal window work. A
Spark error saying a write job was cancelled because the `SparkContext` was
shut down is a final symptom; inspect earlier driver logs for the original
failure. If the prepared outputs already exist and contain `_SUCCESS`, use
them rather than rerunning the expensive write in the notebook.

The preparation script writes separate Parquet directories for each split.
Keep the raw JSONL separate and unchanged. On Windows, local Hadoop-backed
writes may require the matching `winutils.exe`; running the Spark job in
Linux/WSL avoids that Windows-specific filesystem requirement.

The notebook also needs to read those local Parquet files. On Windows, Hadoop
can fail while listing files with
`UnsatisfiedLinkError: NativeIO$Windows.access0` when the matching native
Hadoop library is unavailable. This is a Windows JNI/native-library problem,
not a corrupt Parquet dataset. Run the notebook using the WSL Miniconda kernel
and Java 21 so the Linux Spark runtime reads the files through Linux Hadoop
filesystem code. Set `SPARK_HOME` to the PySpark package belonging to that
same Python environment; otherwise a separate `spark-submit` earlier on
`PATH` can load mismatched Spark/Scala jars.

## Production-shaped workflow at small scale

Production structure can be emulated without processing the entire corpus in
every notebook cell. Use a 5M-row sample for interactive development and a
separate batch job for the full downloaded category.

```text
data/raw/
    Movies_and_TV.jsonl

data/processed/
    bronze/       cleaned interactions
    silver/       filtered interactions
    gold/         train/validation/test features
```

The raw JSONL should be treated as immutable. Cleaning and feature generation
should write new Parquet outputs instead of modifying the raw file. Version
the outputs when experiments need to be reproduced:

```text
data/processed/movies_tv_5m/v1/
data/processed/movies_tv_full/v1/
```

The recommended development flow is:

```text
5M notebook sample
    ↓
model experiments and evaluation
    ↓
full-category batch preprocessing
    ↓
MLflow tracking
    ↓
FastAPI + Redis serving
```

This is production-shaped because it separates raw data, offline batch
processing, model artifacts, and online serving. A real petabyte-scale system
uses object storage, partitioned Parquet or Delta tables, distributed Spark
executors, incremental processing, precomputed embeddings, and an ANN index.
The project currently uses `Spark local[*]`, Parquet, a two-tower model, and
local MLflow. FastAPI, optional Redis caching, Docker Compose, Prometheus,
OpenTelemetry, Kubernetes manifests, Terraform for ECS/Fargate, an Airflow
DAG, Kafka event scripts, and a cache-outage exercise are implemented as
teaching code/configuration. Their unit tests or syntax checks cover only the
pieces described in [the deployment walkthrough](deployment_walkthrough.md).
Docker Compose was run end to end locally; Kubernetes, Airflow, Kafka, and AWS
have not. See [PLAN.md](../PLAN.md) for the verification status of each
component.

## Temporal train/validation/test split

For recommendation, split each user's interactions by time. The older
interactions train the model, the next most recent interaction validates
model choices, and the most recent interaction is held back for final testing.
This simulates recommending from a user's past to predict a later interaction.

```python
from pyspark.sql import Window

user_time_order = Window.partitionBy("user_id").orderBy(
    F.col("timestamp").asc(),
    F.col("item_id").asc(),
)
user_rows = Window.partitionBy("user_id")

ranked = (
    interactions
    .withColumn("position", F.row_number().over(user_time_order))
    .withColumn("user_interaction_count", F.count("*").over(user_rows))
)
```

`Window.partitionBy("user_id")` handles each user's history independently.
The time ordering assigns position 1 to the earliest interaction. `item_id`
breaks timestamp ties consistently. `row_number()` creates each user's
chronological position, and the second window expression adds that user's
interaction count to each row.

The split conditions are:

```text
position < user_interaction_count - 1  → train
position = user_interaction_count - 1  → validation
position = user_interaction_count      → test
```

Because earlier preprocessing requires at least five interactions per user,
every user has at least three training rows, one validation row, and one test
row. This temporal split reduces future-information leakage compared with a
random row split. The ranking and per-user window operations require Spark to
shuffle and sort records by user, so this can be one of the more expensive
preprocessing steps.

### Verify split coverage and chronology

After constructing the splits, audit that users have the expected rows and
that their latest training timestamp does not occur after validation, and
validation does not occur after test. This catches split mistakes before
model evaluation.

```python
train_times = train.groupBy("user_id").agg(
    F.max("timestamp").alias("last_train_ts")
)
validation_times = validation.select(
    "user_id", F.col("timestamp").alias("validation_ts")
)
test_times = test.select("user_id", F.col("timestamp").alias("test_ts"))

audit = (
    train_times
    .join(validation_times, "user_id", "left")
    .join(test_times, "user_id", "left")
    .agg(
        F.sum(F.col("validation_ts").isNull().cast("int")).alias("missing_validation"),
        F.sum(F.col("test_ts").isNull().cast("int")).alias("missing_test"),
        F.sum((F.col("last_train_ts") > F.col("validation_ts")).cast("int")).alias("train_after_validation"),
        F.sum((F.col("validation_ts") > F.col("test_ts")).cast("int")).alias("validation_after_test"),
    )
    .collect()[0]
)
```

The audit joins one timestamp summary per user and checks missing split rows
and out-of-order timestamps. Zero chronology violations supports the intended
past-to-future evaluation. Timestamp ties are allowed by the chronological
comparison; the split's secondary `item_id` sort key makes their assignment
deterministic.

Do not assume the original user-frequency filter guarantees split coverage.
Filtering items can remove many of a user's events afterward. A user may then
have only one remaining event, leaving no training or validation example.
Inspect missing split counts and, if the modeling protocol requires all three
splits for every user, reapply a minimum-interaction filter after item
filtering or use an iterative user/item k-core filter.

For a local exploratory run, audit distinct user counts per split first. A
full chronology audit that joins timestamp summaries for every user can add
another expensive Spark job. If resources are tight, inspect ordering on a
bounded sample of users and clearly label it as a sample check; it does not
prove the invariant for every user. If the JVM exits during a Spark action,
Py4J may subsequently report `ConnectionRefusedError`. That is a symptom that
the Java process has stopped, often from resource pressure, rather than a
network connection that needs to be retried unchanged.

## Diagnosing Spark heap exhaustion in the notebook

The first error is the important one:

```text
java.lang.OutOfMemoryError: Java heap space
```

Here, Spark exhausted the Java driver's heap while sorting/shuffling the
per-user window split. The later `Connection reset` and
`ConnectionRefusedError` messages occur because the Spark JVM terminated;
they are consequences, not a separate network problem.

The notebook had a default Java heap, while the successful batch command used
an 8 GB driver. Configure the heap before PySpark starts its Java gateway:

```python
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 8g pyspark-shell"
```

Changing this after Spark's Java process has started does not resize that JVM.
Restart the notebook kernel after changing the setup cell.

The split DataFrames are lazy. Calling `train.count()`,
`validation.count()`, and `test.count()` separately triggers Spark jobs and
can recompute the upstream window/sort/shuffle lineage for each action. Avoid
those repeated counts in an interactive notebook. The prepared data already
exists under `data/processed/movies_tv_5m/`; load its Parquet directories
instead of repeating the full computation. This keeps the planned split logic
visible while using the outputs of the batch preparation step for subsequent
work.

## Efficient execution for the 5M-row notebook

The optimized notebook preserves the preprocessing and split calculations:

- Supply only the four required fields in an explicit JSON schema. This avoids
  a separate schema-inference scan of the full JSONL. Missing or incompatible
  fields can become null under the JSON reader's permissive mode; inspect the
  quality summary before proceeding.
- Persist the projected reviews, cleaned interactions, and filtered interactions
  only while they are reused. MEMORY_AND_DISK caches them in Spark memory and
  falls back to disk when necessary. It does not make arbitrary workloads immune
  to heap exhaustion.
- Use left-semi joins for eligibility filters: retain an interaction when its
  user/item exists in the eligible set, without carrying columns from that set.
- Compute the filtered row/user/item counts together in one aggregation.
- Cache the shared window result and count the split labels in one aggregation.
  This actually executes and verifies split sizes, while avoiding three separate
  computations of the same window sort. Collect only the three summary rows.
- Release upstream caches after the downstream result is materialized.
- For later model experiments, restart and run setup followed by the prepared
  Parquet-loading cell. This skips all raw-data preprocessing.

The current starting settings are a 16 GB Java heap, local[8], and 128 initial
shuffle partitions with adaptive query execution enabled. Eight concurrent
workers bound simultaneous task memory demand. Adaptive execution may merge
small shuffle partitions. These settings are tuning starting points; fastest
settings depend on WSL memory, CPU availability, storage, and the physical plan.
128 GB physical RAM does not automatically become Spark's Java heap, and the
WSL memory allowance must accommodate Java plus Python and operating-system use.
Restart the kernel before applying the new heap setting.

A single aggregate can still contain several stages and shuffles. Caching avoids
recomputing upstream data; it does not remove required grouping or sorting.
Do not claim an optimal configuration or a speedup until it is benchmarked.

Reference: [Apache Spark SQL performance tuning](https://spark.apache.org/docs/latest/sql-performance-tuning.html).

## Project steps completed so far

Keep this walkthrough updated as each roadmap step is implemented. Explain the
reason for each operation, how to run it, and how to check its behavior. The
README stays focused on the project overview and short commands.

## How the data and ML pipelines fit together

We have started with the **data pipeline**. The full ML pipeline is the next
layer that consumes its prepared output. Keeping these separate makes it clear
which steps transform source data and which steps learn or serve a model.

```text
Data pipeline
Raw Amazon reviews (JSONL)
        ↓
PySpark cleaning and filtering
        ↓
Chronological train / validation / test splits
        ↓
Parquet outputs
        │
        ▼
ML pipeline (partly implemented)
Training data ──→ popularity baseline
        └─────────→ PyTorch two-tower + negative sampling
                           ↓
              validation ranking and cohort metrics
                           ↓
                local MLflow experiment records
                           ↓
        one test evaluation (used for this sample)
                           ↓
              batch bundle → FastAPI / optional Redis
                           ↓
             Docker / observability / AWS ECS path
```

**Implemented:** the modeling path and the code/configuration for its batch
job, API, Docker/Compose stack, Prometheus/OpenTelemetry, Kubernetes, chaos
exercise, AWS/Terraform, Airflow, and Kafka event demo. A matched three-seed
comparison and one test checkpoint are documented in [the two-tower study
guide](two_tower_study_guide.md) and [test report](two_tower_test_report.md).
Two-tower was modestly ahead of popularity on this sample; its test split is
consumed and is not an untouched unbiased estimate.

**Remaining:** runtime verification of Docker/Compose, a Kubernetes cluster,
the Airflow scheduler, Kafka broker flow, a live cache-outage exercise, and
AWS deployment. The 17.4M-row scale is also unverified; current bounded model
steps collect selected data to the driver. Do not tune against the consumed
test split. The [study guide concept map](two_tower_study_guide.md#concept-map-what-this-project-teaches-so-far)
links each concept to its source and evidence.

## Batch processing: current job and production direction

The current `src/data/prepare.py` is our first batch preparation job. It runs
Spark locally with `local[*]`, reads JSONL, applies the cleaning/filtering and
temporal split logic, then writes Parquet. With a 5M-row dataset, the core
transformations can be the same as a production version, and a suitably sized
single machine may be enough. Production readiness is mainly about running
reliably, validating outputs, and making reruns safe; a cluster is not
automatically required just because the job is production-facing.

The differences to close are:

| Area | Current job | Production direction |
|---|---|---|
| Compute | Spark local mode on one machine | A managed Spark cluster when needed; one machine can still work at 5M rows if it has enough resources |
| Storage | Local JSONL input and Parquet folders | Object storage and versioned datasets or a table format |
| Reruns | Writes with `overwrite` to the selected output folders | Write to a new run path, validate it, then publish or mark it current |
| Validation | Helper-level tests and checks during exploration | Validate schema, counts, duplicates, split coverage, and chronology before publishing |
| Run record | Command arguments and Spark logs | Also record input version, code version, parameters, output paths, and counts in a manifest |

This means the 5M-row job does not need different modeling logic just to be
called production. A production-style version should write to an isolated run
directory, verify it, and only then expose it as the current prepared dataset.
It should not destroy the last successful output when a new run fails.

### Scaling the batch job to 17.4M rows

For a 17.4M-row dataset, keep the same high-level transformations but make
inputs, outputs, and execution settings configurable for the deployment
environment. A robust job should:

1. Read immutable JSONL from object storage with an explicit schema and record
   the source location/version.
2. Clean and normalize IDs, timestamps, and feedback fields. Track counts for
   invalid and duplicate records so changes are visible between runs.
3. Apply documented eligibility and split rules. For strict evaluation,
   derive user/item eligibility and mappings from training history, then apply
   them to validation and test to avoid using future activity.
4. Write Parquet (or a suitable table format) under a unique run/version path;
   do not overwrite the last successful dataset as part of the initial write.
5. Validate schema, nonempty outputs, row counts, duplicate keys, user
   coverage, and train/validation/test chronology. Write a manifest containing
   the input version, code version, parameters, output counts, and locations.
6. Publish the new output only after validation succeeds, then let a scheduler
   such as Airflow, Dagster, or a cloud-native workflow trigger and monitor the
   run.

At 17.4M rows, one machine may still be sufficient depending on row width,
available memory, shuffle volume, and storage speed. We would design the job
to run on Spark across a cluster when needed, while keeping the transformation
logic independent of local versus cluster deployment. The first production
iteration can evolve from this project by replacing local paths with object
storage paths, adding versioned writes and validation, and recording run
metadata. These production changes are a design direction, not features that
the current preparation script already provides.

## AWS deployment direction and current status

The project includes Docker packaging for the serving API and Terraform
configuration for an AWS ECS/Fargate path. Docker Compose has now been built
and exercised locally end to end, including API serving, metrics/traces, and a
Redis outage/recovery. This is local verification, not proof of AWS
deployment: Terraform has not been applied. See the dated evidence in the
[deployment walkthrough](deployment_walkthrough.md#end-to-end-run-record-2026-10-05).

A possible deployment shape is:

```text
Raw reviews in S3
        ↓
Containerized Spark batch job
        ↓
Versioned Parquet outputs in S3
        ↓
Containerized recommendation API on ECS/Fargate
        ↓
Clients

Terraform provisions the storage, compute, networking, IAM, logs, and
deployment resources. Lambda may trigger or coordinate small tasks.
```

The Spark preparation job and model inference should not be assigned to Lambda
by default. Spark has substantial JVM and memory needs, and the eventual
PyTorch model may need more than the account's available 512 MB per function.
A container service such as ECS on Fargate is a better starting point for a
long-running API, while a managed/container batch option is a better fit for
large preprocessing jobs. We can choose the exact AWS service after measuring
runtime, peak memory, and cost on our data. Lambda could still be useful for a
small S3 upload trigger, request routing, or workflow notification if that
function fits its memory, duration, and package constraints.

AWS's published Lambda quota currently gives a per-function memory range of
128 MB to 10,240 MB, but AWS notes that new accounts can have reduced memory
and concurrency quotas. Therefore the 512 MB limit you see may be an
account-level quota. We should verify the quota shown for your account before
planning any Lambda workload around a higher value. This account-specific
limit does not prevent us from using Docker and Terraform with ECS/Fargate or
another suitable compute service. Lambda's current published limits are
documented in the [AWS Lambda quotas](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html)
and [memory configuration](https://docs.aws.amazon.com/lambda/latest/dg/configuration-memory.html)
guides.

The AWS phase should proceed in this order:

1. Validate preprocessing, model training/evaluation, and API behavior; record
   runtime, memory, model artifact size, and request shape.
2. Build and run the API image locally. A separate containerized Spark batch
   workload is not part of the current Terraform deployment path.
3. Choose AWS compute from measurements and account quotas. Keep the first
   deployment small and avoid making Lambda a hard dependency for the batch
   or model service.
4. Write Terraform for the selected storage, compute, IAM, network, log, and
   container registry resources. Store secrets outside source control.
5. Deploy to a development environment, test the full path from input data to
   Parquet and API response, then add monitoring and cost limits before any
   broader rollout.

The project contains a Docker image definition and Terraform configuration
for the proposed API path, but neither has been used to deploy AWS resources
from this machine. The deployment walkthrough records commands and
prerequisites. A real deployment still needs local image/runtime verification,
AWS credentials and quota checks, and a deliberate apply of potentially
billable infrastructure.

### Step 1: PySpark preprocessing

The input is newline-delimited JSON. Spark reads it as a DataFrame so the same
transformations can run on a local sample and later scale to a larger dataset.
The preparation job keeps `user_id`, `parent_asin`, `timestamp`, and the
optional `rating`, and renames `parent_asin` to `item_id` for downstream code.

Rows without a user ID, item ID, or timestamp are removed: such records cannot
identify an interaction or be assigned to a time-based split. Duplicate
user/item/timestamp events are dropped so repeated copies do not inflate
frequency counts or affect which event is considered latest. The job then
filters users and items below the configured interaction thresholds (five by
default). This reduces sparse entities and gives collaborative models more
observed relationships, at the cost of excluding some cold-start users and
items.

For each user, events are sorted by timestamp, with item ID as a deterministic
tie-breaker. All but the final two events go into training; the second-to-last
goes into validation; the latest goes into test. This simulates using history
to recommend a later item. The outputs are written as Parquet under separate
`interactions`, `train`, `validation`, and `test` directories. Later steps can
read these columnar files without parsing the original JSONL or rebuilding the
cleaning pipeline.

Run the preparation module with an input review JSONL file and an output
directory. For example:

The raw category review JSONL is not stored in Git. Download the `Movies and
TV` review file from the [official Amazon Reviews 2023 dataset page](https://amazon-reviews-2023.github.io/),
then decompress it to `data/raw/Movies_and_TV.jsonl`. The dataset page links
the category-specific review file and documents its JSON fields, including
`user_id`, `parent_asin`, and `timestamp`. The five-million-row command below
uses that file as a development sample; omit `--limit` only when intentionally
processing the full category.

```bash
python -m src.data.prepare \
  --input data/raw/Movies_and_TV.jsonl \
  --output data/processed/movies_tv_5m \
  --limit 5000000
```

`--limit` is useful for a development sample. Omit it to process the full
input. `--min-user-interactions`, `--min-item-interactions`, and
`--shuffle-partitions` control filtering and local Spark execution. The
preprocessing helpers are covered by `tests/test_prepare.py`, including sparse
entity filtering and the latest-two temporal split.

One benchmark caveat: the current job filters users and items before the
temporal split, so the counts used for eligibility include later events. This
can leak information about which entities appear in validation/test. For a
strict evaluation, derive eligibility and ID mappings from training history
only and apply those mappings to the later splits. The current baseline does
not use ratings; it treats the presence of a review as implicit positive
feedback.

### Step 2: global popularity baseline

This baseline is a simple reference for later personalized models. It counts
each item's interactions in the training split, orders items by decreasing
count, and breaks ties by item ID. Every user receives the same global top-K
list. Since validation and test rows do not contribute to the counts, they
remain held out for evaluation.

Run the module with the directory written by preprocessing **only after the
model and its settings are frozen**. The current CLI reads both validation and
test and prints both hit rates. For iterative model selection, use the
two-tower evaluator's same-sample popularity comparison; it reads validation
only. This prevents accidentally using test results to choose a model.

```bash
python -m src.models.popularity \
  --data data/processed/movies_tv_5m \
  --k 10
```

Hit Rate@K is calculated per user. A user is a hit when at least one of their
distinct held-out items is in the top-K list. The score is the number of users
with a hit divided by the number of evaluated users; duplicate events do not
increase a user's weight. If there are no evaluated users, the score is zero.
The module prints the top items and separate validation and test results. Use
validation while making modeling choices and keep test for final comparison.

`tests/test_popularity.py` exercises interaction-count ordering, stable
tie-breaking, invalid K handling, one-hit-per-user behavior, and empty held-out
data. Spark tests need a working Java runtime. This Windows setup has previously
failed to initialize Spark's JVM; the notebook notes above describe using the
WSL environment with Java 21 when that occurs. Python compilation can catch
syntax errors but does not replace these Spark behavior checks.

### ML pipeline notes

The current model workflow, full-catalog validation protocol, cohort results,
MLflow run commands, test instructions, and implementation function map are
maintained in [the two-tower study guide](two_tower_study_guide.md). The
measured epoch curves and test-by-test outcomes are in
[the test and training report](two_tower_test_report.md). This document keeps
the broader data-pipeline and deployment plan; update it when those pieces are
implemented rather than duplicating detailed model notes here.

## Spark troubleshooting log (WSL setup, 2026-10-03)

This section records the issues encountered while running the 5M-row preparation
and notebook on Windows/WSL. Diagnose the first Java/Spark error in the log; a
later Py4J connection error often just means that the JVM has already exited.

### 1. Windows Hadoop filesystem errors

The Windows run first reported that `HADOOP_HOME` / `hadoop.home.dir` was unset.
After that was addressed, local Parquet access failed with:

```text
UnsatisfiedLinkError: NativeIO$Windows.access0
```

This is Hadoop trying to call its Windows native filesystem library. It can
happen when compatible `winutils.exe`/native Hadoop libraries are not installed.
For this project, running Spark and the notebook in WSL avoids that Windows
native filesystem path. Keep the project files accessible under `/mnt/c/...` or
move working data into the WSL filesystem if mounted-drive I/O becomes a
bottleneck.

### 2. `scala/Serializable` missing at Spark startup

The error:

```text
NoClassDefFoundError: scala/Serializable
```

means SparkSubmit could not load the Scala classes required by its Spark jars.
It is a Spark installation/classpath problem, not an application-code or Java
heap problem. On this machine, `which -a spark-submit` showed multiple launchers
in `~/.local/bin`, Miniconda, and Windows Python Scripts. Putting one launcher
first on `PATH` did not guarantee that it used the intended PySpark jars.

Check that Python, PySpark, SparkSubmit, and the jars come from one environment:

```bash
which -a python spark-submit
python -c 'import pyspark; print(pyspark.__version__); print(pyspark.__file__)'
find "$SPARK_HOME/jars" -maxdepth 1 -name 'scala-library-*.jar' -print
```

For the WSL Miniconda installation used here, derive `SPARK_HOME` from the
active Python package and place its `bin` first on `PATH`:

```bash
export SPARK_HOME="$(python -c 'import pathlib,pyspark; print(pathlib.Path(pyspark.__file__).parent)')"
export PATH="$SPARK_HOME/bin:$CONDA_PREFIX/bin:$PATH"
hash -r
```

If environment variables point to another Spark installation, inspect or clear
those before retrying (`SPARK_HOME`, `PYSPARK_SUBMIT_ARGS`, and
`SPARK_SCALA_VERSION` were checked during this incident). Avoid sourcing a
`find-spark-home` helper until you have inspected which shell utilities it
calls; in this session that helper failed because its `dirname` invocation
received an unexpected `-b` argument.

Reinstalling PySpark is not the first diagnostic step. First identify the
active Python executable, imported PySpark version/location, selected
`spark-submit`, `SPARK_HOME`, and Scala jar. If those point to different
installations, fix the path/classpath mismatch; reinstall only if the selected
installation is actually incomplete or damaged.

### 3. Java compatibility errors

A startup/runtime error mentioning an internal JDK class, such as
`ClassNotFoundException: jdk.internal.ref.Cleaner`, calls for checking the
specific Spark and Java versions together. Do not infer the active Java version
from an installed JDK: verify the one actually selected by the current shell:

```bash
echo "$JAVA_HOME"
java -version
```

The successful WSL setup used Java 21 with the active Miniconda PySpark package.
Keep the Java runtime and PySpark/Spark distribution consistent across the
notebook kernel and command line.

### 4. Driver heap exhaustion during the temporal split

The first causal error was:

```text
java.lang.OutOfMemoryError: Java heap space
```

It occurred in Spark's unsafe sorter/spill reader during the per-user window
sort and shuffle. The subsequent `Connection reset` / `Connection refused`
tracebacks appeared because the Spark JVM died. Retrying the same Py4J call
cannot recover that dead JVM; restart the notebook kernel after adjusting the
configuration.

The batch preparation command succeeded with an 8 GB driver. The notebook was
then configured to request a 16 GB Java heap before PySpark launches its gateway:

```python
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 16g pyspark-shell"
```

That variable must be set before Spark starts; changing it in a live kernel
does not resize the Java process. The notebook also uses `local[8]` to bound
concurrent task memory, 128 initial shuffle partitions, and adaptive execution.
These are starting settings, not proven optimal values. Although the workstation
has 128 GB RAM, WSL has its own memory allowance and Java shares that allowance
with Python and the operating system. Check available memory inside WSL with
`free -h` if heap failures persist.

The notebook avoids three separate `train.count()`, `validation.count()`, and
`test.count()` actions. It labels a cached window result and aggregates the
three split counts in one action, preventing each count from replaying the same
upstream JSON/cleaning/window lineage. For model development after the Parquet
splits have been generated, restart the kernel and run setup plus the prepared
Parquet-loading cell rather than rebuilding the raw-data pipeline.

### 5. Keep one Spark toolchain active

The working combination for this run was the WSL Miniconda Python kernel,
PySpark installed in that same Miniconda environment, Java 21, and the Spark
jars beneath that PySpark package. Windows-side PySpark installations and
launchers should not take precedence in WSL `PATH`. When Spark fails, record
these together before changing versions:

```text
Python executable and version
PySpark version and package location
spark-submit location
SPARK_HOME
Scala library jar in SPARK_HOME/jars
JAVA_HOME and java -version
```

This checklist distinguishes filesystem/native-library failures, a mismatched
Spark classpath, Java compatibility issues, and actual memory exhaustion.
