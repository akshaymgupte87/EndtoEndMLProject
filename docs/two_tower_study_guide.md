# Two-Tower Recommender: Study Guide

This guide follows the implementation one step at a time. Each step explains
the data contract, the reason for the design, how to run it, and how to verify
it. We will add the next step only after you ask to proceed.

## Goal and current position

A two-tower recommender learns one vector for a user and one vector for an
item. It scores a user/item pair by comparing those vectors. At serving time,
item vectors can be computed ahead of time; the system then ranks items by
their dot-product score with the requested user's vector.

The end-to-end model workflow will eventually be:

```text
Prepared temporal splits
    ↓
Integer user/item mappings
    ↓
Positive training pairs and negative examples
    ↓
PyTorch user and item towers
    ↓
Validation-based training and model selection
    ↓
Ranking metrics and final test evaluation
    ↓
Saved model and item vectors for recommendation serving
```

The mapping, sampling, model structure, and training-loop code now exist.
The mapping command has not yet been run on the project dataset, and the
training loop has only been checked on synthetic interactions. Ranking metrics,
real-data training, and serving artifacts remain later steps.

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
translate indices back to original user and item IDs. The command exists, but
this project has not yet generated its two-tower mapping artifact.

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
does not train a network or compute recommendation metrics. This command has
not yet been run on the project data. Step 2 defines positive and negative
examples; Steps 3 and 4 implement the model and training loop.

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

Spark could not initialize in the Windows Python environment during an earlier
baseline test because the selected JVM lacked a class required by Spark. Run
Spark checks from the project's WSL/Java 21 environment when that limitation
applies. We have not yet run this new mapping command against the dataset.

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
objective; ranking metrics will be a later evaluation step.

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
  --output artifacts/two_tower/v1 \
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

The synthetic test does not prove that training improves recommendation
quality on Amazon data. After the Spark environment is ready, the next useful
check is a small real-data run and inspection of the sampled user/pair counts,
validation loss curve, and checkpoint metadata. Do not use the reported
validation BPR loss as a substitute for Recall@10 or NDCG@10.

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
