# Two-Tower Project: Test and Training Report

## Test run

**Date:** 2026-10-05  
**Environment:** WSL Ubuntu, Miniconda Python 3.11, PySpark 4.2.0, Java 21  
**Latest complete WSL result:** 99 passed, 0 failed in 27.17 seconds after separating checkpoint loading from Spark/MLflow training imports. PySpark warned that pandas 3.x compatibility is incomplete; it did not affect the run.

The latest complete test-running commands are in the
[study guide](two_tower_study_guide.md#running-the-projects-tests).

### Windows rerun (2026-10-05)

Running the full suite with the configured Windows Python 3.14 environment
gave **82 passed and 10 Spark startup errors**. Spark's JVM failed to load
`jdk.internal.ref.Cleaner` while starting its local context. The errors are
environment-specific; the same suite previously passed in WSL with Python
3.11 and Java 21. Use the documented WSL command for the complete Spark test
run.

The tests use small controlled examples to check behavior and invariants.
They do not retrain the two-tower model on the full dataset. A passing test
means the tested behavior matched its expectations; it does not mean the model
beats the popularity recommender.

This 99-test run covers matched-comparison safeguards, Pandas/XGBoost ranking
tests, shared-catalog evaluation, ALS source-ID alignment, and comparison
aggregation. See the study guide's “Running the project's tests” section for
the exact WSL command.

### `tests/test_api.py` — 2 passed

Known users received ranked items with training-seen items filtered; unknown
users returned 404. Metrics were exposed, and simulated Redis connection
failures preserved recommendation responses.

### `tests/test_collaborative_filtering.py` — 5 passed

Factor dot products ranked expected items and removed seen items. Tests also
checked a missing factor, saved-model archive contents, source-ID alignment,
and comparison aggregation.

### `tests/test_evaluate_two_tower.py` — 12 passed

Metric calculations matched hand-computed examples. Ranking removed training-
seen items; cohort boundaries and validation/test split isolation behaved as
expected.

### `tests/test_event_schema.py` — 5 passed

Click/view events round-tripped through JSON. Invalid indices, event types,
booleans, and malformed JSON were rejected.

### `tests/test_experiment_tracking.py` — 1 passed

The temporary SQLite store retained parameters, final and epoch-step metrics,
and the history artifact.

### `tests/test_model_registration.py` — 3 passed

Validation reports clearing the gate were accepted; test reports, non-finite
metrics, and recall regression were rejected before registration.

### `tests/test_prepare.py` — 2 passed

Filtering removed the sparse entities in the fixture; the temporal split put
the two latest interactions in validation and test. This checks the chronology
the recommender uses to simulate future interactions.

### `tests/test_popularity.py` — 4 passed

Popularity ordering and tie-breaking were deterministic. Hit Rate counted a
user once even with duplicate held-out rows; empty validation data returned
zero users and zero hit rate. This is the reference behavior for baseline
comparisons.

### `tests/test_sampling.py` — 26 passed

Repeated sampling stayed outside each user's training positives and repeated
with the same seed. Dense histories used the fallback and still returned an
unseen catalog item. Invalid or out-of-range IDs were rejected before model
indexing.

### `tests/test_prepare_tower_data.py` — 2 passed

Mappings were sorted, distinct, and zero-based. Encoding retained known rows,
preserved duplicate known interactions, and counted unknown IDs as excluded.
This verifies that embedding indices can be looked up consistently.

### `tests/test_two_tower.py` — 29 passed

Embedding lookup and aligned dot-product scores had expected shapes and finite
values. In the gradient-step check, the positive-minus-negative score margin
increased and BPR loss decreased after an optimizer-style update. This proves
the loss sends gradients in the intended direction on a controlled pair; it
does not show recommendation quality on real data.

### `tests/test_train_two_tower.py` — 1 passed

The synthetic test used 4 training pairs, 2 validation pairs, and 6 epochs.
The epoch trace was:

| Epoch | Train BPR | Validation BPR | Debugging result |
|---:|---:|---:|---|
| 1 | 0.693159 | 0.690882 | Initial pairwise scores are close; validation starts below training by chance. |
| 2 | 0.690663 | 0.684052 | Both losses fall. |
| 3 | 0.681953 | 0.676006 | Validation keeps improving. |
| 4 | 0.670524 | 0.666031 | Both losses fall; no overfit signal in this tiny fixture. |
| 5 | 0.650915 | 0.654781 | Validation is now slightly above training. |
| 6 | 0.628127 | 0.641865 | Training continues improving; validation also improves but the gap widens. |

The test passed because it reloaded the checkpoint with the lowest validation
loss and verified its scores and metadata. The 6-epoch fixture is a mechanics
check, not a performance estimate.

### `tests/test_xgboost_ranker.py` — 7 passed

Candidate sampling was repeatable, unique, and excluded training-seen items.
Validation negatives excluded both training history and held-out targets.
Pandas created numeric popularity and history-affinity features from training
data. Tests checked that positive training rows subtract their own contribution
from co-occurrence counts. A small grouped `XGBRanker` fit/predict check
passed. These tests check mechanics on synthetic rows; the real-data
experiment is reported separately below.

The run produced one PySpark/Pandas compatibility warning; it did not cause a
test failure. The `/tmp` pytest cache option avoids the earlier mounted-folder permission
warning. Older 76- and 77-test runs and a corrected synthetic ALS assertion
are historical debugging records; the latest 99-test run supersedes those
counts.

## Complete pytest inventory and pass criteria

Pytest collected **99 cases** across the 13 files below. A test function with
`pytest.mark.parametrize` is one named test but creates multiple collected
cases; the case counts here include those parameter variations. A case passes
when its stated output or behavior matches the assertion, or when the
specified invalid input raises the expected exception. These are mostly small
synthetic unit/integration checks, not model-quality thresholds. The tests
exercise Spark through the shared `local[2]` fixture in `tests/conftest.py`.

### `tests/test_api.py` — 2 cases

| Test | Pass criterion |
|---|---|
| `test_api_serves_known_user_hides_seen_and_rejects_unknown` | Health/readiness return 200; known user 77 gets exactly unseen items `item-b`, `item-c`; unknown user returns 404; limit 101 returns 422; metrics endpoint exposes request counter. |
| `test_api_falls_back_when_redis_is_down` | With Redis operations raising `ConnectionError`, recommendation still returns 200 and the expected unseen items; cache-error metric is exposed. |

### `tests/test_collaborative_filtering.py` — 5 cases

| Test | Pass criterion |
|---|---|
| `test_als_top_k_uses_factor_similarity_and_removes_seen_items` | Dot-product ranking matches the expected top two items per user and removes each user's seen item. |
| `test_als_top_k_rejects_missing_user_factor` | Requesting recommendations for a user without a learned ALS vector raises `ValueError` identifying the missing factor. |
| `test_model_archive_contains_saved_spark_model_files` | ZIP archive contains the Spark model file under the expected `als_model/` path. |
| `test_localize_user_factors_maps_source_ids_to_evaluation_rows` | Factors keyed by source user IDs are re-keyed to the requested evaluation-row order without swapping vectors. |
| `test_mean_and_sample_std_summarizes_seed_metrics` | Values `[1, 2, 3]` produce mean 2 and sample standard deviation 1. |

### `tests/test_evaluate_two_tower.py` — 13 cases

| Test | Pass criterion |
|---|---|
| `test_ranking_metrics_at_k_uses_macro_user_averages` | Hand-computed macro Recall, Precision, and NDCG at K=2 match, with two users counted. |
| `test_ranking_metrics_rejects_empty_relevance` | Empty relevant-target data raises `ValueError`. |
| `test_recommend_top_k_excludes_each_users_training_items` | Model top-2 lists match known scores and contain no training-seen item. |
| `test_recommend_top_k_respects_shared_candidate_catalog` | Ranking considers only the supplied catalog, then returns the two highest-scoring unseen candidates. |
| `test_popularity_ranking_also_excludes_each_users_training_items` | Popularity top-2 lists preserve popularity order after per-user seen-item removal. |
| `test_popularity_ranking_respects_shared_candidate_catalog` | Popularity output contains only the supplied candidate IDs, in popularity order. |
| `test_metrics_by_cohort_reports_user_history_and_target_popularity_groups` | Sparse/medium/active user groups and tail/mid/popular target groups each have the expected count; known hit yields recall 1.0. |
| `test_recommend_top_k_rejects_non_positive_k[k=0]` | K=0 raises `ValueError` explaining K must be positive. |
| `test_recommend_top_k_rejects_non_positive_k[k=-1]` | K=-1 raises the same validation error. |
| `test_recommend_top_k_rejects_out_of_range_seen_item` | Seen item index outside the model catalog raises `ValueError`. |
| `test_validation_loader_keeps_selected_users_and_skips_train_seen_targets` | Only selected users are mapped; train histories and eligible validation targets are correct; overlapping held-out target is counted as skipped; popularity counts use train only. |
| `test_evaluation_loader_reads_only_the_requested_holdout_split` | Validation request returns validation target, test request returns test target, and train is rejected as an invalid holdout split. |

### `tests/test_event_schema.py` — 5 cases

| Test | Pass criterion |
|---|---|
| `test_event_round_trip_is_validated` | A valid click event survives encode/decode with all fields unchanged. |
| `test_event_rejects_invalid_values[negative-user]` | Negative user index is rejected by `encode_event`. |
| `test_event_rejects_invalid_values[unsupported-event-type]` | Unsupported event type is rejected. |
| `test_event_rejects_invalid_values[boolean-index]` | Boolean user index is rejected rather than accepted as an integer. |
| `test_event_decoder_rejects_malformed_json` | Invalid JSON bytes raise `ValueError` identifying malformed JSON. |

### `tests/test_experiment_tracking.py` — 1 case

| Test | Pass criterion |
|---|---|
| `test_log_experiment_run_records_parameters_metrics_and_artifacts` | Temporary SQLite MLflow store retains parameters, final metric, epoch steps `[1, 2]`, and `history.json` artifact. |

### `tests/test_model_registration.py` — 3 cases

| Test | Pass criterion |
|---|---|
| `test_model_registration_gate_accepts_validation_gain` | A validation report with Recall@10 not regressing and NDCG@10 lift passes; reported lift is 20%. |
| `test_model_registration_gate_rejects_test_report_and_regression` | Test-split report is rejected; validation Recall@10 below baseline is rejected. |
| `test_model_registration_gate_rejects_non_finite_metrics` | NaN NDCG is rejected as not finite. |

### `tests/test_popularity.py` — 4 cases

| Test | Pass criterion |
|---|---|
| `test_rank_popular_items_orders_counts_and_breaks_ties` | Top two items have the highest counts; equal-count tie resolves alphabetically/deterministically as expected. |
| `test_rank_popular_items_rejects_non_positive_k` | K=0 raises `ValueError`. |
| `test_hit_rate_counts_each_user_once` | Duplicate matching holdout rows count as one hit for that user; two users yield one hit and 0.5 hit rate. |
| `test_hit_rate_handles_empty_held_out_data` | Empty holdout returns 0 users, 0 hits, and 0.0 hit rate. |

### `tests/test_prepare.py` — 2 cases

| Test | Pass criterion |
|---|---|
| `test_filter_interactions_removes_sparse_entities` | With both thresholds set to two, only the interaction between entities meeting both thresholds remains. |
| `test_temporal_split_uses_last_two_interactions` | Timestamp order assigns the earliest event to train, next to validation, latest to test. |

### `tests/test_prepare_tower_data.py` — 2 cases

| Test | Pass criterion |
|---|---|
| `test_build_id_mapping_is_sorted_distinct_and_zero_based` | Nulls and duplicates are removed; IDs sort deterministically and receive contiguous zero-based `LongType` indices. |
| `test_encode_split_drops_pairs_with_unknown_user_or_item` | Known rows (including duplicates) are retained with correct indices; unknown user/item rows are excluded and counted. |

### `tests/test_sampling.py` — 29 cases

| Test | Pass criterion |
|---|---|
| `test_build_seen_items_deduplicates_positive_pairs` | Duplicate user-item pairs collapse into per-user sets. |
| `test_sampler_never_returns_a_training_positive` | Across 100 draws for each fixture user, every negative belongs to catalog and is absent from that user's seen set. |
| `test_sampler_is_repeatable_with_the_same_seed` | Two samplers with equal seed produce the same 20-item sequence. |
| `test_sampler_rejects_a_user_who_has_seen_the_full_catalog` | A history covering the catalog raises `ValueError`. |
| `test_sampler_rejects_unknown_users` | Sampling for a user with no history raises `KeyError`. |
| `test_build_seen_items_rejects_non_integer_indices[value=0.5]` | Float user index is rejected with `TypeError`. |
| `test_build_seen_items_rejects_non_integer_indices[value='0']` | String item index is rejected with `TypeError`. |
| `test_build_seen_items_rejects_non_integer_indices[value=True]` | Boolean index is rejected with `TypeError`. |
| `test_build_seen_items_rejects_negative_indices[pair=(-1,0)]` | Negative user index is rejected with `ValueError`. |
| `test_build_seen_items_rejects_negative_indices[pair=(0,-1)]` | Negative item index is rejected with `ValueError`. |
| `test_sampler_rejects_non_integer_catalog_size[count=1.5]` | Float catalog size is rejected with `TypeError`. |
| `test_sampler_rejects_non_integer_catalog_size[count=True]` | Boolean catalog size is rejected with `TypeError`. |
| `test_sampler_rejects_invalid_history[empty-user-history]` | Empty history is rejected with `ValueError`. |
| `test_sampler_rejects_invalid_history[negative-user-index]` | Negative history user index is rejected. |
| `test_sampler_rejects_invalid_history[negative-item-index]` | Negative history item index is rejected. |
| `test_sampler_rejects_invalid_history[item-outside-catalog]` | History item beyond catalog bounds is rejected. |
| `test_sampler_rejects_non_integer_history[float-user-index]` | Float history user index is rejected with `TypeError`. |
| `test_sampler_rejects_non_integer_history[float-item-index]` | Float history item index is rejected with `TypeError`. |
| `test_sampler_rejects_non_integer_history[boolean-item-index]` | Boolean history item index is rejected with `TypeError`. |
| `test_sampler_keeps_a_read_only_snapshot_of_history` | Later source-history mutation does not change sampler history; exposed snapshot cannot be modified; remaining item can be sampled. |
| `test_dense_fallback_maps_each_available_rank_uniformly[rank=0]` | After 64 rejection attempts, rank 0 maps to first unseen catalog item 1. |
| `test_dense_fallback_maps_each_available_rank_uniformly[rank=1]` | Rank 1 maps to unseen item 4. |
| `test_dense_fallback_maps_each_available_rank_uniformly[rank=2]` | Rank 2 maps to unseen item 6. |
| `test_sample_rejects_invalid_user_indices[user=-1]` | Negative user index is rejected. |
| `test_sample_rejects_invalid_user_indices[user=0.0]` | Float user index is rejected. |
| `test_sample_rejects_invalid_user_indices[user=True]` | Boolean user index is rejected. |

### `tests/test_train_two_tower.py` — 1 case

| Test | Pass criterion |
|---|---|
| `test_training_loop_saves_and_reloads_best_checkpoint` | Six-epoch run writes checkpoint; history length matches epochs; selected epoch/loss equals minimum validation loss and is finite; reloaded model scores exactly match; user IDs, item mapping path, and best epoch metadata round-trip. |

### `tests/test_two_tower.py` — 31 cases

| Test | Pass criterion |
|---|---|
| `test_pairwise_dataset_returns_valid_training_triples` | Dataset returns expected users/positives and a sampled negative outside each user's positive history. |
| `test_pairwise_dataset_rejects_empty_pairs` | Empty positive-pair input raises `ValueError`. |
| `test_two_tower_returns_one_dot_product_score_per_pair` | Forward pass returns one finite positive and negative score per aligned input pair. |
| `test_bpr_loss_is_lower_when_positives_score_higher` | Correctly ordered positive/negative scores yield lower BPR loss than reversed scores. |
| `test_bpr_loss_rejects_mismatched_batch_shapes` | Unequal positive/negative batch shapes raise `ValueError`. |
| `test_dataset_rejects_malformed_pairs[one-field]` | Pair with one field raises `ValueError`. |
| `test_dataset_rejects_malformed_pairs[three-fields]` | Pair with three fields raises `ValueError`. |
| `test_dataset_rejects_malformed_pairs[scalar]` | Non-pair scalar raises `ValueError`. |
| `test_dataset_does_not_silently_coerce_indices[float-user]` | Float user index raises `TypeError`. |
| `test_dataset_does_not_silently_coerce_indices[float-item]` | Float item index raises `TypeError`. |
| `test_dataset_does_not_silently_coerce_indices[boolean-user]` | Boolean user index raises `TypeError`. |
| `test_dataset_does_not_silently_coerce_indices[boolean-item]` | Boolean item index raises `TypeError`. |
| `test_dataset_rejects_pairs_outside_sampler_history[negative-user]` | Negative user index raises `ValueError`. |
| `test_dataset_rejects_pairs_outside_sampler_history[negative-item]` | Negative item index raises `ValueError`. |
| `test_dataset_rejects_pairs_outside_sampler_history[item-outside-catalog]` | Item index outside sampler catalog raises `ValueError`. |
| `test_dataset_rejects_pairs_outside_sampler_history[unknown-user]` | User missing from sampler history raises `ValueError`. |
| `test_dataset_rejects_pairs_outside_sampler_history[unseen-positive]` | Pair whose positive is absent from that user's history raises `ValueError`. |
| `test_dataloader_workers_have_distinct_reproducible_negative_streams` | Worker streams differ from each other, repeat exactly across equal seeded runs, and all negatives are valid unseen catalog IDs. |
| `test_scores_equal_known_dot_products` | Fixed embeddings produce exact expected positive and negative dot products. |
| `test_scores_reject_accidental_broadcasting` | Differently shaped user/item index tensors raise `ValueError`. |
| `test_model_rejects_invalid_dimensions[boolean-user-count]` | Boolean user count raises `TypeError` or `ValueError`. |
| `test_model_rejects_invalid_dimensions[float-item-count]` | Non-integer item count raises `TypeError` or `ValueError`. |
| `test_model_rejects_invalid_dimensions[zero-embedding-dim]` | Zero embedding dimension raises `TypeError` or `ValueError`. |
| `test_bpr_rejects_empty_and_nonfinite_scores[empty]` | Empty score tensor is rejected in either positive or negative position. |
| `test_bpr_rejects_empty_and_nonfinite_scores[nan]` | NaN score tensor is rejected in either position. |
| `test_bpr_rejects_empty_and_nonfinite_scores[infinity]` | Infinite score tensor is rejected in either position. |
| `test_bpr_rejects_integer_scores` | Integer score tensors raise `TypeError`; BPR requires floating-point values. |
| `test_bpr_is_finite_for_large_finite_score_differences` | Extreme finite score differences produce finite loss (1000 in the fixture), without numerical overflow. |
| `test_gradient_step_increases_positive_margin_and_reduces_loss` | Gradients exist and are finite with expected signs; one gradient-descent update increases positive-minus-negative margin and lowers BPR loss. |

### `tests/test_xgboost_ranker.py` — 7 cases

| Test | Pass criterion |
|---|---|
| `test_sample_unseen_items_is_repeatable_unique_and_excludes_history` | Equal seeds yield identical five unique negatives, none in seen history. |
| `test_sample_unseen_items_rejects_impossible_request` | Requesting more negatives than available unseen items raises `ValueError`. |
| `test_training_table_contains_only_train_positives_and_unseen_negatives` | Positive counts per user match training rows; every sampled negative is outside that user's train history. |
| `test_validation_candidates_exclude_train_history_and_all_targets` | All validation targets remain; each user has expected target-plus-negative row count; negatives overlap neither training history nor any held-out target. |
| `test_cooccurrence_counts_are_built_from_unique_user_histories` | Item-pair counts match counts over unique items per user, so duplicate events do not inflate affinity. |
| `test_pandas_features_use_training_counts_and_leave_one_out_affinity` | Log-count and history-affinity features match expected values; ID columns stay integer; feature order is exactly declared. |
| `test_xgboost_ranker_fits_grouped_candidate_rows` | Small grouped `XGBRanker` fit completes and prediction returns one score per candidate row. |

## Other verification checks (not part of the 99 pytest cases)

These checks have different pass criteria and runtime requirements, so they
must not be added to the pytest count:

| Check | Pass criterion and current evidence |
|---|---|
| Full-catalog and shared-catalog offline evaluations | Job completes on the named validation/test split; emits Recall@K, Precision@K, and NDCG@K for the specified cohort/catalog. Quality is interpreted by comparison with the same-sample baseline, not by mere command success. Detailed epoch and metric records are below. |
| XGBoost real-data comparison | Trains and ranks the sampled validation candidates and writes metrics. The recorded experiment is candidate-sampled, not full-catalog, and does not mean the live API uses XGBoost. |
| Batch-job smoke run | 500-user, two-epoch run exits successfully and emits checkpoint, manifest, ranking report, and serving bundle. It is a plumbing check; smoke model's zero Recall@10/20 is not a pass/fail quality threshold. See [deployment walkthrough](deployment_walkthrough.md#batch-smoke-run-log). |
| End-to-end batch-to-API run (2026-10-05) | Batch trained/evaluated/exported on 500 users; the Docker API loaded the bundle and served requests; Prometheus target was up; Collector received spans; stopping Redis preserved HTTP 200 responses and restarting Redis restored cache hits. The outage response took 7.94 seconds, so fallback latency needs improvement. Full record: [deployment walkthrough](deployment_walkthrough.md#end-to-end-run-record-2026-10-05). |
| MLflow run inspection | Training/evaluation run is present with parameters, epoch/final metrics, and expected artifacts. The local SQLite write path itself is also unit-tested by `test_experiment_tracking.py`. |
| Terraform formatting and validation | `terraform fmt -check` reports no formatting changes and `terraform validate` exits successfully after provider initialization. This checks configuration only, not whether AWS deployment works. |
| Docker Compose configuration | `docker compose config --quiet` exits successfully; the stack was also built and run in the end-to-end check below. |
| Local container/API observability exercise | Completed locally: API ready and served recommendations; Prometheus scrape target was up; Collector received spans; Redis outage preserved recommendations and raised cache errors; restart restored cache hits. The outage request was slow (~7.94 seconds). Full evidence: [deployment walkthrough](deployment_walkthrough.md#end-to-end-run-record-2026-10-05). |
| Kubernetes, Airflow, Kafka, AWS deployment | Require their runtimes/accounts. Manifests and runbooks exist, but no live cluster/DAG/broker/AWS apply is claimed by the 99-test result. |

For full-suite and focused commands, see [Running the project's tests in the
study guide](two_tower_study_guide.md#running-the-projects-tests). The
environment-specific WSL requirement and known Windows Spark startup failure
are at the top of this report.

## Epoch-by-epoch real-data training log

This is a separate bounded experiment, not part of the unit-test run. It used
2,000 users, 21,938 training pairs, 2,000 validation pairs, and a 137,070-item
catalog. The checkpoint was selected using sampled validation pairwise loss.
Lower loss means the model more often assigns higher scores to observed pairs
than sampled negatives; it does not directly measure top-K recommendation
quality.

| Epoch | Training BPR loss | Validation BPR loss | Change from prior epoch |
|---:|---:|---:|---|
| 1 | 0.693145 | 0.693146 | Starting point; near the 0.693 loss expected when positive and negative scores are initially similar. |
| 2 | 0.692558 | 0.693131 | Both losses fall slightly; optimization begins to separate pairs. |
| 3 | 0.690732 | 0.693092 | Training loss falls faster; validation improvement remains small. |
| 4 | 0.685939 | 0.692981 | Both improve; validation still near its starting value. |
| 5 | 0.676823 | 0.692765 | Training improves more than validation, an early generalization gap. |
| 6 | 0.662705 | 0.692323 | Validation continues to improve, but much more slowly than training. |
| 7 | 0.643707 | 0.691735 | Pairwise validation signal improves; this is not yet a ranking metric. |
| 8 | 0.620562 | 0.690970 | Both losses decrease; training-to-validation gap widens. |
| 9 | 0.594032 | 0.690048 | Lowest validation loss so far; training continues to fit the sampled pairs. |
| 10 | 0.564963 | 0.688978 | Best validation loss in this run; checkpoint saved at epoch 10. |

**Interpretation:** training loss decreased by about 18.5%, while validation
pairwise loss decreased by about 0.6%. The model learned the training pairs,
and the validation pairwise objective improved modestly. The widening gap
calls for monitoring overfitting in future runs. Epoch 10 is the best of these
10 epochs by this validation loss, not proof that 10 epochs or this model
configuration is generally optimal.

## Full-catalog validation ranking log

After training, the saved model was evaluated for the same 2,000 users against
all 137,070 catalog items. Training-seen items were removed from each user's
candidate list. The evaluator read train and validation only; test remained
untouched. Each user had one eligible validation target, and no target was
removed for overlapping the training history.

| Method | K | Recall@K | Precision@K | NDCG@K | Users with target in top K |
|---|---:|---:|---:|---:|---:|
| Two-tower | 10 | 1.00% | 0.100% | 0.00496 | 20 / 2,000 |
| Popularity, same training sample | 10 | 1.55% | 0.155% | 0.00754 | 31 / 2,000 |
| Two-tower | 20 | 1.45% | 0.0725% | 0.00610 | 29 / 2,000 |
| Popularity, same training sample | 20 | 2.40% | 0.120% | 0.00975 | 48 / 2,000 |

**Result:** popularity is better on Recall, Precision, and NDCG at both K
values. The loss curve shows that pairwise training worked, but full-catalog
ranking shows that this model configuration did not turn that improvement
into better recommendations than the baseline. The next model change should
be judged on validation ranking metrics, not training loss alone. Keep test
held out until model selection is complete.

## Larger-sample diagnostic: 10,000 users, seed 42

To check whether the first result was caused by the small sample, a second
run kept the same seed, model size, optimizer, and 10 epochs, while increasing
the limits to 10,000 users and 200,000 training pairs. The actual sample had
10,000 users, 110,923 training pairs, 10,000 validation targets, and the same
137,070-item catalog. Complete histories were retained, so the pair limit was
not reached. Test was not read.

| Epoch | Training BPR loss | Validation BPR loss |
|---:|---:|---:|
| 1 | 0.693145 | 0.693118 |
| 2 | 0.691150 | 0.692426 |
| 3 | 0.680024 | 0.688252 |
| 4 | 0.651070 | 0.678394 |
| 5 | 0.605329 | 0.664063 |
| 6 | 0.550166 | 0.648187 |
| 7 | 0.493176 | 0.632428 |
| 8 | 0.438829 | 0.617857 |
| 9 | 0.389305 | 0.604925 |
| 10 | 0.345362 | 0.593487 |

Both losses decreased every epoch. Training BPR fell about 50.2%; validation
BPR fell about 14.4%. The gap indicates the training objective improved more
strongly on training pairs, so later runs should keep checking validation
ranking rather than selecting by training loss.

| Method | K | Recall@K | Precision@K | NDCG@K | Users with target in top K |
|---|---:|---:|---:|---:|---:|
| Two-tower | 10 | 2.25% | 0.225% | 0.01163 | 225 / 10,000 |
| Popularity, same training sample | 10 | 2.09% | 0.209% | 0.01029 | 209 / 10,000 |
| Two-tower | 20 | 3.34% | 0.167% | 0.01436 | 334 / 10,000 |
| Popularity, same training sample | 20 | 2.99% | 0.150% | 0.01252 | 299 / 10,000 |

**Result:** the two-tower model beats popularity on all three metrics at both
cutoffs in this larger validation cohort. This suggested that the 2,000-user
run may have had too little data to show an advantage, but a second seed was
needed before treating the gain as repeatable. Recall@10 is still 2.25%, so
recommendation coverage remains low.

## Validation slices and second seed: 10,000 users, seed 7

The same 10,000-user limits and model configuration were trained with seed 7
and evaluated against a separately sampled 10,000-user cohort. Both runs used
all 137,070 catalog items, full-catalog ranking, training-seen masking, and
their own same-sample popularity baseline. Neither read the test split.

| Epoch | Training BPR | Validation BPR | Debugging result |
|---:|---:|---:|---|
| 1 | 0.693138 | 0.693110 | Initial pairwise scores are near the random baseline. |
| 2 | 0.690764 | 0.692092 | Both losses fall. |
| 3 | 0.676937 | 0.686592 | Validation improves with training. |
| 4 | 0.642657 | 0.674994 | Validation continues to improve. |
| 5 | 0.591024 | 0.659054 | Training improves faster; the gap grows. |
| 6 | 0.531812 | 0.641673 | Validation still improves. |
| 7 | 0.472890 | 0.624692 | Both fall; training-to-validation gap widens. |
| 8 | 0.418392 | 0.608830 | Validation remains on a downward trend. |
| 9 | 0.369407 | 0.594395 | Validation improves again. |
| 10 | 0.326602 | 0.581581 | Lowest validation loss; epoch 10 checkpoint selected. |

Across these epochs, training BPR fell 52.9% and validation BPR fell 16.1%.
This shows optimization progress; the ranking comparison below is the check
of whether the learned scores improve recommendations.

| Seed | Method | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|---:|---|---:|---:|---:|---:|
| 42 | Two-tower | 2.25% | 0.01163 | 3.34% | 0.01436 |
| 42 | Popularity | 2.09% | 0.01029 | 2.99% | 0.01252 |
| 7 | Two-tower | 1.89% | 0.00940 | 2.99% | 0.01211 |
| 7 | Popularity | 1.86% | 0.00979 | 2.73% | 0.01198 |

The two-tower model improves Recall at both K values on seed 7, but popularity
has higher NDCG@10. At K=20 the model narrowly leads on Recall and NDCG. This
is not a consistent win across all metrics, cohorts, and seeds. The user
samples differ between seeds, so these runs check end-to-end stability across
random samples; they are not a paired statistical comparison on identical
users.

### Cohort breakdown for the seed 42 run

User cohorts are based on each user's number of training interactions:
1–5, 6–20, and over 20. Target-item cohorts are based on how many times the
validation item occurred in this sampled training set: 0–2, 3–10, and over
10. These popularity bands are sample-relative, not global Amazon item
popularity. Each user has one validation target in this dataset.

| Cohort | Users/targets | Two-tower Recall@10 | Popularity Recall@10 | Two-tower Recall@20 | Popularity Recall@20 |
|---|---:|---:|---:|---:|---:|
| Sparse users, 1–5 train rows | 4,821 users | 2.53% | 2.47% | 3.69% | 3.53% |
| Medium users, 6–20 train rows | 4,053 users | 2.20% | 2.00% | 3.43% | 2.91% |
| Active users, over 20 train rows | 1,126 users | 1.24% | 0.80% | 1.51% | 0.98% |
| Tail targets, 0–2 sample train rows | 5,144 targets | 0.00% | 0.00% | 0.00% | 0.00% |
| Mid targets, 3–10 sample train rows | 2,812 targets | 0.43% | 0.00% | 0.57% | 0.00% |
| Popular targets, over 10 sample train rows | 2,044 targets | 10.42% | 10.23% | 15.56% | 14.63% |

Seed 42 shows gains over popularity in all three user-history groups and all
three target-popularity groups. The tail-target result is zero for both
methods because those targets have at most two occurrences in the sampled
training interactions; it exposes a coverage weakness rather than proving
that ranking code failed. Seed 7's slices are mixed: popularity is stronger
for sparse users and popular targets at K=10, while the tower is stronger for
medium-history users. The complete machine-readable cohort metrics are in
each `validation_metrics.json` file.

### Evaluation conclusion

The evaluation step now reports overall and cohort Recall@K, Precision@K,
and NDCG@K for K=10 and K=20, and compares each run against popularity on the
same sampled users and training interactions. The broader run shows a
promising but inconsistent gain: seed 42 wins every listed aggregate metric,
while seed 7 has mixed results. This is not enough to approve the model as a
stable replacement. Cold-start coverage is separately limited by the
training-only ID mappings: held-out rows with unknown IDs were excluded
during preparation (2,177 validation and 3,427 test rows in the prepared
sample), so the ID-only model cannot rank those cases. Keep test sealed until
model selection is complete.

The machine-readable outputs are
`artifacts/two_tower/movies_tv_5m/v1/history.json`,
`artifacts/two_tower/movies_tv_5m/v1/validation_metrics.json`,
`artifacts/two_tower/movies_tv_5m/v2/history.json`, and
`artifacts/two_tower/movies_tv_5m/v2/validation_metrics.json`, plus the seed 7
run under `artifacts/two_tower/movies_tv_5m/v3/`. They are local generated
artifacts and are excluded from Git.

The training and validation commands now log to the local MLflow experiment
`two-tower-recommender` using `sqlite:///mlflow.db`. The seed-42 v2 validation
and seed-7 v3 training/evaluation runs are recorded. The seed-42 v2 training
preceded tracking integration, so that training curve remains in its JSON
history only. The MLflow database and artifacts are ignored by Git.

## Implicit ALS validation comparison (2026-10-04)

> **Evaluation audit:** The numerical ALS results in this historical section
> are invalid and superseded. The evaluator used local row indices to select
> Spark ALS factor rows, whose IDs are global source IDs. This mismatch paired
> users with the wrong factors. The corrected user-ID localization and the
> fair matched comparison below are the results to use. The historical scores
> remain here only to explain why the audit was necessary.

This run implements the v3 plan's collaborative-filtering milestone. Spark ML
ALS trained in implicit-feedback mode on all 2,687,407 encoded training
interactions. Pair counts were aggregated as confidence weights. Parameters
were rank 16, 10 iterations, regularization 0.1, alpha 1.0, and seed 42. It
learned 238,645 user factors and 137,070 item factors.

Evaluation used the same 10,000 validation users as the seed-42 two-tower
checkpoint. The evaluator read train and validation only; it did not open the
test split. All 10,000 validation targets had ALS item factors and none
overlapped their users' training histories. The popularity comparator was
computed from the full training split.

| Method | K | Recall@K | Precision@K | NDCG@K | Hits |
|---|---:|---:|---:|---:|---:|
| Implicit ALS | 10 | 0.56% | 0.056% | 0.00285 | 56 / 10,000 |
| Full-train popularity | 10 | 2.08% | 0.208% | 0.01068 | 208 / 10,000 |
| Implicit ALS | 20 | 0.91% | 0.0455% | 0.00372 | 91 / 10,000 |
| Full-train popularity | 20 | 3.29% | 0.1645% | 0.01368 | 329 / 10,000 |

**Result:** implicit ALS lost to full-training popularity at both cutoffs and
on all listed metrics. This is one initial configuration, not evidence that
ALS can never help. The direct numerical comparison with existing two-tower
runs needs care: ALS trained on all training users, while the two-tower
checkpoint trained on a selected 10,000-user sample. The held-out test split
remains reserved for a later final evaluation.

Spark ML ALS does not expose a validation-loss value for each training
iteration through this estimator, so there is no ALS epoch-style loss table
for this run. The iteration count and final validation ranking metrics are
recorded instead; two-tower training continues to log its per-epoch BPR loss.

The model, `als_model.zip`, and machine-readable report are in the ignored
local directory `artifacts/two_tower/movies_tv_5m/als_v2/`; the MLflow run ID
is `e0819753ceba4eeb83a526af69d78e77`. This run logs the hyperparameters,
validation metrics, Spark model archive, and report. To reproduce the evaluation cohort,
`artifacts/two_tower/movies_tv_5m/als_validation_users.json` contains the
source user indices read from the seed-42 two-tower checkpoint. The runnable
commands are in the study guide's Step 7.

The corrected matched-seed experiment is recorded below.

## Matched three-model validation comparison (2026-10-05)

### Experimental contract

The comparison fixes 10,000 mapped users, all 110,923 of their training
interactions, their 10,000 validation targets, and a catalog of 50,507 items
that occur in those training interactions. Popularity, ALS, and two-tower
use the exact same train rows and candidates. ALS and two-tower were each
trained at seeds 42, 7, and 123. All ranking metrics use validation only;
the test split was not opened. The denominator remains 10,000 targets for
every method, including 2,613 targets outside the sampled candidate catalog.

The cohort and catalog are reproducibly recorded in
`artifacts/comparisons/movies_tv_5m/cohort_v1/`; their SHA-256 hashes are
`512f9adb171c635a240c8729678324f945bd9c6ee9155f1e01f9a6a9564ab4b4` and
`b487d444334404612ecf36697d38dabf4295fe44f18780177b847f4474f7fc1b`.
These generated artifacts are local and ignored by Git. The experiment
summary is `artifacts/comparisons/movies_tv_5m/summary_v1/comparison_summary.json`
and `.csv`.

### Overall results

Values are mean ± sample standard deviation across the three learned-model
seeds. Popularity is deterministic, so its standard deviation is zero.

| Model | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|---|---:|---:|---:|---:|
| Popularity | 2.09% | 0.010294 ± 0 | 2.99% | 0.012523 ± 0 |
| Implicit ALS | 1.82% ± 0.14 pp | 0.009612 ± 0.000680 | 2.83% ± 0.31 pp | 0.012160 ± 0.001118 |
| Two-tower | 2.21% ± 0.03 pp | 0.011180 ± 0.000401 | 3.23% ± 0.10 pp | 0.013735 ± 0.000547 |

Precision@K is Recall@K divided by K here because each user has exactly one
validation target. The two-tower mean exceeds popularity by 0.12 percentage
points of Recall@10 and 0.24 points of Recall@20; the three-seed spread is
small, but three seeds do not establish statistical significance. ALS loses
to popularity on the three-seed mean at both cutoffs. This supports continuing
with the two-tower candidate for validation-based refinement; it does not yet
pass a production quality gate.

### Per-seed ranking results

| Seed | Model | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|---:|---|---:|---:|---:|---:|
| fixed | Popularity | 2.09% | 0.010294 | 2.99% | 0.012523 |
| 42 | ALS | 1.86% | 0.009718 | 2.78% | 0.012008 |
| 7 | ALS | 1.93% | 0.010233 | 3.17% | 0.013347 |
| 123 | ALS | 1.66% | 0.008886 | 2.55% | 0.011126 |
| 42 | Two-tower | 2.25% | 0.011626 | 3.34% | 0.014363 |
| 7 | Two-tower | 2.19% | 0.011065 | 3.16% | 0.013474 |
| 123 | Two-tower | 2.19% | 0.010849 | 3.19% | 0.013367 |

The user-history breakdown shows average two-tower Recall@10 of 2.52%,
2.15%, and 1.07% for sparse (1–5), medium (6–20), and active (>20) users;
popularity scores are 2.47%, 2.00%, and 0.80%. For target popularity, the
two-tower recall is 0% for targets with 0–2 training occurrences, 0.27% for
3–10, and 10.44% for >10; popularity is 0%, 0%, and 10.23%. This highlights
the main limitation: the learned model helps in the observed tail/mid groups
only slightly, and neither model retrieves the rarest targets at K=10/20.
The candidate catalog also cannot rank the 2,613 held-out targets absent
from this selected training sample; those remain misses in the common
denominator. These are validation slice findings, not evidence about unseen
test performance.

### Two-tower epoch debugging log

Each run trained for 10 epochs on the same rows; validation loss is sampled
pairwise BPR loss used for checkpoint selection. It is not the ranking metric.

| Epoch | Seed 42 train / val | Seed 7 train / val | Seed 123 train / val |
|---:|---:|---:|---:|
| 1 | 0.693145 / 0.693118 | 0.693143 / 0.693127 | 0.693140 / 0.693120 |
| 2 | 0.691150 / 0.692426 | 0.691080 / 0.692237 | 0.691047 / 0.692314 |
| 3 | 0.680024 / 0.688252 | 0.679312 / 0.686942 | 0.678973 / 0.687147 |
| 4 | 0.651070 / 0.678394 | 0.648856 / 0.675185 | 0.647971 / 0.675599 |
| 5 | 0.605329 / 0.664063 | 0.601325 / 0.658985 | 0.599917 / 0.659928 |
| 6 | 0.550166 / 0.648187 | 0.544954 / 0.641491 | 0.543087 / 0.642915 |
| 7 | 0.493176 / 0.632428 | 0.487381 / 0.624655 | 0.485557 / 0.626621 |
| 8 | 0.438829 / 0.617857 | 0.432881 / 0.609374 | 0.431644 / 0.611705 |
| 9 | 0.389305 / 0.604925 | 0.383860 / 0.595684 | 0.382146 / 0.598280 |
| 10 | 0.345362 / 0.593487 | 0.340576 / 0.583794 | 0.338669 / 0.586435 |

All six curves fall each epoch. Training loss falls by about 50.2%–51.1%;
validation pairwise loss falls by about 14.2%–15.7%. The widening gap means
training pairs become easier faster than held-out pairs, so validation
ranking must guide later model selection. The similar final losses do not
guarantee identical rankings, as the per-seed table shows.

### Reproduction and tracking

From the WSL project root, prepare the fixed cohort once:

```bash
python -m src.models.prepare_comparison_cohort \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --output artifacts/comparisons/movies_tv_5m/cohort_v1 \
  --max-users 10000 --max-train-pairs 200000 --seed 42
```

For each seed (`42`, `7`, `123`), train the two-tower model on that cohort and
evaluate it against the shared candidate list:

```bash
python -m src.models.train_two_tower \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --output artifacts/comparisons/movies_tv_5m/two_tower/seed42 \
  --max-users 10000 --max-train-pairs 200000 --epochs 10 --seed 42 \
  --user-ids-file artifacts/comparisons/movies_tv_5m/cohort_v1/cohort.json

python -m src.models.evaluate_two_tower \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --model artifacts/comparisons/movies_tv_5m/two_tower/seed42/best_model.pt \
  --candidate-items artifacts/comparisons/movies_tv_5m/cohort_v1/candidate_items.json \
  --output artifacts/comparisons/movies_tv_5m/two_tower/seed42/validation_metrics.json \
  --k 10 20 --batch-size 128
```

For ALS, replace the seed and output directory for each run:

```bash
python -m src.models.collaborative_filtering \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --output artifacts/comparisons/movies_tv_5m/als/seed42 \
  --user-ids artifacts/comparisons/movies_tv_5m/cohort_v1/cohort.json \
  --candidate-items artifacts/comparisons/movies_tv_5m/cohort_v1/candidate_items.json \
  --train-scope cohort --rank 16 --max-iter 10 --reg-param 0.1 --alpha 1.0 \
  --seed 42 --k 10 20 --shuffle-partitions 32
```

Popularity is calculated by each evaluator from the same cohort train rows.
After all six learned-model reports exist, summarize and check their shared
counts against the cohort manifest:

```bash
python -m src.models.summarize_comparison \
  --comparison-dir artifacts/comparisons/movies_tv_5m \
  --cohort-dir artifacts/comparisons/movies_tv_5m/cohort_v1 \
  --output-dir artifacts/comparisons/movies_tv_5m/summary_v1 \
  --seeds 42 7 123
```

The command writes JSON and CSV with per-seed values and mean/sample standard
deviation, and rejects reports that disagree on users, targets, catalog, or
training scope. MLflow records the six training/evaluation runs locally.
This report summarizes the completed run rather than a fresh rerun; on this
machine, the model artifacts and per-run MLflow IDs are local. No markdown
file or generated model artifact needs to be pushed for reproduction if the
source data is not available; rerun preparation and the commands above.

## XGBoost pairwise ranking experiment (2026-10-05)

### Experiment contract

`src.models.xgboost_ranker` trained `XGBRanker` 3.2.0 with 150 boosting trees
on WSL Python 3.11 and Pandas 3.0.3, using 10,000 users and 110,923 unique
positive training pairs. Two uniformly sampled unseen items per positive
produced 332,769 training candidate rows. Spark selected only the checkpoint's
users; Pandas built their feature tables. The features were `log1p(item_count)`
and `log1p(history_affinity)`, where history affinity sums
training-only co-occurrence counts between a candidate and the user's other
training items. The 2,399,052 distinct item co-occurrence pairs were built
from training histories only. For positive training examples, the current
user's contribution was subtracted from the affinity feature. Validation
targets did not train the ranker or its features.

Validation used 10,000 users with one held-out target each. For every user,
the candidate list contained that target and 99 uniformly sampled items absent
from both training history and the user's validation target set: 1,000,000
rows in total. XGBoost and the seed-42 two-tower checkpoint ranked these exact
same candidates. The test split was not read. This is sampled-candidate
evaluation, not the earlier shared 50,507-item catalog comparison or full
catalog ranking.

### Results

| Ranker | K | Recall@K | Precision@K | NDCG@K | Hits |
|---|---:|---:|---:|---:|---:|
| XGBoost | 10 | 52.59% | 5.259% | 0.374505 | 5,259 / 10,000 |
| Two-tower, same candidates | 10 | 42.25% | 4.225% | 0.299654 | 4,225 / 10,000 |
| XGBoost | 20 | 61.83% | 3.092% | 0.397942 | 6,183 / 10,000 |
| Two-tower, same candidates | 20 | 52.01% | 2.601% | 0.324255 | 5,201 / 10,000 |

On this sampled list, XGBoost gained 10.34 percentage points of Recall@10 and
9.82 points of Recall@20. Feature importance was 0.643 for item popularity
and 0.357 for history affinity. These values are model diagnostics, not
causal explanations. The saved model was 253,275 bytes (about 247 KiB).

**Interpretation limit:** a random list of 100 candidates is much easier than
ranking tens of thousands of items. These recall values must not be compared
with the full-catalog or shared-catalog results above. The result shows that
the XGBoost ranker can order this sampled candidate set better than the
two-tower score in this single validation run; it does not establish a
production gain. The current API still serves only the two-tower bundle.

### Reproduction

From the WSL project root:

```bash
python -m src.models.xgboost_ranker \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --checkpoint artifacts/comparisons/movies_tv_5m/two_tower/seed42/best_model.pt \
  --output artifacts/xgboost_ranker/cohort_v4 \
  --estimators 150 --seed 42 --validation-negatives 99 --k 10 20
```

The recorded ignored outputs are `xgboost_ranker.json` and
`validation_metrics.json` under `artifacts/xgboost_ranker/cohort_v3/`. The
reproduction command uses the next empty directory, `cohort_v4`.

## Frozen-candidate test evaluation (2026-10-05)

### Gate and limitation

Before retraining the candidate, the project gate was written as: validation
NDCG@10 must be at least 5% higher than popularity and Recall@10 must not
regress. The fresh seed-42 run met both conditions: NDCG@10 was 0.011626
versus 0.010294 (+12.9%), and Recall@10 was 2.25% versus 2.09%. I then froze
that checkpoint and ran the evaluator once with `--split test`.

**Interpretation limit:** this is a final project checkpoint, but not a
strictly untouched unbiased test estimate. The same validation cohort and
seed-42 results had already been inspected during the preceding experiments;
the numeric gate was defined before this fresh retraining, but after those
validation results were known. Do not use this test result to tune or select
another model. A future unbiased estimate requires a newly reserved test set
that has not informed prior experimentation.

### Results

The test split produced 9,934 eligible users/targets from the fixed 10,000
user cohort; 66 cohort users had no eligible test target. Of the 9,934
targets, 2,947 were outside the 50,507-item sampled catalog and remained
misses in the metric denominator. No target overlapped the user's training
history.

| Method | K | Recall@K | Precision@K | NDCG@K | Hits |
|---|---:|---:|---:|---:|---:|
| Two-tower | 10 | 1.490% | 0.149% | 0.007616 | 148 / 9,934 |
| Popularity | 10 | 1.399% | 0.140% | 0.007134 | 139 / 9,934 |
| Two-tower | 20 | 2.335% | 0.117% | 0.009739 | 232 / 9,934 |
| Popularity | 20 | 2.184% | 0.109% | 0.009052 | 217 / 9,934 |

Two-tower is modestly ahead on these aggregate test metrics: +0.091 percentage
points Recall@10 and +0.151 points Recall@20. This is directionally consistent
with validation, while absolute recall remains low and tail-item performance
is weak. It is evidence for this sampled experiment only, not a deployment
claim.

### Reproduction record

The frozen checkpoint is at
`artifacts/model_selection/two_tower_seed42/best_model.pt`. Its validation
and test reports are `validation_metrics.json` and `test_metrics.json` in the
same ignored local directory. The test was run with:

```bash
python -m src.models.evaluate_two_tower \
  --data data/processed/movies_tv_5m/two_tower/v1 \
  --checkpoint artifacts/model_selection/two_tower_seed42/best_model.pt \
  --candidate-items artifacts/comparisons/movies_tv_5m/cohort_v1/candidate_items.json \
  --output artifacts/model_selection/two_tower_seed42/test_metrics.json \
  --split test --k 10 20
```

MLflow evaluation run: `b88307a45be944d6851c649acfc2c957`. The test split
should now be considered consumed for this cohort/model iteration.
