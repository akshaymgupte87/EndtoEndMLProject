"""Tests for local MLflow experiment logging."""

import mlflow

from src.models.experiment_tracking import log_experiment_run


def test_log_experiment_run_records_parameters_metrics_and_artifacts(tmp_path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    history = tmp_path / "history.json"
    history.write_text('{"epochs": 2}', encoding="utf-8")

    run_id = log_experiment_run(
        experiment_name="test-recommender",
        run_name="small-test-run",
        parameters={"seed": 17, "model": "two_tower"},
        metrics={"best_validation_loss": 0.25},
        step_metrics=[
            {"epoch": 1, "train_loss": 0.5},
            {"epoch": 2, "train_loss": 0.3},
        ],
        artifacts=[history],
        tracking_uri=tracking_uri,
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert run.data.params["seed"] == "17"
    assert run.data.params["model"] == "two_tower"
    assert run.data.metrics["best_validation_loss"] == 0.25
    assert [metric.step for metric in client.get_metric_history(run_id, "train_loss")] == [1, 2]
    assert [artifact.path for artifact in client.list_artifacts(run_id)] == ["history.json"]
