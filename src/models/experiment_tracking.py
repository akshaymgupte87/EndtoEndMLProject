"""Small MLflow helper for local recommender experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow


def log_experiment_run(
    *,
    experiment_name: str,
    run_name: str,
    parameters: Mapping[str, Any],
    metrics: Mapping[str, float],
    artifacts: Sequence[Path],
    step_metrics: Sequence[Mapping[str, float | int]] = (),
    tags: Mapping[str, str] | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Log params, final/epoch metrics, and files; return the MLflow run ID."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    missing_artifacts = [path for path in artifacts if not path.is_file()]
    if missing_artifacts:
        raise FileNotFoundError(f"MLflow artifacts do not exist: {missing_artifacts}")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, tags=dict(tags or {})) as run:
        mlflow.log_params(
            {
                key: value if isinstance(value, (str, int, float, bool)) else str(value)
                for key, value in parameters.items()
            }
        )
        if metrics:
            mlflow.log_metrics(dict(metrics))
        for record in step_metrics:
            step = int(record["epoch"])
            mlflow.log_metrics(
                {
                    key: float(value)
                    for key, value in record.items()
                    if key != "epoch"
                },
                step=step,
            )
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))
        return run.info.run_id
