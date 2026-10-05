"""Register a two-tower checkpoint only after its validation quality gate passes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.models.train_two_tower import load_two_tower_checkpoint


def check_validation_gate(
    report: dict[str, Any], minimum_ndcg_lift: float = 0.05
) -> dict[str, float]:
    """Return gate measurements or reject non-validation/underperforming runs."""
    if not math.isfinite(minimum_ndcg_lift) or minimum_ndcg_lift < 0:
        raise ValueError("minimum NDCG lift must be a finite non-negative number")
    if report.get("split") != "validation":
        raise ValueError("only a validation report can authorize model registration")
    metrics = report.get("metrics_at_k", {})
    learned = metrics.get("two_tower", {}).get("10", {})
    baseline = metrics.get("popularity_baseline_on_same_user_sample", {}).get("10", {})
    try:
        model_ndcg = float(learned["ndcg"])
        baseline_ndcg = float(baseline["ndcg"])
        model_recall = float(learned["recall"])
        baseline_recall = float(baseline["recall"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("validation report must contain two-tower and popularity Recall/NDCG@10") from error
    if not all(math.isfinite(value) for value in (model_ndcg, baseline_ndcg, model_recall, baseline_recall)):
        raise ValueError("validation gate metrics must be finite numbers")
    if baseline_ndcg <= 0:
        raise ValueError("popularity baseline NDCG@10 must be positive")
    lift = model_ndcg / baseline_ndcg - 1.0
    if lift < minimum_ndcg_lift:
        raise ValueError(
            f"validation gate failed: NDCG@10 lift {lift:.3%} is below {minimum_ndcg_lift:.3%}"
        )
    if model_recall < baseline_recall:
        raise ValueError("validation gate failed: two-tower Recall@10 is below popularity")
    return {
        "ndcg_at_10": model_ndcg,
        "popularity_ndcg_at_10": baseline_ndcg,
        "ndcg_lift": lift,
        "recall_at_10": model_recall,
        "popularity_recall_at_10": baseline_recall,
    }


def register_checkpoint(
    checkpoint: Path,
    validation_report: Path,
    registered_name: str,
    tracking_uri: str = "sqlite:///mlflow.db",
    minimum_ndcg_lift: float = 0.05,
) -> dict[str, str | float]:
    report = json.loads(validation_report.read_text(encoding="utf-8"))
    gate = check_validation_gate(report, minimum_ndcg_lift)
    model, metadata = load_two_tower_checkpoint(checkpoint, device="cpu")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment("two-tower-recommender")
    with mlflow.start_run(run_name=f"register-{registered_name}") as run:
        mlflow.log_params(
            {
                "checkpoint": str(checkpoint),
                "best_epoch": metadata.get("best_epoch", -1),
                "model_users": model.num_users,
                "catalog_items": model.num_items,
                "validation_report": str(validation_report),
            }
        )
        mlflow.log_metrics(gate)
        mlflow.set_tag("stage", "quality_gated_registration")
        model_info = mlflow.pytorch.log_model(
            model,
            name="model",
            serialization_format="pickle",
            registered_model_name=registered_name,
            code_paths=["src"],
            pip_requirements=["torch>=2.6,<3", "numpy>=2,<3"],
        )
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{registered_name}'")
    if not versions:
        raise RuntimeError("MLflow did not create a registered model version")
    version = max(versions, key=lambda entry: int(entry.version))
    client.set_registered_model_alias(registered_name, "candidate", version.version)
    return {
        "registered_name": registered_name,
        "version": str(version.version),
        "alias": "candidate",
        "model_uri": model_info.model_uri,
        "run_id": run.info.run_id,
        **gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--registered-name", default="amazon-reviews-two-tower")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--minimum-ndcg-lift", type=float, default=0.05)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.validation_report.is_file():
        parser.error("checkpoint and validation report must exist")
    if not math.isfinite(args.minimum_ndcg_lift) or args.minimum_ndcg_lift < 0:
        parser.error("minimum-ndcg-lift must be non-negative")
    result = register_checkpoint(
        args.checkpoint,
        args.validation_report,
        args.registered_name,
        args.tracking_uri,
        args.minimum_ndcg_lift,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
