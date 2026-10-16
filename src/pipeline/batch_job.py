"""Run the bounded train/evaluate/export workflow and package serving artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.models.train_two_tower import load_two_tower_checkpoint
from src.models.register_model import check_validation_gate


def _run(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def export_serving_artifacts(
    data: Path,
    output: Path,
    checkpoint: Path,
    lineage: dict[str, object] | None = None,
    popularity_item_indices: list[int] | None = None,
) -> Path:
    model, metadata = load_two_tower_checkpoint(checkpoint, device="cpu")
    lineage = lineage or {}
    output.mkdir(parents=True, exist_ok=True)
    item_mapping_path = data / "item_mapping"
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName("recommender-export-serving-artifacts")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    try:
        mapping = spark.read.parquet(str(item_mapping_path)).orderBy("item_idx").collect()
        train = spark.read.parquet(str(data / "train")).select("user_idx", "item_idx")
        user_ids = [int(value) for value in metadata["model_user_ids"]]
        from pyspark.sql.types import LongType, StructField, StructType

        cohort = spark.createDataFrame(
            [(value,) for value in user_ids],
            StructType([StructField("user_idx", LongType(), nullable=False)]),
        )
        histories = train.join(cohort, "user_idx", "inner").collect()
    finally:
        spark.stop()

    ordered_ids = [int(row.item_idx) for row in mapping]
    if ordered_ids != list(range(model.num_items)):
        raise ValueError("item mapping must match contiguous model item indices")
    model.eval()
    import torch

    with torch.inference_mode():
        vectors = model.encode_items(torch.arange(model.num_items)).cpu().numpy()
    np.save(output / "item_vectors.npy", vectors)
    (output / "item_ids.json").write_text(
        json.dumps([str(row.item_id) for row in mapping]), encoding="utf-8"
    )
    (output / "user_ids.json").write_text(json.dumps(user_ids), encoding="utf-8")
    seen: dict[str, list[int]] = {}
    source_to_local = {source_id: local for local, source_id in enumerate(user_ids)}
    for row in histories:
        seen.setdefault(str(source_to_local[int(row.user_idx)]), []).append(int(row.item_idx))
    (output / "seen_items.json").write_text(json.dumps(seen), encoding="utf-8")
    if popularity_item_indices is not None:
        if len(popularity_item_indices) != len(set(popularity_item_indices)) or any(
            item < 0 or item >= model.num_items for item in popularity_item_indices
        ):
            raise ValueError("popularity fallback indices must be unique catalog indices")
        (output / "popularity_item_indices.json").write_text(
            json.dumps(popularity_item_indices), encoding="utf-8"
        )
    metadata_path = output / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_type": "two_tower_id_embeddings",
                "num_users": model.num_users,
                "num_items": model.num_items,
                "embedding_dim": model.embedding_dim,
                "model_user_ids": user_ids,
                "checkpoint": checkpoint.name,
                **lineage,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "model_type": "two_tower_id_embeddings",
        "num_users": model.num_users,
        "num_items": model.num_items,
        **lineage,
        "files": {},
    }
    artifact_names = ["best_model.pt", "item_vectors.npy", "item_ids.json", "user_ids.json", "seen_items.json", "model_metadata.json"]
    if popularity_item_indices is not None:
        artifact_names.append("popularity_item_indices.json")
    for name in artifact_names:
        path = output / name
        if name == "best_model.pt" and not path.exists():
            import shutil

            shutil.copy2(checkpoint, path)
        manifest["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    bundle_path = output.parent / "model_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    return bundle_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Encoded train/validation data")
    parser.add_argument("--output", type=Path, required=True, help="New run output directory")
    parser.add_argument("--user-ids-file", type=Path, help="Optional fixed cohort JSON")
    parser.add_argument("--max-users", type=int, default=2_000)
    parser.add_argument("--max-train-pairs", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-items", type=Path, help="Optional shared-catalog JSON")
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        help="MLflow tracking URI (defaults to MLFLOW_TRACKING_URI or local SQLite)",
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be new or empty")
    if not args.data.is_dir():
        parser.error("--data must point to an existing encoded dataset")
    args.output.mkdir(parents=True, exist_ok=True)
    pipeline_run_id = str(uuid.uuid4())
    trained_at_utc = datetime.now(timezone.utc).isoformat()
    run_env = os.environ.copy()
    train_cmd = [
        sys.executable, "-m", "src.models.train_two_tower",
        "--data", str(args.data), "--output", str(args.output),
        "--max-users", str(args.max_users), "--max-train-pairs", str(args.max_train_pairs),
        "--epochs", str(args.epochs), "--seed", str(args.seed),
        "--mlflow-tracking-uri", args.mlflow_tracking_uri,
    ]
    if args.user_ids_file:
        train_cmd.extend(["--user-ids-file", str(args.user_ids_file)])
    _run(train_cmd, env=run_env)
    evaluate_cmd = [
        sys.executable, "-m", "src.models.evaluate_two_tower",
        "--data", str(args.data), "--checkpoint", str(args.output / "best_model.pt"),
        "--output", str(args.output / "validation_metrics.json"),
        "--split", "validation", "--k", "10", "20",
        "--mlflow-tracking-uri", args.mlflow_tracking_uri,
    ]
    if args.candidate_items:
        evaluate_cmd.extend(["--candidate-items", str(args.candidate_items)])
    _run(evaluate_cmd, env=run_env)
    report = json.loads((args.output / "validation_metrics.json").read_text(encoding="utf-8"))
    try:
        gate = {"passed": True, **check_validation_gate(report)}
    except ValueError as error:
        gate = {"passed": False, "reason": str(error)}
    serving_policy = "two_tower" if gate["passed"] else "popularity"
    (args.output / "validation_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    evaluation_run = json.loads(
        (args.output / "evaluation_mlflow_run.json").read_text(encoding="utf-8")
    )
    training_run_path = args.output / "training_mlflow_run.json"
    training_run = (
        json.loads(training_run_path.read_text(encoding="utf-8"))
        if training_run_path.is_file()
        else {"run_id": None}
    )
    model_version = "sha256:" + hashlib.sha256(
        (args.output / "best_model.pt").read_bytes()
    ).hexdigest()
    lineage = {
        "model_version": model_version,
        "pipeline_run_id": pipeline_run_id,
        "training_mlflow_run_id": training_run["run_id"],
        "evaluation_mlflow_run_id": evaluation_run["run_id"],
        "trained_at_utc": trained_at_utc,
        "seed": args.seed,
        "serving_policy": serving_policy,
    }
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName("recommender-export-popularity-fallback")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    try:
        popularity_rows = (
            spark.read.parquet(str(args.data / "train"))
            .groupBy("item_idx")
            .count()
            .orderBy("count", "item_idx", ascending=[False, True])
            .select("item_idx")
            .collect()
        )
        popularity_indices = [int(row.item_idx) for row in popularity_rows]
    finally:
        spark.stop()
    bundle = export_serving_artifacts(
        args.data,
        args.output / "serving",
        args.output / "best_model.pt",
        lineage=lineage,
        popularity_item_indices=popularity_indices,
    )
    manifest = {
        **lineage,
        "input_data": str(args.data),
        "users": report["users_evaluated"],
        "validation_metrics": report["metrics_at_k"],
        "validation_gate": gate,
        "serving_policy": serving_policy,
        "validation_report": str(args.output / "validation_metrics.json"),
        "serving_bundle": str(bundle),
    }
    (args.output / "batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
