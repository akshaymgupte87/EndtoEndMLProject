"""Train, validate, promote, and refresh the recommender in a SageMaker Training Job."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import boto3


def _training_parameters() -> dict[str, str]:
    path = Path("/opt/ml/input/config/hyperparameters.json")
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    data_dir = Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    model_dir.mkdir(parents=True, exist_ok=True)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"SageMaker training channel was not mounted: {data_dir}")

    parameters = _training_parameters()
    run_dir = model_dir / "run"
    tracking_uri = f"sqlite:///{model_dir / 'mlflow.db'}"
    command = [
        sys.executable,
        "-m",
        "src.pipeline.batch_job",
        "--data",
        str(data_dir),
        "--output",
        str(run_dir),
        "--max-users",
        str(int(parameters.get("max-users", "2000"))),
        "--max-train-pairs",
        str(int(parameters.get("max-train-pairs", "100000"))),
        "--epochs",
        str(int(parameters.get("epochs", "10"))),
        "--seed",
        str(int(parameters.get("seed", "42"))),
        "--mlflow-tracking-uri",
        tracking_uri,
    ]
    subprocess.run(command, check=True, env=os.environ.copy())

    manifest_path = run_dir / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle_path = Path(manifest["serving_bundle"])
    if not bundle_path.is_file():
        raise FileNotFoundError(f"batch pipeline did not create serving bundle: {bundle_path}")

    model_s3_uri = os.environ["MODEL_S3_URI"]
    if not model_s3_uri.startswith("s3://"):
        raise ValueError("MODEL_S3_URI must be an s3:// URI")
    bucket, _, key = model_s3_uri[5:].partition("/")
    if not bucket or not key:
        raise ValueError("MODEL_S3_URI must include both bucket and key")

    # The batch job writes model_bundle.zip at the parent of run_dir, which is
    # SageMaker's model directory and becomes the Training Job model.tar.gz.
    boto3.client("s3").upload_file(str(bundle_path), bucket, key)
    print(json.dumps({"promoted_model": model_s3_uri, **manifest}, indent=2))

    response = boto3.client("ecs").update_service(
        cluster=os.environ["ECS_CLUSTER"],
        service=os.environ["ECS_SERVICE"],
        desiredCount=1,
        forceNewDeployment=True,
    )
    print(
        json.dumps(
            {
                "ecs_cluster": os.environ["ECS_CLUSTER"],
                "ecs_service": os.environ["ECS_SERVICE"],
                "deployment_id": response["service"].get("deployments", [{}])[0].get("id"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
