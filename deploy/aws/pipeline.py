"""Create and start the SageMaker Training Pipeline for the recommender."""

from __future__ import annotations

import argparse
import time

import boto3
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import TrainingStep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--pipeline-role-arn", required=True)
    parser.add_argument("--training-role-arn", required=True)
    parser.add_argument("--training-image", required=True)
    parser.add_argument("--data-s3-uri", required=True)
    parser.add_argument("--model-s3-uri", required=True)
    parser.add_argument("--ecs-cluster", required=True)
    parser.add_argument("--ecs-service", required=True)
    parser.add_argument("--pipeline-name", default="recommender-cloud-training")
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--max-train-pairs", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--instance-volume-gb", type=int, default=30)
    parser.add_argument("--max-runtime-seconds", type=int, default=14400)
    parser.add_argument("--max-wait-seconds", type=int, default=21600)
    parser.add_argument("--wait", action="store_true", help="Wait for the execution to finish")
    args = parser.parse_args()

    session = boto3.Session(region_name=args.region)
    sagemaker_session = PipelineSession(boto_session=session)
    data_uri = ParameterString(name="TrainingDataS3Uri", default_value=args.data_s3_uri)
    model_uri = ParameterString(name="ModelS3Uri", default_value=args.model_s3_uri)
    cluster = ParameterString(name="EcsCluster", default_value=args.ecs_cluster)
    service = ParameterString(name="EcsService", default_value=args.ecs_service)
    max_users = ParameterInteger(name="MaxUsers", default_value=args.max_users)
    max_pairs = ParameterInteger(name="MaxTrainPairs", default_value=args.max_train_pairs)
    epochs = ParameterInteger(name="Epochs", default_value=args.epochs)
    seed = ParameterInteger(name="Seed", default_value=args.seed)

    artifact_bucket = args.model_s3_uri.removeprefix("s3://").split("/", 1)[0]
    estimator = Estimator(
        image_uri=args.training_image,
        role=args.training_role_arn,
        instance_count=1,
        instance_type=args.instance_type,
        volume_size=args.instance_volume_gb,
        max_run=args.max_runtime_seconds,
        use_spot_instances=True,
        max_wait=args.max_wait_seconds,
        output_path=f"s3://{artifact_bucket}/training-job-output",
        sagemaker_session=sagemaker_session,
        base_job_name="recommender-train",
        environment={
            "MODEL_S3_URI": model_uri,
            "ECS_CLUSTER": cluster,
            "ECS_SERVICE": service,
        },
        hyperparameters={
            "max-users": max_users,
            "max-train-pairs": max_pairs,
            "epochs": epochs,
            "seed": seed,
        },
    )
    step_args = estimator.fit(
        inputs={
            "training": TrainingInput(
                s3_data=data_uri,
                s3_data_type="S3Prefix",
                input_mode="File",
            )
        }
    )
    train_and_deploy = TrainingStep(name="TrainValidatePromoteAndDeploy", step_args=step_args)
    pipeline = Pipeline(
        name=args.pipeline_name,
        parameters=[data_uri, model_uri, cluster, service, max_users, max_pairs, epochs, seed],
        steps=[train_and_deploy],
        sagemaker_session=sagemaker_session,
    )
    pipeline.upsert(role_arn=args.pipeline_role_arn)
    execution = pipeline.start()
    print(f"Pipeline execution started: {execution.arn}")
    if args.wait:
        terminal = {"Succeeded", "Failed", "Stopped"}
        while True:
            description = execution.describe()
            status = description["PipelineExecutionStatus"]
            print(f"Pipeline execution status: {status}")
            if status in terminal:
                if status != "Succeeded":
                    raise SystemExit(f"Pipeline execution ended with status {status}")
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
