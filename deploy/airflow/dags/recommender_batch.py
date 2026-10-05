"""Minimal Airflow DAG: run one local batch training command on demand."""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="recommender_batch_training",
    description="Prepare a two-tower model bundle from already-encoded data",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["learning-demo", "recommender"],
) as dag:
    train_and_package = BashOperator(
        task_id="train_evaluate_export",
        bash_command=(
            "cd {{ params.project_dir }} && "
            "python -m src.pipeline.batch_job "
            "--data {{ params.data_dir }} "
            "--output {{ params.output_root }}/{{ run_id | replace(':', '_') | replace('+', '_') }} "
            "--max-users 2000 --max-train-pairs 100000 --epochs 10 --seed 42"
        ),
        params={
            "project_dir": "/opt/recommender",
            "data_dir": "/opt/recommender/data/processed/movies_tv_5m/two_tower/v1",
            "output_root": "/opt/recommender/artifacts/airflow_runs",
        },
    )
