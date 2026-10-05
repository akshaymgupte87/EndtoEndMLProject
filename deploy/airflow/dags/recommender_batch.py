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
            "export JAVA_HOME={{ var.value.get('recommender_java_home', '/usr/lib/jvm/java-21-openjdk-amd64') }} && "
            "export SPARK_HOME={{ var.value.get('recommender_spark_home', '/opt/venv/lib/python3.11/site-packages/pyspark') }} && "
            "export PATH={{ var.value.get('recommender_spark_home', '/opt/venv/lib/python3.11/site-packages/pyspark') }}/bin:{{ var.value.get('recommender_python_dir', '/opt/venv/bin') }}:{{ var.value.get('recommender_java_home', '/usr/lib/jvm/java-21-openjdk-amd64') }}/bin:$PATH && "
            "export PYSPARK_PYTHON={{ var.value.get('recommender_python', '/opt/venv/bin/python') }} && "
            "export PYTHONPATH={{ var.value.get('recommender_project_dir', '/opt/recommender') }} && "
            "export PYSPARK_SUBMIT_ARGS='--driver-memory {{ var.value.get('recommender_spark_driver_memory', '8g') }} pyspark-shell' && "
            "cd {{ var.value.get('recommender_project_dir', '/opt/recommender') }} && "
            "{{ var.value.get('recommender_python', '/opt/venv/bin/python') }} -m src.pipeline.batch_job "
            "--data {{ var.value.get('recommender_data_dir', '/opt/recommender/data/processed/movies_tv_5m/two_tower/v1') }} "
            "--output {{ var.value.get('recommender_output_root', '/opt/recommender/artifacts/airflow_runs') }}/{{ run_id | replace(':', '_') | replace('+', '_') }} "
            "--max-users {{ dag_run.conf.get('max_users', params.max_users) }} "
            "--max-train-pairs {{ dag_run.conf.get('max_train_pairs', params.max_train_pairs) }} "
            "--epochs {{ dag_run.conf.get('epochs', params.epochs) }} "
            "--seed {{ dag_run.conf.get('seed', params.seed) }}"
        ),
        params={
            "max_users": 2000,
            "max_train_pairs": 100000,
            "epochs": 10,
            "seed": 42,
        },
    )
