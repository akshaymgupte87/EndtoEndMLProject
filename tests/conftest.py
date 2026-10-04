"""Shared small Spark runtime for behavior and Parquet integration tests."""

import os
import sys

import pytest


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    from pyspark.sql import SparkSession

    # Driver and Python workers must use the same interpreter, not PATH's python.
    previous_python = os.environ.get("PYSPARK_PYTHON")
    os.environ["PYSPARK_PYTHON"] = sys.executable
    warehouse = tmp_path_factory.mktemp("spark-warehouse").as_uri()
    session = None
    try:
        session = (
            SparkSession.builder.master("local[2]")
            .appName("recommender-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.default.parallelism", "2")
            .config("spark.sql.warehouse.dir", warehouse)
            .getOrCreate()
        )
        session.sparkContext.setLogLevel("ERROR")
        yield session
    finally:
        if session is not None:
            session.stop()
        if previous_python is None:
            os.environ.pop("PYSPARK_PYTHON", None)
        else:
            os.environ["PYSPARK_PYTHON"] = previous_python
