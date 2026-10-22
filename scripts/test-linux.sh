#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYSPARK_PYTHON:-$(command -v python3 || command -v python || true)}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python executable not found: ${python_bin:-no python3/python on PATH}. Set PYSPARK_PYTHON." >&2
  exit 2
fi

if ! "$python_bin" -c 'import sys, pyspark; assert sys.version_info >= (3, 11), f"Python 3.11+ required, found {sys.version.split()[0]}"; assert pyspark.__version__.split(".", 1)[0] == "4", f"PySpark 4.x required, found {pyspark.__version__}"' 2>/dev/null; then
  echo "The selected Python ($python_bin) must be Python 3.11+ with PySpark 4.x installed." >&2
  echo 'Install the project dependencies in that same Linux environment, then rerun this script.' >&2
  exit 2
fi

if [[ -z "${JAVA_HOME:-}" ]]; then
  java_on_path="$(command -v java || true)"
  if [[ -n "$java_on_path" ]]; then
    JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$java_on_path")")")"
  fi
fi
if [[ -z "${JAVA_HOME:-}" || ! -x "$JAVA_HOME/bin/java" ]]; then
  echo 'Java was not found. Install a Java 17+ JDK and set JAVA_HOME (Java 21 is the tested version).' >&2
  exit 2
fi

java_version="$("$JAVA_HOME/bin/java" -version 2>&1 | sed -nE 's/.*version "([0-9]+).*/\1/p' | head -n 1)"
if [[ ! "$java_version" =~ ^[0-9]+$ ]] || (( java_version < 17 )); then
  echo "PySpark 4.x requires Java 17+; found ${java_version:-unknown} at $JAVA_HOME/bin/java." >&2
  exit 2
fi

export PYSPARK_PYTHON="$python_bin"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-127.0.0.1}"
export SPARK_HOME="${SPARK_HOME:-$($python_bin -c 'import pathlib, pyspark; print(pathlib.Path(pyspark.__file__).parent)')}"
if [[ ! -x "$SPARK_HOME/bin/spark-submit" || ! -d "$SPARK_HOME/jars" ]]; then
  echo "SPARK_HOME does not contain a usable Spark distribution: $SPARK_HOME" >&2
  echo 'Unset SPARK_HOME to use the Spark jars bundled with the selected PySpark package.' >&2
  exit 2
fi
export PATH="$SPARK_HOME/bin:$JAVA_HOME/bin:$PATH"
export PYSPARK_SUBMIT_ARGS="${PYSPARK_SUBMIT_ARGS:---driver-memory 8g pyspark-shell}"

"$python_bin" -c 'import sys, pyspark; print(f"Python: {sys.executable} ({sys.version.split()[0]})"); print(f"PySpark: {pyspark.__version__} ({pyspark.__file__})")'
echo "Java: $java_version ($JAVA_HOME)"
echo "Spark: $SPARK_HOME"
cd "$repo_root"
exec "$python_bin" -m pytest -o "cache_dir=${PYTEST_CACHE_DIR:-/tmp/endtoendml-pytest-cache}" "$@"
