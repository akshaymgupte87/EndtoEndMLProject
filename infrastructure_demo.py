"""Generate local, synthetic ML infrastructure evidence without Spark or cloud jobs."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import mlflow
import numpy as np
import torch

from demo import build_serving_bundle, rank_metrics
from src.api.app import create_app
from src.models.experiment_tracking import log_experiment_run
from src.models.register_model import check_validation_gate, register_checkpoint


async def request(app, user):
    status = 0
    body = bytearray()

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            body.extend(message.get("body", b""))

    path = f"/recommendations/{user}"
    await app({
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "GET", "scheme": "http", "path": path,
        "raw_path": path.encode(), "query_string": b"limit=10", "root_path": "",
        "headers": [], "client": ("127.0.0.1", 12345), "server": ("demo", 80),
    }, receive, send)
    if status != 200 or not json.loads(body).get("items"):
        raise RuntimeError(f"benchmark request failed: {status} {body!r}")


async def benchmark(bundle, users, count):
    app = create_app(bundle)
    results = {}
    async with app.router.lifespan_context(app):
        if app.state.runtime.get("redis") is not None:
            raise RuntimeError("Unset REDIS_URL for the isolated local benchmark")
        for i in range(20):
            await request(app, users[i % len(users)])
        for mode in ("uncached", "memory_cached"):
            if mode == "memory_cached":
                for user in users:
                    await request(app, user)
            samples = []
            for i in range(count):
                if mode == "uncached":
                    app.state.runtime["cache"].clear()
                start = time.perf_counter_ns()
                await request(app, users[i % len(users)])
                samples.append((time.perf_counter_ns() - start) / 1e6)
            results[mode] = {
                "requests": count, "errors": 0,
                **{f"p{p}_ms": float(np.percentile(samples, p)) for p in (50, 95, 99)},
                "samples_ms": samples,
            }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ml_infrastructure"))
    parser.add_argument("--requests", type=int, default=200)
    args = parser.parse_args()
    if args.requests < 20:
        parser.error("--requests must be at least 20")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{output / 'mlflow.db'}"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "device": "cpu", "torch_threads": 1,
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2))
    runs = []
    reports = []
    weights = []
    for repeat in range(2):
        bundle = output / f"repeat_{repeat}"
        data, metrics, info = build_serving_bundle(bundle, seed=42)
        report = {"split": "validation", "metrics_at_k": {}}
        for key, rankings in (("two_tower", "model_rankings"),
                              ("popularity_baseline_on_same_user_sample", "popularity_rankings")):
            measured = rank_metrics(metrics[rankings], data["validation_pairs"], 10)
            report["metrics_at_k"][key] = {"10": {
                "recall": measured["recall@10"], "ndcg": measured["ndcg@10"],
            }}
        report_path = bundle / "validation_metrics.json"
        report_path.write_text(json.dumps(report, indent=2))
        state = torch.load(bundle / "best_model.pt", weights_only=True)["state_dict"]
        weights.append(hashlib.sha256(b"".join(
            state[key].numpy().tobytes() for key in sorted(state)
        )).hexdigest())
        run_id = log_experiment_run(
            experiment_name="ml-infrastructure-synthetic", run_name=f"seed42-repeat{repeat}",
            parameters={**info["training_config"], "seed": 42,
                        "data_sha256": info["dataset_fingerprint_sha256"],
                        "weights_sha256": weights[-1]},
            metrics={"validation_ndcg_at_10": report["metrics_at_k"]["two_tower"]["10"]["ndcg"]},
            artifacts=[report_path, bundle / "run_report.json", bundle / "best_model.pt",
                       bundle / "manifest.json", output / "environment.json"],
            tags={"synthetic": "true"}, tracking_uri=uri,
        )
        # Read back persisted metrics; a successful logging call alone is insufficient.
        persisted = mlflow.tracking.MlflowClient(tracking_uri=uri).get_run(run_id)
        assert persisted.data.metrics["validation_ndcg_at_10"] == report["metrics_at_k"]["two_tower"]["10"]["ndcg"]
        runs.append({"run_id": run_id, "dataset_sha256": info["dataset_fingerprint_sha256"],
                     "weights_sha256": weights[-1], "best_epoch": info["best_epoch"]})
        reports.append(report)
    reproducible = weights[0] == weights[1] and reports[0] == reports[1] and runs[0]["dataset_sha256"] == runs[1]["dataset_sha256"]
    if not reproducible:
        raise RuntimeError("seeded repeatability check failed")
    rejected = {}
    for name, report in (("test_split", {**reports[0], "split": "test"}),
                         ("no_lift", {"split": "validation", "metrics_at_k": {
                             "two_tower": {"10": {"recall": .02, "ndcg": .01}},
                             "popularity_baseline_on_same_user_sample": {"10": {"recall": .02, "ndcg": .01}},
                         }})):
        try:
            check_validation_gate(report)
        except ValueError as error:
            rejected[name] = str(error)
        else:
            raise RuntimeError(f"negative gate case accepted: {name}")
    try:
        gate = check_validation_gate(reports[0])
    except ValueError as error:
        registration = {"registered": False, "reason": str(error)}
    else:
        registration = register_checkpoint(output / "repeat_0/best_model.pt",
            output / "repeat_0/validation_metrics.json", "synthetic-infrastructure-demo", uri)
        registration["gate"] = gate
    result = {
        "scope": "synthetic local workflow evidence; not real-data quality or production capacity",
        "selection": "lowest validation BPR checkpoint; register only with >=5% validation NDCG@10 lift and no Recall@10 regression against matched popularity",
        "tracking_uri": uri, "runs": runs, "reproducible": reproducible,
        "negative_gate_cases": rejected, "registration": registration,
        "latency_protocol": "in-process ASGI, sequential concurrency=1, 20 warmups; excludes startup/network/TLS; cache cleared outside timing for uncached requests",
        "latency": asyncio.run(benchmark(output / "repeat_0", data["user_ids"], args.requests)),
    }
    (output / "evidence.json").write_text(json.dumps(result, indent=2))
    print(f"Evidence: {output / 'evidence.json'}")


if __name__ == "__main__":
    main()
