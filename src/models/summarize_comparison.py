"""Validate and summarize matched multi-seed recommender evaluation reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def mean_and_sample_std(values: list[float]) -> tuple[float, float]:
    """Return arithmetic mean and sample standard deviation."""
    if not values:
        raise ValueError("at least one metric value is required")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _metric_rows(report: dict[str, Any], model_key: str) -> dict[str, dict[str, float]]:
    return {
        str(k): {name: float(value) for name, value in metrics.items() if name != "users"}
        for k, metrics in report["metrics_at_k"][model_key].items()
    }


def _cohort_metric_rows(
    report: dict[str, Any], model_key: str
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for cohort, cohort_report in report.get("metrics_by_cohort", {}).items():
        if model_key not in cohort_report["metrics_at_k"]:
            continue
        result[cohort] = _metric_rows(
            {"metrics_at_k": cohort_report["metrics_at_k"]}, model_key
        )
    return result


def build_summary(comparison_dir: Path, cohort_dir: Path, seeds: list[int]) -> dict[str, Any]:
    manifest = _read_json(cohort_dir / "manifest.json")
    runs: list[dict[str, Any]] = []
    baseline_reference: dict[str, dict[str, float]] | None = None
    common = None

    for seed in seeds:
        tower = _read_json(
            comparison_dir / "two_tower" / f"seed{seed}" / "validation_metrics.json"
        )
        als = _read_json(comparison_dir / "als" / f"seed{seed}" / "validation_metrics.json")
        if als.get("training_scope") != "cohort":
            raise ValueError(f"ALS seed {seed} was not trained on the shared cohort")
        for report in (tower, als):
            current = (
                report["users_evaluated"],
                report["candidate_items"],
                report["validation_targets_evaluated"],
                report["validation_targets_outside_candidate_catalog"],
            )
            if common is None:
                common = current
            elif current != common:
                raise ValueError("model reports do not use the same users, targets, and candidate catalog")
        if common != (
            manifest["users"],
            manifest["candidate_items"],
            manifest["validation_targets"],
            manifest["validation_targets_outside_candidate_catalog"],
        ):
            raise ValueError("run report counts do not match the cohort manifest")

        tower_base = _metric_rows(tower, "popularity_baseline_on_same_user_sample")
        als_base = _metric_rows(als, "popularity_baseline")
        if baseline_reference is None:
            baseline_reference = tower_base
        if tower_base != als_base or tower_base != baseline_reference:
            raise ValueError("popularity baseline changed between model or seed runs")

        runs.append({
            "model": "two_tower",
            "seed": seed,
            "metrics_at_k": _metric_rows(tower, "two_tower"),
            "metrics_by_cohort": _cohort_metric_rows(tower, "two_tower"),
        })
        runs.append({
            "model": "implicit_als",
            "seed": seed,
            "metrics_at_k": _metric_rows(als, "implicit_als"),
            "metrics_by_cohort": _cohort_metric_rows(als, "implicit_als"),
        })

    if common is None or baseline_reference is None:
        raise ValueError("at least one seed is required")
    runs.append({
        "model": "popularity",
        "seed": "deterministic",
        "metrics_at_k": baseline_reference,
        "metrics_by_cohort": _cohort_metric_rows(tower, "popularity_baseline_on_same_user_sample"),
    })

    metric_groups: dict[tuple[str, str, str, str], list[float]] = {}
    for run in runs:
        for k, metrics in run["metrics_at_k"].items():
            for metric, value in metrics.items():
                metric_groups.setdefault((run["model"], "overall", k, metric), []).append(value)
        for cohort, by_k in run["metrics_by_cohort"].items():
            for k, metrics in by_k.items():
                for metric, value in metrics.items():
                    metric_groups.setdefault((run["model"], cohort, k, metric), []).append(value)

    summaries = []
    for (model, cohort, k, metric), values in sorted(metric_groups.items()):
        mean, sample_std = mean_and_sample_std(values)
        summaries.append({
            "model": model,
            "cohort": cohort,
            "k": int(k),
            "metric": metric,
            "mean": mean,
            "sample_std": sample_std,
            "runs": len(values),
        })
    return {
        "split": "validation",
        "test_split_read": False,
        "seeds": seeds,
        "cohort": manifest,
        "runs": runs,
        "mean_and_sample_std": summaries,
        "interpretation_limit": "Three seeds describe run-to-run variation; they do not establish statistical significance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be a non-empty list of unique integers")
    result = build_summary(args.comparison_dir, args.cohort_dir, args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "comparison_summary.json"
    csv_path = args.output_dir / "comparison_summary.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["model", "seed", "cohort", "k", "metric", "value", "sample_std", "runs"],
        )
        writer.writeheader()
        for run in result["runs"]:
            groups = {"overall": run["metrics_at_k"], **run["metrics_by_cohort"]}
            for cohort, by_k in groups.items():
                for k, metrics in by_k.items():
                    for metric, value in metrics.items():
                        writer.writerow({
                            "model": run["model"], "seed": run["seed"],
                            "cohort": cohort, "k": k, "metric": metric,
                            "value": value, "sample_std": "", "runs": "",
                        })
        for summary in result["mean_and_sample_std"]:
            writer.writerow({
                "model": summary["model"],
                "seed": "mean",
                "cohort": summary["cohort"],
                "k": summary["k"],
                "metric": summary["metric"],
                "value": summary["mean"],
                "sample_std": summary["sample_std"],
                "runs": summary["runs"],
            })
    print(f"Summary JSON: {json_path}")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
