"""Run a reproducible, inspectable two-tower recommendation demo."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import random
import tempfile
import time
import webbrowser
import uuid
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import uvicorn

from src.models.train_two_tower import TrainingConfig, train_two_tower


SEED = 42
DEMO_PORT = 8766
TOP_K = 10
USER_COUNT = 96
GENRES = ["Science fiction", "Mystery", "Adventure", "Animation"]
TITLES = {
    "Science fiction": [
        "Orbit of the Blue Moon", "The Last Astronaut", "Signal from Europa",
        "Map of Forgotten Stars", "The Quiet Singularity", "Red Planet Radio",
        "A Ship Called Tomorrow", "The Glass Nebula", "After Earthrise", "Starfall Archive",
    ],
    "Mystery": [
        "The Quiet Detective", "Midnight on Harbor Street", "A Very Small Mystery",
        "The Clockmaker's Secret", "Room 17", "The Vanishing Postcard",
        "A Study in Rain", "The North Pier Case", "Murder at Bell House", "The Unsent Letter",
    ],
    "Adventure": [
        "Beyond the Cedar Pass", "The Sunken Compass", "River of Giants",
        "The Last Mapmaker", "Island at World's End", "The Amber Caravan",
        "Seven Days to the Summit", "The Hidden Waterfall", "Across the Salt Sea", "Wild Frontier",
    ],
    "Animation": [
        "Paper Dragon", "The Little Cloud Engine", "Foxes of Moonwood",
        "The Lantern Garden", "Milo and the Mountain", "A Whale in the Sky",
        "The Button Parade", "The Secret Kite", "Cloudberry Hollow", "The Painted Comet",
    ],
}


def make_dataset(seed: int = SEED) -> dict:
    """Create a fixed synthetic cohort with per-user train/validation/test splits."""
    item_ids: list[str] = []
    item_info: dict[str, dict[str, str]] = {}
    genre_items: dict[str, list[int]] = {}
    for genre_index, genre in enumerate(GENRES):
        genre_items[genre] = []
        for offset, title in enumerate(TITLES[genre]):
            item_index = len(item_ids)
            item_id = f"movie_{item_index:03d}"
            item_ids.append(item_id)
            genre_items[genre].append(item_index)
            item_info[item_id] = {"title": title, "genre": genre}

    user_ids = [10_000 + i for i in range(USER_COUNT)]
    user_genres: dict[int, str] = {}
    train_pairs: list[tuple[int, int]] = []
    validation_pairs: list[tuple[int, int]] = []
    test_pairs: list[tuple[int, int]] = []
    user_histories: dict[int, list[int]] = {}
    for local_user in range(USER_COUNT):
        user_id = user_ids[local_user]
        genre = GENRES[local_user % len(GENRES)]
        user_genres[user_id] = genre
        ordered_items = genre_items[genre].copy()
        random.Random(seed * 10_000 + user_id).shuffle(ordered_items)
        history = ordered_items[:6]
        validation_item, test_item = ordered_items[6:8]
        user_histories[local_user] = history
        train_pairs.extend((local_user, item) for item in history)
        validation_pairs.append((local_user, validation_item))
        test_pairs.append((local_user, test_item))

    return {
        "item_ids": item_ids,
        "item_info": item_info,
        "user_ids": user_ids,
        "user_genres": user_genres,
        "genre_items": genre_items,
        "train_pairs": train_pairs,
        "validation_pairs": validation_pairs,
        "test_pairs": test_pairs,
        "user_histories": user_histories,
    }


def rank_metrics(rankings: list[list[int]], targets: list[tuple[int, int]], k: int) -> dict:
    hits = 0
    ndcg = 0.0
    for ranking, (_, target) in zip(rankings, targets, strict=True):
        try:
            rank = ranking[:k].index(target) + 1
        except ValueError:
            continue
        hits += 1
        ndcg += 1.0 / math.log2(rank + 1)
    users = len(targets)
    return {
        f"recall@{k}": hits / users if users else 0.0,
        f"ndcg@{k}": ndcg / users if users else 0.0,
        "users": users,
        "catalog_coverage@k": len({i for row in rankings for i in row[:k]}),
    }


def evaluate_demo(model, data: dict) -> dict:
    item_count = len(data["item_ids"])
    train_seen = {user: set() for user in range(USER_COUNT)}
    item_popularity = Counter()
    for user, item in data["train_pairs"]:
        train_seen[user].add(item)
        item_popularity[item] += 1
    popularity_order = sorted(range(item_count), key=lambda i: (-item_popularity[i], i))

    model_rankings: list[list[int]] = []
    popularity_rankings: list[list[int]] = []
    with torch.inference_mode():
        all_items = torch.arange(item_count, dtype=torch.long)
        for user in range(USER_COUNT):
            user_tensor = torch.full((item_count,), user, dtype=torch.long)
            scores = model.score(user_tensor, all_items).cpu().numpy()
            candidates = [item for item in range(item_count) if item not in train_seen[user]]
            model_rankings.append(sorted(candidates, key=lambda i: (-float(scores[i]), i)))
            popularity_rankings.append([item for item in popularity_order if item not in train_seen[user]])

    metrics = {
        "two_tower": rank_metrics(model_rankings, data["test_pairs"], TOP_K),
        "popularity": rank_metrics(popularity_rankings, data["test_pairs"], TOP_K),
        "k": TOP_K,
        "train_interactions": len(data["train_pairs"]),
        "validation_targets": len(data["validation_pairs"]),
        "test_targets": len(data["test_pairs"]),
        "users": USER_COUNT,
        "items": item_count,
        "candidate_protocol": "full catalog minus each user's training interactions",
    }
    metrics["model_rankings"] = model_rankings
    metrics["popularity_rankings"] = popularity_rankings
    metrics["item_popularity"] = item_popularity
    return metrics


def build_serving_bundle(destination: Path, seed: int = SEED) -> tuple[dict, dict, dict]:
    run_started = time.perf_counter()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    data = make_dataset(seed)
    destination.mkdir(parents=True, exist_ok=True)
    training_output = io.StringIO()
    with contextlib.redirect_stdout(training_output):
        result = train_two_tower(
            data["train_pairs"],
            data["validation_pairs"],
            num_users=USER_COUNT,
            num_items=len(data["item_ids"]),
            model_user_ids=data["user_ids"],
            config=TrainingConfig(
                embedding_dim=16,
                batch_size=128,
                epochs=35,
                learning_rate=0.01,
                weight_decay=0.0001,
                seed=seed,
            ),
        )

    model = result.model.eval()
    metrics = evaluate_demo(model, data)
    with torch.inference_mode():
        vectors = model.item_embedding.weight.detach().cpu().numpy()
    checkpoint = {
        "format_version": 1,
        "model_config": {
            "num_users": model.num_users,
            "num_items": model.num_items,
            "embedding_dim": model.embedding_dim,
        },
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    torch.save(checkpoint, destination / "best_model.pt")
    np.save(destination / "item_vectors.npy", vectors)
    (destination / "item_ids.json").write_text(json.dumps(data["item_ids"]), encoding="utf-8")
    (destination / "user_ids.json").write_text(json.dumps(data["user_ids"]), encoding="utf-8")
    (destination / "seen_items.json").write_text(
        json.dumps({str(user): sorted(items) for user, items in {
            u: set(data["user_histories"][u]) for u in range(USER_COUNT)
        }.items()}),
        encoding="utf-8",
    )
    popularity_order = sorted(
        range(len(data["item_ids"])),
        key=lambda i: (-metrics["item_popularity"][i], i),
    )
    (destination / "popularity_item_indices.json").write_text(
        json.dumps(popularity_order), encoding="utf-8"
    )
    model_metadata = {
        "model_type": "two_tower_id_embeddings",
        "model_version": f"synthetic-seed-{seed}",
        "serving_policy": "two_tower",
        "seed": seed,
        "synthetic_demo": True,
        "evaluation": {key: value for key, value in metrics.items()
                       if key in {"two_tower", "popularity", "k", "train_interactions",
                                  "validation_targets", "test_targets", "users", "items"}},
    }
    (destination / "model_metadata.json").write_text(json.dumps(model_metadata), encoding="utf-8")
    names = [
        "best_model.pt", "item_vectors.npy", "item_ids.json", "user_ids.json",
        "seen_items.json", "model_metadata.json", "popularity_item_indices.json",
    ]
    manifest = {"files": {
        name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
        for name in names
    }}
    (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    interactions = {
        "train": sorted(data["train_pairs"]),
        "validation": sorted(data["validation_pairs"]),
        "test": sorted(data["test_pairs"]),
    }
    dataset_fingerprint = hashlib.sha256(
        json.dumps(interactions, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    run_info = {
        "run_id": uuid.uuid4().hex[:12],
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - run_started, 3),
        "seed": seed,
        "training_config": {
            "embedding_dim": 16,
            "batch_size": 128,
            "epochs": 35,
            "learning_rate": 0.01,
            "weight_decay": 0.0001,
            "loss": "Bayesian Personalized Ranking (BPR)",
            "checkpoint_selection": "lowest sampled validation BPR loss",
        },
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "dataset_fingerprint_definition": (
            "SHA-256 of compact JSON containing sorted (local_user_idx, item_idx) "
            "pairs for train, validation, and test splits"
        ),
        "checkpoint_sha256": hashlib.sha256(
            (destination / "best_model.pt").read_bytes()
        ).hexdigest(),
        "training_log": training_output.getvalue().strip().splitlines(),
        "best_epoch": result.best_epoch,
        "best_validation_bpr_loss": result.best_validation_loss,
    }
    report = {**run_info, "metrics": {
        key: value for key, value in metrics.items()
        if key in {"two_tower", "popularity", "k", "train_interactions",
                   "validation_targets", "test_targets", "users", "items",
                   "candidate_protocol"}
    }}
    (destination / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return data, metrics, run_info


def create_demo_app(model_dir: Path, data: dict, metrics: dict, run_info: dict, run_root: Path):
    from fastapi import HTTPException, Query
    from fastapi.responses import FileResponse, HTMLResponse, Response
    from src.api.app import create_app, load_runtime

    app = create_app(model_dir)
    project_report_path = (
        Path(__file__).resolve().parent
        / "artifacts" / "model_selection" / "two_tower_seed42" / "test_metrics.json"
    )
    project_model_dir = project_report_path.parent
    project_history_path = project_model_dir / "history.json"
    project_checkpoint_path = project_model_dir / "best_model.pt"
    project_report = (
        json.loads(project_report_path.read_text(encoding="utf-8"))
        if project_report_path.is_file()
        else None
    )
    if project_report is not None and project_checkpoint_path.is_file():
        project_report["checkpoint_sha256"] = hashlib.sha256(
            project_checkpoint_path.read_bytes()
        ).hexdigest()
    sample_users = [data["user_ids"][genre_index] for genre_index in range(len(GENRES))]
    id_to_local = {user_id: local for local, user_id in enumerate(data["user_ids"])}

    def public_metrics() -> dict:
        return {key: value for key, value in metrics.items()
                if key not in {"model_rankings", "popularity_rankings", "item_popularity"}}

    def public_run_report() -> dict:
        return {
            **run_info,
            "metrics": public_metrics(),
        }

    @app.get("/", response_class=HTMLResponse)
    def home():
        return PAGE

    @app.get("/demo-info")
    def info():
        return {
            "users": [{"id": user_id, "cohort": data["user_genres"][user_id]}
                      for user_id in sample_users],
            "items": data["item_info"],
            "metrics": public_metrics(),
            "split": {"train": 6, "validation": 1, "test": 1},
            "seed": run_info["seed"],
            "run_id": run_info["run_id"],
        }

    @app.get("/demo-run-report")
    def run_report():
        return Response(
            json.dumps(public_run_report(), indent=2),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=demo-run-report.json"},
        )

    @app.get("/project-evaluation.json")
    def project_evaluation():
        if project_report is None:
            raise HTTPException(status_code=404, detail="saved real-data test report is not present")
        return Response(
            json.dumps(project_report, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=project-test-evaluation.json"},
        )

    @app.get("/project-training-history.json")
    def project_training_history():
        if not project_history_path.is_file():
            raise HTTPException(status_code=404, detail="saved real-data training history is not present")
        return json.loads(project_history_path.read_text(encoding="utf-8"))

    @app.get("/project-checkpoint.pt")
    def project_checkpoint_download():
        if not project_checkpoint_path.is_file():
            raise HTTPException(status_code=404, detail="saved real-data checkpoint is not present")
        return FileResponse(project_checkpoint_path, filename="real-data-best-model.pt")

    @app.get("/demo-checkpoint.pt")
    def checkpoint_download():
        return FileResponse(model_dir / "best_model.pt", filename="best_model.pt")

    @app.get("/demo-manifest.json")
    def manifest_download():
        return FileResponse(model_dir / "manifest.json", filename="manifest.json")

    @app.get("/demo-dataset.csv")
    def dataset_csv():
        rows = ["user_idx,user_id,item_idx,item_id,split"]
        for split, pairs in (("train", data["train_pairs"]),
                             ("validation", data["validation_pairs"]),
                             ("test", data["test_pairs"])):
            for user, item in sorted(pairs):
                rows.append(
                    f"{user},{data['user_ids'][user]},{item},"
                    f"{data['item_ids'][item]},{split}"
                )
        return Response("\n".join(rows) + "\n", media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=demo-interactions.csv"})

    @app.post("/demo-run")
    def rerun(seed: int = Query(default=SEED, ge=0, le=999_999)):
        nonlocal data, metrics, run_info, model_dir, id_to_local
        candidate_dir = run_root / f"run-{uuid.uuid4().hex[:12]}"
        candidate_data, candidate_metrics, candidate_run_info = build_serving_bundle(
            candidate_dir, seed
        )
        # Validate the checksummed bundle before swapping the live API to this run.
        candidate_runtime = load_runtime(candidate_dir)
        app.state.runtime = candidate_runtime
        data, metrics, run_info, model_dir = (
            candidate_data, candidate_metrics, candidate_run_info, candidate_dir
        )
        id_to_local = {user_id: local for local, user_id in enumerate(data["user_ids"])}
        return public_run_report()

    @app.get("/demo-context/{user_id}")
    def context(user_id: int):
        local = id_to_local.get(user_id)
        if local is None:
            raise HTTPException(status_code=404, detail="sample user not found")
        history = data["user_histories"][local]
        model = app.state.runtime["model"]
        with torch.inference_mode():
            item_indices = torch.arange(len(data["item_ids"]), dtype=torch.long)
            user_indices = torch.full_like(item_indices, local)
            model_scores = model.score(user_indices, item_indices).cpu().tolist()
        evidence = {}
        for item_id, item_index in zip(data["item_ids"], range(len(data["item_ids"])), strict=True):
            same_cohort = [
                other for other, other_genre in data["user_genres"].items()
                if other != user_id and other_genre == data["user_genres"][user_id]
            ]
            support = sum(
                item_index in data["user_histories"][id_to_local[other]]
                for other in same_cohort
            )
            evidence[item_id] = {
                "training_interactions": int(metrics["item_popularity"][item_index]),
                "same_cohort_support": support,
                "model_score": float(model_scores[item_index]),
            }
        return {
            "user_id": user_id,
            "cohort": data["user_genres"][user_id],
            "history": [data["item_ids"][i] for i in history],
            "held_out_test_item": data["item_ids"][data["test_pairs"][local][1]],
            "model_test_rank": metrics["model_rankings"][local].index(
                data["test_pairs"][local][1]
            ) + 1,
            "popularity_test_rank": metrics["popularity_rankings"][local].index(
                data["test_pairs"][local][1]
            ) + 1,
            "evidence": evidence,
        }

    return app


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="recommender-ml-demo-") as temp_dir:
        run_root = Path(temp_dir) / "runs"
        model_dir = run_root / "initial"
        data, metrics, run_info = build_serving_bundle(model_dir)
        app = create_demo_app(model_dir, data, metrics, run_info, run_root)
        address = f"http://127.0.0.1:{DEMO_PORT}"
        print(f"ML recommender demo: {address}")
        print(
            f"Run {run_info['run_id']} · seed {SEED} · {metrics['users']} users · {metrics['items']} items · "
            f"two-tower Recall@{TOP_K}={metrics['two_tower'][f'recall@{TOP_K}']:.3f} · "
            f"popularity Recall@{TOP_K}={metrics['popularity'][f'recall@{TOP_K}']:.3f}"
        )
        print("Synthetic offline experiment; press Ctrl+C to stop.")
        try:
            webbrowser.open(address)
        except webbrowser.Error:
            pass
        uvicorn.run(app, host="127.0.0.1", port=DEMO_PORT, log_level="warning")


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Movie recommendation project demo</title><style>
:root{color-scheme:dark;--bg:#0a1020;--panel:#121b2d;--panel2:#18243a;--line:#2b3a55;--text:#edf3ff;--muted:#a3b1ca;--accent:#80e6c1;--blue:#8bbcff;--gold:#ffd37c}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#111d35 0,var(--bg) 42%);color:var(--text);font:15px/1.55 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:42px 24px 60px}.eyebrow{color:var(--accent);text-transform:uppercase;letter-spacing:.16em;font-size:11px;font-weight:750}.top{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}h1{font-size:clamp(36px,6vw,58px);line-height:1.03;margin:10px 0 14px;letter-spacing:-.04em}.lede{color:var(--muted);max-width:720px;margin:0}.tag{border:1px solid #735d32;color:var(--gold);border-radius:99px;padding:7px 11px;font-size:12px;white-space:nowrap}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:22px}.panel{background:#111a2aee;border:1px solid var(--line);border-radius:15px;padding:19px}.wide{grid-column:1/-1}.panel h2{font-size:16px;margin:0 0 12px}.muted{color:var(--muted)}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}.stat{background:var(--panel2);border:1px solid var(--line);border-radius:11px;padding:12px}.stat strong{font-size:22px;display:block}.stat span{color:var(--muted);font-size:11px}.protocol{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.step{padding:12px;background:var(--panel2);border-radius:10px;border:1px solid var(--line)}.step b{color:var(--accent)}.table{width:100%;border-collapse:collapse}.table th,.table td{text-align:left;border-bottom:1px solid var(--line);padding:10px 8px}.table th{font-size:11px;color:var(--muted);font-weight:600}.model{color:var(--accent)}.base{color:var(--blue)}.cohort{display:flex;gap:8px;flex-wrap:wrap}.user{border:1px solid var(--line);background:var(--panel2);color:var(--text);border-radius:9px;padding:9px 11px;cursor:pointer}.user.active{border-color:var(--accent)}.body{display:grid;grid-template-columns:250px 1fr;gap:12px;margin-top:12px}.history,.recs{background:var(--panel2);border-radius:11px;padding:14px}.history ul{padding-left:19px;margin-bottom:0}.history li{margin:5px 0}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}.movie{border:1px solid var(--line);border-radius:10px;padding:12px}.movie b{display:block}.movie small{color:var(--muted)}.score{font-variant-numeric:tabular-nums;color:var(--accent)}.note{font-size:12px;color:var(--muted);margin-top:11px}.how{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.how article{background:var(--panel2);padding:14px;border-radius:10px}.how b{display:block;margin-bottom:5px}.warning{border-left:3px solid var(--gold);padding:9px 12px;background:#211d17;color:#e7d5b2;border-radius:5px;font-size:12px;margin-top:12px}button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.tabs{display:flex;gap:8px;margin-top:22px;border-bottom:1px solid var(--line)}.tab{padding:11px 15px;background:transparent;color:var(--muted);border:0;border-bottom:2px solid transparent;cursor:pointer}.tab.active{color:var(--accent);border-color:var(--accent)}.tab-panel-hidden{display:none!important}.run-controls{display:flex;gap:9px;align-items:center;flex-wrap:wrap}.run-controls input{width:100px;padding:9px;border:1px solid var(--line);border-radius:8px;background:var(--panel2);color:var(--text)}.run-controls button{padding:10px 14px;border:0;border-radius:9px;background:var(--accent);color:#07140f;font-weight:750;cursor:pointer}.run-meta{font:12px/1.6 ui-monospace,Consolas,monospace;overflow-wrap:anywhere}pre{max-height:260px;overflow:auto;white-space:pre-wrap;background:#080d17;padding:12px;border-radius:9px;color:#c9d8f1;font-size:12px}details summary{cursor:pointer;color:var(--accent)}.run-links{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0}.run-links a{color:var(--blue)}.chart{width:100%;height:auto;background:#080d17;border-radius:10px}.bar-compare{margin:18px 0;padding:14px;background:var(--panel2);border-radius:11px}.bar-row{display:grid;grid-template-columns:145px 1fr 160px;gap:12px;align-items:center;margin:10px 0}.bar-track{height:22px;background:#0a1020;border-radius:99px;overflow:hidden}.bar{height:100%;border-radius:99px;min-width:2px}.model-bar{background:var(--accent)}.base-bar{background:var(--blue)}.bar-row b{font-size:12px;font-variant-numeric:tabular-nums}@media(max-width:600px){.bar-row{grid-template-columns:1fr 100px}.bar-track{grid-column:1/2;grid-row:2}.bar-row b{grid-column:2;grid-row:2}}
@media(max-width:760px){main{padding:26px 14px}.grid{grid-template-columns:1fr}.wide{grid-column:auto}.stats{grid-template-columns:repeat(2,1fr)}.body{grid-template-columns:1fr}.cards{grid-template-columns:1fr}.how{grid-template-columns:1fr}.top{display:block}.tag{display:inline-block;margin-top:12px}}
</style></head><body><main>
<div class="top"><div><div class="eyebrow">End-to-end machine learning project</div><h1>Movie recommendations,<br>from data to API</h1><p class="lede">I built a service that learns from viewing histories, checks its rankings against interactions held out from training, compares with a simple popularity approach, and serves recommendations through an API.</p></div><span class="tag">PROJECT RESULTS + LIVE DEMO</span></div>
<nav class="tabs" role="tablist" aria-label="Demo sections"><button class="tab active" role="tab" aria-selected="true" data-tab="overview">Project overview</button><button class="tab" role="tab" aria-selected="false" data-tab="user">Example user</button><button class="tab" role="tab" aria-selected="false" data-tab="technical">Technical details</button></nav>
<div class="grid"><section class="panel wide" data-tab-panel="overview"><h2>What I built</h2><p class="muted">A complete recommendation workflow: prepare interaction data, train and select a model, measure it against a baseline on held-out examples, then serve ranked movies behind a health-checked API.</p><div class="how"><article><b>1 · Learn</b><span class="muted">The model learns which users and movies tend to go together from past interactions.</span></article><article><b>2 · Check</b><span class="muted">Test interactions are kept aside, so the reported score reflects examples the model did not train on.</span></article><article><b>3 · Serve</b><span class="muted">The selected model powers a recommendation API; run evidence and downloadable artifacts are available in Technical details.</span></article></div></section>
<section class="panel wide" data-tab-panel="overview"><h2>What the measured result says</h2><p class="muted" id="plain-result">Loading the saved project evaluation…</p><div class="bar-compare" aria-label="Comparison of held-out movies found in the top ten"><div class="bar-row"><span>Learned model</span><div class="bar-track"><div class="bar model-bar" id="model-bar"></div></div><b id="model-bar-label">—</b></div><div class="bar-row"><span>Popularity</span><div class="bar-track"><div class="bar base-bar" id="base-bar"></div></div><b id="base-bar-label">—</b></div><p class="note">Share of held-out target movies found in each approach's top 10; bars use a common scale.</p></div><table class="table"><thead><tr><th>Approach</th><th>Users tested</th><th>Movies considered</th><th>Found in top 10</th></tr></thead><tbody id="project-results"><tr><td colspan="4">Loading saved report…</td></tr></tbody></table><p class="note" id="project-run-meta"></p><details><summary>Show exact ranking metrics and training curve</summary><p class="note">Recall@10 is the share of held-out movies appearing in the top ten. NDCG@10 gives more credit when a held-out movie appears nearer the top.</p><table class="table"><thead><tr><th>Approach</th><th>Recall@10</th><th>NDCG@10</th></tr></thead><tbody id="project-metric-details"></tbody></table><canvas id="project-loss-chart" width="900" height="220" class="chart" aria-label="Saved real-data model training and validation loss by epoch"></canvas></details><p class="warning">The model is a little better than popularity in this offline test, but the gain is small. Overall recall is low, tail-item performance is weak, and no online business impact has been measured. This result is useful as an honest baseline for further work, not a claim of production success.</p><div class="run-links"><a href="/project-evaluation.json">Test report (JSON)</a><a href="/project-training-history.json">Training history (JSON)</a><a href="/project-checkpoint.pt">Real-data checkpoint</a></div><h3>What I’d improve next</h3><p class="muted">Investigate weak performance on less popular movies, compare against a stronger tuned baseline, and evaluate with a time-based split that better reflects how recommendations are used.</p></section>
<section class="panel wide" data-tab-panel="technical"><h2>Training and comparison setup</h2><div class="how"><article><b>Two-tower model</b><span class="muted">16-dimensional user and item ID embeddings trained with AdamW and pairwise BPR loss. Candidate score is the embedding dot product.</span></article><article><b>Popularity baseline</b><span class="muted">Ranks movies by training interaction count; ties break by item ID. Validation and test labels are not used.</span></article><article><b>Checkpoint selection</b><span class="muted">The project trainer restores the epoch with the lowest sampled validation BPR loss.</span></article></div><p class="note">The saved project result uses real project data and is distinct from the small synthetic workflow below.</p></section>
<section class="panel" data-tab-panel="technical"><h2>Live synthetic workflow check</h2><table class="table"><thead><tr><th>Ranker</th><th>Recall@K</th><th>NDCG@K</th><th>Catalog coverage</th></tr></thead><tbody id="results"></tbody></table><p class="note">This small generated dataset exists to make a fresh training run inspectable on a laptop. It is not the source of the saved project results.</p></section>
<section class="panel wide" data-tab-panel="technical"><div class="row"><div><h2>Run evidence · synthetic retraining</h2><p class="note">This is the live retraining path: 96 generated users, 576 training interactions, and 40 movies. It runs quickly at laptop scale; elapsed time is measured, not simulated. The saved real-data result on the overview is a recorded evaluation and is not retrained by this button.</p></div><div class="run-controls"><label for="seed">Seed</label><input id="seed" type="number" min="0" max="999999" value="42"><button id="run-button" type="button">Run experiment</button></div></div><pre id="run-meta"></pre><canvas id="loss-chart" width="900" height="220" class="chart" aria-label="Training and validation BPR loss by epoch"></canvas><details><summary>Download run artifacts</summary><div class="run-links"><a href="/demo-dataset.csv">Interaction splits (CSV)</a><a href="/demo-run-report">Run report (JSON)</a><a href="/demo-checkpoint.pt">Trained checkpoint</a><a href="/demo-manifest.json">Artifact manifest</a><a href="/project-evaluation.json">Saved project report (JSON)</a><a href="/project-checkpoint.pt">Saved project checkpoint</a></div></details><details><summary>Show captured epoch training log</summary><pre id="training-log"></pre></details><details><summary>Why the synthetic score is high</summary><p class="note">The generator creates four balanced genres, with ten titles each. Users interact mostly within one generated genre, creating a clear collaborative signal. High scores show the pipeline can learn that controlled pattern; they do not demonstrate recommendation quality on real audiences.</p><p class="note">This is laptop-scale by design. The run panel reports measured time, not a throughput benchmark.</p></details></section>
<section class="panel wide" data-tab-panel="user"><h2>Inspect a held-out user</h2><p class="note">This trace uses generated example data so you can inspect individual training history, recommendations, and held-out ranking behavior.</p><div class="cohort" id="users"></div><div id="inspection" class="body"><p class="muted">Select a profile to inspect its training history and ranked candidates.</p></div><div class="warning">The model is collaborative and ID-only. “Same-cohort support” is descriptive evidence from the synthetic data generator, not a causal feature attribution. Real-world recommendation quality requires real data and a properly governed evaluation.</div></section>
</div></main><script>
const titleById={};let info;
async function init(){info=await(await fetch('/demo-info')).json();for(const [id,item] of Object.entries(info.items))titleById[id]=item.title;
const m=info.metrics,k=info.metrics.k;
document.getElementById('results').innerHTML=[['Two-tower',m.two_tower,'model'],['Popularity',m.popularity,'base']].map(([name,v,c])=>`<tr><td class="${c}">${name}</td><td>${(100*v[`recall@${k}`]).toFixed(1)}%</td><td>${v[`ndcg@${k}`].toFixed(3)}</td><td>${v[`catalog_coverage@${k}`]} / ${m.items}</td></tr>`).join('');
document.getElementById('users').innerHTML=info.users.map(u=>`<button class="user" data-id="${u.id}">User ${u.id} · ${u.cohort}</button>`).join('');document.querySelectorAll('.user').forEach(b=>b.onclick=()=>inspect(b.dataset.id,b));}
async function inspect(id,button){document.querySelectorAll('.user').forEach(b=>b.classList.toggle('active',b===button));const [ctx,res]=await Promise.all([fetch(`/demo-context/${id}`).then(r=>r.json()),fetch(`/recommendations/${id}?limit=6`).then(r=>r.json())]);const history=ctx.history.map(item=>`<li>${titleById[item]} <small>· ${info.items[item].genre}</small></li>`).join('');const recs=res.items.map((item,i)=>{const e=ctx.evidence[item];return `<article class="movie"><b>${i+1}. ${titleById[item]}</b><small>${info.items[item].genre} · score ${e.model_score.toFixed(4)}</small><div class="note">${e.training_interactions} training interactions; ${e.same_cohort_support} of 23 synthetic peer profiles interacted. The score comes from learned user and movie ID embeddings; peer counts are descriptive, not causal explanations.</div></article>`}).join('');document.getElementById('inspection').innerHTML=`<div class="history"><b>User ${id} · ${ctx.cohort}</b><p class="note">Training history (six interactions)</p><ul>${history}</ul></div><div class="recs"><b>Top recommendations from the live API</b><div class="cards" style="margin-top:10px">${recs}</div><div class="note" style="margin-top:14px"><b>Held-out check:</b> ${titleById[ctx.held_out_test_item]} was not used for training or model selection. Its rank was #${ctx.model_test_rank} for the model and #${ctx.popularity_test_rank} for popularity. The model learned interaction patterns from training histories; this generated profile is illustrative, not real user evidence.</div></div>`;}
init();
async function loadProjectReport(){const response=await fetch('/project-evaluation.json');const body=document.getElementById('project-results');if(!response.ok){body.innerHTML='<tr><td colspan="4">Saved real-data report is unavailable in this checkout.</td></tr>';document.getElementById('plain-result').textContent='The saved real-data evaluation is not available in this checkout; open Technical details for the separate synthetic live run.';return;}const r=await response.json(),tower=r.metrics_at_k.two_tower['10'],pop=r.metrics_at_k.popularity_baseline_on_same_user_sample['10'];const percent=m=>(100*m.recall).toFixed(2),found=m=>Math.round(m.recall*r.users_evaluated).toLocaleString();body.innerHTML=[['Learned recommendation model',tower],['Popularity only',pop]].map(([name,m])=>`<tr><td>${name}</td><td>${r.users_evaluated.toLocaleString()}</td><td>${r.candidate_items.toLocaleString()}</td><td>${found(m)} (${percent(m)}%)</td></tr>`).join('');document.getElementById('project-metric-details').innerHTML=[['Learned recommendation model',tower],['Popularity only',pop]].map(([name,m])=>`<tr><td>${name}</td><td>${percent(m)}%</td><td>${m.ndcg.toFixed(5)}</td></tr>`).join('');const delta=((tower.recall-pop.recall)*100).toFixed(2);document.getElementById('plain-result').textContent=`On ${r.users_evaluated.toLocaleString()} held-out user examples, the learned model put the target movie in its first ten recommendations ${found(tower)} times, compared with ${found(pop)} for popularity alone. That's about ${delta} percentage points higher in this offline test.`;for(const [element,m] of [['model-bar',tower],['base-bar',pop]])document.getElementById(element).style.width=`${Math.max(2,100*m.recall/tower.recall)}%`;document.getElementById('model-bar-label').textContent=`${percent(tower)}% · ${found(tower)}`;document.getElementById('base-bar-label').textContent=`${percent(pop)}% · ${found(pop)}`;}
loadProjectReport();
async function loadProjectHistory(){const response=await fetch('/project-training-history.json'),meta=document.getElementById('project-run-meta');if(!response.ok){meta.textContent='Saved real-data training history is unavailable in this checkout.';return;}const h=await response.json(),evaluation=await(await fetch('/project-evaluation.json')).json();meta.textContent=`Saved real-data run: ${h.model_users.toLocaleString()} users · ${h.train_pairs.toLocaleString()} training pairs · ${h.catalog_items.toLocaleString()} items · best epoch ${h.best_epoch} · validation BPR ${h.best_validation_loss.toFixed(5)} · test users ${evaluation.users_evaluated.toLocaleString()} · candidates ${evaluation.candidate_items.toLocaleString()} · checkpoint SHA-256 ${evaluation.checkpoint_sha256||'not available'}`;drawLossChart(h.epochs.map(e=>`epoch=${e.epoch} train_loss=${e.train_loss} validation_loss=${e.validation_loss}`),'project-loss-chart');}
loadProjectHistory();
function activateTab(name){document.querySelectorAll('[data-tab-panel]').forEach(panel=>panel.classList.toggle('tab-panel-hidden',panel.dataset.tabPanel!==name));document.querySelectorAll('.tab').forEach(tab=>{const active=tab.dataset.tab===name;tab.classList.toggle('active',active);tab.setAttribute('aria-selected',String(active));});}
document.querySelectorAll('.tab').forEach(tab=>tab.onclick=()=>activateTab(tab.dataset.tab));activateTab('overview');
function drawLossChart(lines,canvasId='loss-chart'){const canvas=document.getElementById(canvasId),ctx=canvas.getContext('2d'),rows=lines.map(line=>{const m=line.match(/epoch=(\d+) train_loss=([\d.]+) validation_loss=([\d.]+)/);return m?{epoch:+m[1],train:+m[2],validation:+m[3]}:null;}).filter(Boolean);ctx.clearRect(0,0,canvas.width,canvas.height);if(!rows.length)return;const pad={l:48,r:18,t:25,b:32},values=rows.flatMap(r=>[r.train,r.validation]),low=Math.min(...values),high=Math.max(...values),span=Math.max(high-low,0.001),x=i=>pad.l+i*(canvas.width-pad.l-pad.r)/Math.max(rows.length-1,1),y=v=>pad.t+(high-v)*(canvas.height-pad.t-pad.b)/span;ctx.strokeStyle='#34435d';ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,canvas.height-pad.b);ctx.lineTo(canvas.width-pad.r,canvas.height-pad.b);ctx.stroke();for(const [key,color,label] of [['train','#8bbcff','train BPR'],['validation','#80e6c1','validation BPR']]){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();rows.forEach((r,i)=>i?ctx.lineTo(x(i),y(r[key])):ctx.moveTo(x(i),y(r[key])));ctx.stroke();ctx.fillStyle=color;ctx.fillRect(pad.l+(key==='train'?0:110),8,10,10);ctx.fillStyle='#cbd6e8';ctx.font='12px system-ui';ctx.fillText(label,pad.l+(key==='train'?15:125),17);}ctx.fillStyle='#a3b1ca';ctx.font='11px system-ui';ctx.fillText('epoch 1',pad.l,canvas.height-9);ctx.fillText(`epoch ${rows[rows.length-1].epoch}`,canvas.width-pad.r-52,canvas.height-9);ctx.fillText('lower is better',canvas.width-105,17);}
async function loadRunEvidence(){const response=await fetch('/demo-run-report');const run=await response.json();document.getElementById('seed').value=run.seed;document.getElementById('run-meta').textContent=`Run ID: ${run.run_id}\nElapsed: ${run.elapsed_seconds}s · completed UTC: ${run.completed_at_utc}\nSeed: ${run.seed} · best epoch: ${run.best_epoch} · best validation BPR: ${run.best_validation_bpr_loss.toFixed(5)}\nDataset SHA-256: ${run.dataset_fingerprint_sha256}\nCheckpoint SHA-256: ${run.checkpoint_sha256}`;document.getElementById('training-log').textContent=run.training_log.join('\n');drawLossChart(run.training_log);}
loadRunEvidence();
document.getElementById('run-button').onclick=async()=>{const button=document.getElementById('run-button');button.disabled=true;button.textContent='Training…';document.getElementById('run-meta').textContent='Training and evaluating on the server. The active API model will switch after the new bundle validates.';try{const seed=document.getElementById('seed').value;const response=await fetch(`/demo-run?seed=${encodeURIComponent(seed)}`,{method:'POST'});if(!response.ok)throw Error(`Run failed (${response.status}): ${await response.text()}`);await init();await loadRunEvidence();await loadProjectReport();document.querySelector('details').open=true;}catch(error){document.getElementById('run-meta').textContent=error.message;}finally{button.disabled=false;button.textContent='Run experiment';}};
</script></body></html>'''


if __name__ == "__main__":
    main()
