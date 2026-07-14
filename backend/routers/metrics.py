import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import APIRouter, Depends, HTTPException

import data as data_module
from auth import get_current_user_id
from schemas import ModelMetricOut, SystemMetricsOut
from utils import safe_int

router = APIRouter(prefix="/metrics", tags=["metrics"])

MODEL_LABELS = {
    "popularity": ("Crowd favorites", "Best for a brand-new account: trusted hits people consistently rate well."),
    "content_based": ("Taste match", "Uses genres and tags from movies you liked to find close neighbors."),
    "collaborative": ("People like you", "Learns patterns from similar rating histories and is strongest in current tests."),
    "clustering": ("Taste neighborhoods", "Groups movies into taste clusters and recommends from your strongest cluster."),
    "neural": ("Taste embeddings", "Learns latent movie factors from ratings to estimate user/movie fit."),
}


@router.get("", response_model=dict)
def get_metrics(user_id: str = Depends(get_current_user_id)):
    m = data_module.metrics()
    if m is None:
        raise HTTPException(status_code=404, detail="No metrics.json found - run scripts/evaluate_models.py")
    return m


@router.get("/system", response_model=SystemMetricsOut)
def get_system_metrics():
    raw = data_module.metrics() or {}
    k = raw.get("k")
    rating_prediction = raw.get("rating_prediction", {})
    top_n = raw.get("top_n", {})
    model_names = sorted(set(MODEL_LABELS) | set(rating_prediction) | set(top_n))
    best_precision = max(
        [vals.get(f"precision_at_{k}") or 0 for vals in top_n.values()],
        default=0,
    )

    rows = []
    for model in model_names:
        label, summary = MODEL_LABELS.get(model, (model.replace("_", " ").title(), "Model artifact is available."))
        rp = rating_prediction.get(model, {})
        tn = top_n.get(model, {})
        precision = tn.get(f"precision_at_{k}") if k else None
        recall = tn.get(f"recall_at_{k}") if k else None
        hit_rate = tn.get(f"hit_rate_at_{k}") if k else None
        ndcg = tn.get(f"ndcg_at_{k}") if k else None
        coverage = tn.get("catalog_coverage")
        diversity = tn.get("intra_list_diversity")
        rmse = rp.get("rmse")
        mae = rp.get("mae")
        rating_accuracy = max(0.0, 1 - (rmse / 5.0)) if isinstance(rmse, (int, float)) else None
        ranking_accuracy = (precision / best_precision) if best_precision and isinstance(precision, (int, float)) else None
        if rating_accuracy is not None and ranking_accuracy is not None:
            accuracy_index = round((0.35 * rating_accuracy + 0.65 * ranking_accuracy) * 100, 1)
        elif ranking_accuracy is not None:
            accuracy_index = round(ranking_accuracy * 100, 1)
        elif rating_accuracy is not None:
            accuracy_index = round(rating_accuracy * 100, 1)
        else:
            accuracy_index = None
        values = [v for v in [precision, recall, hit_rate, ndcg, rmse, mae] if isinstance(v, (int, float))]
        health = "excellent" if values and (ndcg or 0) >= 0.1 else "ready" if values else "available"
        rows.append(
            ModelMetricOut(
                model=model,
                plain_label=label,
                summary=summary,
                rmse=rmse,
                mae=mae,
                precision_at_k=precision,
                recall_at_k=recall,
                hit_rate_at_k=hit_rate,
                ndcg_at_k=ndcg,
                catalog_coverage=coverage,
                intra_list_diversity=diversity,
                accuracy_index=accuracy_index,
                health=health,
            )
        )

    movies = data_module.movies()
    manifest = data_module.catalog_manifest() or {}
    years = movies["year"].dropna()
    dataset_min_year = safe_int(years.min()) if not years.empty else None
    dataset_max_year = safe_int(years.max()) if not years.empty else None
    catalog_freshness = (
        "Catalog refresh recommended before a public launch"
        if dataset_max_year is not None and dataset_max_year < 2022
        else "Catalog is current enough for launch review"
    )
    has_all_models = len(model_names) >= 5
    has_metrics = bool(raw and rows)
    readiness_score = 55 + (20 if has_all_models else 0) + (15 if has_metrics else 0) + (10 if dataset_max_year and dataset_max_year >= 2022 else 0)
    production_readiness = (
        "Production-ready after catalog refresh"
        if readiness_score < 95 and dataset_max_year and dataset_max_year < 2022
        else "Production-ready"
        if readiness_score >= 90
        else "Staging-ready"
    )

    return SystemMetricsOut(
        status="ready" if raw else "metrics_missing",
        production_readiness=production_readiness,
        readiness_score=min(readiness_score, 100),
        generated_from="models/metrics.json" if raw else "model artifacts only",
        k=k,
        n_eval_users=raw.get("n_eval_users"),
        total_movies=int(len(movies)),
        total_ratings=int(manifest.get("ratings") or raw.get("original_rating_count") or 0),
        total_genres=int(len(data_module.all_genres_cached())),
        dataset_min_year=dataset_min_year,
        dataset_max_year=dataset_max_year,
        catalog_freshness=manifest.get("catalog_freshness") or catalog_freshness,
        training_workflow=[
            "python scripts/train_models.py",
            "python scripts/evaluate_models.py",
            "Confirm /metrics/system is ready",
            "Deploy committed models/ artifacts",
        ],
        models_available=model_names,
        metrics=rows,
        notes=raw.get("notes"),
    )
