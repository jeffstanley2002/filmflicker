from fastapi import APIRouter, Depends, HTTPException

from backend import data as data_module
from backend.auth import get_current_user_id
from backend.schemas import ModelMetricOut, SystemMetricsOut
from backend.utils import safe_int

router = APIRouter(prefix="/metrics", tags=["metrics"])

MODEL_LABELS = {
    "popularity": ("Crowd favorites", "Best for a brand-new account: trusted hits people consistently rate well."),
    "content_based": ("Taste match", "Uses genres and tags from movies you liked to find close neighbors."),
    "collaborative": ("People like you", "Learns patterns from similar rating histories and is strongest in current tests."),
    "clustering": ("Taste neighborhoods", "Groups movies into taste clusters and recommends from your strongest cluster."),
    "neural": ("Taste embeddings", "Learns latent movie factors from ratings to estimate user/movie fit."),
}

MODEL_ARTIFACTS = {
    "popularity": {"popularity.csv"},
    "content_based": {"content_tfidf_vectorizer.joblib", "content_tfidf_matrix.npz", "content_movie_ids.npy"},
    "collaborative": {"collaborative_svd.joblib"},
    "clustering": {"clustering.joblib"},
    "neural": {"neural_weights.npz"},
}


def _available_models(artifact_status: dict) -> list[str]:
    available_files = set(artifact_status["available"])
    return [
        model
        for model, required_files in MODEL_ARTIFACTS.items()
        if required_files.issubset(available_files)
    ]


def _metric_rows(raw: dict, model_names: list[str]) -> list[ModelMetricOut]:
    k = raw.get("k")
    rating_prediction = raw.get("rating_prediction", {})
    top_n = raw.get("top_n", {})
    rows = []
    for model in model_names:
        label, summary = MODEL_LABELS[model]
        prediction = rating_prediction.get(model, {})
        ranking = top_n.get(model, {})

        def ranking_metric(name: str):
            return ranking.get(f"{name}_at_{k}") if k else None

        measured_values = [
            ranking_metric("precision"),
            ranking_metric("recall"),
            ranking_metric("hit_rate"),
            ranking_metric("ndcg"),
            prediction.get("rmse"),
            prediction.get("mae"),
        ]
        rows.append(
            ModelMetricOut(
                model=model,
                plain_label=label,
                summary=summary,
                rmse=prediction.get("rmse"),
                mae=prediction.get("mae"),
                precision_at_k=ranking_metric("precision"),
                recall_at_k=ranking_metric("recall"),
                hit_rate_at_k=ranking_metric("hit_rate"),
                ndcg_at_k=ranking_metric("ndcg"),
                catalog_coverage=ranking.get("catalog_coverage"),
                intra_list_diversity=ranking.get("intra_list_diversity"),
                accuracy_index=None,
                health=(
                    "evaluated"
                    if any(isinstance(value, (int, float)) for value in measured_values)
                    else "artifact available"
                ),
            )
        )
    return rows


def _readiness(
    artifact_status: dict,
    raw: dict,
    rows: list[ModelMetricOut],
    catalog_freshness: str,
) -> tuple[bool, int, str]:
    all_models_available = len(rows) == len(MODEL_ARTIFACTS)
    has_metrics = bool(raw and rows and raw.get("n_eval_users"))
    catalog_current = "recommended" not in catalog_freshness.lower()
    bundle_ready = artifact_status["ready"] and all_models_available and has_metrics
    score = (45 if bundle_ready else 20) + (35 if has_metrics else 0) + (20 if catalog_current else 0)
    label = "Deployment-ready portfolio build" if bundle_ready else "Not deployment-ready"
    if bundle_ready and not catalog_current:
        label = "Portfolio deployment-ready; catalog refresh required for a public product"
    return bundle_ready, min(score, 100), label


@router.get("", response_model=dict)
def get_metrics(user_id: str = Depends(get_current_user_id)):
    m = data_module.metrics()
    if m is None:
        raise HTTPException(status_code=404, detail="No metrics.json found - run scripts/evaluate_models.py")
    return m


@router.get("/system", response_model=SystemMetricsOut)
def get_system_metrics():
    raw = data_module.metrics() or {}
    artifact_status = data_module.artifact_status(verify_checksums=False)
    model_names = _available_models(artifact_status)
    rows = _metric_rows(raw, model_names)

    movies = data_module.movies()
    manifest = data_module.catalog_manifest() or {}
    years = movies["year"].dropna()
    dataset_min_year = safe_int(years.min()) if not years.empty else None
    dataset_max_year = safe_int(years.max()) if not years.empty else None
    catalog_freshness = manifest.get("catalog_freshness") or "Catalog freshness has not been verified"
    bundle_ready, readiness_score, production_readiness = _readiness(
        artifact_status, raw, rows, catalog_freshness
    )

    return SystemMetricsOut(
        status="ready" if bundle_ready else "not_ready",
        production_readiness=production_readiness,
        readiness_score=readiness_score,
        generated_from=f"artifact generation {artifact_status['generation_id'] or 'unversioned'}",
        k=raw.get("k"),
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
