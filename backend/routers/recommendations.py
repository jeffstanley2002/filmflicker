from fastapi import APIRouter, Depends, Query

from backend import data as data_module
from backend import db
from backend.auth import get_current_user_id
from backend.movie_mapper import recommendation_list
from backend.rate_limit import rate_limit
from backend.schemas import RecommendationOut

from src.recommenders import content_based, ensemble

router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"],
    dependencies=[Depends(rate_limit("recommendations", 30, 60))],
)


@router.get("", response_model=list[RecommendationOut])
def get_recommendations(
    model: str = Query(..., pattern="^(popularity|content_based|collaborative|clustering|neural)$"),
    n: int = Query(12, ge=1, le=50),
    user_id: str = Depends(get_current_user_id),
):
    """Returns model-specific recommendations with cold-start fallbacks."""
    engine = db.get_connection()
    watched_ids, rated = db.get_profile(engine, user_id)
    disliked_ids = db.get_not_interested_ids(engine, user_id)
    watchlist_ids = db.get_watchlist_ids(engine, user_id)
    low_signal = data_module.low_signal()
    pop_df = data_module.popularity_table()
    movies_df = data_module.movies_indexed()
    links = data_module.links_indexed()

    artifacts = {"content_based": data_module.content_based_artifacts()}
    if model == "collaborative":
        artifacts["collaborative"] = data_module.collaborative_artifacts()
    elif model == "clustering":
        artifacts["clustering"] = data_module.clustering_artifacts()
    elif model == "neural":
        artifacts["neural"] = data_module.neural_model()
    recs = ensemble.recommend(
        model=model,
        n=n,
        rated=rated,
        watched_ids=watched_ids,
        disliked_ids=disliked_ids | watchlist_ids,
        movies_df=movies_df,
        pop_df=pop_df,
        low_signal=low_signal,
        artifacts=artifacts,
    )

    return recommendation_list(recs, movies_df, links)


@router.get("/similar/{movie_id}", response_model=list[RecommendationOut])
def similar_movies(movie_id: int, n: int = Query(8, ge=1, le=50), user_id: str = Depends(get_current_user_id)):
    engine = db.get_connection()
    watched_ids = db.get_watched_ids(engine, user_id)
    exclude = watched_ids | db.get_not_interested_ids(engine, user_id) | data_module.low_signal()
    artifacts = data_module.content_based_artifacts()
    recs = content_based.similar_to_movie(movie_id, artifacts, n=max(n * 6, n), exclude_ids=exclude)

    movies_df = data_module.movies_indexed()
    links = data_module.links_indexed()
    from src.recommenders import ranking
    recs = ranking.rerank_candidates(recs, movies_df, data_module.popularity_table(), n=n, primary_model="content_based")
    return recommendation_list(recs, movies_df, links)
