import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from fastapi import APIRouter, Depends

import data as data_module
import db
from auth import get_current_user_id
from schemas import AnalyticsOut
from src.recommenders import clustering

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("", response_model=AnalyticsOut)
def get_analytics(user_id: str = Depends(get_current_user_id)):
    engine = db.get_connection()
    watched_ids, ratings = db.get_profile(engine, user_id)

    if not watched_ids:
        return AnalyticsOut(
            movies_watched=0,
            movies_rated=0,
            avg_rating=None,
            genre_breakdown={},
            rating_distribution={},
            decade_breakdown={},
            cluster_breakdown={},
        )

    movies_df = data_module.movies_indexed()

    genre_counts = Counter()
    decade_counts = Counter()
    for mid in watched_ids:
        if mid in movies_df.index:
            genre_counts.update(movies_df.loc[mid, "genre_list"])
            year = movies_df.loc[mid, "year"]
            if pd.notna(year):
                decade = int(year) // 10 * 10
                decade_counts[f"{decade}s"] += 1

    rating_counts = Counter(f"{r:.1f}" for r in ratings.values())

    artifacts = data_module.clustering_artifacts()
    cluster_counts = clustering.profile_cluster_counts(watched_ids, artifacts)
    cluster_breakdown = {clustering.cluster_label(c, artifacts): n for c, n in cluster_counts.items()}

    return AnalyticsOut(
        movies_watched=len(watched_ids),
        movies_rated=len(ratings),
        avg_rating=(sum(ratings.values()) / len(ratings)) if ratings else None,
        genre_breakdown=dict(genre_counts),
        rating_distribution=dict(rating_counts),
        decade_breakdown=dict(decade_counts),
        cluster_breakdown=cluster_breakdown,
    )
