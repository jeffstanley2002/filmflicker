"""Production recommendation ensemble shared by serving and evaluation."""
from src.recommenders import clustering, collaborative, content_based, popularity
from src.recommenders import ranking


def _seed_genres(movies_df, rated: dict) -> list:
    genres = []
    for movie_id, rating in rated.items():
        if rating >= 4.0 and movie_id in movies_df.index:
            genres.extend(movies_df.loc[movie_id, "genre_list"])
    return genres


def recommend(
    model: str,
    n: int,
    rated: dict,
    watched_ids: set,
    disliked_ids: set,
    movies_df,
    pop_df,
    low_signal: set,
    artifacts: dict,
    ranking_config: dict | None = None,
):
    exclude = set(watched_ids) | set(disliked_ids) | set(low_signal)
    preferences = dict(rated)
    preferences.update({movie_id: 1.0 for movie_id in disliked_ids if movie_id not in preferences})
    pool_n = min(200, max(n * 8, 40))

    if not watched_ids and not disliked_ids:
        primary = popularity.top_trending(pop_df, n=pool_n, exclude_ids=exclude)
    elif model == "popularity":
        primary = popularity.because_you_watched(
            pop_df, _seed_genres(movies_df, rated), exclude_ids=exclude, n=pool_n
        )
    elif model == "content_based":
        primary = content_based.recommend_for_profile(
            preferences, artifacts["content_based"], n=pool_n, exclude_ids=exclude
        )
    elif model == "collaborative":
        primary = collaborative.recommend_for_profile(
            preferences, artifacts["collaborative"], n=pool_n, exclude_ids=exclude
        )
    elif model == "clustering":
        primary = clustering.recommend_for_profile(
            preferences if preferences else watched_ids,
            artifacts["clustering"],
            pop_df,
            n=pool_n,
            exclude_ids=exclude,
        )
    else:
        primary = artifacts["neural"].recommend_for_profile(preferences, n=pool_n, exclude_ids=exclude)

    candidates = list(primary)
    if preferences and model != "content_based":
        candidates.extend(
            content_based.recommend_for_profile(
                preferences, artifacts["content_based"], n=max(n * 4, 24), exclude_ids=exclude
            )
        )
    if model != "popularity":
        candidates.extend(
            popularity.because_you_watched(
                pop_df, _seed_genres(movies_df, rated), exclude_ids=exclude, n=max(n * 3, 20)
            )
        )

    if not candidates:
        candidates = popularity.top_trending(pop_df, n=pool_n, exclude_ids=exclude)
    return ranking.rerank_candidates(
        candidates,
        movies_df,
        pop_df,
        n=n,
        primary_model=model,
        config=ranking_config,
    )
