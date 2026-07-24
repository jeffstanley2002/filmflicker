"""Rule-based recommender: Bayesian-weighted popularity + genre-overlap
"because you watched" logic. No training needed beyond aggregating ratings,
so this doubles as the explicit cold-start lane for brand-new profiles.
"""
from pathlib import Path

import pandas as pd

from src import data_utils
from src.recommenders.base import MODELS_DIR, Recommendation

ARTIFACT = MODELS_DIR / "popularity.csv"


def train_and_save(movies: pd.DataFrame, ratings: pd.DataFrame, out_path: Path = ARTIFACT):
    table = data_utils.popularity_table(movies, ratings)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    return table


def load(path: Path = ARTIFACT) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["movieId", "genres", "count", "mean", "weighted_score"],
    )
    df["genre_list"] = df["genres"].apply(
        lambda g: [] if pd.isna(g) or g == "(no genres listed)" else g.split("|")
    )
    return df


def top_trending(pop_df: pd.DataFrame, n: int = 20, exclude_ids=None, genre: str = None) -> list:
    df = pop_df
    if genre:
        df = df[df["genre_list"].apply(lambda gs: genre in gs)]
    if exclude_ids:
        df = df[~df["movieId"].isin(exclude_ids)]
    df = df.sort_values("weighted_score", ascending=False).head(n)
    return [
        Recommendation(
            movie_id=row.movieId,
            score=float(row.weighted_score),
            reason=f"Popular pick ({row.count:.0f} ratings, {row.mean:.1f}★ avg)",
            model="popularity",
        )
        for row in df.itertuples()
    ]


def because_you_watched(pop_df: pd.DataFrame, seed_genres: list, exclude_ids=None, n: int = 10) -> list:
    """Simple genre-overlap rule: score = (# shared genres) weighted by popularity."""
    if not seed_genres:
        return top_trending(pop_df, n=n, exclude_ids=exclude_ids)
    seed = set(seed_genres)
    df = pop_df.copy()
    df["overlap"] = df["genre_list"].apply(lambda gs: len(seed & set(gs)))
    df = df[df["overlap"] > 0]
    if exclude_ids:
        df = df[~df["movieId"].isin(exclude_ids)]
    df["rule_score"] = df["overlap"] * df["weighted_score"]
    df = df.sort_values("rule_score", ascending=False).head(n)
    return [
        Recommendation(
            movie_id=row.movieId,
            score=float(row.rule_score),
            reason=f"Shares {row.overlap} genre(s) with movies you liked",
            model="popularity",
        )
        for row in df.itertuples()
    ]
