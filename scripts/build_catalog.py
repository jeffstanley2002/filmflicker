"""Build the normalized FilmFlicker movie catalog.

Data roles:
- MovieLens supplies ratings/tags/links and the stable training signal.
- TMDB optionally enriches movie metadata, posters, overviews, and current
  release metadata when a TMDB key is configured.

Examples:
    python scripts/build_catalog.py --movielens-dir data/ml-latest-small
    TMDB_API_KEY=... python scripts/build_catalog.py --movielens-dir data/ml-32m --tmdb-limit 5000

Outputs:
    data/processed/movies.csv
    data/processed/ratings.csv
    data/processed/tags.csv
    data/processed/links.csv
    data/processed/catalog_manifest.json
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from datetime import date

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data_utils

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MOVIELENS_DIR = ROOT / "data" / "ml-latest-small"
DEFAULT_OUT_DIR = ROOT / "data" / "processed"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_SYNTHETIC_ID_OFFSET = 10_000_000


def _headers() -> dict[str, str]:
    token = os.environ.get("TMDB_BEARER_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _params() -> dict[str, str]:
    key = os.environ.get("TMDB_API_KEY")
    return {"api_key": key} if key else {}


def _tmdb_enabled() -> bool:
    return bool(os.environ.get("TMDB_API_KEY") or os.environ.get("TMDB_BEARER_TOKEN"))


def _fetch_tmdb_movie(tmdb_id: int, session: requests.Session) -> dict[str, Any] | None:
    url = f"{TMDB_API_BASE}/movie/{tmdb_id}"
    params = {"language": "en-US", **_params()}
    try:
        resp = session.get(url, params=params, headers=_headers(), timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        print(f"TMDB fetch failed for {tmdb_id}: {exc}")
        return None


def _fetch_tmdb_genres(session: requests.Session) -> dict[int, str]:
    url = f"{TMDB_API_BASE}/genre/movie/list"
    try:
        resp = session.get(url, params={"language": "en-US", **_params()}, headers=_headers(), timeout=15)
        resp.raise_for_status()
        return {int(g["id"]): g["name"] for g in resp.json().get("genres", [])}
    except requests.RequestException as exc:
        print(f"TMDB genre fetch failed: {exc}")
        return {}


def _append_current_tmdb_movies(
    movies: pd.DataFrame,
    links: pd.DataFrame,
    pages: int,
    sleep: float,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if pages <= 0 or not _tmdb_enabled():
        return movies, links, 0

    session = requests.Session()
    genre_lookup = _fetch_tmdb_genres(session)
    existing_tmdb = set(links["tmdbId"].dropna().astype(int))
    rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []

    for page in range(1, pages + 1):
        try:
            resp = session.get(
                f"{TMDB_API_BASE}/discover/movie",
                params={
                    "language": "en-US",
                    "include_adult": "false",
                    "include_video": "false",
                    "sort_by": "primary_release_date.desc",
                    "primary_release_date.lte": date.today().isoformat(),
                    "vote_count.gte": 5,
                    "page": page,
                    **_params(),
                },
                headers=_headers(),
                timeout=20,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"TMDB discover page {page} failed: {exc}")
            continue

        for item in resp.json().get("results", []):
            tmdb_id = item.get("id")
            if not tmdb_id or int(tmdb_id) in existing_tmdb:
                continue
            release_date = item.get("release_date")
            genres = [genre_lookup[g] for g in item.get("genre_ids", []) if g in genre_lookup]
            movie_id = TMDB_SYNTHETIC_ID_OFFSET + int(tmdb_id)
            rows.append(
                {
                    "movieId": movie_id,
                    "title": _title_with_year(item.get("title") or item.get("original_title") or f"TMDB {tmdb_id}", release_date),
                    "genres": "|".join(genres) if genres else "(no genres listed)",
                    "year": _year_from_date(release_date),
                    "poster_url": f"{TMDB_IMAGE_BASE}{item['poster_path']}" if item.get("poster_path") else None,
                    "overview": item.get("overview"),
                    "tmdb_release_date": release_date,
                    "metadata_source": "tmdb_current",
                }
            )
            link_rows.append({"movieId": movie_id, "imdbId": None, "tmdbId": int(tmdb_id)})
            existing_tmdb.add(int(tmdb_id))
        print(f"TMDB current page {page}/{pages}: added {len(rows):,} total movies")
        if sleep:
            time.sleep(sleep)

    if rows:
        movies = pd.concat([movies, pd.DataFrame(rows)], ignore_index=True)
        links = pd.concat([links, pd.DataFrame(link_rows)], ignore_index=True)
    return movies, links, len(rows)


def _title_with_year(title: str, release_date: str | None) -> str:
    year = _year_from_date(release_date)
    return f"{title} ({year})" if year and f"({year})" not in title else title


def _enrich_with_tmdb(movies: pd.DataFrame, links: pd.DataFrame, limit: int | None, sleep: float) -> pd.DataFrame:
    if not _tmdb_enabled():
        print("TMDB_API_KEY/TMDB_BEARER_TOKEN not set; writing MovieLens-only catalog.")
        movies["poster_url"] = None
        movies["overview"] = None
        movies["tmdb_release_date"] = None
        movies["metadata_source"] = "movielens"
        return movies

    links_by_movie = links.set_index("movieId")
    eligible = movies[movies["movieId"].isin(links_by_movie.index)].copy()
    if limit:
        # Prioritize newer movies first so portfolio demos surface fresh catalog metadata.
        eligible = eligible.sort_values("year", ascending=False, na_position="last").head(limit)

    session = requests.Session()
    enriched: dict[int, dict[str, Any]] = {}
    for i, row in enumerate(eligible.itertuples(), start=1):
        tmdb_id = links_by_movie.loc[row.movieId, "tmdbId"]
        if pd.isna(tmdb_id):
            continue
        data = _fetch_tmdb_movie(int(tmdb_id), session)
        if not data:
            continue
        genres = [g["name"] for g in data.get("genres", []) if g.get("name")]
        enriched[row.movieId] = {
            "tmdb_title": data.get("title") or data.get("original_title"),
            "tmdb_release_date": data.get("release_date"),
            "tmdb_year": _year_from_date(data.get("release_date")),
            "tmdb_genres": "|".join(genres) if genres else None,
            "poster_url": f"{TMDB_IMAGE_BASE}{data['poster_path']}" if data.get("poster_path") else None,
            "overview": data.get("overview"),
            "metadata_source": "tmdb",
        }
        if i % 100 == 0:
            print(f"TMDB enriched {i:,}/{len(eligible):,} candidates")
        if sleep:
            time.sleep(sleep)

    enriched_df = pd.DataFrame.from_dict(enriched, orient="index")
    enriched_df.index.name = "movieId"
    out = movies.join(enriched_df, on="movieId")
    out["poster_url"] = out.get("poster_url")
    out["overview"] = out.get("overview")
    out["tmdb_release_date"] = out.get("tmdb_release_date")
    out["metadata_source"] = out["metadata_source"].fillna("movielens")
    use_tmdb_genres = out["tmdb_genres"].notna()
    out.loc[use_tmdb_genres, "genres"] = out.loc[use_tmdb_genres, "tmdb_genres"]
    use_tmdb_year = out["tmdb_year"].notna()
    out.loc[use_tmdb_year, "year"] = out.loc[use_tmdb_year, "tmdb_year"]
    drop_cols = [c for c in ["tmdb_title", "tmdb_genres", "tmdb_year"] if c in out.columns]
    return out.drop(columns=drop_cols)


def _year_from_date(value: str | None):
    if not value or len(value) < 4:
        return None
    try:
        return int(value[:4])
    except ValueError:
        return None


def build_catalog(movielens_dir: Path, out_dir: Path, tmdb_limit: int | None, tmdb_current_pages: int, sleep: float):
    if not movielens_dir.exists():
        raise FileNotFoundError(f"MovieLens directory not found: {movielens_dir}")

    previous_data_dir = data_utils.DATA_DIR
    data_utils.DATA_DIR = movielens_dir
    try:
        movies = data_utils.load_movies()
        ratings = data_utils.load_ratings()
        tags = data_utils.load_tags()
        links = data_utils.load_links()
    finally:
        data_utils.DATA_DIR = previous_data_dir

    movies = _enrich_with_tmdb(movies, links, tmdb_limit, sleep)
    movies, links, current_added = _append_current_tmdb_movies(movies, links, tmdb_current_pages, sleep)

    out_dir.mkdir(parents=True, exist_ok=True)
    movies.to_csv(out_dir / "movies.csv", index=False)
    ratings.to_csv(out_dir / "ratings.csv", index=False)
    tags.to_csv(out_dir / "tags.csv", index=False)
    links.to_csv(out_dir / "links.csv", index=False)

    manifest = {
        "movielens_dir": str(movielens_dir),
        "movies": int(len(movies)),
        "ratings": int(len(ratings)),
        "tags": int(len(tags)),
        "min_year": int(movies["year"].min()) if movies["year"].notna().any() else None,
        "max_year": int(movies["year"].max()) if movies["year"].notna().any() else None,
        "tmdb_enabled": _tmdb_enabled(),
        "tmdb_enriched_movies": int((movies["metadata_source"] == "tmdb").sum()) if "metadata_source" in movies else 0,
        "tmdb_current_movies_added": current_added,
        "catalog_freshness": (
            "Catalog includes current TMDB releases"
            if current_added > 0
            else "Catalog refresh recommended before a public launch"
        ),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "catalog_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movielens-dir", type=Path, default=DEFAULT_MOVIELENS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tmdb-limit", type=int, default=None, help="Limit TMDB enrichment for fast smoke runs.")
    parser.add_argument("--tmdb-current-pages", type=int, default=0, help="Append current TMDB discover/movie pages as unrated catalog items.")
    parser.add_argument("--sleep", type=float, default=0.02, help="Seconds to sleep between TMDB calls.")
    parser.add_argument("--replace-app-data", action="store_true", help="Copy processed files over the active MovieLens data directory.")
    args = parser.parse_args()

    build_catalog(args.movielens_dir, args.out_dir, args.tmdb_limit, args.tmdb_current_pages, args.sleep)
    if args.replace_app_data:
        for name in ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]:
            shutil.copy2(args.out_dir / name, DEFAULT_MOVIELENS_DIR / name)


if __name__ == "__main__":
    main()
