import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src import cache, db
from src.recommenders import clustering, collaborative, content_based, neural, popularity
from src.recommenders.base import low_signal_movie_ids
from src.ui import inject_custom_css, movie_card, profile_sidebar, require_profile

st.set_page_config(page_title="Recommendations — CineMatch", page_icon="✨", layout="wide")
inject_custom_css()
profile_sidebar()
profile_id, profile_name = require_profile()
st.title("✨ Recommendations")

conn = cache.get_db_conn()
movies = cache.load_movies().set_index("movieId")
links = cache.load_links().set_index("movieId")
pop_df = cache.load_popularity()
low_signal = low_signal_movie_ids(pop_df)

watched_ids = db.get_watched_ids(conn, profile_id)
rated = db.get_ratings(conn, profile_id)
exclude = watched_ids | low_signal

MODEL_OPTIONS = {
    "Popularity (trending + genre overlap)": "popularity",
    "Content-based (TF-IDF similarity)": "content_based",
    "Collaborative filtering (SVD)": "collaborative",
    "Clustering (explore your taste cluster)": "clustering",
    "Neural network (two-tower embeddings)": "neural",
}

if not watched_ids:
    st.info(
        "You haven't marked anything watched yet, so every model would just be guessing. "
        "Showing **trending picks** until you add a few movies on the Browse page."
    )
    recs = popularity.top_trending(pop_df, n=12, exclude_ids=low_signal)
else:
    choice_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    model_key = MODEL_OPTIONS[choice_label]

    if model_key == "popularity":
        seed_genres = []
        for mid, r in rated.items():
            if r >= 4.0 and mid in movies.index:
                seed_genres.extend(movies.loc[mid, "genre_list"])
        recs = popularity.because_you_watched(pop_df, seed_genres, exclude_ids=exclude, n=12)

    elif model_key == "content_based":
        if not rated:
            st.warning("Content-based needs a few star ratings to build your taste profile — showing trending instead.")
            recs = popularity.top_trending(pop_df, n=12, exclude_ids=exclude)
        else:
            artifacts = cache.load_content_based()
            recs = content_based.recommend_for_profile(rated, artifacts, n=12, exclude_ids=exclude)

    elif model_key == "collaborative":
        if not rated:
            st.warning("Collaborative filtering needs a few star ratings to fold you into the model — showing trending instead.")
            recs = popularity.top_trending(pop_df, n=12, exclude_ids=exclude)
        else:
            artifacts = cache.load_collaborative()
            recs = collaborative.recommend_for_profile(rated, artifacts, n=12, exclude_ids=exclude)

    elif model_key == "clustering":
        artifacts = cache.load_clustering()
        recs = clustering.recommend_for_profile(watched_ids, artifacts, pop_df, n=12, exclude_ids=exclude)

    else:  # neural
        if not rated:
            st.warning("The neural model needs a few star ratings — showing trending instead.")
            recs = popularity.top_trending(pop_df, n=12, exclude_ids=exclude)
        else:
            model = cache.load_neural()
            recs = model.recommend_for_profile(rated, n=12, exclude_ids=exclude)

@st.fragment
def _render_recs_grid(recs, movies, links, conn, profile_id):
    imdb_ids = [
        links.loc[r.movie_id, "imdbId"]
        for r in recs
        if r.movie_id in movies.index and r.movie_id in links.index
    ]
    with st.spinner("Loading posters..."):
        meta_by_imdb = cache.fetch_omdb_batch(imdb_ids)

    cols = st.columns(4)
    for i, rec in enumerate(recs):
        if rec.movie_id not in movies.index:
            continue
        row = movies.loc[rec.movie_id]
        with cols[i % 4]:
            movie_row = {"title": row["title"], "genre_list": row["genre_list"], "year": row["year"]}
            links_row = links.loc[rec.movie_id] if rec.movie_id in links.index else None
            imdb_id = links_row["imdbId"] if links_row is not None else None
            with st.container(border=True):
                movie_card(movie_row, links_row, badge=rec.reason, meta=meta_by_imdb.get(imdb_id))
                if st.button("Mark watched", key=f"rec_watch_{rec.movie_id}", use_container_width=True):
                    db.toggle_watched(conn, profile_id, rec.movie_id)
                    st.rerun(scope="fragment")


if not recs:
    st.warning("Not enough signal yet for this model — try rating a few more movies, or switch models.")
else:
    _render_recs_grid(recs, movies, links, conn, profile_id)

st.divider()
st.subheader("Because you watched...")
liked = [mid for mid, r in rated.items() if r >= 4.0 and mid in movies.index]
if not liked:
    st.caption("Rate a movie 4★ or higher to unlock similar-movie recommendations here.")
else:
    seed_titles = {mid: movies.loc[mid, "title"] for mid in liked}
    seed_id = st.selectbox("Pick a movie you liked", list(seed_titles.keys()), format_func=lambda m: seed_titles[m])
    artifacts = cache.load_content_based()
    similar = content_based.similar_to_movie(seed_id, artifacts, n=8, exclude_ids=exclude)
    if similar:
        similar_imdb_ids = [
            links.loc[r.movie_id, "imdbId"]
            for r in similar
            if r.movie_id in movies.index and r.movie_id in links.index
        ]
        with st.spinner("Loading posters..."):
            similar_meta = cache.fetch_omdb_batch(similar_imdb_ids)

        cols = st.columns(4)
        for i, rec in enumerate(similar):
            if rec.movie_id not in movies.index:
                continue
            row = movies.loc[rec.movie_id]
            with cols[i % 4]:
                movie_row = {"title": row["title"], "genre_list": row["genre_list"], "year": row["year"]}
                links_row = links.loc[rec.movie_id] if rec.movie_id in links.index else None
                imdb_id = links_row["imdbId"] if links_row is not None else None
                with st.container(border=True):
                    movie_card(movie_row, links_row, badge=f"Similarity {rec.score:.2f}", meta=similar_meta.get(imdb_id))
