import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src import cache, db
from src.ui import movie_card, profile_sidebar, require_profile

st.set_page_config(page_title="Watched — CineMatch", page_icon="✅", layout="wide")
profile_sidebar()
profile_id, profile_name = require_profile()

st.title(f"✅ {profile_name}'s watched movies")

conn = cache.get_db_conn()
movies = cache.load_movies().set_index("movieId")
links = cache.load_links().set_index("movieId")

watched_ids = db.get_watched_ids(conn, profile_id)
ratings = db.get_ratings(conn, profile_id)

if not watched_ids:
    st.info("No watched movies yet — head to **Browse** and mark a few, or rate them right here after adding.")
    st.stop()

st.caption(f"{len(watched_ids)} movies watched, {len(ratings)} rated.")

sorted_ids = sorted(watched_ids, key=lambda m: ratings.get(m, 0), reverse=True)

cols = st.columns(4)
for i, movie_id in enumerate(sorted_ids):
    if movie_id not in movies.index:
        continue
    row = movies.loc[movie_id]
    with cols[i % 4]:
        movie_row = {"title": row["title"], "genre_list": row["genre_list"], "year": row["year"]}
        links_row = links.loc[movie_id] if movie_id in links.index else None
        movie_card(movie_row, links_row)

        current = ratings.get(movie_id, 0.0)
        new_rating = st.select_slider(
            "Your rating",
            options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            value=current if current else 3.0,
            key=f"rate_{movie_id}",
            label_visibility="collapsed",
        )
        c1, c2 = st.columns(2)
        if c1.button("Save rating", key=f"save_{movie_id}", use_container_width=True):
            db.set_rating(conn, profile_id, movie_id, new_rating)
            st.rerun()
        if c2.button("Remove", key=f"remove_{movie_id}", use_container_width=True):
            db.toggle_watched(conn, profile_id, movie_id)
            st.rerun()
        st.divider()
