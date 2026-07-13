import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src import cache, db
from src.ui import inject_custom_css, movie_card, profile_sidebar

st.set_page_config(page_title="Browse — CineMatch", page_icon="🎬", layout="wide")
inject_custom_css()
profile_sidebar()
st.title("🎬 Browse movies")

movies = cache.load_movies()
links = cache.load_links().set_index("movieId")
pop_df = cache.load_popularity().set_index("movieId")

from src.data_utils import all_genres

genres = all_genres(movies)

with st.form("filters"):
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    query = c1.text_input("Search title", placeholder="e.g. Matrix")
    genre_filter = c2.multiselect("Genre", genres)
    year_min, year_max = int(movies["year"].min()), int(movies["year"].max())
    year_range = c3.slider("Year", year_min, year_max, (year_min, year_max))
    sort_by = c4.selectbox("Sort by", ["Popularity", "Rating", "Newest", "Title"])
    submitted = st.form_submit_button("Search", type="primary")

if submitted:
    st.session_state["browse_page"] = 0

filtered = movies.copy()
if query:
    # regex=False: the search box is a plain-text field, not a regex prompt -
    # without this, characters like "(" (common in movie titles/queries)
    # raise an uncaught re.error and crash the page.
    filtered = filtered[filtered["title"].str.contains(query, case=False, na=False, regex=False)]
if genre_filter:
    filtered = filtered[filtered["genre_list"].apply(lambda gs: any(g in gs for g in genre_filter))]
# A few dozen MovieLens titles have no parseable release year (e.g. "Moonlight",
# "Ready Player One"). Keep them visible regardless of the year slider instead
# of silently hiding real, popular movies because of a data-quality gap.
year_col = filtered["year"]
filtered = filtered[year_col.isna() | ((year_col >= year_range[0]) & (year_col <= year_range[1]))]

filtered = filtered.join(pop_df[["count", "mean", "weighted_score"]], on="movieId")
sort_map = {
    "Popularity": ("weighted_score", False),
    "Rating": ("mean", False),
    "Newest": ("year", False),
    "Title": ("title", True),
}
sort_col, ascending = sort_map[sort_by]
filtered = filtered.sort_values(sort_col, ascending=ascending, na_position="last")

st.caption(f"{len(filtered):,} movies match your filters.")

PAGE_SIZE = 20
if "browse_page" not in st.session_state:
    st.session_state["browse_page"] = 0
start = st.session_state["browse_page"] * PAGE_SIZE
page_df = filtered.iloc[start : start + PAGE_SIZE]

conn = cache.get_db_conn()
profile_id = st.session_state.get("profile_id")


@st.fragment
def _render_grid(page_df, links, conn, profile_id):
    # Fetch all posters for this page in parallel up front instead of one
    # blocking network call per card - cuts a 20-poster cold page load from
    # ~14s to ~1-2s (warm/cached loads are already instant either way).
    imdb_ids = [
        links.loc[mid, "imdbId"] for mid in page_df["movieId"] if mid in links.index
    ]
    with st.spinner("Loading posters..."):
        meta_by_imdb = cache.fetch_omdb_batch(imdb_ids)

    watched_ids = db.get_watched_ids(conn, profile_id) if profile_id else set()

    cols = st.columns(4)
    for i, row in enumerate(page_df.itertuples()):
        with cols[i % 4]:
            movie_row = {"title": row.title, "genre_list": row.genre_list, "year": row.year}
            links_row = links.loc[row.movieId] if row.movieId in links.index else None
            imdb_id = links_row["imdbId"] if links_row is not None else None
            meta = meta_by_imdb.get(imdb_id)
            rating_txt = f"{row.mean:.1f}★ ({int(row.count)} ratings)" if row.count else "No ratings yet"
            with st.container(border=True):
                movie_card(movie_row, links_row, badge=rating_txt, meta=meta)

                if not profile_id:
                    st.caption("Pick a profile on Home to track watched movies.")
                else:
                    is_watched = row.movieId in watched_ids
                    label = "✅ Watched" if is_watched else "Mark watched"
                    if st.button(label, key=f"watch_{row.movieId}", use_container_width=True):
                        db.toggle_watched(conn, profile_id, row.movieId)
                        st.rerun(scope="fragment")


_render_grid(page_df, links, conn, profile_id)

nav1, nav2, nav3 = st.columns([1, 2, 1])
with nav1:
    if st.session_state["browse_page"] > 0 and st.button("← Previous"):
        st.session_state["browse_page"] -= 1
        st.rerun()
with nav3:
    if start + PAGE_SIZE < len(filtered) and st.button("Next →"):
        st.session_state["browse_page"] += 1
        st.rerun()
