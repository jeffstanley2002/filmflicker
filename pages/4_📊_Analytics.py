import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd
import streamlit as st

from src import cache, db
from src.recommenders import clustering
from src.ui import profile_sidebar, require_profile

st.set_page_config(page_title="Analytics — CineMatch", page_icon="📊", layout="wide")
profile_sidebar()
profile_id, profile_name = require_profile()
st.title(f"📊 {profile_name}'s taste profile")

conn = cache.get_db_conn()
movies = cache.load_movies().set_index("movieId")
watched_ids = db.get_watched_ids(conn, profile_id)
rated = db.get_ratings(conn, profile_id)

if not watched_ids:
    st.info("Watch and rate a few movies to see your taste profile here.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Movies watched", len(watched_ids))
c2.metric("Movies rated", len(rated))
c3.metric("Average rating given", f"{(sum(rated.values()) / len(rated)):.1f}★" if rated else "—")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Genre breakdown")
    genre_counts = Counter()
    for mid in watched_ids:
        if mid in movies.index:
            genre_counts.update(movies.loc[mid, "genre_list"])
    if genre_counts:
        df = pd.DataFrame(genre_counts.items(), columns=["genre", "count"]).sort_values("count", ascending=False)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(x=alt.X("count:Q", title="Movies watched"), y=alt.Y("genre:N", sort="-x", title=None))
        )
        st.altair_chart(chart, use_container_width=True)

with right:
    st.subheader("Rating distribution")
    if rated:
        df = pd.DataFrame({"rating": list(rated.values())})
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(x=alt.X("rating:Q", bin=alt.Bin(step=0.5), title="Stars"), y=alt.Y("count()", title="Movies"))
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("No ratings yet — rate movies on the Watched page.")

st.divider()

left2, right2 = st.columns(2)
with left2:
    st.subheader("Most-watched decade")
    decades = Counter()
    for mid in watched_ids:
        if mid in movies.index and pd.notna(movies.loc[mid, "year"]):
            decade = int(movies.loc[mid, "year"]) // 10 * 10
            decades[f"{decade}s"] += 1
    if decades:
        df = pd.DataFrame(decades.items(), columns=["decade", "count"]).sort_values("decade")
        chart = alt.Chart(df).mark_bar().encode(x="decade:N", y="count:Q")
        st.altair_chart(chart, use_container_width=True)

with right2:
    st.subheader("Taste clusters (KMeans)")
    artifacts = cache.load_clustering()
    counts = clustering.profile_cluster_counts(watched_ids, artifacts)
    if counts:
        df = pd.DataFrame(
            [{"cluster": f"Cluster {c}", "count": n} for c, n in counts.items()]
        ).sort_values("count", ascending=False)
        chart = alt.Chart(df).mark_arc().encode(theta="count:Q", color="cluster:N")
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            f"Your watched movies span {len(counts)} of the model's clusters — "
            "the Recommendations page pulls from your most common one."
        )
