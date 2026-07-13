"""CineMatch — a multi-model movie recommender. Home page: intro + profile picker."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src import cache, db
from src.ui import inject_custom_css, profile_sidebar

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")
inject_custom_css()
profile_sidebar()

st.markdown(
    """
    <div style="padding: 1.75rem 0 0.5rem 0;">
        <div style="font-size: 2.75rem; font-weight: 800; line-height: 1.1;">
            🎬 CineMatch
        </div>
        <div style="font-size: 1.15rem; opacity: 0.8; margin-top: 0.5rem; max-width: 760px;">
            A movie recommender that runs <b>5 different ML approaches side by side</b>
            on the MovieLens dataset — popularity ranking, content-based filtering,
            matrix factorization, clustering, and a neural network — so you can see
            how each one makes a different call about what you'd like next.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

movies = cache.load_movies()
ratings = cache.load_ratings()
col1, col2, col3 = st.columns(3)
col1.metric("Movies", f"{len(movies):,}")
col2.metric("Ratings in dataset", f"{len(ratings):,}")
col3.metric("MovieLens users", f"{ratings['userId'].nunique():,}")

st.divider()

st.subheader("Your profile")
st.caption(
    "No password needed — profiles are just named buckets so you (or anyone "
    "trying the demo) can keep a separate watched list and ratings."
)

conn = cache.get_db_conn()
profiles = db.get_profiles(conn)

with st.container(border=True):
    left, right = st.columns(2)
    with left:
        st.markdown("**Use an existing profile**")
        if profiles:
            names = [p["name"] for p in profiles]
            choice = st.selectbox("Profile", names, label_visibility="collapsed")
            if st.button("Continue as this profile", type="primary", use_container_width=True):
                match = next(p for p in profiles if p["name"] == choice)
                st.session_state["profile_id"] = match["id"]
                st.session_state["profile_name"] = match["name"]
                st.rerun()
        else:
            st.caption("No profiles yet — create the first one.")

    with right:
        st.markdown("**Create a new profile**")
        new_name = st.text_input("Name", label_visibility="collapsed", placeholder="e.g. jeff")
        if st.button("Create & continue", use_container_width=True) and new_name.strip():
            profile_id = db.get_or_create_profile(conn, new_name)
            st.session_state["profile_id"] = profile_id
            st.session_state["profile_name"] = new_name.strip()
            st.rerun()

if "profile_name" in st.session_state:
    st.success(
        f"Active profile: **{st.session_state['profile_name']}**. "
        "Head to **Browse** to start marking movies watched, then check **Recommendations**."
    )

st.divider()
with st.expander("How the 5 models work"):
    st.markdown(
        """
- **Popularity / rule-based** — Bayesian-weighted ratings (like IMDb's Top 250) plus genre-overlap "because you watched X" logic. Also the cold-start fallback for brand-new profiles.
- **Content-based** — TF-IDF over genres + community tags, cosine similarity. Only needs *what a movie is about*, not other users.
- **Collaborative filtering** — matrix factorization (TruncatedSVD) over the user–item ratings matrix; predicts your rating for movies based on patterns across everyone's ratings.
- **Clustering** — KMeans groups movies into taste clusters; you get recommendations from whichever cluster your watched movies fall into most.
- **Neural network** — a two-tower model (user/item embeddings projected into a shared space, combined via dot product) trained offline with Keras/TensorFlow, served at runtime with plain NumPy (no TensorFlow needed in production).

See the **Model Comparison** page for measured accuracy (RMSE/MAE, Precision@10/Recall@10) on a held-out test split.
        """
    )
