"""Small shared UI helpers used by every page: profile session state, the
sidebar profile switcher, and a reusable movie-card renderer."""
import streamlit as st

from src import cache, db, omdb


def require_profile():
    if "profile_id" not in st.session_state:
        st.warning("Pick or create a profile on the **Home** page first.")
        st.stop()
    return st.session_state["profile_id"], st.session_state["profile_name"]


def profile_sidebar():
    conn = cache.get_db_conn()
    st.sidebar.markdown("### Profile")
    if "profile_name" in st.session_state:
        st.sidebar.success(f"Active: **{st.session_state['profile_name']}**")
        if st.sidebar.button("Switch profile", use_container_width=True):
            del st.session_state["profile_id"]
            del st.session_state["profile_name"]
            st.rerun()
    else:
        st.sidebar.info("No profile selected yet — go to **Home**.")
    st.sidebar.divider()
    st.sidebar.caption(
        "CineMatch runs 5 independent recommender models side by side — "
        "see the Model Comparison page for how they stack up."
    )


def movie_card(movie_row, links_row=None, badge=None):
    """Renders a poster (or genre-colored placeholder) + title/year/genres
    inside the current Streamlit container. Returns nothing; caller decides
    what interactive widgets go below the card."""
    imdb_id = links_row["imdbId"] if links_row is not None else None
    meta = cache.fetch_omdb(imdb_id) if imdb_id else {"poster_url": None}

    genres = movie_row.get("genre_list", [])
    year = movie_row.get("year")
    title = movie_row["title"]

    if meta and meta.get("poster_url"):
        st.image(meta["poster_url"], use_container_width=True)
    else:
        color = omdb.genre_color(genres)
        st.markdown(
            f"""<div style="background:{color};height:220px;border-radius:8px;
            display:flex;align-items:center;justify-content:center;padding:12px;
            text-align:center;color:white;font-weight:600;font-size:0.9rem;">
            {title}</div>""",
            unsafe_allow_html=True,
        )

    caption = f"**{title}**"
    if badge:
        caption += f"  \n{badge}"
    st.caption(", ".join(genres) if genres else "—")
    st.markdown(caption)
