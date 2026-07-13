"""Small shared UI helpers used by every page: profile session state, the
sidebar profile switcher, custom theming, and a reusable movie-card
renderer."""
import streamlit as st

from src import cache, db, omdb

_CSS_INJECTED_KEY = "_cinematch_css_injected"


def inject_custom_css():
    """Card hover/shadow polish, a nicer font, and tighter spacing - layered
    on top of the base theme in .streamlit/config.toml. Idempotent per
    session (Streamlit reruns the whole script on every interaction, so
    without this guard the same <style> block would be injected repeatedly).
    """
    if st.session_state.get(_CSS_INJECTED_KEY):
        return
    st.session_state[_CSS_INJECTED_KEY] = True

    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* Poster images: rounded corners + shadow + hover lift */
        [data-testid="stImage"] img {
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.35);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        [data-testid="stImage"] img:hover {
            transform: translateY(-4px) scale(1.015);
            box-shadow: 0 10px 24px rgba(0,0,0,0.5);
        }

        /* Buttons: smoother corners + hover lift, consistent across the app */
        .stButton > button {
            border-radius: 8px;
            transition: transform 0.12s ease, border-color 0.12s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            border-color: #e11d48;
            color: #e11d48;
        }

        /* Sidebar + card captions: slightly tighter, calmer typography */
        [data-testid="stCaptionContainer"] { opacity: 0.75; }

        /* Custom scrollbar */
        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #3a3f4b; border-radius: 6px; }
        ::-webkit-scrollbar-thumb:hover { background: #545b6b; }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def movie_card(movie_row, links_row=None, badge=None, meta=None):
    """Renders a poster (or genre-colored placeholder) + title/genres inside
    the current Streamlit container. Returns nothing; caller decides what
    interactive widgets go below the card.

    `meta` lets the caller pass pre-fetched OMDb data (see
    cache.fetch_omdb_batch) so a grid of cards does one parallel fetch
    instead of N sequential ones. Falls back to a single fetch if omitted.
    """
    if meta is None:
        imdb_id = links_row["imdbId"] if links_row is not None else None
        meta = cache.fetch_omdb(imdb_id) if imdb_id else {"poster_url": None}

    genres = movie_row.get("genre_list", [])
    title = movie_row["title"]

    if meta and meta.get("poster_url"):
        # streamlit==1.38.0 (pinned in requirements.txt) predates st.image's
        # use_container_width param - use_column_width is the equivalent for
        # this version.
        st.image(meta["poster_url"], use_column_width=True)
    else:
        color = omdb.genre_color(genres)
        st.markdown(
            f"""<div style="background:{color};height:220px;border-radius:10px;
            display:flex;align-items:center;justify-content:center;padding:12px;
            text-align:center;color:white;font-weight:600;font-size:0.9rem;
            box-shadow:0 2px 10px rgba(0,0,0,0.35);
            transition:transform 0.15s ease, box-shadow 0.15s ease;"
            onmouseover="this.style.transform='translateY(-4px) scale(1.015)'; this.style.boxShadow='0 10px 24px rgba(0,0,0,0.5)';"
            onmouseout="this.style.transform=''; this.style.boxShadow='0 2px 10px rgba(0,0,0,0.35)';">
            {title}</div>""",
            unsafe_allow_html=True,
        )

    caption = f"**{title}**"
    if badge:
        caption += f"  \n{badge}"
    st.caption(", ".join(genres) if genres else "—")
    st.markdown(caption)
