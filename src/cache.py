"""Streamlit-caching wrappers around the plain-Python loaders in data_utils
and the recommenders. Centralized here (not in data_utils.py) so the offline
training/eval scripts can import data_utils without ever importing streamlit.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from src import data_utils, db, omdb
from src.recommenders import clustering, collaborative, content_based, neural, popularity
from src.recommenders.base import MODELS_DIR


@st.cache_data(show_spinner=False)
def load_movies():
    return data_utils.load_movies()


@st.cache_data(show_spinner=False)
def load_ratings():
    return data_utils.load_ratings()


@st.cache_data(show_spinner=False)
def load_tags():
    return data_utils.load_tags()


@st.cache_data(show_spinner=False)
def load_links():
    return data_utils.load_links()


@st.cache_resource(show_spinner=False)
def get_db_conn():
    return db.get_connection()


@st.cache_resource(show_spinner=False)
def get_omdb_conn():
    return omdb.get_cache_conn()


@st.cache_resource(show_spinner="Loading popularity model...")
def load_popularity():
    return popularity.load()


@st.cache_resource(show_spinner="Loading content-based model...")
def load_content_based():
    return content_based.load()


@st.cache_resource(show_spinner="Loading collaborative filtering model...")
def load_collaborative():
    return collaborative.load()


@st.cache_resource(show_spinner="Loading clustering model...")
def load_clustering():
    return clustering.load()


@st.cache_resource(show_spinner="Loading neural model...")
def load_neural():
    return neural.load()


@st.cache_data(show_spinner=False)
def load_metrics():
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def fetch_omdb(imdb_id):
    # get_omdb_conn() is itself st.cache_resource-wrapped, so this reuses
    # the one shared connection instead of opening a new sqlite connection
    # on every single poster lookup.
    return omdb.fetch_metadata(imdb_id, conn=get_omdb_conn())


def fetch_omdb_batch(imdb_ids) -> dict:
    """Fetches metadata for many movies concurrently instead of one blocking
    network round-trip at a time. Cache hits (the common case after the
    first page load) resolve instantly inside the thread; only genuine
    cache misses actually hit the network, and even those overlap instead
    of serializing - a page of 20 uncached posters drops from ~14s to ~1-2s.
    """
    ids = [i for i in dict.fromkeys(imdb_ids) if i]  # de-dupe, preserve order, drop falsy
    if not ids:
        return {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = pool.map(fetch_omdb, ids)
    return dict(zip(ids, results))
