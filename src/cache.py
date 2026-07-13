"""Streamlit-caching wrappers around the plain-Python loaders in data_utils
and the recommenders. Centralized here (not in data_utils.py) so the offline
training/eval scripts can import data_utils without ever importing streamlit.
"""
import json

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
    # Cache key is imdb_id only; the sqlite connection itself is not
    # hashable-friendly for st.cache_data, so we open it fresh here (cheap).
    return omdb.fetch_metadata(imdb_id)
