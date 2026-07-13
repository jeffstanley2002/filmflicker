import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd
import streamlit as st

from src import cache
from src.ui import inject_custom_css, profile_sidebar

st.set_page_config(page_title="Model Comparison — CineMatch", page_icon="🧪", layout="wide")
inject_custom_css()
profile_sidebar()
st.title("🧪 Model comparison")
st.caption(
    "Measured on an 80/20 held-out split of ratings.csv — see the Methodology note at the bottom."
)

metrics = cache.load_metrics()
if not metrics:
    st.warning("No metrics yet. Run `python scripts/evaluate_models.py` to generate `models/metrics.json`.")
    st.stop()

st.subheader("Rating prediction accuracy")
st.caption("Lower is better. Only collaborative filtering and the neural net predict star ratings directly.")
rp = metrics["rating_prediction"]
df = pd.DataFrame(
    [{"model": m, "metric": k, "value": v} for m, vals in rp.items() for k, v in vals.items()]
)
chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(x="model:N", y="value:Q", color="metric:N", xOffset="metric:N", tooltip=["model", "metric", "value"])
)
st.altair_chart(chart, use_container_width=True)

st.divider()

st.subheader(f"Top-{metrics['k']} recommendation quality")
st.caption(
    f"Precision@{metrics['k']} / Recall@{metrics['k']} against each test profile's held-out 4★+ movies, "
    f"averaged over {metrics['n_eval_users']} sampled users. Higher is better."
)
topn = metrics["top_n"]
df2 = pd.DataFrame(
    [{"model": m, "metric": k, "value": v} for m, vals in topn.items() for k, v in vals.items()]
)
chart2 = (
    alt.Chart(df2)
    .mark_bar()
    .encode(x="model:N", y="value:Q", color="metric:N", xOffset="metric:N", tooltip=["model", "metric", "value"])
)
st.altair_chart(chart2, use_container_width=True)

st.divider()
st.subheader("What each model is actually doing")
st.markdown(
    """
| Model | Approach | Strengths | Weaknesses |
|---|---|---|---|
| **Popularity** | Bayesian-weighted average rating + genre overlap | Zero cold-start cost, hard to beat for "safe" picks | Not personalized beyond genre |
| **Content-based** | TF-IDF (genres + tags) + cosine similarity | Works from day one with just a few ratings, explainable | Only sees surface metadata, not taste patterns |
| **Collaborative (SVD)** | Rank-5 matrix factorization over the user–item ratings matrix | Best on every metric measured here (RMSE, MAE, Precision@10, Recall@10) | Needs a fold-in approximation for brand-new users, "cold" for niche items |
| **Clustering (KMeans)** | Groups movies by genre + popularity features | Good for discovery/exploration outside your usual picks | Coarser signal than factorization |
| **Neural net (two-tower)** | User/item towers projecting into a shared latent space, dot product + biases, trained with Keras, served via NumPy | Second-best top-N precision; architecture directly targets the "leans on popularity" failure mode a simpler concat-MLP design had | Still behind SVD here — a small (~100K-rating) dataset favors SVD's lower parameter count |
"""
)

st.divider()
with st.expander("Methodology notes"):
    st.write(metrics["notes"])
    st.caption(
        "Movies with fewer than 5 ratings are excluded as recommendation candidates for every "
        "model (not just neural) — sparse items give any embedding-based model noisy signal, "
        "and it's a bad user experience regardless of model."
    )
    st.caption(
        "Collaborative's n_components=5 and the neural model's two-tower architecture were both "
        "chosen via held-out sweeps, not defaults — see the README's 'Measured results' section "
        "for the before/after numbers that justified each change."
    )
