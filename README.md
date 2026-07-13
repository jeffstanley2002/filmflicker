# CineMatch

A movie recommender that runs **5 independent ML approaches side by side** —
popularity/rule-based, content-based filtering, collaborative filtering
(matrix factorization), clustering, and a neural network — over the
[MovieLens](https://grouplens.org/datasets/movielens/) dataset, with a
Streamlit UI, multi-profile watch/rating tracking, and a measured
model-comparison dashboard.

**Live demo:** _add your deployed Streamlit Cloud URL here_

## Why 5 models instead of 1

Most recommender demos ship one model and call it done. CineMatch is built
to make the *tradeoffs between approaches* visible and measurable:

| Model | Approach | Best at | Weaker at |
|---|---|---|---|
| Popularity | Bayesian-weighted average rating + genre overlap | Cold start, zero training cost | Not personalized beyond genre |
| Content-based | TF-IDF (genres + tags) + cosine similarity | Works with just a few ratings, explainable | Only sees metadata, not behavior |
| Collaborative filtering | TruncatedSVD matrix factorization | Best top-N precision in eval below | New users need a fold-in approximation |
| Clustering | KMeans over genre + popularity features | Discovery / "more like my usual taste" | Coarser than factorization |
| Neural network | Embeddings + MLP (Keras, offline-trained) | Best rating-prediction accuracy (RMSE/MAE) | Weaker top-N ranking — see Model Comparison page |

Every model's actual measured accuracy is on the app's **Model Comparison**
page, computed from a held-out 80/20 split — not just claimed.

## Architecture

```
MovieLens CSVs (data/ml-latest-small/)
        │
        ▼
scripts/train_models.py  ──offline──►  models/*.{joblib,npz,csv,json}
  (TF-IDF, SVD, KMeans,                       │
   Keras neural net)                          │  read-only
                                               ▼
                                    src/recommenders/*.py
                                    (pure Python + NumPy/sklearn,
                                     NO TensorFlow at serve time)
                                               │
                                               ▼
                            Streamlit app (app.py + pages/)
                                     │
                                     ▼
                        SQLite (app_data/app.db) — profiles,
                        watched list, ratings (per deployment)
```

The neural model is trained offline with Keras/TensorFlow
(`requirements-train.txt`), but only its learned weight matrices are
exported to `models/neural_weights.npz`. At serve time,
`src/recommenders/neural.py` reimplements the forward pass in plain NumPy —
so the **deployed app never imports TensorFlow**, which keeps the Docker
image small and avoids free-tier memory limits. Verified with a clean-venv
install: `requirements.txt` has zero TensorFlow/ML-training dependencies.

## Measured results (80/20 held-out split, k=10, 150 sampled test users)

| Model | RMSE ↓ | MAE ↓ | Precision@10 ↑ | Recall@10 ↑ |
|---|---|---|---|---|
| Collaborative (SVD) | **0.99** | **0.77** | **0.095** | **0.068** |
| Neural network | 1.12 | 0.84 | 0.013 | 0.013 |
| Clustering | — | — | 0.046 | 0.043 |
| Popularity | — | — | 0.025 | 0.019 |
| Content-based | — | — | 0.011 | 0.006 |

Regenerate with `python scripts/evaluate_models.py` (writes
`models/metrics.json`, which the Model Comparison page reads).

**Two tuning passes behind these numbers**, both driven by held-out
evaluation rather than guesswork:

1. **SVD's rank was overfit.** An offline sweep over `n_components` in
   [3, 5, 8, 10, 15, 20, 30, 50, 100, 150] showed RMSE rising monotonically
   past rank 5 (0.976 → 1.49 at rank 150) - with only 610 users and ~9.7K
   movies, a high-rank factorization has more parameters than the data can
   support. Dropping the default from 50 to 5 components nearly doubled
   Precision@10 (0.051 → 0.095) and cut RMSE by ~22%.
2. **The neural model's architecture was the bottleneck, not its
   training.** The original concat-embeddings-into-an-MLP design leaned on
   item popularity instead of learning sharp personalization - Precision@10
   was an almost-random 0.001. Replacing it with a two-tower / dot-product
   architecture (separate user and item towers projecting into a shared
   latent space, forcing an explicit interaction term) lifted Precision@10
   13x to 0.013. It's still the weakest top-N model of the five - a fair,
   documented outcome for a small dataset, not a hidden flaw - and its RMSE
   ticked up slightly (1.10 → 1.12) even as MAE and top-N both improved, a
   real mixed result worth stating plainly rather than rounding up.

See the Model Comparison page's Methodology notes for why movies with <5
ratings are excluded as candidates for every model (their embeddings
otherwise overfit to a single noisy data point and dominate every user's
top-N, regardless of model).

## Tech stack

- **App:** Streamlit (multi-page), Altair for charts
- **ML:** scikit-learn (TF-IDF, TruncatedSVD, KMeans), Keras/TensorFlow (offline only)
- **Data:** MovieLens `ml-latest-small` (~100K ratings, 9.7K movies), committed to the repo
- **Storage:** SQLAlchemy Core over SQLite (local/default) or Postgres (once `DATABASE_URL` is set - see below), OMDb response cache
- **Optional:** OMDb API for real posters (free key), graceful placeholder fallback otherwise

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

42 tests covering data/feature-engineering logic, the SQLite data layer
(idempotency, upserts, cross-profile isolation, DB-level constraint
enforcement), the metric functions, and integration tests against the real
trained artifacts for all 5 recommenders (exclude-list contracts, score
ranges, and a direct regression test for the low-signal-movie bug described
above).

## Running locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py       # fetches MovieLens (skips if already present)

# Models are already trained and committed under models/. To retrain:
pip install -r requirements-train.txt  # adds TensorFlow, needed only for this step
python scripts/train_models.py
python scripts/evaluate_models.py      # optional, regenerates models/metrics.json

streamlit run app.py
```

To enable real posters, copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and paste in a free key from
[omdbapi.com/apikey.aspx](https://www.omdbapi.com/apikey.aspx). Without a
key the app runs fine — movies just render as genre-colored placeholder
cards instead of posters.

## Setting up Postgres (Supabase) for real persistent users

Local dev and a quick demo deploy both work with **zero setup** - `src/db.py`
falls back to a local SQLite file automatically. But Streamlit Community
Cloud's filesystem is ephemeral (it resets on every restart/redeploy), so
if real users are going to create profiles and expect their watched
list/ratings to stick around, point the app at a free hosted Postgres
instead:

1. Create a free project at [supabase.com](https://supabase.com).
2. In your project's **Settings → Database**, copy the **Connection
   string** (URI format, "Session pooler" mode is fine for this app's
   traffic level).
3. Append `?sslmode=require` to the end of it.
4. Set it as `DATABASE_URL`:
   - **Locally** (optional, to test against the real DB before deploying):
     add to `.streamlit/secrets.toml` alongside `OMDB_API_KEY`.
   - **Deployed**: paste into Streamlit Cloud's **Settings → Secrets**.
5. Redeploy (or just rerun locally) - `src/db.py` detects `DATABASE_URL`
   automatically and switches from SQLite to Postgres, creating the same
   `profiles`/`watched`/`ratings` tables (Postgres-flavored DDL) on first
   connection. No code changes needed either way.

If `DATABASE_URL` is never set, the app just keeps using local SQLite -
fine for a personal demo, not for real users who expect their data to
persist.

## Deploying for free (Streamlit Community Cloud)

1. Push this repo to your own GitHub account (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**, and point it at this repo with main file
   path `app.py`.
3. In the app's **Settings → Secrets**, paste:
   ```toml
   OMDB_API_KEY = "your_key_here"
   DATABASE_URL = "postgresql://...?sslmode=require"  # optional, see above
   ```
   (Both are optional — the app degrades gracefully without either.)
4. Deploy. First build takes a few minutes; `models/` and `data/` are
   already committed, so no training happens at deploy time.

## Project structure

```
data/ml-latest-small/     MovieLens CSVs (committed)
models/                   Trained artifacts + metrics.json (committed, generated by scripts/train_models.py)
scripts/
  download_data.py        Fetch MovieLens
  train_models.py         Offline training for all 5 models
  evaluate_models.py       Held-out RMSE/MAE + Precision@K/Recall@K -> models/metrics.json
src/
  data_utils.py            Plain-Python data loading/feature engineering (no Streamlit dependency)
  db.py                    SQLite: profiles, watched, ratings
  omdb.py                  Optional poster/metadata fetch + cache
  evaluate.py               Shared metric functions
  cache.py / ui.py          Streamlit-specific caching + shared UI components
  recommenders/             One module per model (popularity, content_based, collaborative, clustering, neural)
app.py                     Home page: intro + profile picker
pages/                     Browse, Watched, Recommendations, Analytics, Model Comparison
tests/                     pytest suite (data layer, DB layer, metrics, recommenders)
```
