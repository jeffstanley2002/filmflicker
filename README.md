<p align="center">
  <img src="frontend/public/brand/filmflicker-lockup.png" alt="FilmFlicker" width="320">
</p>

<p align="center">
  <strong>Personal movie recommendations powered by five different recommender models.</strong>
</p>

---

> **Note:** FilmFlicker is no longer deployed. The live site and API were taken down to cut hosting costs, and the project is no longer actively maintained. The code is still here for anyone who wants to read it, learn from it, or run it locally.

## About

FilmFlicker is a full-stack movie recommendation app. You sign up, browse a catalog of tens of thousands of movies, mark what you've watched, and rate what you've seen. The more you rate, the better your recommendations get.

Instead of relying on one ranking trick, FilmFlicker runs five recommender models side by side. You can switch between them and see how each one reads your taste differently.

### Features

- **Browse**: search and filter the MovieLens catalog, with posters and metadata
- **For You**: personalized recommendations, with a choice of which model to use
- **Watchlist and Watched**: keep track of movies you want to see and ones you've already seen
- **Ratings**: half-star ratings that feed straight back into your recommendations
- **Not interested**: hide movies so they stop showing up
- **Taste analytics**: charts of your genres, rating habits, and viewing history
- **System design page**: a public walkthrough of how a rating turns into a recommendation
- **Accounts**: email/password and Google sign-in through Supabase Auth, with every user's data kept separate

## The models

| Model | What it does |
| --- | --- |
| **Content-based** | Recommends movies similar to what you've rated highly, based on genres, tags, and metadata |
| **Collaborative hybrid** | Matrix factorization over 32M ratings to find people with similar taste |
| **Taste embeddings** | A two-tower neural recommender trained on all 32,000,204 MovieLens ratings |
| **Popularity** | A strong baseline built on well-rated, widely-watched movies |
| **Taste neighborhoods** | Clusters users into taste groups and recommends from your group |

### Results

All models were evaluated on a held-out set of each user's most recent ratings (a chronological split, so no future data leaks into training), across 1,000 users from the full MovieLens 32M dataset.

| Model | RMSE | Precision@10 | Recall@10 | Hit Rate@10 | NDCG@10 | Diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Content-based | - | **1.50%** | **4.30%** | **12.5%** | **3.28%** | 54.6% |
| Collaborative hybrid | **0.899** | 1.00% | 2.83% | 9.0% | 2.10% | **86.6%** |
| Taste embeddings | 0.966 | 0.63% | 1.77% | 5.6% | 1.23% | 80.2% |
| Popularity | - | 0.17% | 0.44% | 1.7% | 0.43% | 56.7% |
| Taste neighborhoods | - | 0.00% | 0.00% | 0.0% | 0.00% | 60.0% |

Content-based matching was the strongest top-10 recommender, while the collaborative model was best at predicting the exact rating. The neural taste embeddings overlap with collaborative filtering on only 17.3% of their top-10 picks, so the two models surface different movies.

## Tech stack

- **Frontend**: React, TypeScript, Vite, Recharts
- **Backend**: FastAPI, SQLAlchemy Core
- **Database and auth**: Supabase (Postgres with Row-Level Security, plus Supabase Auth)
- **ML**: NumPy, pandas, scikit-learn, trained on [MovieLens 32M](https://grouplens.org/datasets/movielens/)
- **Hosting (formerly)**: Render for the API, Vercel for the frontend

## Project structure

```
backend/     FastAPI app: routes, auth, database migrations
frontend/    React single-page app
src/         Recommender models and evaluation code
scripts/     Data download, catalog building, training, tuning, and evaluation
models/      Trained model artifacts
tests/       Backend and recommender tests
```

## Running it locally

You'll need Python 3, Node.js, and a free [Supabase](https://supabase.com) project.

**Backend**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp backend/.env.example backend/.env   # fill in your Supabase values
uvicorn backend.main:app --reload
```

`backend/.env` needs `DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_JWT_SECRET`, and `ALLOWED_ORIGINS`. Database migrations run automatically when the API starts.

**Frontend**

```bash
cd frontend
npm install
cp .env.example .env.local   # fill in VITE_API_URL and your Supabase keys
npm run dev
```

Then open http://localhost:5173.

**Retraining the models (optional)**

```bash
python scripts/download_movielens.py 32m
python scripts/build_catalog.py --movielens-dir data/ml-32m --tmdb-current-pages 0
FILMFLICKER_DATA_DIR=data/ml-32m python scripts/train_models.py
python scripts/validate_model_export.py
```

**Tests**

```bash
pytest
cd frontend && npm run lint && npm run build
```

## Author

Built by [Jeffrey Stanley](https://github.com/jeffstanley2002).
