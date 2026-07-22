# CineMatch

CineMatch is a production-oriented movie recommendation app with a React frontend, FastAPI backend, Supabase Auth, and five trained recommender strategies.

## Stack

- Frontend: React, Vite, TypeScript, Supabase Auth, Recharts
- Backend: FastAPI, SQLAlchemy Core, Supabase Postgres
- Models: popularity, content-based, collaborative filtering, clustering, and latent taste embeddings
- Data: MovieLens 32M processed catalog plus committed model artifacts under `models/`

## Model Documentation

For the complete model history, training formulas, tuning rounds, final metrics, deployment checks, limitations, and exact reproduction workflow, see [MODEL_TRAINING_AND_EVALUATION.md](MODEL_TRAINING_AND_EVALUATION.md).

For a module map, request/model flows, architectural invariants, and guidance on where to make common changes, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).

## Local Development

Install backend dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
cp .env.example .env.local
```

Set `frontend/.env.local`:

```bash
VITE_API_URL=http://localhost:8000
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
```

Set `backend/.env`:

```bash
DATABASE_URL=postgresql://...
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_JWT_SECRET=your-supabase-jwt-secret
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
```

Run the API:

```bash
uvicorn backend.main:app --reload
```

Run the frontend:

```bash
cd frontend
npm run dev
```

Open:

- App: `http://localhost:5173`
- API readiness JSON: `http://localhost:8000/metrics/system`

## Supabase Schema

The API applies serialized, versioned migrations from `backend/migrations/` at startup when `DATABASE_URL` is configured. The initial schema is:

```sql
create schema if not exists cinematch_v2;

create table if not exists cinematch_v2.watched (
  id serial primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  movie_id integer not null,
  watched_at timestamptz not null default now(),
  unique (user_id, movie_id)
);

create table if not exists cinematch_v2.ratings (
  id serial primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  movie_id integer not null,
  rating real not null check (rating between 0.5 and 5.0),
  rated_at timestamptz not null default now(),
  unique (user_id, movie_id)
);

create table if not exists cinematch_v2.not_interested (
  id serial primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  movie_id integer not null,
  created_at timestamptz not null default now(),
  unique (user_id, movie_id)
);

alter table cinematch_v2.watched enable row level security;
alter table cinematch_v2.ratings enable row level security;
alter table cinematch_v2.not_interested enable row level security;
```

Every API query is scoped by the verified Supabase JWT subject, so users only read and mutate their own watch, rating, and not-interested rows. The migrations also enable Row-Level Security with owner-scoped policies for `authenticated` users and lock down any legacy `public` tables left from older local-storage experiments.

## Model Metrics

Current metrics are generated from `models/metrics.json` using a per-user temporal holdout of each user's latest interactions. The final untouched test contains 1,000 complete user histories:

| Model | RMSE | Precision@10 | Recall@10 | Hit Rate@10 | NDCG@10 | Diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Collaborative hybrid | 0.907 | 1.40% | 4.07% | 12.2% | 3.08% | 78.8% |
| Content-based | - | 1.27% | 3.55% | 11.3% | 3.02% | 69.1% |
| Popularity | - | 1.11% | 3.14% | 10.0% | 2.29% | 69.6% |
| Taste embeddings | 0.932 | 1.07% | 3.21% | 9.0% | 2.48% | 78.6% |
| Taste neighborhoods | - | 1.03% | 3.14% | 9.4% | 2.19% | 66.6% |

Tune only on the chronological validation split:

```bash
python scripts/tune_models.py --max-ratings 1000000 --n-validation-users 200
```

Regenerate the untouched final metrics with:

```bash
python scripts/evaluate_models.py --max-ratings 1000000 --n-eval-users 1000 --max-rating-predictions 100000
```

Validate exported model artifacts before deployment:

```bash
python scripts/validate_model_export.py
```

## Fresh Catalog And Retraining

The production data architecture is:

- MovieLens 32M: free catalog, ratings, tags, and links for supervised recommender training
- Supabase: live user watch/rating events for future personalization

Download a larger MovieLens training set:

```bash
python scripts/download_movielens.py 32m
```

Build a normalized processed catalog. Without a TMDB key, this still writes a clean MovieLens catalog. With `TMDB_API_KEY` or `TMDB_BEARER_TOKEN`, it enriches movies with TMDB posters and metadata:

```bash
python scripts/build_catalog.py --movielens-dir data/ml-32m --tmdb-current-pages 0
```

For a fast smoke test:

```bash
python scripts/build_catalog.py --movielens-dir data/ml-latest-small --tmdb-current-pages 0
```

The free MovieLens 32M catalog currently reaches 2023. Very new releases should be added later through a separate free-or-approved catalog source plus enough interaction data to make personalization meaningful. After the processed catalog is built, retrain, evaluate, and validate export artifacts:

```bash
python scripts/tune_models.py --max-ratings 1000000 --n-validation-users 200
# Promote the validation winner in collaborative.py and ranking.py, then:
python scripts/train_models.py
python scripts/evaluate_models.py --max-ratings 1000000 --n-eval-users 1000 --max-rating-predictions 100000
python scripts/validate_model_export.py
```

The default taste-embedding export is derived from trained SVD factors, so local training does not require TensorFlow.

## Checks

```bash
pytest
python scripts/validate_model_export.py
cd frontend && npm run lint && npm run build
```

## Deployment

The root `Dockerfile` runs the API as one worker so the large read-only model cache is not duplicated. `render.yaml` declares the required secrets and readiness probe. Deploy `frontend/` separately on Vercel; `frontend/vercel.json` provides the SPA fallback. Set the production frontend origin in `ALLOWED_ORIGINS` and its API URL in `VITE_API_URL`.

The `.github/workflows/keepalive.yml` workflow can ping the deployed API twice a day. Set the GitHub Actions secret `CINEMATCH_API_URL` to the deployed API origin, for example `https://cinematch-api.onrender.com`. The `/health` endpoint checks the database, which creates regular Supabase activity. Supabase Pro is still the only guaranteed way to prevent Free Plan inactivity pauses.

Before the first production build, set all three Vercel variables: `VITE_API_URL`, `VITE_SUPABASE_URL`, and `VITE_SUPABASE_ANON_KEY`. Use the exact Vercel origin (without a trailing slash) in the API's `ALLOWED_ORIGINS`.

The API is designed for one worker. Typical startup is about 379 MB RSS, a neural request peaks near 432 MB, and cycling through every strategy can transiently reach about 538 MB with the current artifacts. The Render Blueprint therefore pins the 2 GB `standard` plan. For another host, use at least 1 GB RAM.

### First-deploy checklist

1. Commit and push every source, migration, model, manifest, Docker, and hosting file.
2. Deploy the Render API and set `DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_JWT_SECRET`, and `ALLOWED_ORIGINS`.
3. Confirm the deployed API's `/health` response reports configuration, database, and models ready.
4. In Vercel, set `VITE_API_URL` to the Render HTTPS origin and set both Supabase frontend variables before building.
5. In Supabase Auth URL Configuration, set the Site URL to the exact Vercel production origin and allow the exact `https://your-domain/app/browse` redirect used by Google sign-in.
6. Enable/configure the Google provider in Supabase if Google sign-in should be available.
7. Open the production site in a private browser window and smoke-test email login, Google login, Browse, For You, Watched, Taste, sign-out confirmation, and a mobile viewport.

References: [Supabase redirect URL configuration](https://supabase.com/docs/guides/auth/redirect-urls), [Supabase Google login](https://supabase.com/docs/guides/auth/social-login/auth-google), and [Render instance types](https://render.com/docs/compute-plans).
