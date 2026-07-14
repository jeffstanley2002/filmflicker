# CineMatch

CineMatch is a production-oriented movie recommendation app with a React frontend, FastAPI backend, Supabase Auth, and five trained recommender strategies.

## Stack

- Frontend: React, Vite, TypeScript, Supabase Auth, Recharts
- Backend: FastAPI, SQLAlchemy Core, Supabase Postgres
- Models: popularity, content-based, collaborative filtering, clustering, embedding-powered deep taste
- Data: MovieLens 32M processed catalog plus committed model artifacts under `models/`

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

The API creates these tables automatically at startup when `DATABASE_URL` is configured. You can also run this SQL in Supabase SQL Editor:

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
```

Every API query is scoped by the verified Supabase JWT subject, so users only read and mutate their own watch and rating rows.

## Model Metrics

Current metrics are generated from `models/metrics.json`. Regenerate them with:

```bash
python scripts/evaluate_models.py --max-ratings 1000000 --n-eval-users 250 --max-rating-predictions 100000
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
python scripts/train_models.py
python scripts/evaluate_models.py --max-ratings 1000000 --n-eval-users 250 --max-rating-predictions 100000
python scripts/validate_model_export.py
```

The default deep taste export uses an embedding artifact derived from trained SVD factors, so local training does not require TensorFlow.

## Checks

```bash
pytest
cd frontend && npm run lint && npm run build
```
