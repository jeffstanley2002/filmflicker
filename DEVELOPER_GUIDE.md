# FilmFlicker Developer Guide

This guide is the shortest path from cloning the repository to making a safe change. The README covers setup and deployment; `MODEL_TRAINING_AND_EVALUATION.md` is the detailed model record.

## Repository map

| Path | Responsibility |
| --- | --- |
| `frontend/src/` | React pages, reusable UI components, API client, and Supabase browser session |
| `backend/main.py` | FastAPI assembly, startup validation, middleware, and health endpoints |
| `backend/routers/` | Thin HTTP handlers grouped by product capability |
| `backend/db.py` | All application-owned Postgres queries and mutations |
| `backend/data.py` | Process-level catalog/model loading and caching |
| `backend/movie_mapper.py` | Catalog/model records converted to public API response schemas |
| `backend/schemas.py` | Pydantic request and response contracts |
| `backend/migrations/` | Ordered, immutable database migrations |
| `src/recommenders/` | Framework-independent recommendation and reranking logic |
| `scripts/` | Offline catalog, training, tuning, evaluation, and release validation commands |
| `models/` | Versioned serving artifacts and machine-readable evaluation results |
| `tests/` | Model, data, security, and backend correctness tests |

## Request flow

```text
React page
  -> frontend/src/lib/api.ts
  -> FastAPI router
  -> verified Supabase JWT subject
  -> backend/db.py and/or backend/data.py
  -> backend/movie_mapper.py
  -> Pydantic response
```

Authenticated state is never accepted from a request body. `backend/auth.py` verifies the configured Supabase issuer and returns the token's UUID subject. Every user-data query receives that UUID explicitly.

## Recommendation flow

```text
ratings + watched + not interested
  -> selected candidate generator
  -> content/popularity supplements
  -> shared quality/diversity reranker
  -> catalog display-data join
  -> RecommendationOut
```

The model named by the user is the serving strategy. `source_model` records which candidate generator produced each final item. Preserve both fields when changing recommendation output.

## Important invariants

1. Never derive a JWKS URL or accepted issuer from an unverified token.
2. Every watched/rating/not-interested query must be scoped by `user_id`.
3. Training writes artifacts to staging and publishes the manifest last.
4. The API must fail startup when configuration, migrations, checksums, or model loading fail.
5. Do not load training-only user matrices into the serving artifact.
6. Tune on validation data only. Run the 1,000-user test after configuration is frozen.
7. Keep routers thin: database SQL belongs in `backend/db.py`; response joins belong in `backend/movie_mapper.py`; recommendation math belongs in `src/recommenders/`.
8. A catalog freshness warning is a product limitation, not permission to label the system commercially production-ready.
9. Keep one API worker. The content artifact stays resident and the optional collaborative/clustering/neural cache intentionally holds only one model at a time to bound memory. Budget at least 1 GB RAM; the Render Blueprint uses its 2 GB `standard` plan.

## Where to make common changes

| Change | Primary files |
| --- | --- |
| Add an API field | `backend/schemas.py`, `backend/movie_mapper.py`, `frontend/src/lib/types.ts` |
| Add a user mutation | `backend/db.py`, one router, `frontend/src/lib/api.ts` |
| Change recommendation ranking | `src/recommenders/ranking.py`, model quality tests, tuning/evaluation scripts |
| Add a recommender | `src/recommenders/`, `backend/data.py`, ensemble, schemas/types, evaluation |
| Change authentication | `backend/auth.py`, `backend/config.py`, security tests |
| Add a table/index | New numbered SQL file under `backend/migrations/` |
| Change movie-card data | `backend/movie_mapper.py`, `backend/schemas.py`, `MovieCard.tsx` |

## Model release workflow

```bash
backend/.venv/bin/python scripts/tune_models.py --max-ratings 1000000 --n-validation-users 200
backend/.venv/bin/python scripts/train_models.py --neural-sample-size 1000000 --neural-epochs 8
backend/.venv/bin/python scripts/evaluate_models.py --max-ratings 1000000 --n-eval-users 1000 --max-rating-predictions 100000 --neural-sample-size 1000000 --neural-epochs 8
backend/.venv/bin/python scripts/validate_model_export.py
```

Do not deploy artifacts between training and evaluation. Training removes stale metrics; evaluation writes fresh metrics and refreshes the checksum manifest. `validate_model_export.py` rejects taste-embedding artifacts unless they declare `training_mode=two_tower_neural`.

## Required checks

```bash
backend/.venv/bin/python -m pytest -q
backend/.venv/bin/python scripts/validate_model_export.py
backend/.venv/bin/pip-audit -r backend/requirements.txt
cd frontend
npm run lint
npm run build
npm audit --omit=dev
```

For a runtime smoke test, start one API worker and confirm `/health` reports configuration, database, and models ready. Stop the server afterward.
