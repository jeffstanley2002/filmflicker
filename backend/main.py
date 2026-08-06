import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from backend import data as data_module
from backend import db
from backend.auth import get_current_user_id
from backend.config import get_settings
from backend.routers import analytics, feedback, metrics, movies, posters, ratings, recommendations, watched, watchlist

logger = logging.getLogger("filmflicker.api")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    errors = settings.validate()
    if errors:
        raise RuntimeError("Invalid runtime configuration: " + "; ".join(errors))
    db.get_connection()
    status = data_module.artifact_status(verify_checksums=True)
    if not status["ready"]:
        raise RuntimeError("Model bundle is not ready: " + "; ".join(status["errors"]))
    data_module.preload()
    logger.info("startup_ready generation_id=%s", status["generation_id"])
    yield


app = FastAPI(title="FilmFlicker API", version="1.1.0", lifespan=lifespan)

_allowed_origins = list(get_settings().allowed_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(movies.router)
app.include_router(posters.router)
app.include_router(feedback.router)
app.include_router(watched.router)
app.include_router(watchlist.router)
app.include_router(ratings.router)
app.include_router(recommendations.router)
app.include_router(analytics.router)
app.include_router(metrics.router)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed request_id=%s path=%s", request_id, request.url.path)
        raise
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    logger.info(
        json.dumps(
            {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": elapsed_ms,
            }
        )
    )
    return response


@app.get("/health/live")
def liveness():
    return {"status": "ok"}


@app.get("/healthz")
def uptime_probe():
    return {"status": "ok"}


@app.get("/health")
def readiness():
    checks = {"configuration": not get_settings().validate(), "database": False, "models": False}
    details = []
    try:
        with db.get_connection().connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        details.append("database unavailable")
    model_status = data_module.artifact_status(verify_checksums=False)
    checks["models"] = model_status["ready"]
    details.extend(model_status["errors"])
    ready = all(checks.values())
    payload = {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
        "generation_id": model_status["generation_id"],
        "details": details,
    }
    return JSONResponse(payload, status_code=200 if ready else 503)


@app.get("/me")
def me(user_id: str = Depends(get_current_user_id)):
    return {"user_id": user_id}
